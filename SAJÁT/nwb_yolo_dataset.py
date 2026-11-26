"""Build staged YOLO26 detection datasets from the filtered NWB recording.

Images are 128 channels x 128 samples of the bandpass + median-referenced signal.
gray: fixed asinh curve.  rgb: R = gray, G = whitened signal, B = NEO energy.
Labels come from the spike sorting, with one centred box per unit.

Stages (cumulative):
  stage1_single_spike     one labelled spike
  stage2_up_to_3_spikes   adds windows with 2-3 labelled spikes
  stage3_any_spike_count  adds sliding windows
Each stage contains the earlier ones; --stages picks which are written (default: stage 3 only).

Usage (spike env):
    python nwb_yolo_dataset.py                        # stage 3
    python nwb_yolo_dataset.py --stages 1 2 3         # all three stages
    python nwb_yolo_dataset.py --quick                # small test build
    python nwb_yolo_dataset.py --image-mode gray rgb  # both encodings
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import sys
import time
import unicodedata
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

SPLITS = ("train", "val", "test")
# Default data folder = two levels above this file (yolo26/, next to the yolo26neuro repo): the NWB and
# DATASETS/ live there on every computer.  NFC: macOS gives "Ö" decomposed, the cache key stores it composed.
DATA_ROOT = Path(unicodedata.normalize("NFC", str(Path(__file__).resolve().parents[2])))


@dataclass
class Config:
    nwb_path: str = str(DATA_ROOT / "Rat01_Insertion1_Depth2.nwb")
    output_root: str = str(DATA_ROOT / "DATASETS" / "nwb_asinh_stages_newdataset") 
    # Zero-phase bandpass; the margin makes the cache bit-identical to reading the NWB chain.
    freq_min: float = 300.0
    freq_max: float = 3000.0
    filter_margin_ms: float = 50.0
    cache_dir: str = str(DATA_ROOT / "DATASETS" / "nwb_pp_cache")  # ~12 GB preprocessed copy, shared by builds
    window: int = 128
    # gray = 127.5 * (1 + asinh(v / knee) / asinh(vmax / knee)), v in uV (noise is ~5 uV).
    gray_knee_uv: float = 10.0
    gray_vmax_uv: float = 1000.0
    image_mode: str = "rgb"  # "gray", or "rgb": R = gray, G = whitened signal, B = NEO energy
    # rgb only: local whitening radius, and the asinh curves for G (noise sd) and B (uV^2).
    whiten_radius_um: float = 50.0
    rgb_white_knee_sd: float = 2.0
    rgb_white_vmax_sd: float = 200.0
    rgb_neo_knee_uv2: float = 100.0
    rgb_neo_vmax_uv2: float = 250000.0
    # Chronological split with gaps; time_fraction < 1 reads only part of each split (--quick).
    train_fraction: float = 0.70
    val_fraction: float = 0.15
    split_gap_s: float = 1.0
    chunk_s: float = 4.0
    time_fraction: dict = field(default_factory=lambda: {"train": 1.0, "val": 1.0, "test": 1.0})
    # Templates: samples before / after the spike, averaged from up to template_max_per_chunk spikes
    # per unit and chunk; units with fewer than template_min_spikes use the stored NWB waveform.
    template_before: int = 40
    template_after: int = 40
    template_max_per_chunk: int = 10
    template_min_spikes: int = 50
    # Box: samples above max(box_fraction x peak, box_min_sigma x noise), symmetric around the peak.
    box_fraction: float = 0.25
    box_min_sigma: float = 3.0
    box_min_half_width: int = 2
    box_min_half_height: int = 2  # rows; 4 = one probe depth step (same column) above and below the peak
    # Fallback units only: align the stored waveform to the peak time measured in the recording.
    center_search: int = 5
    center_min_spikes: int = 20
    visible_sigma: float = 5.0  # negatives have no sample beyond this x noise
    edge_band: int = 2  # a box centre this close to an edge strip makes the window ambiguous
    # Negatives: share of each stage's images, and the spike-free margin around each one.
    negative_fraction: float = 0.10
    negative_margin: int = 40
    max_per_unit_count: dict | None = None  # windows per (split, unit, label count); None = no cap
    stages: tuple = (3,)  # stages to write, from 1, 2 and 3
    stage3_stride: int = 64
    stage3_max_images: dict | None = field(default_factory=lambda: {"train": 40000, "val": 8000, "test": 8000})
    seed: int = 0
    workers: int = max(1, min((os.cpu_count() or 2) - 2, 61))  # Windows allows at most 61 pool workers
    preview_count: int = 12


# --------------------------------------------------------------------------- recording

def load_recording(nwb_path, uv_per_bit, freq_min, freq_max, margin_ms, electrical_series_path=None):
    import spikeinterface.full as si

    recording = si.read_nwb(nwb_path, electrical_series_path=electrical_series_path)
    recording.set_channel_gains([uv_per_bit] * recording.get_num_channels())
    recording.set_channel_offsets([0.0] * recording.get_num_channels())
    recording_bp, recording_pp = preprocess_chain(recording, freq_min, freq_max, margin_ms, with_bandpass=True)
    return recording, recording_bp, recording_pp


def preprocess_chain(recording, freq_min, freq_max, margin_ms, with_bandpass=False):
    """The one preprocessing definition: bandpass (zero-phase) -> global median CMR, int16."""
    import spikeinterface.full as si

    bp = si.bandpass_filter(recording, freq_min=freq_min, freq_max=freq_max, margin_ms=margin_ms)
    pp = si.common_reference(bp, reference="global", operator="median")
    return (bp, pp) if with_bandpass else pp


def whiten_chain(cache_path, shape, fs, uv_per_bit, locations, train_end, radius_um):
    """Third preprocessing step: SpikeInterface local whitening of the bandpass + CMR signal.

    It runs on the cached int16 copy of that signal (the same samples as the NWB chain, fast to
    read).  The whitening matrix W is estimated on the train part only and then fixed for the
    whole recording: whitened (float32, unit noise) = traces_int16 @ W.
    Returns the lazy whitened recording and W, (channels, channels) in NWB column order.
    """
    import spikeinterface.full as si

    rec = si.read_binary(cache_path, fs, "int16", num_channels=shape[1], gain_to_uV=uv_per_bit, offset_to_uV=0.0)
    rec.set_channel_locations(locations)
    train = rec.frame_slice(start_frame=0, end_frame=train_end)
    W = np.asarray(si.whiten(train, dtype="float32", mode="local", radius_um=radius_um, seed=0)._kwargs["W"], np.float32)
    return si.whiten(rec, dtype="float32", W=W), W


def _copy_block(task):
    """Decompress one NWB block (a whole HDF5 chunk, read exactly once) into the raw memmap."""
    import h5py

    nwb_path, dset_name, raw_path, shape, a, b = task
    with h5py.File(nwb_path, "r") as f:
        block = f[dset_name][a:b]
    out = np.memmap(raw_path, np.int16, "r+", shape=shape)
    out[a:b] = block
    out.flush()


def _init_pp_writer(state):
    import spikeinterface.full as si

    _W.clear()
    raw = si.read_binary(state["raw_path"], state["fs"], "int16", num_channels=state["shape"][1],
                         gain_to_uV=state["uv_per_bit"], offset_to_uV=0.0)
    _W["pp"] = preprocess_chain(raw, state["freq_min"], state["freq_max"], state["margin_ms"])
    _W["out"] = np.memmap(state["pp_path"], np.int16, "r+", shape=state["shape"])


def _write_pp(task):
    a, b = task
    _W["out"][a:b] = _W["pp"].get_traces(start_frame=a, end_frame=b)


def ensure_cache(cfg, uv_per_bit, fs, dset_name, shape, block_frames):
    """Preprocessed int16 signal of the whole recording, (frames, channels) in NWB column order.

    Written once: (1) every gzip block of the NWB is decompressed exactly once into a raw
    int16 file, (2) the same SpikeInterface chain as `load_recording` (bandpass + global
    median CMR, same margin) runs on that file in parallel, (3) the raw file is deleted.
    The bandpass margin makes the result bit-identical to reading the NWB chain directly;
    notebook step 11 re-checks this against the NWB.  Reused while `meta.json` matches.
    Returns (path, status), status "built" or "reused".
    """
    import spikeinterface

    key = {
        "nwb_path": str(Path(cfg.nwb_path).resolve()), "nwb_size": os.path.getsize(cfg.nwb_path),
        "nwb_mtime": os.path.getmtime(cfg.nwb_path), "dataset": dset_name, "shape": list(shape),
        "uv_per_bit": uv_per_bit, "freq_min": cfg.freq_min, "freq_max": cfg.freq_max,
        "filter_margin_ms": cfg.filter_margin_ms, "chain": "bandpass_filter -> common_reference(global, median)",
        "spikeinterface_minor": ".".join(spikeinterface.__version__.split(".")[:2]),
    }
    folder = Path(cfg.cache_dir) / f"{Path(cfg.nwb_path).stem}_bp{cfg.freq_min:g}-{cfg.freq_max:g}_m{cfg.filter_margin_ms:g}_cmr"
    pp_path, meta_path = folder / "traces_pp_int16.raw", folder / "meta.json"
    nbytes = int(shape[0]) * int(shape[1]) * 2  # Python ints: np.prod overflows int32 on Windows with numpy 1.x
    if meta_path.exists() and pp_path.exists() and pp_path.stat().st_size == nbytes:
        old = json.loads(meta_path.read_text())
        old.pop("spikeinterface_verified", None)  # written by earlier versions of this script
        if old == key:
            return pp_path, "reused"
        changed = {k: (old.get(k), key.get(k)) for k in key.keys() | old.keys() if old.get(k) != key.get(k)}
        print(f"cache {folder} was made with other settings {changed}; rebuilding it", flush=True)

    folder.mkdir(parents=True, exist_ok=True)
    meta_path.unlink(missing_ok=True)
    free = shutil.disk_usage(folder).free
    if free < 2 * nbytes + 2e9:
        raise OSError(f"cache needs {2 * nbytes / 1e9:.0f} GB free in {folder} while it is built, "
                      f"{free / 1e9:.0f} GB available; set cache_dir elsewhere")
    t0 = time.time()
    raw_path = folder / "traces_raw_int16.tmp"
    np.memmap(raw_path, np.int16, "w+", shape=shape).flush()
    blocks = [(cfg.nwb_path, dset_name, str(raw_path), shape, a, min(a + block_frames, shape[0]))
              for a in range(0, shape[0], block_frames)]
    # A decompressed block is ~1 GB and h5py holds a second copy while reading: keep 3 in flight.
    with ProcessPoolExecutor(min(cfg.workers, 3)) as pool:
        list(pool.map(_copy_block, blocks))
    print(f"cache: {len(blocks)} NWB blocks decompressed ({time.time() - t0:.0f} s)", flush=True)

    np.memmap(pp_path, np.int16, "w+", shape=shape).flush()
    step = int(round(10 * fs))
    state = {"raw_path": str(raw_path), "pp_path": str(pp_path), "shape": shape, "fs": fs, "uv_per_bit": uv_per_bit,
             "freq_min": cfg.freq_min, "freq_max": cfg.freq_max, "margin_ms": cfg.filter_margin_ms}
    with ProcessPoolExecutor(cfg.workers, initializer=_init_pp_writer, initargs=(state,)) as pool:
        list(pool.map(_write_pp, [(a, min(a + step, shape[0])) for a in range(0, shape[0], step)]))
    raw_path.unlink()
    meta_path.write_text(json.dumps(key, indent=1))
    print(f"cache: preprocessed signal written to {pp_path} ({time.time() - t0:.0f} s)", flush=True)
    return pp_path, "built"


def open_cache(path, shape):
    return np.memmap(path, np.int16, "r", shape=tuple(shape))


def asinh_curve(uv, knee_uv, vmax_uv):
    return np.arcsinh(np.clip(uv, -vmax_uv, vmax_uv) / knee_uv) / np.arcsinh(vmax_uv / knee_uv)


def gray_lut(uv_per_bit, knee_uv, vmax_uv):
    """uint8 gray for every int16 value of the preprocessed signal (index = value + 32768)."""
    uv = np.arange(-32768, 32768, dtype=np.float64) * uv_per_bit
    return np.rint((asinh_curve(uv, knee_uv, vmax_uv) + 1.0) * 127.5).astype(np.uint8)


IMAGE_MODES = ("gray", "rgb")


def make_encoder(cfg, ctx):
    """Everything needed to turn int16 traces into an image (picklable, for the workers)."""
    enc = {"mode": cfg.image_mode, "lut": ctx["lut"]}
    if cfg.image_mode == "rgb":
        o = ctx["order"]
        enc.update({
            "uv_per_bit": ctx["uv_per_bit"],
            # The whitening matrix for rows in image order: whitened rows = white @ int16 rows.
            "white": np.ascontiguousarray(ctx["whiten_W"][np.ix_(o, o)].T),
            "white_knee_sd": cfg.rgb_white_knee_sd, "white_vmax_sd": cfg.rgb_white_vmax_sd,
            "neo_knee_uv2": cfg.rgb_neo_knee_uv2, "neo_vmax_uv2": cfg.rgb_neo_vmax_uv2,
        })
    return enc


def encode(enc, x):
    """(ALL rows in image order, W + 2) int16 with ONE extra sample on each side -> uint8 (rows, W[, 3])."""
    img = enc["lut"][x[:, 1:-1].astype(np.int32) + 32768]
    if enc["mode"] != "rgb":
        return img
    counts = x.astype(np.float32)
    g = asinh_curve(enc["white"] @ counts[:, 1:-1], enc["white_knee_sd"], enc["white_vmax_sd"])
    uv = counts * enc["uv_per_bit"]
    core = uv[:, 1:-1]
    neo = np.clip(core * core - uv[:, :-2] * uv[:, 2:], 0.0, enc["neo_vmax_uv2"])
    b = np.arcsinh(neo / enc["neo_knee_uv2"]) / np.arcsinh(enc["neo_vmax_uv2"] / enc["neo_knee_uv2"])
    return np.dstack([img, np.rint((g + 1.0) * 127.5), np.rint(b * 255.0)]).astype(np.uint8)


# --------------------------------------------------------------------------- workers

_W = {}


def _init_worker(state):
    _W.clear()
    _W.update(state)
    _W["traces"] = open_cache(state["cache_path"], state["cache_shape"])


def _read_chunk(start, end):
    return np.ascontiguousarray(_W["traces"][start:end, _W["order"]].T)  # (channels in image order, time) int16


def _render_windows(task):
    """Write one PNG per window start (`pool_dir/<frame>.png`) from the cache."""
    starts, pool_dir = task
    W, enc = _W["window"], _W["enc"]
    for ws in starts:
        img = encode(enc, _read_chunk(ws - 1, ws + W + 1))
        Image.fromarray(img).save(Path(pool_dir) / f"{ws}.png")  # L or RGB


def _pass1_chunk(task):
    """Noise sigma (MAD) per channel, and per unit the summed peak-row trace around its spikes."""
    start, end = task
    uv = _read_chunk(start, end).astype(np.float32) * _W["uv_per_bit"]
    sigma = np.median(np.abs(uv - np.median(uv, axis=1, keepdims=True)), axis=1) / 0.6745
    h, peak_row, n_units = _W["center_search"], _W["peak_row"], len(_W["peak_row"])
    frames, classes = _W["event_frames"], _W["event_classes"]
    lo, hi = np.searchsorted(frames, [start + h, end - h])
    t = (frames[lo:hi] - start).astype(np.int64)
    k = classes[lo:hi].astype(np.int64)
    snips = uv[peak_row[k][:, None], t[:, None] + np.arange(-h, h + 1)]  # (spikes, 2h+1)
    sums = np.zeros((n_units, 2 * h + 1), np.float64)
    np.add.at(sums, k, snips)
    # Full templates: up to template_max_per_chunk spikes per unit, evenly spread over the chunk.
    before, after, cap = _W["template_before"], _W["template_after"], _W["template_max_per_chunk"]
    lo2, hi2 = np.searchsorted(frames, [start + before, end - after - 1])
    t2 = (frames[lo2:hi2] - start).astype(np.int64)
    k2 = classes[lo2:hi2].astype(np.int64)
    pick = []
    for u in np.unique(k2):
        idx = np.nonzero(k2 == u)[0]
        pick.append(idx[np.linspace(0, len(idx) - 1, min(cap, len(idx))).round().astype(int)])
    pick = np.concatenate(pick) if pick else np.zeros(0, np.int64)
    full = np.zeros((n_units, uv.shape[0], before + after + 1), np.float32)
    for i in pick:
        full[k2[i]] += uv[:, t2[i] - before:t2[i] + after + 1]
    return sigma, sums, np.bincount(k, minlength=n_units), full, np.bincount(k2[pick], minlength=n_units)


def _pass2_chunk(task):
    """Windows and labels for one chunk.  Items hold positions only; images are rendered by
    `write_dataset` (or `window_image`).

    Edge rule: a spike is labelled only if its box centre lies in the valid zone of the window
    (not in the edge_margin strip at either side); within edge_band samples of the border the
    spike is ambiguous.  Spikes are ordered by centre, so for an anchor spike (or a group of
    2-3 consecutive spikes) the window starts that label exactly these spikes form one interval
    [lo, hi] computed from the two neighbours.  One uniformly random start in it is the jitter.
    """
    chunk_index, start, end, split, stage3_cap = task
    W = _W["window"]
    rng = np.random.default_rng([_W["seed"], chunk_index])
    x = _read_chunk(start, end)
    T = x.shape[1]
    z = x.astype(np.float32) * (_W["uv_per_bit"] / _W["sigma"][:, None])
    # (n_units, 4): left, right (samples from the sorted spike frame), top row, bottom row + 1;
    # the box centre is frame + center_dt (the template peak), the box symmetric around it.
    boxes, center_dt = _W["boxes"], _W["center_dt"]
    margin, band = _W["edge_margin"], _W["edge_band"]
    lab_lo, lab_hi = margin + band, W - margin - band  # labelled:   lab_lo <= centre column < lab_hi
    out_lo, out_hi = margin - band, W - margin + band  # unlabelled: centre column < out_lo or >= out_hi
    pad = W // 2
    ws_min, ws_max = pad, T - pad - W  # every window lies inside the chunk, away from its edges

    # Sorted events in the chunk, ordered by box centre.
    frames, classes = _W["event_frames"], _W["event_classes"]
    lo, hi = np.searchsorted(frames, [start, end])
    ev_t = (frames[lo:hi] - start).astype(np.int64)
    ev_k = classes[lo:hi].astype(np.int64)
    inside = (ev_t >= 3) & (ev_t < T - 3)
    ev_t, ev_k = ev_t[inside], ev_k[inside]
    by_centre = np.argsort(ev_t + center_dt[ev_k], kind="stable")
    ev_t, ev_k = ev_t[by_centre], ev_k[by_centre]
    c = ev_t + center_dt[ev_k]                                       # box centre (sample in the chunk)
    n = len(ev_t)

    x0 = ev_t - boxes[ev_k, 0]                                       # box [x0, x1)
    x1 = ev_t + boxes[ev_k, 1] + 1
    y0, y1 = boxes[ev_k, 2], boxes[ev_k, 3]
    reach = int(np.abs(boxes[:, :2]).max()) + 2

    def window_objects(ws):
        """Labels (class, x0, y0, x1, y1) for the window starting at ws.  Returns the labels, the
        number of visible but unlabelled spikes (centre in a strip or outside the picture), the
        number of ambiguous spikes (centre in a border band) and the labelled spike indices."""
        a, b = np.searchsorted(c, [ws - reach, ws + W + reach])
        rel = c[a:b] - ws
        labelled = (rel >= lab_lo) & (rel < lab_hi)
        ambiguous = ((rel >= out_lo) & (rel < lab_lo)) | ((rel >= lab_hi) & (rel < out_hi))
        visible = (x1[a:b] > ws) & (x0[a:b] < ws + W)
        li = np.arange(a, b)[labelled]
        lab = np.stack([ev_k[li], np.clip(x0[li] - ws, 0, W), y0[li], np.clip(x1[li] - ws, 0, W), y1[li]], axis=1)
        return lab, int((visible & ~labelled & ~ambiguous).sum()), int(ambiguous.sum()), li

    def interval(i, j):
        """Window starts [lo, hi] that label exactly the spikes i..j: their centres in the valid
        zone, the centres of the neighbours i-1 and j+1 (so of all others) in a strip or outside."""
        lo_, hi_ = int(c[j]) - (lab_hi - 1), int(c[i]) - lab_lo
        if i > 0:
            lo_ = max(lo_, int(c[i - 1]) - (out_lo - 1))
        if j + 1 < n:
            hi_ = min(hi_, int(c[j + 1]) - out_hi)
        return max(lo_, ws_min), min(hi_, ws_max)

    counts = {"stage1_windows": 0, "stage2_groups": 0, "no_window": 0, "plan_mismatch": 0,
              "stage3_ambiguous_dropped": 0}
    items = []

    def add(kind, i, j, lo_, hi_):
        ws = int(rng.integers(lo_, hi_ + 1))
        labels, n_unlab, n_ambig, members = window_objects(ws)
        if n_ambig or members.tolist() != list(range(i, j + 1)):
            counts["plan_mismatch"] += 1  # must stay 0: the interval guarantees exactly this content
            return False
        items.append({
            "kind": kind, "split": split, "labels": labels, "window_start_frame": start + ws,
            "anchor_class": int(ev_k[i]), "n_labels": len(labels), "n_sliver": n_unlab,
            "jitter_lo": start + lo_, "jitter_hi": start + hi_,
        })
        return True

    # Stage 1: every spike that can be the only labelled one.  Stage 2 adds a window for every
    # group of 2-3 consecutive spikes that can be labelled together.
    for i in range(n):
        lo_, hi_ = interval(i, i)
        if lo_ <= hi_:
            counts["stage1_windows"] += add("anchored", i, i, lo_, hi_)
        else:
            counts["no_window"] += 1
    for size in (2, 3):
        for i in range(n - size + 1):
            lo_, hi_ = interval(i, i + size - 1)
            if lo_ <= hi_:
                counts["stage2_groups"] += add("group", i, i + size - 1, lo_, hi_)

    # Stage 3 (only if built): a sliding window with stride stage3_stride from a random offset;
    # windows with an ambiguous spike or without any label are dropped.
    if _W["build_stage3"]:
        stride = _W["stage3_stride"]
        tiles = np.arange(ws_min + int(rng.integers(0, stride)), ws_max + 1, stride)
        for ws in rng.permutation(tiles)[:stage3_cap]:
            ws = int(ws)
            labels, n_unlab, n_ambig, _ = window_objects(ws)
            if n_ambig:
                counts["stage3_ambiguous_dropped"] += 1
                continue
            if not len(labels):
                continue
            items.append({
                "kind": "tile", "split": split, "labels": labels,
                "window_start_frame": start + ws, "anchor_class": -1, "n_labels": len(labels),
                "n_sliver": n_unlab, "jitter_lo": start + ws, "jitter_hi": start + ws,
            })

    # Negatives: non-overlapping windows at a random phase with no sorted spike (visible or
    # not) within negative_margin samples and no sample beyond visible_sigma on any channel.
    m = min(_W["negative_margin"], pad)
    crossings = np.concatenate([[0], np.cumsum(np.abs(z).max(axis=0) >= _W["visible_sigma"])])
    events = np.concatenate([[0], np.cumsum(np.bincount(frames[lo:hi] - start, minlength=T))])
    starts = np.arange(pad + int(rng.integers(0, W)), T - pad - W + 1, W)
    clean = (crossings[starts + W] == crossings[starts]) & (events[starts + W + m] == events[starts - m])
    for ws in rng.permutation(starts[clean]):
        ws = int(ws)
        items.append({
            "kind": "negative", "split": split,
            "labels": np.zeros((0, 5), np.int64), "window_start_frame": start + ws, "anchor_class": -1,
            "n_labels": 0, "n_sliver": 0, "jitter_lo": start + ws, "jitter_hi": start + ws,
        })

    return items, counts


# --------------------------------------------------------------------------- build steps

def plan_chunks(cfg, num_frames, fs, rng):
    gap = int(round(cfg.split_gap_s * fs))
    train_end = int(num_frames * cfg.train_fraction)
    val_end = int(num_frames * (cfg.train_fraction + cfg.val_fraction))
    intervals = {
        "train": (0, train_end),
        "val": (train_end + gap, val_end),
        "test": (val_end + gap, num_frames),
    }
    chunk = int(round(cfg.chunk_s * fs))
    chunks = {}
    for split, (a, b) in intervals.items():
        starts = np.arange(a, b - chunk + 1, chunk)
        n = max(1, int(round(len(starts) * cfg.time_fraction[split])))
        chosen = np.sort(rng.choice(starts, size=min(n, len(starts)), replace=False))
        chunks[split] = [(int(s), int(s) + chunk) for s in chosen]
    return intervals, chunks


def stored_peaks(cfg, stored):
    """Peak row and time offset (from the sorted frame) of every stored mean waveform,
    searched within +-5 samples of the frame (a later, larger co-firing spike is ignored)."""
    c = cfg.template_before
    rows, dts = [], []
    for w in stored:
        wt = w.T  # (rows, 81)
        near = np.abs(wt[:, c - 5:c + 6])
        r, t = np.unravel_index(np.argmax(near), near.shape)
        rows.append(r); dts.append(t - 5)
    return np.array(rows, np.int64), np.array(dts, np.int64)


def shift_time(w, shift):
    """w (rows, time) moved `shift` samples earlier (later if negative), zero-filled."""
    out = np.zeros_like(w)
    if shift > 0:
        out[:, :-shift] = w[:, shift:]
    elif shift < 0:
        out[:, -shift:] = w[:, :shift]
    else:
        out[:] = w
    return out


def build_templates(cfg, stored, sigma, shift=None, recomputed=None):
    """Per-unit templates (image rows x time) and CENTRED boxes.

    The box centre is the waveform peak (row r, time frame + center_dt); its half width and
    half height are the largest distance from the peak to any above-threshold sample of the
    peak's component, so the box is symmetric in time and in rows (clipped at the probe ends).
    Returned boxes are (left, right, top, bottom + 1) relative to the sorted spike frame.
    `recomputed[k]` (rows x 81, or None) is unit k's template averaged from the recording and is
    used when given; otherwise the stored waveform, moved `shift[k]` samples earlier.
    """
    center = cfg.template_before
    shift = np.zeros(len(stored), np.int64) if shift is None else shift
    templates = np.stack([
        recomputed[k] if recomputed is not None and recomputed[k] is not None else shift_time(w.T, int(d))
        for k, (w, d) in enumerate(zip(stored, shift))]).astype(np.float32)  # (units, rows, 81)
    n_units, n_rows, _ = templates.shape
    boxes = np.zeros((n_units, 4), np.int64)
    peak_row = np.zeros(n_units, np.int64)
    center_dt = np.zeros(n_units, np.int64)
    for k in range(n_units):
        absw = np.abs(templates[k])
        # The sorted frame is the trough, so search the peak only near it; the mean waveform
        # can contain a larger, later spike of a co-firing unit.
        near = absw[:, center - 5:center + 6]
        r, t = np.unravel_index(np.argmax(near), near.shape)
        t += center - 5
        peak_row[k], center_dt[k] = r, t - center
        thr = np.maximum(cfg.box_fraction * absw[r, t], cfg.box_min_sigma * sigma)
        mask = absw >= thr[:, None]
        if not mask[r, t]:  # peak below 3 sigma (or sigma NaN): keep the peak so the box is never empty
            print(f"warning: unit {k} peak {absw[r, t]:.3g} uV is below its box threshold {thr[r]:.3g} uV "
                  f"(noise sigma {np.asarray(sigma)[r]:.3g} uV on row {r}); it gets the minimum box", flush=True)
            mask[r, t] = True
        # Rows are depth-first over 4 probe columns, so the same column at the next
        # depth is 4 rows away: bridge +-4 rows before taking the peak's component.
        bridged = ndimage.binary_dilation(mask, structure=np.ones((5, 1), bool))
        components, _ = ndimage.label(bridged, structure=np.ones((3, 3), bool))
        rr, tt = np.nonzero(mask & (components == components[r, t]))
        hw = max(t - tt.min(), tt.max() - t, cfg.box_min_half_width)
        hh = max(r - rr.min(), rr.max() - r, cfg.box_min_half_height)
        dt = int(center_dt[k])
        boxes[k] = (hw - dt, hw + dt, max(r - hh, 0), min(r + hh + 1, n_rows))
    return templates, boxes, peak_row, center_dt


def prepare(cfg: Config):
    """Load recording + sorting, channel order, events, stored templates, split plan and LUT."""
    import pynwb
    import spikeinterface.full as si

    with pynwb.NWBHDF5IO(cfg.nwb_path, "r") as io:
        series = list(io.read().acquisition.values())[0]
        uv_per_bit, nwb_offset, nwb_unit = float(series.conversion), float(series.offset), series.unit
        # Given explicitly: some SpikeInterface versions cannot find the series / t_start themselves.
        series_path, series_rate = f"acquisition/{series.name}", float(series.rate)
        series_t_start = float(series.starting_time or 0.0)
        dset_name, dset_shape = series.data.name, tuple(series.data.shape)
        block_frames = series.data.chunks[0] if series.data.chunks else 1 << 20
    if not np.isclose(nwb_offset, 0.0):
        raise ValueError(f"Unexpected NWB offset {nwb_offset}")

    recording, recording_bp, recording_pp = load_recording(
        cfg.nwb_path, uv_per_bit, cfg.freq_min, cfg.freq_max, cfg.filter_margin_ms, electrical_series_path=series_path
    )
    sorting = si.read_nwb_sorting(cfg.nwb_path, electrical_series_path=series_path,
                                  sampling_frequency=series_rate, t_start=series_t_start)
    fs = float(recording.get_sampling_frequency())
    num_frames = int(recording.get_num_frames())
    if recording.get_num_segments() != 1 or sorting.get_num_segments() != 1:
        raise ValueError("Expected one recording segment and one sorting segment.")
    if recording.get_num_channels() != cfg.window:
        raise ValueError(f"Expected {cfg.window} channels, found {recording.get_num_channels()}")
    if recording_pp.get_dtype() != np.int16:
        raise ValueError("The int16 lookup-table mapping needs an int16 preprocessing pipeline.")
    if dset_shape != (num_frames, recording.get_num_channels()):
        raise ValueError(f"NWB dataset {dset_name} has shape {dset_shape}, SpikeInterface sees "
                         f"{(num_frames, recording.get_num_channels())}")
    cache_path, cache_status = ensure_cache(cfg, uv_per_bit, fs, dset_name, dset_shape, block_frames)
    traces = open_cache(cache_path, dset_shape)
    if cache_status == "built":  # a new cache must equal the NWB chain exactly (also checked in notebook step 11)
        seam = int(round(10 * fs))  # ensure_cache writes 10 s tasks; NWB gzip blocks are block_frames long
        for a in sorted({num_frames // 3, 37 * seam - 1000, block_frames - 1000, num_frames - 2000}):
            a = min(max(a, 0), num_frames - 2000)
            if not np.array_equal(traces[a:a + 2000], recording_pp.get_traces(start_frame=a, end_frame=a + 2000)):
                raise RuntimeError(f"cache {cache_path} differs from the NWB preprocessing chain at frame {a}; "
                                   "delete the cache folder and rebuild")

    channel_ids = [str(c) for c in recording.get_channel_ids()]
    locations = recording.get_channel_locations()
    # Top of the probe first (descending y), then x.  This equals the raw NWB channel
    # order, which the mean-waveform pretrained checkpoint was trained on; ascending y
    # would show every unit upside down to that model.
    order = np.lexsort((locations[:, 0], -locations[:, 1]))
    order_text = "\n".join(f"{channel_ids[i]},{locations[i, 0]:g},{locations[i, 1]:g}" for i in order)

    unit_ids = list(sorting.get_unit_ids())
    # Class k = k-th row of the NWB units table, named unit_<cluster_id> exactly like the
    # mean-waveform checkpoint, so its class head keeps meaning the same units.
    if "cluster_id" in sorting.get_property_keys():
        class_names = [f"unit_{int(c)}" for c in sorting.get_property("cluster_id")]
    else:
        class_names = [str(u) for u in unit_ids]
    frames, classes = [], []
    for k, u in enumerate(unit_ids):
        st = np.asarray(sorting.get_unit_spike_train(unit_id=u), dtype=np.int64)
        frames.append(st)
        classes.append(np.full(len(st), k, np.int64))
    frames, classes = np.concatenate(frames), np.concatenate(classes)
    if len(frames) == 0 or frames.min() < 0 or frames.max() >= num_frames:
        raise RuntimeError(f"sorted spike frames {frames.min() if len(frames) else '-'}..{frames.max() if len(frames) else '-'} "
                           f"do not fit the recording (0..{num_frames - 1}): the sorting's sampling frequency or "
                           f"t_start is wrong (expected {series_rate} Hz, t_start {series_t_start} s)")
    o = np.argsort(frames, kind="stable")
    stored = []
    for u in unit_ids:
        w = np.asarray(sorting.get_property("mean_waveform_all_channels_filt", [u])[0], np.float32)
        w = w if w.shape[1] == len(channel_ids) else w.T
        stored.append(w[:, order])

    rng = np.random.default_rng(cfg.seed)
    intervals, chunks = plan_chunks(cfg, num_frames, fs, rng)
    recording_white, whiten_W = whiten_chain(str(cache_path), dset_shape, fs, uv_per_bit, locations,
                                             intervals["train"][1], cfg.whiten_radius_um)
    return {
        "t_start": time.time(), "rng": rng,
        "uv_per_bit": uv_per_bit, "nwb_unit": nwb_unit, "fs": fs, "num_frames": num_frames,
        # recording*: the lazy NWB chain (the deployment contract, slow to read);
        # traces: the same signal from the cache, (frames, channels) int16 memmap in NWB column order.
        "recording": recording, "recording_bp": recording_bp, "recording_pp": recording_pp,
        # recording_white: lazy SpikeInterface whitening of the cached signal; whiten_W: its fixed matrix.
        "recording_white": recording_white, "whiten_W": whiten_W,
        "traces": traces, "cache_path": str(cache_path),
        "channel_ids": channel_ids, "locations": locations, "order": order, "order_text": order_text,
        "order_hash": hashlib.sha256(order_text.encode()).hexdigest(),
        "unit_ids": unit_ids, "class_names": class_names,
        "event_frames": frames[o], "event_classes": classes[o], "stored": stored,
        "unit_spike_count": np.bincount(classes, minlength=len(unit_ids)),
        "intervals": intervals, "chunks": chunks,
        "lut": gray_lut(uv_per_bit, cfg.gray_knee_uv, cfg.gray_vmax_uv),
    }


def worker_state(cfg, ctx):
    state = {
        "cache_path": ctx["cache_path"], "cache_shape": ctx["traces"].shape, "window": cfg.window,
        "uv_per_bit": ctx["uv_per_bit"], "order": ctx["order"], "event_frames": ctx["event_frames"],
        "event_classes": ctx["event_classes"],
    }
    if "boxes" in ctx:
        state.update({
            "seed": cfg.seed, "sigma": ctx["sigma"], "boxes": ctx["boxes"], "center_dt": ctx["center_dt"],
            "visible_sigma": cfg.visible_sigma,
            "edge_margin": ctx["edge_margin"], "edge_band": cfg.edge_band,
            "negative_margin": cfg.negative_margin,
            "build_stage3": 3 in cfg.stages, "stage3_stride": cfg.stage3_stride,
        })
    return state


def run_pass1(cfg, ctx):
    """Noise sigma per channel and per-unit templates averaged from the recording (train chunks
    only); boxes come from those templates.  Units with too few spikes fall back to the
    stored NWB mean waveform, moved in time to the peak measured in this pipeline."""
    chunks = ctx["chunks"]["train"]
    stored_row, stored_dt = stored_peaks(cfg, ctx["stored"])
    state = {**worker_state(cfg, ctx), "peak_row": stored_row, "center_search": cfg.center_search,
             "template_before": cfg.template_before, "template_after": cfg.template_after,
             "template_max_per_chunk": cfg.template_max_per_chunk}
    h = cfg.center_search
    n_units = len(stored_row)
    sigmas, sums, counts = [], np.zeros((n_units, 2 * h + 1)), np.zeros(n_units, np.int64)
    tsum, tcount = None, np.zeros(n_units, np.int64)
    with ProcessPoolExecutor(cfg.workers, initializer=_init_worker, initargs=(state,)) as pool:
        for sg, sm, ct, full, fc in pool.map(_pass1_chunk, chunks):
            sigmas.append(sg); sums += sm; counts += ct
            tsum = full.astype(np.float64) if tsum is None else tsum + full
            tcount += fc
    recomputed = [tsum[k] / tcount[k] if tcount[k] >= cfg.template_min_spikes else None for k in range(n_units)]
    sigma = np.median(np.stack(sigmas), axis=0).astype(np.float32)
    measured = np.abs(sums).argmax(axis=1) - h  # peak of the averaged peak-row trace, from the sorted frame
    data_dt = np.where(counts >= cfg.center_min_spikes, measured, stored_dt)
    shift = stored_dt - data_dt  # > 0: the stored waveform is late, move it earlier (fallback units)
    templates, boxes, peak_row, center_dt = build_templates(
        cfg, ctx["stored"], sigma, shift, recomputed)
    source = ["recomputed" if recomputed[k] is not None
              else ("nwb_mean_waveform" if shift[k] == 0 else f"nwb_mean_waveform_shifted_{int(shift[k]):+d}")
              for k in range(n_units)]
    fallback = [k for k in range(n_units) if recomputed[k] is None]
    # Spike times must line up with the signal: then every averaged template has a clear peak. Averaged
    # noise (misaligned times) has none, every box falls back to the minimum and all labels are useless.
    peak_uv = np.abs(templates[np.arange(n_units), peak_row, cfg.template_before + center_dt])
    weak = peak_uv < cfg.box_min_sigma * sigma[peak_row]
    if not np.isfinite(sigma).all() or weak[[k for k in range(n_units) if recomputed[k] is not None]].mean() > 0.1:
        raise RuntimeError(
            f"{int(weak.sum())}/{n_units} unit templates have no peak above {cfg.box_min_sigma} x noise "
            f"(median peak {np.nanmedian(peak_uv):.1f} uV, median noise {np.nanmedian(sigma):.1f} uV): the spike "
            "times do not line up with the signal. Check the SpikeInterface version (0.104.x), the sorting's "
            "sampling frequency / t_start, and delete the cache folder if it was copied from another computer.")
    widest =int((boxes[:, 0] + boxes[:, 1] + 1).max())
    edge_margin = (widest + 1) // 2
    if 2 * (edge_margin + cfg.edge_band) >= cfg.window:
        raise ValueError(f"edge_margin {edge_margin} leaves no valid zone in a {cfg.window}-sample window")
    ctx.update({"sigma": sigma, "templates": templates, "boxes": boxes, "peak_row": peak_row,
                "center_dt": center_dt, "template_source": source,
                "timing_shift": np.where([r is None for r in recomputed], shift, 0), "timing_spikes": counts,
                "template_spikes": tcount, "edge_margin": int(edge_margin)})
    print(f"templates: {n_units - len(fallback)} units averaged from the recording "
          f"(median {int(np.median(tcount))} spikes), {len(fallback)} on the stored waveform "
          f"({[ctx['class_names'][k] for k in fallback][:10]})", flush=True)
    width = boxes[:, 0] + boxes[:, 1] + 1
    print(f"pass 1 done ({time.time() - ctx['t_start']:.0f} s, {len(chunks)} chunks): noise sigma median "
          f"{np.median(sigma):.2f} uV; centred boxes from the templates: width "
          f"{width.min()}-{int(np.median(width))}-{width.max()} samples, height "
          f"{int(np.median(boxes[:, 3] - boxes[:, 2]))} rows (median); edge strips {edge_margin} samples "
          f"(valid zone: columns {edge_margin}-{cfg.window - edge_margin - 1})", flush=True)


def pass2_tasks(cfg, ctx):
    tasks, index = [], 0
    for split in SPLITS:
        cap = (None if cfg.stage3_max_images is None
               else int(np.ceil(1.5 * cfg.stage3_max_images[split] / len(ctx["chunks"][split]))))
        for a, b in ctx["chunks"][split]:
            tasks.append((index, a, b, split, cap))
            index += 1
    return tasks


def run_pass2(cfg, ctx):
    """Candidate images and labels for all planned chunks."""
    by_kind = {"anchored": [], "group": [], "tile": [], "negative": []}
    rejects = {}
    with ProcessPoolExecutor(cfg.workers, initializer=_init_worker, initargs=(worker_state(cfg, ctx),)) as pool:
        for items, rej in pool.map(_pass2_chunk, pass2_tasks(cfg, ctx)):
            for it in items:
                by_kind[it["kind"]].append(it)
            for r, c in rej.items():
                rejects[r] = rejects.get(r, 0) + c
    ctx.update({"anchored": by_kind["anchored"], "groups": by_kind["group"], "tiles": by_kind["tile"],
                "negatives": by_kind["negative"], "window_counts": rejects})
    if rejects.get("plan_mismatch"):
        raise RuntimeError(f"{rejects['plan_mismatch']} windows did not contain exactly their planned spikes")
    print(f"window search: {rejects}", flush=True)
    print(f"pass 2 done ({time.time() - ctx['t_start']:.0f} s): {len(by_kind['anchored'])} stage-1 windows, "
          f"{len(by_kind['group'])} stage-2 windows, {len(by_kind['tile'])} stage-3 windows, "
          f"{len(by_kind['negative'])} negatives", flush=True)


def _cap_per_unit(items, caps, rng):
    """At most caps[split] items per (split, anchor unit, label count); caps None = all."""
    if caps is None:
        return list(items)
    groups = {}
    for it in items:
        groups.setdefault((it["split"], it["anchor_class"], it["n_labels"]), []).append(it)
    return [group[j] for (split, _, _), group in groups.items() for j in rng.permutation(len(group))[:caps[split]]]


def select_items(cfg, ctx):
    """Cumulative stages: stage 1 = stage-1 windows; stage 2 = stage 1 + stage-2 windows;
    stage 3 (if built) = stage 2 + sliding windows.  Each stage adds negative_fraction negatives
    for its NEW windows from one shuffled pool per split, so earlier negatives stay included."""
    rng = ctx["rng"]
    new = {
        "stage1_single_spike": _cap_per_unit(ctx["anchored"], cfg.max_per_unit_count, rng),
        "stage2_up_to_3_spikes": _cap_per_unit(ctx["groups"], cfg.max_per_unit_count, rng),
    }
    if 3 in cfg.stages:
        tiles = []
        for split in SPLITS:
            full = [it for it in ctx["tiles"] if it["split"] == split]
            cap = None if cfg.stage3_max_images is None else int(round(
                cfg.stage3_max_images[split] * (1 - cfg.negative_fraction)))
            tiles += [full[i] for i in rng.permutation(len(full))[:cap]]
        new["stage3_any_spike_count"] = tiles

    pools = {}
    for split in SPLITS:
        negs = [it for it in ctx["negatives"] if it["split"] == split]
        pools[split] = [negs[i] for i in rng.permutation(len(negs))]
    stage_items, so_far, n_neg = {}, [], {sp: 0 for sp in SPLITS}
    for stage_name, items in new.items():
        so_far = so_far + items
        frac = cfg.negative_fraction
        for split in SPLITS:
            n_new = sum(it["split"] == split for it in items)
            want = n_neg[split] + int(round(n_new * frac / (1 - frac)))
            if want > len(pools[split]):
                print(f"warning: {stage_name} {split} wants {want} negatives, only {len(pools[split])} found",
                      flush=True)
            n_neg[split] = min(want, len(pools[split]))
        stage_items[stage_name] = so_far + [it for sp in SPLITS for it in pools[sp][:n_neg[sp]]]
    for items in stage_items.values():
        items.sort(key=lambda it: (SPLITS.index(it["split"]), it["window_start_frame"]))
    return stage_items


def stage_stats(items, n_units):
    per_split = {}
    for split in SPLITS:
        its = [it for it in items if it["split"] == split]
        n_labels = [len(it["labels"]) for it in its]
        cls = np.concatenate([it["labels"][:, 0] for it in its]) if its else np.zeros(0, int)
        per_split[split] = {
            "images": len(its),
            "negative_images": int(sum(it["kind"] == "negative" for it in its)),
            "windows_by_kind": {k: int(sum(it["kind"] == k for it in its)) for k in ("anchored", "group", "tile")},
            "labels_per_image_mean": round(float(np.mean(n_labels)), 3) if its else 0.0,
            "label_count_histogram": {int(c): int(n) for c, n in zip(*np.unique(n_labels, return_counts=True))},
            "classes_present": f"{len(np.unique(cls))}/{n_units}",
            "images_with_unlabelled_slivers": int(sum(it["n_sliver"] > 0 for it in its)),
        }
    return per_split


def _link(src, dst):
    try:
        os.link(src, dst)  # same bytes on disk for every stage that uses the window
    except OSError:
        shutil.copyfile(src, dst)


def write_stage(stage_dir, items, class_names, cfg, pool_dir, height):
    W, H = cfg.window, height  # image width = samples, height = channels
    rows = []
    for split in SPLITS:
        (stage_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (stage_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        for i, it in enumerate(it for it in items if it["split"] == split):
            stem = f"{split}_{i:06d}_f{it['window_start_frame']}"
            _link(pool_dir / f"{it['window_start_frame']}.png", stage_dir / "images" / split / f"{stem}.png")
            lines = [
                f"{c} {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} {(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}\n"
                for c, x0, y0, x1, y1 in it["labels"].tolist()
            ]
            (stage_dir / "labels" / split / f"{stem}.txt").write_text("".join(lines), encoding="utf-8")
            rows.append({
                "split": split, "image": f"images/{split}/{stem}.png",
                "window_start_frame": it["window_start_frame"], "kind": it["kind"],
                "anchor_class": it["anchor_class"], "n_labels": len(it["labels"]),
                "n_sliver_unlabelled": it["n_sliver"], "jitter_lo_frame": it["jitter_lo"], "jitter_hi_frame": it["jitter_hi"],
                "classes": " ".join(str(c) for c in it["labels"][:, 0].tolist()),
            })
    with (stage_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    names = "\n".join(f"  {i}: {json.dumps(n)}" for i, n in enumerate(class_names))
    (stage_dir / "data.yaml").write_text(
        f"path: {stage_dir}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        f"nc: {len(class_names)}\nnames:\n{names}\n",
        encoding="utf-8",
    )


def window_image(ctx, window_start_frame, window=128, enc=None):
    """uint8 image (channels in image order, samples[, 3]) of one window, read from the cache.
    `enc` (from make_encoder) gives any image mode; otherwise the gray lookup table ctx["lut"]."""
    if enc is None:
        enc = {"mode": "lut", "lut": ctx["lut"]}
    x = ctx["traces"][window_start_frame - 1:window_start_frame + window + 1, ctx["order"]].T
    return encode(enc, x)


def plot_items(items, title, count, rng, ctx, path=None, enc=None):
    """Random examples with their boxes, rendered from the cache."""
    import matplotlib.pyplot as plt

    picks = [items[i] for i in rng.choice(len(items), size=min(count, len(items)), replace=False)]
    cols = 4
    nrows = int(np.ceil(len(picks) / cols))
    fig, axes = plt.subplots(nrows, cols, figsize=(4 * cols, 4 * nrows), squeeze=False)
    for ax, it in zip(axes.flat, picks):
        img = window_image(ctx, it["window_start_frame"], enc=enc)
        ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        for c, x0, y0, x1, y1 in it["labels"].tolist():
            color = "red" if c == it["anchor_class"] else "orange"
            ax.add_patch(plt.Rectangle((x0 - 0.5, y0 - 0.5), x1 - x0, y1 - y0, fill=False, ec=color, lw=1.2))
        if "edge_margin" in ctx:  # border between the invalid strips and the valid zone
            for xb in (ctx["edge_margin"] - 0.5, img.shape[1] - ctx["edge_margin"] - 0.5):
                ax.axvline(xb, color="cyan", lw=0.8, ls="--")
        ax.set_title(f"{it['split']} f={it['window_start_frame']} n={len(it['labels'])} unlabelled={it['n_sliver']}"
                     + (" NEG" if it["kind"] == "negative" else ""), fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes.flat[len(picks):]:
        ax.set_visible(False)
    fig.suptitle(f"{title}  (red = anchor spike, orange = other labelled spikes, dashed = edge strips: no labels)")
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=80)
        plt.close(fig)
    return fig


def write_dataset(cfg, ctx, stage_items):
    """Write the three stages plus everything deployment needs to reproduce the images."""
    import matplotlib
    import pynwb
    import scipy
    import spikeinterface

    if not matplotlib.is_interactive():
        matplotlib.use("Agg")
    unit_ids = ctx["unit_ids"]
    build_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = Path(cfg.output_root) / f"build_{build_id}_{cfg.image_mode}"
    root.mkdir(parents=True)
    class_names = ctx["class_names"]
    enc = make_encoder(cfg, ctx)
    lut = enc["lut"]

    # Render every distinct window once (stages share windows: stage 1 is a subset of
    # stage 2 and all stages draw negatives from one pool), then link it into the stages.
    pool_dir = root / "_rendered"
    pool_dir.mkdir()
    starts = sorted({it["window_start_frame"] for items in stage_items.values() for it in items})
    state = {**worker_state(cfg, ctx), "enc": enc}
    with ProcessPoolExecutor(cfg.workers, initializer=_init_worker, initargs=(state,)) as pool:
        list(pool.map(_render_windows, [(part.tolist(), str(pool_dir))
                                        for part in np.array_split(np.array(starts, np.int64), cfg.workers * 4)]))
    print(f"rendered {len(starts)} distinct windows ({time.time() - ctx['t_start']:.0f} s)", flush=True)

    stats = {"stages": {}}
    preview_rng = np.random.default_rng([cfg.seed, 2])  # own rng: previews do not depend on the image mode
    for stage_name, items in stage_items.items():
        write_stage(root / stage_name, items, class_names, cfg, pool_dir, height=len(ctx["order"]))
        plot_items(items, stage_name, cfg.preview_count, preview_rng, ctx=ctx, enc=enc,
                   path=root / f"preview_{stage_name}.png")
        stats["stages"][stage_name] = stage_stats(items, len(unit_ids))
    stats["window_search"] = ctx.get("window_counts", {})

    seg = ctx["recording_bp"]._recording_segments[0]
    shutil.rmtree(pool_dir)  # the stage folders hold links to the same files
    boxes = ctx["boxes"]
    np.save(root / "gray_lut_int16.npy", lut)
    np.savez(root / "preprocessing_arrays.npz", channel_order=ctx["order"],
             noise_sigma_uv_image_order=ctx["sigma"], bandpass_coeff=np.asarray(seg.coeff),
             whitening_matrix=ctx["whiten_W"])
    np.savez(root / "unit_templates.npz", templates_uv=ctx["templates"], boxes=boxes, peak_row=ctx["peak_row"],
             center_dt=ctx["center_dt"], unit_ids=np.array([str(u) for u in unit_ids]),
             class_names=np.array(class_names))
    (root / "channel_order.csv").write_text("channel_id,x_um,y_um\n" + ctx["order_text"] + "\n", encoding="utf-8")
    with (root / "class_map.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "class_name", "unit_id", "template_source", "spike_count", "box_left", "box_right",
                         "box_top_row", "box_bottom_row", "center_dt", "stored_timing_shift", "timing_spikes", "template_spikes"])
        for k, u in enumerate(unit_ids):
            writer.writerow([k, class_names[k], u, ctx["template_source"][k], int(ctx["unit_spike_count"][k]),
                             *boxes[k].tolist(), int(ctx["center_dt"][k]), int(ctx["timing_shift"][k]),
                             int(ctx["timing_spikes"][k]), int(ctx["template_spikes"][k])])
    preprocessing = {
        "build_id": build_id,
        "nwb": {"path": cfg.nwb_path, "size_bytes": os.path.getsize(cfg.nwb_path), "series_unit": ctx["nwb_unit"],
                "uv_per_bit": ctx["uv_per_bit"], "sampling_frequency_hz": ctx["fs"], "num_frames": ctx["num_frames"]},
        "splits_frames": ctx["intervals"],
        "split_gap_s": cfg.split_gap_s,
        "chunks": ctx["chunks"],
        "label_policy": {"edge_margin": ctx["edge_margin"]},
        "config": asdict(cfg),
        "versions": {"python": sys.version.split()[0], "platform": platform.platform(), "numpy": np.__version__,
                     "scipy": scipy.__version__, "spikeinterface": spikeinterface.__version__,
                     "pynwb": pynwb.__version__},
        "build_seconds": round(time.time() - ctx["t_start"], 1),
    }
    (root / "preprocessing.json").write_text(json.dumps(preprocessing, indent=2, default=str), encoding="utf-8")
    (root / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"wrote {root} in {time.time() - ctx['t_start']:.0f} s")
    return root, stats


def build(cfg: Config, image_modes=None):
    ctx = prepare(cfg)
    print(f"{len(ctx['unit_ids'])} units, {len(ctx['event_frames']):,} spikes, "
          f"{ctx['num_frames'] / ctx['fs']:.0f} s; chunks per split: "
          + ", ".join(f"{s}={len(c)}" for s, c in ctx["chunks"].items()), flush=True)
    run_pass1(cfg, ctx)
    run_pass2(cfg, ctx)
    # Every stage is selected, so a stage's content does not depend on which stages are written.
    stage_items = {name: items for name, items in select_items(cfg, ctx).items()
                   if int(name[len("stage")]) in cfg.stages}
    roots = []
    for mode in image_modes or [cfg.image_mode]:  # same windows and labels, one build per encoding
        root, stats = write_dataset(replace(cfg, image_mode=mode), ctx, stage_items)
        roots.append(root)
    print(json.dumps(stats["stages"], indent=1))
    return roots


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="small smoke-test build")
    parser.add_argument("--nwb", dest="nwb_path", default=None, help=f"NWB file (default {Config.nwb_path})")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--image-mode", choices=IMAGE_MODES, nargs="+", default=None,
                        help="one or more encodings; each gets its own build folder with the same windows")
    parser.add_argument("--cache-dir", default=None, help="where the ~12 GB preprocessed copy is kept")
    parser.add_argument("--stages", type=int, nargs="+", choices=(1, 2, 3), default=None,
                        help="stages to write (default: 3)")
    args = parser.parse_args()
    cfg = Config()
    if args.quick:
        cfg.time_fraction = {"train": 0.02, "val": 0.03, "test": 0.03}
        cfg.stage3_max_images = {"train": 300, "val": 100, "test": 100}
        cfg.template_min_spikes = 20  # few chunks: at most ~80 spikes per unit are averaged
    if args.stages:
        cfg.stages = tuple(sorted(set(args.stages)))
    if args.image_mode:
        cfg.image_mode = args.image_mode[0]
    for name in ("nwb_path", "cache_dir", "output_root", "workers"):
        if getattr(args, name):
            setattr(cfg, name, getattr(args, name))
    build(cfg, args.image_mode)


if __name__ == "__main__":
    main()
