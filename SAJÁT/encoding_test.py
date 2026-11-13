"""One-off test: which image encoding should the spike detector get?

Takes ONE gray build of nwb_yolo_dataset.py (its windows and labels) and renders the other arms
from the preprocessed cache, so every arm has exactly the same windows and labels:

  A  gray  uV            the base build itself (asinh of microvolts)
  B  gray  white         asinh of the locally whitened signal (noise units)
  C  rgb   uV, nbr, neo  the current rgb_feat: neighbour-electrode mean and NEO energy
  D  rgb   uV, white, neo

Whitening needs no second cache: the 128 x 128 matrix is computed once with SpikeInterface
(`whiten`, local, train part of the recording only) and applied to each window when it is rendered.

Training is identical for every arm: class-agnostic, from yolo26n.pt, no augmentation, AdamW 1e-3
with cosine decay, a fixed number of epochs, validation at 128 x 128 (not Ultralytics' padded
rect validation).  Each run is tested on the stage-1 test split and on the stage-3 test split.

    python encoding_test.py BASE_BUILD --render                    # write arms B, C, D + preview.png
    python encoding_test.py BASE_BUILD --train --epochs 30         # all arms, seed 0
    python encoding_test.py BASE_BUILD --train --arms A D --seeds 0 1 2
    python encoding_test.py BASE_BUILD                             # print the results table

Every training run is logged to Weights & Biases (project `--wandb-project`, default encoding_test).
Results are kept in <BASE_BUILD>_arms/results.json; runs already in it are skipped.
Delete this script and the _arms folder once the encoding is chosen.
"""
import argparse
import csv
import json
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # the local ultralytics fork
sys.path.insert(0, str(HERE))
from nwb_yolo_dataset import asinh_curve, gray_lut, open_cache  # noqa: E402


def neighbor_matrix(locations_image_order, radius_um):
    """(rows, rows) row-normalised mean over the electrodes within radius_um (self included); arm C."""
    d = np.linalg.norm(locations_image_order[:, None, :] - locations_image_order[None, :, :], axis=2)
    m = (d <= radius_um).astype(np.float32)
    return m / m.sum(axis=1, keepdims=True)

ARMS = {"A": ["uv"], "B": ["white"], "C": ["uv", "nbr", "neo"], "D": ["uv", "white", "neo"]}
STAGES = ("stage1_single_spike", "stage3_any_spike_count")
SPLITS = ("train", "val", "test")
WHITEN_RADIUS_UM = 50.0
WHITE_KNEE, WHITE_VMAX = 2.0, 200.0  # noise units; the uV curve is knee 10, max 1000 at ~5 uV noise
IMGSZ = 128


# --------------------------------------------------------------------------- rendering

def load_base(base):
    """Everything the views need, read from the base build's own files."""
    pre = json.loads((base / "preprocessing.json").read_text())
    cfg, nwb = pre["config"], pre["nwb"]
    order = np.load(base / "preprocessing_arrays.npz")["channel_order"]
    rows = list(csv.DictReader(open(base / "channel_order.csv")))
    locations = np.array([[float(r["x_um"]), float(r["y_um"])] for r in rows])  # image order
    return {
        "cache_path": str(Path(pre["images_cut_from"]["cache"])), "cache_shape": (nwb["num_frames"], len(order)),
        "fs": nwb["sampling_frequency_hz"], "uv_per_bit": nwb["uv_per_bit"], "order": order,
        "locations": locations, "train_end": pre["splits_frames"]["train"][1],
        "lut": gray_lut(nwb["uv_per_bit"], cfg["gray_knee_uv"], cfg["gray_vmax_uv"]),
        "vmax_uv": cfg["gray_vmax_uv"], "nbr_knee_uv": 5.0,
        "nb": neighbor_matrix(locations, 33.0),
        "neo_knee": cfg["rgb_neo_knee_uv2"], "neo_vmax": cfg["rgb_neo_vmax_uv2"],
    }


def whitening_matrix(state, path):
    """(channels, channels) matrix W of SpikeInterface's local whitening, estimated on the train
    part of the cache: whitened = traces_int16 @ W, unit noise variance.  Saved for reuse."""
    if path.exists():
        return np.load(path)
    import spikeinterface.full as si

    n = state["cache_shape"][1]
    rec = si.read_binary(state["cache_path"], state["fs"], "int16", num_channels=n,
                         gain_to_uV=state["uv_per_bit"], offset_to_uV=0.0)
    loc = np.zeros((n, 2))
    loc[state["order"]] = state["locations"]
    rec.set_channel_locations(loc)
    train = rec.frame_slice(start_frame=0, end_frame=state["train_end"])
    W = np.asarray(si.whiten(train, dtype="float32", mode="local", radius_um=WHITEN_RADIUS_UM, seed=0)._kwargs["W"],
                   np.float32)
    np.save(path, W)
    return W


def to_u8(curve):
    return np.rint((curve + 1.0) * 127.5).astype(np.uint8)


def views(state, block):
    """block: (W + 2, channels) int16 from the cache, one extra sample on each side.
    Returns the four uint8 views, each (channels in image order, W)."""
    order = state["order"]
    x = block[:, order].T
    uv = x.astype(np.float32) * state["uv_per_bit"]
    core = uv[:, 1:-1]
    neo = np.clip(core * core - uv[:, :-2] * uv[:, 2:], 0.0, state["neo_vmax"])
    white = (block[1:-1].astype(np.float32) @ state["W"])[:, order].T
    return {
        "uv": state["lut"][x[:, 1:-1].astype(np.int32) + 32768],
        "nbr": to_u8(asinh_curve(state["nb"] @ core, state["nbr_knee_uv"], state["vmax_uv"])),
        "neo": np.rint(255.0 * np.arcsinh(neo / state["neo_knee"])
                       / np.arcsinh(state["neo_vmax"] / state["neo_knee"])).astype(np.uint8),
        "white": to_u8(asinh_curve(white, WHITE_KNEE, WHITE_VMAX)),
    }


_S = {}


def _init(state):
    _S.update(state)
    _S["traces"] = open_cache(state["cache_path"], state["cache_shape"])


def _render(task):
    for ws, targets in task:
        v = views(_S, _S["traces"][ws - 1:ws + IMGSZ + 1])
        for arm, path in targets:
            chans = [v[name] for name in ARMS[arm]]
            Image.fromarray(chans[0] if len(chans) == 1 else np.dstack(chans)).save(path)


def _link(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def render(base, out, arms, workers):
    state = load_base(base)
    out.mkdir(exist_ok=True)
    state["W"] = whitening_matrix(state, out / "whitening_W.npy")
    arms = [a for a in arms if a != "A"]  # arm A is the base build

    # The whitened signal must have unit noise and no correlation between neighbouring channels.
    tr = open_cache(state["cache_path"], state["cache_shape"])
    a0 = state["train_end"] // 2
    w = (tr[a0:a0 + 80000].astype(np.float32) @ state["W"])[:, state["order"]]
    raw = tr[a0:a0 + 80000][:, state["order"]].astype(np.float32)
    mad = lambda z: float(np.median(np.median(np.abs(z - np.median(z, axis=0)), axis=0) / 0.6745))
    nbcorr = lambda z: float(np.mean([np.corrcoef(z[:, i], z[:, i + 4])[0, 1] for i in range(0, 124, 7)]))
    print(f"whitening check: noise sd {mad(w):.2f} (want ~1), neighbour correlation {nbcorr(w):.3f} "
          f"(before whitening {nbcorr(raw):.3f})", flush=True)

    tasks = []
    for stage in STAGES:
        rows = list(csv.DictReader(open(base / stage / "manifest.csv")))
        only_test = stage != STAGES[0]  # stage 3 is only tested on; its train and val are large
        if only_test:
            rows = [r for r in rows if r["split"] == "test"]
        for arm in arms:
            dst = out / arm / stage
            for split in SPLITS:
                (dst / "images" / split).mkdir(parents=True, exist_ok=True)
                (dst / "labels" / split).mkdir(parents=True, exist_ok=True)
            yaml = (base / stage / "data.yaml").read_text().splitlines()
            if only_test:  # Ultralytics wants train and val to exist: point them at the test images
                yaml = [line.replace("images/train", "images/test").replace("images/val", "images/test") for line in yaml]
            (dst / "data.yaml").write_text("\n".join([f"path: {dst}"] + yaml[1:]) + "\n")
            shutil.copyfile(base / stage / "manifest.csv", dst / "manifest.csv")
        for r in rows:
            label = Path(r["image"].replace("images/", "labels/", 1)).with_suffix(".txt")
            targets = []
            for arm in arms:
                dst = out / arm / stage
                if not (dst / label).exists():
                    _link(base / stage / label, dst / label)
                targets.append((arm, str(dst / r["image"])))
            tasks.append((int(r["window_start_frame"]), targets))
    parts = [tasks[i::workers * 8] for i in range(workers * 8)]
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(state,)) as pool:
        list(pool.map(_render, [p for p in parts if p]))
    print(f"rendered {len(tasks)} windows for arms {arms}", flush=True)

    # Same pixels where the arms share a view: channel 0 of C and D is the base gray image.
    rows = list(csv.DictReader(open(base / STAGES[0] / "manifest.csv")))[::97]
    for arm in arms:
        if ARMS[arm][0] == "uv":
            for r in rows:
                a = np.asarray(Image.open(base / STAGES[0] / r["image"]))
                b = np.asarray(Image.open(out / arm / STAGES[0] / r["image"]))[..., 0]
                if not np.array_equal(a, b):
                    raise RuntimeError(f"arm {arm}: channel 0 differs from the base gray image {r['image']}")
    print("channel 0 of the RGB arms equals the base gray images", flush=True)
    preview(state, base, out / "preview.png")


def preview(state, base, path, n=5):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in csv.DictReader(open(base / STAGES[1] / "manifest.csv")) if r["split"] == "val"]
    rng = np.random.default_rng(0)
    tr = open_cache(state["cache_path"], state["cache_shape"])
    names = ["uv", "white", "nbr", "neo"]
    fig, axes = plt.subplots(n, 4, figsize=(12, 3 * n))
    for i, k in enumerate(rng.choice(len(rows), n, replace=False)):
        ws = int(rows[k]["window_start_frame"])
        v = views(state, tr[ws - 1:ws + IMGSZ + 1])
        lab = (base / STAGES[1] / rows[k]["image"].replace("images/", "labels/", 1)).with_suffix(".txt")
        for j, name in enumerate(names):
            ax = axes[i, j]
            ax.imshow(v[name], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            for line in lab.read_text().splitlines():
                _, cx, cy, w, h = map(float, line.split())
                ax.add_patch(plt.Rectangle(((cx - w / 2) * IMGSZ - 0.5, (cy - h / 2) * IMGSZ - 0.5), w * IMGSZ, h * IMGSZ,
                                           fill=False, ec="lime", lw=0.6))
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(name)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)
    print(f"wrote {path}", flush=True)


# --------------------------------------------------------------------------- training

def no_rect_trainer():
    """DetectionTrainer that validates at 128 x 128.  Ultralytics hard-codes rect=True for the
    validation set during training, which pads 128 x 128 images to 160 x 160 with a grey frame."""
    from ultralytics.data import build_yolo_dataset
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.utils.torch_utils import unwrap_model

    class NoRectTrainer(DetectionTrainer):
        def build_dataset(self, img_path, mode="train", batch=None):
            gs = max(int(unwrap_model(self.model).stride.max()), 32)
            return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=False, stride=gs)

    return NoRectTrainer


def check_val_shape(trainer):
    shape = tuple(next(iter(trainer.test_loader))["img"].shape[-2:])
    if shape != (IMGSZ, IMGSZ):
        raise RuntimeError(f"validation images are {shape}, expected {(IMGSZ, IMGSZ)}")


def wandb_step_logger(project, name):
    """W&B callbacks that log against the training step (batch) instead of the epoch."""
    import wandb

    state = {"step": 0, "run": None}

    def on_pretrain_routine_start(trainer):
        state["run"] = wandb.init(project=project, name=name, config=dict(vars(trainer.args)), reinit="finish_previous")

    def on_train_batch_end(trainer):
        state["step"] += 1
        nb = max(len(trainer.train_loader), 1)
        log = {} if trainer.loss_items is None else trainer.label_loss_items(trainer.loss_items, prefix="train")
        log["epoch"] = trainer.epoch + ((state["step"] - 1) % nb + 1) / nb
        state["run"].log(log, step=state["step"])

    def on_fit_epoch_end(trainer):
        state["run"].log({**trainer.metrics, "epoch": trainer.epoch + 1}, step=state["step"])

    return {"on_pretrain_routine_start": on_pretrain_routine_start, "on_train_batch_end": on_train_batch_end,
            "on_fit_epoch_end": on_fit_epoch_end}, state


def box_metrics(m):
    return {"P": round(float(m.box.mp), 3), "R": round(float(m.box.mr), 3),
            "mAP50": round(float(m.box.map50), 3), "mAP50-95": round(float(m.box.map), 3)}


def spike_recall(model, base, arm_dir, device, conf=0.25):
    """Stage-1 test images hold one spike each: share found (a box with IoU >= 0.5 at conf >= 0.25),
    for all spikes and for the weakest 25 % (smallest contrast in the base gray image)."""
    stage = STAGES[0]
    rows = [r for r in csv.DictReader(open(base / stage / "manifest.csv"))
            if r["split"] == "test" and r["kind"] != "negative"]
    hit, contrast = [], []
    for k in range(0, len(rows), 256):
        part = rows[k:k + 256]
        results = model.predict([str(arm_dir / stage / r["image"]) for r in part], imgsz=IMGSZ, conf=conf,
                                device=device, save=False, verbose=False)
        for r, res in zip(part, results):
            lab = (base / stage / r["image"].replace("images/", "labels/", 1)).with_suffix(".txt")
            _, cx, cy, w, h = (float(v) * IMGSZ for v in lab.read_text().split())
            g = np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
            p = res.boxes.xyxy.cpu().numpy().reshape(-1, 4)
            iw = np.clip(np.minimum(p[:, 2], g[2]) - np.maximum(p[:, 0], g[0]), 0, None)
            ih = np.clip(np.minimum(p[:, 3], g[3]) - np.maximum(p[:, 1], g[1]), 0, None)
            union = (p[:, 2] - p[:, 0]) * (p[:, 3] - p[:, 1]) + w * h - iw * ih
            hit.append(bool(len(p)) and bool((iw * ih / union >= 0.5).any()))
            gray = np.asarray(Image.open(base / stage / r["image"])).astype(np.int16)
            y0, y1, x0, x1 = (int(round(v)) for v in (g[1], g[3], g[0], g[2]))
            contrast.append(int(np.abs(gray[y0:y1, x0:x1] - 128).max()))
    hit, contrast = np.array(hit), np.array(contrast)
    weak = contrast <= np.percentile(contrast, 25)
    return {"recall": round(float(hit.mean()), 3), "recall_weak25": round(float(hit[weak].mean()), 3)}


def train_arm(base, out, arm, seed, epochs, device, weights, wandb_project):
    from ultralytics import YOLO
    import ultralytics.utils.callbacks.wb as ultralytics_wb

    ultralytics_wb.callbacks.clear()  # this script logs to W&B itself, against the training step
    arm_dir = base if arm == "A" else out / arm
    tag = f"{arm}_seed{seed}_{epochs}ep"
    project = str(out / "runs")
    model = YOLO(weights)
    model.add_callback("on_train_start", check_val_shape)
    callbacks, wb = wandb_step_logger(wandb_project, tag)
    for event, fn in callbacks.items():
        model.add_callback(event, fn)
    model.train(
        data=str(arm_dir / STAGES[0] / "data.yaml"), single_cls=True, name=tag, exist_ok=True, trainer=no_rect_trainer(),
        imgsz=IMGSZ, epochs=epochs, patience=epochs, batch=64, seed=seed, deterministic=True, device=device, workers=0,
        project=project, val=True, plots=False, cache="ram", rect=False,
        optimizer="AdamW", lr0=1e-3, lrf=0.01, cos_lr=True, momentum=0.9, weight_decay=5e-4,
        warmup_epochs=1.0, warmup_bias_lr=0.0,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, degrees=0.0, translate=0.0, scale=0.0, shear=0.0, perspective=0.0,
        flipud=0.0, fliplr=0.0, bgr=0.0, mosaic=0.0, mixup=0.0, cutmix=0.0, copy_paste=0.0, erasing=0.0,
        close_mosaic=0, augment=False,
    )
    in_training = float(model.trainer.metrics["metrics/mAP50(B)"])  # the trainer's final validation of best.pt
    best = YOLO(Path(model.trainer.save_dir) / "weights" / "best.pt")

    def val(stage, split):
        return box_metrics(best.val(data=str(arm_dir / stage / "data.yaml"), split=split, imgsz=IMGSZ, batch=64,
                                    device=device, workers=0, single_cls=True, plots=False, verbose=False, rect=False,
                                    project=project, name=f"{tag}_{split}_{stage}", exist_ok=True))

    result = {"arm": arm, "channels": ARMS[arm], "seed": seed, "epochs": epochs,
              "val_stage1": val(STAGES[0], "val"), "test_stage1": val(STAGES[0], "test"),
              "test_stage3": val(STAGES[1], "test"), **spike_recall(best, base, arm_dir, device)}
    wb["run"].summary.update({f"{k}/{m}": v for k, r in result.items() if isinstance(r, dict) for m, v in r.items()})
    wb["run"].summary.update({"recall": result["recall"], "recall_weak25": result["recall_weak25"]})
    wb["run"].finish()
    if abs(in_training - result["val_stage1"]["mAP50"]) > 0.01:
        raise RuntimeError(f"{tag}: validation inside training gave mAP50 {in_training:.3f}, the same weights "
                           f"validated alone give {result['val_stage1']['mAP50']:.3f}")
    return tag, result


def table(results):
    print(f"\n{'run':16s} {'channels':16s} | stage 1 test: {'mAP50':>6s} {'mAP50-95':>8s} {'recall':>7s} {'weak25%':>8s} "
          f"| stage 3 test: {'mAP50':>6s} {'mAP50-95':>8s} {'P':>6s} {'R':>6s}")
    for tag, r in sorted(results.items()):
        s1, s3 = r["test_stage1"], r["test_stage3"]
        print(f"{tag:16s} {','.join(r['channels']):16s} |               {s1['mAP50']:6.3f} {s1['mAP50-95']:8.3f} "
              f"{r['recall']:7.3f} {r['recall_weak25']:8.3f} |               {s3['mAP50']:6.3f} {s3['mAP50-95']:8.3f} "
              f"{s3['P']:6.3f} {s3['R']:6.3f}")
    by_arm = {}
    for r in results.values():
        by_arm.setdefault(r["arm"], []).append(r["test_stage3"]["mAP50"])
    if any(len(v) > 1 for v in by_arm.values()):
        print("\nstage 3 test mAP50 over seeds: " + ", ".join(
            f"{a}: {np.mean(v):.3f} +- {np.std(v):.3f} (n={len(v)})" for a, v in sorted(by_arm.items())))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("base_build", type=Path, help="a gray build of nwb_yolo_dataset.py with stage 3")
    p.add_argument("--render", action="store_true")
    p.add_argument("--train", action="store_true")
    p.add_argument("--arms", nargs="+", choices=list(ARMS), default=list(ARMS))
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--device", default="mps")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    p.add_argument("--weights", default=str(HERE / "yolo26n.pt"))
    p.add_argument("--wandb-project", default="encoding_test", help="W&B project the runs are logged to")
    a = p.parse_args()
    base = a.base_build.resolve()
    out = base.parent / f"{base.name}_arms"
    results_path = out / "results.json"

    if a.render:
        render(base, out, a.arms, a.workers)
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    if a.train:
        import wandb

        wandb.login()  # uses the saved API key (~/.netrc or WANDB_API_KEY); asks for it once otherwise
        for seed in a.seeds:
            for arm in a.arms:
                if f"{arm}_seed{seed}_{a.epochs}ep" in results:
                    continue
                tag, result = train_arm(base, out, arm, seed, a.epochs, a.device, a.weights, a.wandb_project)
                results[tag] = result
                results_path.write_text(json.dumps(results, indent=1))
                table(results)
    if results:
        table(results)


if __name__ == "__main__":
    main()
