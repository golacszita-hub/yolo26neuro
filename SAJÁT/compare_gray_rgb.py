"""Compare a "gray" and an RGB build ("rgb3" or "rgb_feat") of nwb_yolo_dataset.py (same Config except image_mode).

Part 1 (always): checks the two builds hold the same windows and labels, that the RGB R channel
equals the gray image, and measures per channel how readable spikes are:
  * noise_std     gray-level std of the negatives (pure noise), i.e. how much texture noise has
  * box_contrast  median over boxes of the largest |pixel - baseline| in the box (baseline 128 for
                  signed channels, 0 for the unsigned NEO energy channel B of rgb_feat)
  * snr           box_contrast / noise_std
  * weak_snr      the same on the weakest 25 % of boxes (by gray contrast): do small units survive?
  * saturated     share of pixels at 0 or 255
plus the correlation between channels (how much new information G and B add) and a figure.

Part 2 (--train): trains both builds with identical settings (class-agnostic and 100-class) and tests
each on its stage's test split and on stage 3.  No augmentation at all, rect=False, and a fixed
optimizer / learning rate (AdamW, lr0 1e-3, cosine to 1e-5, 1 warmup epoch) instead of
Ultralytics' "auto" choice.  W&B logs against the training step (batch).

    python compare_gray_rgb.py GRAY_BUILD RGB_BUILD [--stage stage1_single_spike] [--train --epochs 50 --patience 15]
                               [--weights yolo26n.pt] [--wandb-project gray_vs_rgb_v2]
    (run with PYTHONPATH=<repo root> so the local ultralytics fork is imported)
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

SPLITS = ("train", "val", "test")
PRETRAINED = Path(__file__).parent / "runs/detect/spike_yolo_overfit/mean_waveforms_100_spike_detection-6/weights/best.pt"
CHANNEL_TITLES = {
    "rgb3": ["R (= gray)", "G linear", "B asinh high knee"],
    "rgb_feat": ["R (= gray)", "G spatial neighbour mean", "B NEO energy"],
}


def image_mode(build):
    return json.loads((build / "preprocessing.json").read_text())["config"]["image_mode"]


def load_manifest(stage_dir):
    return list(csv.DictReader(open(stage_dir / "manifest.csv", encoding="utf-8")))


def read_boxes(label_path, W=128):
    boxes = []
    for line in label_path.read_text().splitlines():
        _, cx, cy, w, h = map(float, line.split())
        x0, x1 = int(round((cx - w / 2) * W)), int(round((cx + w / 2) * W))
        y0, y1 = int(round((cy - h / 2) * W)), int(round((cy + h / 2) * W))
        boxes.append((x0, y0, x1, y1))
    return boxes


def check_same(gray_stage, rgb_stage):
    mg, mr = load_manifest(gray_stage), load_manifest(rgb_stage)
    assert mg == mr, f"{gray_stage.name}: manifests differ (different windows), compare builds with the same Config"
    for r in mg:
        lab = Path(r["image"].replace("images/", "labels/", 1)).with_suffix(".txt")
        assert (gray_stage / lab).read_text() == (rgb_stage / lab).read_text(), lab
    return mg


def channel_stats(gray_stage, rgb_stage, rows, split, mode="rgb3"):
    rows = [r for r in rows if r["split"] == split]
    names = ["gray", "R", "G", "B"]
    baseline = {n: 0.0 if (mode == "rgb_feat" and n == "B") else 128.0 for n in names}
    noise = {n: [] for n in names}
    contrast = {n: [] for n in names}
    sat = {n: [0, 0] for n in names}
    corr_px = []
    r_equals_gray = True
    for r in rows:
        g = np.asarray(Image.open(gray_stage / r["image"]))
        rgb = np.asarray(Image.open(rgb_stage / r["image"]))
        assert g.ndim == 2 and rgb.shape == g.shape + (3,), (g.shape, rgb.shape)
        r_equals_gray &= bool(np.array_equal(rgb[..., 0], g))
        chans = {"gray": g, "R": rgb[..., 0], "G": rgb[..., 1], "B": rgb[..., 2]}
        for n, c in chans.items():
            sat[n][0] += int(((c == 0) | (c == 255)).sum()); sat[n][1] += c.size
        if r["kind"] == "negative":
            for n, c in chans.items():
                noise[n].append(float(c.std()))
            continue
        lab = Path(r["image"].replace("images/", "labels/", 1)).with_suffix(".txt")
        for x0, y0, x1, y1 in read_boxes(rgb_stage / lab):
            for n, c in chans.items():
                contrast[n].append(float(np.abs(c[y0:y1, x0:x1].astype(np.float32) - baseline[n]).max()))
        if len(corr_px) < 200:
            corr_px.append(rgb.reshape(-1, 3)[::7].astype(np.float32))
    contrast = {n: np.array(v) for n, v in contrast.items()}
    weak = contrast["gray"] <= np.percentile(contrast["gray"], 25)
    out = {}
    for n in names:
        ns = float(np.median(noise[n])) if noise[n] else float("nan")
        out[n] = {
            "noise_std": round(ns, 2),
            "box_contrast": round(float(np.median(contrast[n])), 1),
            "snr": round(float(np.median(contrast[n])) / ns, 1),
            "weak_snr": round(float(np.median(contrast[n][weak])) / ns, 1),
            "saturated_%": round(100 * sat[n][0] / sat[n][1], 3),
        }
    px = np.concatenate(corr_px)
    cc = np.corrcoef(px.T)
    return out, r_equals_gray, {"R-G": round(cc[0, 1], 3), "R-B": round(cc[0, 2], 3), "G-B": round(cc[1, 2], 3)}, len(rows)


def figure(gray_stage, rgb_stage, rows, path, n=5, seed=0, mode="rgb3"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    pos = [r for r in rows if r["split"] == "val" and r["kind"] != "negative"]
    neg = [r for r in rows if r["split"] == "val" and r["kind"] == "negative"]
    pick = [pos[i] for i in rng.choice(len(pos), n - 1, replace=False)] + ([neg[0]] if neg else [])
    cols = ["gray", *CHANNEL_TITLES[mode], "RGB composite"]
    fig, axes = plt.subplots(len(pick), len(cols), figsize=(3 * len(cols), 3 * len(pick)))
    for i, r in enumerate(pick):
        g = np.asarray(Image.open(gray_stage / r["image"]))
        rgb = np.asarray(Image.open(rgb_stage / r["image"]))
        lab = Path(r["image"].replace("images/", "labels/", 1)).with_suffix(".txt")
        for j, img in enumerate([g, rgb[..., 0], rgb[..., 1], rgb[..., 2], rgb]):
            ax = axes[i, j]
            ax.imshow(img, cmap=None if img.ndim == 3 else "gray", vmin=0, vmax=255, interpolation="nearest")
            for x0, y0, x1, y1 in read_boxes(rgb_stage / lab):
                ax.add_patch(plt.Rectangle((x0 - 0.5, y0 - 0.5), x1 - x0, y1 - y0, fill=False, ec="lime", lw=0.8))
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(cols[j])
        axes[i, 0].set_ylabel(f"{r['kind']} cls {r['classes'] or '-'}", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def wandb_step_logger(project, name):
    """W&B callbacks that log against the training step (= batch), not the epoch.

    Every batch: train losses and learning rates.  Every epoch end: validation metrics at the
    step reached.  `epoch` (fractional) is logged too, so the x-axis can be switched in the UI.
    """
    import wandb

    state = {"step": 0, "run": None}

    def on_pretrain_routine_start(trainer):
        state["run"] = wandb.init(project=project, name=name, config=dict(vars(trainer.args)), reinit="finish_previous")

    def on_train_batch_end(trainer):
        state["step"] += 1
        nb = max(len(trainer.train_loader), 1)
        log = {} if trainer.loss_items is None else trainer.label_loss_items(trainer.loss_items, prefix="train")
        log.update({f"lr/pg{i}": g["lr"] for i, g in enumerate(trainer.optimizer.param_groups)})
        log["epoch"] = trainer.epoch + ((state["step"] - 1) % nb + 1) / nb
        state["run"].log(log, step=state["step"])

    def on_fit_epoch_end(trainer):
        state["run"].log({**trainer.metrics, "epoch": trainer.epoch + 1}, step=state["step"])

    def on_train_end(trainer):  # trainer.metrics now hold the final validation of best.pt
        state["run"].summary.update({f"best/{k}": v for k, v in trainer.metrics.items()})

    callbacks = {"on_pretrain_routine_start": on_pretrain_routine_start, "on_train_batch_end": on_train_batch_end,
                 "on_fit_epoch_end": on_fit_epoch_end, "on_train_end": on_train_end}
    return callbacks, state


def no_rect_trainer():
    """DetectionTrainer whose validation set is also built with rect=False.

    Ultralytics hard-codes rect=True for the validation set during training
    (models/yolo/detect/train.py), and rect batches get half a stride of padding
    (data/base.py), so 128x128 images were validated padded to 160x160 with a grey-114 frame.
    """
    from ultralytics.data import build_yolo_dataset
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.utils.torch_utils import unwrap_model

    class NoRectTrainer(DetectionTrainer):
        def build_dataset(self, img_path, mode="train", batch=None):
            gs = max(int(unwrap_model(self.model).stride.max()), 32)
            return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=False, stride=gs)

    return NoRectTrainer


def train_and_test(builds, stage, epochs, patience, device, out_project, wandb_project="gray_vs_rgb_v2",
                   weights=PRETRAINED):
    import ultralytics.utils.callbacks.wb as ultralytics_wb
    from ultralytics import YOLO

    # Switch off Ultralytics' own W&B logger (step = epoch) in this process only; the global
    # setting, and so the notebook, stay unchanged.  wandb_step_logger replaces it.
    ultralytics_wb.callbacks.clear()

    common = dict(
        imgsz=128, epochs=epochs, batch=64, seed=42, deterministic=True, device=device, workers=0,
        patience=patience, project=str(out_project), val=True, plots=False, cache="ram", rect=False,
        # Learning rate fixed: optimizer="auto" picked MuSGD lr 0.01 (best epoch right after warmup,
        # validation jumping +-0.07); AdamW 1e-3 with a cosine decay to 1e-5 fine-tunes calmly.
        optimizer="AdamW", lr0=1e-3, lrf=0.01, cos_lr=True, momentum=0.9, weight_decay=5e-4,
        warmup_epochs=1.0, warmup_bias_lr=0.0,
        # No augmentation at all (time placement of the anchor spike is varied in the dataset itself).
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, degrees=0.0, translate=0.0, scale=0.0, shear=0.0, perspective=0.0,
        flipud=0.0, fliplr=0.0, bgr=0.0, mosaic=0.0, mixup=0.0, cutmix=0.0, copy_paste=0.0, erasing=0.0,
        close_mosaic=0, augment=False,
    )
    results = {}
    Path(out_project).mkdir(parents=True, exist_ok=True)  # Ultralytics itself writes under runs/detect/<project>
    # Agnostic gray vs RGB first: that pair answers "does RGB help" soonest.
    for single_cls in (True, False):
        for mode, root in builds.items():
            data = str(root / stage / "data.yaml")
            tag = f"{stage}_{mode}_{'agnostic' if single_cls else 'multiclass'}_{epochs}ep"
            model = YOLO(weights)
            callbacks, wb_state = wandb_step_logger(wandb_project, tag)
            for event, fn in callbacks.items():
                model.add_callback(event, fn)
            model.train(data=data, single_cls=single_cls, name=tag, exist_ok=True, trainer=no_rect_trainer(), **common)
            best = YOLO(Path(model.trainer.save_dir) / "weights" / "best.pt")
            # Test on the stage's own test split and on stage 3 (regular tiles = what deployment sees).
            for test_stage in dict.fromkeys([stage, "stage3_any_spike_count"]):
                m = best.val(data=str(root / test_stage / "data.yaml"), split="test", imgsz=128, batch=64, device=device,
                             workers=0, single_cls=single_cls, plots=False, verbose=False, rect=False, project=str(out_project),
                             name=f"{tag}_test_{test_stage}", exist_ok=True)
                r = {"P": round(float(m.box.mp), 3), "R": round(float(m.box.mr), 3),
                     "mAP50": round(float(m.box.map50), 3), "mAP50-95": round(float(m.box.map), 3)}
                results[f"{tag} -> test {test_stage}"] = r
                wb_state["run"].summary.update({f"test_{test_stage}/{k}": v for k, v in r.items()})
            wb_state["run"].finish()
            (Path(out_project) / "results_so_far.json").write_text(json.dumps(results, indent=1))
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("gray_build", type=Path)
    p.add_argument("rgb_build", type=Path)
    p.add_argument("--stage", default="stage1_single_spike")
    p.add_argument("--split", default="train", help="split used for the channel statistics")
    p.add_argument("--train", action="store_true")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--skip-stats", action="store_true", help="skip the dataset comparison, only train")
    p.add_argument("--device", default="mps")
    p.add_argument("--weights", default=str(PRETRAINED), help="start weights (default: the mean-waveform model)")
    p.add_argument("--wandb-project", default="gray_vs_rgb_v2")
    a = p.parse_args()
    mode = image_mode(a.rgb_build)
    out_dir = a.rgb_build / "compare_with_gray"
    out_dir.mkdir(exist_ok=True)
    report = {"gray_build": str(a.gray_build), "rgb_build": str(a.rgb_build), "stages": {}}

    for stage_dir in [] if a.skip_stats else sorted(a.gray_build.glob("stage*")):
        rows = check_same(stage_dir, a.rgb_build / stage_dir.name)
        stats, r_eq, corr, n = channel_stats(stage_dir, a.rgb_build / stage_dir.name, rows, a.split, mode)
        report["stages"][stage_dir.name] = {"images": n, "R_equals_gray": r_eq, "channel_correlation": corr, "channels": stats}
        print(f"\n{stage_dir.name} ({a.split}, {n} images): same windows+labels OK, R == gray: {r_eq}, corr {corr}")
        print(f"  {'':5s} {'noise_std':>9s} {'contrast':>9s} {'snr':>6s} {'weak_snr':>9s} {'sat %':>7s}")
        for ch, s in stats.items():
            print(f"  {ch:5s} {s['noise_std']:9.2f} {s['box_contrast']:9.1f} {s['snr']:6.1f} {s['weak_snr']:9.1f} {s['saturated_%']:7.3f}")
        figure(stage_dir, a.rgb_build / stage_dir.name, rows, out_dir / f"{stage_dir.name}_gray_vs_rgb.png", mode=mode)

    if a.train:
        res = train_and_test({"gray": a.gray_build, mode: a.rgb_build}, a.stage, a.epochs, a.patience, a.device,
                             Path("runs") / a.wandb_project, a.wandb_project, a.weights)
        report["test_results"] = res
        print(f"\ntest split, {a.stage}:")
        for tag, r in res.items():
            print(f"  {tag:75s} {r}")

    if a.skip_stats and (out_dir / "report.json").exists():  # keep the stats written by an earlier run
        report = {**json.loads((out_dir / "report.json").read_text()), **{k: v for k, v in report.items() if k != "stages"}}
    (out_dir / "report.json").write_text(json.dumps(report, indent=1))
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()
