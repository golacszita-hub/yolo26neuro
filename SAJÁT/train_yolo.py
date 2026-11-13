"""Train YOLO26n on one or more builds of nwb_yolo_dataset.py with identical settings.

    python train_yolo.py GRAY_BUILD RGB_BUILD --epochs 20 --weights path/to/best.pt

For every build: train on `--stage` (default stage 1), then validate the best weights again on
their own (val split) and test them on the stage-1 and stage-3 test splits.

* Weights & Biases is always on: `wandb.login()` uses the saved API key, losses are logged per
  training step, and the validation / test numbers and plots are added to the run.
* Validation is never rectangular: Ultralytics hard-codes rect=True for the validation set
  during training, which pads 128 x 128 images to 160 x 160 with a grey frame.  The trainer
  below builds it with rect=False, every `val` call passes rect=False, and the script stops if
  the validation images are not 128 x 128 or if the validation inside training and the
  separate validation of the same weights disagree.
* plots=True: Ultralytics saves the PR / F1 curves, the confusion matrix and pictures of
  validation batches with labels and predictions in each run folder; they are uploaded to W&B.
* No augmentation, AdamW 1e-3 with cosine decay, fixed number of epochs.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # the local ultralytics fork

IMGSZ = 128
TEST_STAGES = ("stage1_single_spike", "stage3_any_spike_count")


def no_rect_trainer():
    from ultralytics.data import build_yolo_dataset
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.utils.torch_utils import unwrap_model

    class NoRectTrainer(DetectionTrainer):
        def build_dataset(self, img_path, mode="train", batch=None):
            gs = max(int(unwrap_model(self.model).stride.max()), 32)
            return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=False, stride=gs)

    return NoRectTrainer


def check_val_loader(trainer):
    """Stop before the first epoch if the validation set is rectangular or padded."""
    dataset = trainer.test_loader.dataset
    shape = tuple(next(iter(trainer.test_loader))["img"].shape[-2:])
    if dataset.rect or shape != (IMGSZ, IMGSZ):
        raise RuntimeError(f"validation set: rect={dataset.rect}, image shape {shape}; expected rect=False, {(IMGSZ, IMGSZ)}")
    print(f"validation set: rect={dataset.rect}, images {shape}", flush=True)


def wandb_step_logger(project, name, config):
    """W&B callbacks that log against the training step (batch) instead of the epoch."""
    import wandb

    state = {"step": 0, "run": None}

    def on_pretrain_routine_start(trainer):
        state["run"] = wandb.init(project=project, name=name, config={**dict(vars(trainer.args)), **config},
                                  reinit="finish_previous")

    def on_train_batch_end(trainer):
        state["step"] += 1
        nb = max(len(trainer.train_loader), 1)
        log = {} if trainer.loss_items is None else trainer.label_loss_items(trainer.loss_items, prefix="train")
        log.update({f"lr/pg{i}": g["lr"] for i, g in enumerate(trainer.optimizer.param_groups)})
        log["epoch"] = trainer.epoch + ((state["step"] - 1) % nb + 1) / nb
        state["run"].log(log, step=state["step"])

    def on_fit_epoch_end(trainer):
        state["run"].log({**trainer.metrics, "epoch": trainer.epoch + 1}, step=state["step"])

    return {"on_pretrain_routine_start": on_pretrain_routine_start, "on_train_batch_end": on_train_batch_end,
            "on_fit_epoch_end": on_fit_epoch_end}, state


def data_yaml(build, stage):
    """data.yaml of a stage with `path:` pointing at where the build is now (builds move between machines)."""
    src = build / stage / "data.yaml"
    lines = [f"path: {src.parent}" if line.startswith("path:") else line for line in src.read_text().splitlines()]
    out = HERE / "runs" / "data" / f"{build.name}_{stage}.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    return str(out)


def box_metrics(m):
    return {"P": round(float(m.box.mp), 4), "R": round(float(m.box.mr), 4),
            "mAP50": round(float(m.box.map50), 4), "mAP50-95": round(float(m.box.map), 4)}


def log_plots(run, folder, prefix):
    import wandb

    images = sorted(p for p in Path(folder).glob("*") if p.suffix in (".png", ".jpg"))
    if images:
        run.log({f"{prefix}/{p.stem}": wandb.Image(str(p)) for p in images})


def train_build(build, a):
    import ultralytics.utils.callbacks.wb as ultralytics_wb
    from ultralytics import YOLO

    ultralytics_wb.callbacks.clear()  # this script logs to W&B itself, against the training step
    mode = json.loads((build / "preprocessing.json").read_text())["config"]["image_mode"]
    tag = f"{a.stage}_{mode}_{'multiclass' if a.multiclass else 'agnostic'}_{a.epochs}ep"
    project = str(HERE / "runs" / a.wandb_project)
    model = YOLO(a.weights)
    model.add_callback("on_train_start", check_val_loader)
    callbacks, wb = wandb_step_logger(a.wandb_project, tag, {"build": str(build), "image_mode": mode,
                                                             "start_weights": str(a.weights)})
    for event, fn in callbacks.items():
        model.add_callback(event, fn)
    model.train(
        data=data_yaml(build, a.stage), single_cls=not a.multiclass, name=tag, exist_ok=True,
        trainer=no_rect_trainer(), imgsz=IMGSZ, epochs=a.epochs, patience=a.epochs, batch=64, seed=a.seed,
        deterministic=True, device=a.device, workers=0, project=project, val=True, plots=True, cache=a.cache, rect=False,
        optimizer="AdamW", lr0=1e-3, lrf=0.01, cos_lr=True, momentum=0.9, weight_decay=5e-4,
        warmup_epochs=1.0, warmup_bias_lr=0.0,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, degrees=0.0, translate=0.0, scale=0.0, shear=0.0, perspective=0.0,
        flipud=0.0, fliplr=0.0, bgr=0.0, mosaic=0.0, mixup=0.0, cutmix=0.0, copy_paste=0.0, erasing=0.0,
        close_mosaic=0, augment=False,
    )
    save_dir = Path(model.trainer.save_dir)
    in_training = float(model.trainer.metrics["metrics/mAP50(B)"])  # the trainer's final validation of best.pt
    best = YOLO(save_dir / "weights" / "best.pt")
    run = wb["run"]
    log_plots(run, save_dir, "train_plots")

    def evaluate(stage, split):
        name = f"{tag}_{split}_{stage}"
        m = best.val(data=data_yaml(build, stage), split=split, imgsz=IMGSZ, batch=64, device=a.device,
                     workers=0, single_cls=not a.multiclass, plots=True, verbose=False, rect=False,
                     project=project, name=name, exist_ok=True)
        r = box_metrics(m)
        run.summary.update({f"{split}_{stage}/{k}": v for k, v in r.items()})
        log_plots(run, Path(project) / name, f"{split}_{stage}")
        return r

    result = {"build": str(build), "image_mode": mode, "weights": str(save_dir / "weights" / "best.pt"),
              "val": evaluate(a.stage, "val")}
    for stage in TEST_STAGES:
        result[f"test_{stage}"] = evaluate(stage, "test")
    result["val_inside_training_mAP50"] = round(in_training, 4)
    run.summary["val_inside_training_mAP50"] = result["val_inside_training_mAP50"]
    run.finish()
    if abs(in_training - result["val"]["mAP50"]) > 0.01:
        raise RuntimeError(f"{tag}: validation inside training gave mAP50 {in_training:.4f}, the same weights "
                           f"validated alone give {result['val']['mAP50']:.4f}")
    return tag, result


def table(results):
    print(f"\n{'run':48s} {'split':32s} {'P':>7s} {'R':>7s} {'mAP50':>7s} {'mAP50-95':>9s}")
    for tag, r in results.items():
        for key in ["val"] + [f"test_{s}" for s in TEST_STAGES]:
            m = r[key]
            print(f"{tag:48s} {key:32s} {m['P']:7.4f} {m['R']:7.4f} {m['mAP50']:7.4f} {m['mAP50-95']:9.4f}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("builds", type=Path, nargs="+", help="build folders of nwb_yolo_dataset.py (gray and / or rgb)")
    p.add_argument("--stage", default="stage1_single_spike", help="stage to train on")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--weights", nargs="+", default=[str(HERE / "yolo26n.pt")],
                   help="start weights: one file for all builds, or one per build in the same order")
    p.add_argument("--multiclass", action="store_true", help="one class per unit (default: one class, spike)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="mps")
    p.add_argument("--wandb-project", default="spikeyolo")
    p.add_argument("--cache", default="ram", help="Ultralytics image cache: ram (default), disk or False; "
                   "use False for stage 2 / 3 (135k+ images do not fit in 18 GB RAM)")
    a = p.parse_args()
    a.cache = False if a.cache.lower() == "false" else a.cache
    if len(a.weights) not in (1, len(a.builds)):
        p.error(f"--weights: give 1 file or {len(a.builds)} (one per build), got {len(a.weights)}")
    weights = a.weights * len(a.builds) if len(a.weights) == 1 else a.weights

    import wandb

    wandb.login()  # uses the saved API key (~/.netrc or WANDB_API_KEY); asks for it once otherwise
    results = {}
    out = HERE / "runs" / a.wandb_project / "results.json"
    for build, a.weights in zip(a.builds, weights):
        tag, result = train_build(build.resolve(), a)
        results[tag] = result
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({**(json.loads(out.read_text()) if out.exists() else {}), **results}, indent=1))
        table(results)
    print(f"\nresults: {out}")


if __name__ == "__main__":
    main()
