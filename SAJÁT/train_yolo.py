"""Train YOLO26n on one stage of a dataset build. Change the settings below and run the file."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # the local ultralytics fork

import torch
import wandb
from ultralytics import YOLO, settings
from ultralytics.data import build_yolo_dataset
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils.torch_utils import unwrap_model

# ---- settings ----
# Build folder inside DATASETS/ (it differs per computer); its name ends in _gray or _rgb.
BUILD = HERE.parent.parent / "DATASETS/nwb_asinh_stages_newdataset/build_20261121T161459Z_rgb"
STAGE = "stage3_any_spike_count"  # "stage1_single_spike" | "stage2_up_to_3_spikes" | "stage3_any_spike_count"
SINGLE_CLS = False  # True: spike / no spike, False: one class per unit
EPOCHS = 100
PATIENCE = 20       # stop when validation has not improved for this many epochs
BATCH = 512         # what the Nov 2026 Windows runs used; lower it if memory runs out
WEIGHTS = HERE / "yolo26n.pt"  # or a trained best.pt to continue from (weights are not in git)
DEVICE = 0 if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"  # NVIDIA, Mac, else CPU

MODE = BUILD.name.rsplit("_", 1)[-1]  # "gray" or "rgb", from the build folder name
DATA = BUILD / STAGE / "data.yaml"
NAME = f"{STAGE}_{MODE}_{'agnostic' if SINGLE_CLS else 'multiclass'}_{EPOCHS}ep"


# Needed for no rect: Ultralytics always builds the validation set with rect=True during training.
class NoRectTrainer(DetectionTrainer):
    def build_dataset(self, img_path, mode="train", batch=None):
        gs = max(int(unwrap_model(self.model).stride.max()), 32)
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=False, stride=gs)


# Log to W&B by batch: Ultralytics' own W&B callback logs only once per epoch, so it is turned off here.
settings.update({"wandb": False})
wandb.init(project="spikeyolo", name=NAME)
step = 0  # batches seen so far


def log_batch(trainer):
    global step
    step += 1
    if trainer.loss_items is not None:  # None when a batch was dropped after a CUDA out-of-memory
        lr = {f"lr/pg{i}": g["lr"] for i, g in enumerate(trainer.optimizer.param_groups)}
        wandb.log({**trainer.label_loss_items(trainer.loss_items, prefix="batch"), **lr, "epoch": trainer.epoch + 1}, step=step)


def log_epoch(trainer):  # epoch-average train losses and validation metrics, at the epoch's last batch
    wandb.log({**trainer.label_loss_items(trainer.tloss, prefix="train"), **trainer.metrics}, step=step)


model = YOLO(WEIGHTS)
model.add_callback("on_train_batch_end", log_batch)
model.add_callback("on_fit_epoch_end", log_epoch)
model.add_callback("on_train_end", lambda trainer: wandb.finish())

model.train(
    data=str(DATA), 
    single_cls=SINGLE_CLS, 
    epochs=EPOCHS,
    patience=PATIENCE,
    trainer=NoRectTrainer,
    imgsz=128, 
    batch=BATCH,
    device=DEVICE,
    workers=0, 
    seed=0,
    project=str(HERE / "runs/spikeyolo"), 
    name=NAME, 
    exist_ok=True,
    plots=True, 
    rect=False,
    # no augmentation
    hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, degrees=0.0, translate=0.0, scale=0.0, shear=0.0, perspective=0.0,
    flipud=0.0, fliplr=0.0, bgr=0.0, mosaic=0.0, mixup=0.0, cutmix=0.0, copy_paste=0.0, erasing=0.0, close_mosaic=0,
)
