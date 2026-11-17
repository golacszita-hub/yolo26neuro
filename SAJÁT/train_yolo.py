"""Train YOLO26n on one stage of the gray or rgb dataset. Change the settings below and run the file."""
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
MODE = "rgb"                  # "gray" | "rgb"
STAGE = "stage2_up_to_3_spikes"  # "stage1_single_spike" | "stage2_up_to_3_spikes" | "stage3_any_spike_count"
SINGLE_CLS = False             # True: spike / no spike, False: one class per unit
EPOCHS = 20
WEIGHTS = HERE / "runs/spikeyolo/stage2_up_to_3_spikes_rgb_agnostic_15ep/weights/best.pt"  # not in git: copy it over, or use HERE / "yolo26n.pt"
DEVICE = 0 if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"  # NVIDIA, Mac, else CPU

DATASETS = {
    "gray": HERE.parent.parent / "DATASETS/nwb_asinh_stages/build_20261112T160151Z_gray",
    "rgb": HERE.parent.parent / "DATASETS/nwb_asinh_stages/build_20261112T161035Z_rgb",
}
DATA = DATASETS[MODE] / STAGE / "data.yaml"
NAME = f"{STAGE}_{MODE}_{'agnostic' if SINGLE_CLS else 'multiclass'}_{EPOCHS}ep"


# Needed for no rect: Ultralytics always builds the validation set with rect=True during training.
class NoRectTrainer(DetectionTrainer):
    def build_dataset(self, img_path, mode="train", batch=None):
        gs = max(int(unwrap_model(self.model).stride.max()), 32)
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=False, stride=gs)


settings.update({"wandb": True})
wandb.init(project="spikeyolo", name=NAME)  # Ultralytics' W&B callback logs into this run and closes it

YOLO(WEIGHTS).train(
    data=str(DATA), 
    single_cls=SINGLE_CLS, 
    epochs=EPOCHS, 
    trainer=NoRectTrainer,
    imgsz=128, 
    batch=64, 
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
