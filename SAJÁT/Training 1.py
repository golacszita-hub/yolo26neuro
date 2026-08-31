import json
import wandb
from ultralytics import YOLO, settings


with open("dataset_info.json") as f:
    info = json.load(f)
nc = info["nc"]

yaml_content = f"""
path: /Users/Zita/Documents/EGYETEM/Önlab/yolo26/dataset_test2
train: images/train
val: images/val
test: images/test
nc: {nc}
names: {list(range(nc))}
"""
with open("spikeyolo_dataset.yaml", "w") as f:
    f.write(yaml_content)


model = YOLO("yolo26n.pt")


model.train(
    data="spikeyolo_dataset.yaml",
    imgsz=128,
    epochs=100,
    batch=-1,
    device="mps",
    workers=4,
    project="spikeyolo",
    name="run4",
    patience=15,

    # no augmentation
    hsv_h=0,
    hsv_s=0,
    hsv_v=0,
    fliplr=0,
    flipud=0,
    degrees=0,
    translate=0,
    scale=0,
    shear=0,
    perspective=0,
    mixup=0,
    copy_paste=0,
    mosaic=0,

    # useful for small objects
    close_mosaic=0
)