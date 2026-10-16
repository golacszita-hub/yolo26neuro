"""Compare scratch predictions and Ultralytics validation with identical 128px inputs."""

from pathlib import Path
import csv
import sys

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "runs/detect/spike_yolo_overfit/mean_waveforms_100_spike_detection-5/weights/best.pt"
DATASET = ROOT / "yolo26.09_mean_waveform_overfit/YOLO26_mean_waveform_detection"
CONF = 0.25
NMS_IOU = 0.45
MATCH_IOU = 0.45
IMGSZ = 128
BATCH = 8


def pairwise_iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU for xyxy boxes."""
    if not len(boxes_a) or not len(boxes_b):
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=float)
    top_left = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    bottom_right = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    wh = np.maximum(bottom_right - top_left, 0)
    intersection = wh[..., 0] * wh[..., 1]
    area_a = np.prod(np.maximum(boxes_a[:, 2:] - boxes_a[:, :2], 0), axis=1)
    area_b = np.prod(np.maximum(boxes_b[:, 2:] - boxes_b[:, :2], 0), axis=1)
    union = area_a[:, None] + area_b[None, :] - intersection
    return intersection / np.maximum(union, 1e-12)


def normalized_cm(matrix: np.ndarray) -> np.ndarray:
    """Normalize each true-class column, as Ultralytics does for its plot."""
    totals = matrix.sum(axis=0, keepdims=True)
    normalized = np.divide(
        matrix,
        totals,
        out=np.zeros_like(matrix, dtype=float),
        where=totals != 0,
    )
    normalized[normalized < 0.005] = np.nan
    return normalized


def write_cm_csv(path: Path, matrix: np.ndarray, class_names: list[str]) -> None:
    """Write a labeled raw-count confusion matrix."""
    labels = [*class_names, "background"]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["predicted\\true", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *row.tolist()])


def main() -> None:
    image_dir = DATASET / "images" / "val"
    label_dir = DATASET / "labels" / "val"
    image_paths = sorted(image_dir.glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No validation PNG images found in {image_dir}")
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")

    output_dir = MODEL_PATH.parent.parent / "prediction_confusion_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(MODEL_PATH))
    n_classes = len(model.names)
    class_names = [model.names[i] for i in range(n_classes)]
    background = n_classes

    # Generate the reference confusion matrix with the Ultralytics validator.
    yolo_metrics = model.val(
        data=str(DATASET / "data.yaml"),
        split="val",
        imgsz=IMGSZ,
        batch=BATCH,
        rect=False,
        conf=CONF,
        iou=NMS_IOU,
        plots=True,
        verbose=False,
        project=str(output_dir),
        name="ultralytics_eval",
        exist_ok=True,
    )
    yolo_cm = np.asarray(yolo_metrics.confusion_matrix.matrix, dtype=np.int64)

    # Separately predict, then construct the matrix below without Ultralytics' CM helper.
    predictions = model.predict(
        source=[str(path) for path in image_paths],
        imgsz=IMGSZ,
        batch=BATCH,
        conf=CONF,
        iou=NMS_IOU,
        max_det=300,
        rect=False,
        device="cpu",
        verbose=False,
    )
    if len(predictions) != len(image_paths):
        raise RuntimeError(f"Expected {len(image_paths)} predictions, got {len(predictions)}")

    scratch_cm = np.zeros((n_classes + 1, n_classes + 1), dtype=np.int64)
    for image_path, result in zip(image_paths, predictions):
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            raise FileNotFoundError(f"Missing label file: {label_path}")

        label_text = label_path.read_text().strip()
        rows = [list(map(float, line.split())) for line in label_text.splitlines() if line.strip()]
        if any(len(row) != 5 for row in rows):
            raise ValueError(f"Invalid YOLO label format in {label_path}")

        gt_classes = np.asarray([int(row[0]) for row in rows], dtype=int)
        if np.any((gt_classes < 0) | (gt_classes >= n_classes)):
            raise ValueError(f"Class ID outside [0, {n_classes - 1}] in {label_path}")

        height, width = result.orig_shape
        gt_boxes = np.asarray(
            [
                [
                    (xc - box_w / 2) * width,
                    (yc - box_h / 2) * height,
                    (xc + box_w / 2) * width,
                    (yc + box_h / 2) * height,
                ]
                for _, xc, yc, box_w, box_h in rows
            ],
            dtype=float,
        ).reshape(-1, 4)

        if result.boxes is None or not len(result.boxes):
            pred_boxes = np.empty((0, 4), dtype=float)
            pred_classes = np.empty(0, dtype=int)
        else:
            keep = np.asarray(result.boxes.conf) > CONF
            pred_boxes = np.asarray(result.boxes.xyxy)[keep]
            pred_classes = np.asarray(result.boxes.cls, dtype=int)[keep]

        ious = pairwise_iou(gt_boxes, pred_boxes)
        candidates = np.where(ious > MATCH_IOU)
        if len(candidates[0]):
            matches = np.column_stack(
                (np.stack(candidates, axis=1), ious[candidates])
            )
            # Same greedy one-to-one IoU matching order as Ultralytics.
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            matched_gt, matched_pred, _ = matches.transpose().astype(int)
        else:
            matched_gt = np.empty(0, dtype=int)
            matched_pred = np.empty(0, dtype=int)

        for gt_idx, pred_idx in zip(matched_gt, matched_pred):
            scratch_cm[pred_classes[pred_idx], gt_classes[gt_idx]] += 1
        for gt_idx, gt_class in enumerate(gt_classes):
            if gt_idx not in matched_gt:
                scratch_cm[background, gt_class] += 1
        for pred_idx, pred_class in enumerate(pred_classes):
            if pred_idx not in matched_pred:
                scratch_cm[pred_class, background] += 1

    if yolo_cm.shape != scratch_cm.shape:
        raise RuntimeError(f"Matrix shape mismatch: YOLO {yolo_cm.shape}, scratch {scratch_cm.shape}")

    difference = scratch_cm - yolo_cm
    exact_match = np.array_equal(scratch_cm, yolo_cm)
    print(f"Images: {len(image_paths)}")
    print(f"Ground-truth objects: {int(scratch_cm[:, :n_classes].sum())}")
    print(f"Predictions counted: {int(scratch_cm[:n_classes, :].sum())}")
    print(f"Raw integer matrices exactly equal: {exact_match}")
    print(f"Differing cells: {int(np.count_nonzero(difference))}")
    if not exact_match:
        for pred_idx, true_idx in np.argwhere(difference)[:20]:
            pred_name = "background" if pred_idx == background else class_names[pred_idx]
            true_name = "background" if true_idx == background else class_names[true_idx]
            print(
                f"  predicted={pred_name}, true={true_name}: "
                f"scratch={scratch_cm[pred_idx, true_idx]}, YOLO={yolo_cm[pred_idx, true_idx]}"
            )

    ticks = list(range(0, n_classes, 10)) + [background]
    tick_labels = [str(i) for i in range(0, n_classes, 10)] + ["Background"]
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), constrained_layout=True)
    for ax, matrix, title in zip(
        axes,
        (yolo_cm, scratch_cm),
        ("Ultralytics validation", "Scratch from predictions"),
    ):
        image = ax.imshow(
            normalized_cm(matrix),
            cmap="Blues",
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )
        ax.set(title=title, xlabel="True class", ylabel="Predicted class")
        ax.set_xticks(ticks, tick_labels, rotation=45, ha="right")
        ax.set_yticks(ticks, tick_labels)
    fig.colorbar(image, ax=axes, label="Fraction of true class", shrink=0.85)

    comparison_path = output_dir / "confusion_matrix_comparison.png"
    fig.savefig(comparison_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    scratch_path = output_dir / "scratch_confusion_matrix.npy"
    np.save(scratch_path, scratch_cm)
    yolo_csv = output_dir / "ultralytics_confusion_matrix.csv"
    scratch_csv = output_dir / "scratch_confusion_matrix.csv"
    write_cm_csv(yolo_csv, yolo_cm, class_names)
    write_cm_csv(scratch_csv, scratch_cm, class_names)
    print(f"Ultralytics plot: {output_dir / 'ultralytics_eval' / 'confusion_matrix.png'}")
    print(f"Side-by-side comparison: {comparison_path}")
    print(f"Raw count CSVs: {yolo_csv} and {scratch_csv}")
    print(f"Scratch NumPy matrix: {scratch_path}")


if __name__ == "__main__":
    main()
