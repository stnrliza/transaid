"""
YOLO Instance Segmentation Training Module for Dental Caries Detection.

This module trains and validates an Ultralytics YOLO instance segmentation model
(e.g., YOLOv8s-seg / YOLO11s-seg) on Near-Infrared Light Transillumination (NILT)
caries datasets using AdamW optimization, cosine learning rate decay, and tailored
geometric/photometric data augmentations.
"""

import os
from pathlib import Path
import torch
from ultralytics import YOLO

# ── PATHS & CONFIGURATION ───────────────────────────
BASE_DIR         = Path(__file__).resolve().parent
DATA_YAML        = Path(os.getenv("DATA_YAML", str(BASE_DIR / "data.yaml")))
OUTPUT_DIR       = Path(os.getenv("YOLO_OUTPUT_DIR", str(BASE_DIR / "runs")))
PRETRAINED_MODEL = os.getenv("PRETRAINED_MODEL", "yolov8s-seg.pt")
EPOCHS           = int(os.getenv("EPOCHS", "500"))
IMGSZ            = int(os.getenv("IMGSZ", "640"))
BATCH_SIZE       = int(os.getenv("BATCH_SIZE", "-1"))   # -1 enables automated batch sizing
SAVE_PERIOD      = int(os.getenv("SAVE_PERIOD", "50"))
DEVICE           = 0 if torch.cuda.is_available() else "cpu"
# ────────────────────────────────────────────────────


def train_model(data_yaml: Path = DATA_YAML,
                pretrained_model: str = PRETRAINED_MODEL,
                epochs: int = EPOCHS,
                imgsz: int = IMGSZ,
                batch_size: int = BATCH_SIZE,
                output_dir: Path = OUTPUT_DIR,
                device: str | int = DEVICE):
    """
    Train and validate a YOLO instance segmentation model on the specified dataset.

    Args:
        data_yaml: Path to the dataset configuration YAML file.
        pretrained_model: Initial model weights checkpoint name or file path.
        epochs: Number of training epochs.
        imgsz: Target image square resolution.
        batch_size: Training batch size (-1 for auto-batch).
        output_dir: Destination project output directory.
        device: Target compute device (GPU index or 'cpu').

    Returns:
        Ultralytics validation metrics object.
    """
    if not data_yaml.exists():
        print(f"[ERROR] Dataset YAML configuration not found at: {data_yaml}")
        print("Please configure DATA_YAML in your environment or provide a valid data.yaml path.")
        return None

    print("=" * 60)
    print("YOLO Instance Segmentation Training")
    print("=" * 60)
    print(f"Model   : {pretrained_model}")
    print(f"Device  : {'GPU ' + str(device) if device != 'cpu' else 'CPU'}")
    print(f"Data    : {data_yaml}")
    print(f"Epochs  : {epochs}")
    print("=" * 60)

    model = YOLO(pretrained_model)

    model.train(
        data         = str(data_yaml),
        epochs       = epochs,
        imgsz        = imgsz,
        device       = device,
        batch        = batch_size,
        project      = str(output_dir),
        name         = "exp_yolov8s_caries",
        save_period  = SAVE_PERIOD,
        exist_ok     = True,
        val          = True,
        pretrained   = True,
        plots        = True,
        amp          = True,

        # ── Optimizer & Scheduler Hyperparameters ───
        optimizer       = "AdamW",
        lr0             = 0.002,
        lrf             = 0.01,
        cos_lr          = True,
        warmup_epochs   = 3,
        warmup_momentum = 0.8,
        warmup_bias_lr  = 0.1,
        weight_decay    = 0.0005,

        # ── Regularization & Early Stopping ─────────
        dropout      = 0.0,
        patience     = 50,

        # ── Augmentation Settings ───────────────────
        fliplr       = 0.5,
        flipud       = 0.5,
        degrees      = 15,
        translate    = 0.1,
        scale        = 0.2,
        shear        = 10,
        mosaic       = 0.8,
        close_mosaic = 10,
        copy_paste   = 0.2,
        mixup        = 0.2,
        hsv_h        = 0.015,
        hsv_s        = 0.7,
        hsv_v        = 0.4,
        erasing      = 0.4,
    )

    print("\n📊 Evaluating best checkpoint (best.pt)...")
    metrics = model.val(
        data   = str(data_yaml),
        split  = "val",
        imgsz  = imgsz,
        device = device,
        batch  = batch_size,
    )

    print("\n[Done] Model training and evaluation finished.")
    print(f"[Done] Artifacts saved to: {output_dir / 'exp_yolov8s_caries'}")
    if hasattr(metrics, "box"):
        print(f"📈 Box mAP50: {metrics.box.map50:.3f}, mAP50-95: {metrics.box.map:.3f}")
    if hasattr(metrics, "seg"):
        print(f"📈 Mask mAP50: {metrics.seg.map50:.3f}, mAP50-95: {metrics.seg.map:.3f}")

    return metrics


def main():
    """Entry point for training script execution."""
    train_model()


if __name__ == "__main__":
    main()
