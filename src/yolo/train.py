"""
train_exp1_olddataset.py
------------------------
EKSPERIMEN 1: Isolasi variabel dataset.

Tujuan:
  Menguji apakah YOLO11l-seg bisa mencapai precision ~0.90 jika menggunakan
  dataset lama (CariesDatasetCleanSG) yang sama dengan run berhasil (train_best_l2).

  Hyperparameter disamakan PERSIS dengan train_best_l2 (yolov8l-seg, precision 0.90).

Hipotesis:
  - Jika berhasil (precision >= 0.80) → masalah ada di dataset baru, bukan arsitektur YOLO11
  - Jika gagal (precision tetap rendah) → masalah ada di arsitektur YOLO11 → coba yolov8l-seg
"""

import torch
from pathlib import Path
from ultralytics import YOLO


# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
# Dataset lama yang terbukti berhasil
DATA_YAML  = Path("/home/guest/Workshop/skripsi-lija/skripsi/BAB_4/1.yolov8seg/dataset/CariesDatasetCleanSG/data.yaml")
OUTPUT_DIR = BASE_DIR / "runs"


# ─────────────────────────────────────────────
# CONFIG — sama persis dengan train_best_l2
# ─────────────────────────────────────────────
PRETRAINED_MODEL = "yolov8l-seg.pt"
EPOCHS           = 500
IMGSZ            = 640
BATCH_SIZE       = -1       # auto batch, sama dengan train_best_l2
SAVE_PERIOD      = 50
DEVICE           = 0 if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    if not DATA_YAML.exists():
        print(f"[ERROR] Data YAML tidak ditemukan: {DATA_YAML}")
        return

    print("=" * 60)
    print("EKSPERIMEN 1: YOLO11l + Dataset Lama (CariesDatasetCleanSG)")
    print("=" * 60)
    print(f"Model   : {PRETRAINED_MODEL}")
    print(f"Device  : {'GPU' if DEVICE == 0 else 'CPU'}")
    print(f"Data    : {DATA_YAML}")
    print(f"Epochs  : {EPOCHS}")
    print("=" * 60)

    model = YOLO(PRETRAINED_MODEL)

    model.train(
        data         = str(DATA_YAML),
        epochs       = EPOCHS,
        imgsz        = IMGSZ,
        device       = DEVICE,
        batch        = BATCH_SIZE,
        project      = str(OUTPUT_DIR),
        name         = "exp1_yolov8l_olddataset",
        save_period  = SAVE_PERIOD,
        exist_ok     = True,
        val          = True,
        pretrained   = True,
        plots        = True,
        amp          = True,

        # ── Optimizer & Scheduler — sama dengan train_best_l2 ────────
        optimizer       = "AdamW",
        lr0             = 0.002,
        lrf             = 0.01,
        cos_lr          = True,
        warmup_epochs   = 3,
        warmup_momentum = 0.8,
        warmup_bias_lr  = 0.1,
        weight_decay    = 0.0005,

        # ── Regularisasi — sama dengan train_best_l2 ─────────────────
        dropout      = 0.0,
        patience     = 50,

        # ── Augmentasi — sama dengan train_best_l2 ───────────────────
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

    print("\n📊 Evaluasi model terbaik (best.pt)...")
    metrics = model.val(
        data   = str(DATA_YAML),
        split  = "val",
        imgsz  = IMGSZ,
        device = DEVICE,
        batch  = BATCH_SIZE,
    )

    print("\n[done] Eksperimen 1 selesai.")
    print(f"[done] Hasil di: {OUTPUT_DIR / 'exp1_yolo11l_olddataset'}")
    print(f"📈 mAP50: {metrics.box.map50:.3f}, mAP50-95: {metrics.box.map:.3f}")


if __name__ == "__main__":
    main()

