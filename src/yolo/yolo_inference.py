"""
YOLO Instance Segmentation Inference Module for NILT Dental Imagery.

This module handles Computer Vision inference exclusively:
- Loading and caching the trained YOLOv8/11 instance segmentation model.
- Running inference on Near-Infrared Light Transillumination (NILT) images.
- Deriving clinical severity classifications from detection confidence scores.
- Producing segmentation mask overlays without RAG, LLM, PDF, or interactive I/O.
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from PIL import Image

# Lazy import for YOLO inside get_yolo_model

load_dotenv()

# ── CONFIGURATION & ENVIRONMENT ─────────────────────
BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent.parent
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", str(BASE_DIR / "weights" / "best.pt"))
# ────────────────────────────────────────────────────

# Module-level model cache for lazy instantiation
_yolo_model: Optional[object] = None


def get_yolo_model(model_path: Optional[str] = None):
    """
    Retrieve or lazily initialize the YOLO instance segmentation model.

    Args:
        model_path: Path to the trained YOLO weights file (.pt). If omitted,
                    uses YOLO_MODEL_PATH from environment.

    Returns:
        Loaded Ultralytics YOLO model instance.
    """
    global _yolo_model
    if _yolo_model is None:
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "The 'ultralytics' package is required for YOLO inference. "
                "Please install it via 'pip install ultralytics'."
            ) from e

        target_path_str = model_path or YOLO_MODEL_PATH
        target_path = Path(target_path_str)
        if not target_path.is_absolute():
            # Check relative to cwd, then relative to repo root
            if not target_path.exists() and (REPO_ROOT / target_path).exists():
                target_path = REPO_ROOT / target_path

        if not target_path.exists():
            raise FileNotFoundError(
                f"YOLO model weights not found at: {target_path}. "
                "Please configure YOLO_MODEL_PATH in your environment or .env file."
            )
        _yolo_model = YOLO(str(target_path))
    return _yolo_model


def severity_label(confidence: float) -> str:
    """
    Classify clinical severity based on YOLO detection confidence.

    Args:
        confidence: Detection confidence score between 0.0 and 1.0.

    Returns:
        Indonesian/English severity string: 'Tinggi (High)', 'Sedang (Moderate)', or 'Rendah (Low)'.
    """
    if confidence >= 0.75:
        return "Tinggi (High)"
    elif confidence >= 0.5:
        return "Sedang (Moderate)"
    return "Rendah (Low)"


def run_yolo(image_path: str, save_overlay_path: Optional[str] = None) -> dict:
    """
    Run YOLO instance segmentation inference on an input NILT dental image.

    Args:
        image_path: Filesystem path to the input NILT dental image.
        save_overlay_path: Optional destination path to save the segmentation overlay.

    Returns:
        Dictionary containing:
            - class_name: Detected class label (str) or None if healthy.
            - confidence: Detection confidence score (float) or None.
            - severity: Severity label string or None.
            - result: The Ultralytics Results object.
            - orig_path: Resolved path to the original input image.
            - vis_path: Destination path where overlay was saved, or None.
    """
    resolved_img_path = Path(image_path).resolve()
    if not resolved_img_path.exists():
        raise FileNotFoundError(f"Input image not found at: {image_path}")

    model = get_yolo_model()
    results = model(str(resolved_img_path))
    result = results[0]

    vis_path = None
    if save_overlay_path:
        out_file = Path(save_overlay_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        result.save(filename=str(out_file))
        vis_path = str(out_file)

    orig_path = str(resolved_img_path)

    # Extract primary detection (one primary lesion evaluated per tooth crop)
    if len(result.boxes) == 0:
        return {
            "class_name": None,
            "confidence": None,
            "severity": None,
            "result": result,
            "orig_path": orig_path,
            "vis_path": vis_path,
        }

    box = result.boxes[0]
    class_id = int(box.cls[0])
    confidence = float(box.conf[0])
    class_name = model.names[class_id]

    return {
        "class_name": class_name,
        "confidence": confidence,
        "severity": severity_label(confidence),
        "result": result,
        "orig_path": orig_path,
        "vis_path": vis_path,
    }


def save_overlay(result, save_path: str | Path) -> str:
    """
    Save the Ultralytics segmentation overlay visualization to disk.

    Args:
        result: Ultralytics Results object from run_yolo.
        save_path: Destination path for the overlay image.

    Returns:
        String path of the saved overlay.
    """
    out_file = Path(save_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    result.save(filename=str(out_file))
    return str(out_file)
