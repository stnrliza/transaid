"""
YOLO Inference Latency and Throughput Benchmarking on NVIDIA Jetson.

Benchmarks inference latency (mean, median, standard deviation, P95) and FPS
comparing ONNX Runtime (CPU/CUDA) against TensorRT FP16 execution on embedded
NVIDIA Jetson devices (e.g., Jetson Nano, Xavier, Orin).

Usage:
    python jetson_inference.py --mode onnx
    python jetson_inference.py --mode tensorrt
    python jetson_inference.py --mode both      (default)
"""

import argparse
import csv
from datetime import datetime
import os
from pathlib import Path
import platform
import random
import subprocess
import time
from typing import Optional

import numpy as np
from PIL import Image

# ── CONFIGURATION & ENVIRONMENT ─────────────────────
BASE_DIR      = Path(__file__).resolve().parent
MODEL_ONNX    = Path(os.getenv("JETSON_MODEL_ONNX", str(BASE_DIR / "models" / "yolov8s_best.onnx")))
MODEL_TRT     = Path(os.getenv("JETSON_MODEL_TRT", str(BASE_DIR / "models" / "yolov8s_best.engine")))
TEST_IMAGE    = Path(os.getenv("JETSON_TEST_IMAGE_DIR", str(BASE_DIR / "images")))
OUTPUT_DIR    = Path(os.getenv("JETSON_OUTPUT_DIR", str(BASE_DIR / "benchmark_results")))
IMG_SIZE      = int(os.getenv("IMG_SIZE", "640"))
WARMUP_RUNS   = int(os.getenv("WARMUP_RUNS", "10"))
BENCH_RUNS    = int(os.getenv("BENCH_RUNS", "100"))
CONF_THRESH   = float(os.getenv("CONF_THRESH", "0.10"))
IOU_THRESH    = float(os.getenv("IOU_THRESH", "0.30"))
# ────────────────────────────────────────────────────


def get_image_list(image_dir: str | Path) -> list[Path]:
    """
    Collect all valid image file paths from a directory.

    Args:
        image_dir: Filesystem path to directory containing benchmark images.

    Returns:
        List of Path objects for all matching image files.

    Raises:
        NotADirectoryError: If image_dir does not exist or is not a directory.
        FileNotFoundError: If no valid image files are found.
    """
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    img_dir = Path(image_dir)
    if not img_dir.is_dir():
        raise NotADirectoryError(f"Image directory does not exist: {image_dir}")
    paths = [p for p in img_dir.iterdir() if p.suffix.lower() in exts]
    if not paths:
        raise FileNotFoundError(f"No valid image files found in: {image_dir}")
    return paths


def get_jetson_info() -> dict:
    """
    Query host hardware and system specifications on Jetson Linux environments.

    Returns:
        Dictionary containing platform, architecture, device tree model, total RAM, and tegrastats snapshot.
    """
    info = {
        "platform": platform.platform(),
        "machine":  platform.machine(),
    }
    try:
        with open("/proc/device-tree/model", "r") as f:
            info["jetson_model"] = f.read().strip().replace("\x00", "")
    except Exception:
        info["jetson_model"] = "Unknown Jetson"
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if "MemTotal" in line:
                    kb = int(line.split()[1])
                    info["ram_gb"] = round(kb / 1e6, 2)
                    break
    except Exception:
        info["ram_gb"] = "N/A"
    try:
        result = subprocess.run(
            ["tegrastats", "--interval", "100"],
            capture_output=True, text=True, timeout=2
        )
        info["tegrastats_sample"] = result.stdout.split("\n")[0]
    except Exception:
        info["tegrastats_sample"] = "tegrastats unavailable"
    return info


def prepare_blob_onnx(path: Path) -> np.ndarray:
    """
    Load, resize, normalize, and format an image into an NCHW FP32 tensor for ONNX Runtime.

    Args:
        path: Path to input image file.

    Returns:
        NumPy array of shape (1, 3, IMG_SIZE, IMG_SIZE) with float32 values normalized to [0, 1].
    """
    img = Image.open(str(path)).resize((IMG_SIZE, IMG_SIZE)).convert("RGB")
    blob = np.array(img).astype(np.float32) / 255.0
    return np.transpose(blob, (2, 0, 1))[np.newaxis, ...]


def prepare_blob_trt(path: Path) -> np.ndarray:
    """
    Load, resize, normalize, and format an image into a contiguous NCHW FP16 array for TensorRT.

    Args:
        path: Path to input image file.

    Returns:
        Contiguous NumPy array of shape (1, 3, IMG_SIZE, IMG_SIZE) with float16 values.
    """
    img = Image.open(str(path)).resize((IMG_SIZE, IMG_SIZE)).convert("RGB")
    blob = np.array(img).astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]
    return np.ascontiguousarray(blob, dtype=np.float16)


def compute_stats(latencies: list[float], format_name: str, device: str) -> dict:
    """
    Calculate summary statistics from an array of measured execution latencies.

    Args:
        latencies: List of per-inference elapsed times in milliseconds.
        format_name: Model execution runtime format ('ONNX' or 'TensorRT-FP16').
        device: Device execution target ('CPU', 'CUDA', or 'GPU').

    Returns:
        Dictionary containing mean, median, std, min, max, FPS, and 95th percentile latency.
    """
    arr = np.array(latencies)
    return {
        "format":    format_name,
        "device":    device,
        "runs":      len(latencies),
        "mean_ms":   round(float(np.mean(arr)), 2),
        "median_ms": round(float(np.median(arr)), 2),
        "std_ms":    round(float(np.std(arr)), 2),
        "min_ms":    round(float(np.min(arr)), 2),
        "max_ms":    round(float(np.max(arr)), 2),
        "fps":       round(1000 / float(np.mean(arr)), 2),
        "p95_ms":    round(float(np.percentile(arr, 95)), 2),
    }


def benchmark_onnx(image_list: list[Path], model_path: Path = MODEL_ONNX,
                   warmup: int = WARMUP_RUNS, runs: int = BENCH_RUNS) -> Optional[dict]:
    """
    Benchmark ONNX Runtime inference throughput and latency.

    Args:
        image_list: List of sample image paths.
        model_path: Path to target ONNX model.
        warmup: Number of untimed warmup iterations.
        runs: Number of timed benchmark iterations.

    Returns:
        Dictionary of computed performance statistics, or None if onnxruntime is unavailable.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("[SKIP] onnxruntime is not installed.")
        return None

    if not model_path.exists():
        print(f"[ERROR] ONNX model file not found: {model_path}")
        return None

    print("\n" + "=" * 55)
    print(f"[ONNX] Loading {model_path} ...")

    providers = ort.get_available_providers()
    print(f"[ONNX] Available execution providers: {providers}")

    if "CUDAExecutionProvider" in providers:
        ep = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        device_label = "CUDA"
    else:
        ep = ["CPUExecutionProvider"]
        device_label = "CPU"
    print(f"[ONNX] Using primary provider: {ep[0]}")

    sess = ort.InferenceSession(str(model_path), providers=ep)
    input_name = sess.get_inputs()[0].name
    print(f"[ONNX] Input tensor name: {input_name}")
    print(f"[ONNX] Benchmark image pool: {len(image_list)} files")

    print(f"[ONNX] Warming up ({warmup} iterations) ...")
    for _ in range(warmup):
        blob = prepare_blob_onnx(random.choice(image_list))
        sess.run(None, {input_name: blob})

    print(f"[ONNX] Running benchmark ({runs} iterations) ...")
    latencies = []
    for i in range(runs):
        blob = prepare_blob_onnx(random.choice(image_list))
        t0 = time.perf_counter()
        sess.run(None, {input_name: blob})
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)
        if (i + 1) % 20 == 0:
            print(f"  [{i + 1}/{runs}] {latencies[-1]:.1f} ms")

    return compute_stats(latencies, "ONNX", device_label)


def benchmark_tensorrt(image_list: list[Path], model_path: Path = MODEL_TRT,
                       warmup: int = WARMUP_RUNS, runs: int = BENCH_RUNS) -> Optional[dict]:
    """
    Benchmark NVIDIA TensorRT FP16 engine latency and throughput.

    Args:
        image_list: List of sample image paths.
        model_path: Path to serialized TensorRT plan/engine file.
        warmup: Number of untimed warmup iterations.
        runs: Number of timed benchmark iterations.

    Returns:
        Dictionary of performance metrics, or None if TensorRT/PyCUDA are missing.
    """
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit
    except ImportError as e:
        print(f"[SKIP] TensorRT or PyCUDA unavailable: {e}")
        return None

    if not model_path.exists():
        print(f"[ERROR] TensorRT engine file not found: {model_path}")
        return None

    print("\n" + "=" * 55)
    print(f"[TensorRT] Loading {model_path} ...")

    trt_logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(trt_logger)

    with open(model_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()
    print("[TensorRT] Engine loaded successfully.")
    print(f"[TensorRT] Benchmark image pool: {len(image_list)} files")

    input_idx  = engine.get_binding_index(engine.get_binding_name(0))
    output_idx = engine.get_binding_index(engine.get_binding_name(1))

    input_shape  = tuple(engine.get_binding_shape(input_idx))
    output_shape = tuple(engine.get_binding_shape(output_idx))

    input_nbytes  = int(np.prod(input_shape)) * np.dtype(np.float16).itemsize
    output_nbytes = int(np.prod(output_shape)) * np.dtype(np.float16).itemsize

    d_input  = cuda.mem_alloc(input_nbytes)
    d_output = cuda.mem_alloc(output_nbytes)
    bindings = [int(d_input), int(d_output)]

    h_output = cuda.pagelocked_empty(int(np.prod(output_shape)), dtype=np.float16)
    stream   = cuda.Stream()

    def _infer(blob: np.ndarray) -> None:
        """Execute asynchronous TensorRT inference and synchronize CUDA stream."""
        cuda.memcpy_htod_async(d_input, blob, stream)
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        stream.synchronize()

    print(f"[TensorRT] Warming up ({warmup} iterations) ...")
    for _ in range(warmup):
        blob = prepare_blob_trt(random.choice(image_list))
        _infer(blob)

    print(f"[TensorRT] Running benchmark ({runs} iterations) ...")
    latencies = []
    for i in range(runs):
        blob = prepare_blob_trt(random.choice(image_list))
        t0 = time.perf_counter()
        _infer(blob)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)
        if (i + 1) % 20 == 0:
            print(f"  [{i + 1}/{runs}] {latencies[-1]:.1f} ms")

    return compute_stats(latencies, "TensorRT-FP16", "GPU")


def print_results(results: list[dict]) -> None:
    """
    Format and print benchmark summary metrics in an aligned terminal table.

    Args:
        results: List of performance metrics dictionaries.
    """
    print("\n" + "=" * 65)
    print("BENCHMARK RESULTS")
    print("=" * 65)
    header = "{:<16} {:<8} {:<10} {:<10} {:<8} {:<8} {:<10}".format(
        "Format", "Device", "Mean(ms)", "Median", "Std", "FPS", "P95(ms)")
    print(header)
    print("-" * len(header))
    for r in results:
        if r is None:
            continue
        print("{:<16} {:<8} {:<10} {:<10} {:<8} {:<8} {:<10}".format(
            r["format"], r["device"], r["mean_ms"],
            r["median_ms"], r["std_ms"], r["fps"], r["p95_ms"]))


def save_csv(results: list[dict], sysinfo: dict, output_dir: Path = OUTPUT_DIR) -> str:
    """
    Export benchmark metrics and device metadata into a timestamped CSV spreadsheet.

    Args:
        results: List of performance metrics dictionaries.
        sysinfo: System hardware information dictionary.
        output_dir: Destination folder for output CSV.

    Returns:
        Filesystem path to the written CSV file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = str(output_dir / f"benchmark_jetson_{timestamp}.csv")
    fieldnames = ["format", "device", "runs", "mean_ms", "median_ms",
                  "std_ms", "min_ms", "max_ms", "fps", "p95_ms",
                  "jetson_model", "ram_gb"]
    with open(fname, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            if r is None:
                continue
            row = {**r,
                   "jetson_model": sysinfo.get("jetson_model", ""),
                   "ram_gb":       sysinfo.get("ram_gb", "")}
            for fn in fieldnames:
                if fn not in row:
                    row[fn] = ""
            writer.writerow({k: row[k] for k in fieldnames})
    print(f"\n[CSV] Benchmark results saved to: {fname}")
    return fname


def main():
    """CLI entry point for running Jetson inference benchmarks."""
    parser = argparse.ArgumentParser(description="YOLO Inference Benchmark on NVIDIA Jetson")
    parser.add_argument(
        "--mode",
        choices=["onnx", "tensorrt", "both"],
        default="both",
        help="Inference runtime format to benchmark (default: both)"
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default=str(TEST_IMAGE),
        help=f"Directory containing test images (default: {TEST_IMAGE})"
    )
    args = parser.parse_args()

    print("=" * 65)
    print("YOLO INFERENCE BENCHMARK")
    print(f"Timestamp : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode      : {args.mode}")
    print("=" * 65)

    sysinfo = get_jetson_info()
    print("\n[System Info]")
    for k, v in sysinfo.items():
        print(f"  {k}: {v}")

    try:
        image_list = get_image_list(args.image_dir)
        print(f"\n[Images] Found {len(image_list)} images in: {args.image_dir}")
    except (NotADirectoryError, FileNotFoundError) as e:
        print(f"\n[ERROR] {e}")
        exit(1)

    results = []

    if args.mode in ("onnx", "both"):
        r = benchmark_onnx(image_list)
        results.append(r)

    if args.mode in ("tensorrt", "both"):
        r = benchmark_tensorrt(image_list)
        results.append(r)

    if any(results):
        print_results(results)
        save_csv(results, sysinfo)
    else:
        print("\n[ERROR] No benchmark results produced.")

    print("\n[Done] Benchmark execution finished.")


if __name__ == "__main__":
    main()