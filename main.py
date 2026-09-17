#!/usr/bin/env python3
"""
TransAID: Clinical Decision Support System (CDSS) for Secondary Caries Detection.

Main Entry Point for the TransAID pipeline:
  1. Full pipeline: YOLO inference -> report generation -> G-Eval -> RAGAS
  2. Report-only mode (--skip-eval): YOLO inference -> report generation
  3. Eval-only mode (--geval and/or --ragas): Evaluate existing reports with optional UID prefix filter

Usage Examples:
  python main.py path/to/nilt_image.jpg
  python main.py path/to/nilt_image.jpg --name Liza --tooth 36 --lokasi oklusal
  python main.py path/to/nilt_image.jpg --skip-eval
  python main.py --geval --patient-id liza
  python main.py --ragas --patient-id liza_36
  python main.py --geval --ragas
"""

import argparse
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# ── 1. ENVIRONMENT INITIALIZATION ────────────────────
# load_dotenv MUST be called before importing any modules from src/
from dotenv import load_dotenv

load_dotenv()

# Ensure repository root is on sys.path so 'src' can be imported cleanly
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ── 2. MODULE IMPORTS ────────────────────────────────
from src.yolo.yolo_inference import run_yolo
from src.cdss.report import (
    build_query,
    build_query_healthy,
    translate_query_to_english,
    retrieve_context,
    generate_report,
    generate_healthy_report_static,
    generate_pdf,
    save_report_json,
)
from src.cdss.geval import run_geval
from src.cdss.ragas import run_ragas


# ── 3. HELPER FUNCTIONS ──────────────────────────────
def slugify_name(name: str) -> str:
    """
    Produce a filesystem-safe lowercase slug from patient name.
    Anything non-alphanumeric is collapsed to a single underscore.
    """
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return slug if slug else "patient"


def generate_uid(name: str, tooth: str, timestamp: str) -> str:
    """
    Generate unique session identifier string: <name>_<tooth>_<timestamp>
    """
    slug = slugify_name(name)
    tooth_clean = str(tooth).strip()
    return f"{slug}_{tooth_clean}_{timestamp}"


def resolve_output_dir() -> Path:
    """
    Resolve the root output directory from OUTPUT_DIR environment variable.
    """
    dir_env = os.getenv("OUTPUT_DIR", "src/outputs")
    p = Path(dir_env)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p


def prompt_manual_input() -> tuple[str, str, str]:
    """
    Interactively collect clinician input via terminal.

    Returns:
        Tuple of (patient_name, tooth_number, lesion_surface).
    """
    patient_name = input("Patient name: ").strip()
    while not patient_name:
        print("Patient name cannot be empty.")
        patient_name = input("Patient name: ").strip()

    tooth = input("Tooth number (FDI notation, 11-48): ").strip()
    while not tooth:
        print("Tooth number cannot be empty.")
        tooth = input("Tooth number (FDI notation, 11-48): ").strip()

    lokasi = input("Lesion location / surface (mesial/distal/oklusal/servikal/bukal/lingual): ").strip().lower()
    while not lokasi:
        print("Lesion location cannot be empty.")
        lokasi = input("Lesion location / surface (mesial/distal/oklusal/servikal/bukal/lingual): ").strip().lower()

    return patient_name, tooth, lokasi


# ── 4. CLI ARGUMENT PARSER & VALIDATION ──────────────
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="TransAID: Clinical Decision Support System for Secondary Caries Detection."
    )
    # Mode 1: Image path for report generation
    parser.add_argument(
        "image",
        nargs="?",
        default=None,
        help="Path to NILT dental image for report generation.",
    )

    # Optional metadata flags for non-interactive / scriptable execution
    parser.add_argument("--name", type=str, default=None, help="Patient name (slugified in UID).")
    parser.add_argument("--tooth", type=str, default=None, help="FDI tooth number (e.g. '36', '15').")
    parser.add_argument(
        "--lokasi",
        type=str,
        default=None,
        help="Lesion location / surface (oklusal, mesial, distal, bukal, lingual, servikal).",
    )

    # Report-only modifier
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Generate report only; skip running G-Eval and RAGAS evaluation suites.",
    )

    # Mode 2: Evaluation-only flags
    parser.add_argument(
        "--geval",
        action="store_true",
        help="Run G-Eval LLM-as-a-judge evaluation on existing report JSONs.",
    )
    parser.add_argument(
        "--ragas",
        action="store_true",
        help="Run RAGAS reference-free evaluation suite on existing report JSONs.",
    )
    parser.add_argument(
        "--patient-id",
        type=str,
        default=None,
        help="Optional UID prefix filter for evaluation mode (e.g. 'liza', 'liza_36').",
    )

    return parser


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    has_image = args.image is not None
    has_eval_mode = args.geval or args.ragas

    # Rule 1: Exactly one mode {image path, --geval/--ragas}
    if has_image and has_eval_mode:
        parser.error(
            "Ambiguous invocation: cannot pass both an image path and evaluation flags (--geval / --ragas). "
            "Choose report generation mode (<image>) or evaluation-only mode (--geval/--ragas)."
        )

    if not has_image and not has_eval_mode:
        parser.error(
            "No operation specified. Provide an image path to run report generation, "
            "or pass --geval / --ragas to run evaluation mode."
        )

    # Rule 2: --patient-id is only valid in eval-only mode
    if args.patient_id is not None and has_image:
        parser.error("--patient-id can only be used in evaluation-only mode (--geval / --ragas), not with an image path.")

    # Rule 3: --name, --tooth, --lokasi must be given all together or not at all
    manual_flags = [args.name is not None, args.tooth is not None, args.lokasi is not None]
    if any(manual_flags) and not all(manual_flags):
        parser.error("--name, --tooth, and --lokasi must all be provided together when using non-interactive mode.")

    if any(manual_flags) and not has_image:
        parser.error("--name, --tooth, and --lokasi are only valid during report generation with an image path.")

    # Rule 4: --skip-eval is only valid with an image
    if args.skip_eval and not has_image:
        parser.error("--skip-eval is only valid during report generation with an image path.")


# ── 5. PIPELINE EXECUTION MODES ──────────────────────
def run_report_pipeline(args: argparse.Namespace, output_root: Path) -> None:
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image file not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    # Collect metadata (non-interactive or interactive)
    if args.name is not None:
        patient_name = args.name.strip()
        tooth = args.tooth.strip()
        lokasi = args.lokasi.strip().lower()
    else:
        print("\n--- Practitioner Input ---")
        patient_name, tooth, lokasi = prompt_manual_input()

    # Generate session timestamp and UID once at session start
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    uid = generate_uid(patient_name, tooth, timestamp)

    print(f"\n{'='*55}")
    print(f"TransAID Session: {uid}")
    print(f"Input image:      {image_path.resolve()}")
    print(f"Patient name:     {patient_name}")
    print(f"Tooth (FDI):      {tooth}")
    print(f"Location:         {lokasi}")
    print(f"{'='*55}\n")

    # Step 1: YOLO instance segmentation inference
    print("[1/5] Running YOLO instance segmentation inference...")
    detection = run_yolo(str(image_path))
    is_healthy = (detection.get("class_name") is None)

    if is_healthy:
        outcome_dir = output_root / "healthy"
        print("      Result: No caries lesion detected (Healthy Case).")
    else:
        outcome_dir = output_root / "caries"
        conf_pct = detection["confidence"] * 100 if detection.get("confidence") is not None else 0.0
        print(f"      Result: {detection['class_name']} detected ({conf_pct:.1f}% confidence, {detection.get('severity')}).")

    outcome_dir.mkdir(parents=True, exist_ok=True)

    # Save segmentation overlay as <uid>.png
    png_path = outcome_dir / f"{uid}.png"
    detection["result"].save(filename=str(png_path))
    detection["vis_path"] = str(png_path)
    detection["orig_path"] = str(image_path.resolve())
    print(f"      Segmentation overlay saved: {png_path}")

    # Step 2: Query Construction
    manual_input = {"patient_name": patient_name, "no_gigi": tooth, "lokasi": lokasi}
    print("[2/5] Constructing clinical query...")
    if is_healthy:
        query = build_query_healthy(manual_input)
    else:
        query = build_query(detection, manual_input)
    print(f"      Clinical query: \"{query}\"")

    # Step 3: Cross-Lingual RAG Retrieval & Report Generation
    if is_healthy:
        print("[3/5] Generating clinical reports (deterministic healthy template)...")
        report = generate_healthy_report_static(tooth, lokasi)
        chunks = []
    else:
        print("[3/5] Translating query and retrieving RAG context from ChromaDB...")
        english_query = translate_query_to_english(query)
        print(f"      Optimized English query: \"{english_query}\"")
        chunks = retrieve_context(english_query)
        print(f"      Retrieved {len(chunks)} evidence chunks from dental literature.")
        print("[4/5] Generating dual-audience clinical reports via LLM...")
        report = generate_report(query, chunks)

    print("      Clinical reports generated successfully ✅")

    # Step 4: Typeset PDF Report
    print("[5/5] Typesetting clinical PDF report...")
    pdf_path = generate_pdf(
        detection=detection,
        report=report,
        query=query,
        patient_name=patient_name,
        tooth=tooth,
        lokasi=lokasi,
        is_healthy=is_healthy,
        output_dir=outcome_dir,
        uid=uid,
    )
    print(f"      PDF report saved: {pdf_path}")

    # Step 5: Persist Full Case Metadata JSON
    json_path = save_report_json(
        detection=detection,
        report=report,
        query=query,
        contexts=chunks,
        pdf_path=pdf_path,
        patient_name=patient_name,
        tooth=tooth,
        lokasi=lokasi,
        image_path=str(image_path),
        output_dir=outcome_dir,
        uid=uid,
    )
    print(f"      JSON metadata saved: {json_path}")

    # Step 6: Evaluations (unless --skip-eval)
    geval_sidecar = None
    ragas_sidecar = None
    if not args.skip_eval:
        print(f"\n--- Running Quality Evaluation Suites for UID: {uid} ---")
        try:
            print("[Eval: G-Eval LLM-as-a-Judge]")
            run_geval(output_dir=output_root, patient_id=uid)
            geval_sidecar = outcome_dir / f"{uid}_geval.json"
        except Exception as e:
            print(f"  [G-Eval Failed] {e}", file=sys.stderr)

        try:
            print("\n[Eval: RAGAS Suite]")
            run_ragas(output_dir=output_root, patient_id=uid)
            ragas_sidecar = outcome_dir / f"{uid}_ragas.json"
        except Exception as e:
            print(f"  [RAGAS Failed] {e}", file=sys.stderr)
    else:
        print("\nEvaluation skipped (--skip-eval specified).")

    # Final Execution Summary
    print(f"\n{'='*55}")
    print("TransAID Pipeline Run Summary")
    print(f"{'='*55}")
    print(f"UID:         {uid}")
    print(f"Outcome:     {'Healthy (No Caries)' if is_healthy else 'Caries Detected'}")
    print(f"Overlay:     {png_path}")
    print(f"PDF Report:  {pdf_path}")
    print(f"Case Data:   {json_path}")
    if geval_sidecar and geval_sidecar.exists():
        print(f"G-Eval JSON: {geval_sidecar}")
        print(f"G-Eval CSV:  {output_root / 'geval.csv'}")
    if ragas_sidecar and ragas_sidecar.exists():
        print(f"RAGAS JSON:  {ragas_sidecar}")
        print(f"RAGAS CSV:   {output_root / 'ragas.csv'}")
    print(f"{'='*55}\n")


def run_eval_pipeline(args: argparse.Namespace, output_root: Path) -> None:
    filter_desc = f" (filter prefix: '{args.patient_id}')" if args.patient_id else " (all pending reports)"
    print(f"\n{'='*55}")
    print(f"TransAID Evaluation-Only Mode{filter_desc}")
    print(f"Output Root: {output_root}")
    print(f"{'='*55}\n")

    if args.geval:
        print("--- Running G-Eval (LLM-as-a-Judge) Suite ---")
        run_geval(output_dir=output_root, patient_id=args.patient_id)
        print(f"G-Eval aggregate CSV: {output_root / 'geval.csv'}\n")

    if args.ragas:
        print("--- Running RAGAS Evaluation Suite ---")
        run_ragas(output_dir=output_root, patient_id=args.patient_id)
        print(f"RAGAS aggregate CSV: {output_root / 'ragas.csv'}\n")

    print(f"{'='*55}")
    print("Evaluation Complete ✅")
    print(f"{'='*55}\n")


# ── 6. MAIN ROUTINE ──────────────────────────────────
def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args, parser)

    output_root = resolve_output_dir()
    output_root.mkdir(parents=True, exist_ok=True)

    if args.image is not None:
        run_report_pipeline(args, output_root)
    else:
        run_eval_pipeline(args, output_root)


if __name__ == "__main__":
    main()
