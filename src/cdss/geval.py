"""
05_geval.py — G-Eval (reference-free LLM-as-judge) for NILT CDSS reports.

Evaluates the two report variants produced by 03_pipeline.py:
  - laporan_dokter (dentist-facing): coherence, completeness, relevance, fluency
  - laporan_pasien (patient-facing): simplicity, clarity, fluency, conciseness

Method: G-Eval form-filling paradigm (Liu et al., 2023) — each criterion has a
fixed chain-of-thought rubric baked into the prompt; the judge LLM returns a
1-5 score + short reasoning per criterion as JSON.

NOTE on fidelity to the original G-Eval paper: the paper weights the final
score by the token-level probability distribution over scores 1-5 (using
logprobs from the judge model) to get a continuous score, which smooths out
single-sample judge noise. This script does NOT do that — we take the judge's
single direct score per criterion instead. This is a documented simplification;
if you want to approximate the paper's behavior, increase N_SAMPLES below to
self-consistency-average multiple independent judge calls per criterion instead
of relying on logprobs.

Judge LLM: Qwen3:14b via local Ollama instance.

Run:
    python 05_geval.py
"""

import csv
import json
import os
import random
import re
import sys
import time
import glob
from dataclasses import dataclass
from typing import Optional

import requests


# --------------------------------------------------------------------------- #
# Config — edit here, no CLI needed
# --------------------------------------------------------------------------- #

OLLAMA_BASE_URL     = "http://localhost:11434"   # Ollama server address
JUDGE_MODEL         = "qwen3:14b"
INPUT_DIR           = "/home/guest/Workshop/skripsi-lija/cdss/outputs/03_reports/fix"
OUTPUT_DIR          = "/home/guest/Workshop/skripsi-lija/cdss/outputs/05_geval/fix"
REPORT_PATTERN      = "laporan_*.json"
SLEEP_BETWEEN_CALLS = 1.0   # seconds between each judge call.
                            # Running locally via Ollama — no rate limits,
                            # but a small pause avoids overwhelming the GPU.
SKIP_EXISTING       = True  # set False to re-score reports that already have a sidecar


# --------------------------------------------------------------------------- #
# Criteria definitions (G-Eval form-filling style: name + CoT evaluation steps)
# --------------------------------------------------------------------------- #

@dataclass
class Criterion:
    name: str
    description: str
    evaluation_steps: list[str]


DOKTER_CRITERIA: list[Criterion] = [
    Criterion(
        name="coherence",
        description=(
            "Coherence measures whether the report's findings, clinical "
            "reasoning, and recommendation form a logical, connected flow "
            "without contradictions or non-sequiturs."
        ),
        evaluation_steps=[
            "Read the Temuan (finding), Interpretasi (interpretation), and "
            "Rekomendasi (recommendation) sections in order.",
            "Check whether the interpretation logically follows from the "
            "stated finding (same tooth, location, condition, confidence).",
            "Check whether the recommendation logically follows from the "
            "interpretation (no recommendation that contradicts or ignores "
            "the clinical reasoning given).",
            "Penalize abrupt topic shifts, contradictions between sections, "
            "or reasoning that does not connect to the specific case.",
            "Assign a score from 1 (incoherent, disconnected sections) to "
            "5 (fully logical, tightly connected flow).",
        ],
    ),
    Criterion(
        name="completeness",
        description=(
            "Completeness measures whether all clinically expected elements "
            "are present given the detection input (diagnosis/class, tooth "
            "number, location/surface, severity, and an actionable "
            "treatment/follow-up recommendation)."
        ),
        evaluation_steps=[
            "Identify what clinical elements are available in the input "
            "(detection class, tooth number, location, confidence, severity).",
            "Check whether the report's Temuan correctly reflects the "
            "detected tooth, location, and condition.",
            "Check whether the Interpretasi addresses clinical significance "
            "(not just restating the finding).",
            "Check whether the Rekomendasi gives a concrete next step "
            "(further exam, treatment option, monitoring) rather than being "
            "vague or generic.",
            "Assign a score from 1 (missing multiple expected elements) to "
            "5 (all expected elements present and substantively addressed).",
        ],
    ),
    Criterion(
        name="relevance",
        description=(
            "Relevance measures whether every statement in the report is "
            "pertinent to this specific patient case, as opposed to generic "
            "boilerplate or filler not tied to the actual detection."
        ),
        evaluation_steps=[
            "For each sentence in the report, check whether it references "
            "or is clearly tied to the specific detection (tooth, location, "
            "condition, confidence) rather than being a generic statement "
            "that could apply to any case.",
            "Flag sentences that appear to be filler, overly generic "
            "textbook statements, or unrelated tangents.",
            "Assign a score from 1 (mostly generic/irrelevant content) to "
            "5 (every statement is specific and pertinent to this case).",
        ],
    ),
    Criterion(
        name="fluency",
        description=(
            "Fluency measures grammatical correctness, professional "
            "clinical register, and natural sentence construction in "
            "Indonesian medical writing."
        ),
        evaluation_steps=[
            "Check grammar, spelling, and sentence construction.",
            "Check whether the register is appropriate for a "
            "dentist-to-dentist clinical report (professional, precise "
            "terminology) rather than overly casual or awkwardly phrased.",
            "Assign a score from 1 (frequent grammatical errors, awkward "
            "phrasing) to 5 (fluent, natural, professional clinical "
            "Indonesian).",
        ],
    ),
]

PASIEN_CRITERIA: list[Criterion] = [
    Criterion(
        name="simplicity",
        description=(
            "Simplicity measures whether the report avoids unexplained "
            "clinical jargon, using plain language a non-dentist patient "
            "can understand."
        ),
        evaluation_steps=[
            "Identify any clinical/technical terms used in the report "
            "(e.g. disease names, anatomical terms, procedure names).",
            "Check whether each such term is either avoided, replaced with "
            "plain language, or briefly explained in lay terms when used.",
            "Assign a score from 1 (dense with unexplained jargon) to 5 "
            "(fully accessible plain language).",
        ],
    ),
    Criterion(
        name="clarity",
        description=(
            "Clarity measures whether a patient can understand, in a "
            "single read, what the problem is and what they should do next."
        ),
        evaluation_steps=[
            "Read the Ringkasan (summary) and Saran (advice) as a patient "
            "with no dental background would.",
            "Check whether it is immediately clear what was found and why "
            "it matters to the patient.",
            "Check whether the advice gives an unambiguous, concrete next "
            "action (e.g. what to do, when to follow up).",
            "Assign a score from 1 (confusing, unclear what to do) to 5 "
            "(immediately clear problem and next action).",
        ],
    ),
    Criterion(
        name="fluency",
        description=(
            "Fluency measures grammatical correctness and a natural, "
            "friendly tone appropriate for communicating with a patient."
        ),
        evaluation_steps=[
            "Check grammar, spelling, and sentence construction.",
            "Check whether the tone is warm and approachable rather than "
            "clinical/cold or awkwardly translated-sounding.",
            "Assign a score from 1 (poor grammar, cold/awkward tone) to 5 "
            "(fluent, natural, friendly Indonesian).",
        ],
    ),
    Criterion(
        name="conciseness",
        description=(
            "Conciseness measures whether the report conveys the necessary "
            "information without redundant restatement or padding."
        ),
        evaluation_steps=[
            "Check for sentences or phrases that repeat information "
            "already stated elsewhere in the report.",
            "Check for padding, hedging, or filler that adds length "
            "without adding information the patient needs.",
            "Assign a score from 1 (repetitive/padded) to 5 (every "
            "sentence adds distinct, necessary information).",
        ],
    ),
]


# --------------------------------------------------------------------------- #
# Judge LLM call (Ollama /api/generate endpoint)
# --------------------------------------------------------------------------- #

PROMPT_TEMPLATE = """You are an expert dental clinician evaluating an AI-generated clinical report for quality. You will evaluate ONE criterion at a time using the chain-of-thought steps provided.

# Criterion: {criterion_name}
{criterion_description}

# Evaluation steps (follow these in order before scoring)
{steps}

# Case context
Detection input: {query}
Manual input: tooth={tooth}, location={lokasi}
Severity: {severity}

# Report under evaluation ({doc_type_label})
{report_text}

# Instructions
Follow the evaluation steps above internally, then output ONLY a JSON object with EXACTLY these two keys and no others, no markdown fences, no extra text:
{{"score": <integer 1-5>, "reasoning": "<justifikasi 1-3 kalimat dalam Bahasa Indonesia, mengutip bagian spesifik dari laporan>"}}
"""


def build_prompt(criterion: Criterion, doc_type_label: str, report_text: str,
                  query: str, tooth: str, lokasi: str, severity: str) -> str:
    steps = "\n".join(f"{i+1}. {s}" for i, s in enumerate(criterion.evaluation_steps))
    return PROMPT_TEMPLATE.format(
        criterion_name=criterion.name,
        criterion_description=criterion.description,
        steps=steps,
        query=query,
        tooth=tooth,
        lokasi=lokasi,
        severity=severity,
        doc_type_label=doc_type_label,
        report_text=report_text,
    )


def call_ollama(prompt: str, model: str, base_url: str,
                max_retries: int = 5, timeout: int = 300) -> dict:
    """Call Ollama via its /api/generate endpoint. Returns parsed dict.

    NOTE: We do NOT use Ollama's format='json' parameter because it is
    incompatible with Qwen3 on Ollama 0.24.0 (produces empty '{}').
    Instead, we rely on prompt instructions and parse JSON from the raw
    text response after stripping <think> tags and markdown fences.

    num_predict is set to 2048 because Qwen3 is a thinking model that
    wraps internal reasoning in <think>...</think> tags.  With complex
    evaluation prompts, the thinking alone can exceed 512 tokens, leaving
    nothing for the actual JSON output.  2048 gives ample room.
    """
    url = f"{base_url}/api/generate"

    # Prepend system instruction to the prompt.
    full_prompt = (
        "You are an expert evaluator. Always respond with valid JSON only. "
        "The JSON must have exactly two keys: \"score\" (integer 1-5) and "
        "\"reasoning\" (string in Indonesian). "
        "No markdown fences, no extra text, no explanation outside the JSON.\n\n"
        f"{prompt}"
    )

    payload = {
        "model": model,
        "prompt": full_prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 2048,
        },
    }

    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        if attempt > 1:
            payload["options"]["temperature"] = min(0.2 * (attempt - 1), 1.0)

        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            if resp.status_code >= 500:
                wait = min(10 * (2 ** (attempt - 1)) + random.uniform(0, 5), 120)
                last_err = RuntimeError(f"HTTP {resp.status_code} server error (attempt {attempt})")
                print(f"  [server-error {resp.status_code}] retry {attempt}/{max_retries} in {wait:.0f}s...",
                      file=sys.stderr)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            raw_text = data["response"]
            # Strip any <think>...</think> tags that Qwen3 may emit
            text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
            # Strip markdown code fences if present
            text = re.sub(r"^\s*$", "", text).strip()
            # Extract the JSON object from the text
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if match:
                return json.loads(match.group())
            # Debug: log what we got so we can diagnose
            print(f"  [debug] raw response length={len(raw_text)}, "
                  f"after strip length={len(text)}, "
                  f"text={text[:200]!r}",
                  file=sys.stderr)
            raise json.JSONDecodeError("No JSON object found in response", text, 0)
        except requests.exceptions.Timeout as e:
            last_err = e
            wait = min(30 * (2 ** (attempt - 1)), 300)
            print(f"  [timeout] retry {attempt}/{max_retries} in {wait}s...",
                  file=sys.stderr)
            time.sleep(wait)
        except (requests.RequestException, KeyError, IndexError, json.JSONDecodeError) as e:
            last_err = e
            wait = 2 ** attempt
            print(f"  [error: {e}] retry {attempt}/{max_retries} in {wait}s...",
                  file=sys.stderr)
            time.sleep(wait)

    raise RuntimeError(f"Ollama call failed after {max_retries} retries: {last_err}")


def extract_score(parsed: dict, criterion_name: str) -> tuple[Optional[int], str]:
    """Extract score and reasoning from parsed JSON, with fuzzy key matching.

    Qwen3 sometimes uses variant key names (e.g. 'skor' instead of 'score',
    'alasan' instead of 'reasoning'). This function tries common alternatives.
    """
    # Try common key variants for score
    score = None
    for key in ("score", "skor", "nilai", "Score", "SCORE"):
        if key in parsed:
            score = parsed[key]
            break
    # If still None, try to find any integer value in the dict
    if score is None:
        for v in parsed.values():
            if isinstance(v, int) and 1 <= v <= 5:
                score = v
                break

    # Try common key variants for reasoning
    reasoning = ""
    for key in ("reasoning", "reason", "alasan", "justifikasi",
                "penjelasan", "Reasoning", "REASONING"):
        if key in parsed:
            reasoning = str(parsed[key])
            break
    # If still empty, try to find the longest string value as reasoning
    if not reasoning:
        for v in parsed.values():
            if isinstance(v, str) and len(v) > len(reasoning):
                reasoning = v

    if isinstance(score, str):
        m = re.search(r"\d+", score)
        score = int(m.group()) if m else None
    if score is not None:
        try:
            score = int(score)
        except (TypeError, ValueError):
            score = None
    if score is not None and not (1 <= score <= 5):
        print(f"  [warn] out-of-range score {score} for {criterion_name}, clamping",
              file=sys.stderr)
        score = max(1, min(5, score))

    if score is None:
        print(f"  [warn] could not extract score for {criterion_name} from: {parsed}",
              file=sys.stderr)

    return score, reasoning


# --------------------------------------------------------------------------- #
# Report loading / text assembly
# --------------------------------------------------------------------------- #

def assemble_dokter_text(report: dict) -> str:
    d = report["report"]["laporan_dokter"]
    return (
        f"Temuan: {d.get('temuan', '')}\n"
        f"Interpretasi: {d.get('interpretasi', '')}\n"
        f"Rekomendasi: {d.get('rekomendasi', '')}"
    )


def assemble_pasien_text(report: dict) -> str:
    p = report["report"]["laporan_pasien"]
    return (
        f"Ringkasan: {p.get('ringkasan', '')}\n"
        f"Saran: {p.get('saran', '')}"
    )


def load_report(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# Main evaluation loop
# --------------------------------------------------------------------------- #

def evaluate_report(report: dict, report_id: str, model: str, base_url: str,
                     sleep_between_calls: float) -> dict:
    query = report.get("query", "")
    manual = report.get("manual_input", {})
    tooth = manual.get("no_gigi", "")
    lokasi = manual.get("lokasi", "")
    severity = report.get("detection", {}).get("severity", "")

    result = {"report_id": report_id, "dokter": {}, "pasien": {}}

    dokter_text = assemble_dokter_text(report)
    for crit in DOKTER_CRITERIA:
        prompt = build_prompt(crit, "Dentist-facing report (laporan_dokter)",
                               dokter_text, query, tooth, lokasi, severity)
        parsed = call_ollama(prompt, model, base_url)
        score, reasoning = extract_score(parsed, crit.name)
        result["dokter"][crit.name] = {"score": score, "reasoning": reasoning}
        print(f"  dokter/{crit.name}: {score}")
        time.sleep(sleep_between_calls)

    pasien_text = assemble_pasien_text(report)
    for crit in PASIEN_CRITERIA:
        prompt = build_prompt(crit, "Patient-facing report (laporan_pasien)",
                               pasien_text, query, tooth, lokasi, severity)
        parsed = call_ollama(prompt, model, base_url)
        score, reasoning = extract_score(parsed, crit.name)
        result["pasien"][crit.name] = {"score": score, "reasoning": reasoning}
        print(f"  pasien/{crit.name}: {score}")
        time.sleep(sleep_between_calls)

    return result


def write_sidecar(result: dict, output_dir: str, report_id: str) -> str:
    out_path = os.path.join(output_dir, f"{report_id}_geval.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return out_path


def append_to_csv_rows(result: dict, rows: list[dict]) -> None:
    for doc_type in ("dokter", "pasien"):
        for criterion, payload in result[doc_type].items():
            rows.append({
                "report_id": result["report_id"],
                "doc_type": doc_type,
                "criterion": criterion,
                "score": payload["score"],
                "reasoning": payload["reasoning"],
            })


def write_aggregate_csv(rows: list[dict], output_dir: str) -> str:
    out_path = os.path.join(output_dir, "geval_scores.csv")
    fieldnames = ["report_id", "doc_type", "criterion", "score", "reasoning"]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def main():
    # Quick connectivity check — make sure Ollama is reachable
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        r.raise_for_status()
        models = [m["name"] for m in r.json().get("models", [])]
        if not any(JUDGE_MODEL in m for m in models):
            print(f"WARNING: model '{JUDGE_MODEL}' not found in Ollama. "
                  f"Available: {models}. Will attempt to pull on first call.",
                  file=sys.stderr)
        else:
            print(f"Using Ollama model: {JUDGE_MODEL}")
    except requests.RequestException as e:
        print(f"ERROR: cannot reach Ollama at {OLLAMA_BASE_URL}: {e}",
              file=sys.stderr)
        print("Make sure 'ollama serve' is running.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    paths = sorted(glob.glob(os.path.join(INPUT_DIR, REPORT_PATTERN)))
    if not paths:
        print(f"No files matched '{REPORT_PATTERN}' in {INPUT_DIR}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(paths)} report(s) in {INPUT_DIR}")
    all_rows: list[dict] = []

    for path in paths:
        report_id = os.path.splitext(os.path.basename(path))[0]
        sidecar_path = os.path.join(OUTPUT_DIR, f"{report_id}_geval.json")

        if SKIP_EXISTING and os.path.exists(sidecar_path):
            print(f"[skip] {report_id} (sidecar exists)")
            with open(sidecar_path, "r", encoding="utf-8") as f:
                result = json.load(f)
            append_to_csv_rows(result, all_rows)
            continue

        print(f"[eval] {report_id}")
        try:
            report = load_report(path)
            result = evaluate_report(report, report_id, JUDGE_MODEL,
                                     OLLAMA_BASE_URL, SLEEP_BETWEEN_CALLS)
            write_sidecar(result, OUTPUT_DIR, report_id)
            append_to_csv_rows(result, all_rows)
        except Exception as e:
            print(f"  [FAILED] {report_id}: {e}", file=sys.stderr)
            continue

    csv_path = write_aggregate_csv(all_rows, OUTPUT_DIR)
    print(f"\nDone. Aggregate CSV: {csv_path}")
    print(f"Sidecar JSONs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
