"""
RAGAS-Inspired (Reference-Free) Evaluation for NILT CDSS Reports.

Evaluates dentist-facing reports (laporan_dokter) produced by the CDSS pipeline
using four reference-free metrics adapted from the RAGAS framework:

  - Faithfulness      : Are factual claims in the report grounded either in the
                        retrieved literature contexts or in established dental clinical knowledge?
  - Answer Relevancy  : Does the report directly and comprehensively address the detection query?
  - Context Precision : Are the retrieved literature chunks useful and relevant to this specific case?
                        (Automatically skipped for healthy cases where contexts are empty by design).
  - Context Recall    : How completely did retrieval locate the necessary evidence to support the case?
                        (Automatically skipped for healthy cases).

I/O & Skip Architecture:
- Discovers reports under outputs/healthy/ and outputs/caries/ (ignoring sidecars).
- Reads outputs/ragas.csv once to build an in-memory set of already evaluated UIDs.
- Writes <uid>_ragas.json sidecar first, then appends to outputs/ragas.csv (crash safety).
"""

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()

# ── CONFIGURATION & ENVIRONMENT ─────────────────────
BASE_DIR            = Path(__file__).resolve().parent
REPO_ROOT           = BASE_DIR.parent.parent
OLLAMA_BASE_URL     = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
JUDGE_MODEL         = os.getenv("JUDGE_MODEL", "qwen3:14b")

_output_dir_env     = os.getenv("OUTPUT_DIR", os.getenv("RAGAS_OUTPUT_DIR", "src/outputs"))
OUTPUT_DIR          = Path(_output_dir_env)
if not OUTPUT_DIR.is_absolute():
    OUTPUT_DIR      = REPO_ROOT / OUTPUT_DIR

SLEEP_BETWEEN_CALLS = float(os.getenv("SLEEP_BETWEEN_CALLS", "1.0"))
SKIP_EXISTING       = os.getenv("SKIP_EXISTING", "True").lower() in ("true", "1", "yes")
MAX_CONTEXT_CHUNKS  = int(os.getenv("MAX_CONTEXT_CHUNKS", "5"))

CSV_FIELDNAMES      = ["uid", "metric", "score", "reasoning"]
# ────────────────────────────────────────────────────


@dataclass
class Criterion:
    """Evaluation criterion structure with detailed evaluation instructions."""
    name: str
    description: str
    evaluation_steps: list[str]


RAGAS_CRITERIA: list[Criterion] = [
    Criterion(
        name="faithfulness",
        description=(
            "Faithfulness mengukur apakah klaim faktual dalam laporan_dokter "
            "(Temuan, Interpretasi, Rekomendasi) dapat dibenarkan secara klinis. "
            "Klaim dianggap valid jika (a) dapat ditelusuri ke salah satu chunk "
            "konteks yang diambil, ATAU (b) konsisten dengan pengetahuan klinis "
            "dental yang mapan yang wajar diketahui oleh LLM medis. "
            "Klaim yang dianggap sebagai halusinasi HANYA jika: bertentangan "
            "dengan input deteksi (nomor gigi, permukaan, confidence), "
            "menyebutkan gigi/permukaan yang salah, atau membuat pernyataan "
            "klinis yang akan dianggap tidak benar oleh dokter gigi."
        ),
        evaluation_steps=[
            "Baca chunk konteks yang tersedia — catat topik klinisnya.",
            "Untuk setiap klaim dalam Temuan, Interpretasi, dan Rekomendasi: "
            "periksa apakah klaim tersebut (a) didukung oleh salah satu chunk "
            "konteks, ATAU (b) merupakan pengetahuan dental klinis umum yang "
            "valid (misalnya: interpretasi confidence threshold, protokol "
            "severity standar, teknik NILT untuk deteksi karies sekunder).",
            "Tandai sebagai halusinasi HANYA klaim yang: (i) bertentangan "
            "dengan input deteksi, (ii) menyebut nomor gigi atau permukaan "
            "yang berbeda dari query, atau (iii) membuat rekomendasi klinis "
            "yang secara medis tidak tepat untuk severitas yang disebutkan.",
            "Berikan skor 1 jika terdapat halusinasi faktual yang jelas; "
            "skor 3 jika semua klaim wajar secara klinis meski sebagian tidak "
            "terdapat dalam konteks; skor 5 jika semua klaim valid DAN "
            "sebagian besar dapat ditelusuri ke konteks.",
        ],
    ),
    Criterion(
        name="answer_relevancy",
        description=(
            "Answer relevancy mengukur apakah laporan_dokter secara langsung "
            "dan lengkap menjawab query deteksi — yaitu gigi spesifik, "
            "permukaan, kondisi yang terdeteksi, dan severitas yang disebutkan "
            "dalam input. Laporan yang relevan tetap fokus pada kasus ini "
            "dan tidak menyimpang ke konten klinis yang tidak terkait."
        ),
        evaluation_steps=[
            "Baca query deteksi: nomor gigi, permukaan/lokasi, kelas yang "
            "terdeteksi, confidence, severitas.",
            "Periksa apakah Temuan secara akurat mencerminkan deteksi spesifik "
            "yang disebutkan dalam query (gigi yang sama, permukaan yang sama, "
            "kondisi yang sama).",
            "Periksa apakah Interpretasi dan Rekomendasi spesifik untuk "
            "deteksi ini — bukan pernyataan generik yang bisa berlaku untuk "
            "kasus karies manapun.",
            "Kurangi skor jika ada konten off-topic, detail deteksi yang "
            "hilang, atau rekomendasi yang tidak sesuai dengan severitas "
            "yang terdeteksi.",
            "Berikan skor 1 (laporan sebagian besar mengabaikan atau "
            "melewatkan detail query) hingga 5 (laporan secara langsung dan "
            "lengkap membahas semua aspek query deteksi).",
        ],
    ),
    Criterion(
        name="context_precision",
        description=(
            "Context precision mengukur apakah chunk konteks yang diambil "
            "dari knowledge base benar-benar relevan dan berguna untuk "
            "menghasilkan laporan tentang deteksi spesifik ini. "
            "Knowledge base terdiri dari jurnal dental berbahasa Inggris; "
            "chunk berbahasa Inggris bukan masalah selama topiknya relevan. "
            "PENTING: chunk dianggap relevan jika membahas salah satu topik "
            "berikut — deteksi karies (termasuk karies primer maupun sekunder), "
            "staging lesi (ICDAS, kedalaman radiolusensi, radiolucency grading), "
            "deskripsi klinis lesi pada permukaan gigi (mesial, oklusal, bukal, dll), "
            "teknik deteksi (NILT, visual-taktil, radiografi), atau pendekatan "
            "manajemen karies (observasi, re-restorasi, kriteria intervensi). "
            "Chunk TIDAK perlu menyebutkan frasa 'karies sekunder' secara eksplisit "
            "untuk dianggap relevan — overlap topik klinis sudah cukup. "
            "Chunk yang tidak relevan: bibliografi/daftar penulis, epidemiologi "
            "edentulisme, topik sama sekali di luar karies atau restorasi."
        ),
        evaluation_steps=[
            "Baca setiap chunk konteks secara berurutan.",
            "Untuk setiap chunk, tentukan apakah chunk berisi informasi yang "
            "berguna secara klinis untuk laporan ini. Topik yang RELEVAN meliputi: "
            "deteksi karies (primer atau sekunder), staging lesi (ICDAS, "
            "radiolucency depth), deskripsi permukaan lesi, teknik deteksi "
            "(NILT, visual-taktil, radiografi bitewing/periapikal), atau kriteria "
            "manajemen karies. Chunk TIDAK harus menyebut 'karies sekunder' "
            "secara eksplisit — kesamaan topik klinis sudah cukup untuk relevansi.",
            "Catat apakah chunk paling berguna muncul di awal daftar (ranking "
            "bagus) atau tersembunyi di bawah (ranking buruk).",
            "Tandai sebagai tidak relevan HANYA: daftar penulis/bibliografi, "
            "topik yang sama sekali tidak berkaitan dengan karies atau restorasi "
            "(misalnya: epidemiologi edentulisme tanpa konteks preventif, "
            "prosedur bedah mulut yang tidak terkait karies).",
            "Berikan skor 1 (sebagian besar chunk tidak relevan secara klinis) "
            "hingga 5 (hampir semua chunk relevan secara klinis, chunk terbaik "
            "muncul di awal).",
        ],
    ),
    Criterion(
        name="context_recall",
        description=(
            "Context recall mengukur kelengkapan retrieval: sejauh mana "
            "informasi yang dibutuhkan untuk menjawab query klinis "
            "berhasil ditemukan oleh sistem retrieval dari knowledge base. "
            "Knowledge base terdiri dari jurnal dental berbahasa Inggris. "
            "PENTING: topik informasi kunci harus dinilai secara KONSEPTUAL, "
            "bukan secara verbatim. Chunk yang membahas staging lesi, kedalaman "
            "radiolusensi, atau kriteria intervensi dianggap mencakup topik "
            "'severity/manajemen karies' meskipun tidak menggunakan frasa "
            "'karies sekunder' atau 'protokol manajemen' secara eksplisit. "
            "Berikan kredit parsial secara proporsional: jika 3 dari 4 topik "
            "kunci terpenuhi = skor 4, bukan skor 1 hanya karena 1 topik hilang."
        ),
        evaluation_steps=[
            "Baca query deteksi dan tentukan 4 topik informasi kunci yang "
            "idealnya dibutuhkan: (1) karakteristik/mekanisme lesi karies "
            "(primer atau sekunder), (2) interpretasi severity atau staging "
            "lesi (ICDAS, confidence threshold, kedalaman radiolusensi), "
            "(3) teknik deteksi atau konfirmasi (NILT, visual-taktil, "
            "radiografi), (4) pendekatan manajemen berdasarkan severity "
            "(observasi, monitoring, intervensi, re-restorasi).",
            "Untuk setiap topik, periksa apakah setidaknya satu chunk "
            "mencakupnya SECARA KONSEPTUAL — tidak harus menggunakan frasa "
            "yang sama. Contoh: chunk tentang 'ICDAS code 3-4 radiolucency' "
            "mencakup topik (2) severity/staging; chunk tentang 'visual-taktil "
            "examination' mencakup topik (3) teknik deteksi.",
            "Hitung jumlah topik yang tercakup (0-4) dan konversi ke skor "
            "menggunakan tabel berikut SECARA EKSAK: "
            "4 topik tercakup = skor 5, "
            "3 topik tercakup = skor 4, "
            "2 topik tercakup = skor 3, "
            "1 topik tercakup = skor 2, "
            "0 topik tercakup = skor 1.",
            "Berikan skor sesuai tabel di atas. JANGAN mengurangi skor "
            "hanya karena frasa 'karies sekunder' atau 'protokol manajemen' "
            "tidak muncul verbatim dalam chunk — overlap konseptual sudah cukup.",
        ],
    ),
]

METRIC_KEYS = [c.name for c in RAGAS_CRITERIA]
CONTEXT_PRECISION_KEY = "context_precision"
CONTEXT_RECALL_KEY = "context_recall"
CONTEXT_METRICS_SKIP = {CONTEXT_PRECISION_KEY, CONTEXT_RECALL_KEY}


PROMPT_TEMPLATE = """Kamu adalah dokter gigi ahli yang mengevaluasi laporan klinis yang dihasilkan oleh sistem AI. Evaluasi SATU kriteria pada satu waktu menggunakan langkah-langkah chain-of-thought yang diberikan.

# Kriteria: {criterion_name}
{criterion_description}

# Langkah evaluasi (ikuti secara berurutan sebelum memberikan skor)
{steps}

# Konteks kasus
Input deteksi: {query}
Gigi: {tooth} | Lokasi: {lokasi} | Severitas: {severity}

# Chunk konteks yang diambil dari knowledge base
{contexts}

# Laporan yang dievaluasi (laporan_dokter)
{report_text}

# Instruksi
Ikuti langkah evaluasi di atas secara internal, lalu output HANYA objek JSON dengan TEPAT dua kunci ini dan tidak ada yang lain, tanpa markdown fence, tanpa teks tambahan:
{{"score": <integer 1-5>, "reasoning": "<justifikasi 1-3 kalimat dalam Bahasa Indonesia, mengutip bagian spesifik dari laporan atau konteks>"}}
"""

PROMPT_TEMPLATE_NO_CONTEXT = """Kamu adalah dokter gigi ahli yang mengevaluasi laporan klinis yang dihasilkan oleh sistem AI. Evaluasi SATU kriteria pada satu waktu.

# Kriteria: {criterion_name}
{criterion_description}

# Langkah evaluasi (ikuti secara berurutan sebelum memberikan skor)
{steps}

# Konteks kasus
Input deteksi: {query}
Gigi: {tooth} | Lokasi: {lokasi} | Severitas: {severity}

CATATAN: Tidak ada chunk konteks yang diambil dari knowledge base untuk kasus ini (kasus gigi sehat — pipeline secara sengaja melewati retrieval RAG).

# Laporan yang dievaluasi (laporan_dokter)
{report_text}

# Instruksi
Ikuti langkah evaluasi di atas secara internal, lalu output HANYA objek JSON dengan TEPAT dua kunci ini dan tidak ada yang lain, tanpa markdown fence, tanpa teks tambahan:
{{"score": <integer 1-5>, "reasoning": "<justifikasi 1-3 kalimat dalam Bahasa Indonesia, mengutip bagian spesifik dari laporan>"}}
"""


def build_prompt(criterion: Criterion, report_text: str, contexts_text: str,
                 query: str, tooth: str, lokasi: str, severity: str,
                 no_context: bool = False) -> str:
    """
    Construct evaluation prompt with or without RAG context chunks.

    Args:
        criterion: Target RAGAS evaluation criterion.
        report_text: Clinician report text under test.
        contexts_text: Formatted literature context passages.
        query: Clinical detection query string.
        tooth: FDI tooth number.
        lokasi: Lesion surface.
        severity: Severity category string.
        no_context: Flag indicating whether context chunks are absent.

    Returns:
        Formatted prompt ready for model evaluation.
    """
    steps = "\n".join(f"{i+1}. {s}" for i, s in enumerate(criterion.evaluation_steps))
    template = PROMPT_TEMPLATE_NO_CONTEXT if no_context else PROMPT_TEMPLATE
    fmt = dict(
        criterion_name=criterion.name,
        criterion_description=criterion.description,
        steps=steps,
        query=query,
        tooth=tooth,
        lokasi=lokasi,
        severity=severity,
        report_text=report_text,
    )
    if not no_context:
        fmt["contexts"] = contexts_text
    return template.format(**fmt)


def call_ollama(prompt: str, model: str = JUDGE_MODEL, base_url: str = OLLAMA_BASE_URL,
                max_retries: int = 5, timeout: int = 300) -> dict:
    """
    Call Ollama judge model with exponential retry backoff and robust JSON extraction.

    Args:
        prompt: Evaluation prompt string.
        model: Ollama model name.
        base_url: Server HTTP endpoint.
        max_retries: Number of retry attempts.
        timeout: Request timeout seconds.

    Returns:
        Parsed evaluation dictionary containing 'score' and 'reasoning'.
    """
    url = f"{base_url}/api/generate"
    full_prompt = (
        "Kamu adalah evaluator ahli. Selalu respons dengan JSON valid saja. "
        "JSON harus memiliki tepat dua kunci: \"score\" (integer 1-5) dan "
        "\"reasoning\" (string dalam Bahasa Indonesia). "
        "Tanpa markdown fence, tanpa teks tambahan, tanpa penjelasan di luar JSON.\n\n"
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
                last_err = RuntimeError(f"HTTP {resp.status_code} (attempt {attempt})")
                print(f"  [server-error {resp.status_code}] retry {attempt}/{max_retries} "
                      f"in {wait:.0f}s...", file=sys.stderr)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            raw_text = data["response"]
            # Strip reasoning tags
            text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
            text = re.sub(r"^\s*$", "", text).strip()
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if match:
                return json.loads(match.group())
            print(f"  [debug] raw_len={len(raw_text)}, stripped_len={len(text)}, "
                  f"preview={text[:200]!r}", file=sys.stderr)
            raise json.JSONDecodeError("No JSON object found in response", text, 0)
        except requests.exceptions.Timeout as e:
            last_err = e
            wait = min(30 * (2 ** (attempt - 1)), 300)
            print(f"  [timeout] retry {attempt}/{max_retries} in {wait}s...", file=sys.stderr)
            time.sleep(wait)
        except (requests.RequestException, KeyError, IndexError, json.JSONDecodeError) as e:
            last_err = e
            wait = 2 ** attempt
            print(f"  [error: {e}] retry {attempt}/{max_retries} in {wait}s...", file=sys.stderr)
            time.sleep(wait)

    raise RuntimeError(f"Ollama call failed after {max_retries} retries: {last_err}")


def extract_score(parsed: dict, criterion_name: str) -> tuple[Optional[int], str]:
    """
    Extract integer score and reasoning from JSON dictionary using fuzzy key heuristics.

    Args:
        parsed: Response dictionary from Ollama judge.
        criterion_name: Name of evaluated metric.

    Returns:
        Tuple of (score_int, reasoning_str).
    """
    score = None
    for key in ("score", "skor", "nilai", "Score", "SCORE"):
        if key in parsed:
            score = parsed[key]
            break
    if score is None:
        for v in parsed.values():
            if isinstance(v, int) and 1 <= v <= 5:
                score = v
                break

    reasoning = ""
    for key in ("reasoning", "reason", "alasan", "justifikasi",
                "penjelasan", "Reasoning", "REASONING"):
        if key in parsed:
            reasoning = str(parsed[key])
            break
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


def load_report(path: str) -> dict:
    """
    Load and parse a clinical report JSON file from disk.

    Args:
        path: Path to target JSON file.

    Returns:
        Parsed JSON dictionary.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def assemble_report_text(report: dict) -> str:
    """
    Format clinician report sections into a unified text block for evaluation.

    Args:
        report: Parsed report sidecar dictionary.

    Returns:
        String containing Temuan, Interpretasi, and Rekomendasi sections.
    """
    d = report["report"]["laporan_dokter"]
    return (
        f"Temuan: {d.get('temuan', '')}\n"
        f"Interpretasi: {d.get('interpretasi', '')}\n"
        f"Rekomendasi: {d.get('rekomendasi', '')}"
    )


def assemble_contexts_text(report: dict) -> str:
    """
    Format retrieved knowledge base context passages into labeled text blocks.

    Args:
        report: Parsed report sidecar dictionary containing 'contexts' list.

    Returns:
        Concatenated chunk text blocks or empty string if none present.
    """
    contexts = report.get("contexts", [])
    if not contexts:
        return ""
    chunks = contexts[:MAX_CONTEXT_CHUNKS]
    return "\n\n".join(f"[Chunk {i+1}]\n{c}" for i, c in enumerate(chunks))


def evaluate_report(report: dict, report_id: str,
                    model: str = JUDGE_MODEL,
                    base_url: str = OLLAMA_BASE_URL,
                    sleep_between_calls: float = SLEEP_BETWEEN_CALLS) -> dict:
    """
    Evaluate a single clinical report across all configured RAGAS metrics.

    Args:
        report: Parsed report dictionary.
        report_id: Unique report identifier (UID).
        model: Ollama judge model name.
        base_url: Ollama HTTP base endpoint.
        sleep_between_calls: Delay in seconds between successive metric evaluations.

    Returns:
        Dictionary mapping metric names to their score and reasoning payloads.
    """
    query       = report.get("query", "")
    manual      = report.get("manual_input", {})
    tooth       = manual.get("no_gigi", "")
    lokasi      = manual.get("lokasi", "")
    severity    = report.get("detection", {}).get("severity") or "N/A (healthy)"

    report_text   = assemble_report_text(report)
    contexts_text = assemble_contexts_text(report)
    has_contexts  = bool(contexts_text)

    result = {"uid": report_id, "report_id": report_id}

    for crit in RAGAS_CRITERIA:
        # Context-dependent metrics are skipped for healthy cases
        if crit.name in CONTEXT_METRICS_SKIP and not has_contexts:
            result[crit.name] = {
                "score":     None,
                "reasoning": "N/A — healthy case, no context chunks retrieved (by design)."
            }
            print(f"  {crit.name}: N/A (no contexts)")
            continue

        prompt = build_prompt(
            crit, report_text, contexts_text,
            query, tooth, lokasi, severity,
            no_context=(not has_contexts),
        )
        parsed = call_ollama(prompt, model=model, base_url=base_url)
        score, reasoning = extract_score(parsed, crit.name)
        result[crit.name] = {"score": score, "reasoning": reasoning}
        print(f"  {crit.name}: {score}")
        time.sleep(sleep_between_calls)

    return result


def find_reports(output_dir: Path, patient_id_prefix: Optional[str] = None) -> list[Path]:
    """
    Discover all candidate report JSON files under output_dir (healthy/ and caries/).

    Excludes per-report evaluation sidecars (*_geval.json, *_ragas.json).
    Optionally filters by patient_id prefix (case-insensitive).

    Args:
        output_dir: Root output directory.
        patient_id_prefix: Optional UID prefix filter.

    Returns:
        Sorted list of candidate Path objects.
    """
    subdirs = [output_dir / "healthy", output_dir / "caries"]
    candidates: list[Path] = []

    for d in subdirs:
        if d.exists() and d.is_dir():
            for p in d.glob("*.json"):
                if not (p.name.endswith("_geval.json") or p.name.endswith("_ragas.json")):
                    candidates.append(p)

    if not candidates and output_dir.exists() and output_dir.is_dir():
        for p in output_dir.glob("*.json"):
            if not (p.name.endswith("_geval.json") or p.name.endswith("_ragas.json")):
                candidates.append(p)

    candidates = sorted(set(candidates))

    if patient_id_prefix:
        prefix_norm = patient_id_prefix.lower()
        candidates = [p for p in candidates if p.stem.lower().startswith(prefix_norm)]

    return candidates


def get_evaluated_uids(csv_path: Path) -> set[str]:
    """
    Read the aggregate CSV once and return the set of already evaluated UIDs.

    Args:
        csv_path: Path to ragas.csv.

    Returns:
        Set of UIDs already present in the CSV.
    """
    evaluated: set[str] = set()
    if csv_path.exists() and csv_path.stat().st_size > 0:
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                uid = row.get("uid") or row.get("report_id")
                if uid:
                    evaluated.add(uid.strip())
    return evaluated


def append_to_csv_rows(result: dict, rows: list[dict]) -> None:
    """
    Flatten evaluation results into row items for spreadsheet export.

    Args:
        result: Evaluation dictionary.
        rows: Destination list for CSV row records.
    """
    uid = result.get("uid") or result.get("report_id", "")
    for metric in METRIC_KEYS:
        payload = result.get(metric, {})
        if isinstance(payload, dict):
            score     = payload.get("score")
            reasoning = payload.get("reasoning", "")
        else:
            score     = payload
            reasoning = ""
        rows.append({
            "uid":       uid,
            "metric":    metric,
            "score":     score,
            "reasoning": reasoning,
        })


def append_csv_rows(csv_path: Path, rows: list[dict]) -> None:
    """
    Append formatted rows to the aggregate CSV file, initializing with headers if new.

    Args:
        csv_path: Path to target CSV file.
        rows: Rows to append.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists() and csv_path.stat().st_size > 0
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def check_ollama(model: str = JUDGE_MODEL, base_url: str = OLLAMA_BASE_URL) -> None:
    """
    Verify network connectivity and judge model presence on local Ollama server.
    """
    try:
        r = requests.get(f"{base_url}/api/tags", timeout=5)
        r.raise_for_status()
        models = [m["name"] for m in r.json().get("models", [])]
        if not any(model in m for m in models):
            print(f"WARNING: model '{model}' not found in Ollama. "
                  f"Available: {models}. Will attempt to pull on first call.",
                  file=sys.stderr)
        else:
            print(f"Using Ollama judge model: {model}")
    except requests.RequestException as e:
        print(f"ERROR: cannot reach Ollama at {base_url}: {e}", file=sys.stderr)
        print("Ensure 'ollama serve' is running.", file=sys.stderr)
        sys.exit(1)


def run_ragas(output_dir: Optional[Path] = None,
              patient_id: Optional[str] = None,
              model: str = JUDGE_MODEL,
              base_url: str = OLLAMA_BASE_URL,
              sleep_between_calls: float = SLEEP_BETWEEN_CALLS) -> list[str]:
    """
    Run RAGAS evaluation suite across all matching clinical reports.

    Uses outputs/ragas.csv as the skip-existing index.
    Writes <uid>_ragas.json sidecar first, then appends to outputs/ragas.csv.

    Args:
        output_dir: Destination root directory containing healthy/ and caries/.
        patient_id: Optional UID prefix filter.
        model: Judge LLM model identifier.
        base_url: Ollama base endpoint URL.
        sleep_between_calls: Delay in seconds between evaluation calls.

    Returns:
        List of UIDs evaluated in this execution.
    """
    target_dir = Path(output_dir) if output_dir is not None else OUTPUT_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    csv_path = target_dir / "ragas.csv"

    check_ollama(model=model, base_url=base_url)

    report_paths = find_reports(target_dir, patient_id_prefix=patient_id)
    if not report_paths:
        filter_msg = f" matching prefix '{patient_id}'" if patient_id else ""
        print(f"No clinical report JSONs found in {target_dir}{filter_msg}.")
        return []

    evaluated_uids = get_evaluated_uids(csv_path) if SKIP_EXISTING else set()
    evaluated_count = 0
    evaluated_list: list[str] = []

    print(f"Found {len(report_paths)} report(s) in {target_dir}. Already evaluated in CSV: {len(evaluated_uids)}")

    for path in report_paths:
        uid = path.stem
        if SKIP_EXISTING and uid in evaluated_uids:
            print(f"[skip] {uid} (already evaluated in CSV)")
            continue

        print(f"[eval] {uid}")
        try:
            report_data = load_report(str(path))
            result = evaluate_report(report_data, uid, model=model, base_url=base_url,
                                     sleep_between_calls=sleep_between_calls)

            # Crash-safety write order: 1. write sidecar JSON, 2. append CSV row
            sidecar_path = path.parent / f"{uid}_ragas.json"
            with open(sidecar_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            rows: list[dict] = []
            append_to_csv_rows(result, rows)
            append_csv_rows(csv_path, rows)

            evaluated_uids.add(uid)
            evaluated_list.append(uid)
            evaluated_count += 1
        except Exception as e:
            print(f"  [FAILED] {uid}: {e}", file=sys.stderr)
            continue

    print(f"\nRAGAS completed. {evaluated_count} report(s) evaluated.")
    print(f"Aggregate CSV: {csv_path}")
    return evaluated_list


def main():
    """CLI entry point for standalone RAGAS execution."""
    parser = argparse.ArgumentParser(description="Run RAGAS reference-free evaluation suite.")
    parser.add_argument("--patient-id", type=str, default=None,
                        help="Optional UID prefix filter (e.g. 'liza' or 'liza_36')")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Root output directory (defaults to OUTPUT_DIR env var)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    run_ragas(output_dir=out_dir, patient_id=args.patient_id)


if __name__ == "__main__":
    main()
