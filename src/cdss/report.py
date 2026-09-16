"""
End-to-End Clinical Decision Support System (CDSS) Reporting Pipeline.

This module orchestrates the complete diagnostic and reporting workflow:
1. YOLOv8 Instance Segmentation: Detects secondary caries in Near-Infrared Light
   Transillumination (NILT) dental images.
2. Clinical Query Construction: Translates visual detections and practitioner input
   (FDI tooth number, surface location) into structured clinical queries.
3. Cross-Lingual RAG Retrieval: Translates queries to English and retrieves relevant
   evidence-based dental literature from ChromaDB.
4. LLM Report Synthesis: Uses a clinical LLM (e.g., Gemma, Qwen) to synthesize dual-audience
   reports in Indonesian (clinician-facing and patient-facing).
5. PDF Document Generation: Typesets a multi-page clinical report with side-by-side
   image overlays, diagnostic interpretations, and tailored oral hygiene guidance.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import chromadb
from fpdf import FPDF
from openai import OpenAI
from PIL import Image
from sentence_transformers import SentenceTransformer
from ultralytics import YOLO

# ── CONFIGURATION & ENVIRONMENT ─────────────────────
BASE_DIR        = Path(__file__).resolve().parent
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", str(BASE_DIR.parent / "yolo" / "weights" / "best.pt"))
CHROMA_PATH     = Path(os.getenv("CHROMA_PATH", str(BASE_DIR / "chroma_db")))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "karies_knowledge")
OUTPUT_FOLDER   = Path(os.getenv("CDSS_OUTPUT_DIR", str(BASE_DIR / "outputs" / "reports")))
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
LLM_MODEL       = os.getenv("LLM_MODEL", "gemma3:12b")
LLM_API_KEY     = os.getenv("LLM_API_KEY", "ollama")
TOP_K_CHUNKS    = int(os.getenv("TOP_K_CHUNKS", "5"))
# ────────────────────────────────────────────────────

# Module-level component caches for lazy instantiation
_yolo_model: Optional[YOLO] = None
_embedder: Optional[SentenceTransformer] = None
_chroma_client: Optional[chromadb.PersistentClient] = None
_chroma_collection = None
_llm_client: Optional[OpenAI] = None


def get_yolo_model(model_path: str = YOLO_MODEL_PATH) -> YOLO:
    """
    Retrieve or lazily initialize the YOLOv8 segmentation model.

    Args:
        model_path: Path to the trained YOLO weights file (.pt).

    Returns:
        Loaded Ultralytics YOLO model instance.
    """
    global _yolo_model
    if _yolo_model is None:
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"YOLO model weights not found at: {model_path}. "
                "Please configure YOLO_MODEL_PATH in your environment or .env file."
            )
        _yolo_model = YOLO(model_path)
    return _yolo_model


def get_embedder() -> SentenceTransformer:
    """
    Retrieve or lazily initialize the biomedical text embedding model.

    Returns:
        SentenceTransformer instance (NeuML/pubmedbert-base-embeddings).
    """
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("NeuML/pubmedbert-base-embeddings")
    return _embedder


def get_chroma_collection(chroma_path: Path = CHROMA_PATH,
                          collection_name: str = COLLECTION_NAME):
    """
    Retrieve or lazily connect to the ChromaDB vector database collection.

    Args:
        chroma_path: Path to ChromaDB persistent storage directory.
        collection_name: Name of the target vector collection.

    Returns:
        ChromaDB collection instance.
    """
    global _chroma_client, _chroma_collection
    if _chroma_collection is None:
        if _chroma_client is None:
            _chroma_client = chromadb.PersistentClient(path=str(chroma_path))
        _chroma_collection = _chroma_client.get_collection(collection_name)
    return _chroma_collection


def get_llm_client(base_url: str = OLLAMA_BASE_URL,
                   api_key: str = LLM_API_KEY) -> OpenAI:
    """
    Retrieve or lazily initialize the OpenAI-compatible LLM client (Ollama/OpenAI).

    Args:
        base_url: Base endpoint URL for the inference server.
        api_key: API authorization key.

    Returns:
        Configured OpenAI client instance.
    """
    global _llm_client
    if _llm_client is None:
        _llm_client = OpenAI(base_url=base_url, api_key=api_key)
    return _llm_client


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


# FDI tooth notation to Indonesian colloquial name mapping
# Used exclusively in patient reports to ensure patients understand their tooth locations.
FDI_TOOTH_NAME: dict[str, str] = {
    # Quadrant 1 — Upper Right
    "11": "gigi seri pertama atas kanan",
    "12": "gigi seri kedua atas kanan",
    "13": "gigi taring atas kanan",
    "14": "gigi premolar pertama atas kanan",
    "15": "gigi premolar kedua atas kanan",
    "16": "gigi geraham pertama atas kanan",
    "17": "gigi geraham kedua atas kanan",
    "18": "gigi geraham bungsu atas kanan",
    # Quadrant 2 — Upper Left
    "21": "gigi seri pertama atas kiri",
    "22": "gigi seri kedua atas kiri",
    "23": "gigi taring atas kiri",
    "24": "gigi premolar pertama atas kiri",
    "25": "gigi premolar kedua atas kiri",
    "26": "gigi geraham pertama atas kiri",
    "27": "gigi geraham kedua atas kiri",
    "28": "gigi geraham bungsu atas kiri",
    # Quadrant 3 — Lower Left
    "31": "gigi seri pertama bawah kiri",
    "32": "gigi seri kedua bawah kiri",
    "33": "gigi taring bawah kiri",
    "34": "gigi premolar pertama bawah kiri",
    "35": "gigi premolar kedua bawah kiri",
    "36": "gigi geraham pertama bawah kiri",
    "37": "gigi geraham kedua bawah kiri",
    "38": "gigi geraham bungsu bawah kiri",
    # Quadrant 4 — Lower Right
    "41": "gigi seri pertama bawah kanan",
    "42": "gigi seri kedua bawah kanan",
    "43": "gigi taring bawah kanan",
    "44": "gigi premolar pertama bawah kanan",
    "45": "gigi premolar kedua bawah kanan",
    "46": "gigi geraham pertama bawah kanan",
    "47": "gigi geraham kedua bawah kanan",
    "48": "gigi geraham bungsu bawah kanan",
}


def get_tooth_name(no_gigi: str) -> str:
    """
    Return descriptive Indonesian tooth name from FDI notation with fallback.

    Args:
        no_gigi: Two-digit FDI tooth notation string (e.g., '36').

    Returns:
        Colloquial Indonesian tooth name.
    """
    return FDI_TOOTH_NAME.get(str(no_gigi).strip(), f"gigi {no_gigi}")


# ── COMPONENT 1: YOLOv8 INFERENCE ───────────────────
def run_yolo(image_path: str, output_folder: Path = OUTPUT_FOLDER) -> dict:
    """
    Run YOLOv8 segmentation inference and save side-by-side visualization images.

    Args:
        image_path: Filesystem path to the input NILT dental image.
        output_folder: Destination folder for output overlay images.

    Returns:
        Dictionary containing class_name, confidence, severity, vis_path, and orig_path.
    """
    model = get_yolo_model()
    output_folder.mkdir(parents=True, exist_ok=True)

    results = model(image_path)
    result  = results[0]

    # Save copy of original image for side-by-side comparison
    orig_path = str(output_folder / "yolo_original.jpg")
    Image.open(image_path).convert("RGB").save(orig_path)

    # Save segmentation mask overlay
    vis_path = str(output_folder / "yolo_result.jpg")
    result.save(filename=vis_path)

    # Extract primary detection (assuming one lesion per cropped tooth image)
    if len(result.boxes) == 0:
        return {
            "class_name": None,
            "confidence": None,
            "severity":   None,
            "vis_path":   vis_path,
            "orig_path":  orig_path,
        }

    box        = result.boxes[0]
    class_id   = int(box.cls[0])
    confidence = float(box.conf[0])
    class_name = model.names[class_id]

    return {
        "class_name":  class_name,
        "confidence":  confidence,
        "vis_path":    vis_path,
        "orig_path":   orig_path,
        "severity":    severity_label(confidence),
    }


# ── COMPONENT 2: QUERY BUILDER ──────────────────────
def get_manual_input() -> dict:
    """
    Prompt dentist for manual clinical metadata via interactive terminal.

    Returns:
        Dictionary with keys 'no_gigi' (FDI notation) and 'lokasi' (lesion surface).
    """
    no_gigi = input("Tooth number (FDI notation, 11-48): ").strip()
    lokasi  = input("Lesion location / surface (mesial/distal/oklusal/servikal/bukal/lingual): ").strip().lower()
    return {"no_gigi": no_gigi, "lokasi": lokasi}


def build_query(detection: dict, manual: dict) -> str:
    """
    Synthesize YOLO detection output and manual clinical metadata into a clinical query.

    Args:
        detection: Detection result dictionary from run_yolo.
        manual: Practitioner input dictionary with tooth number and surface.

    Returns:
        Formatted clinical query sentence in Indonesian.
    """
    if detection is None or detection.get("class_name") is None:
        return build_query_healthy(manual)

    conf       = detection["confidence"]
    conf_pct   = round(conf * 100)
    tooth_name = get_tooth_name(manual["no_gigi"])

    if conf >= 0.75:
        severity_label_str = "tinggi"
        urgency            = "memerlukan tindakan klinis segera"
    elif conf >= 0.50:
        severity_label_str = "sedang"
        urgency            = "memerlukan konfirmasi radiografi bitewing"
    else:
        severity_label_str = "rendah"
        urgency            = "memerlukan observasi dan konfirmasi klinis lanjutan"

    return (
        f"Deteksi karies sekunder pada {tooth_name} (gigi {manual['no_gigi']}) "
        f"permukaan {manual['lokasi']} dengan confidence {conf_pct}% "
        f"(severitas {severity_label_str}). "
        f"Temuan ini {urgency}. "
        f"Hasil segmentasi instance dari citra NILT."
    )


def build_query_healthy(manual: dict) -> str:
    """
    Construct clinical query string for healthy teeth without detected caries.

    Args:
        manual: Practitioner input dictionary with tooth number and surface.

    Returns:
        Formatted clinical query sentence in Indonesian for healthy findings.
    """
    tooth_name = get_tooth_name(manual["no_gigi"])
    return (
        f"Tidak terdeteksi karies sekunder pada {tooth_name} (gigi {manual['no_gigi']}) "
        f"di lokasi {manual['lokasi']}. Hasil segmentasi instance dari "
        f"citra NILT tidak menunjukkan adanya lesi."
    )


def translate_query_to_english(query: str) -> str:
    """
    Translate an Indonesian clinical query into an English query optimized for PubMedBERT retrieval.

    Args:
        query: Clinical query string in Indonesian.

    Returns:
        Optimized English search query.
    """
    client = get_llm_client()
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert dental information retrieval system. Extract the core clinical concepts "
                        "from the given Indonesian patient query to form a broad, keyword-rich English search query "
                        "optimized for semantic search in a dental journal database.\n\n"
                        "Rules:\n"
                        "- Focus on the condition (e.g. secondary caries), location (e.g. mesial surface), "
                        "and required action (e.g. clinical management, observation, immediate treatment).\n"
                        "- DROP highly specific patient details like exact tooth numbers (e.g. tooth 31, gigi 46), "
                        "exact confidence scores (e.g. 75%), and AI terminology (e.g. instance segmentation).\n"
                        "- Output ONLY the optimized English search query, no explanation, no quotes, no markdown."
                    )
                },
                {"role": "user", "content": query}
            ],
            temperature=0.0,
            max_tokens=256,
        )
        translated = response.choices[0].message.content.strip()
        translated = translated.replace('"', '').replace("'", "").strip()
        return translated
    except Exception as e:
        print(f"      [Warning] Query translation failed: {e}. Falling back to original query.")
        return query


# ── COMPONENT 3: RAG RETRIEVAL ──────────────────────
def retrieve_context(query: str, top_k: int = TOP_K_CHUNKS) -> list[str]:
    """
    Retrieve Top-K most relevant document chunks from ChromaDB knowledge base.

    Args:
        query: Semantic query text in English.
        top_k: Number of chunks to retrieve.

    Returns:
        List of text chunks from relevant dental journals.
    """
    embedder = get_embedder()
    collection = get_chroma_collection()

    query_embedding = embedder.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )
    return results["documents"][0]


# ── COMPONENT 4: LLM REPORT GENERATION ──────────────
SYSTEM_PROMPT = """Kamu adalah sistem CDSS untuk dokter gigi.
Gunakan referensi klinis yang diberikan untuk membuat laporan akurat.

ATURAN PENTING:
- Tulis SELURUH laporan dalam Bahasa Indonesia. JANGAN gunakan Bahasa Inggris.
- Untuk istilah medis, gunakan istilah Indonesia yang umum dipakai
  (contoh: "karies sekunder" bukan "secondary caries",
  "restorasi" bukan "restoration", "pemeriksaan klinis" bukan "clinical examination").

LAPORAN DOKTER RULES:
- JANGAN ekspansi nomor gigi ke nama Indonesia (misal JANGAN tulis "gigi 36 (gigi geraham atas kanan)", cukup "gigi 36")
- JANGAN ekspansi singkatan NILT — tulis "NILT" saja, tanpa kepanjangan
- JANGAN definisikan apa itu karies sekunder — asumsikan dokter gigi sudah tahu
- JANGAN jelaskan apa itu radiografi — gunakan "radiografi bitewing" atau "radiografi periapikal" langsung, tanpa keterangan dalam kurung
- JANGAN sertakan saran penanganan generik — sesuaikan rekomendasi dengan tingkat keparahan yang disebutkan di query:
  * Rendah (<50%): monitoring aktif dan re-evaluasi klinis pada kunjungan berikutnya; tidak perlu intervensi segera.
  * Sedang (50–75%): konfirmasi dengan radiografi bitewing untuk menilai kedalaman lesi dan integritas margin; pertimbangkan re-restorasi bila ditemukan kebocoran marginal.
  * Tinggi (>75%): indikasi re-restorasi atau perawatan endodontik jika lesi telah mencapai dentin dalam; lakukan radiografi periapikal untuk evaluasi keterlibatan pulpa.
- JANGAN awali rekomendasi dengan kata "Rekomendasi:" — langsung tulis isi rekomendasinya
- JANGAN sertakan faktor etiologi (diet, kebersihan mulut) di laporan dokter — itu masuk ke laporan pasien
- Maksimal 2–3 kalimat per field. Padat dan klinis.

LAPORAN PASIEN RULES:
- WAJIB gunakan bahasa awam — JANGAN gunakan istilah kedokteran gigi (odontologi) yang tidak dikenal masyarakat umum
- Gunakan padanan berikut secara KONSISTEN (kiri = istilah teknis yang DILARANG, kanan = yang HARUS dipakai):
  * "molar" / "geraham" → "gigi geraham"
  * "premolar" / "bikuspid" → "gigi geraham kecil" atau gunakan nama lengkap dari query
  * "insisor" / "insisivus" → "gigi seri"
  * "kaninus" / "kuspid" → "gigi taring"
  * "oklusal" → "permukaan kunyah"
  * "bukal" → "sisi luar gigi (yang menghadap pipi)"
  * "lingual" / "palatal" → "sisi dalam gigi (yang menghadap lidah)"
  * "mesial" / "distal" → "sela-sela antar gigi"
  * "servikal" → "pangkal gigi (dekat gusi)"
  * "restorasi" / "tumpatan" → "tambalan"
  * "lesi" → "kerusakan" atau "lubang"
  * "margin restorasi" → "tepi tambalan"
  * "kebocoran marginal" → "tambalan yang bocor atau tidak rapat"
  * "karies sekunder" → BOLEH digunakan tapi WAJIB dijelaskan dalam 1 kalimat pertama
  * "radiografi bitewing" / "radiografi periapikal" → "foto rontgen gigi"
  * "dentin" → "lapisan dalam gigi"
  * "pulpa" → "saraf gigi"
  * "endodontik" → "perawatan saraf gigi"
  * "plak" → "plak (kotoran yang menempel di gigi)"
  * "kalkulus" → "karang gigi"
  * "profilaksis" → "pembersihan gigi"
  * "NILT" → JANGAN disebut sama sekali di laporan pasien
  * "confidence" / "probabilitas" / "segmentasi" / "AI" → JANGAN disebut sama sekali
- JANGAN tampilkan skor confidence mentah sebagai ketidakyakinan dokter (JANGAN tulis "dokter tidak yakin" atau "tingkat kepercayaan X%")
- Jika confidence rendah/sedang, framing sebagai "diperlukan pemeriksaan lanjutan untuk konfirmasi", BUKAN "dokter ragu"
- JANGAN gunakan kata "kami" — gunakan "dokter gigi" atau bentuk pasif ("disarankan")
- JANGAN tulis frasa penutup informal seperti "jangan khawatir" atau "kami akan menjelaskan semuanya"
- Di bagian "ringkasan": WAJIB sebutkan nama gigi dalam Bahasa Indonesia yang sudah disediakan di query (contoh: "gigi premolar kedua atas kanan (gigi 15)")
- Di bagian "ringkasan": WAJIB jelaskan secara singkat apa itu karies sekunder dalam 1 kalimat sederhana (contoh: "Karies sekunder adalah kerusakan gigi baru yang muncul di sekitar tambalan yang sudah ada.")
- Di bagian "saran": sesuaikan dengan lokasi lesi yang disebutkan di query:
  * oklusal (permukaan kunyah): tekankan menyikat permukaan kunyah secara menyeluruh
  * bukal (sisi luar gigi yang menghadap pipi): tekankan menyikat sisi luar gigi dengan gerakan memutar
  * lingual (sisi dalam gigi yang menghadap lidah): tekankan menyikat sisi dalam gigi dengan gerakan memutar
  * mesial/distal (sela-sela antar gigi): tekankan penggunaan benang gigi (dental floss) setiap hari
  * servikal (pangkal gigi dekat gusi): tekankan menyikat pangkal gigi dengan lembut menggunakan gerakan memutar kecil agar tidak melukai gusi
- Di bagian "saran": sesuaikan urgensi kunjungan dengan severitas:
  * Tinggi: "Disarankan untuk segera menjadwalkan kunjungan dalam waktu dekat"
  * Sedang: "Disarankan untuk menjadwalkan kunjungan pemeriksaan lanjutan"
  * Rendah: "Disarankan untuk tetap melakukan pemeriksaan rutin berkala"
- Gunakan bahasa sederhana, bermartabat, dan formal
- Maksimal 3 kalimat per field.

KASUS GIGI SEHAT (TIDAK ADA DETEKSI):
- Jika tidak ada deteksi, buat laporan "tidak ditemukan karies"
- JANGAN tulis "kami tidak menemukan" — gunakan "tidak ditemukan" (pasif)
- Tekankan pentingnya pemantauan rutin di kedua jenis laporan

=== CONTOH OUTPUT YANG BENAR ===

# Contoh 1 — Severitas TINGGI (confidence 83%)
Query: Deteksi karies sekunder pada gigi 35 permukaan oklusal dengan confidence 83% (severitas tinggi). Temuan ini memerlukan tindakan klinis segera.

Output:
{
  "laporan_dokter": {
    "temuan": "Terdeteksi karies sekunder pada permukaan oklusal gigi 35 dengan confidence 83% berdasarkan analisis citra NILT.",
    "interpretasi": "Confidence 83% mengindikasikan lesi aktif dengan probabilitas tinggi adanya kebocoran marginal pada restorasi eksisting. Perlu konfirmasi melalui pemeriksaan visual-taktil dan radiografi periapikal untuk menilai keterlibatan dentin.",
    "rekomendasi": "Radiografi periapikal direkomendasikan untuk evaluasi kedalaman lesi dan kemungkinan keterlibatan pulpa. Jika ditemukan kebocoran marginal luas atau keterlibatan dentin dalam, indikasi re-restorasi atau perawatan endodontik perlu dipertimbangkan."
  },
  "laporan_pasien": {
    "ringkasan": "Karies sekunder adalah kerusakan gigi baru yang muncul di sekitar tambalan yang sudah ada. Hasil pemeriksaan pada gigi geraham kecil kedua bawah kiri (gigi 35) menunjukkan indikasi kerusakan seperti ini di permukaan kunyah, dan dokter gigi akan melakukan pemeriksaan lanjutan untuk memastikan kondisinya.",
    "saran": "Disarankan untuk segera menjadwalkan kunjungan dalam waktu dekat untuk pemeriksaan dan penanganan lebih lanjut. Saat menyikat gigi, pastikan permukaan kunyah gigi belakang disikat secara menyeluruh karena sisa makanan mudah menumpuk di area tersebut."
  }
}

# Contoh 2 — Severitas SEDANG (confidence 62%)
Query: Deteksi karies sekunder pada gigi 15 permukaan oklusal dengan confidence 62% (severitas sedang). Temuan ini memerlukan konfirmasi radiografi bitewing.

Output:
{
  "laporan_dokter": {
    "temuan": "Terdeteksi karies sekunder pada permukaan oklusal gigi 15 dengan confidence 62% berdasarkan analisis citra NILT.",
    "interpretasi": "Confidence 62% menunjukkan indikasi lesi pada area margin restorasi yang memerlukan konfirmasi lebih lanjut. Kemungkinan terdapat kebocoran marginal yang belum mencapai dentin dalam.",
    "rekomendasi": "Pemeriksaan radiografi bitewing direkomendasikan untuk menilai integritas margin restorasi dan kedalaman lesi. Jika ditemukan kebocoran marginal, pertimbangkan re-restorasi dengan protokol adhesif total-etch atau self-etch."
  },
  "laporan_pasien": {
    "ringkasan": "Karies sekunder adalah kerusakan gigi baru yang muncul di sekitar tambalan yang sudah ada. Hasil pemeriksaan pada gigi geraham kecil kedua atas kanan (gigi 15) menunjukkan kemungkinan adanya kerusakan seperti ini di permukaan kunyah, sehingga diperlukan pemeriksaan lanjutan untuk memastikannya.",
    "saran": "Disarankan untuk menjadwalkan kunjungan pemeriksaan lanjutan, termasuk foto rontgen gigi, untuk menilai kondisi tambalan secara lebih akurat. Pastikan permukaan kunyah gigi belakang disikat dengan baik setiap kali menyikat gigi, dan kurangi konsumsi makanan atau minuman manis."
  }
}

# Contoh 3 — Severitas RENDAH (confidence 33%)
Query: Deteksi karies sekunder pada gigi 16 permukaan oklusal dengan confidence 33% (severitas rendah). Temuan ini memerlukan observasi dan konfirmasi klinis lanjutan.

Output:
{
  "laporan_dokter": {
    "temuan": "Terdapat indikasi karies sekunder pada permukaan oklusal gigi 16 dengan confidence 33% berdasarkan analisis citra NILT.",
    "interpretasi": "Confidence 33% menunjukkan sinyal lesi yang lemah dan belum konklusif. Temuan ini kemungkinan merupakan lesi awal atau artefak citra yang memerlukan observasi klinis.",
    "rekomendasi": "Lakukan pemeriksaan visual-taktil langsung pada kunjungan ini untuk mengkonfirmasi integritas margin restorasi. Jika tidak ditemukan tanda klinis kebocoran, jadwalkan monitoring pada kunjungan recall berikutnya tanpa intervensi segera."
  },
  "laporan_pasien": {
    "ringkasan": "Karies sekunder adalah kerusakan gigi baru yang muncul di sekitar tambalan yang sudah ada. Hasil pemeriksaan pada gigi geraham pertama atas kanan (gigi 16) menunjukkan sinyal awal yang perlu dikonfirmasi lebih lanjut melalui pemeriksaan langsung oleh dokter gigi.",
    "saran": "Disarankan untuk tetap melakukan pemeriksaan rutin berkala agar kondisi tambalan dapat dipantau dengan baik. Perhatikan kebersihan permukaan kunyah gigi belakang saat menyikat gigi, dan batasi konsumsi makanan manis untuk mencegah kerusakan lebih lanjut."
  }
}
=== AKHIR CONTOH ===

Respond HANYA dalam format JSON berikut, tanpa teks tambahan, tanpa markdown code block:
{
  "laporan_dokter": {
    "temuan": "...",
    "interpretasi": "...",
    "rekomendasi": "..."
  },
  "laporan_pasien": {
    "ringkasan": "...",
    "saran": "..."
  }
}"""


def generate_report(query: str, context_chunks: list[str]) -> dict:
    """
    Generate dual-audience clinical reports (dentist & patient) using LLM with RAG context.

    Args:
        query: Clinical query string describing the detected case.
        context_chunks: Evidence-based literature passages retrieved from ChromaDB.

    Returns:
        Parsed JSON dictionary containing 'laporan_dokter' and 'laporan_pasien'.
    """
    client = get_llm_client()
    context = "\n\n---\n\n".join(context_chunks)

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT + f"\n\nReferensi Klinis:\n{context}"
            },
            {"role": "user", "content": query}
        ],
        temperature=0.3,
        top_p=0.85,
        max_tokens=1024,
        extra_body={
            "options": {
                "top_k": 40,
                "repeat_penalty": 1.2,
                "num_predict": 1024,
            }
        }
    )

    raw = response.choices[0].message.content
    # Strip markdown fences if emitted by model
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


def generate_healthy_report_static(no_gigi: str, lokasi: str) -> dict:
    """
    Generate deterministic clinical report for healthy teeth without requiring an LLM call.

    Args:
        no_gigi: FDI tooth number string.
        lokasi: Tooth surface inspected.

    Returns:
        Dictionary containing static 'laporan_dokter' and 'laporan_pasien' objects.
    """
    tooth_name  = get_tooth_name(no_gigi)
    lokasi_norm = lokasi.lower()

    # Oral hygiene recommendations tailored to tooth surface
    if lokasi_norm in ("mesial", "distal"):
        hygiene_tip = (
            "Penggunaan benang gigi (dental floss) setiap hari sangat "
            "dianjurkan untuk membersihkan sela-sela gigi yang tidak "
            "terjangkau sikat gigi biasa."
        )
    elif lokasi_norm == "bukal":
        hygiene_tip = (
            "Pastikan sisi luar gigi (yang menghadap pipi) disikat dengan "
            "gerakan memutar secara menyeluruh agar kotoran tidak menumpuk."
        )
    elif lokasi_norm in ("lingual", "palatal"):
        hygiene_tip = (
            "Pastikan sisi dalam gigi (yang menghadap lidah) disikat dengan "
            "gerakan memutar secara menyeluruh agar kotoran tidak menumpuk."
        )
    elif lokasi_norm == "servikal":
        hygiene_tip = (
            "Pastikan pangkal gigi (area dekat gusi) disikat dengan lembut "
            "menggunakan gerakan memutar kecil agar kotoran di sekitar gusi "
            "terangkat tanpa melukai gusi."
        )
    else:  # oklusal / default
        hygiene_tip = (
            "Pastikan permukaan kunyah gigi belakang disikat secara "
            "menyeluruh setiap kali menyikat gigi untuk mencegah "
            "penumpukan kotoran di area tersebut."
        )

    return {
        "laporan_dokter": {
            "temuan": (
                f"Tidak ditemukan indikasi karies sekunder pada permukaan "
                f"{lokasi} gigi {no_gigi} berdasarkan analisis citra NILT."
            ),
            "interpretasi": (
                "Citra NILT tidak menunjukkan pola transmitansi yang "
                "mengindikasikan lesi karies di bawah atau di sekitar "
                "restorasi yang ada. Kondisi margin restorasi tampak "
                "intak secara radiometrik."
            ),
            "rekomendasi": (
                "Pemeriksaan klinis rutin tetap direkomendasikan pada "
                "kunjungan berkala. Pemantauan berkala dengan NILT atau "
                "radiografi bitewing dapat dilakukan sesuai protokol "
                "recall pasien."
            ),
        },
        "laporan_pasien": {
            "ringkasan": (
                f"Karies sekunder adalah kerusakan gigi baru yang muncul "
                f"di sekitar tambalan yang sudah ada. "
                f"Hasil pemeriksaan citra pada {tooth_name} (gigi {no_gigi}) "
                f"tidak menunjukkan tanda-tanda kerusakan seperti itu, "
                f"sehingga kondisi gigi Anda saat ini terlihat baik."
            ),
            "saran": (
                f"Disarankan untuk tetap melakukan pemeriksaan rutin berkala "
                f"agar kondisi tambalan dapat terus dipantau. "
                f"{hygiene_tip}"
            ),
        },
    }


# ── COMPONENT 5: PDF GENERATOR ──────────────────────
# Color palette matching clinical sample specifications
NAVY        = (21, 67, 96)      # Primary header / dark column labels
NAVY_LIGHT  = (52, 110, 145)    # Secondary column labels
GREEN_DARK  = (39, 116, 84)     # Patient report header
GREEN_LIGHT = (236, 247, 241)   # Patient report background card
GRAY_LIGHT  = (240, 240, 240)   # Clinician info box background
TEXT_DARK   = (40, 40, 40)
GRAY_TEXT   = (110, 110, 110)


def _section_row(pdf: FPDF, label: str, text: str, fill_color: tuple) -> None:
    """
    Render a two-column section row with a colored label badge on the left and body text on the right.

    Args:
        pdf: FPDF instance.
        label: Badge label string.
        text: Multiline body text.
        fill_color: RGB tuple for the label background.
    """
    label_w = 38
    page_w  = pdf.w - pdf.l_margin - pdf.r_margin
    text_w  = page_w - label_w

    start_y = pdf.get_y()

    # Pre-calculate content height without rendering (dry_run)
    pdf.set_xy(pdf.l_margin + label_w, start_y)
    pdf.set_font("Helvetica", size=10.5)
    measured_h = pdf.multi_cell(text_w, 5.5, text, dry_run=True, output="HEIGHT")
    row_h = max(measured_h, 10)

    # Draw colored badge background
    pdf.set_fill_color(*fill_color)
    pdf.rect(pdf.l_margin, start_y, label_w, row_h, style="F")

    # Write badge text (white, bold) centered vertically
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 9.5)
    n_lines = max(1, round(row_h / 6))
    pdf.set_xy(pdf.l_margin, start_y + (row_h - n_lines * 5) / 2)
    pdf.multi_cell(label_w, 5, label, align="L")

    # Write section content text
    pdf.set_xy(pdf.l_margin + label_w, start_y)
    pdf.set_text_color(*TEXT_DARK)
    pdf.set_font("Helvetica", size=10.5)
    pdf.multi_cell(text_w, 5.5, text)

    pdf.set_text_color(*TEXT_DARK)
    pdf.set_xy(pdf.l_margin, start_y + row_h)
    pdf.ln(2)


def generate_pdf(detection: dict, report: dict, query: str,
                 patient_name: str = "-", tooth: str = "-", lokasi: str = "-",
                 report_no: str = None, is_healthy: bool = False,
                 output_folder: Path = OUTPUT_FOLDER) -> str:
    """
    Typeset and generate a formatted multi-page clinical PDF report.

    Args:
        detection: YOLOv8 detection metadata dictionary.
        report: Dual-audience report dictionary ('laporan_dokter' & 'laporan_pasien').
        query: Formulated clinical query string.
        patient_name: Patient identifier.
        tooth: FDI tooth number.
        lokasi: Lesion surface.
        report_no: Unique clinical report identifier.
        is_healthy: Boolean flag indicating healthy tooth finding.
        output_folder: Destination filesystem directory.

    Returns:
        Filesystem path to the generated PDF document.
    """
    output_folder.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if is_healthy:
        pdf_path = str(output_folder / f"laporan_healthy_gigi{tooth}_{lokasi}_{timestamp}.pdf")
    else:
        pdf_path = str(output_folder / f"laporan_{timestamp}.pdf")
    if report_no is None:
        report_no = f"NILT-{datetime.now().strftime('%Y%m%d')}-{timestamp[-4:]}"

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # ── Page 1: Clinician Report Header ──
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(*NAVY)
    pdf.cell(0, 9, "LAPORAN KLINIS CDSS", ln=True)
    pdf.set_font("Helvetica", "I", 10)
    pdf.set_text_color(*GRAY_TEXT)
    pdf.cell(0, 6, "Clinical Decision Support System NILT Caries Detection", ln=True)

    # Date and Report Number in upper right corner
    pdf.set_xy(pdf.w - pdf.r_margin - 60, 12)
    pdf.set_font("Helvetica", size=9)
    pdf.set_text_color(*GRAY_TEXT)
    pdf.multi_cell(60, 5, f"No: {report_no}\nTanggal: {datetime.now().strftime('%d %B %Y')}",
                   align="R")

    pdf.set_draw_color(*NAVY)
    pdf.set_line_width(0.6)
    pdf.line(pdf.l_margin, 26, pdf.w - pdf.r_margin, 26)
    pdf.ln(8)

    # ── Detection Summary Box ──
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(*NAVY)
    pdf.cell(0, 7, "INFORMASI DETEKSI", ln=True)
    pdf.ln(1)

    col_w   = (pdf.w - pdf.l_margin - pdf.r_margin) / 2
    y_start = pdf.get_y()
    pdf.set_fill_color(*GRAY_LIGHT)
    pdf.rect(pdf.l_margin, y_start, col_w * 2, 22, style="F")

    pdf.set_xy(pdf.l_margin + 3, y_start + 2)
    pdf.set_font("Helvetica", "B", 9.5)
    pdf.set_text_color(*TEXT_DARK)

    if is_healthy:
        pdf.multi_cell(col_w - 6, 5.5,
                        f"Pasien: {patient_name}\n"
                        f"Gigi: {tooth}\n"
                        f"Kelas Deteksi: Tidak ada")
        pdf.set_xy(pdf.l_margin + col_w + 3, y_start + 2)
        pdf.multi_cell(col_w - 6, 5.5,
                        f"Confidence: -\n"
                        f"Tingkat Keparahan: -\n"
                        f"Lokasi: {lokasi}")
    else:
        pdf.multi_cell(col_w - 6, 5.5,
                        f"Pasien: {patient_name}\n"
                        f"Gigi: {tooth}\n"
                        f"Kelas Deteksi: {detection['class_name']}")
        pdf.set_xy(pdf.l_margin + col_w + 3, y_start + 2)
        pdf.multi_cell(col_w - 6, 5.5,
                        f"Confidence: {detection['confidence']*100:.0f}%\n"
                        f"Tingkat Keparahan: {detection.get('severity', '-')}\n"
                        f"Lokasi Lesi: {lokasi}")
    pdf.set_xy(pdf.l_margin, y_start + 24)
    pdf.ln(3)

    # ── Side-by-side Images: Original NILT vs. Segmentation ──
    if detection.get("vis_path"):
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(*NAVY)
        pdf.cell(0, 7, "HASIL DETEKSI YOLOv8 - INSTANCE SEGMENTATION", ln=True)
        pdf.ln(1)

        img_w  = col_w - 6
        img_y  = pdf.get_y()
        orig_path = detection.get("orig_path", detection["vis_path"])

        with Image.open(orig_path) as _im:
            img_h = img_w * _im.height / _im.width
        MAX_IMG_H = 70  # Constrain image height to fit onto a single page
        if img_h > MAX_IMG_H:
            scale = MAX_IMG_H / img_h
            img_w_actual = img_w * scale
            img_h = MAX_IMG_H
        else:
            img_w_actual = img_w

        img_offset = (col_w - img_w_actual) / 2

        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(*NAVY)
        pdf.set_xy(pdf.l_margin, img_y)
        pdf.cell(col_w, 5, "Citra Original (NILT)", align="C")
        pdf.set_xy(pdf.l_margin + col_w, img_y)
        pdf.cell(col_w, 5, "Hasil Segmentasi YOLOv8", align="C")

        pdf.image(orig_path, x=pdf.l_margin + img_offset, y=img_y + 6, w=img_w_actual)
        pdf.image(detection["vis_path"], x=pdf.l_margin + col_w + img_offset, y=img_y + 6, w=img_w_actual)

        pdf.set_y(img_y + 6 + img_h + 3)
        pdf.set_font("Helvetica", "I", 8.5)
        pdf.set_text_color(*GRAY_TEXT)
        pdf.cell(0, 6,
                 "Kiri: Citra asli NILT  |  Kanan: Hasil instance segmentation "
                 "(mask area karies ditampilkan dalam overlay biru)",
                 ln=True, align="C")
        pdf.ln(4)

    # ── Clinician Sections ──
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(*NAVY)
    pdf.cell(0, 8, "LAPORAN KLINIS - UNTUK DOKTER GIGI", ln=True)
    pdf.ln(1)

    dokter = report["laporan_dokter"]
    _section_row(pdf, "TEMUAN", dokter["temuan"], NAVY)
    _section_row(pdf, "INTERPRETASI", dokter["interpretasi"], NAVY_LIGHT)
    _section_row(pdf, "REKOMENDASI", dokter["rekomendasi"], NAVY)

    pdf.ln(2)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(*GRAY_TEXT)
    pdf.multi_cell(0, 4.5,
        "Laporan ini dihasilkan secara otomatis oleh sistem CDSS berbasis "
        "YOLOv8 + RAG + LLM. Tidak menggantikan diagnosis klinis oleh "
        "tenaga medis profesional.")

    # ── Page 2: Patient-facing Report ──
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(*NAVY)
    pdf.cell(0, 9, "LAPORAN KLINIS CDSS", ln=True)
    pdf.set_font("Helvetica", "I", 10)
    pdf.set_text_color(*GRAY_TEXT)
    pdf.cell(0, 6, "Clinical Decision Support System NILT Caries Detection", ln=True)
    pdf.set_xy(pdf.w - pdf.r_margin - 60, 12)
    pdf.set_font("Helvetica", size=9)
    pdf.multi_cell(60, 5, f"No: {report_no}\nTanggal: {datetime.now().strftime('%d %B %Y')}",
                   align="R")
    pdf.set_draw_color(*NAVY)
    pdf.line(pdf.l_margin, 26, pdf.w - pdf.r_margin, 26)
    pdf.ln(8)

    # Detection Info (Patient Version)
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(*NAVY)
    pdf.cell(0, 7, "INFORMASI DETEKSI", ln=True)
    pdf.ln(1)

    col_w   = (pdf.w - pdf.l_margin - pdf.r_margin) / 2
    y_start = pdf.get_y()
    pdf.set_fill_color(*GRAY_LIGHT)
    pdf.rect(pdf.l_margin, y_start, col_w * 2, 22, style="F")

    pdf.set_xy(pdf.l_margin + 3, y_start + 2)
    pdf.set_font("Helvetica", "B", 9.5)
    pdf.set_text_color(*TEXT_DARK)
    pdf.multi_cell(col_w - 6, 5.5,
                    f"Pasien: {patient_name}\n"
                    f"Gigi: {tooth}")

    pdf.set_xy(pdf.l_margin + col_w + 3, y_start + 2)
    if is_healthy:
        pdf.multi_cell(col_w - 6, 5.5,
                        f"Jenis: Tidak ada deteksi\n"
                        f"Lokasi: {lokasi}\n"
                        f"Tingkat Keparahan: -")
    else:
        pdf.multi_cell(col_w - 6, 5.5,
                        f"Jenis: {detection['class_name']}\n"
                        f"Lokasi: {lokasi}\n"
                        f"Tingkat Keparahan: {detection.get('severity', '-')}")
    pdf.set_xy(pdf.l_margin, y_start + 22)
    pdf.ln(3)

    # Side-by-side images on patient page
    if detection.get("vis_path"):
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(*NAVY)
        pdf.cell(0, 7, "HASIL PEMERIKSAAN", ln=True)
        pdf.ln(1)

        img_w  = col_w - 6
        img_y  = pdf.get_y()
        orig_path = detection.get("orig_path", detection["vis_path"])

        with Image.open(orig_path) as _im:
            img_h = img_w * _im.height / _im.width
        MAX_IMG_H = 70
        if img_h > MAX_IMG_H:
            scale = MAX_IMG_H / img_h
            img_w_actual = img_w * scale
            img_h = MAX_IMG_H
        else:
            img_w_actual = img_w

        img_offset = (col_w - img_w_actual) / 2

        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(*NAVY)
        pdf.set_xy(pdf.l_margin, img_y)
        pdf.cell(col_w, 5, "Citra Original (NILT)", align="C")
        pdf.set_xy(pdf.l_margin + col_w, img_y)
        pdf.cell(col_w, 5, "Hasil Segmentasi YOLOv8", align="C")

        pdf.image(orig_path, x=pdf.l_margin + img_offset, y=img_y + 6, w=img_w_actual)
        pdf.image(detection["vis_path"], x=pdf.l_margin + col_w + img_offset, y=img_y + 6, w=img_w_actual)

        pdf.set_y(img_y + 6 + img_h + 3)
        pdf.set_font("Helvetica", "I", 8.5)
        pdf.set_text_color(*GRAY_TEXT)
        pdf.cell(0, 6,
                 "Kiri: Citra asli NILT  |  Kanan: Hasil instance segmentation "
                 "(mask area karies ditampilkan dalam overlay biru)",
                 ln=True, align="C")
        pdf.ln(4)

    # Patient Narrative Cards
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(*GREEN_DARK)
    pdf.cell(0, 7, "LAPORAN UNTUK PASIEN", ln=True)
    pdf.ln(2)

    pasien = report["laporan_pasien"]
    for label, key in [("Apa yang ditemukan?", "ringkasan"),
                       ("Apa yang perlu dilakukan?", "saran")]:
        box_y = pdf.get_y()
        page_w = pdf.w - pdf.l_margin - pdf.r_margin
        pdf.set_xy(pdf.l_margin + 3, box_y + 8)
        pdf.set_font("Helvetica", size=10.5)
        text_h = pdf.multi_cell(page_w - 6, 5.5, pasien[key],
                                dry_run=True, output="HEIGHT")
        box_h = text_h + 10

        pdf.set_fill_color(*GREEN_LIGHT)
        pdf.rect(pdf.l_margin, box_y, page_w, box_h, style="F")

        pdf.set_xy(pdf.l_margin + 3, box_y + 2)
        pdf.set_font("Helvetica", "B", 10.5)
        pdf.set_text_color(*GREEN_DARK)
        pdf.cell(0, 6, label, ln=True)

        pdf.set_xy(pdf.l_margin + 3, box_y + 8)
        pdf.set_font("Helvetica", size=10.5)
        pdf.set_text_color(*TEXT_DARK)
        pdf.multi_cell(page_w - 6, 5.5, pasien[key])

        pdf.set_y(box_y + box_h + 5)

    pdf.set_text_color(*TEXT_DARK)
    pdf.output(pdf_path)
    return pdf_path


# ── MAIN PIPELINE EXECUTION ──────────────────────────
def run(image_path: str, output_folder: Path = OUTPUT_FOLDER) -> dict:
    """
    Execute the full end-to-end CDSS pipeline on an input image.

    Args:
        image_path: Filesystem path to the NILT input image.
        output_folder: Destination folder for output artifacts (PDF, JSON, overlays).

    Returns:
        Dictionary containing detection, manual metadata, query, contexts, report, and pdf_path.
    """
    output_folder.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*50}")
    print(f"Input: {image_path}")

    print("[1/5] Running YOLOv8 segmentation inference...")
    detection = run_yolo(image_path, output_folder=output_folder)

    is_healthy = (detection.get("class_name") is None)

    if is_healthy:
        print("      No caries detected — healthy tooth case.")
    else:
        print(f"      Detected: {detection['class_name']} ({detection['confidence']*100:.1f}%)")

    print("[2/5] Collecting clinician input (tooth number & lesion location)...")
    manual = get_manual_input()

    print("[3/5] Constructing clinical query and retrieving RAG context...")
    query = build_query(detection, manual)

    if is_healthy:
        # Healthy cases use deterministic template; bypass RAG retrieval and LLM call
        chunks = []
        print("      Healthy case — using deterministic clinical template.")
        print("[4/5] Generating reports (static template)...")
        report = generate_healthy_report_static(manual["no_gigi"], manual["lokasi"])
    else:
        # Cross-lingual RAG: translate query to English for biomedical retrieval
        print("      Translating query to English for PubMedBERT retrieval...")
        english_query = translate_query_to_english(query)
        print(f"      Indonesian Query: {query}")
        print(f"      English Query:    {english_query}")

        chunks = retrieve_context(english_query)
        print(f"      {len(chunks)} relevant chunks retrieved from knowledge base")
        print("[4/5] Generating clinical reports via LLM...")
        report = generate_report(query, chunks)

    print("      Reports generated successfully ✅")

    print("[5/5] Generating PDF report...")
    pdf_path = generate_pdf(detection, report, query,
                            tooth=manual["no_gigi"], lokasi=manual["lokasi"],
                            is_healthy=is_healthy, output_folder=output_folder)
    print(f"      PDF saved: {pdf_path}")

    # Persist structured execution metadata to JSON for offline evaluation
    if is_healthy:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = str(
            output_folder / f"laporan_healthy_gigi{manual['no_gigi']}_{manual['lokasi']}_{timestamp}.json"
        )
    else:
        json_path = pdf_path.replace(".pdf", ".json")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "image_path":   image_path,
            "query":        query,
            "contexts":     chunks,
            "report":       report,
            "pdf_path":     pdf_path,
            "manual_input": manual,
            "detection": {
                "class_name":  detection.get("class_name"),
                "confidence":  detection.get("confidence"),
                "severity":    detection.get("severity"),
            }
        }, f, ensure_ascii=False, indent=2)
    print(f"      JSON saved: {json_path}")

    print(f"{'='*50}\n")
    return {
        "detection": detection,
        "manual":    manual,
        "query":     query,
        "contexts":  chunks,
        "report":    report,
        "pdf_path":  pdf_path
    }


def generate_healthy_reports(output_folder: Path = OUTPUT_FOLDER) -> None:
    """
    Batch generate standard healthy control case reports and evaluation sidecars.

    Args:
        output_folder: Destination folder for output JSON sidecars.
    """
    output_folder.mkdir(parents=True, exist_ok=True)
    healthy_cases = [
        {"no_gigi": "46", "lokasi": "oklusal"},
        {"no_gigi": "26", "lokasi": "oklusal"},
        {"no_gigi": "14", "lokasi": "bukal"},
        {"no_gigi": "25", "lokasi": "oklusal"},
        {"no_gigi": "47", "lokasi": "oklusal"},
    ]

    print(f"\n{'='*50}")
    print(f"Batch generating {len(healthy_cases)} healthy tooth reports...")
    print(f"{'='*50}")

    for i, case in enumerate(healthy_cases, 1):
        no_gigi = case["no_gigi"]
        lokasi  = case["lokasi"]
        print(f"\n[{i}/{len(healthy_cases)}] Tooth {no_gigi} - {lokasi}")

        report = generate_healthy_report_static(no_gigi, lokasi)
        query  = build_query_healthy({"no_gigi": no_gigi, "lokasi": lokasi})

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = str(
            output_folder / f"laporan_healthy_gigi{no_gigi}_{lokasi}_{timestamp}.json"
        )
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "query":        query,
                "contexts":     [],
                "report":       report,
                "manual_input": case,
                "detection": {
                    "class_name": None,
                    "confidence": None,
                    "severity":   None,
                },
            }, f, ensure_ascii=False, indent=2)
        print(f"      JSON saved: {json_path}")

    print(f"\n{'='*50}")
    print(f"Done — {len(healthy_cases)} healthy reports generated.")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--healthy":
        generate_healthy_reports()
    else:
        image_path = sys.argv[1] if len(sys.argv) > 1 else "test_image.jpg"
        run(image_path)
