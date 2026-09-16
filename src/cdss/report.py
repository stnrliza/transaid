"""
Pipeline lengkap CDSS:
YOLOv8 → Query Builder → RAG → LLM → PDF laporan
"""

import json
import os
from datetime import datetime
from pathlib import Path

import chromadb
from fpdf import FPDF
from openai import OpenAI
from PIL import Image
from sentence_transformers import SentenceTransformer
from ultralytics import YOLO

# ── CONFIG ──────────────────────────────────────────
YOLO_MODEL_PATH = "/home/guest/Workshop/skripsi-lija/others/skripsi/BAB_4/1.yolov8seg/Training/train_best_l2/weights/best.pt"       # ganti path
CHROMA_PATH     = "./chroma_db/"
COLLECTION_NAME = "karies_knowledge"
OUTPUT_FOLDER   = "/home/guest/Workshop/skripsi-lija/cdss/outputs/03_reports/fix"
OLLAMA_BASE_URL = "http://localhost:11434/v1"
LLM_MODEL       = "gemma3:12b"
TOP_K_CHUNKS    = 5
# ────────────────────────────────────────────────────

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Init semua komponen
yolo_model  = YOLO(YOLO_MODEL_PATH)
embedder    = SentenceTransformer("NeuML/pubmedbert-base-embeddings")
chroma      = chromadb.PersistentClient(path=CHROMA_PATH)
collection  = chroma.get_collection(COLLECTION_NAME)
llm_client  = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")


def severity_label(confidence: float) -> str:
    """Klasifikasikan tingkat keparahan berdasarkan confidence deteksi."""
    if confidence >= 0.75:
        return "Tinggi (High)"
    elif confidence >= 0.5:
        return "Sedang (Moderate)"
    return "Rendah (Low)"


# Lookup nama gigi dalam Bahasa Indonesia berdasarkan notasi FDI
# Digunakan HANYA di laporan_pasien agar pasien memahami lokasi giginya
FDI_TOOTH_NAME: dict[str, str] = {
    # Kuadran 1 — Rahang atas kanan
    "11": "gigi seri pertama atas kanan",
    "12": "gigi seri kedua atas kanan",
    "13": "gigi taring atas kanan",
    "14": "gigi premolar pertama atas kanan",
    "15": "gigi premolar kedua atas kanan",
    "16": "gigi geraham pertama atas kanan",
    "17": "gigi geraham kedua atas kanan",
    "18": "gigi geraham bungsu atas kanan",
    # Kuadran 2 — Rahang atas kiri
    "21": "gigi seri pertama atas kiri",
    "22": "gigi seri kedua atas kiri",
    "23": "gigi taring atas kiri",
    "24": "gigi premolar pertama atas kiri",
    "25": "gigi premolar kedua atas kiri",
    "26": "gigi geraham pertama atas kiri",
    "27": "gigi geraham kedua atas kiri",
    "28": "gigi geraham bungsu atas kiri",
    # Kuadran 3 — Rahang bawah kiri
    "31": "gigi seri pertama bawah kiri",
    "32": "gigi seri kedua bawah kiri",
    "33": "gigi taring bawah kiri",
    "34": "gigi premolar pertama bawah kiri",
    "35": "gigi premolar kedua bawah kiri",
    "36": "gigi geraham pertama bawah kiri",
    "37": "gigi geraham kedua bawah kiri",
    "38": "gigi geraham bungsu bawah kiri",
    # Kuadran 4 — Rahang bawah kanan
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
    """Kembalikan nama gigi dalam Bahasa Indonesia. Fallback ke nomor FDI jika tidak dikenal."""
    return FDI_TOOTH_NAME.get(str(no_gigi).strip(), f"gigi {no_gigi}")


# ── KOMPONEN 1: YOLOv8 ──────────────────────────────
def run_yolo(image_path: str) -> dict:
    """Jalankan inference YOLOv8 dan simpan hasil visualisasi."""
    results = yolo_model(image_path)
    result  = results[0]

    # Simpan salinan citra original untuk perbandingan side-by-side
    orig_path = os.path.join(OUTPUT_FOLDER, "yolo_original.jpg")
    Image.open(image_path).convert("RGB").save(orig_path)

    # Simpan gambar dengan mask segmentasi
    vis_path = os.path.join(OUTPUT_FOLDER, "yolo_result.jpg")
    result.save(filename=vis_path)

    # Ambil deteksi pertama (asumsi 1 lesi per foto)
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
    class_name = yolo_model.names[class_id]

    return {
        "class_name":  class_name,
        "confidence":  confidence,
        "vis_path":    vis_path,
        "orig_path":   orig_path,
        "severity":    severity_label(confidence),
    }


# ── KOMPONEN 2: Query Builder ────────────────────────
def get_manual_input() -> dict:
    """Input manual dokter via terminal (nomor gigi + lokasi lesi)."""
    no_gigi = input("Nomor gigi (notasi FDI, 11-48): ").strip()
    lokasi  = input("Lokasi lesi (mesial/distal/oklusal/servikal/bukal/lingual): ").strip().lower()
    return {"no_gigi": no_gigi, "lokasi": lokasi}


def build_query(detection: dict, manual: dict) -> str:
    """Konversi output YOLO + input manual dokter ke kalimat klinis untuk RAG query."""
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
    """Query string untuk kasus gigi sehat (tidak ada deteksi karies)."""
    tooth_name = get_tooth_name(manual["no_gigi"])
    return (
        f"Tidak terdeteksi karies sekunder pada {tooth_name} (gigi {manual['no_gigi']}) "
        f"di lokasi {manual['lokasi']}. Hasil segmentasi instance dari "
        f"citra NILT tidak menunjukkan adanya lesi."
    )


def translate_query_to_english(query: str) -> str:
    """Translate Indonesian CDSS query to English for optimal biomedical retrieval."""
    try:
        response = llm_client.chat.completions.create(
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
        print(f"      [Warning] Translation failed: {e}. Using original query.")
        return query


# ── KOMPONEN 3: RAG Retrieval ────────────────────────
def retrieve_context(query: str) -> list[str]:
    """Ambil Top-K chunks paling relevan dari ChromaDB."""
    query_embedding = embedder.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=TOP_K_CHUNKS
    )
    return results["documents"][0]


# ── KOMPONEN 4: LLM Generation ──────────────────────
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
    """Generate laporan dokter + pasien via LLM dengan konteks RAG."""
    context = "\n\n---\n\n".join(context_chunks)

    response = llm_client.chat.completions.create(
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
    # Bersihkan kalau ada markdown code block
    raw = raw.replace("", "").strip()
    return json.loads(raw)


def generate_healthy_report_static(no_gigi: str, lokasi: str) -> dict:
    """Generate laporan statis untuk kasus gigi sehat (tanpa LLM call)."""
    tooth_name  = get_tooth_name(no_gigi)
    lokasi_norm = lokasi.lower()

    # Saran kebersihan spesifik berdasarkan lokasi lesi
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


# ── KOMPONEN 5: PDF Generator ────────────────────────
# Palet warna mengikuti referensi laporan_cdss_sample.pdf
NAVY        = (21, 67, 96)      # header / label kolom kiri (gelap)
NAVY_LIGHT  = (52, 110, 145)    # label kolom kanan (lebih terang)
GREEN_DARK  = (39, 116, 84)     # header "Laporan Pasien"
GREEN_LIGHT = (236, 247, 241)   # background box pasien
GRAY_LIGHT  = (240, 240, 240)   # background info box dokter
TEXT_DARK   = (40, 40, 40)
GRAY_TEXT   = (110, 110, 110)


def _section_row(pdf, label, text, fill_color):
    """Satu baris label (kiri, berwarna) + isi (kanan, multi-baris)."""
    label_w = 38
    page_w  = pdf.w - pdf.l_margin - pdf.r_margin
    text_w  = page_w - label_w

    start_y = pdf.get_y()

    # Hitung tinggi konten dulu TANPA menggambar (dry_run, fpdf2 >= 2.7)
    pdf.set_xy(pdf.l_margin + label_w, start_y)
    pdf.set_font("Helvetica", size=10.5)
    measured_h = pdf.multi_cell(text_w, 5.5, text, dry_run=True, output="HEIGHT")
    row_h = max(measured_h, 10)

    # Gambar background label
    pdf.set_fill_color(*fill_color)
    pdf.rect(pdf.l_margin, start_y, label_w, row_h, style="F")

    # Tulis label (putih, bold), dicentang vertikal sederhana
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 9.5)
    n_lines = max(1, round(row_h / 6))
    pdf.set_xy(pdf.l_margin, start_y + (row_h - n_lines * 5) / 2)
    pdf.multi_cell(label_w, 5, label, align="L")

    # Tulis isi teks (satu kali saja)
    pdf.set_xy(pdf.l_margin + label_w, start_y)
    pdf.set_text_color(*TEXT_DARK)
    pdf.set_font("Helvetica", size=10.5)
    pdf.multi_cell(text_w, 5.5, text)

    pdf.set_text_color(*TEXT_DARK)
    pdf.set_xy(pdf.l_margin, start_y + row_h)
    pdf.ln(2)


def generate_pdf(detection: dict, report: dict, query: str,
                  patient_name: str = "-", tooth: str = "-", lokasi: str = "-",
                  report_no: str = None, is_healthy: bool = False) -> str:
    """Generate PDF laporan rapi (mengikuti template laporan_cdss_sample.pdf)."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if is_healthy:
        pdf_path = os.path.join(OUTPUT_FOLDER,
                                f"laporan_healthy_gigi{tooth}_{lokasi}_{timestamp}.pdf")
    else:
        pdf_path = os.path.join(OUTPUT_FOLDER, f"laporan_{timestamp}.pdf")
    if report_no is None:
        report_no = f"NILT-{datetime.now().strftime('%Y%m%d')}-{timestamp[-4:]}"

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # ── Header ──
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(*NAVY)
    pdf.cell(0, 9, "LAPORAN KLINIS CDSS", ln=True)
    pdf.set_font("Helvetica", "I", 10)
    pdf.set_text_color(*GRAY_TEXT)
    pdf.cell(0, 6, "Clinical Decision Support System NILT Caries Detection", ln=True)

    # No & tanggal di kanan atas (ditulis di posisi sebelumnya)
    pdf.set_xy(pdf.w - pdf.r_margin - 60, 12)
    pdf.set_font("Helvetica", size=9)
    pdf.set_text_color(*GRAY_TEXT)
    pdf.multi_cell(60, 5, f"No: {report_no}\nTanggal: {datetime.now().strftime('%d %B %Y')}",
                   align="R")

    pdf.set_draw_color(*NAVY)
    pdf.set_line_width(0.6)
    pdf.line(pdf.l_margin, 26, pdf.w - pdf.r_margin, 26)
    pdf.ln(8)

    # ── Info Deteksi ──
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

    # ── Gambar: original vs hasil segmentasi side-by-side ──
    if detection.get("vis_path"):
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(*NAVY)
        pdf.cell(0, 7, "HASIL DETEKSI YOLOv8 - INSTANCE SEGMENTATION", ln=True)
        pdf.ln(1)

        img_w  = col_w - 6
        img_y  = pdf.get_y()
        orig_path = detection.get("orig_path", detection["vis_path"])

        # Hitung tinggi gambar, batasi agar laporan dokter muat 1 halaman
        with Image.open(orig_path) as _im:
            img_h = img_w * _im.height / _im.width
        MAX_IMG_H = 70  # mm — batas agar konten + gambar muat 1 halaman
        if img_h > MAX_IMG_H:
            # Scale down: gunakan h sebagai constraint, hitung w proporsional
            scale = MAX_IMG_H / img_h
            img_w_actual = img_w * scale
            img_h = MAX_IMG_H
        else:
            img_w_actual = img_w

        # Offset untuk centering gambar di dalam kolom masing-masing
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

    # ── Laporan Dokter ──
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

    # ── Laporan Pasien (halaman baru) ──
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

    # ── Info Deteksi (ringkas, versi pasien) ──
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

    # ── Gambar: original vs hasil segmentasi side-by-side (halaman pasien) ──
    if detection.get("vis_path"):
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(*NAVY)
        pdf.cell(0, 7, "HASIL PEMERIKSAAN", ln=True)
        pdf.ln(1)

        img_w  = col_w - 6
        img_y  = pdf.get_y()
        orig_path = detection.get("orig_path", detection["vis_path"])

        # Hitung tinggi gambar, batasi agar laporan pasien muat 1 halaman
        with Image.open(orig_path) as _im:
            img_h = img_w * _im.height / _im.width
        MAX_IMG_H = 70  # mm — batas agar konten + gambar muat 1 halaman
        if img_h > MAX_IMG_H:
            scale = MAX_IMG_H / img_h
            img_w_actual = img_w * scale
            img_h = MAX_IMG_H
        else:
            img_w_actual = img_w

        # Offset untuk centering gambar di dalam kolom masing-masing
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


# ── MAIN ─────────────────────────────────────────────
def run(image_path: str):
    print(f"\n{'='*50}")
    print(f"Input: {image_path}")

    print("[1/5] Menjalankan YOLOv8...")
    detection = run_yolo(image_path)

    is_healthy = (detection.get("class_name") is None)

    if is_healthy:
        print("      Tidak ada deteksi karies — kasus gigi sehat.")
    else:
        print(f"      Deteksi: {detection['class_name']} ({detection['confidence']*100:.1f}%)")

    print("[2/5] Input manual dokter (nomor gigi & lokasi lesi)...")
    manual = get_manual_input()

    print("[3/5] Membangun query dan retrieval RAG...")
    query = build_query(detection, manual)

    if is_healthy:
        # Kasus gigi sehat: gunakan template statis, skip RAG & LLM
        chunks = []
        print("      Kasus sehat — menggunakan template statis.")
        print("[4/5] Generating laporan (template statis)...")
        report = generate_healthy_report_static(manual["no_gigi"], manual["lokasi"])
        # detection sudah memiliki vis_path dan orig_path dari run_yolo
    else:
        # Translate query to English for optimal biomedical retrieval (cross-lingual RAG)
        print("      Translating query to English for RAG retrieval...")
        english_query = translate_query_to_english(query)
        print(f"      Indonesian Query: {query}")
        print(f"      English Query:    {english_query}")
        
        chunks = retrieve_context(english_query)
        print(f"      {len(chunks)} chunks ditemukan dari knowledge base")
        print("[4/5] Generating laporan via LLM...")
        report = generate_report(query, chunks)

    print("      Laporan berhasil digenerate ✅")

    print("[5/5] Generating PDF...")
    pdf_path = generate_pdf(detection, report, query,
                             tooth=manual["no_gigi"], lokasi=manual["lokasi"],
                             is_healthy=is_healthy)
    print(f"      PDF tersimpan: {pdf_path}")

    # Simpan hasil pipeline ke JSON untuk evaluasi (04_evaluate.py)
    if is_healthy:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = os.path.join(
            OUTPUT_FOLDER,
            f"laporan_healthy_gigi{manual['no_gigi']}_{manual['lokasi']}_{timestamp}.json"
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
    print(f"      JSON tersimpan: {json_path}")

    print(f"{'='*50}\n")
    return {
        "detection": detection,
        "manual":    manual,
        "query":     query,
        "contexts":  chunks,
        "report":    report,
        "pdf_path":  pdf_path
    }


def generate_healthy_reports():
    """Generate 5 laporan gigi sehat sesuai tabel di report_revision_guide.md (Part 3)."""
    healthy_cases = [
        {"no_gigi": "46", "lokasi": "oklusal"},
        {"no_gigi": "26", "lokasi": "oklusal"},
        {"no_gigi": "14", "lokasi": "bukal"},
        {"no_gigi": "25", "lokasi": "oklusal"},
        {"no_gigi": "47", "lokasi": "oklusal"},
    ]

    print(f"\n{'='*50}")
    print(f"Generating {len(healthy_cases)} laporan gigi sehat...")
    print(f"{'='*50}")

    for i, case in enumerate(healthy_cases, 1):
        no_gigi = case["no_gigi"]
        lokasi  = case["lokasi"]
        print(f"\n[{i}/{len(healthy_cases)}] Gigi {no_gigi} - {lokasi}")

        report = generate_healthy_report_static(no_gigi, lokasi)
        query  = build_query_healthy({"no_gigi": no_gigi, "lokasi": lokasi})

        # Detection dict minimal untuk batch generation, bisa saja vis_path = None
        # karena tidak ada citra asli untuk batch command ini
        detection = {
            "class_name": None,
            "confidence": None,
            "severity":   None,
            "vis_path":   None,
            "orig_path":  None,
        }

        # Simpan JSON
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = os.path.join(
            OUTPUT_FOLDER,
            f"laporan_healthy_gigi{no_gigi}_{lokasi}_{timestamp}.json"
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
        print(f"      JSON tersimpan: {json_path}")

    print(f"\n{'='*50}")
    print(f"Selesai — {len(healthy_cases)} laporan gigi sehat berhasil digenerate.")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--healthy":
        generate_healthy_reports()
    else:
        image_path = sys.argv[1] if len(sys.argv) > 1 else "test_image.jpg"
        run(image_path)
