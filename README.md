# TransAID: AI-Powered Clinical Decision Support System (CDSS) for Secondary Caries Detection

An end-to-end medical AI pipeline that couples **Computer Vision (YOLOv8/11 Instance Segmentation)** on **Near-Infrared Light Transillumination (NILT)** dental imagery with **Cross-Lingual Retrieval-Augmented Generation (RAG)** and **Clinical Large Language Models (LLMs)**.

The system analyzes transilluminated tooth images, segments caries lesions around existing restorations, retrieves evidence-based clinical literature from peer-reviewed dental journals, and synthesizes structured, publication-grade **dual-audience diagnostic reports** (clinician-facing and patient-facing).

---

## Key Features

- **Computer Vision (YOLOv8/11 Instance Segmentation)**: Detects and segments recurrent and residual caries margins on 980 nm NILT dental images.
- **Cross-Lingual RAG Pipeline**: Converts Indonesian clinical queries to domain-optimized English search concepts to query a PubMedBERT-embedded vector database of dental literature.
- **Biomedical Vector Store (ChromaDB)**: Curated knowledge base built from peer-reviewed dental publications (ICDAS, ICCMS, selective caries removal protocols).
- **Dual-Audience Report Generation**:
  - **Clinician-Facing**: Precise FDI tooth notation, radiometric interpretation, margin integrity assessment, and severity-tailored intervention pathways (active monitoring vs. bitewing confirmation vs. re-restoration / endodontics).
  - **Patient-Facing**: Accessible lay explanations in Indonesian, reassuring tone, and targeted daily brushing/flossing recommendations tailored to the exact tooth surface (occlusal, buccal, lingual, interproximal).
- **Automated PDF Typesetting**: Produces professional two-page clinical reports with side-by-side original and segmented lesion overlays.
- **Rigorous Offline Evaluation**: Built-in reference-free evaluation suites using **RAGAS** (Faithfulness, Relevancy, Precision, Recall) and **G-Eval LLM-as-a-Judge** (Coherence, Completeness, Relevance, Simplicity, Clarity), indexed with crash-safe CSV skip-existing logic.
- **Embedded Hardware Benchmarking**: Benchmark scripts comparing ONNX Runtime against TensorRT FP16 on NVIDIA Jetson edge devices.

---

## System Architecture

```mermaid
flowchart TD
    A[NILT 980nm Dental Image] --> B[YOLOv8/11 Segmentation]
    B --> C[Lesion Mask, Confidence & Severity]
    D[Practitioner Input: Tooth Number & Surface] --> E[Clinical Query Builder]
    C --> E
    
    subgraph Cross-Lingual RAG Pipeline
        E -->|Indonesian Query| F[Query Concept Translation]
        F -->|English Biomedical Concepts| G[PubMedBERT Embedder]
        G --> H[(ChromaDB Vector Store\nDental Literature)]
        H -->|Top-K Clinical Chunks| I[Context Assembler]
    end
    
    subgraph LLM Generation & Typesetting
        I --> J[Clinical LLM\nGemma-3 / Qwen]
        E --> J
        J --> K[Dual-Audience JSON Report]
        K --> L[FPDF2 PDF Generator]
        B --> L
    end
    
    L --> M[Clinician Report: Page 1]
    L --> N[Patient Report: Page 2]
    
    subgraph Evaluation Framework
        K --> O[RAGAS Evaluation]
        K --> P[G-Eval LLM-as-a-Judge]
    end
```

---

## Project Structure

```
transaid/
├── .env                      # Local environment configuration (copied from .env.example)
├── .env.example              # Environment template
├── README.md                 # Project documentation
├── main.py                   # Single entry point (generation + evaluation)
├── pimnas/                   # Archived legacy GUI + SQLite version
└── src/
    ├── cdss/                 # Clinical Decision Support System & RAG
    │   ├── build_kb.py       # PDF literature extractor, chunker & ChromaDB builder
    │   ├── report.py         # RAG retrieval, clinical LLM synthesis & PDF typesetting library
    │   ├── geval.py          # G-Eval reference-free LLM-as-a-judge evaluation suite
    │   ├── ragas.py          # RAGAS-inspired reference-free evaluation suite
    │   └── knowledge_base/   # Peer-reviewed dental journal literature (PDFs)
    ├── yolo/                 # Computer Vision & Instance Segmentation
    │   ├── train.py          # YOLO instance segmentation training script
    │   └── yolo_inference.py # YOLO model loader, inference & severity classifier
    ├── deployment/           # Embedded Edge Deployment & Benchmarking
    │   └── jetson_inference.py # Jetson Nano/Xavier/Orin ONNX vs. TensorRT benchmark
    └── outputs/              # Generated outputs (resolved via OUTPUT_DIR)
        ├── healthy/          # Sessions where no caries was detected
        │   ├── <uid>.pdf          # Typeset clinical report
        │   ├── <uid>.png          # YOLO segmentation overlay
        │   ├── <uid>.json         # Full structured case data (evaluation input)
        │   ├── <uid>_geval.json   # Per-session G-Eval scores & reasoning
        │   └── <uid>_ragas.json   # Per-session RAGAS scores & reasoning
        ├── caries/           # Sessions where caries lesion was detected
        │   ├── <uid>.pdf
        │   ├── <uid>.png
        │   ├── <uid>.json
        │   ├── <uid>_geval.json
        │   └── <uid>_ragas.json
        ├── geval.csv         # Aggregate G-Eval scores across all sessions
        └── ragas.csv         # Aggregate RAGAS scores across all sessions
```

---

## Requirements & Environment Setup

### 1. Prerequisites
- Python 3.10 or higher
- NVIDIA GPU with CUDA support (recommended for YOLO inference/training & local LLM execution)
- [Ollama](https://ollama.ai/) installed and running locally

### 2. Install Python Dependencies

```bash
pip install ultralytics torch torchvision torchaudio \
    chromadb sentence-transformers nltk pypdf fpdf2 \
    openai pillow requests numpy python-dotenv
```

### 3. Pull Ollama LLM Models

The pipeline uses `gemma3:12b` for clinical report generation and `qwen3:14b` for reference-free LLM judging:

```bash
ollama pull gemma3:12b
ollama pull qwen3:14b
```

### 4. Configure Environment Variables

Copy `.env.example` in the project root to `.env`:

```bash
cp .env.example .env
```

Environment variables are loaded automatically via `python-dotenv`:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `YOLO_MODEL_PATH` | `src/yolo/weights/best.pt` | Path to trained YOLO instance segmentation weights |
| `DATA_YAML` | `src/yolo/data.yaml` | Dataset YAML configuration for training |
| `CHROMA_PATH` | `src/cdss/chroma_db` | Storage path for ChromaDB vector database |
| `JOURNALS_FOLDER` | `src/cdss/knowledge_base` | Directory containing dental journal PDFs |
| `COLLECTION_NAME` | `karies_knowledge` | ChromaDB collection name |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama HTTP endpoint |
| `LLM_MODEL` | `gemma3:12b` | Clinical generation LLM |
| `JUDGE_MODEL` | `qwen3:14b` | Evaluation LLM for G-Eval and RAGAS |
| `OUTPUT_DIR` | `src/outputs` | Root output directory (`healthy/`, `caries/`, `geval.csv`, `ragas.csv`) |

---

## Workflow & Usage Guide

### Step 1: Train YOLO Segmentation Model (Optional)

To train or fine-tune YOLO instance segmentation on NILT images:

```bash
python src/yolo/train.py
```

- Hyperparameters are tuned for transillumination dental photography: AdamW optimizer, cosine annealing schedule, and mosaic/mixup augmentations.
- Checkpoints, plots, and validation metrics are saved to `runs/exp_yolov8s_caries`.

### Step 2: Build the ChromaDB Knowledge Base

Extract, chunk, and index dental literature PDFs into the vector store:

```bash
python src/cdss/build_kb.py
```

- Strips academic bibliographies and reference sections.
- Applies sentence-aware sliding window chunking (target: 150 words, overlap: 30 words) with NLTK.
- Encodes passages into 768-dimensional biomedical vector embeddings via `NeuML/pubmedbert-base-embeddings`.

### Step 3: Run the Main CDSS Pipeline

`main.py` is the unified entry point for report generation and evaluation.

#### 1. Full Pipeline (Interactive Mode)
Runs YOLO inference -> generates report -> runs G-Eval -> runs RAGAS:

```bash
python main.py path/to/nilt_image.jpg
```

You will be prompted interactively for:
1. **Patient name**: Name of the patient (e.g. `Liza`).
2. **Tooth number**: FDI notation (e.g. `36`, `15`, `46`).
3. **Lesion surface / location**: Surface inspected (`oklusal`, `mesial`, `distal`, `bukal`, `lingual`, `servikal`).

#### 2. Full Pipeline (Scriptable / Non-Interactive Mode)
Pass `--name`, `--tooth`, and `--lokasi` to bypass interactive prompts:

```bash
python main.py path/to/nilt_image.jpg --name Liza --tooth 36 --lokasi oklusal
```

> [!NOTE]
> All three flags (`--name`, `--tooth`, `--lokasi`) must be provided together when running non-interactively.

#### 3. Report Generation Only (Skip Evaluation)
Decouple report generation from evaluation with `--skip-eval`:

```bash
python main.py path/to/nilt_image.jpg --skip-eval
python main.py path/to/nilt_image.jpg --name Liza --tooth 36 --lokasi oklusal --skip-eval
```

---

## Session UID Scheme & Output Organization

Each session receives a standardized unique identifier generated at session start:

```
<name>_<tooth>_<timestamp>
```

- **`name`**: Patient name, lowercased and slugified (non-alphanumeric characters collapsed to `_`). E.g., `Liza` -> `liza`, `Dr. John Doe` -> `dr_john_doe`.
- **`tooth`**: Two-digit FDI tooth number (e.g. `36`).
- **`timestamp`**: 14-digit timestamp in `YYYYMMDDHHMMSS` format (e.g. `20260917143210`).

Example UID: `liza_36_20260917143210`

### Automatic Output Categorization
Outputs are organized by clinical finding based directly on YOLO detection results:
- **`outputs/healthy/`**: Real patient cases where YOLO detected no lesion (`class_name is None`). Synthesizes deterministic healthy report and oral hygiene recommendations without requiring LLM generation.
- **`outputs/caries/`**: Cases where a caries lesion was detected. Retrieves literature via cross-lingual RAG and generates LLM reports.

Generated files per session:
- `<uid>.png`: YOLO instance segmentation overlay mask.
- `<uid>.pdf`: Professional two-page typeset PDF report.
- `<uid>.json`: Structured case metadata (input to evaluation suites).
- `<uid>_geval.json`: G-Eval per-criterion evaluation scores and reasoning.
- `<uid>_ragas.json`: RAGAS per-metric evaluation scores and reasoning.

---

## Evaluation Suites (G-Eval & RAGAS)

Run offline evaluations against existing reports in `outputs/healthy/` and `outputs/caries/`:

```bash
# Run both evaluation suites on all un-evaluated reports
python main.py --geval --ragas

# Run G-Eval only
python main.py --geval

# Run RAGAS only
python main.py --ragas

# Filter evaluation by patient UID prefix (case-insensitive)
python main.py --geval --patient-id liza
python main.py --ragas --patient-id liza_36
python main.py --geval --ragas --patient-id liza_36_20260917
```

### Skip-Existing & Crash Safety
- **Aggregate CSV Index**: `outputs/geval.csv` and `outputs/ragas.csv` serve as the in-memory skip index. UIDs already present in the CSV are skipped on subsequent runs.
- **Crash-Safe Write Order**: Evaluators write the `<uid>_geval.json` / `<uid>_ragas.json` sidecar first, and append rows to the CSV last. If a run is interrupted mid-way, partial evaluations are safely reprocessed.

### Metrics Evaluated

#### G-Eval LLM-as-a-Judge Suite:
- **Dentist Report**: Coherence, Completeness, Relevance, Fluency (1–5 scale).
- **Patient Report**: Simplicity, Clarity, Fluency, Conciseness (1–5 scale).

#### RAGAS Metric Suite:
- **Faithfulness**: Verifies claims against literature or established dental knowledge.
- **Answer Relevancy**: Checks coverage of tooth, surface, and severity.
- **Context Precision**: Rates the clinical usefulness of retrieved literature (skipped for healthy cases by design).
- **Context Recall**: Verifies whether essential clinical concepts were retrieved (skipped for healthy cases by design).

---

## Hardware Benchmarking on NVIDIA Jetson

To benchmark inference speed on embedded NVIDIA Jetson platforms:

```bash
python src/deployment/jetson_inference.py --mode both --image-dir path/to/images/
```

- Compares **ONNX Runtime** (CPU/CUDA) vs. **TensorRT FP16** engine.
- Records mean/median/std latency, FPS, and 95th-percentile metrics alongside tegrastats device telemetry into CSV format.

---

## Clinical Report Structure

### Page 1: Clinician Report (`Laporan Dokter`)
- **Examination Metadata**: Date, unique report number, patient ID, tooth FDI number, detected class, detection confidence, and severity level.
- **Side-by-Side Imagery**: Original NILT image alongside the YOLO instance segmentation mask overlay.
- **Temuan (Findings)**: Direct radiometrical statement of detected lesion.
- **Interpretasi (Interpretation)**: Diagnostic assessment of marginal integrity and probable dentin/pulpal involvement.
- **Rekomendasi (Recommendations)**: Concrete, severity-stratified next steps:
  - *Low (<50%)*: Active monitoring and clinical re-evaluation on next recall.
  - *Moderate (50–75%)*: Bitewing radiography confirmation to assess lesion depth and margin leakage; consider total-etch/self-etch re-restoration.
  - *High (>75%)*: Indication for restoration replacement or endodontic intervention; periapical radiograph for pulpal involvement.

### Page 2: Patient Report (`Laporan Pasien`)
- **Apa yang ditemukan? (Summary)**: Clear explanation in Indonesian avoiding technical dental jargon; introduces secondary caries simply as *"kerusakan gigi baru yang muncul di sekitar tambalan yang sudah ada"*.
- **Apa yang perlu dilakukan? (Action Plan)**: Visit urgency and home care advice customized to the tooth surface:
  - *Interproximal (mesial/distal)*: Daily dental flossing.
  - *Occlusal*: Thorough brushing of biting surfaces.
  - *Cervical*: Gentle circular brushing at the gumline.

---

## Citation & Reference Literature

The knowledge base leverages consensus guidelines and peer-reviewed dental literature, including:
- **ICCMS™**: International Caries Classification and Management System guidelines.
- **ICDAS**: International Caries Detection and Assessment System criteria manual.
- Consensus statements on management of secondary and residual caries around composite restorations.
