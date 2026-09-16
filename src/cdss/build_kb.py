"""
Build knowledge base dari jurnal PDF ke ChromaDB.
Jalankan sekali saja — hasilnya tersimpan permanen.

Improvements dari versi sebelumnya:
- Sentence-aware chunking (nltk) — tidak memotong di tengah kalimat
- Chunk size lebih kecil (150 kata) — embedding lebih focused
- Biomedical embedding model (pubmedbert) — lebih relevan untuk teks dental/klinis
- Metadata per chunk (source filename, chunk index) — untuk debugging & filtering
- Wipe and recreate collection untuk clean rebuild
"""

import os
import re
import nltk
from pypdf import PdfReader
import chromadb
from sentence_transformers import SentenceTransformer

# Download nltk sentence tokenizer data jika belum ada
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# ── CONFIG ──────────────────────────────────────────
JOURNALS_FOLDER        = "./knowledge_base/"
CHROMA_PATH            = "./chroma_db/"
COLLECTION_NAME        = "karies_knowledge"
CHUNK_TARGET_WORDS     = 150   # target kata per chunk
CHUNK_OVERLAP_WORDS    = 30    # overlap antar chunk (dalam kata)
BATCH_SIZE             = 100
BIBLIOGRAPHY_THRESHOLD = 0.3
EMBEDDING_MODEL        = "NeuML/pubmedbert-base-embeddings"
# ────────────────────────────────────────────────────


# ── BIBLIOGRAPHY FILTERING ──────────────────────────
REFERENCES_HEADING_RE = re.compile(
    r"\n\s*(references|daftar\s+pustaka|bibliography)\s*\n",
    re.IGNORECASE,
)


def _looks_like_citation(line: str) -> bool:
    if re.match(r"^\s*[\[\(]?\d+[\]\)]?\.?\s+[A-Z]", line):
        if re.search(r"\d{4}", line):
            return True
    if re.match(r"^[A-Z][a-zA-Z-]+\s+[A-Z]{1,3}[,.\s]", line):
        return True
    if re.search(r"\b\d{4}\s*[;:]\s*\d", line):
        return True
    return False


def strip_references_section(text: str) -> str:
    matches = list(REFERENCES_HEADING_RE.finditer(text))
    if not matches:
        return text
    if len(matches) == 1:
        return text[: matches[0].start()]
    for match in reversed(matches):
        ref_start = match.start()
        pos = match.end()
        last_citation_end = pos
        non_citation_streak = 0
        while pos < len(text):
            line_end = text.find("\n", pos)
            if line_end == -1:
                line_end = len(text)
            line = text[pos:line_end].strip()
            pos = line_end + 1
            if not line:
                continue
            if _looks_like_citation(line):
                non_citation_streak = 0
                last_citation_end = pos
            else:
                non_citation_streak += 1
                if non_citation_streak >= 3:
                    break
        text = text[:ref_start] + text[last_citation_end:]
    return text


def is_bibliography_chunk(chunk: str, threshold: float = BIBLIOGRAPHY_THRESHOLD) -> bool:
    segments = re.split(r"(?<=\.)\s+(?=[A-Z])", chunk)
    if len(segments) < 3:
        return False
    citation_like = sum(1 for s in segments if _looks_like_citation(s))
    return (citation_like / len(segments)) >= threshold
# ────────────────────────────────────────────────────


def extract_text(pdf_path: str) -> tuple[str, int]:
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    original_len = len(text)
    text = strip_references_section(text)
    return text, original_len - len(text)


def chunk_text(text: str) -> tuple[list[str], int, int]:
    """
    Sentence-aware chunking:
    - Tokenize ke kalimat dulu pakai nltk
    - Gabungkan kalimat sampai mencapai CHUNK_TARGET_WORDS
    - Overlap: ambil kembali kalimat-kalimat terakhir dari chunk sebelumnya
      yang totalnya <= CHUNK_OVERLAP_WORDS
    """
    sentences = nltk.sent_tokenize(text)
    chunks = []
    dropped_short = 0
    dropped_biblio = 0

    i = 0
    overlap_sentences = []  # kalimat overlap dari chunk sebelumnya

    while i < len(sentences):
        current_sentences = list(overlap_sentences)  # mulai dengan overlap
        current_word_count = sum(len(s.split()) for s in current_sentences)

        # Tambahkan kalimat sampai mencapai target
        while i < len(sentences):
            sentence = sentences[i]
            sentence_words = len(sentence.split())
            if current_word_count + sentence_words > CHUNK_TARGET_WORDS and current_sentences:
                break
            current_sentences.append(sentence)
            current_word_count += sentence_words
            i += 1

        if not current_sentences:
            i += 1
            continue

        chunk = " ".join(current_sentences)

        # Filter chunk terlalu pendek
        if len(chunk.split()) <= 30:
            dropped_short += 1
            # Reset overlap
            overlap_sentences = []
            continue

        # Filter bibliography
        if is_bibliography_chunk(chunk):
            dropped_biblio += 1
            overlap_sentences = []
            continue

        chunks.append(chunk)

        # Hitung overlap: ambil kalimat-kalimat terakhir <= CHUNK_OVERLAP_WORDS
        overlap_sentences = []
        overlap_word_count = 0
        for s in reversed(current_sentences):
            s_words = len(s.split())
            if overlap_word_count + s_words > CHUNK_OVERLAP_WORDS:
                break
            overlap_sentences.insert(0, s)
            overlap_word_count += s_words

    return chunks, dropped_short, dropped_biblio


def deduplicate_chunks(chunks: list[str], metadatas: list[dict]) -> tuple[list[str], list[dict]]:
    seen = set()
    unique_chunks = []
    unique_metadatas = []
    for chunk, meta in zip(chunks, metadatas):
        normalized = " ".join(chunk.split())
        if normalized not in seen:
            seen.add(normalized)
            unique_chunks.append(chunk)
            unique_metadatas.append(meta)
    return unique_chunks, unique_metadatas


def main():
    # 1. Baca semua PDF
    all_chunks = []
    all_metadatas = []

    for filename in sorted(os.listdir(JOURNALS_FOLDER)):
        if not filename.endswith(".pdf"):
            continue
        path = os.path.join(JOURNALS_FOLDER, filename)
        text, chars_stripped = extract_text(path)
        chunks, dropped_short, dropped_biblio = chunk_text(text)

        for idx, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metadatas.append({
                "source": filename,
                "chunk_index": idx,
            })

        print(
            f"  {filename}: {len(chunks)} chunks kept "
            f"({dropped_short} dropped short, {dropped_biblio} dropped bibliography, "
            f"{chars_stripped} chars stripped by heading filter)"
        )

    print(f"\nTotal chunks before dedup: {len(all_chunks)}")
    all_chunks, all_metadatas = deduplicate_chunks(all_chunks, all_metadatas)
    print(f"Total chunks after dedup:  {len(all_chunks)}")

    # 2. Load embedding model
    print(f"\nLoading embedding model: {EMBEDDING_MODEL} ...")
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    # 3. Setup ChromaDB — wipe and recreate
    print("Setting up ChromaDB (wipe and recreate)...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"  Existing collection '{COLLECTION_NAME}' deleted.")
    except Exception:
        print(f"  No existing collection to delete.")
    collection = client.create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    # 4. Embed dan simpan per batch
    print("Embedding dan menyimpan ke ChromaDB...")
    for i in range(0, len(all_chunks), BATCH_SIZE):
        batch_docs = all_chunks[i:i + BATCH_SIZE]
        batch_meta = all_metadatas[i:i + BATCH_SIZE]
        embeddings = embedder.encode(batch_docs, show_progress_bar=False).tolist()
        collection.add(
            documents=batch_docs,
            embeddings=embeddings,
            metadatas=batch_meta,
            ids=[f"chunk_{i + j}" for j in range(len(batch_docs))],
        )
        print(f"  Progress: {min(i + BATCH_SIZE, len(all_chunks))}/{len(all_chunks)}")

    print("\nKnowledge base selesai ✅")
    print(f"Total tersimpan: {collection.count()} chunks")


if __name__ == "__main__":
    main()
