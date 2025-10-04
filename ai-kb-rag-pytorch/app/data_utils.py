import json, re
from pathlib import Path
from typing import List, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from .config import (
    KB_FILE, INDEX_FILE, CHUNKS_FILE, EMBEDDING_MODEL, FAISS_DIR, PAIRS_FILE
)

def _normalize_ws(text: str) -> str:
    # Keep line breaks for FAQ parsing; trim spaces per line and collapse extra blank lines
    lines = [ln.strip() for ln in text.strip().splitlines()]
    # drop runaway empty clusters while keeping structure
    cleaned = []
    last_blank = False
    for ln in lines:
        is_blank = (ln == "")
        if is_blank and last_blank:
            continue
        cleaned.append(ln)
        last_blank = is_blank
    return "\n".join(cleaned)

def load_kb() -> str:
    if not KB_FILE.exists():
        KB_FILE.write_text(
            "Sample KB.\nQ: How to request refund?\nA: Email support with order ID.\n",
            encoding="utf-8"
        )
    return KB_FILE.read_text(encoding="utf-8")

def chunk_text(text: str, chunk_size: int = 160, overlap: int = 40) -> List[str]:
    words = _normalize_ws(text).split(" ")
    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
        i += max(1, chunk_size - overlap)
    return [c for c in chunks if len(c.split()) > 20] or [text]

def build_or_load_faiss():
    FAISS_DIR.mkdir(parents=True, exist_ok=True)
    if INDEX_FILE.exists() and CHUNKS_FILE.exists():
        index = faiss.read_index(str(INDEX_FILE))
        chunks = json.loads(CHUNKS_FILE.read_text())
        return index, chunks

    text = load_kb()
    chunks = chunk_text(text)
    model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    embeddings = model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype(np.float32))
    faiss.write_index(index, str(INDEX_FILE))
    CHUNKS_FILE.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    return index, chunks

def embed_queries(queries: List[str]) -> np.ndarray:
    model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    return model.encode(queries, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)

def top_k(query: str, k: int) -> List[Tuple[int, float]]:
    index, _ = build_or_load_faiss()
    vec = embed_queries([query])
    D, I = index.search(vec, k)
    return list(zip(I[0].tolist(), D[0].tolist()))

def ensure_pairs(chunks: List[str], out_file: Path = None, negatives_per_pos: int = 2):
    """Create synthetic (query, chunk, label) pairs if file doesn't exist."""
    if out_file is None:
        out_file = PAIRS_FILE

    import random
    rng = random.Random(13)

    if Path(out_file).exists():
        return

    items = []
    for i, ch in enumerate(chunks):
        first_sent = re.split(r"(?<=[.!?])\s+", ch.strip())[0][:180]
        q = f"What does the document say about: {first_sent[:120]}?"
        items.append({"query": q, "chunk": ch, "label": 1.0})
        for _ in range(negatives_per_pos):
            j = rng.randrange(0, len(chunks))
            if j == i:
                j = (j + 1) % len(chunks)
            items.append({"query": q, "chunk": chunks[j], "label": 0.0})

    with open(out_file, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
