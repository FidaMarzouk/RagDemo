"""Chunk the sample docs, embed each chunk with Ollama, and load into pgvector.

Run with: python ingest.py
Safe to re-run — it truncates the table first so ingestion is idempotent.
"""
import os
import glob

import psycopg2
from dotenv import load_dotenv
from pypdf import PdfReader

from embeddings import embed_text

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://raguser:ragpass@localhost:5432/ragdb")
DOCS_DIR = os.path.join(os.path.dirname(__file__), "sample_docs")

CHUNK_SIZE = 500  # characters
CHUNK_OVERLAP = 50


def extract_text(path: str) -> str:
    """Return raw text for a .txt or .pdf file, dispatching on extension."""
    if path.lower().endswith(".pdf"):
        reader = PdfReader(path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Simple fixed-size character chunking with overlap. Good enough for a demo corpus."""
    text = " ".join(text.split())  # normalize whitespace
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return [c for c in chunks if c.strip()]


def main():
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()

    cur.execute("TRUNCATE TABLE rag_demo_chunks RESTART IDENTITY;")

    doc_paths = sorted(
        glob.glob(os.path.join(DOCS_DIR, "*.txt")) + glob.glob(os.path.join(DOCS_DIR, "*.pdf"))
    )
    if not doc_paths:
        raise SystemExit(f"No .txt or .pdf files found in {DOCS_DIR}")

    total_chunks = 0
    for path in doc_paths:
        source_name = os.path.basename(path)
        raw_text = extract_text(path)
        if not raw_text.strip():
            print(f"WARNING: no extractable text in {source_name} (skipped)")
            continue

        chunks = chunk_text(raw_text)
        print(f"{source_name}: {len(chunks)} chunks to embed...")
        for i, chunk in enumerate(chunks, 1):
            vector = embed_text(chunk)
            cur.execute(
                "INSERT INTO rag_demo_chunks (source, content, embedding) VALUES (%s, %s, %s)",
                (source_name, chunk, vector),
            )
            total_chunks += 1
            if i % 5 == 0 or i == len(chunks):
                print(f"  {source_name}: {i}/{len(chunks)} chunks embedded", flush=True)

        print(f"Ingested {source_name}")

    conn.commit()
    cur.close()
    conn.close()
    print(f"Done. Ingested {total_chunks} chunks from {len(doc_paths)} documents.")


if __name__ == "__main__":
    main()