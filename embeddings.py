"""Embedding helper — calls local Ollama's /api/embeddings endpoint.

Kept separate from generation so the embedding model is easy to swap
(e.g. to bge-small via sentence-transformers) without touching retrieval
or generation code.
"""
import os
import requests

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")


def embed_text(text: str) -> list[float]:
    """Return an embedding vector for a single piece of text via Ollama."""
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["embedding"]


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Ollama's embeddings endpoint is single-input; loop for a small demo corpus."""
    return [embed_text(t) for t in texts]
