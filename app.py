"""
FastAPI wrapper around the RAG pipeline (rag_core.run_for_eval).

Exposes run_for_eval as an HTTP /query endpoint so evaluation harnesses
(like test_elyaeval_rag_qa.py) can call the RAG logic over HTTP without
importing rag_core's dependencies directly. This decouples the eval suite
from the SUT's internal stack (pgvector, Ollama, Groq, etc.).

To run:
    uvicorn app:app --host 0.0.0.0 --port 8000

Environment variables (all optional, fallback to sensible defaults):
  - DATABASE_URL: PostgreSQL connection string (default: postgresql://raguser:ragpass@localhost:5432/ragdb)
  - OLLAMA_BASE_URL: Ollama endpoint (default: http://localhost:11434)
  - EMBED_MODEL: embedding model name in Ollama (default: nomic-embed-text)
  - GROQ_API_KEY: Groq API key (required for generation)
  - GROQ_MODEL: Groq model to use (default: llama-3.3-70b-versatile)
"""

from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

from rag_core import run_for_eval

app = FastAPI(
    title="RAG Demo Evaluation API",
    description="HTTP endpoint for the RAG pipeline, decoupled from eval harnesses.",
)


class QueryRequest(BaseModel):
    """HTTP request body for /query endpoint."""

    query: str
    top_k: Optional[int] = 3
    expected_output: Optional[str] = None


class QueryResponse(BaseModel):
    """HTTP response body from /query endpoint — matches run_for_eval's return shape."""

    query: str
    answer: str
    retrieval_context: list[str]


@app.post("/query")
async def query(request: QueryRequest) -> QueryResponse:
    """
    Execute a single RAG query and return the answer + retrieval context.

    This endpoint wraps run_for_eval() from rag_core, making it available
    over HTTP. The eval harness (test_elyaeval_rag_qa.py) POSTs a query here
    instead of importing rag_core directly — allowing the same eval suite to
    run independently of rag_core's internal dependencies.

    Args:
        request: QueryRequest with query string, optional top_k, optional expected_output

    Returns:
        QueryResponse with answer, retrieval_context, and echoed query
    """
    result = run_for_eval(
        query=request.query,
        top_k=request.top_k,
        expected_output=request.expected_output,
    )
    return QueryResponse(**result)


@app.get("/health")
async def health():
    """Liveness probe — returns 200 if the service is up."""
    return {"status": "ok"}