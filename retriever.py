import os

import psycopg2
from dotenv import load_dotenv
from deepeval.tracing import observe, update_current_span
from deepeval.test_case import LLMTestCase

from embeddings import embed_text
from elyaeval.metrics import RETRIEVAL_METRICS

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://raguser:ragpass@localhost:5432/ragdb")


@observe(metrics=RETRIEVAL_METRICS)
def retrieve(query: str, top_k: int = 3, expected_output: str | None = None) -> list[str]:
    """Embed the query and return the top_k most similar chunks by cosine distance."""
    query_vector = embed_text(query)

    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT content
        FROM rag_demo_chunks
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """,
        (query_vector, top_k),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    chunks = [row[0] for row in rows]
    update_current_span(test_case=LLMTestCase(
        input=query, retrieval_context=chunks, expected_output=expected_output,
    ))
    return chunks