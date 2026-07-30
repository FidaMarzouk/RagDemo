import os

import psycopg2
from dotenv import load_dotenv

from embeddings import embed_text

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://raguser:ragpass@localhost:5432/ragdb")


def retrieve(query: str, top_k: int = 3) -> list[str]:
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

    return [row[0] for row in rows]
