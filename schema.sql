CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS rag_demo_chunks (
    id SERIAL PRIMARY KEY,
    source TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding VECTOR(768) NOT NULL
);

-- ivfflat index for cosine similarity search (build after ingest for best results,
-- but fine to create up front for a small demo corpus)
CREATE INDEX IF NOT EXISTS rag_demo_chunks_embedding_idx
    ON rag_demo_chunks
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 15);
