# RAG Demo — pgvector + Ollama embeddings + Groq generator

A minimal RAG system built specifically to demo a deepeval-based evaluation.

**Stack**
- Vector store: Postgres + pgvector (Docker)
- Embeddings: `nomic-embed-text` via local Ollama
- Generator: Groq API (`llama-3.3-70b-versatile` by default)
- Judge (eval only): local Ollama `llama3.1` — kept separate from the generator
  so the same model never grades its own answers
- Corpus: 3 short sample docs on renewable energy (solar, wind, storage)

## 1. Prerequisites

- Docker + Docker Compose
- Ollama running locally with two models pulled:
  ```
  ollama pull llama3.1
  ollama pull nomic-embed-text
  ```
- A free Groq API key: https://console.groq.com/keys

## 2. Setup

```bash
cp .env.example .env
# edit .env and set GROQ_API_KEY

python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt

docker compose up -d       # starts Postgres with pgvector, applies schema.sql
python ingest.py           # chunks sample_docs/*.txt, embeds, loads into pgvector
```

## 3. Run the API

```bash
uvicorn app:app --reload
```

- `GET /health` → `{"status": "ok"}`
- `POST /run_for_eval` with `{"query": "...", "top_k": 3}` →
  `{"query", "answer", "retrieval_context"}`

## 4. Run the deepeval evaluation

```bash
python eval_deepeval.py
```

This runs 4 golden questions through the RAG pipeline and scores each answer
with `AnswerRelevancyMetric` and `FaithfulnessMetric`, judged by local Ollama
`llama3.1`.

## Notes / easy extension points

- **Swap the embedding model**: only `embeddings.py` needs to change (e.g. to
  `bge-small-en-v1.5` via `sentence-transformers`) — just make sure the
  `VECTOR(768)` dimension in `schema.sql` matches the new model's output size.
- **Swap the generator**: only `generator.py` needs to change to point at a
  different API or a local Ollama model.
- **Add more metrics**: deepeval also supports `ContextualPrecisionMetric` and
  `ContextualRecallMetric` if you want to evaluate retrieval quality
  specifically, not just the generated answer.
- **Bigger corpus**: drop more `.txt` files into `sample_docs/` and re-run
  `ingest.py` (it truncates and reloads the table each time).
