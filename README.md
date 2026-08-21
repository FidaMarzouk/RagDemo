# rag-demo — ElyaEval case study

A minimal RAG app (pgvector + Ollama embeddings + Groq generation), built to demonstrate
[ElyaEval](https://github.com/FidaMarzouk/ElyaEval) end to end: e2e HTTP evaluation, component
(span-level) evaluation, and CI via the shared Tekton recipe.

**Stack**
- Vector store: Postgres + pgvector (Docker)
- Embeddings: `nomic-embed-text` via local Ollama
- Generator: Groq API (`openai/gpt-oss-120b` by default — see `generator.py`)
- Judge (eval only): local Ollama `llama3.1`, kept separate from the generator so the same model
  never grades its own answers
- Corpus: sample docs under `sample_docs/` (add your own `.txt`/`.pdf` files — none are bundled)

## 1. Prerequisites

- Docker + Docker Compose
- Ollama running locally with two models pulled:
  ```
  ollama pull llama3.1
  ollama pull nomic-embed-text
  ```
- A free Groq API key: https://console.groq.com/keys
- `elyaeval` installed (this app's `retriever.py`/`generator.py` import `elyaeval.metrics` for
  component-level scoring):
  ```
  pip install "elyaeval @ git+https://github.com/FidaMarzouk/ElyaEval.git@v0.1.4"
  ```

## 2. Setup

```bash
cp .env.example .env
# edit .env and set GROQ_API_KEY (and DATABASE_URL/OLLAMA_BASE_URL if not using the defaults)

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

docker compose up -d          # starts Postgres with pgvector, applies schema.sql
python ingest.py              # chunks sample_docs/*.txt|*.pdf, embeds, loads into pgvector
```

## 3. Run the API

```bash
uvicorn app:app --reload
```

- `GET /health` → `{"status": "ok"}`
- `POST /query` with `{"query": "...", "top_k": 3}` → `{"query", "answer", "retrieval_context"}`

## 4. Run the evaluation

**e2e** (calls the running API over HTTP, scores the final answer):
```bash
deepeval test run test_elyaeval_rag_qa.py
```
Runs every `rag_qa` / `nightly` golden in `goldens_rag_demo.jsonl` through `/query` and scores
each with `RAG_METRICS` (answer relevancy, faithfulness, the three contextual metrics). Results:
`report/results_rag_qa.csv` (one row per golden per metric) + `report/junit.xml`.

**component** (in-process, scores retrieval and generation as separate spans — regenerate with
`elyaeval init --task-type rag_qa --eval-mode component` if you don't already have this file):
```bash
deepeval test run test_elyaeval_rag_qa_component.py
```
Tells you *which stage* regressed, rather than just whether the final answer did — useful
alongside the e2e suite, not instead of it.

## 5. CI (Tekton)

`pipelinerun.yaml` in this repo was generated with `elyaeval init-ci` and runs the e2e suite above
against a deployed instance of this API through the shared `elyaeval-rag-recipe` Pipeline. See
[ElyaEval's `recipes/tekton/README.md`]
for cluster setup and secrets, then:

```bash
$env:SUT_URL = "http://host.docker.internal:8000"
$env:JUDGE_PROVIDER = "local"
$env:JUDGE_MODEL_NAME = "llama3.1"
$env:JUDGE_BASE_URL = "http://host.docker.internal:11434/v1"
envsubst < pipelinerun.yaml | kubectl create -n elyaeval-ci -f -
```

## Notes / easy extension points

- **Swap the embedding model**: only `embeddings.py` needs to change (e.g. to
  `bge-small-en-v1.5` via `sentence-transformers`)
- **`conftest.py`** patches a known DeepEval bug in sync `evals_iterator()` runs (missing
  `trace.end_time` fallback) — remove it once that fix ships upstream; see the file's docstring.
