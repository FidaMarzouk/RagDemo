from fastapi import FastAPI
from rag_core import run_for_eval

app = FastAPI()

@app.post("/query")
def query(payload: QueryRequest):
    return run_for_eval(payload.query, top_k=payload.top_k)