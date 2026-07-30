from retriever import retrieve
from generator import generate_answer


def run_for_eval(query: str, top_k: int = 3) -> dict:
    """Full RAG pass for a single query — call directly from eval scripts or other Python code."""
    retrieval_context = retrieve(query, top_k=top_k)
    answer = generate_answer(query, retrieval_context)
    return {
        "query": query,
        "answer": answer,
        "retrieval_context": retrieval_context,
    }