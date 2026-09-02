from deepeval.tracing import observe, update_current_trace

from retriever import retrieve
from generator import generate_answer


@observe()
def run_for_eval(query: str, top_k: int = 5, expected_output: str | None = None) -> dict:
    """Full RAG pass for a single query — call directly from eval scripts or other Python code."""
    retrieval_context = retrieve(query, top_k=top_k, expected_output=expected_output)
    answer = generate_answer(query, retrieval_context)
    update_current_trace(input=query, output=answer)
    return {
        "query": query,
        "answer": answer,
        "retrieval_context": retrieval_context,
    }