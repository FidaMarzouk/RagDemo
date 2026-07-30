"""Evaluate the RAG demo with deepeval, using local Ollama llama3.1 as the judge.

The judge model is intentionally different from the generator (Groq) to avoid
a model grading its own answers.

Run with: python eval_deepeval.py
"""
import os

from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.models import OllamaModel
from deepeval.test_case import LLMTestCase

from rag_core import run_for_eval

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "llama3.1")

# A tiny golden set covering the sample corpus (solar, wind, storage).
GOLDENS = [
    {
        "query": "Why does solar power need battery storage?",
        "expected_output": (
            "Solar power is intermittent — panels only generate electricity when "
            "the sun is shining — so battery storage or backup generation is "
            "needed to provide a steady electricity supply."
        ),
    },
    {
        "query": "What is the main advantage of offshore wind turbines over onshore ones?",
        "expected_output": (
            "Offshore wind turbines generally produce more consistent output "
            "because winds over open water are stronger and steadier than onshore, "
            "though they cost more to install and maintain."
        ),
    },
    {
        "query": "What is the largest form of grid-scale energy storage today?",
        "expected_output": (
            "Pumped-storage hydropower is the largest form of grid-scale energy "
            "storage worldwide by capacity."
        ),
    },
    {
        "query": "How much has the cost of solar PV dropped since 2010?",
        "expected_output": "The cost of solar PV has dropped by more than 80% since 2010.",
    },
]


def main():
    judge = OllamaModel(model=JUDGE_MODEL, base_url=OLLAMA_BASE_URL)

    metrics = [
        AnswerRelevancyMetric(model=judge, threshold=0.5),
        FaithfulnessMetric(model=judge, threshold=0.5),
    ]

    test_cases = []
    for golden in GOLDENS:
        result = run_for_eval(golden["query"])
        test_cases.append(
            LLMTestCase(
                input=golden["query"],
                actual_output=result["answer"],
                retrieval_context=result["retrieval_context"],
                expected_output=golden["expected_output"],
            )
        )

    evaluate(test_cases, metrics)


if __name__ == "__main__":
    main()
