import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions using only the provided context. "
    "If the context does not contain the answer, say you don't have enough information. "
    "Keep answers concise and grounded in the context."
)


def generate_answer(query: str, context_chunks: list[str]) -> str:
    context_block = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(context_chunks))
    user_prompt = f"Context:\n{context_block}\n\nQuestion: {query}\n\nAnswer:"

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=500,
    )
    return response.choices[0].message.content.strip()
