"""Steps 4 & 5 - Answer generation with citations.

Two modes:
  * extractive (default, offline): the answer is assembled from the retrieved
    sentences themselves, each tagged [n]. Cannot invent facts.
  * LLM (optional): set LLM_PROVIDER=anthropic and ANTHROPIC_API_KEY. The model
    gets ONLY the retrieved chunks and must cite them. Its answer is then run
    through the hallucination guard; if it fails, we fall back to extractive.
"""
import json
import os
import urllib.request

from .chunking import Chunk
from .text_utils import split_sentences

INSUFFICIENT = "INSUFFICIENT_CONTEXT"

SYSTEM_PROMPT = (
    "You are an assistant for renewable-energy operations staff. Answer ONLY from the "
    "numbered context passages. Rules: (1) end every sentence with the citation number "
    "of the passage that supports it, like [1]; (2) never add facts, numbers or steps "
    f"that are not in the passages; (3) if the passages do not answer the question, reply "
    f"exactly {INSUFFICIENT}. Keep the answer short and practical."
)


def extractive_answer(context: list[Chunk]) -> str:
    parts = []
    for i, chunk in enumerate(context, start=1):
        for sent in split_sentences(chunk.text):
            parts.append(f"{sent.rstrip('.')} [{i}].")
    return " ".join(parts)


def llm_enabled() -> bool:
    return os.getenv("LLM_PROVIDER", "").lower() == "anthropic" and bool(os.getenv("ANTHROPIC_API_KEY"))


def llm_answer(question: str, context: list[Chunk], timeout: int = 20) -> str:
    """Call the Anthropic Messages API (stdlib only). Raises on any failure."""
    passages = "\n".join(f"[{i}] ({c.source}) {c.text}" for i, c in enumerate(context, start=1))
    body = {
        "model": os.getenv("ANTHROPIC_MODEL", "claude-haiku-5-5"),
        "max_tokens": 400,
        "temperature": 0,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": f"Context:\n{passages}\n\nQuestion: {question}"}],
    }
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(body).encode(),
        headers={
            "content-type": "application/json",
            "x-api-key": os.environ["ANTHROPIC_API_KEY"],
            "anthropic-version": "2023-06-01",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.load(resp)
    return "".join(b.get("text", "") for b in data["content"] if b.get("type") == "text").strip()
