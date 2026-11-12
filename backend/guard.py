"""Step 6 - Hallucination check (faithfulness guard).

Every sentence of an answer must pass ALL of these tests against the chunks it
cites:
  1. it carries a citation [n] and n points at a real context chunk;
  2. >= SUPPORT_THRESHOLD (80%) of its meaningful words appear in the cited chunk(s);
  3. every number in it (e.g. 75C) appears in the cited chunk(s);
  4. it does not introduce a negation ("not", "never"...) the source lacks.

This is a deterministic, explainable check. It catches invented facts, wrong
numbers and wrong citations; it cannot catch subtle meaning changes that reuse
the source's words (see README limitations).
"""
import re

from .chunking import Chunk
from .text_utils import CITATION_RE, NEGATIONS, content_tokens, raw_tokens, split_sentences

SUPPORT_THRESHOLD = 0.8


def _normalise_citation_position(answer: str) -> str:
    """Turn 'text. [1]' into 'text [1].' so a sentence and its citation stay together."""
    return re.sub(r"\.\s*((?:\[\d+\]\s*)+)", lambda m: " " + m.group(1).strip() + ".", answer)


def check_answer(answer: str, context: list[Chunk], threshold: float = SUPPORT_THRESHOLD) -> dict:
    """`context[i]` is the chunk the model sees as citation number i+1."""
    results = []
    for sent in split_sentences(_normalise_citation_position(answer)):
        cited = [int(n) for n in CITATION_RE.findall(sent)]
        valid = [n for n in cited if 1 <= n <= len(context)]
        text = re.sub(r"\s+([.,])", r"\1", CITATION_RE.sub("", sent))
        tokens = set(content_tokens(text))
        problems = []

        if not cited:
            problems.append("no citation")
        elif len(valid) != len(cited):
            problems.append("cites a source that does not exist")

        pool_chunks = [context[n - 1] for n in valid] or []
        pool_text = " ".join(c.index_text for c in pool_chunks)
        pool_tokens = set(content_tokens(pool_text))
        support = len(tokens & pool_tokens) / len(tokens) if tokens else 1.0
        if pool_chunks and support < threshold:
            missing = sorted(tokens - pool_tokens)
            problems.append(f"only {support:.0%} of its key words are in the cited source (unsupported: {', '.join(missing)})")

        numbers = {t for t in tokens if any(ch.isdigit() for ch in t)}
        bad_numbers = sorted(numbers - pool_tokens)
        if pool_chunks and bad_numbers:
            problems.append(f"number(s) not in cited source: {', '.join(bad_numbers)}")

        sent_neg = NEGATIONS & set(raw_tokens(text))
        src_neg = NEGATIONS & set(raw_tokens(pool_text))
        if pool_chunks and sent_neg - src_neg:
            problems.append("adds a negation not present in the source")

        results.append({
            "sentence": text.strip(),
            "cited": valid,
            "support": round(support, 2),
            "supported": not problems,
            "problems": problems,
        })

    grounded = bool(results) and all(r["supported"] for r in results)
    return {
        "grounded": grounded,
        "min_support": min((r["support"] for r in results), default=0.0),
        "sentences": results,
    }
