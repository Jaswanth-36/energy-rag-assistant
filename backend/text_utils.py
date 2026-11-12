"""Shared text helpers: tokenising, light stemming, sentence splitting.

Kept deliberately simple and dependency-free so every step can be explained.
"""
import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
_SUFFIXES = ("ations", "ation", "ings", "ing", "ions", "ion", "ed", "es", "s")
NEGATIONS = {"not", "no", "never", "without", "cannot"}
CITATION_RE = re.compile(r"\[(\d+)\]")


def raw_tokens(text: str) -> list[str]:
    """Lower-case word tokens. '75°C' -> '75c', "don't" -> 'do not'."""
    text = text.lower().replace("°", "").replace("n't", " not")
    return _TOKEN_RE.findall(text)


def stem(word: str) -> str:
    """Very light stemmer so 'checked'/'checks'/'check' all match.

    Numbers (anything containing a digit) are never altered.
    """
    if any(c.isdigit() for c in word):
        return word
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)]
            break
    if word.endswith("e") and len(word) > 4:
        word = word[:-1]
    return word


def content_tokens(text: str) -> list[str]:
    """Stemmed tokens with stop-words removed (the 'meaningful' words)."""
    return [stem(t) for t in raw_tokens(text) if t not in ENGLISH_STOP_WORDS]


def split_sentences(text: str) -> list[str]:
    """Split on sentence-ending punctuation followed by whitespace."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def strip_citations(text: str) -> str:
    return CITATION_RE.sub("", text).replace("  ", " ").strip()
