"""Step 1 - Chunking: turn markdown SOP files into small, citable pieces."""
from dataclasses import dataclass
from pathlib import Path

from .text_utils import split_sentences


@dataclass(frozen=True)
class Chunk:
    id: str          # e.g. "solar_inverter_sop.md#1"  (this is what we cite)
    source: str      # file name
    title: str       # first '# heading' of the file
    text: str        # the passage shown to the user
    index_text: str  # title + text: what gets embedded (title gives context)


def load_documents(data_dir: Path) -> dict[str, str]:
    return {p.name: p.read_text(encoding="utf-8") for p in sorted(data_dir.glob("*.md"))}


def chunk_document(name: str, raw: str, max_words: int = 20, overlap_sentences: int = 0) -> list[Chunk]:
    """Sentence-aware chunking.

    Sentences are packed together until `max_words` would be exceeded. A chunk
    never cuts a sentence in half, so every citation is a readable passage.
    `overlap_sentences` repeats the last N sentences in the next chunk, which
    helps when an answer spans a chunk boundary (0 here: the docs are tiny).
    """
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    title = name
    body: list[str] = []
    for ln in lines:
        if ln.startswith("#"):
            if title == name:
                title = ln.lstrip("# ").strip()
        else:
            body.append(ln)
    sentences = split_sentences(" ".join(body))

    groups: list[list[str]] = []
    current: list[str] = []
    words = 0
    for sent in sentences:
        n = len(sent.split())
        if current and words + n > max_words:
            groups.append(current)
            current = current[-overlap_sentences:] if overlap_sentences else []
            words = sum(len(s.split()) for s in current)
        current.append(sent)
        words += n
    if current:
        groups.append(current)

    chunks = []
    for i, group in enumerate(groups):
        text = " ".join(group)
        chunks.append(Chunk(f"{name}#{i}", name, title, text, f"{title}. {text}"))
    return chunks


def build_chunks(data_dir: Path, max_words: int = 20, overlap_sentences: int = 0) -> list[Chunk]:
    chunks: list[Chunk] = []
    for name, raw in load_documents(data_dir).items():
        chunks.extend(chunk_document(name, raw, max_words, overlap_sentences))
    return chunks
