"""Steps 2 & 3 - Embeddings and retrieval.

Two embedding backends, chosen with the EMBEDDING_BACKEND environment variable:

  tfidf  (default) sparse lexical vectors. Offline, instant, explainable.
  hybrid           TF-IDF + a neural (dense) sentence-embedding model run locally
                   through `fastembed` (ONNX, no PyTorch, no API key). The two scores
                   are fused, so a question can match a passage by meaning even when
                   few words overlap ("running hot" -> "temperature exceeds").

If the dense model cannot be loaded (package missing, no download, low memory) the
app falls back to TF-IDF automatically and says so in /api/health and in each answer.
"""
import os
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .chunking import Chunk
from .text_utils import content_tokens

DEFAULT_DENSE_MODEL = "BAAI/bge-small-en-v1.5"


def _analyzer(text: str) -> list[str]:
    toks = content_tokens(text)
    return toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]


@dataclass
class Hit:
    chunk: Chunk
    score: float                 # score used for ranking and the refusal gate
    coverage: float              # share of the question's keywords found in the chunk
    lexical: float = 0.0         # TF-IDF cosine
    dense: float | None = None   # raw neural-embedding cosine (hybrid mode only)


class DenseScorer:
    """Neural sentence embeddings through fastembed. Raises if unavailable."""

    def __init__(self, texts: list[str], model_name: str = DEFAULT_DENSE_MODEL):
        from fastembed import TextEmbedding  # optional dependency

        self.model_name = model_name
        self._model = TextEmbedding(model_name=model_name)
        self._matrix = self._unit(np.array(list(self._model.embed(texts)), dtype=float))

    @staticmethod
    def _unit(m: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(m, axis=-1, keepdims=True)
        return m / np.where(norm == 0, 1, norm)

    def scores(self, question: str) -> np.ndarray:
        encode = getattr(self._model, "query_embed", self._model.embed)
        q = self._unit(np.array(list(encode([question])), dtype=float))[0]
        return self._matrix @ q


class Retriever:
    def __init__(self, chunks: list[Chunk], dense=None, alpha: float = 0.5, dense_floor: float = 0.5):
        self.chunks = chunks
        self.dense = dense
        self.alpha = alpha              # weight of the dense score in the fused score
        self.dense_floor = dense_floor  # cosine below this counts as 'unrelated' (maps to 0)
        self.backend = "tfidf"
        self.note = ""
        self.vectorizer = TfidfVectorizer(analyzer=_analyzer, sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform([c.index_text for c in chunks])
        # title included: a split chunk loses its subject (e.g. 'inverter'), the title restores it
        self._chunk_terms = [set(content_tokens(c.index_text)) for c in chunks]

    def search(self, question: str, k: int = 3) -> list[Hit]:
        q_vec = self.vectorizer.transform([question])
        lexical = (self.matrix @ q_vec.T).toarray().ravel()  # rows are L2-normalised -> cosine
        if self.dense is not None:
            raw = np.asarray(self.dense.scores(question), dtype=float)
            adjusted = np.clip((raw - self.dense_floor) / (1 - self.dense_floor), 0.0, 1.0)
            fused = (1 - self.alpha) * lexical + self.alpha * adjusted
        else:
            raw, fused = None, lexical
        q_terms = set(content_tokens(question))
        order = fused.argsort()[::-1][:k]
        hits = []
        for i in order:
            cov = len(q_terms & self._chunk_terms[i]) / len(q_terms) if q_terms else 0.0
            hits.append(Hit(self.chunks[i], float(fused[i]), cov, float(lexical[i]),
                            None if raw is None else float(raw[i])))
        return hits


def build_retriever(chunks: list[Chunk]) -> Retriever:
    """Create the retriever for the configured backend, falling back to TF-IDF on any failure."""
    if os.getenv("EMBEDDING_BACKEND", "tfidf").lower() == "hybrid":
        model = os.getenv("EMBEDDING_MODEL", DEFAULT_DENSE_MODEL)
        try:
            dense = DenseScorer([c.index_text for c in chunks], model)
            r = Retriever(chunks, dense=dense,
                          alpha=float(os.getenv("DENSE_WEIGHT", "0.5")),
                          dense_floor=float(os.getenv("DENSE_FLOOR", "0.5")))
            r.backend = f"hybrid (TF-IDF + {model})"
            return r
        except Exception as exc:  # missing package, no network for the model download, low memory...
            r = Retriever(chunks)
            r.note = f"Neural embeddings unavailable ({type(exc).__name__}); using TF-IDF only."
            return r
    return Retriever(chunks)
