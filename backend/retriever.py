"""Steps 2 & 3 - Embeddings and retrieval.

Embedding = TF-IDF vectors (unigrams + bigrams over stemmed, stop-word-free
tokens). It is a *lexical* embedding: it needs no internet or API key, so the
project runs anywhere. The `Retriever` interface (fit on chunks, `search`
returns scored hits) is the only thing a neural embedding model would need to
replace - see README "Production gaps".
"""
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer

from .chunking import Chunk
from .text_utils import content_tokens


def _analyzer(text: str) -> list[str]:
    toks = content_tokens(text)
    return toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]


@dataclass
class Hit:
    chunk: Chunk
    score: float      # cosine similarity, 0..1
    coverage: float   # share of the question's keywords found in the chunk


class Retriever:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(analyzer=_analyzer, sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform([c.index_text for c in chunks])
        # title included: a split chunk loses its subject (e.g. 'inverter'), the title restores it
        self._chunk_terms = [set(content_tokens(c.index_text)) for c in chunks]

    def search(self, question: str, k: int = 3) -> list[Hit]:
        q_vec = self.vectorizer.transform([question])
        scores = (self.matrix @ q_vec.T).toarray().ravel()  # rows are L2-normalised -> cosine
        q_terms = set(content_tokens(question))
        order = scores.argsort()[::-1][:k]
        hits = []
        for i in order:
            cov = len(q_terms & self._chunk_terms[i]) / len(q_terms) if q_terms else 0.0
            hits.append(Hit(self.chunks[i], float(scores[i]), cov))
        return hits
