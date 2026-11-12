"""The whole RAG flow in one place: retrieve -> decide -> generate -> verify."""
import os
import time
from pathlib import Path

from . import generator, guard
from .chunking import build_chunks
from .retriever import build_retriever

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

MIN_SCORE = 0.12      # below this cosine similarity the best chunk is not relevant enough
MIN_COVERAGE = 0.5    # at least 50% of the question's keywords must appear in the best chunk
REL_CUTOFF = 0.5      # also use chunks scoring >= 50% of the best score
MAX_CONTEXT = 3

ABSTAIN_TEXT = "I could not find this in the provided operations documents, so I won't guess."


class RAGPipeline:
    def __init__(self, data_dir: Path = DATA_DIR, max_words: int = 20, retriever=None, dense_accept=None):
        self.chunks = build_chunks(data_dir, max_words=max_words)
        self.retriever = retriever or build_retriever(self.chunks)
        self._by_id = {c.id: c for c in self.chunks}
        # Neural 'accept' threshold: a question with few shared keywords is still answered when the
        # best passage is this similar by meaning. Off (None) until calibrated with the eval script.
        env = os.getenv("DENSE_ACCEPT")
        self.dense_accept = dense_accept if dense_accept is not None else (float(env) if env else None)

    def retrieve_context(self, question: str, top_k: int = 3):
        """Returns (all_hits, used_chunks, abstain_reason_or_None)."""
        hits = self.retriever.search(question, k=max(top_k, 1))
        best = hits[0]
        if best.score < MIN_SCORE:
            return hits, [], f"best similarity {best.score:.2f} is below {MIN_SCORE}"
        accepted_by_meaning = (self.dense_accept is not None and best.dense is not None
                               and best.dense >= self.dense_accept)
        if best.coverage < MIN_COVERAGE and not accepted_by_meaning:
            return hits, [], f"only {best.coverage:.0%} of the question's keywords appear in the best match"
        used = [h.chunk for h in hits if h.score >= max(MIN_SCORE, REL_CUTOFF * best.score)][:MAX_CONTEXT]
        # Neighbour expansion: the sentence that answers a question is often right next to the
        # one that matched best (e.g. 'what may indicate...' / 'recommended checks...').
        for nb in self._neighbours(best.chunk):
            if nb not in used and len(used) < MAX_CONTEXT:
                used.append(nb)
        # Present the passages in the order a reader needs them: best document first,
        # and inside a document in its own order (so step 1 comes before step 2).
        doc_rank = {}
        for i, h in enumerate(hits):
            doc_rank.setdefault(h.chunk.source, i)
        used.sort(key=lambda c: (doc_rank.get(c.source, len(hits)), int(c.id.rpartition("#")[2])))
        return hits, used, None

    def _neighbours(self, chunk):
        name, _, idx = chunk.id.partition("#")
        return [self._by_id[f"{name}#{j}"] for j in (int(idx) + 1, int(idx) - 1) if f"{name}#{j}" in self._by_id]

    def ask(self, question: str, top_k: int = 3, use_llm: bool | None = None) -> dict:
        t0 = time.perf_counter()
        question = question.strip()
        if not question:
            raise ValueError("question is empty")
        hits, used, abstain_reason = self.retrieve_context(question, top_k)
        used_ids = {c.id for c in used}
        retrieval = [
            {"chunk_id": h.chunk.id, "source": h.chunk.source, "text": h.chunk.text,
             "score": round(h.score, 3), "coverage": round(h.coverage, 2), "used": h.chunk.id in used_ids,
             "lexical": round(h.lexical, 3), "dense": None if h.dense is None else round(h.dense, 3)}
            for h in hits
        ]
        out = {"question": question, "retrieval": retrieval, "citations": [], "check": None, "notes": [],
               "embedding_backend": self.retriever.backend}
        if self.retriever.note:
            out["notes"].append(self.retriever.note)

        if abstain_reason:
            out.update(mode="abstained", answer=ABSTAIN_TEXT)
            out["notes"].append(f"Abstained: {abstain_reason}.")
        else:
            out["citations"] = [{"id": i, "chunk_id": c.id, "source": c.source, "title": c.title, "text": c.text}
                                for i, c in enumerate(used, start=1)]
            want_llm = generator.llm_enabled() if use_llm is None else use_llm
            answer, mode = None, "extractive"
            if want_llm:
                try:
                    draft = generator.llm_answer(question, used)
                    if generator.INSUFFICIENT in draft:
                        out.update(mode="abstained", answer=ABSTAIN_TEXT, citations=[])
                        out["notes"].append("LLM judged the context insufficient.")
                        draft = None
                        answer = ABSTAIN_TEXT
                    else:
                        verdict = guard.check_answer(draft, used)
                        if verdict["grounded"]:
                            answer, mode = draft, "llm"
                        else:
                            out["notes"].append("LLM draft failed the hallucination check; used extractive answer instead.")
                            mode = "llm_fallback"
                except Exception as exc:  # network error, bad key, timeout...
                    out["notes"].append(f"LLM unavailable ({type(exc).__name__}); used extractive answer.")
                    mode = "llm_fallback"
            if answer is None:
                answer = generator.extractive_answer(used)
            if out.get("mode") != "abstained":
                out.update(mode=mode, answer=answer)
                out["check"] = guard.check_answer(answer, used)

        out["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return out
