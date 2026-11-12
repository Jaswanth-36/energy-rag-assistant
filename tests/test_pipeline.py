"""Run with:  python -m pytest -q   (or)   python tests/test_pipeline.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend import guard
from backend.chunking import chunk_document
from backend.pipeline import RAGPipeline

pipe = RAGPipeline()


def test_chunks_never_cut_sentences_and_keep_source():
    chunks = chunk_document("x.md", "# T\nFirst sentence here. Second sentence here.", max_words=3)
    assert [c.text for c in chunks] == ["First sentence here.", "Second sentence here."]
    assert all(c.source == "x.md" and c.title == "T" for c in chunks)


def test_in_scope_question_is_answered_with_citation_from_right_document():
    r = pipe.ask("What should I verify before inspecting a spillway gate?")
    assert r["mode"] == "extractive"
    assert r["citations"][0]["source"] == "hydro_safety_brief.md"
    assert "[1]" in r["answer"] and r["check"]["grounded"]


def test_out_of_scope_question_abstains():
    r = pipe.ask("What is the capital of France?")
    assert r["mode"] == "abstained" and r["citations"] == []


def test_guard_rejects_invented_number_and_missing_citation():
    ctx = [pipe.chunks[2]]  # solar inverter SOP, first chunk (75C)
    assert not guard.check_answer("Inspect the inverter if it exceeds 85C [1].", ctx)["grounded"]
    assert not guard.check_answer("Inspect the cooling fans.", ctx)["grounded"]
    assert guard.check_answer("If inverter temperature exceeds 75C, inspect cooling fans [1].", ctx)["grounded"]


class FakeDense:
    """Stand-in for a neural model: 'running hot' means temperature, even with no shared words."""

    def __init__(self, chunks):
        self.chunks = chunks

    def scores(self, question):
        import numpy as np
        return np.array([0.9 if ("hot" in question and "exceeds" in c.text) else 0.3 for c in self.chunks])


def test_hybrid_ranks_by_meaning_when_words_do_not_overlap():
    from backend.retriever import Retriever
    q = "What if the inverter is running hot?"
    plain = Retriever(pipe.chunks).search(q, k=1)[0]
    hybrid = Retriever(pipe.chunks, dense=FakeDense(pipe.chunks)).search(q, k=1)[0]
    assert "exceeds" in hybrid.chunk.text and hybrid.dense == 0.9
    assert hybrid.score > plain.score


def test_paraphrase_is_refused_by_keyword_rule_but_accepted_by_meaning_when_calibrated():
    from backend.retriever import Retriever
    q = "What if the inverter is running hot?"
    r = Retriever(pipe.chunks, dense=FakeDense(pipe.chunks))
    off = RAGPipeline(retriever=r).ask(q)
    on = RAGPipeline(retriever=r, dense_accept=0.8).ask(q)
    assert off["mode"] == "abstained"
    assert on["mode"] == "extractive" and on["citations"][0]["source"] == "solar_inverter_sop.md"
    # an unrelated question is still refused even with the neural gate on
    assert RAGPipeline(retriever=r, dense_accept=0.8).ask("What is the capital of France?")["mode"] == "abstained"


def test_missing_neural_package_falls_back_to_tfidf(monkeypatch=None):
    import os
    from backend.retriever import build_retriever
    os.environ["EMBEDDING_BACKEND"] = "hybrid"
    os.environ["EMBEDDING_MODEL"] = "no-such-model"
    try:
        r = build_retriever(pipe.chunks)
        assert r.dense is None and r.backend == "tfidf" and "using TF-IDF only" in r.note
    finally:
        del os.environ["EMBEDDING_BACKEND"], os.environ["EMBEDDING_MODEL"]


def test_answer_steps_follow_document_order():
    r = pipe.ask("What should be checked for inverter derating?")
    assert r["answer"].index("exceeds 75C") < r["answer"].index("repeated derating")


def test_guard_rejects_reversed_direction_word():
    ctx = [pipe.chunks[2]]
    assert not guard.check_answer("Inspect cooling fans if the inverter temperature is below 75C [1].", ctx)["grounded"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
