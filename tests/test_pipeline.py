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


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
