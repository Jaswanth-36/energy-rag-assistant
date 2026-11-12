"""Evaluation: retrieval quality, abstention, and hallucination-guard tests.

Run from the project root:   python -m evaluation.run_eval
Writes evaluation/results.json and evaluation/results.md.
"""
import argparse
import csv
import json
from pathlib import Path

from backend import generator, guard
from backend.pipeline import DATA_DIR, RAGPipeline

OUT = Path(__file__).parent

# (question, answer text, should_pass_guard, what the test is about)
GUARD_CASES = [
    ("What should be checked for inverter derating?",
     "Inspect the cooling fans and dust filters when the inverter temperature goes above 75C [@cooling].",
     True, "faithful paraphrase"),
    ("How should gearbox vibration be investigated?",
     "Check the vibration spectrum, oil level and oil contamination [@spectrum].",
     True, "faithful paraphrase"),
    ("What should I verify before inspecting a spillway gate?",
     "Verify lockout-tagout and the water level before a spillway gate inspection [@lockout].",
     True, "faithful paraphrase"),
    ("What temperature should trigger an inverter inspection?",
     "Inspect the inverter if its temperature exceeds 85C [@75C].",
     False, "wrong number"),
    ("How should gearbox vibration be investigated?",
     "Replace the gearbox immediately if vibration exceeds 12 mm/s [@spectrum].",
     False, "invented number + invented action"),
    ("What should be checked for inverter derating?",
     "Shut down the whole plant and call the regulator [@cooling].",
     False, "invented action"),
    ("What should I verify before inspecting a spillway gate?",
     "Verify lockout-tagout and the water level before inspection.",
     False, "missing citation"),
    ("What should I verify before inspecting a spillway gate?",
     "Verify lockout-tagout and the water level before inspection [5].",
     False, "citation to a non-existent source"),
    ("What should be checked for inverter derating?",
     "Do not inspect the cooling fans [@cooling].",
     False, "flipped meaning (negation added)"),
    ("What should be checked for inverter derating?",
     "Inspect the cooling fans, dust filters and the solar panel glass for cracks [@cooling].",
     False, "partly invented detail"),
    # Known weakness, reported honestly: reuses source words, changes the meaning.
    ("What should be checked for inverter derating?",
     "Create a priority maintenance ticket and inspect cooling fans if the inverter temperature is below 75C [@75C] [@priority].",
     False, "direction reversed (below vs exceeds)"),
    # Honest remaining blind spot: same words, but the order of the steps is changed.
    ("What should be checked for inverter derating?",
     "Verify SCADA alarms first, then create a priority maintenance ticket if repeated derating occurs [@priority].",
     False, "steps reordered using source words (known blind spot)"),
]


def resolve_citations(answer: str, used) -> str:
    """Replace [@word] with the citation number of the retrieved chunk containing `word`,
    so each test cites the chunk a careful writer would actually cite."""
    import re

    def repl(m):
        for i, c in enumerate(used, start=1):
            if m.group(1).lower() in c.text.lower():
                return f"[{i}]"
        return "[1]"
    return re.sub(r"\[@([^\]]+)\]", repl, answer)


def dense_calibration(pipe, rows) -> list[str]:
    """Raw neural-embedding similarity of each question's best chunk, so thresholds can be set from data."""
    if pipe.retriever.dense is None:
        return []
    ans = [float(pipe.retriever.dense.scores(r["question"]).max()) for r in rows if r["answerable"] == "yes"]
    out = [float(pipe.retriever.dense.scores(r["question"]).max()) for r in rows if r["answerable"] == "no"]
    lines = ["", "## Neural-embedding calibration", "",
             f"Best raw cosine for answerable questions: min {min(ans):.2f}, max {max(ans):.2f}",
             f"Best raw cosine for out-of-scope questions: min {min(out):.2f}, max {max(out):.2f}"]
    if min(ans) > max(out):
        mid = (min(ans) + max(out)) / 2
        lines += [f"The two groups separate cleanly. Suggested settings: DENSE_FLOOR={max(out):.2f}, DENSE_ACCEPT={mid:.2f}.",
                  "DENSE_ACCEPT lets a paraphrased question through when its best passage is at least that similar by meaning."]
    else:
        lines += ["The two groups overlap, so no safe DENSE_ACCEPT exists: leave it unset and keep the keyword rule."]
    return lines


def run_llm_eval(pipe, answerable) -> None:
    """Needs LLM_PROVIDER=anthropic and ANTHROPIC_API_KEY. Measures the real LLM path end to end."""
    if not generator.llm_enabled():
        raise SystemExit("LLM mode is off. Set LLM_PROVIDER=anthropic and ANTHROPIC_API_KEY, then rerun with --llm.")
    modes = {"llm": 0, "llm_fallback": 0, "abstained": 0, "extractive": 0}
    rows = []
    for r in answerable:
        res = pipe.ask(r["question"], use_llm=True)
        modes[res["mode"]] = modes.get(res["mode"], 0) + 1
        rows.append(f"| {r['question']} | {res['mode']} | {(res['check'] or {}).get('grounded')} | {res['answer'][:110]} |")
    n = len(answerable)
    md = ["# LLM-mode evaluation", "",
          f"Model: {__import__('os').getenv('ANTHROPIC_MODEL', 'claude-haiku-5-5')}  |  questions: {n}", "",
          f"- LLM answer passed the hallucination guard: {modes['llm']}/{n}",
          f"- LLM draft rejected, extractive fallback used: {modes['llm_fallback']}/{n}",
          f"- Wrongly refused: {modes['abstained']}/{n}", "",
          "| Question | Mode | Grounded | Answer (first 110 chars) |", "|---|---|---|---|"] + rows
    (OUT / "results_llm.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true", help="evaluate the real LLM answer path (needs an API key)")
    args = ap.parse_args()
    pipe = RAGPipeline()
    rows = list(csv.DictReader(open(DATA_DIR / "eval_questions.csv", encoding="utf-8")))
    answerable = [r for r in rows if r["answerable"] == "yes"]
    unanswerable = [r for r in rows if r["answerable"] == "no"]

    if args.llm:
        run_llm_eval(pipe, answerable)
        return

    # --- Retrieval (answerable questions): did the right document come back? ---
    hit1 = hit3 = 0
    rr_sum = 0.0
    abstained_wrongly = []
    details = []
    for r in answerable:
        hits = pipe.retriever.search(r["question"], k=3)
        sources = [h.chunk.source for h in hits]
        rank = sources.index(r["expected_source"]) + 1 if r["expected_source"] in sources else None
        hit1 += rank == 1
        hit3 += rank is not None
        rr_sum += 1 / rank if rank else 0.0
        res = pipe.ask(r["question"])
        if res["mode"] == "abstained":
            abstained_wrongly.append(r["question"])
        details.append({"question": r["question"], "type": r.get("type", "keyword"), "expected": r["expected_source"], "rank": rank,
                        "mode": res["mode"], "grounded": (res["check"] or {}).get("grounded")})
    n = len(answerable)

    by_type = {}
    for d in details:
        t = by_type.setdefault(d["type"], {"n": 0, "hit1": 0, "refused": 0})
        t["n"] += 1
        t["hit1"] += d["rank"] == 1
        t["refused"] += d["mode"] == "abstained"

    # --- Abstention (out-of-scope questions): did we refuse instead of guessing? ---
    correct_abstain = 0
    answered_wrongly = []
    for r in unanswerable:
        res = pipe.ask(r["question"])
        if res["mode"] == "abstained":
            correct_abstain += 1
        else:
            answered_wrongly.append(r["question"])

    # --- Faithfulness of our own answers: do they pass the guard? ---
    grounded = sum(1 for d in details if d["grounded"])

    # --- Guard test: can it tell good answers from fabricated ones? ---
    guard_rows = []
    for q, ans, should_pass, kind in GUARD_CASES:
        _, used, _ = pipe.retrieve_context(q)
        ans = resolve_citations(ans, used)
        verdict = guard.check_answer(ans, used)
        guard_rows.append({"kind": kind, "expected_pass": should_pass, "got_pass": verdict["grounded"],
                           "correct": verdict["grounded"] == should_pass, "answer": ans})
    bad = [g for g in guard_rows if not g["expected_pass"]]
    good = [g for g in guard_rows if g["expected_pass"]]
    caught = sum(1 for g in bad if not g["got_pass"])
    false_alarms = sum(1 for g in good if not g["got_pass"])

    results = {
        "n_answerable": n, "n_unanswerable": len(unanswerable),
        "retrieval_hit_at_1": hit1 / n, "retrieval_hit_at_3": hit3 / n, "mrr": rr_sum / n,
        "wrongly_abstained": abstained_wrongly,
        "abstention_accuracy": correct_abstain / len(unanswerable), "wrongly_answered": answered_wrongly,
        "answers_passing_guard": f"{grounded}/{n - len(abstained_wrongly)}",
        "guard_fabrications_caught": f"{caught}/{len(bad)}",
        "guard_false_alarms_on_good_answers": f"{false_alarms}/{len(good)}",
        "by_type": by_type, "guard_cases": guard_rows, "retrieval_details": details,
    }
    (OUT / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    md = ["# Evaluation results", "",
          f"Embedding backend: {pipe.retriever.backend}" + (f"  ({pipe.retriever.note})" if pipe.retriever.note else ""),
          f"Questions: {n} answerable, {len(unanswerable)} out-of-scope. Guard test cases: {len(GUARD_CASES)}.", "",
          "| Metric | Result |", "|---|---|",
          f"| Retrieval hit@1 (right document ranked first) | {hit1}/{n} = {hit1 / n:.0%} |",
          f"| Retrieval hit@3 | {hit3}/{n} = {hit3 / n:.0%} |",
          f"| MRR | {rr_sum / n:.2f} |",
          f"| Answerable questions wrongly refused | {len(abstained_wrongly)}/{n} |",
          f"| Out-of-scope questions correctly refused | {correct_abstain}/{len(unanswerable)} = {correct_abstain / len(unanswerable):.0%} |",
          f"| Guard: fabricated answers caught | {caught}/{len(bad)} |",
          f"| Guard: faithful answers wrongly flagged | {false_alarms}/{len(good)} |", ""]
    if answered_wrongly:
        md += ["Out-of-scope questions the system wrongly answered:"] + [f"- {q}" for q in answered_wrongly] + [""]
    md += ["## Answerable questions by type", "",
           "keyword = shares words with the document; paraphrase = different words, same meaning.", "",
           "| Type | Questions | Right document first | Wrongly refused |", "|---|---|---|---|"]
    md += [f"| {k} | {v['n']} | {v['hit1']}/{v['n']} | {v['refused']}/{v['n']} |" for k, v in by_type.items()]
    md += ["", "## Guard test cases", "", "| Case | Should pass | Did pass | OK |", "|---|---|---|---|"]
    md += [f"| {g['kind']} | {g['expected_pass']} | {g['got_pass']} | {'yes' if g['correct'] else '**NO**'} |" for g in guard_rows]
    md += dense_calibration(pipe, rows)
    (OUT / "results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
