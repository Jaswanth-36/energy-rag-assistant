# Interview walkthrough notes (follows Case_Study_Presentation_Guide.docx)

**Do not start with code.** Open with the problem, then the flow, then the demo.

## 1. Problem in under a minute
"Field engineers need correct SOP steps fast, and a wrong or invented instruction is a safety
risk. This assistant answers from the company documents only, shows exactly which sentence it used,
checks its own answer, and refuses when the documents don't cover the question."
Success = right document found, answer cited, no invented facts, refusal when out of scope.

## 2. Flow to draw
question -> chunk -> embed (TF-IDF) -> retrieve -> refuse-or-answer -> cite -> verify -> screen/API

## 3. Be precise about what this is (the guide stresses this)
- **RAG: yes**, answers are built from retrieved document chunks.
- **Agent: no**, it's a fixed pipeline; no model chooses tools.
- **LLM:** optional. Default is extractive (no LLM). If enabled, the LLM only rewrites retrieved
  text and its output is verified by Python.
- **Python calculates:** retrieval, refusal decision, hallucination check. **LLM generates:** only wording.
- **ML model / hyperparameters:** there is no trained model. Explain TF-IDF and the tunable settings:
  chunk size (20 words), refuse thresholds (similarity 0.12, coverage 0.5), guard support (0.8).

## 4. Repo walkthrough (one or two sentences each)
README -> `data/` -> `backend/chunking.py` -> `backend/retriever.py` -> `backend/pipeline.py`
(refusal + neighbour expansion) -> `backend/generator.py` -> `backend/guard.py` ->
`backend/app.py` + `frontend/index.html` -> `evaluation/run_eval.py`.

## 5. Demo (rehearse these three)
1. "What should be checked for inverter derating?" -> cited answer, all sentences verified.
2. "How should gearbox vibration be investigated?" -> shows neighbour chunk being pulled in.
3. "What is the capital of France?" -> refuses.

## 6. Honest talking points
- Eval set is tiny and self-written, so results are optimistic. Two thresholds were tuned after seeing failures (say so).
- Known guard blind spot: "below 75C" passes because it reuses source words.
- Challenge worth telling: splitting into sentence chunks made a chunk lose its subject ("inverter"), so I count the title as context for keyword coverage. Also my first guard test "failed" because I cited the wrong chunk, and the guard was right.
- Next step: neural embeddings + NLI-based check.
- Credit AI assistance if you used it, and be able to explain every file.

## Likely questions
- *Why not just use an LLM?* It can invent steps; here every sentence is traceable and checked.
- *What if the key feature (title context) is removed?* Chunks lose their subject and recall drops (inverter derating question is the example).
- *How do you know it works on new data?* Honestly: only on a small set; the next step is a larger independent set.
- *What does the LLM generate vs Python compute?* See section 3.
