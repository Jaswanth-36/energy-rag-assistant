# Energy Operations Document RAG Assistant

**Live demo:** https://web-production-77fb7066.up.railway.app

A document assistant for renewable-energy operations. An engineer asks a question
("What should be checked for inverter derating?") and gets an answer **taken only from the
SOP documents**, with **citations**, the **retrieved evidence**, and a **hallucination check**
on every sentence. If the documents don't cover the question, it **refuses** instead of guessing.

## Screenshots

| Cited and verified answer | Out-of-scope refusal |
|---|---|
| ![Gearbox answer](screenshots/1_cited_answer_gearbox.png) | ![Refusal](screenshots/2_refusal_out_of_scope.png) |

| Second cited answer | Railway deployment |
|---|---|
| ![Inverter answer](screenshots/3_cited_answer_inverter.png) | ![Railway active](screenshots/4_railway_active.png) |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn backend.app:app --reload                       # open http://127.0.0.1:8000
python -m evaluation.run_eval                          # evaluation report
python -m pytest -q                                    # tests
```

The default mode needs no API key and no internet: TF-IDF retrieval and extractive answers.

### Optional: neural embeddings and LLM answers

```bash
pip install -r requirements-dense.txt                  # neural embedding model (runs locally, ONNX)
export EMBEDDING_BACKEND=hybrid                        # Windows PowerShell: $env:EMBEDDING_BACKEND="hybrid"
export LLM_PROVIDER=anthropic ANTHROPIC_API_KEY=...    # never commit the key
uvicorn backend.app:app --reload
python -m evaluation.run_eval                          # retrieval eval + embedding calibration
python -m evaluation.run_eval --llm                    # evaluates the real LLM answer path
```

The evaluation ends with a "Neural-embedding calibration" section. If it prints suggested
`DENSE_FLOOR` and `DENSE_ACCEPT` values, set them and rerun. If it reports that the answerable and
out-of-scope groups overlap (as it does on the bundled question set, see below), leave
`DENSE_ACCEPT` unset:

```bash
export DENSE_FLOOR=0.xx DENSE_ACCEPT=0.xx              # only when the calibration section suggests values
python -m evaluation.run_eval
```

`GET /api/health` shows which backends are active. If the neural model cannot load, the app
falls back to TF-IDF automatically and says so in the health output and in each answer.

## How it works (input -> process -> output)

```
question
  -> 1 chunking      split each .md SOP into sentence-aware chunks (backend/chunking.py)
  -> 2 embedding     TF-IDF vectors; optionally fused with neural embeddings (backend/retriever.py)
  -> 3 retrieval     fused cosine similarity, top-k, plus neighbour chunks, in document order
  -> 4 decide        too dissimilar / too few question keywords found?  -> REFUSE
  -> 5 generate      extractive answer (default) or LLM-written answer, every sentence tagged [n]
  -> 6 verify        hallucination guard checks each sentence against its cited chunk
  -> JSON: answer, citations, retrieval scores, per-sentence check, mode, embedding backend, notes
```

| Step | Design choice | Why |
|---|---|---|
| Chunking | Pack whole sentences up to 20 words; never cut mid-sentence; title kept as context | A citation must be a readable passage. Sentence chunks keep the exact instruction intact. |
| Embeddings | Default: TF-IDF (1-2 word grams, stemming, stop-words removed). Optional `hybrid`: TF-IDF + a local neural model (BAAI/bge-small-en-v1.5), scores fused 50/50 | TF-IDF is offline, instant and explainable. The neural part matches by meaning ("running hot" -> "temperature exceeds"). Fallback to TF-IDF keeps the app running if the model is unavailable. |
| Retrieval | Fused cosine top-k + neighbour expansion, passages shown in document order | The sentence that answers a question often sits next to the best match; step 1 should appear before step 2. |
| Refusal | Refuse if the best fused similarity is below 0.12 or fewer than 50% of the question's keywords appear in the best chunk, unless the optional `DENSE_ACCEPT` meaning check passes | In a safety setting "I don't know" beats a confident guess. |
| Generation | Extractive by default; optional LLM (`LLM_PROVIDER=anthropic`) that sees only the retrieved passages | Extractive answers cannot invent facts. An LLM draft must pass the guard or we fall back to extractive. |
| Guard | Per sentence: valid citation, >= 80% of key words in the cited chunk, all numbers present, no new negation, no flipped direction word (above/below) | Deterministic and explainable: it tells you *why* a sentence failed. |

**Not used:** no agent (the flow is a fixed pipeline, not a model choosing tools). RAG **is** used
because answers are built from retrieved document chunks.

## Evaluation (`python -m evaluation.run_eval`)

Data (`data/questions.csv`): 15 answerable questions (10 *keyword* questions that share words
with the documents, 5 *paraphrase* questions that use different words) + 9 out-of-scope questions,
plus 12 hand-written answers to test the guard. Both backends were measured on this same set.

| Metric | TF-IDF (default) | Hybrid (TF-IDF + bge-small-en-v1.5) |
|---|---|---|
| Retrieval hit@1 | 14/15 | 15/15 |
| Retrieval hit@3 | 14/15 | 15/15 |
| MRR | 0.93 | 1.00 |
| Answerable questions wrongly refused | 4/15 (all paraphrases) | 4/15 (all paraphrases) |
| Out-of-scope questions correctly refused | 9/9 | 9/9 |
| Guard: fabricated answers caught | 8/9 | 8/9 |
| Guard: faithful answers wrongly flagged | 0/3 | 0/3 |

| Question type | Questions | Right document first (TF-IDF / hybrid) | Wrongly refused (both) |
|---|---|---|---|
| keyword | 10 | 10/10 / 10/10 | 0/10 |
| paraphrase | 5 | 4/5 / 5/5 | 4/5 |

### Neural-embedding calibration (measured)

Best raw cosine similarity per question, hybrid backend:

| Group | Min | Max |
|---|---|---|
| Answerable questions | 0.63 | 0.91 |
| Out-of-scope questions | 0.39 | 0.69 |

The two ranges overlap, so no single similarity threshold separates answerable from out-of-scope
questions on this set. The calibration step therefore suggests no safe `DENSE_ACCEPT`, and it is
**left unset by default**: the keyword rule stays in charge of the accept/refuse decision.

**Main finding.** Neural embeddings fix *ranking*: the right document is first for all 15
questions, including every paraphrase (TF-IDF missed one). They do **not** fix the *refusal
decision*: 4 of 5 paraphrase questions, such as *"What if the inverter is running hot?"* (document
says *"temperature exceeds 75C"*), are still refused, because the gate relies on keyword overlap.
This is the safe direction of error (refuse a valid question rather than answer an out-of-scope
one), and all 9 out-of-scope questions are refused, but it remains a real weakness. Opening the gate
with a cosine cut-off would need a value that only a larger paraphrase set could justify; a cut-off
chosen from 24 questions would be overfitted.

**Read these numbers with care.** The corpus is 4 documents (6 chunks) and the questions were
written by the same person who built the system, so scores are optimistic and say little about
real-world accuracy. With 24 questions the figures show direction, not statistical precision.
Three settings were changed *after* seeing failures, so they are tuned to this small set:
keyword-coverage cutoff 0.4 -> 0.5 (a "blade replacement" question matched only on the words
"wind turbine"), guard support 0.6 -> 0.8 (a half-invented sentence slipped through), and a new
direction-word rule added after "...temperature is **below** 75C" passed the guard. The one
remaining guard miss is deliberate and honest: *"Verify SCADA alarms first, then create a
ticket..."* reorders the steps using the source's own words.

## Limitations

- Default TF-IDF matches words, not meaning, so paraphrased questions are refused (4 of 5 in the
  evaluation). Hybrid mode improves ranking (hit@1 15/15) but does not change this: the same 4
  paraphrases are still refused, because raw neural similarity for answerable questions (0.63-0.91)
  overlaps with out-of-scope questions (0.39-0.69) and no safe `DENSE_ACCEPT` exists on this set.
- The guard is rule-based: it checks words, numbers, negations and direction words, not full
  meaning, so reordered steps or subtle rewording can pass (shown above).
- Tiny corpus; no PDFs, tables or versioning; no authentication; no conversation memory.
- The LLM path was tested with a stand-in for its fallback and guard logic. Run
  `python -m evaluation.run_eval --llm` with a key to measure it on your own machine.

## Production-readiness gaps / next steps

1. Collect many more paraphrased questions (answerable and out-of-scope) so a neural accept
   threshold, or a trained relevance classifier / cross-encoder reranker, can be fitted from data.
2. Add an entailment (NLI) model or LLM-as-judge to the guard for meaning-level checks.
3. Larger, independently written evaluation set; track metrics in CI.
4. Document ingestion for PDF/Word, document versions and effective dates, access control.
5. Logging, rate limiting, and human feedback on answers.

## Configuration (all optional)

Copy `.env.example`. Never commit real keys; set them as Railway variables.

| Variable | Meaning |
|---|---|
| `EMBEDDING_BACKEND=hybrid` | Adds neural embeddings (needs `requirements-dense.txt`); default `tfidf` |
| `EMBEDDING_MODEL` | Defaults to `BAAI/bge-small-en-v1.5` |
| `DENSE_ACCEPT` | Answer a question with few shared keywords when the best passage's neural similarity is at least this value. **Unset (off) by default and recommended to stay off**: on the bundled evaluation set no value separates answerable from out-of-scope questions. |
| `DENSE_WEIGHT` / `DENSE_FLOOR` | Weight of the neural score (0.5) and the cosine below which text counts as unrelated (0.5) |
| `LLM_PROVIDER=anthropic` | Turns on LLM answers (needs the key below) |
| `ANTHROPIC_API_KEY` | Secret key, set in Railway -> Variables |
| `ANTHROPIC_MODEL` | Defaults to `claude-haiku-5-5` |

## API

- `GET /` frontend, `GET /api/health`, `GET /api/sources` (all chunks)
- `POST /api/ask` with `{"question": "...", "top_k": 3}`

## Deploy to Railway

1. Push this folder to a GitHub repository (the `.gitignore` already excludes `.env`).
2. Sign in at railway.com -> **New Project** -> **Deploy from GitHub repo** -> pick the repo.
3. Railway detects Python from `requirements.txt` and starts the app with the `Procfile`:
   `web: uvicorn backend.app:app --host 0.0.0.0 --port $PORT`. A `.python-version` file
   containing `3.12` pins the Python version; without it Railway chose 3.13 and the first
   deployment crashed with the pinned package versions.
4. Open the service -> **Settings -> Networking -> Generate Domain**.
5. Visit `https://<your-domain>/api/health`. Expect `{"status":"ok","chunks":6,...}`, then open the root URL.
6. (Optional) **Variables** tab: add `LLM_PROVIDER` and `ANTHROPIC_API_KEY`. For neural
   embeddings also add the line `fastembed>=0.3.0` to `requirements.txt` and set
   `EMBEDDING_BACKEND=hybrid`; check `/api/health` afterwards to see which backend is active.
7. Deployed at https://web-production-77fb7066.up.railway.app (see the Railway screenshot above).

If a build fails, check the deploy logs; the usual causes are a Python-version or pinned-package
mismatch, fixed by editing `.python-version` or `requirements.txt`.