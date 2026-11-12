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
| Refusal | Refuse if the best fused similarity is below 0.12 or fewer than 50% of the question's keywords appear in the best chunk | In a safety setting "I don't know" beats a confident guess. The keyword rule stays as a safety net even in hybrid mode. |
| Generation | Extractive by default; optional LLM (`LLM_PROVIDER=anthropic`) that sees only the retrieved passages | Extractive answers cannot invent facts. An LLM draft must pass the guard or we fall back to extractive. |
| Guard | Per sentence: valid citation, >= 80% of key words in the cited chunk, all numbers present, no new negation, no flipped direction word (above/below) | Deterministic and explainable: it tells you *why* a sentence failed. |

**Not used:** no agent (the flow is a fixed pipeline, not a model choosing tools). RAG **is** used
because answers are built from retrieved document chunks.

## Evaluation (`python -m evaluation.run_eval`)

Data: 10 answerable + 6 out-of-scope questions (`data/eval_questions.csv`), plus 12 hand-written
answers to test the guard. Results below are for the default TF-IDF backend.

| Metric | Result |
|---|---|
| Retrieval hit@1 / hit@3 / MRR | 10/10 / 10/10 / 1.00 |
| Answerable questions wrongly refused | 0/10 |
| Out-of-scope questions correctly refused | 6/6 |
| Guard: fabricated answers caught | 8/9 |
| Guard: faithful answers wrongly flagged | 0/3 |

**Read these numbers with care.** The corpus is 4 documents (6 chunks) and the questions were
written by the same person who built the system, so scores are optimistic and say little about
real-world accuracy. Three settings were changed *after* seeing failures, so they are tuned to
this small set: keyword-coverage cutoff 0.4 -> 0.5 (a "blade replacement" question matched only
on the words "wind turbine"), guard support 0.6 -> 0.8 (a half-invented sentence slipped
through), and a new direction-word rule added after "...temperature is **below** 75C" passed the
guard. The one remaining guard miss is deliberate and honest: *"Verify SCADA alarms first, then
create a ticket..."* reorders the steps using the source's own words.

## Limitations

- Default TF-IDF matches words, not meaning. Hybrid mode adds meaning, but its similarity scale
  is uncalibrated here: the eval prints calibration numbers and `DENSE_FLOOR` should be set from
  them. The keyword-coverage refusal rule is kept, so a paraphrased question with no shared
  keywords is still refused (safe, but conservative).
- The guard is rule-based: it checks words, numbers, negations and direction words, not full
  meaning, so reordered steps or subtle rewording can pass (shown above).
- Tiny corpus; no PDFs, tables or versioning; no authentication; no conversation memory.
- The neural-embedding and LLM paths were tested with stand-ins for their fusion, fallback and
  guard logic. Run `python -m evaluation.run_eval` with `EMBEDDING_BACKEND=hybrid`, and
  `--llm` with a key, to produce real results for your own machine.

## Production-readiness gaps / next steps

1. Calibrate neural similarity on a larger set; relax the keyword rule once calibrated.
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
