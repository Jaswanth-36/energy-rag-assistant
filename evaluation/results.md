# Evaluation results

Embedding backend: hybrid (TF-IDF + BAAI/bge-small-en-v1.5)
Questions: 15 answerable, 9 out-of-scope. Guard test cases: 12.

| Metric | Result |
|---|---|
| Retrieval hit@1 (right document ranked first) | 15/15 = 100% |
| Retrieval hit@3 | 15/15 = 100% |
| MRR | 1.00 |
| Answerable questions wrongly refused | 4/15 |
| Out-of-scope questions correctly refused | 9/9 = 100% |
| Guard: fabricated answers caught | 8/9 |
| Guard: faithful answers wrongly flagged | 0/3 |

## Answerable questions by type

keyword = shares words with the document; paraphrase = different words, same meaning.

| Type | Questions | Right document first | Wrongly refused |
|---|---|---|---|
| keyword | 10 | 10/10 | 0/10 |
| paraphrase | 5 | 5/5 | 4/5 |

## Guard test cases

| Case | Should pass | Did pass | OK |
|---|---|---|---|
| faithful paraphrase | True | True | yes |
| faithful paraphrase | True | True | yes |
| faithful paraphrase | True | True | yes |
| wrong number | False | False | yes |
| invented number + invented action | False | False | yes |
| invented action | False | False | yes |
| missing citation | False | False | yes |
| citation to a non-existent source | False | False | yes |
| flipped meaning (negation added) | False | False | yes |
| partly invented detail | False | False | yes |
| direction reversed (below vs exceeds) | False | False | yes |
| steps reordered using source words (known blind spot) | False | True | **NO** |

## Neural-embedding calibration

Best raw cosine for answerable questions: min 0.63, max 0.91
Best raw cosine for out-of-scope questions: min 0.39, max 0.69
The two groups overlap, so no safe DENSE_ACCEPT exists: leave it unset and keep the keyword rule.
