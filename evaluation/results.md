# Evaluation results

Embedding backend: tfidf
Questions: 15 answerable, 9 out-of-scope. Guard test cases: 12.

| Metric | Result |
|---|---|
| Retrieval hit@1 (right document ranked first) | 14/15 = 93% |
| Retrieval hit@3 | 14/15 = 93% |
| MRR | 0.93 |
| Answerable questions wrongly refused | 4/15 |
| Out-of-scope questions correctly refused | 9/9 = 100% |
| Guard: fabricated answers caught | 8/9 |
| Guard: faithful answers wrongly flagged | 0/3 |

## Answerable questions by type

keyword = shares words with the document; paraphrase = different words, same meaning.

| Type | Questions | Right document first | Wrongly refused |
|---|---|---|---|
| keyword | 10 | 10/10 | 0/10 |
| paraphrase | 5 | 4/5 | 4/5 |

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
