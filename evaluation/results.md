# Evaluation results

Embedding backend: tfidf
Questions: 10 answerable, 6 out-of-scope. Guard test cases: 12.

| Metric | Result |
|---|---|
| Retrieval hit@1 (right document ranked first) | 10/10 = 100% |
| Retrieval hit@3 | 10/10 = 100% |
| MRR | 1.00 |
| Answerable questions wrongly refused | 0/10 |
| Out-of-scope questions correctly refused | 6/6 = 100% |
| Guard: fabricated answers caught | 8/9 |
| Guard: faithful answers wrongly flagged | 0/3 |

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
