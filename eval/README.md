# Retrieval evaluation data

Create an expert-reviewed `qdoc.jsonl` file here. Keep one JSON object per line. The source of truth for each query is the complete set of applicable clauses and document versions, not just the first retrieved result.

Suggested fields:

```json
{
  "query_id": "Q-0001",
  "query": "审核员的真实问题",
  "institution": "institution-id",
  "product_line": "product-line-id",
  "as_of": "2026-10-01",
  "expected": [
    {"doc_id": "POLICY-001", "version": "2.1", "section": "材料要求 > 收入证明"}
  ],
  "query_type": "procedural",
  "reviewer_verified": true
}
```

Include supported questions, no-answer questions, conflicting rule versions, out-of-scope product lines, and date-boundary cases. Keep applicant names, identity numbers, account numbers, and original case documents out of this evaluation set. Any derived query content must follow institutional data-classification rules.

The evaluator reports section Recall@K, Mean Reciprocal Rank, and no-answer false-positive rate. It can optionally use configured private embedding and reranking services so comparisons match the online retrieval path. The bundled demo file is synthetic and intentionally small; production thresholds must be set and measured on reviewer-validated data.

`qdoc-a.demo.jsonl` contains synthetic end-to-end answer checks:

```sh
PYTHONPATH=src python3 -m creditdoc_precheck_agent.evaluate_answers \
  --index .local/demo-index.json --qdoca eval/qdoc-a.demo.jsonl --generator simulated
```

The answer evaluator reports answerability accuracy, citation grounding, expected-source precision/coverage, and required-term coverage. Simulated scores are not model-quality claims.


Synthetic precheck evaluation cases live in `precheck.demo.jsonl`. Run them with `make eval-precheck`. Replace these fictional cases with expert-reviewed labels before setting acceptance thresholds.

Precheck labels may include `expected_differences` (configured field names), `applicant_group_id`, and `split` (`train`, `dev`, or `acceptance`). The report includes field-level TP/FP/FN, positive-label counts, group/split coverage, and applicant groups that cross splits. Keep all records from one applicant in one split. Missing group/split metadata does not invalidate the demo run, but the report exposes its coverage and marks the leakage check incomplete; it must be complete for an acceptance dataset.

Field recall gates are evaluated independently for identity (99.5%), financial (98%), and auxiliary (95%) fields. The report also computes the positive-example count needed for a one-sided exact binomial 95% lower bound when zero misses are observed: `ceil(log(alpha) / log(target_recall))`. Identity uses Bonferroni correction across the three critical fields (`alpha = 0.05 / 3`), requiring 817 positive examples per field; financial and auxiliary fields require 149 and 59, respectively. `adequate` means only that the sample count is sufficient for this zero-miss criterion; any false negatives fail that criterion. A field with no positive examples has no measurable recall and does not count as having observed zero misses. This does not replace expert review, representative sampling, or separate false-positive and operational acceptance gates.

The current demo has only four synthetic cases, so its reported 100% category scores and field scores are plumbing checks, not acceptance evidence. It is expected to report inadequate sample sizes. Run all local tests and evaluations with `make demo-check`.
