# CreditDoc PreCheck Agent RAG architecture

## Role and trust boundary

The RAG knowledge base contains institution-approved rules and reviewer guidance. Case-specific application forms and proofs remain in the per-application OCR and extraction path; they are not ingested into this shared index. The RAG layer finds applicable rule evidence and citations. Deterministic checks compute field differences, and the reviewer remains the final decision maker.

## Request flow

```text
Precheck orchestrator / async worker queue
  ├─ durable job metadata, idempotency, queue cutoff and hard deadline
  ├─ private document parser (OCR/VL + schema field extraction; page provenance)
  ├─ deterministic material/field checks (missing items, confidence, comparisons)
  └─ RAG evidence request (institution, product line, as-of date, question)
       → enforce required institution/product filters
       → filter approved, effective documents
       → domain dictionary expansion + Chinese character BM25 retrieval
       → optional private-vector fusion + optional cross-encoder reranking
       → return child evidence + parent context + source/version/section/line citation
  → assemble reviewer-facing result; unsupported or conflicting rules go to human review
```

The rule index accepts Markdown and plain text, preserves heading paths and line locations, and writes its index to `.local/index.json`. Applicant PDFs and images use a separate private parser adapter; extracted fields retain document/page references and are not added to the shared rule index. The index contains source text and optional vectors and must remain in an access-controlled local environment. An empty retrieval is an abstention signal; it is not proof that a requirement does not exist. Query conjunctions and approved dictionary aliases are expanded into focused variants. Reciprocal Rank Fusion combines lexical and optional vector rankings.

## Knowledge document lifecycle

1. A knowledge owner registers the source with a stable document ID, version, approval status, effective dates, institution, and product-line scope.
2. The importer validates the registration and source path, then parses approved UTF-8 Markdown/TXT.
3. Chunking preserves heading boundaries and creates parent context plus smaller retrieval chunks. Each chunk retains its source location and version metadata.
4. The new index is written atomically to a temporary file and renamed into place after construction.
5. Retrieval filters by institution, product line, approval status, and effective period before ranking. It uses BM25 by default and adds vector similarity only when the index was built with the matching private embedding model.
6. An optional private cross-encoder reranker scores up to 50 candidates and returns the requested top results.
7. The answer layer either uses the simulated extractive generator for the isolated demo corpus or calls an explicitly configured private OpenAI-compatible endpoint. Model citations are checked against retrieved chunk IDs. A missing citation or failed model call routes the result to human review.

Multiple versions of a document may coexist if their `(doc_id, version)` pairs are unique. Effective-date filtering chooses which versions are candidates for a query.

## Evaluation and rollout

Create an expert-reviewed Q-DOC set mapping reviewer questions to every applicable document version and clause. Include no-answer, conflicting-policy, out-of-scope product, and effective-date boundary cases. Q-DOC-A evaluates answerability, required content, citation grounding, and expected source precision/coverage. Do not use synthetic question generation as the sole source of ground truth.

Queue cutoff and hard deadline are configurable defaults of 90s and 180s respectively. Queue timeout returns a manual-review disposition; a running job beyond the hard deadline is terminal for the workbench, and late completion is discarded. Disposition SLO and Agent-result SLO are counted separately. SQLite stores job metadata only; case payloads and results are memory-only in this demo. A restart routes unrecoverable jobs to manual review. Production needs a durable queue and an approved encrypted case/result store.

The bundled synthetic generator is extractive and is not a production language model. Before production, choose approved private generation and embedding endpoints, then evaluate grounded answers for correctness, exact citation support, abstention, conflict handling, and latency. Retrieval or model failure must leave the normal precheck/manual path available.

## Current limitations

- Rule ingestion is Markdown/TXT only; applicant PDF/PNG/JPEG parsing is an adapter contract and requires a configured private OCR-plus-field-extraction endpoint. Local demo parsing uses fixed fictional outputs and does not inspect uploads. DOCX is not supported.
- The demo index uses lexical BM25 only. Optional private vector retrieval is implemented but requires a configured endpoint and full re-embedding when the model changes.
- No production-grade distributed queue, reviewer-workbench integration, institutional identity integration, or external audit sink; the included local queue is single-process demo infrastructure.
- No production model endpoint is configured in this workspace. The demo mode makes no model network calls.
- This is a development foundation, not a production-ready deployment. Encryption, access control, operational monitoring, retention, and load/SLO validation remain required before production use.
