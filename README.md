# CreditDoc PreCheck Agent — RAG foundation

This repository contains a runnable development RAG flow for approved credit precheck policies. It can ingest policy text, retrieve versioned evidence, and return cited answers through a local API. The bundled generator is explicitly simulated and accepts only the synthetic demo index. A private OpenAI-compatible generation and embedding adapter is available for deployment when the institution configures its internal model endpoints.

## Current scope

- Input: approved UTF-8 Markdown and TXT policy/guidance files listed in a manifest.
- Chunking: heading-aware parent/child chunks with source line locations.
- Retrieval: Chinese character BM25 with conjunction query splitting and RRF; an optional private embedding endpoint adds vector retrieval with RRF. Institution, product-line, approval-status, and effective-date filters run before ranking.
- Query expansion: configurable domain dictionary for standard terms, abbreviations, colloquial phrases, and known typos.
- Applicant material flow: `/v1/parse-document` sends PDF/PNG/JPEG bytes only to a private parser adapter and returns page-grounded structured OCR fields; `/v1/precheck` checks missing materials, OCR confidence, and configured cross-document fields, then retrieves rule evidence. Every result stays in human review and explicitly makes no credit decision.
- Output: evidence and citations, or an empty result that should be handled as “no reliable rule evidence; send to human review.”
- Index file: `.local/index.json`, excluded from Git because it contains full source text.
- API: authenticated `/v1/parse-document`, `/v1/precheck`, `/v1/precheck/upload-jobs`, job polling and history, SLO summary, `/v1/retrieve`, and `/v1/answer` endpoints. Request bodies are not logged by this application. API bearer tokens are scoped to one institution and an explicit product-line allowlist. Successful API requests write metadata-only events to `.local/audit.jsonl` (owner-only file permissions); configure `CREDITDOC_RAG_AUDIT_PATH` for the approved audit location. Job state transitions are also recorded in an append-only metadata table in the owner-only SQLite file.
- Evaluation: Q-DOC section Recall@K, MRR, and no-answer false-positive rate.

DOCX parsing, a production distributed queue, reviewer-workbench integration, production identity management, external audit integration, and operational SLO validation are not implemented. PDF/image OCR uses the included private parser adapter contract; an institution-hosted PaddleOCR-VL plus field-extraction service must be configured before production parsing works. The local demo parser only returns fixed fictional results for five fixed filenames and ignores document bytes. Private embedding and cross-encoder reranking adapters are available but need institution-hosted endpoints to run. The JSON index is suitable for the small demo corpus only; a production corpus needs an approved vector/search store and institution-specific access controls.

## First ingestion

1. Place only approved policy or precheck guidance in `knowledge/approved/`.
2. Copy `config/manifest.example.json` to `config/manifest.json` and `config/keyword_dict.demo.json` to `config/keyword_dict.json`; replace all example identifiers, scope, effective dates, source names, and demo terms.
3. Build the local index:

```sh
PYTHONPATH=src python3 -m creditdoc_precheck_agent.cli ingest \
  --knowledge-dir knowledge/approved \
  --manifest config/manifest.json \
  --dictionary config/keyword_dict.json
```

4. Retrieve evidence with the institution and product scope supplied:

```sh
PYTHONPATH=src python3 -m creditdoc_precheck_agent.cli query \
  --text '个人经营贷需要哪些收入证明材料' \
  --institution YOUR_INSTITUTION_ID \
  --product-line YOUR_PRODUCT_LINE
```

The example manifest is intentionally not usable as-is. Do not ingest real applicant files into this shared knowledge base.

For local development, install the optional test dependencies with `python3 -m pip install -e '.[test]'`.

The full synthetic demo flow is available through `make demo-check` (one command to run Python and browser-renderer tests plus all three evaluations), `make demo-index`, `make test`, `make eval-retrieval`, `make eval-answer`, `make eval-precheck`, and `make run-demo`. Node.js is needed to run the workbench renderer tests. The async worker defaults to a 90-second queue cutoff, a 180-second hard deadline, two workers, and a 200-job backlog cap; these are configurable and remain unvalidated sizing assumptions. Runtime settings are documented in `config/runtime.demo.env.example` and `config/runtime.private.env.example`. The end-to-end local demo flow is mapped in [贷前材料预审流程图.md](贷前材料预审流程图.md), and the local-only scope and verification checklist is in [LOCAL_DEMO_COMPLETION.md](LOCAL_DEMO_COMPLETION.md).

## Synthetic demo corpus

The repo includes fictional rules, a checklist, and an isolated conflicting-rules fixture in `knowledge/demo/`, plus a manifest and sample retrieval questions. The content is not a real institution policy; conflicting rules are isolated to `DEMO_CONFLICT_TEST_PRODUCT`. Build its isolated index only with the explicit demo flag:

```sh
PYTHONPATH=src python3 -m creditdoc_precheck_agent.cli ingest \
  --knowledge-dir knowledge/demo \
  --manifest config/manifest.demo.json \
  --dictionary config/keyword_dict.demo.json \
  --output .local/demo-index.json \
  --allow-demo
```

Query the demo index with `--institution DEMO_ONLY --product-line DEMO_PERSONAL_BUSINESS_LOAN`. Keep this corpus separate from any production index.

Run the demo API locally (the default port is 8080; override it if already occupied):

```sh
mkdir -m 700 -p /private/tmp/creditdoc-demo-run
CREDITDOC_RAG_PORT=18789 \
DEMO_INDEX=/private/tmp/creditdoc-demo-index.json \
DEMO_AUDIT_LOG=/private/tmp/creditdoc-demo-run/audit.jsonl \
DEMO_JOBS_DB=/private/tmp/creditdoc-demo-run/jobs.sqlite \
make run-demo
```

启动后打开 <http://127.0.0.1:18789/> 可进入本地模拟预审工作台（使用默认端口时地址为 `http://127.0.0.1:8080/`）。页面可选择五种虚构场景，查看异步任务状态和预审结果，刷新 SLO 汇总，并查询模拟规则。“最近预审任务”列表可按当前令牌的机构、产品线和审核员范围浏览任务并重新打开结果。状态元数据保存在本地 SQLite，材料和结果只保存在进程内存；服务重启后遗留任务会转人工，历史记录不会恢复预审结果。任务轮询会按 API 返回的硬截止时间等待终态；查询中断或页面刷新后，可使用“刷新任务状态”继续查询同一任务，转人工终态不会被迟到结果覆盖。页面仅在合成索引、模拟生成器和模拟解析器全部启用时提供；默认令牌与演示配置一致。页面不保存令牌，上传内容由固定模拟解析器忽略。

Send an authenticated request:

```sh
curl http://127.0.0.1:18789/v1/answer \
  -H 'Authorization: Bearer local-demo-only-change-this' \
  -H 'Content-Type: application/json' \
  -d '{"query":"收入证明和申请表金额不一致怎么办","institution":"DEMO_ONLY","product_line":"DEMO_PERSONAL_BUSINESS_LOAN"}'
```

The private model and document-parser adapters require institution-managed endpoints. The parser contract accepts PDF/PNG/JPEG uploads up to 15 MB and returns page-numbered extracted fields; only normalized fields and page references are returned by the API. `CREDITDOC_RAG_DOCUMENT_PARSER=simulated` enables five fixed fictional filenames for the local demo and ignores uploaded bytes; it is not OCR. Configure the internal PaddleOCR-VL plus schema-constrained extractor in `config/runtime.private.env.example`. Example settings are in that file. For vector retrieval, ingest with `--embed` after configuring `CREDITDOC_RAG_EMBEDDING_BASE_URL` and `CREDITDOC_RAG_EMBEDDING_MODEL`, then run the API with the same embedding model. For reranking, configure `CREDITDOC_RAG_RERANK_BASE_URL` and `CREDITDOC_RAG_RERANK_MODEL`. Private or explicitly allowlisted hosts are required; adapters reject arbitrary public model hosts.

Run the synthetic retrieval evaluation:

```sh
PYTHONPATH=src python3 -m creditdoc_precheck_agent.evaluate \
  --index .local/demo-index.json --qdoc eval/qdoc.demo.jsonl --top-k 5
```

Add `--embed` and/or `--rerank` when evaluating an index built with the corresponding private services.

The synthetic Q-DOC set has 9 queries (6 answerable, 3 expected abstentions), including wrong-product filtering, the day before a rule becomes effective, the inclusive effective-from boundary, and two conflicting rules isolated under `DEMO_CONFLICT_TEST_PRODUCT`. Current section Recall@5 is 1.00, MRR is 0.75, and the expected no-answer false-positive rate is 0.00. The 7-case Q-DOC-A set checks answerability, citations, required terms, and the human-review flag, including the conflicting-rules case. These small fictional fixtures demonstrate pipeline behavior only; they are not expert-validated quality evidence.

Run end-to-end synthetic answer evaluation:

```sh
PYTHONPATH=src python3 -m creditdoc_precheck_agent.evaluate_answers \
  --index .local/demo-index.json --qdoca eval/qdoc-a.demo.jsonl --generator simulated
```

The 7 synthetic answer cases currently report 1.00 answerability accuracy, citation grounding, expected-source precision/coverage, required-term coverage, and human-review rate. The demo generator only extracts the top two retrieved snippets; these scores do not measure real LLM quality.

Run end-to-end field-precheck evaluation:

```sh
make eval-precheck
```

The four fictional cases currently produce exact issue-category match 1.00 across missing materials, field differences, low OCR confidence, and a clean case. This is a plumbing check only, not validated accuracy. The upload-to-precheck API round trip also verifies the example income mismatch of 12,000 vs. 10,000 yields a 2,000 difference with page references and retrieved rules. `POST /v1/precheck/upload-jobs` runs parsing and checks in workers; poll `GET /v1/precheck/jobs/{job_id}`. `GET /v1/precheck/slo` reports timely dispositions separately from timely Agent results. At the 90-second queue cutoff, queued work is immediately marked `manual_review`; after the 180-second hard deadline, running work is marked `manual_review` and late results cannot overwrite that state. A timely manual disposition counts only in the disposition metric, never as an Agent result. If rule retrieval yields no evidence, the precheck result still returns with a human-verification marker; this differs from a timed-out job being transferred to manual review.

## Next build stages

1. Replace synthetic rules with approved, versioned institution knowledge and reviewer-validated Q-DOC/Q-DOC-A labels.
2. Configure and validate internal embedding, reranker, and generation endpoints, then re-embed and rerun evaluations.
3. Connect the included PDF/PNG/JPEG parser adapter to the institution-approved private OCR and field-extraction service; DOCX parsing and staged publication remain future work.
4. Connect the local queue API to a production durable queue and encrypted case/result store, then to the reviewer workbench, institutional identity, and an institution-managed audit sink. The current SQLite database stores metadata only; payloads and completed results remain in memory and orphaned work goes to manual review after restart. The included local JSONL audit sink is a development fallback, not a production retention or tamper-evidence solution.
5. Run workload and OCR-quality validation against approved, reviewer-labeled cases before enabling automatic routing; local worker timing tests do not establish production SLO compliance.
