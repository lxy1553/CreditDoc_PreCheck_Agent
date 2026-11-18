PYTHON ?= python3
PYTHONPATH := src
DEMO_INDEX ?= .local/demo-index.json
DEMO_API_TOKENS := {"local-demo-only-change-this":{"subject":"demo-reviewer","institution":"DEMO_ONLY","product_lines":["DEMO_PERSONAL_BUSINESS_LOAN"]}}
DEMO_AUDIT_LOG ?= .local/demo-audit.jsonl
DEMO_JOBS_DB ?= .local/demo-jobs.sqlite

.PHONY: demo-index test eval-retrieval eval-answer eval-precheck demo-check run-demo

demo-index:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m creditdoc_precheck_agent.cli ingest \
	  --knowledge-dir knowledge/demo \
	  --manifest config/manifest.demo.json \
	  --dictionary config/keyword_dict.demo.json \
	  --output $(DEMO_INDEX) \
	  --allow-demo

test:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest -p no:cacheprovider -q
	node --test tests/workbench-render.test.mjs

eval-retrieval: demo-index
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m creditdoc_precheck_agent.evaluate \
	  --index $(DEMO_INDEX) --qdoc eval/qdoc.demo.jsonl --top-k 5

eval-answer: demo-index
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m creditdoc_precheck_agent.evaluate_answers \
	  --index $(DEMO_INDEX) --qdoca eval/qdoc-a.demo.jsonl --generator simulated

eval-precheck: demo-index
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m creditdoc_precheck_agent.evaluate_precheck \
	  --index $(DEMO_INDEX) --policy config/precheck.demo.json --cases eval/precheck.demo.jsonl

demo-check: demo-index
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest -p no:cacheprovider -q
	node --test tests/workbench-render.test.mjs
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m creditdoc_precheck_agent.evaluate \
	  --index $(DEMO_INDEX) --qdoc eval/qdoc.demo.jsonl --top-k 5
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m creditdoc_precheck_agent.evaluate_answers \
	  --index $(DEMO_INDEX) --qdoca eval/qdoc-a.demo.jsonl --generator simulated
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m creditdoc_precheck_agent.evaluate_precheck \
	  --index $(DEMO_INDEX) --policy config/precheck.demo.json --cases eval/precheck.demo.jsonl

run-demo: demo-index
	CREDITDOC_RAG_INDEX=$(DEMO_INDEX) \
	CREDITDOC_RAG_GENERATOR=simulated \
	CREDITDOC_RAG_DOCUMENT_PARSER=simulated \
	CREDITDOC_RAG_PRECHECK_POLICY=config/precheck.demo.json \
	CREDITDOC_RAG_AUDIT_PATH=$(DEMO_AUDIT_LOG) \
	CREDITDOC_PRECHECK_JOBS_DB=$(DEMO_JOBS_DB) \
	CREDITDOC_PRECHECK_WORKERS=2 \
	CREDITDOC_PRECHECK_QUEUE_TIMEOUT_SECONDS=90 \
	CREDITDOC_PRECHECK_DEADLINE_SECONDS=180 \
	CREDITDOC_PRECHECK_BACKLOG_MAX=200 \
	CREDITDOC_RAG_API_TOKENS='$(DEMO_API_TOKENS)' \
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m creditdoc_precheck_agent.api
