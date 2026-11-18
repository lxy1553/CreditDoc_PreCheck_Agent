import asyncio
from pathlib import Path

import httpx
import pytest
from httpx import ASGITransport

from creditdoc_precheck_agent.api import create_app
from creditdoc_precheck_agent.audit import MemoryAuditSink
from creditdoc_precheck_agent.generation import RAGService, SimulatedAnswerGenerator
from creditdoc_precheck_agent.rag import build_index
from creditdoc_precheck_agent.precheck import PrecheckEngine
from creditdoc_precheck_agent.document_parser import SimulatedDocumentParser
from creditdoc_precheck_agent.jobs import PrecheckJobCoordinator, SQLiteJobStore


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def app(tmp_path):
    index = build_index(ROOT / "knowledge" / "demo", ROOT / "config" / "manifest.demo.json", allow_demo=True)
    audit = MemoryAuditSink()
    engine = PrecheckEngine.from_file(ROOT / "config/precheck.demo.json", RAGService(index, SimulatedAnswerGenerator()))
    coordinator = PrecheckJobCoordinator(SQLiteJobStore(tmp_path / "jobs.sqlite"), engine.review,
                                          poll_interval_seconds=0.01)
    app = create_app(
        RAGService(index, SimulatedAnswerGenerator()),
        {"demo-secret": {"subject": "demo-reviewer", "institution": "DEMO_ONLY",
                         "product_lines": ["DEMO_PERSONAL_BUSINESS_LOAN"]}},
        audit_sink=audit,
        document_parser=SimulatedDocumentParser(environment="synthetic-demo"),
        precheck_engine=engine,
        job_coordinator=coordinator,
    )
    app.state.test_audit = audit
    yield app
    coordinator.close()


def test_answer_api_requires_bearer_token(app):
    async def request():
        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            return await client.post("/v1/answer", json={
                "query": "预审材料清单", "institution": "DEMO_ONLY",
                "product_line": "DEMO_PERSONAL_BUSINESS_LOAN",
            })

    response = asyncio.run(request())
    assert response.status_code == 401


def test_non_ascii_bearer_token_is_rejected_without_server_error(app):
    async def request():
        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            return await client.get("/v1/precheck/history", headers={b"Authorization": b"Bearer \xff"})

    response = asyncio.run(request())
    assert response.status_code == 401


def test_demo_workbench_is_served_only_when_simulated_adapters_are_enabled(app):
    async def request():
        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            return await client.get("/")

    response = asyncio.run(request())
    assert response.status_code == 200
    assert "本地模拟预审工作台" in response.text
    assert "DEMO_ONLY" in response.text
    assert "工作台需要通过本地 API 服务打开" in response.text
    assert "刷新任务状态" in response.text
    assert all(value in response.text for value in (
        'value="income-difference"', 'value="missing-material"',
        'value="low-confidence"', 'value="consistent"',
        'value="rule-evidence-missing"'))
    assert "2025-12-31" in response.text

    app.state.document_parser = None
    disabled = asyncio.run(request())
    assert disabled.status_code == 404


def test_workbench_loads_its_readable_result_renderer(app):
    async def request():
        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            return await client.get("/static/workbench-render.js")

    response = asyncio.run(request())
    assert response.status_code == 200
    assert "formatIssue" in response.text
    assert "application/javascript" in response.headers["content-type"]


def test_answer_api_returns_grounded_demo_response(app):
    async def request():
        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            return await client.post("/v1/answer", headers={"Authorization": "Bearer demo-secret", "X-Request-ID": "req-demo-001"}, json={
                "query": "收入证明和申请表金额不一致怎么办", "institution": "DEMO_ONLY",
                "product_line": "DEMO_PERSONAL_BUSINESS_LOAN",
            })

    response = asyncio.run(request())
    assert response.status_code == 200
    result = response.json()
    assert result["mode"] == "simulated"
    assert result["needs_human_review"] is True
    assert result["citations"]


def test_blocking_answer_service_does_not_block_health_requests(app):
    import threading
    import time

    started = threading.Event()
    release = threading.Event()
    started_at = []

    class SlowGenerator:
        mode = "simulated"

        def generate(self, query, evidence, *, environment):
            started_at.append(time.monotonic())
            started.set()
            release.wait(1.2)
            return SimulatedAnswerGenerator().generate(query, evidence, environment=environment)

    app.state.rag_service.generator = SlowGenerator()
    timer = threading.Timer(1.2, release.set)
    timer.start()

    async def request():
        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            answer_task = asyncio.create_task(client.post("/v1/answer", headers={
                "Authorization": "Bearer demo-secret",
            }, json={"query": "收入证明和申请表金额不一致怎么办", "institution": "DEMO_ONLY",
                    "product_line": "DEMO_PERSONAL_BUSINESS_LOAN"}))
            assert await asyncio.to_thread(started.wait, 1)
            health = await client.get("/healthz")
            health_latency = time.monotonic() - started_at[0]
            answer = await answer_task
            return health, health_latency, answer

    try:
        health, health_latency, answer = asyncio.run(request())
    finally:
        release.set()
        timer.cancel()

    assert health.status_code == 200
    assert health_latency < 0.4
    assert answer.status_code == 200


def test_answer_api_rejects_institution_outside_token_scope(app):
    async def request():
        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            return await client.post("/v1/answer", headers={"Authorization": "Bearer demo-secret"}, json={
                "query": "预审材料清单", "institution": "OTHER_BANK",
                "product_line": "DEMO_PERSONAL_BUSINESS_LOAN",
            })

    assert asyncio.run(request()).status_code == 403


def test_answer_api_persists_metadata_only_audit(app):
    async def request():
        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            return await client.post("/v1/answer", headers={
                "Authorization": "Bearer demo-secret", "X-Request-ID": "req-demo-001",
            }, json={
                "query": "收入证明和申请表金额不一致怎么办 SENSITIVE_QUERY_DO_NOT_LOG", "institution": "DEMO_ONLY",
                "product_line": "DEMO_PERSONAL_BUSINESS_LOAN",
            })

    response = asyncio.run(request())
    assert response.status_code == 200
    event = app.state.test_audit.events[0]
    assert event["request_id"] == "req-demo-001"
    assert event["subject"] == "demo-reviewer"
    assert event["institution"] == "DEMO_ONLY"
    assert "query" not in event and "answer" not in event and "evidence" not in event
    assert event["elapsed_ms"] >= 0
    assert event["status"]


def test_precheck_api_returns_human_review_and_audits_no_material_values(app):
    async def request():
        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            return await client.post("/v1/precheck", headers={"Authorization": "Bearer demo-secret"}, json={
                "institution": "DEMO_ONLY", "product_line": "DEMO_PERSONAL_BUSINESS_LOAN",
                "as_of": "2026-10-03", "documents": [{
                    "document_type": "application_form", "source_ref": "申请表第2页",
                    "fields": {"applicant_name": {"value": "PII_EXAMPLE_VALUE", "confidence": 0.99}},
                }],
            })

    response = asyncio.run(request())
    assert response.status_code == 200
    result = response.json()
    assert result["decision"] == "no_credit_decision"
    assert result["needs_human_review"] is True
    event = app.state.test_audit.events[-1]
    assert "PII_EXAMPLE_VALUE" not in str(event)
    assert "申请表第2页" not in str(event)


def test_precheck_api_enforces_tenant_scope(app):
    async def request():
        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            return await client.post("/v1/precheck", headers={"Authorization": "Bearer demo-secret"}, json={
                "institution": "OTHER_BANK", "product_line": "DEMO_PERSONAL_BUSINESS_LOAN", "documents": [],
            })

    assert asyncio.run(request()).status_code == 403


def test_document_parse_api_returns_provenance_without_logging_material(app):
    import base64
    async def request():
        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            return await client.post("/v1/parse-document", headers={"Authorization": "Bearer demo-secret"}, json={
                "institution": "DEMO_ONLY", "product_line": "DEMO_PERSONAL_BUSINESS_LOAN",
                "filename": "demo-income-proof.pdf", "content_type": "application/pdf",
                "content_base64": base64.b64encode(b"%PDF-DEMO").decode("ascii"),
            })

    response = asyncio.run(request())
    assert response.status_code == 200
    result = response.json()
    assert result["document_type"] == "income_proof"
    assert result["source_ref"] == "demo-income-proof.pdf"
    assert result["fields"]["monthly_income"]["source_ref"] == "demo-income-proof.pdf#page=1"
    event = app.state.test_audit.events[-1]
    assert "10000" not in str(event)
    assert "demo-income-proof.pdf" not in str(event)


def test_document_parse_api_rejects_wrong_tenant(app):
    import base64
    async def request():
        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            return await client.post("/v1/parse-document", headers={"Authorization": "Bearer demo-secret"}, json={
                "institution": "OTHER", "product_line": "DEMO_PERSONAL_BUSINESS_LOAN",
                "filename": "demo-income-proof.pdf", "content_type": "application/pdf",
                "content_base64": base64.b64encode(b"x").decode("ascii"),
            })

    assert asyncio.run(request()).status_code == 403


def test_simulated_upload_to_precheck_roundtrip(app):
    import base64
    async def request():
        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            headers = {"Authorization": "Bearer demo-secret"}
            docs = []
            for filename in ("demo-application.pdf", "demo-identity.pdf", "demo-income-proof.pdf"):
                response = await client.post("/v1/parse-document", headers=headers, json={
                    "institution": "DEMO_ONLY", "product_line": "DEMO_PERSONAL_BUSINESS_LOAN",
                    "filename": filename, "content_type": "application/pdf",
                    "content_base64": base64.b64encode(b"synthetic-only").decode("ascii"),
                })
                assert response.status_code == 200
                parsed = response.json()
                docs.append({"document_type": parsed["document_type"], "source_ref": parsed["source_ref"],
                             "fields": parsed["fields"]})
            result = await client.post("/v1/precheck", headers=headers, json={
                "institution": "DEMO_ONLY", "product_line": "DEMO_PERSONAL_BUSINESS_LOAN",
                "as_of": "2026-10-03", "documents": docs,
            })
            return result

    response = asyncio.run(request())
    assert response.status_code == 200
    result = response.json()
    assert result["decision"] == "no_credit_decision"
    assert any(issue["category"] == "field_difference" for issue in result["issues"])
    difference = next(issue for issue in result["issues"] if issue["category"] == "field_difference")
    assert difference["details"]["observations"][0]["source_ref"].endswith("#page=1")
    assert difference["rule_evidence"]
    income_issue = next(issue for issue in result["issues"]
                        if issue["category"] == "field_difference" and issue["details"]["field"] == "monthly_income")
    assert income_issue["details"]["numeric_differences"][0]["difference"] == "2000"


def test_async_job_is_idempotent_pollable_and_reports_split_slo(app):
    async def request():
        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            headers = {"Authorization": "Bearer demo-secret", "Idempotency-Key": "same-case-001"}
            body = {"institution": "DEMO_ONLY", "product_line": "DEMO_PERSONAL_BUSINESS_LOAN",
                    "as_of": "2026-10-03", "documents": []}
            first = await client.post("/v1/precheck/jobs", headers=headers, json=body)
            second = await client.post("/v1/precheck/jobs", headers=headers, json=body)
            conflicting = await client.post("/v1/precheck/jobs", headers=headers,
                                            json={**body, "as_of": "2026-10-04"})
            assert first.status_code == 202
            assert second.status_code == 202
            assert conflicting.status_code == 409
            assert first.json()["job_id"] == second.json()["job_id"]
            job_id = first.json()["job_id"]
            result = first.json()
            for _ in range(100):
                result = (await client.get("/v1/precheck/jobs/" + job_id,
                                          headers={"Authorization": "Bearer demo-secret"})).json()
                if result["status"] in {"completed", "manual_review"}:
                    break
                await asyncio.sleep(0.01)
            slo = await client.get("/v1/precheck/slo", headers={"Authorization": "Bearer demo-secret"})
            return result, slo

    result, slo = asyncio.run(request())
    assert result["status"] == "completed"
    assert result["result"]["decision"] == "no_credit_decision"
    assert result["workbench_status"] == "Agent预审完成"
    assert slo.status_code == 200
    assert slo.json()["timely_disposition_count"] == 1
    assert slo.json()["timely_agent_result_count"] == 1


def test_async_job_polling_does_not_disclose_other_tenant(app):
    async def request():
        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/v1/precheck/jobs", headers={
                "Authorization": "Bearer demo-secret", "Idempotency-Key": "private-job-1",
            }, json={"institution": "DEMO_ONLY", "product_line": "DEMO_PERSONAL_BUSINESS_LOAN", "documents": []})
            job_id = response.json()["job_id"]
            app.state.job_coordinator.store.create(institution="OTHER_BANK", product_line="OTHER",
                subject="different-user", idempotency_key="another-key", now=1.0)
            hidden = await client.get("/v1/precheck/jobs/unknown", headers={"Authorization": "Bearer demo-secret"})
            return hidden

    assert asyncio.run(request()).status_code == 404


def test_recent_job_history_is_authenticated_and_limited_to_token_scope(app):
    store = app.state.job_coordinator.store
    own = store.create(institution="DEMO_ONLY", product_line="DEMO_PERSONAL_BUSINESS_LOAN",
                       subject="demo-reviewer", idempotency_key="history-own", now=1000)
    store.create(institution="OTHER_BANK", product_line="DEMO_PERSONAL_BUSINESS_LOAN",
                 subject="demo-reviewer", idempotency_key="history-other-tenant", now=1001)
    store.create(institution="DEMO_ONLY", product_line="DEMO_AUTO_PERSONAL_LOAN",
                 subject="demo-reviewer", idempotency_key="history-other-product", now=1002)
    store.create(institution="DEMO_ONLY", product_line="DEMO_PERSONAL_BUSINESS_LOAN",
                 subject="another-reviewer", idempotency_key="history-other-reviewer", now=1003)

    async def request():
        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            unauthorized = await client.get("/v1/precheck/history")
            response = await client.get("/v1/precheck/history?limit=10", headers={
                "Authorization": "Bearer demo-secret",
            })
            return unauthorized, response

    unauthorized, response = asyncio.run(request())
    assert unauthorized.status_code == 401
    assert response.status_code == 200
    assert [item["job_id"] for item in response.json()["items"]] == [own["job_id"]]
    assert "subject" not in response.json()["items"][0]


def test_expired_running_job_api_response_is_terminal_manual_review(app):
    async def request():
        store = app.state.job_coordinator.store
        job = store.create(institution="DEMO_ONLY", product_line="DEMO_PERSONAL_BUSINESS_LOAN",
                           subject="demo-reviewer", idempotency_key="api-hard-deadline", now=1000)
        assert store.claim(job["job_id"], now=1010, queue_timeout_seconds=90, deadline_seconds=180)
        store.expire(now=1181, queue_timeout_seconds=90, deadline_seconds=180)
        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            return await client.get("/v1/precheck/jobs/" + job["job_id"],
                                    headers={"Authorization": "Bearer demo-secret"})

    response = asyncio.run(request())
    assert response.status_code == 200
    state = response.json()
    assert state["status"] == "manual_review"
    assert state["workbench_status"] == "转人工预审"
    assert state["reason"] == "hard_deadline"
    assert "result" not in state


def test_async_upload_job_runs_ocr_comparison_and_citations_in_worker(app):
    import base64
    async def request():
        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            headers = {"Authorization": "Bearer demo-secret", "Idempotency-Key": "upload-case-001"}
            files = ["demo-application.pdf", "demo-identity.pdf", "demo-income-proof.pdf"]
            body = {"institution": "DEMO_ONLY", "product_line": "DEMO_PERSONAL_BUSINESS_LOAN",
                    "as_of": "2026-10-03", "documents": [
                        {"filename": name, "content_type": "application/pdf",
                         "content_base64": base64.b64encode(b"synthetic-only").decode("ascii")}
                        for name in files
                    ]}
            submitted = await client.post("/v1/precheck/upload-jobs", headers=headers, json=body)
            assert submitted.status_code == 202, submitted.text
            job_id = submitted.json()["job_id"]
            result = submitted.json()
            for _ in range(100):
                response = await client.get("/v1/precheck/jobs/" + job_id,
                                            headers={"Authorization": "Bearer demo-secret"})
                result = response.json()
                if result["status"] in {"completed", "manual_review"}:
                    break
                await asyncio.sleep(0.01)
            return result

    result = asyncio.run(request())
    assert result["status"] == "completed"
    income_issue = next(issue for issue in result["result"]["issues"]
                        if issue["category"] == "field_difference" and issue["details"]["field"] == "monthly_income")
    assert income_issue["details"]["numeric_differences"][0]["difference"] == "2000"
    assert income_issue["details"]["observations"][0]["source_ref"].endswith("#page=1")
    assert income_issue["rule_evidence"]
    persisted_state = app.state.job_coordinator.store.path.read_bytes()
    assert "模拟申请人甲".encode("utf-8") not in persisted_state
    assert b"12000" not in persisted_state and b"10000" not in persisted_state


def test_upload_job_rejects_unsupported_file_before_queuing(app):
    import base64
    async def request():
        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            headers = {"Authorization": "Bearer demo-secret", "Idempotency-Key": "bad-upload-type"}
            response = await client.post("/v1/precheck/upload-jobs", headers=headers, json={
                "institution": "DEMO_ONLY", "product_line": "DEMO_PERSONAL_BUSINESS_LOAN",
                "documents": [{"filename": "sample.docx", "content_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                               "content_base64": base64.b64encode(b"synthetic").decode("ascii")}],
            })
            slo = await client.get("/v1/precheck/slo", headers={"Authorization": "Bearer demo-secret"})
            return response, slo

    response, slo = asyncio.run(request())
    assert response.status_code == 422
    assert "PDF, PNG, or JPEG" in response.json()["detail"]
    assert slo.json()["total_jobs"] == 0


def test_upload_job_rejects_empty_file_before_queuing(app):
    import base64
    async def request():
        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/v1/precheck/upload-jobs", headers={
                "Authorization": "Bearer demo-secret", "Idempotency-Key": "empty-upload",
            }, json={"institution": "DEMO_ONLY", "product_line": "DEMO_PERSONAL_BUSINESS_LOAN",
                    "documents": [{"filename": "demo-application.pdf", "content_type": "application/pdf",
                                   "content_base64": base64.b64encode(b"").decode("ascii")} ]})
            slo = await client.get("/v1/precheck/slo", headers={"Authorization": "Bearer demo-secret"})
            return response, slo

    response, slo = asyncio.run(request())
    assert response.status_code == 422
    assert slo.json()["total_jobs"] == 0


@pytest.mark.parametrize(("scenario", "files", "expected_category"), [
    ("missing", ["demo-application.pdf"], "missing_material"),
    ("low-confidence", ["demo-application-blurry.pdf", "demo-identity.pdf", "demo-income-proof.pdf"], "low_ocr_confidence"),
    ("consistent", ["demo-application.pdf", "demo-identity.pdf", "demo-income-proof-matching.pdf"], None),
])
def test_local_upload_scenarios_cover_precheck_branches(app, scenario, files, expected_category):
    import base64
    async def request():
        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            submitted = await client.post("/v1/precheck/upload-jobs", headers={
                "Authorization": "Bearer demo-secret", "Idempotency-Key": "scenario-" + scenario,
            }, json={"institution": "DEMO_ONLY", "product_line": "DEMO_PERSONAL_BUSINESS_LOAN",
                    "as_of": "2026-10-03", "documents": [
                        {"filename": name, "content_type": "application/pdf",
                         "content_base64": base64.b64encode(b"synthetic-placeholder").decode("ascii")}
                        for name in files
                    ]})
            assert submitted.status_code == 202
            job_id = submitted.json()["job_id"]
            result = submitted.json()
            for _ in range(100):
                response = await client.get("/v1/precheck/jobs/" + job_id,
                                            headers={"Authorization": "Bearer demo-secret"})
                result = response.json()
                if result["status"] in {"completed", "manual_review"}:
                    break
                await asyncio.sleep(0.01)
            return result

    result = asyncio.run(request())
    assert result["status"] == "completed"
    categories = {issue["category"] for issue in result["result"]["issues"]}
    if expected_category:
        assert expected_category in categories
    else:
        assert categories == set()


def test_no_applicable_rule_is_marked_for_human_verification_in_result(app):
    import base64
    async def request():
        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            submitted = await client.post("/v1/precheck/upload-jobs", headers={
                "Authorization": "Bearer demo-secret", "Idempotency-Key": "rule-evidence-missing",
            }, json={"institution": "DEMO_ONLY", "product_line": "DEMO_PERSONAL_BUSINESS_LOAN",
                    "as_of": "2025-12-31", "documents": [
                        {"filename": name, "content_type": "application/pdf",
                         "content_base64": base64.b64encode(b"synthetic-only").decode("ascii")}
                        for name in ("demo-application.pdf", "demo-identity.pdf", "demo-income-proof.pdf")
                    ]})
            assert submitted.status_code == 202
            job_id = submitted.json()["job_id"]
            for _ in range(100):
                result = (await client.get("/v1/precheck/jobs/" + job_id,
                                           headers={"Authorization": "Bearer demo-secret"})).json()
                if result["status"] in {"completed", "manual_review"}:
                    return result
                await asyncio.sleep(0.01)
            return result

    result = asyncio.run(request())
    assert result["status"] == "completed"
    income_issue = next(issue for issue in result["result"]["issues"]
                        if issue["category"] == "field_difference" and issue["details"]["field"] == "monthly_income")
    assert income_issue["rule_evidence_status"] == "not_found"
    assert income_issue["rule_evidence"] == []
    assert income_issue["needs_human_review"] is True
    assert "请人工核对" in income_issue["rule_evidence_message"]
