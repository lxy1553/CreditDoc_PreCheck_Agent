"""Tenant-scoped FastAPI wrapper for the RAG service.

Request bodies are not logged. Audit events contain metadata only and exclude
queries, generated answers, and evidence text.
"""

from __future__ import annotations

import base64
import binascii
import hmac
import json
import os
import re
import sqlite3
import time
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Set

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .audit import LocalJsonlAuditSink
from .embedding import OpenAICompatibleEmbeddingProvider
from .document_parser import DocumentParseError, PrivateDocumentParser, SimulatedDocumentParser
from .generation import OpenAICompatibleGenerator, RAGService, SimulatedAnswerGenerator
from .jobs import IdempotencyConflict, PrecheckJobCoordinator, SQLiteJobStore
from .rag import RAGInputError, RetrievalUnavailable, load_index
from .rerank import PrivateReranker
from .precheck import PrecheckEngine, PrecheckInputError


class MaterialField(BaseModel):
    value: Any
    confidence: float = Field(1.0, ge=0, le=1)
    page: Optional[int] = Field(default=None, ge=1)
    source_ref: Optional[str] = Field(default=None, max_length=360)


class MaterialDocument(BaseModel):
    document_type: str = Field(..., min_length=1, max_length=120)
    source_ref: str = Field(..., min_length=1, max_length=240)
    fields: Dict[str, MaterialField] = Field(default_factory=dict)


class ParseDocumentRequest(BaseModel):
    institution: str = Field(..., min_length=1, max_length=120)
    product_line: str = Field(..., min_length=1, max_length=120)
    filename: str = Field(..., min_length=1, max_length=255)
    content_type: str = Field(..., min_length=1, max_length=100)
    content_base64: str = Field(..., min_length=1, max_length=20 * 1024 * 1024)


class UploadDocument(BaseModel):
    filename: str = Field(..., min_length=1, max_length=255)
    content_type: str = Field(..., min_length=1, max_length=100)
    content_base64: str = Field(..., min_length=1, max_length=20 * 1024 * 1024)


class UploadPrecheckRequest(BaseModel):
    institution: str = Field(..., min_length=1, max_length=120)
    product_line: str = Field(..., min_length=1, max_length=120)
    as_of: Optional[date] = None
    documents: list[UploadDocument]


class PrecheckRequest(BaseModel):
    institution: str = Field(..., min_length=1, max_length=120)
    product_line: str = Field(..., min_length=1, max_length=120)
    as_of: Optional[date] = None
    documents: list[MaterialDocument]


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    institution: str = Field(..., min_length=1, max_length=120)
    product_line: str = Field(..., min_length=1, max_length=120)
    as_of: Optional[date] = None
    top_k: int = Field(5, ge=1, le=20)


MAX_DOCUMENT_BYTES = 15 * 1024 * 1024


def _decode_document_upload(filename: str, content_type: str, encoded: str) -> tuple[str, bytes]:
    expected_type = PrivateDocumentParser.ALLOWED_TYPES.get(Path(filename).suffix.lower())
    if expected_type is None or content_type.lower() != expected_type:
        raise HTTPException(status_code=422, detail="only PDF, PNG, or JPEG uploads with matching content types are supported")
    try:
        content = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=422, detail="content_base64 is invalid") from exc
    if not content:
        raise HTTPException(status_code=422, detail="uploaded document is empty")
    if len(content) > MAX_DOCUMENT_BYTES:
        raise HTTPException(status_code=413, detail="uploaded document exceeds the 15 MB per-file size limit")
    return expected_type, content


def _public_job(job: Dict[str, Any]) -> Dict[str, Any]:
    labels = {"queued": "预审排队中", "running": "预审处理中",
              "manual_review": "转人工预审", "completed": "Agent预审完成"}
    return {key: job.get(key) for key in (
        "job_id", "status", "reason", "created_at", "started_at", "disposition_at",
        "agent_result_at", "timely_disposition", "agent_result_slo_met", "result",
    ) if key in job} | {"workbench_status": labels.get(job["status"], "转人工预审")}


def create_app(service: RAGService, api_tokens: Mapping[str, Mapping[str, Any]], audit_sink: Any = None,
               precheck_engine: Optional[PrecheckEngine] = None, document_parser: Any = None,
               job_coordinator: Optional[PrecheckJobCoordinator] = None) -> FastAPI:
    if not api_tokens:
        raise ValueError("at least one scoped API bearer token is required")
    scopes: Dict[str, Dict[str, Any]] = {}
    for token, scope in api_tokens.items():
        institution = str(scope.get("institution", ""))
        product_lines = scope.get("product_lines")
        subject = str(scope.get("subject", ""))
        if not token or not institution or not subject or not isinstance(product_lines, list) or not product_lines:
            raise ValueError("each API token needs subject, institution, and product_lines")
        scopes[token] = {"subject": subject, "institution": institution,
                         "product_lines": frozenset(str(value) for value in product_lines)}
    sink = audit_sink or LocalJsonlAuditSink(Path(os.environ.get("CREDITDOC_RAG_AUDIT_PATH", ".local/audit.jsonl")))
    app = FastAPI(title="CreditDoc PreCheck RAG", version="0.2.0")
    app.state.rag_service = service
    app.state.audit_sink = sink
    app.state.precheck_engine = precheck_engine
    app.state.document_parser = document_parser
    app.state.job_coordinator = job_coordinator
    if job_coordinator is not None and precheck_engine is not None:
        previous_processor = job_coordinator.processor
        def process_job(payload: Dict[str, Any]) -> Dict[str, Any]:
            if "_upload_documents" not in payload:
                return previous_processor(payload)
            if document_parser is None:
                raise DocumentParseError("document parser is not configured")
            documents = []
            for upload in payload["_upload_documents"]:
                parsed = document_parser.parse(upload["content"], filename=upload["filename"],
                                               content_type=upload["content_type"])
                fields = {}
                for name, value in parsed["fields"].items():
                    fields[name] = {**value, "source_ref": "%s#page=%d" % (upload["filename"], value["page"])}
                documents.append({"document_type": parsed["document_type"],
                                  "source_ref": upload["filename"], "fields": fields})
            case = {key: payload[key] for key in ("institution", "product_line", "as_of") if payload.get(key) is not None}
            case["documents"] = documents
            return precheck_engine.review(case)
        job_coordinator.processor = process_job

    async def require_scope(authorization: Optional[str] = Header(default=None)) -> Dict[str, Any]:
        supplied = authorization[7:] if authorization and authorization.startswith("Bearer ") else ""
        matched: Optional[Dict[str, Any]] = None
        # Compare every configured token to reduce token-position timing leakage.
        for expected, scope in scopes.items():
            if hmac.compare_digest(supplied.encode("utf-8"), expected.encode("utf-8")):
                matched = scope
        if matched is None:
            raise HTTPException(status_code=401, detail="unauthorized")
        return matched

    def check_scope(payload: QueryRequest, scope: Dict[str, Any]) -> None:
        if payload.institution != scope["institution"] or payload.product_line not in scope["product_lines"]:
            raise HTTPException(status_code=403, detail="token is not authorized for this institution or product line")

    def audit(request: Request, scope: Dict[str, Any], payload: QueryRequest,
              route: str, started: float, result: Dict[str, Any]) -> None:
        citations = result.get("citations") or []
        evidence = result.get("evidence") or []
        event = {
            "event_id": str(uuid.uuid4()),
            "request_id": getattr(request.state, "request_id", str(uuid.uuid4())),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "route": route,
            "subject": scope["subject"],
            "institution": payload.institution,
            "product_line": payload.product_line,
            "status": result.get("status", "evidence_found" if evidence else "needs_human_review"),
            "elapsed_ms": max(0, round((time.monotonic() - started) * 1000)),
            "evidence_count": len(evidence),
            "document_ids": sorted({str(item.get("citation", {}).get("doc_id", "")) for item in evidence} - {""}),
            "document_versions": sorted({str(item.get("citation", {}).get("version", "")) for item in evidence} - {""}),
            "citation_chunk_ids": sorted({str(item.get("chunk_id", "")) for item in citations} - {""}),
            "generator_mode": result.get("mode", "retrieval_only"),
        }
        try:
            app.state.audit_sink.write(event)
        except Exception as exc:
            raise HTTPException(status_code=503, detail="audit service unavailable; route to manual review") from exc

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        supplied = request.headers.get("X-Request-ID", "")
        request_id = supplied if re.fullmatch(r"[A-Za-z0-9._:-]{1,100}", supplied) else str(uuid.uuid4())
        request.state.request_id = request_id
        request.state.request_started_at = time.time()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    @app.get("/", include_in_schema=False)
    async def demo_workbench():
        is_demo = (service.index.get("environment") == "synthetic-demo"
                   and getattr(service.generator, "mode", None) == "simulated"
                   and isinstance(app.state.document_parser, SimulatedDocumentParser)
                   and app.state.precheck_engine is not None
                   and app.state.job_coordinator is not None)
        if not is_demo:
            raise HTTPException(status_code=404, detail="not found")
        page = Path(__file__).with_name("static") / "workbench.html"
        if not page.is_file():
            raise HTTPException(status_code=404, detail="demo workbench is unavailable")
        return FileResponse(page, media_type="text/html", headers={
            "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff",
        })

    @app.get("/static/workbench-render.js", include_in_schema=False)
    async def demo_workbench_renderer():
        is_demo = (service.index.get("environment") == "synthetic-demo"
                   and getattr(service.generator, "mode", None) == "simulated"
                   and isinstance(app.state.document_parser, SimulatedDocumentParser)
                   and app.state.precheck_engine is not None
                   and app.state.job_coordinator is not None)
        if not is_demo:
            raise HTTPException(status_code=404, detail="not found")
        script = Path(__file__).with_name("static") / "workbench-render.js"
        if not script.is_file():
            raise HTTPException(status_code=404, detail="demo workbench renderer is unavailable")
        return FileResponse(script, media_type="application/javascript", headers={
            "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff",
        })

    @app.post("/v1/parse-document")
    def parse_document_endpoint(payload: ParseDocumentRequest, request: Request,
                                scope: Dict[str, Any] = Depends(require_scope)):
        check_scope(payload, scope)
        if app.state.document_parser is None:
            raise HTTPException(status_code=503, detail="private document parser is not configured; route to manual review")
        filename = Path(payload.filename).name
        if filename != payload.filename or filename in {".", ".."}:
            raise HTTPException(status_code=422, detail="filename must be a plain file name")
        content_type, content = _decode_document_upload(filename, payload.content_type, payload.content_base64)
        started = time.monotonic()
        try:
            parsed = app.state.document_parser.parse(content, filename=filename, content_type=content_type)
        except DocumentParseError as exc:
            raise HTTPException(status_code=503, detail="document parser unavailable; route to manual review") from exc
        field_results = {}
        for name, value in parsed["fields"].items():
            source_ref = "%s#page=%d" % (filename, value["page"])
            field_results[name] = {**value, "source_ref": source_ref}
        event = {
            "event_id": str(uuid.uuid4()), "request_id": request.state.request_id,
            "timestamp": datetime.now(timezone.utc).isoformat(), "route": "/v1/parse-document",
            "subject": scope["subject"], "institution": payload.institution,
            "product_line": payload.product_line, "status": "document_parsed",
            "elapsed_ms": max(0, round((time.monotonic() - started) * 1000)),
            "document_type": parsed["document_type"], "page_count": len(parsed["pages"]),
            "generator_mode": "private_document_parser",
        }
        try:
            app.state.audit_sink.write(event)
        except Exception as exc:
            raise HTTPException(status_code=503, detail="audit service unavailable; route to manual review") from exc
        return {"document_type": parsed["document_type"], "source_ref": filename,
                "pages": [{"page": page["page"]} for page in parsed["pages"]],
                "fields": field_results}

    @app.post("/v1/precheck")
    def precheck_endpoint(payload: PrecheckRequest, request: Request,
                          scope: Dict[str, Any] = Depends(require_scope)):
        check_scope(payload, scope)
        if app.state.precheck_engine is None:
            raise HTTPException(status_code=503, detail="precheck engine is not configured")
        started = time.monotonic()
        try:
            result = app.state.precheck_engine.review(payload.model_dump() if hasattr(payload, "model_dump") else payload.dict())
            aggregate = {"status": result["status"], "mode": "deterministic_precheck",
                         "citations": [cite for issue in result["issues"] for cite in issue.get("citations", [])],
                         "evidence": [item for issue in result["issues"] for item in issue.get("rule_evidence", [])]}
            audit(request, scope, payload, "/v1/precheck", started, aggregate)
            return result
        except PrecheckInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

    @app.post("/v1/precheck/jobs", status_code=202)
    def submit_precheck_job(payload: PrecheckRequest, request: Request,
                            idempotency_key: str = Header(..., alias="Idempotency-Key", min_length=1, max_length=128),
                            scope: Dict[str, Any] = Depends(require_scope)):
        check_scope(payload, scope)
        if app.state.precheck_engine is None or app.state.job_coordinator is None:
            raise HTTPException(status_code=503, detail="async precheck worker is not configured; route to manual review")
        started = time.monotonic()
        normalized = payload.model_dump() if hasattr(payload, "model_dump") else payload.dict()
        try:
            job = app.state.job_coordinator.submit(
                normalized, institution=payload.institution, product_line=payload.product_line,
                subject=scope["subject"], idempotency_key=idempotency_key,
                received_at=request.state.request_started_at,
            )
        except (RuntimeError, sqlite3.Error) as exc:
            raise HTTPException(status_code=503, detail="precheck queue unavailable; route to manual review") from exc
        except IdempotencyConflict as exc:
            raise HTTPException(status_code=409, detail="idempotency key was already used with a different request") from exc
        result = {"status": job["status"], "mode": "async_precheck", "citations": [], "evidence": []}
        audit(request, scope, payload, "/v1/precheck/jobs", started, result)
        return _public_job(job)

    @app.post("/v1/precheck/upload-jobs", status_code=202)
    def submit_upload_job(payload: UploadPrecheckRequest, request: Request,
                          idempotency_key: str = Header(..., alias="Idempotency-Key", min_length=1, max_length=128),
                          scope: Dict[str, Any] = Depends(require_scope)):
        check_scope(payload, scope)
        if app.state.precheck_engine is None or app.state.job_coordinator is None or app.state.document_parser is None:
            raise HTTPException(status_code=503, detail="private parser or async worker is unavailable; route to manual review")
        if not payload.documents or len(payload.documents) > 30:
            raise HTTPException(status_code=422, detail="upload job requires 1 to 30 documents")
        uploads = []
        total_bytes = 0
        for item in payload.documents:
            if Path(item.filename).name != item.filename or item.filename in {".", ".."}:
                raise HTTPException(status_code=422, detail="filename must be a plain file name")
            content_type, content = _decode_document_upload(item.filename, item.content_type,
                                                              item.content_base64)
            total_bytes += len(content)
            if total_bytes > 30 * 1024 * 1024:
                raise HTTPException(status_code=413, detail="upload job exceeds the 30 MB total size limit")
            uploads.append({"filename": item.filename, "content_type": content_type, "content": content})
        normalized = {"institution": payload.institution, "product_line": payload.product_line,
                      "as_of": payload.as_of.isoformat() if payload.as_of else None,
                      "_upload_documents": uploads}
        started = time.monotonic()
        try:
            job = app.state.job_coordinator.submit(normalized, institution=payload.institution,
                                                   product_line=payload.product_line, subject=scope["subject"],
                                                   idempotency_key=idempotency_key,
                                                   received_at=request.state.request_started_at)
        except (RuntimeError, sqlite3.Error) as exc:
            raise HTTPException(status_code=503, detail="precheck queue unavailable; route to manual review") from exc
        except IdempotencyConflict as exc:
            raise HTTPException(status_code=409, detail="idempotency key was already used with a different request") from exc
        result = {"status": job["status"], "mode": "async_upload_precheck", "citations": [], "evidence": []}
        audit(request, scope, payload, "/v1/precheck/upload-jobs", started, result)
        return _public_job(job)

    @app.get("/v1/precheck/jobs/{job_id}")
    def get_precheck_job(job_id: str, scope: Dict[str, Any] = Depends(require_scope)):
        if app.state.job_coordinator is None:
            raise HTTPException(status_code=503, detail="async precheck worker is not configured")
        job = None
        for product_line in scope["product_lines"]:
            job = app.state.job_coordinator.get(job_id, institution=scope["institution"],
                                                product_line=product_line, subject=scope["subject"])
            if job is not None:
                break
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        return _public_job(job)

    @app.get("/v1/precheck/history")
    def precheck_job_history(limit: int = Query(20, ge=1, le=50),
                             scope: Dict[str, Any] = Depends(require_scope)):
        if app.state.job_coordinator is None:
            raise HTTPException(status_code=503, detail="async precheck worker is not configured")
        jobs = app.state.job_coordinator.store.list_for_scope(
            institution=scope["institution"], product_lines=scope["product_lines"],
            subject=scope["subject"], limit=limit,
        )
        return {"items": [_public_job(job) for job in jobs]}

    @app.get("/v1/precheck/slo")
    def precheck_slo(scope: Dict[str, Any] = Depends(require_scope)):
        if app.state.job_coordinator is None:
            raise HTTPException(status_code=503, detail="async precheck worker is not configured")
        summary = app.state.job_coordinator.store.slo_summary(institution=scope["institution"],
                                                               product_lines=scope["product_lines"],
                                                               subject=scope["subject"])
        summary["queue_timeout_seconds"] = app.state.job_coordinator.queue_timeout_seconds
        summary["deadline_seconds"] = app.state.job_coordinator.deadline_seconds
        summary["worker_count"] = app.state.job_coordinator.workers
        summary["queue_depth"] = app.state.job_coordinator._queue.qsize()
        return summary

    @app.post("/v1/retrieve")
    def retrieve_endpoint(payload: QueryRequest, request: Request, scope: Dict[str, Any] = Depends(require_scope)):
        check_scope(payload, scope)
        started = time.monotonic()
        try:
            evidence = app.state.rag_service.get_evidence(
                payload.query, institution=payload.institution,
                product_line=payload.product_line, as_of=payload.as_of, top_k=payload.top_k,
            )
            result = {"status": "evidence_found" if evidence else "needs_human_review",
                      "mode": "retrieval_only", "evidence": evidence, "citations": [],
                      "needs_human_review": not bool(evidence)}
            audit(request, scope, payload, "/v1/retrieve", started, result)
            return {"evidence": evidence, "needs_human_review": not bool(evidence)}
        except RAGInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except RetrievalUnavailable:
            raise HTTPException(status_code=503, detail="retrieval service unavailable")

    @app.post("/v1/answer")
    def answer_endpoint(payload: QueryRequest, request: Request, scope: Dict[str, Any] = Depends(require_scope)):
        check_scope(payload, scope)
        started = time.monotonic()
        try:
            result = app.state.rag_service.answer(
                payload.query, institution=payload.institution,
                product_line=payload.product_line, as_of=payload.as_of, top_k=payload.top_k,
            )
            audit(request, scope, payload, "/v1/answer", started, result)
            return result
        except RAGInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

    if job_coordinator is not None:
        app.add_event_handler("shutdown", job_coordinator.close)
    return app


def app_from_environment() -> FastAPI:
    index_path = Path(os.environ.get("CREDITDOC_RAG_INDEX", ".local/index.json"))
    raw_tokens = os.environ.get("CREDITDOC_RAG_API_TOKENS", "")
    try:
        tokens = json.loads(raw_tokens)
    except json.JSONDecodeError as exc:
        raise ValueError("CREDITDOC_RAG_API_TOKENS must be a JSON object") from exc
    if not isinstance(tokens, dict):
        raise ValueError("CREDITDOC_RAG_API_TOKENS must be a JSON object")
    generator_mode = os.environ.get("CREDITDOC_RAG_GENERATOR", "private_model")
    index = load_index(index_path)
    embedding_provider = None
    if index.get("embedding_model"):
        embedding_provider = OpenAICompatibleEmbeddingProvider.from_environment()
        if embedding_provider.model_name != index["embedding_model"]:
            raise ValueError("configured embedding model does not match the model used to build this index")
    reranker = PrivateReranker.from_environment() if os.environ.get("CREDITDOC_RAG_RERANK_BASE_URL") else None
    if generator_mode == "simulated":
        if index.get("environment") != "synthetic-demo":
            raise ValueError("simulated generator requires a synthetic-demo index")
        generator = SimulatedAnswerGenerator()
    elif generator_mode == "private_model":
        generator = OpenAICompatibleGenerator.from_environment()
    else:
        raise ValueError("CREDITDOC_RAG_GENERATOR must be simulated or private_model")
    service = RAGService(index, generator, embedding_provider, reranker)
    policy_path = Path(os.environ.get("CREDITDOC_RAG_PRECHECK_POLICY", "config/precheck.json"))
    precheck_engine = PrecheckEngine.from_file(policy_path, service) if policy_path.is_file() else None
    parser_mode = os.environ.get("CREDITDOC_RAG_DOCUMENT_PARSER", "private")
    if parser_mode == "simulated":
        document_parser = SimulatedDocumentParser(environment=index.get("environment", ""))
    elif parser_mode == "private":
        document_parser = PrivateDocumentParser.from_environment() if os.environ.get("CREDITDOC_RAG_DOCUMENT_PARSER_URL") else None
    else:
        raise ValueError("CREDITDOC_RAG_DOCUMENT_PARSER must be simulated or private")
    job_store = SQLiteJobStore(Path(os.environ.get("CREDITDOC_PRECHECK_JOBS_DB", ".local/precheck-jobs.sqlite"))) if precheck_engine else None
    job_coordinator = None
    if job_store:
        job_coordinator = PrecheckJobCoordinator(
            job_store, precheck_engine.review,
            workers=int(os.environ.get("CREDITDOC_PRECHECK_WORKERS", "2")),
            queue_timeout_seconds=float(os.environ.get("CREDITDOC_PRECHECK_QUEUE_TIMEOUT_SECONDS", "90")),
            deadline_seconds=float(os.environ.get("CREDITDOC_PRECHECK_DEADLINE_SECONDS", "180")),
            max_queue_size=int(os.environ.get("CREDITDOC_PRECHECK_BACKLOG_MAX", "200")),
        )
    application = create_app(service, tokens, precheck_engine=precheck_engine,
                             document_parser=document_parser, job_coordinator=job_coordinator)
    return application


def main() -> None:
    import uvicorn

    uvicorn.run(app_from_environment(), host=os.environ.get("CREDITDOC_RAG_HOST", "127.0.0.1"),
                port=int(os.environ.get("CREDITDOC_RAG_PORT", "8080")), access_log=False)


if __name__ == "__main__":
    main()
