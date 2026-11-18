"""Private OCR/document-structure adapter contract.

The institution-hosted parser may use PaddleOCR-VL plus a schema-constrained
field extractor. This client sends applicant bytes only to a private or
explicitly allowlisted endpoint and validates page provenance on return.
"""

from __future__ import annotations

import base64
import json
import math
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional, Set

from .security import is_private_or_loopback, open_without_redirect


class DocumentParseError(RuntimeError):
    """Raised when the private parser fails or returns untrusted structure."""


class PrivateDocumentParser:
    ALLOWED_TYPES = {
        ".pdf": "application/pdf",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
    }

    def __init__(self, base_url: str, model: str, *, api_key: Optional[str] = None,
                 allowed_hosts: Optional[Set[str]] = None, timeout_seconds: float = 45.0,
                 max_bytes: int = 15 * 1024 * 1024) -> None:
        parsed = urllib.parse.urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("document parser endpoint must be an absolute HTTP(S) URL")
        allowed = {host.strip().lower() for host in (allowed_hosts or set())}
        if not is_private_or_loopback(parsed.hostname) and parsed.hostname.lower() not in allowed:
            raise ValueError("document parser host is not private or explicitly allowlisted")
        if parsed.scheme == "http" and not is_private_or_loopback(parsed.hostname):
            raise ValueError("non-private parser endpoints must use HTTPS")
        if not model.strip():
            raise ValueError("document parser model name is required")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.max_bytes = max_bytes

    @classmethod
    def from_environment(cls) -> "PrivateDocumentParser":
        base_url = os.environ.get("CREDITDOC_RAG_DOCUMENT_PARSER_URL", "")
        model = os.environ.get("CREDITDOC_RAG_DOCUMENT_PARSER_MODEL", "")
        api_key = os.environ.get("CREDITDOC_RAG_DOCUMENT_PARSER_API_KEY")
        allowed_hosts = set(filter(None, os.environ.get("CREDITDOC_RAG_DOCUMENT_PARSER_ALLOWED_HOSTS", "").split(",")))
        if not base_url or not model:
            raise ValueError("CREDITDOC_RAG_DOCUMENT_PARSER_URL and CREDITDOC_RAG_DOCUMENT_PARSER_MODEL are required")
        return cls(base_url, model, api_key=api_key, allowed_hosts=allowed_hosts)

    def parse(self, content: bytes, *, filename: str, content_type: str) -> Dict[str, Any]:
        suffix = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        expected_type = self.ALLOWED_TYPES.get(suffix)
        if not expected_type or content_type.lower() != expected_type:
            raise DocumentParseError("only PDF, PNG, and JPEG uploads with matching content types are supported")
        if not content:
            raise DocumentParseError("uploaded document is empty")
        if len(content) > self.max_bytes:
            raise DocumentParseError("uploaded document exceeds the configured size limit")
        body = json.dumps({"model": self.model, "filename": filename, "content_type": content_type,
                           "content_base64": base64.b64encode(content).decode("ascii")}).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        request = urllib.request.Request(self.base_url + "/v1/parse", data=body, headers=headers, method="POST")
        try:
            with open_without_redirect(request, timeout=self.timeout_seconds) as response:
                raw = response.read(4 * 1024 * 1024 + 1)
            if len(raw) > 4 * 1024 * 1024:
                raise DocumentParseError("document parser response exceeds the configured size limit")
            parsed = json.loads(raw.decode("utf-8"))
        except DocumentParseError:
            raise
        except (urllib.error.URLError, TimeoutError, OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DocumentParseError("private document parser failed or returned invalid JSON") from exc
        return self._validate(parsed)

    @staticmethod
    def _validate(payload: Any) -> Dict[str, Any]:
        if not isinstance(payload, dict) or not isinstance(payload.get("document_type"), str) or not payload["document_type"]:
            raise DocumentParseError("parser response requires a document_type")
        pages = payload.get("pages")
        if not isinstance(pages, list) or not pages:
            raise DocumentParseError("parser response requires page provenance")
        page_numbers = set()
        clean_pages = []
        for item in pages:
            if not isinstance(item, dict) or not isinstance(item.get("page"), int) or isinstance(item.get("page"), bool) or item["page"] < 1 or not isinstance(item.get("text"), str):
                raise DocumentParseError("parser page entries need a positive page number and text")
            if item["page"] in page_numbers:
                raise DocumentParseError("parser response contains duplicate page numbers")
            page_numbers.add(item["page"])
            clean_pages.append({"page": item["page"], "text": item["text"]})
        fields = payload.get("fields", {})
        if not isinstance(fields, dict):
            raise DocumentParseError("parser fields must be an object")
        clean_fields = {}
        for name, item in fields.items():
            if not isinstance(name, str) or not isinstance(item, dict) or not isinstance(item.get("value"), (str, int, float)):
                raise DocumentParseError("each parsed field needs a scalar value")
            confidence, page = item.get("confidence"), item.get("page")
            if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
                raise DocumentParseError("parsed field confidence must be between 0 and 1")
            if not isinstance(page, int) or isinstance(page, bool) or page not in page_numbers:
                raise DocumentParseError("parsed field page must reference a returned page")
            clean_fields[name] = {"value": str(item["value"]), "confidence": float(confidence), "page": page}
        return {"document_type": payload["document_type"], "pages": clean_pages, "fields": clean_fields}


class SimulatedDocumentParser:
    """Fixed fictional OCR fixtures for local demo flow; never reads file contents."""

    FIXTURES = {
        "demo-application.pdf": ("application_form", {
            "applicant_name": ("模拟申请人甲", 0.99), "monthly_income": ("12000 元", 0.99),
        }),
        "demo-application-blurry.pdf": ("application_form", {
            "applicant_name": ("模拟申请人甲", 0.62), "monthly_income": ("12000 元", 0.99),
        }),
        "demo-identity.pdf": ("identity_document", {
            "applicant_name": ("模拟申请人甲", 0.99),
        }),
        "demo-income-proof.pdf": ("income_proof", {
            "monthly_income": ("10000 元", 0.98),
        }),
        "demo-income-proof-matching.pdf": ("income_proof", {
            "monthly_income": ("12000 元", 0.98),
        }),
    }

    def __init__(self, *, environment: str) -> None:
        if environment != "synthetic-demo":
            raise ValueError("simulated document parser requires a synthetic demo index")

    def parse(self, content: bytes, *, filename: str, content_type: str) -> Dict[str, Any]:
        suffix = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if suffix not in PrivateDocumentParser.ALLOWED_TYPES or content_type != PrivateDocumentParser.ALLOWED_TYPES[suffix]:
            raise DocumentParseError("only PDF, PNG, and JPEG uploads with matching content types are supported")
        if not content or len(content) > 15 * 1024 * 1024:
            raise DocumentParseError("uploaded document is empty or exceeds the configured size limit")
        fixture = self.FIXTURES.get(filename)
        if fixture is None:
            raise DocumentParseError("simulated parser only accepts a documented demo filename")
        document_type, fields = fixture
        return {
            "document_type": document_type,
            "pages": [{"page": 1, "text": "虚构模拟 OCR 页面；未读取上传内容。"}],
            "fields": {name: {"value": value, "confidence": confidence, "page": 1}
                       for name, (value, confidence) in fields.items()},
        }
