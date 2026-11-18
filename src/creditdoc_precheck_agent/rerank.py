"""Cohere-compatible cross-encoder reranker for a private endpoint."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Sequence, Set

from .security import is_private_or_loopback, open_without_redirect


class RerankError(RuntimeError):
    """Raised when a private reranker fails or returns invalid candidates."""


class PrivateReranker:
    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: Optional[str] = None,
        allowed_hosts: Optional[Set[str]] = None,
        timeout_seconds: float = 15.0,
    ) -> None:
        parsed = urllib.parse.urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("reranker endpoint must be an absolute HTTP(S) URL")
        allowed = {item.strip().lower() for item in (allowed_hosts or set())}
        if not is_private_or_loopback(parsed.hostname) and parsed.hostname.lower() not in allowed:
            raise ValueError("reranker endpoint host is not private or explicitly allowlisted")
        if parsed.scheme == "http" and not is_private_or_loopback(parsed.hostname):
            raise ValueError("non-private reranker endpoints must use HTTPS")
        if not model.strip():
            raise ValueError("reranker model name is required")
        self.base_url = base_url.rstrip("/")
        self.model_name = model
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    @classmethod
    def from_environment(cls) -> "PrivateReranker":
        base_url = os.environ.get("CREDITDOC_RAG_RERANK_BASE_URL", "")
        model = os.environ.get("CREDITDOC_RAG_RERANK_MODEL", "")
        api_key = os.environ.get("CREDITDOC_RAG_RERANK_API_KEY") or None
        allowed_hosts = set(filter(None, os.environ.get("CREDITDOC_RAG_RERANK_ALLOWED_HOSTS", "").split(",")))
        if not base_url or not model:
            raise ValueError("CREDITDOC_RAG_RERANK_BASE_URL and CREDITDOC_RAG_RERANK_MODEL are required")
        return cls(base_url, model, api_key=api_key, allowed_hosts=allowed_hosts)

    def rerank(self, query: str, candidates: Sequence[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        if not candidates:
            return []
        body = json.dumps({
            "model": self.model_name,
            "query": query,
            "documents": [item["evidence"] for item in candidates],
            "top_n": min(top_k, len(candidates)),
        }, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        request = urllib.request.Request(self.base_url + "/rerank", data=body, headers=headers, method="POST")
        try:
            with open_without_redirect(request, timeout=self.timeout_seconds) as response:
                raw = json.loads(response.read().decode("utf-8"))
            items = raw["results"]
            expected_count = min(top_k, len(candidates))
            if not isinstance(items, list) or len(items) != expected_count:
                raise RerankError("reranker returned an unexpected result count")
            indices = [item.get("index") if isinstance(item, dict) else None for item in items]
            if any(not isinstance(index, int) or isinstance(index, bool) for index in indices):
                raise RerankError("reranker returned a non-integer candidate index")
        except (urllib.error.URLError, TimeoutError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RerankError("private reranker failed or returned invalid results") from exc
        if len(set(indices)) != len(indices) or any(index < 0 or index >= len(candidates) for index in indices):
            raise RerankError("reranker returned an invalid candidate index")
        return [candidates[index] for index in indices[:top_k]]
