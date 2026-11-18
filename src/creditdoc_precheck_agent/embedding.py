"""OpenAI-compatible embeddings for a private model endpoint."""

from __future__ import annotations

import json
import math
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import List, Optional, Sequence, Set

from .generation import GenerationError
from .security import is_private_or_loopback, open_without_redirect


class OpenAICompatibleEmbeddingProvider:
    """Creates document/query vectors using an allowlisted private endpoint."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: Optional[str] = None,
        allowed_hosts: Optional[Set[str]] = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        parsed = urllib.parse.urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("embedding endpoint must be an absolute HTTP(S) URL")
        allowed = {item.strip().lower() for item in (allowed_hosts or set())}
        if not is_private_or_loopback(parsed.hostname) and parsed.hostname.lower() not in allowed:
            raise ValueError("embedding endpoint host is not private or explicitly allowlisted")
        if parsed.scheme == "http" and not is_private_or_loopback(parsed.hostname):
            raise ValueError("non-private embedding endpoints must use HTTPS")
        if not model.strip():
            raise ValueError("embedding model name is required")
        self.base_url = base_url.rstrip("/")
        self.model_name = model
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    @classmethod
    def from_environment(cls) -> "OpenAICompatibleEmbeddingProvider":
        base_url = os.environ.get("CREDITDOC_RAG_EMBEDDING_BASE_URL", "")
        model = os.environ.get("CREDITDOC_RAG_EMBEDDING_MODEL", "")
        api_key = os.environ.get("CREDITDOC_RAG_EMBEDDING_API_KEY")
        allowed_hosts = set(filter(None, os.environ.get("CREDITDOC_RAG_EMBEDDING_ALLOWED_HOSTS", "").split(",")))
        if not base_url or not model:
            raise ValueError("CREDITDOC_RAG_EMBEDDING_BASE_URL and CREDITDOC_RAG_EMBEDDING_MODEL are required")
        return cls(base_url, model, api_key=api_key, allowed_hosts=allowed_hosts)

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        vectors: List[List[float]] = []
        for start in range(0, len(texts), 64):
            batch = list(texts[start:start + 64])
            body = json.dumps({"model": self.model_name, "input": batch}, ensure_ascii=False).encode("utf-8")
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = "Bearer " + self.api_key
            request = urllib.request.Request(self.base_url + "/embeddings", data=body, headers=headers, method="POST")
            try:
                with open_without_redirect(request, timeout=self.timeout_seconds) as response:
                    raw = json.loads(response.read().decode("utf-8"))
                items = raw["data"]
                if not isinstance(items, list) or len(items) != len(batch):
                    raise GenerationError("embedding service returned an unexpected vector count")
                indices = [item.get("index") if isinstance(item, dict) else None for item in items]
                if (any(not isinstance(index, int) or isinstance(index, bool) for index in indices)
                        or sorted(indices) != list(range(len(batch)))):
                    raise GenerationError("embedding service returned invalid item indices")
                items = sorted(items, key=lambda item: item["index"])
                batch_vectors = [[float(value) for value in item["embedding"]] for item in items]
            except (urllib.error.URLError, TimeoutError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise GenerationError("private embedding call failed or returned invalid vectors") from exc
            if len(batch_vectors) != len(batch) or not batch_vectors or not batch_vectors[0]:
                raise GenerationError("embedding service returned an unexpected vector count or dimension")
            if len({len(vector) for vector in batch_vectors}) != 1:
                raise GenerationError("embedding service returned inconsistent dimensions")
            if any(not math.isfinite(value) for vector in batch_vectors for value in vector):
                raise GenerationError("embedding service returned a non-finite vector value")
            vectors.extend(batch_vectors)
        return vectors
