from datetime import date
import json
from io import BytesIO
from unittest.mock import patch

import pytest

from creditdoc_precheck_agent.embedding import OpenAICompatibleEmbeddingProvider
from creditdoc_precheck_agent.generation import GenerationError, OpenAICompatibleGenerator
from creditdoc_precheck_agent.rag import retrieve
from creditdoc_precheck_agent.rerank import PrivateReranker, RerankError


class FixedEmbeddingProvider:
    model_name = "fixed-test-vector"

    def embed_texts(self, texts):
        return [[0.0, 1.0] if text == "query" or "vector match" in text else [1.0, 0.0] for text in texts]


def test_optional_embedding_channel_can_retrieve_semantic_match_without_token_overlap():
    index = {
        "schema_version": 1,
        "chunks": [
            {
                "chunk_id": "lexical", "content": "nothing relevant", "parent_content": "nothing relevant",
                "embedding": [1.0, 0.0], "doc_id": "LEX", "doc_title": "lexical", "doc_version": "1",
                "doc_source": "lex.md", "section": "lex", "line_start": 1, "line_end": 1,
                "status": "approved", "effective_from": "2026-01-01", "effective_to": None,
                "institution": "DEMO_ONLY", "scope": {"product_line": ["DEMO"]},
            },
            {
                "chunk_id": "semantic", "content": "vector match", "parent_content": "vector match",
                "embedding": [0.0, 1.0], "doc_id": "SEM", "doc_title": "semantic", "doc_version": "1",
                "doc_source": "sem.md", "section": "sem", "line_start": 1, "line_end": 1,
                "status": "approved", "effective_from": "2026-01-01", "effective_to": None,
                "institution": "DEMO_ONLY", "scope": {"product_line": ["DEMO"]},
            },
        ],
    }

    results = retrieve(index, "query", institution="DEMO_ONLY", product_line="DEMO",
                       as_of=date(2026, 10, 3), embedding_provider=FixedEmbeddingProvider())

    assert results[0]["citation"]["doc_id"] == "SEM"


@pytest.mark.parametrize("adapter", [
    lambda: OpenAICompatibleEmbeddingProvider("https://public.example.com/v1", "embed"),
    lambda: OpenAICompatibleGenerator("https://public.example.com/v1", "chat"),
    lambda: PrivateReranker("https://public.example.com/v1", "rerank"),
])
def test_model_adapters_reject_unallowlisted_public_hosts(adapter):
    with pytest.raises(ValueError, match="private or explicitly allowlisted"):
        adapter()


def test_embedding_provider_rejects_duplicate_response_indices():
    payload = {"data": [
        {"index": 0, "embedding": [1.0, 0.0]},
        {"index": 0, "embedding": [0.0, 1.0]},
    ]}
    provider = OpenAICompatibleEmbeddingProvider("http://127.0.0.1:9999", "embed")
    with patch("creditdoc_precheck_agent.embedding.open_without_redirect",
               return_value=BytesIO(json.dumps(payload).encode("utf-8"))):
        with pytest.raises(GenerationError, match="indices"):
            provider.embed_texts(["first", "second"])


@pytest.mark.parametrize("results", [
    [{"index": 0.5}],
    [{"index": True}],
    [{"index": 0}],
])
def test_reranker_rejects_invalid_or_incomplete_candidate_indices(results):
    candidates = [{"evidence": "first"}, {"evidence": "second"}]
    reranker = PrivateReranker("http://127.0.0.1:9999", "rerank")
    with patch("creditdoc_precheck_agent.rerank.open_without_redirect",
               return_value=BytesIO(json.dumps({"results": results}).encode("utf-8"))):
        with pytest.raises(RerankError, match="index|count"):
            reranker.rerank("query", candidates, 2)
