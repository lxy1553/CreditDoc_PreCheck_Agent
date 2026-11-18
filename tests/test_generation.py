from datetime import date
from pathlib import Path

import pytest

from creditdoc_precheck_agent.generation import GenerationError, RAGService, SimulatedAnswerGenerator
from creditdoc_precheck_agent.rag import RAGInputError, RetrievalUnavailable, build_index, retrieve, rewrite_query


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def demo_index():
    return build_index(ROOT / "knowledge" / "demo", ROOT / "config" / "manifest.demo.json", allow_demo=True)


def test_demo_answer_uses_retrieved_source_citations(demo_index):
    service = RAGService(demo_index, SimulatedAnswerGenerator())

    result = service.answer(
        "申请表月收入和收入证明不一致怎么办",
        institution="DEMO_ONLY",
        product_line="DEMO_PERSONAL_BUSINESS_LOAN",
        as_of=date(2026, 10, 3),
    )

    assert result["mode"] == "simulated"
    assert result["needs_human_review"] is True
    assert result["citations"]
    retrieved_ids = {item["citation"]["chunk_id"] for item in result["evidence"]}
    assert all(item["chunk_id"] in retrieved_ids for item in result["citations"])
    assert "不一致" in result["answer"] or "收入" in result["answer"]


def test_demo_answer_abstains_when_no_evidence(demo_index):
    service = RAGService(demo_index, SimulatedAnswerGenerator())

    result = service.answer(
        "火星基地的天气如何",
        institution="DEMO_ONLY",
        product_line="DEMO_PERSONAL_BUSINESS_LOAN",
    )

    assert result["needs_human_review"] is True
    assert result["citations"] == []
    assert result["evidence"] == []
    assert result["answer"]


def test_simulated_generator_refuses_non_demo_index():
    generator = SimulatedAnswerGenerator()

    with pytest.raises(ValueError, match="synthetic demo index"):
        generator.generate("query", [], environment="approved-knowledge")


def test_query_rewriter_splits_compound_precheck_questions():
    rewritten = rewrite_query("扫描模糊时能不能推测缺失字段并自动退回申请？")

    assert rewritten[0] == "扫描模糊时能不能推测缺失字段并自动退回申请？"
    assert "扫描模糊时能不能推测缺失字段" in rewritten
    assert "自动退回申请" in rewritten


def test_query_rewriter_expands_approved_domain_synonyms():
    dictionary = {"terms": [{"standard_term": "收入证明", "synonyms": ["工资证明", "收入材料"]}]}

    rewritten = rewrite_query("工资证明", dictionary)

    assert "收入证明" in " ".join(rewritten)


def test_retrieval_dependency_failure_returns_human_fallback():
    class BrokenEmbeddingProvider:
        def embed_texts(self, texts):
            raise RetrievalUnavailable("offline")

    index = {
        "schema_version": 1,
        "chunks": [{
            "chunk_id": "one", "content": "收入证明要求", "parent_content": "收入证明要求",
            "embedding": [1.0, 0.0], "doc_id": "D", "doc_title": "rule", "doc_version": "1",
            "doc_source": "rule.md", "section": "收入", "line_start": 1, "line_end": 1,
            "status": "approved", "effective_from": "2026-01-01", "effective_to": None,
            "institution": "DEMO_ONLY", "scope": {"product_line": ["DEMO"]},
        }],
    }
    service = RAGService(index, SimulatedAnswerGenerator(), BrokenEmbeddingProvider())

    result = service.answer("收入证明", institution="DEMO_ONLY", product_line="DEMO")

    assert result["status"] == "retrieval_unavailable"
    assert result["needs_human_review"] is True
    assert result["evidence"] == []


def test_service_applies_reranker_to_retrieved_candidates(demo_index):
    class ReverseReranker:
        def rerank(self, query, candidates, top_k):
            return list(reversed(candidates))[:top_k]

    service = RAGService(demo_index, SimulatedAnswerGenerator(), reranker=ReverseReranker())

    results = service.get_evidence(
        "收入证明和申请表金额不一致", institution="DEMO_ONLY",
        product_line="DEMO_PERSONAL_BUSINESS_LOAN", top_k=2,
    )

    assert len(results) == 2
    assert results[0]["citation"]["chunk_id"] != results[1]["citation"]["chunk_id"]


def test_generation_failure_keeps_evidence_and_routes_to_human(demo_index):
    class BrokenGenerator:
        mode = "test"

        def generate(self, query, evidence, *, environment):
            raise GenerationError("offline")

    service = RAGService(demo_index, BrokenGenerator())

    result = service.answer(
        "收入证明月收入与申请表不一致", institution="DEMO_ONLY",
        product_line="DEMO_PERSONAL_BUSINESS_LOAN",
    )

    assert result["status"] == "generator_unavailable"
    assert result["evidence"]
    assert result["needs_human_review"] is True


def test_retrieval_filters_wrong_institution_product_and_effective_date(demo_index):
    query = "收入证明材料"

    assert retrieve(demo_index, query, institution="OTHER", product_line="DEMO_PERSONAL_BUSINESS_LOAN") == []
    assert retrieve(demo_index, query, institution="DEMO_ONLY", product_line="OTHER") == []
    assert retrieve(demo_index, query, institution="DEMO_ONLY", product_line="DEMO_PERSONAL_BUSINESS_LOAN",
                    as_of=date(2025, 12, 31)) == []


def test_synthetic_manifest_requires_explicit_demo_flag():
    with pytest.raises(RAGInputError, match="--allow-demo"):
        build_index(ROOT / "knowledge" / "demo", ROOT / "config" / "manifest.demo.json")
