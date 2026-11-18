"""Evaluate end-to-end answers and their grounding on a Q-DOC-A JSONL set."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any, Dict, List

from .embedding import OpenAICompatibleEmbeddingProvider
from .generation import OpenAICompatibleGenerator, RAGService, SimulatedAnswerGenerator
from .rag import load_index
from .rerank import PrivateReranker


def evaluate_answers(
    index: Dict[str, Any], qdoca_path: Path, generator: Any,
    embedding_provider: Any = None, reranker: Any = None,
) -> Dict[str, Any]:
    service = RAGService(index, generator, embedding_provider, reranker)
    answerable_correct = 0
    answerable_total = 0
    citation_supported = 0
    citation_total = 0
    expected_sources_found = 0
    expected_sources_total = 0
    relevant_citations = 0
    answerable_citations = 0
    required_terms_found = 0
    required_terms_total = 0
    human_review_count = 0
    rows = []
    for line_number, raw_line in enumerate(qdoca_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            item = json.loads(raw_line)
            expected_answerable = item["answerable"]
            expected_docs = set(item.get("expected_doc_ids", []))
            result = service.answer(
                item["query"], institution=item["institution"], product_line=item["product_line"],
                as_of=date.fromisoformat(item["as_of"]),
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("invalid Q-DOC-A at line %d: %s" % (line_number, exc)) from exc
        if not isinstance(expected_answerable, bool):
            raise ValueError("Q-DOC-A answerable must be a boolean at line %d" % line_number)
        if expected_answerable != bool(expected_docs):
            raise ValueError("expected_doc_ids must match answerability at line %d" % line_number)
        actual_answerable = result["status"] == "evidence_found"
        needs_human_review = result.get("needs_human_review") is True
        human_review_count += int(needs_human_review)
        answerable_correct += int(actual_answerable == expected_answerable)
        answerable_total += 1
        retrieved_ids = {item["citation"]["chunk_id"] for item in result["evidence"]}
        citations = result["citations"]
        if citations:
            citation_supported += sum(1 for citation in citations if citation.get("chunk_id") in retrieved_ids)
            citation_total += len(citations)
        elif not expected_answerable:
            citation_supported += 1
            citation_total += 1
        cited_docs = {citation.get("doc_id") for citation in citations}
        expected_sources_found += len(expected_docs & cited_docs)
        expected_sources_total += len(expected_docs)
        if expected_answerable:
            relevant_citations += sum(1 for citation in citations if citation.get("doc_id") in expected_docs)
            answerable_citations += len(citations)
        terms = item.get("required_terms", [])
        matched_terms = sum(1 for term in terms if term.lower() in result["answer"].lower())
        required_terms_found += matched_terms
        required_terms_total += len(terms)
        rows.append({
            "query_id": item.get("query_id", "line-%d" % line_number),
            "expected_answerable": expected_answerable,
            "actual_status": result["status"],
            "needs_human_review": needs_human_review,
            "answerability_correct": actual_answerable == expected_answerable,
            "expected_docs": sorted(expected_docs),
            "cited_docs": sorted(doc for doc in cited_docs if doc),
            "required_terms_hit": matched_terms,
            "required_terms_total": len(terms),
        })
    return {
        "queries": len(rows),
        "answerability_accuracy": round(answerable_correct / float(answerable_total), 4) if answerable_total else 0.0,
        "citation_grounding_rate": round(citation_supported / float(citation_total), 4) if citation_total else 0.0,
        "expected_source_coverage": round(expected_sources_found / float(expected_sources_total), 4) if expected_sources_total else 1.0,
        "expected_source_precision": round(relevant_citations / float(answerable_citations), 4) if answerable_citations else 0.0,
        "required_term_coverage": round(required_terms_found / float(required_terms_total), 4) if required_terms_total else 1.0,
        "human_review_rate": round(human_review_count / float(answerable_total), 4) if answerable_total else 0.0,
        "items": rows,
        "note": "Synthetic Q-DOC-A results validate the demo flow only; they are not evidence of model quality.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(prog="creditdoc-rag-answer-eval")
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--qdoca", type=Path, required=True)
    parser.add_argument("--generator", choices=("simulated", "private_model"), default="simulated")
    parser.add_argument("--embed", action="store_true")
    parser.add_argument("--rerank", action="store_true")
    args = parser.parse_args()
    index = load_index(args.index)
    if args.generator == "simulated":
        if index.get("environment") != "synthetic-demo":
            parser.error("simulated generator requires a synthetic-demo index")
        generator = SimulatedAnswerGenerator()
    else:
        generator = OpenAICompatibleGenerator.from_environment()
    embedding_provider = OpenAICompatibleEmbeddingProvider.from_environment() if args.embed else None
    if bool(index.get("embedding_model")) != bool(embedding_provider):
        parser.error("use --embed with the same private embedding model used to build the index")
    if embedding_provider and embedding_provider.model_name != index["embedding_model"]:
        parser.error("configured embedding model does not match the index")
    reranker = PrivateReranker.from_environment() if args.rerank else None
    result = evaluate_answers(index, args.qdoca, generator, embedding_provider, reranker)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
