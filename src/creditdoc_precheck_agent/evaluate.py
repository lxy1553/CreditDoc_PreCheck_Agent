"""Evaluate retrieval against an expert- or reviewer-labeled Q-DOC JSONL set."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any, Dict, List

from .embedding import OpenAICompatibleEmbeddingProvider
from .generation import RAGService
from .rag import load_index
from .rerank import PrivateReranker


def evaluate(
    index: Dict[str, Any], qdoc_path: Path, top_k: int = 5,
    embedding_provider: Any = None, reranker: Any = None,
) -> Dict[str, Any]:
    service = RAGService(index, None, embedding_provider, reranker)
    total_expected = 0
    total_found = 0
    reciprocal_ranks: List[float] = []
    rows = []
    no_answer_count = 0
    no_answer_false_positives = 0
    for line_number, raw_line in enumerate(qdoc_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            item = json.loads(raw_line)
            expected = item["expected"]
            if not isinstance(expected, list):
                raise ValueError("expected must be a list")
            answerable = item.get("answerable", bool(expected))
            if answerable and not expected:
                raise ValueError("answerable Q-DOC must have expected evidence")
            if not answerable and expected:
                raise ValueError("no-answer Q-DOC must have an empty expected list")
            results = service.get_evidence(
                item["query"], institution=item["institution"],
                product_line=item["product_line"], as_of=date.fromisoformat(item["as_of"]),
                top_k=top_k,
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("invalid Q-DOC at line %d: %s" % (line_number, exc)) from exc
        expected_keys = {(entry["doc_id"], entry["version"], entry.get("section")) for entry in expected}
        ranked_keys = [
            (result["citation"]["doc_id"], result["citation"]["version"], result["citation"]["section"])
            for result in results
        ]
        hits = [position for position, key in enumerate(ranked_keys, start=1) if key in expected_keys]
        found = len(set(ranked_keys) & expected_keys)
        if answerable:
            total_expected += len(expected_keys)
            total_found += found
            reciprocal_ranks.append(1.0 / hits[0] if hits else 0.0)
        else:
            no_answer_count += 1
            no_answer_false_positives += int(bool(results))
        rows.append({
            "query_id": item.get("query_id", "line-%d" % line_number),
            "expected_count": len(expected_keys),
            "found_count": found,
            "answerable": answerable,
            "reciprocal_rank": round(1.0 / hits[0], 4) if answerable and hits else 0.0,
            "false_positive": bool(results) if not answerable else False,
            "retrieved": [
                {"doc_id": key[0], "version": key[1], "section": key[2]}
                for key in ranked_keys
            ],
        })
    count = sum(1 for row in rows if row["answerable"])
    return {
        "queries": len(rows),
        "answerable_queries": count,
        "top_k": top_k,
        "section_recall_at_k": round(total_found / float(total_expected), 4) if total_expected else 0.0,
        "mean_reciprocal_rank": round(sum(reciprocal_ranks) / count, 4) if count else 0.0,
        "no_answer_queries": no_answer_count,
        "no_answer_false_positive_rate": round(no_answer_false_positives / float(no_answer_count), 4) if no_answer_count else 0.0,
        "items": rows,
        "note": "Synthetic demo scores validate the pipeline only; they are not production quality evidence.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(prog="creditdoc-rag-eval")
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--qdoc", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--embed", action="store_true")
    parser.add_argument("--rerank", action="store_true")
    args = parser.parse_args()
    index = load_index(args.index)
    embedding_provider = OpenAICompatibleEmbeddingProvider.from_environment() if args.embed else None
    if bool(index.get("embedding_model")) != bool(embedding_provider):
        parser.error("use --embed with the same private embedding model used to build the index")
    if embedding_provider and embedding_provider.model_name != index["embedding_model"]:
        parser.error("configured embedding model does not match the index")
    reranker = PrivateReranker.from_environment() if args.rerank else None
    result = evaluate(index, args.qdoc, top_k=args.top_k,
                      embedding_provider=embedding_provider, reranker=reranker)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
