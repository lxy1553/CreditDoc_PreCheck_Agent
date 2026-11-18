"""Command line interface for building and querying a local RAG evidence index."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Optional, Sequence

from .embedding import OpenAICompatibleEmbeddingProvider
from .generation import RAGService
from .rag import RAGInputError, RetrievalUnavailable, build_index, load_index, save_index
from .rerank import PrivateReranker


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="creditdoc-rag")
    commands = parser.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser("ingest", help="build an index from approved Markdown/TXT policies")
    ingest.add_argument("--knowledge-dir", type=Path, required=True)
    ingest.add_argument("--manifest", type=Path, required=True)
    ingest.add_argument("--output", type=Path, default=Path(".local/index.json"))
    ingest.add_argument("--allow-demo", action="store_true", help="explicitly permit a synthetic demo manifest")
    ingest.add_argument("--embed", action="store_true", help="create vectors using the configured private embedding endpoint")
    ingest.add_argument("--dictionary", type=Path, help="approved domain keyword dictionary JSON")
    query = commands.add_parser("query", help="retrieve cited evidence from a local index")
    query.add_argument("--index", type=Path, default=Path(".local/index.json"))
    query.add_argument("--text", required=True)
    query.add_argument("--institution", required=True)
    query.add_argument("--product-line", required=True)
    query.add_argument("--as-of", type=date.fromisoformat, default=date.today())
    query.add_argument("--top-k", type=int, default=5)
    query.add_argument("--embed", action="store_true", help="use the private embedding endpoint for vector retrieval")
    query.add_argument("--rerank", action="store_true", help="use the configured private cross-encoder reranker")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "ingest":
            provider = OpenAICompatibleEmbeddingProvider.from_environment() if args.embed else None
            index = build_index(args.knowledge_dir, args.manifest, allow_demo=args.allow_demo,
                                embedding_provider=provider, keyword_dictionary_path=args.dictionary)
            save_index(index, args.output)
            print(json.dumps({"documents": len(index["documents"]), "chunks": len(index["chunks"]), "index": str(args.output)}, ensure_ascii=False))
            return 0
        index = load_index(args.index)
        provider = OpenAICompatibleEmbeddingProvider.from_environment() if args.embed else None
        if bool(index.get("embedding_model")) != bool(provider):
            raise RAGInputError("use --embed with the same private embedding model used to build the index")
        if provider and provider.model_name != index["embedding_model"]:
            raise RAGInputError("configured embedding model does not match the index")
        reranker = PrivateReranker.from_environment() if args.rerank else None
        service = RAGService(index, None, provider, reranker)
        results = service.get_evidence(args.text, as_of=args.as_of, institution=args.institution,
                                       product_line=args.product_line, top_k=args.top_k)
        print(json.dumps({"query": args.text, "results": results, "needs_human_review": not bool(results)}, ensure_ascii=False, indent=2))
        return 0
    except (RAGInputError, RetrievalUnavailable) as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
