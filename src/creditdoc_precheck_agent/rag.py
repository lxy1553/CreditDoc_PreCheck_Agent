"""Offline ingestion and local lexical retrieval for approved policy documents.

This first slice deliberately uses a dependency-free Chinese character BM25
baseline. The module returns evidence bundles; it does not make credit decisions
or send document content to a model endpoint.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


INDEX_SCHEMA_VERSION = 1
PARENT_CHARS = 1000
CHILD_CHARS = 320
CHILD_OVERLAP = 50
SUPPORTED_SUFFIXES = {".md", ".txt"}
DOMAIN_STOPWORDS = {"要求", "如何", "哪些", "是否", "应该", "可以", "需要", "怎么", "能不", "不能", "以及", "并且"}


class RAGInputError(ValueError):
    """Raised when a knowledge source or index manifest is invalid."""


class RetrievalUnavailable(RuntimeError):
    """Raised when a configured retrieval dependency is unavailable."""


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _parse_date(value: Optional[str], field: str, required: bool = False) -> Optional[date]:
    if value is None and not required:
        return None
    if not isinstance(value, str):
        raise RAGInputError("%s must be an ISO date string" % field)
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise RAGInputError("%s must use YYYY-MM-DD" % field) from exc


def _validate_document(record: Dict[str, Any]) -> Dict[str, Any]:
    required = ("doc_id", "source", "title", "version", "status", "effective_from", "institution", "scope")
    missing = [key for key in required if not record.get(key)]
    if missing:
        raise RAGInputError("document is missing required fields: %s" % ", ".join(missing))
    for key in ("doc_id", "source", "title", "version", "status", "institution"):
        if not isinstance(record[key], str):
            raise RAGInputError("%s must be a string" % key)
    if record["status"] != "approved":
        raise RAGInputError("only documents with status='approved' may be indexed: %s" % record["doc_id"])
    if not isinstance(record["scope"], dict) or not record["scope"].get("product_line"):
        raise RAGInputError("scope.product_line is required for %s" % record["doc_id"])
    if isinstance(record["scope"]["product_line"], str):
        record["scope"]["product_line"] = [record["scope"]["product_line"]]
    if not isinstance(record["scope"]["product_line"], list):
        raise RAGInputError("scope.product_line must be a string or list for %s" % record["doc_id"])
    if not all(isinstance(item, str) and item for item in record["scope"]["product_line"]):
        raise RAGInputError("scope.product_line entries must be non-empty strings for %s" % record["doc_id"])
    start = _parse_date(record["effective_from"], "effective_from", required=True)
    end = _parse_date(record.get("effective_to"), "effective_to")
    if end and start and end < start:
        raise RAGInputError("effective_to precedes effective_from for %s" % record["doc_id"])
    return record


def _load_keyword_dictionary(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {"version": "empty", "terms": []}
    try:
        dictionary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RAGInputError("cannot read keyword dictionary: %s" % exc) from exc
    if not isinstance(dictionary, dict) or not isinstance(dictionary.get("terms"), list):
        raise RAGInputError("keyword dictionary must contain a terms list")
    for item in dictionary["terms"]:
        if not isinstance(item, dict) or not isinstance(item.get("standard_term"), str) or not item["standard_term"]:
            raise RAGInputError("each keyword dictionary term requires a standard_term")
        for key in ("synonyms", "abbreviations", "colloquial", "common_typos"):
            values = item.get(key, [])
            if not isinstance(values, list) or not all(isinstance(value, str) and value for value in values):
                raise RAGInputError("keyword dictionary %s must be a list of non-empty strings" % key)
    return dictionary


def _read_sections(text: str) -> List[Tuple[str, str, int, int]]:
    """Return (section path, text, first line, last line), preserving headings."""
    sections: List[Tuple[str, str, int, int]] = []
    heading_stack: List[Tuple[int, str]] = []
    current_lines: List[str] = []
    current_start = 1
    current_path = "Document"

    def flush() -> None:
        nonlocal current_lines
        first = 0
        last = len(current_lines)
        while first < last and not current_lines[first].strip():
            first += 1
        while last > first and not current_lines[last - 1].strip():
            last -= 1
        body_lines = current_lines[first:last]
        body = "\n".join(body_lines)
        if body:
            sections.append((current_path, body, current_start + first, current_start + last - 1))
        current_lines = []

    lines = text.splitlines()
    for line_no, line in enumerate(lines, start=1):
        match = re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", line)
        if match:
            flush()
            level, title = len(match.group(1)), match.group(2).strip()
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
            current_path = " > ".join(item[1] for item in heading_stack)
            current_start = line_no
            current_lines = [line]
        else:
            if not current_lines:
                current_start = line_no
            current_lines.append(line)
    flush()
    return sections


def _windows(text: str, max_chars: int, overlap: int) -> Iterable[Tuple[int, int, str]]:
    if not text:
        return
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        if end < len(text):
            boundary = max(text.rfind("\n", start, end), text.rfind("。", start, end), text.rfind("；", start, end))
            if boundary > start + max_chars // 2:
                end = boundary + 1
        snippet = text[start:end].strip()
        if snippet:
            left_trim = len(text[start:end]) - len(text[start:end].lstrip())
            right_trim = len(text[start:end].rstrip())
            yield start + left_trim, start + right_trim, snippet
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)


def _build_chunks(record: Dict[str, Any], text: str, source_ref: str) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    source_hash = _sha256(text)
    for section_no, (section_path, section_text, first_line, _last_line) in enumerate(_read_sections(text)):
        for parent_no, (parent_start, parent_end, parent_text) in enumerate(
            _windows(section_text, PARENT_CHARS, CHILD_OVERLAP), start=1
        ):
            parent_id = _sha256("%s|%s|%d|%d" % (record["doc_id"], record["version"], section_no, parent_no))[:24]
            for child_no, (child_start, child_end, child_text) in enumerate(
                _windows(parent_text, CHILD_CHARS, CHILD_OVERLAP), start=1
            ):
                absolute_start = parent_start + child_start
                absolute_end = parent_start + child_end
                line_start = first_line + section_text.count("\n", 0, absolute_start)
                line_end = first_line + section_text.count("\n", 0, max(absolute_start, absolute_end - 1))
                chunk_id = _sha256("%s|%s|%d|%d|%d" % (parent_id, source_hash, parent_no, child_no, absolute_start))[:24]
                chunks.append({
                    "chunk_id": chunk_id,
                    "parent_id": parent_id,
                    "doc_id": record["doc_id"],
                    "doc_title": record["title"],
                    "doc_version": record["version"],
                    "doc_source": source_ref,
                    "doc_hash": source_hash,
                    "status": record["status"],
                    "effective_from": record["effective_from"],
                    "effective_to": record.get("effective_to"),
                    "institution": record["institution"],
                    "scope": record["scope"],
                    "section": section_path,
                    "line_start": line_start,
                    "line_end": max(line_start, line_end),
                    "content": child_text,
                    "parent_content": parent_text,
                })
    return chunks


def build_index(
    knowledge_dir: Path, manifest_path: Path, *, allow_demo: bool = False,
    embedding_provider: Any = None, keyword_dictionary_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Build an index from an explicit manifest of approved Markdown/TXT docs."""
    root = knowledge_dir.resolve()
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RAGInputError("cannot read manifest: %s" % exc) from exc
    records = manifest.get("documents") if isinstance(manifest, dict) else None
    if not isinstance(records, list) or not records:
        raise RAGInputError("manifest must contain a non-empty documents list")
    if manifest.get("environment") == "synthetic-demo" and not allow_demo:
        raise RAGInputError("synthetic demo data requires the explicit --allow-demo option")

    documents: List[Dict[str, Any]] = []
    chunks: List[Dict[str, Any]] = []
    seen_ids = set()
    for original in records:
        if not isinstance(original, dict):
            raise RAGInputError("each manifest document must be an object")
        record = _validate_document(dict(original))
        version_key = (record["doc_id"], record["version"])
        if version_key in seen_ids:
            raise RAGInputError("duplicate document version: %s@%s" % version_key)
        seen_ids.add(version_key)
        source = (root / record["source"]).resolve()
        if root not in source.parents:
            raise RAGInputError("source path escapes knowledge directory: %s" % record["source"])
        if source.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise RAGInputError("only Markdown and TXT sources are supported in this build: %s" % source.name)
        if not source.is_file():
            raise RAGInputError("source file does not exist: %s" % source)
        text = source.read_text(encoding="utf-8").strip()
        if not text:
            raise RAGInputError("source file is empty: %s" % source.name)
        start_date = date.fromisoformat(record["effective_from"])
        end_date = date.fromisoformat(record["effective_to"]) if record.get("effective_to") else None
        documents.append({
            "doc_id": record["doc_id"], "title": record["title"], "version": record["version"],
            "source": record["source"], "source_hash": _sha256(text), "status": record["status"],
            "effective_from": start_date.isoformat(), "effective_to": end_date.isoformat() if end_date else None,
            "institution": record["institution"], "scope": record["scope"],
        })
        chunks.extend(_build_chunks(record, text, record["source"]))
    if not chunks:
        raise RAGInputError("no searchable content found in manifest sources")
    embedding_model = None
    if embedding_provider is not None:
        try:
            vectors = embedding_provider.embed_texts([chunk["content"] for chunk in chunks])
        except Exception as exc:
            raise RAGInputError("embedding index build failed: %s" % exc) from exc
        if len(vectors) != len(chunks) or not vectors or not vectors[0]:
            raise RAGInputError("embedding provider returned an unexpected vector count or dimension")
        if len({len(vector) for vector in vectors}) != 1:
            raise RAGInputError("embedding provider returned inconsistent vector dimensions")
        for chunk, vector in zip(chunks, vectors):
            chunk["embedding"] = vector
        embedding_model = getattr(embedding_provider, "model_name", "unknown")
    keyword_dictionary = _load_keyword_dictionary(keyword_dictionary_path)
    return {
        "schema_version": INDEX_SCHEMA_VERSION,
        "environment": manifest.get("environment", "approved-knowledge"),
        "embedding_model": embedding_model,
        "keyword_dictionary": keyword_dictionary,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "documents": documents,
        "chunks": chunks,
    }


def save_index(index: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temp_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(output_path)


def tokenize(text: str) -> List[str]:
    """Tokenize Chinese as character bigrams and preserve Latin/numeric terms."""
    normalized = text.lower()
    tokens: List[str] = []
    for part in re.findall(r"[a-z0-9][a-z0-9._/-]*|[\u3400-\u9fff]+", normalized):
        if re.match(r"[a-z0-9]", part):
            tokens.append(part)
        elif len(part) == 1:
            tokens.append(part)
        else:
            tokens.extend(part[i:i + 2] for i in range(len(part) - 1))
    return [token for token in tokens if token not in DOMAIN_STOPWORDS]


def rewrite_query(query: str, keyword_dictionary: Optional[Dict[str, Any]] = None) -> List[str]:
    """Keep the original query and add focused variants for explicit conjunctions."""
    variants = [query.strip()]
    parts = re.split(r"\s*(?:并且|并|同时|以及|；|;)\s*", query.strip())
    for part in parts:
        cleaned = part.strip(" ，,。！？!?；;\t\n")
        if len(cleaned) >= 3 and cleaned not in variants:
            variants.append(cleaned)
    for item in (keyword_dictionary or {}).get("terms", []):
        standard = item["standard_term"]
        aliases = item.get("synonyms", []) + item.get("abbreviations", []) + item.get("colloquial", []) + item.get("common_typos", [])
        if standard in query or any(alias in query for alias in aliases):
            expanded = query + " " + standard + " " + " ".join(aliases)
            if expanded not in variants:
                variants.append(expanded)
        if len(variants) >= 12:
            break
    return variants


def _bm25_scores(query_tokens: Sequence[str], documents: Sequence[Sequence[str]]) -> List[float]:
    if not documents:
        return []
    doc_len = [len(tokens) for tokens in documents]
    avg_len = sum(doc_len) / float(len(doc_len)) or 1.0
    term_freq = [Counter(tokens) for tokens in documents]
    doc_freq: Dict[str, int] = defaultdict(int)
    for token in set(query_tokens):
        doc_freq[token] = sum(1 for frequencies in term_freq if frequencies.get(token, 0) > 0)
    scores = [0.0] * len(documents)
    k1, b = 1.5, 0.75
    for token in query_tokens:
        df = doc_freq[token]
        if not df:
            continue
        idf = math.log(1.0 + (len(documents) - df + 0.5) / (df + 0.5))
        for idx, frequencies in enumerate(term_freq):
            tf = frequencies.get(token, 0)
            if tf:
                denom = tf + k1 * (1.0 - b + b * doc_len[idx] / avg_len)
                scores[idx] += idf * tf * (k1 + 1.0) / denom
    return scores


def _matches_scope(chunk: Dict[str, Any], institution: Optional[str], product_line: Optional[str]) -> bool:
    if institution and chunk.get("institution") != institution:
        return False
    if product_line:
        lines = chunk.get("scope", {}).get("product_line", [])
        if "*" not in lines and product_line not in lines:
            return False
    return True


def retrieve(
    index: Dict[str, Any], query: str, *, as_of: Optional[date] = None,
    institution: Optional[str] = None, product_line: Optional[str] = None, top_k: int = 5,
    embedding_provider: Any = None,
) -> List[Dict[str, Any]]:
    if index.get("schema_version") != INDEX_SCHEMA_VERSION:
        raise RAGInputError("unsupported index schema version")
    if not query.strip():
        raise RAGInputError("query must not be empty")
    if not institution or not product_line:
        raise RAGInputError("institution and product_line are required retrieval filters")
    if top_k < 1:
        raise RAGInputError("top_k must be positive")
    day = as_of or date.today()
    candidates = []
    for chunk in index.get("chunks", []):
        start = date.fromisoformat(chunk["effective_from"])
        end = date.fromisoformat(chunk["effective_to"]) if chunk.get("effective_to") else None
        if chunk.get("status") != "approved" or start > day or (end and end < day):
            continue
        if not _matches_scope(chunk, institution, product_line):
            continue
        candidates.append(chunk)
    query_tokens = tokenize(query)
    if not query_tokens or not candidates:
        return []
    tokenized_candidates = [tokenize(chunk["content"]) for chunk in candidates]
    fused_scores: Dict[str, float] = defaultdict(float)
    raw_scores: Dict[str, float] = defaultdict(float)
    chunks_by_id = {chunk["chunk_id"]: chunk for chunk in candidates}
    for rewritten in rewrite_query(query, index.get("keyword_dictionary")):
        rewritten_tokens = tokenize(rewritten)
        if not rewritten_tokens:
            continue
        scores = _bm25_scores(rewritten_tokens, tokenized_candidates)
        ranking = sorted(zip(scores, candidates), key=lambda item: (-item[0], item[1]["chunk_id"]))
        rank = 0
        for score, chunk in ranking:
            if score <= 0:
                continue
            rank += 1
            fused_scores[chunk["chunk_id"]] += 1.0 / (60.0 + rank)
            raw_scores[chunk["chunk_id"]] = max(raw_scores[chunk["chunk_id"]], score)
    if embedding_provider is not None and candidates and all(chunk.get("embedding") for chunk in candidates):
        try:
            query_vector = embedding_provider.embed_texts([query])[0]
        except Exception as exc:
            raise RetrievalUnavailable("query embedding service is unavailable") from exc
        dense_scores = []
        query_norm = math.sqrt(sum(float(value) ** 2 for value in query_vector))
        for chunk in candidates:
            vector = chunk["embedding"]
            if len(vector) != len(query_vector):
                raise RAGInputError("query and index embedding dimensions do not match")
            vector_norm = math.sqrt(sum(float(value) ** 2 for value in vector))
            dot = sum(float(left) * float(right) for left, right in zip(query_vector, vector))
            cosine = dot / (query_norm * vector_norm) if query_norm and vector_norm else 0.0
            dense_scores.append((cosine, chunk))
        for rank, (score, chunk) in enumerate(
            sorted(dense_scores, key=lambda item: (-item[0], item[1]["chunk_id"])), start=1
        ):
            if score > 0:
                fused_scores[chunk["chunk_id"]] += 1.0 / (60.0 + rank)
                raw_scores[chunk["chunk_id"]] = max(raw_scores[chunk["chunk_id"]], score)
    ranked = sorted(
        ((fused_scores[chunk_id], raw_scores[chunk_id], chunk) for chunk_id, chunk in chunks_by_id.items() if fused_scores[chunk_id] > 0),
        key=lambda item: (-item[0], -item[1], item[2]["chunk_id"]),
    )
    results = []
    for fused_score, _raw_score, chunk in ranked:
        results.append({
            "score": round(fused_score, 6),
            "evidence": chunk["content"],
            "context": chunk["parent_content"],
            "citation": {
                "doc_id": chunk["doc_id"], "title": chunk["doc_title"], "version": chunk["doc_version"],
                "source": chunk["doc_source"], "section": chunk["section"],
                "lines": [chunk["line_start"], chunk["line_end"]], "chunk_id": chunk["chunk_id"],
            },
        })
        if len(results) >= top_k:
            break
    return results


def load_index(path: Path) -> Dict[str, Any]:
    try:
        index = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RAGInputError("cannot read index: %s" % exc) from exc
    if not isinstance(index, dict) or index.get("schema_version") != INDEX_SCHEMA_VERSION:
        raise RAGInputError("invalid or unsupported index")
    return index
