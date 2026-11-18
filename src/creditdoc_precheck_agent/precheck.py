"""Deterministic checks over OCR-normalized applicant material fields.

This layer reports missing material, low OCR confidence, and field differences.
It never makes an eligibility or credit decision and never sends field values
as RAG queries; only fixed policy questions are retrieved.
"""

from __future__ import annotations

import json
import math
import re
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


class PrecheckInputError(ValueError):
    """Raised when normalized case material does not match the input contract."""


class PrecheckEngine:
    def __init__(self, policy: Dict[str, Any], rag_service: Any) -> None:
        self.policy = policy
        self.rag_service = rag_service

    @classmethod
    def from_file(cls, path: Path, rag_service: Any) -> "PrecheckEngine":
        try:
            policy = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PrecheckInputError("cannot read precheck policy configuration") from exc
        if not isinstance(policy, dict) or not isinstance(policy.get("required_any"), list) or not isinstance(policy.get("comparisons"), list):
            raise PrecheckInputError("precheck policy requires required_any and comparisons lists")
        threshold = policy.get("confidence_threshold")
        if (not isinstance(threshold, (int, float)) or isinstance(threshold, bool)
                or not math.isfinite(threshold) or not 0 <= threshold <= 1):
            raise PrecheckInputError("confidence_threshold must be between 0 and 1")
        for rule in policy["required_any"]:
            if (not isinstance(rule, dict) or not isinstance(rule.get("document_types"), list)
                    or not rule["document_types"]
                    or any(not isinstance(value, str) or not value for value in rule["document_types"])):
                raise PrecheckInputError("each required_any rule needs non-empty document_types")
            for field in ("label", "rule_question"):
                if field in rule and not isinstance(rule[field], str):
                    raise PrecheckInputError("required_any %s must be a string" % field)
        for rule in policy["comparisons"]:
            if (not isinstance(rule, dict) or not isinstance(rule.get("field"), str) or not rule["field"]
                    or not isinstance(rule.get("document_types"), list) or not rule["document_types"]
                    or any(not isinstance(value, str) or not value for value in rule["document_types"])):
                raise PrecheckInputError("each comparison rule needs a field and non-empty document_types")
            for field in ("label", "rule_question"):
                if field in rule and not isinstance(rule[field], str):
                    raise PrecheckInputError("comparison %s must be a string" % field)
            if "numeric_tolerance" in rule:
                tolerance = rule["numeric_tolerance"]
                if (not isinstance(tolerance, (int, float, str)) or isinstance(tolerance, bool)):
                    raise PrecheckInputError("comparison numeric_tolerance must be a finite non-negative number")
                try:
                    numeric_tolerance = Decimal(str(tolerance))
                except InvalidOperation as exc:
                    raise PrecheckInputError("comparison numeric_tolerance must be a finite non-negative number") from exc
                if not numeric_tolerance.is_finite() or numeric_tolerance < 0:
                    raise PrecheckInputError("comparison numeric_tolerance must be a finite non-negative number")
        return cls(policy, rag_service)

    def review(self, case: Dict[str, Any]) -> Dict[str, Any]:
        institution = case.get("institution")
        product_line = case.get("product_line")
        raw_date = case.get("as_of")
        docs = case.get("documents")
        if not isinstance(institution, str) or not institution or not isinstance(product_line, str) or not product_line:
            raise PrecheckInputError("institution and product_line are required")
        if raw_date is None:
            as_of = date.today()
        elif isinstance(raw_date, date):
            as_of = raw_date
        else:
            try:
                as_of = date.fromisoformat(raw_date)
            except (TypeError, ValueError) as exc:
                raise PrecheckInputError("as_of must be an ISO date") from exc
        if not isinstance(docs, list):
            raise PrecheckInputError("documents must be a list")
        documents = self._validate_documents(docs)
        submitted_types = {doc["document_type"] for doc in documents}
        issues: List[Dict[str, Any]] = []

        for rule in self.policy["required_any"]:
            accepted = set(rule.get("document_types", []))
            if not accepted or not any(item in submitted_types for item in accepted):
                issues.append(self._issue(
                    "missing_material", rule.get("label", rule.get("id", "缺少材料")),
                    {"accepted_document_types": sorted(accepted)}, rule.get("rule_question", "贷前预审材料清单要求"),
                    institution, product_line, as_of,
                ))

        threshold = float(self.policy["confidence_threshold"])
        for doc in documents:
            for field_name, field_value in doc["fields"].items():
                confidence = field_value["confidence"]
                if confidence < threshold:
                    issues.append(self._issue(
                        "low_ocr_confidence", "字段识别置信度偏低",
                        {"field": field_name, "document_type": doc["document_type"],
                         "source_ref": field_value.get("source_ref") or doc["source_ref"], "confidence": confidence},
                        "扫描件模糊或关键字段无法可靠识别时应如何处理", institution, product_line, as_of,
                    ))

        for rule in self.policy["comparisons"]:
            field_name = rule.get("field")
            allowed_types = set(rule.get("document_types", []))
            observed = []
            for doc in documents:
                value = doc["fields"].get(field_name)
                if value is not None and doc["document_type"] in allowed_types:
                    observed.append({"document_type": doc["document_type"],
                                     "source_ref": value.get("source_ref") or doc["source_ref"],
                                     "value": value["value"]})
            if len(observed) < 2:
                continue
            baseline = observed[0]["value"]
            differs = any(not self._equivalent(baseline, item["value"], rule) for item in observed[1:])
            if differs:
                details = {"field": field_name, "observations": observed}
                numeric_differences = (self._numeric_differences(observed)
                                       if "numeric_tolerance" in rule else [])
                if numeric_differences:
                    details["numeric_differences"] = numeric_differences
                issues.append(self._issue(
                    "field_difference", rule.get("label", field_name), details,
                    rule.get("rule_question", "材料字段不一致时如何处理"), institution, product_line, as_of,
                ))

        return {
            "status": "needs_human_review",
            "decision": "no_credit_decision",
            "policy_version": self.policy.get("version", "unspecified"),
            "issues": issues,
            "needs_human_review": True,
            "summary": {"issue_count": len(issues), "missing_material_count": sum(i["category"] == "missing_material" for i in issues),
                        "field_difference_count": sum(i["category"] == "field_difference" for i in issues),
                        "low_confidence_count": sum(i["category"] == "low_ocr_confidence" for i in issues)},
        }

    @staticmethod
    def _validate_documents(items: Sequence[Any]) -> List[Dict[str, Any]]:
        if len(items) > 100:
            raise PrecheckInputError("at most 100 documents are allowed per request")
        result = []
        for index, item in enumerate(items):
            if not isinstance(item, dict) or not isinstance(item.get("document_type"), str) or not item.get("document_type"):
                raise PrecheckInputError("documents[%d].document_type is required" % index)
            if not isinstance(item.get("source_ref"), str) or not item["source_ref"]:
                raise PrecheckInputError("documents[%d].source_ref is required" % index)
            fields = item.get("fields", {})
            if not isinstance(fields, dict):
                raise PrecheckInputError("documents[%d].fields must be an object" % index)
            clean_fields = {}
            for name, field_value in fields.items():
                if (not isinstance(name, str) or not isinstance(field_value, dict)
                        or not isinstance(field_value.get("value"), (str, int, float))
                        or isinstance(field_value.get("value"), bool)):
                    raise PrecheckInputError("documents[%d].fields entries need a scalar value" % index)
                confidence = field_value.get("confidence", 1.0)
                if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not 0 <= confidence <= 1:
                    raise PrecheckInputError("field confidence must be between 0 and 1")
                clean_fields[name] = {"value": str(field_value["value"]), "confidence": float(confidence),
                                      "page": field_value.get("page"), "source_ref": field_value.get("source_ref")}
            result.append({"document_type": item["document_type"], "source_ref": item["source_ref"], "fields": clean_fields})
        return result

    @staticmethod
    def _equivalent(left: str, right: str, rule: Dict[str, Any]) -> bool:
        raw_tolerance = rule.get("numeric_tolerance")
        if raw_tolerance is not None:
            tolerance = Decimal(str(raw_tolerance))
            left_number = PrecheckEngine._number(left)
            right_number = PrecheckEngine._number(right)
            if (tolerance.is_finite() and tolerance >= 0
                    and left_number is not None and right_number is not None):
                return abs(left_number - right_number) <= tolerance
        normalize = lambda text: re.sub(r"\s+", "", text).casefold()
        return normalize(left) == normalize(right)

    @staticmethod
    def _numeric_differences(observations: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
        if len(observations) < 2:
            return []
        baseline = PrecheckEngine._number(observations[0]["value"])
        if baseline is None:
            return []
        differences = []
        for item in observations[1:]:
            value = PrecheckEngine._number(item["value"])
            if value is not None and baseline != value:
                differences.append({"left_source_ref": observations[0]["source_ref"],
                                    "right_source_ref": item["source_ref"],
                                    "difference": format(abs(baseline - value).normalize(), "f")})
        return differences

    @staticmethod
    def _number(value: str) -> Optional[Decimal]:
        normalized = re.sub(r"[\s,，元¥￥]", "", value)
        try:
            number = Decimal(normalized)
            return number if number.is_finite() else None
        except InvalidOperation:
            return None

    def _issue(self, category: str, label: str, details: Dict[str, Any], question: str,
               institution: str, product_line: str, as_of: date) -> Dict[str, Any]:
        try:
            evidence = self.rag_service.get_evidence(question, institution=institution,
                                                      product_line=product_line, as_of=as_of, top_k=3)
            evidence_status = "found" if evidence else "not_found"
        except Exception:
            evidence = []
            evidence_status = "unavailable"
        evidence_messages = {
            "not_found": "未检索到适用的规则证据，请人工核对相关制度。",
            "unavailable": "规则检索暂不可用，请人工核对相关制度。",
        }
        return {"category": category, "label": label, "details": details,
                "rule_evidence": evidence, "rule_evidence_status": evidence_status,
                "rule_evidence_message": evidence_messages.get(evidence_status),
                "needs_human_review": True}
