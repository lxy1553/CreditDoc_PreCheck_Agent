"""Evaluate deterministic issue detection on JSONL synthetic/reviewer cases."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Set

from .generation import RAGService, SimulatedAnswerGenerator
from .precheck import PrecheckEngine
from .rag import load_index

ISSUE_CATEGORIES = ("missing_material", "field_difference", "low_ocr_confidence")
FIELD_CLASS_TARGETS = {"identity": 0.995, "financial": 0.98, "auxiliary": 0.95}
FIELD_CLASSES = {
    "identity": ("applicant_name", "id_number", "id_expiry"),
    "financial": ("monthly_income", "annual_income", "loan_amount"),
    "auxiliary": ("address", "phone_number"),
}


def _zero_miss_sample_size(target: float, alpha: float) -> int:
    """Minimum positive examples for a one-sided exact binomial zero-miss bound."""
    return math.ceil(math.log(alpha) / math.log(target))


def evaluate_precheck(index: Dict[str, Any], policy_path: Path, cases_path: Path, service: RAGService) -> Dict[str, Any]:
    engine = PrecheckEngine.from_file(policy_path, service)
    counts = {category: {"tp": 0, "fp": 0, "fn": 0} for category in ISSUE_CATEGORIES}
    configured_fields = {item["field"] for item in engine.policy.get("comparisons", [])}
    field_counts = {field: {"tp": 0, "fp": 0, "fn": 0} for field in configured_fields}
    items = []
    group_splits: Dict[str, Set[str]] = {}
    grouped_cases = 0
    for line_number, line in enumerate(cases_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            case = json.loads(line)
            expected: Set[str] = set(case["expected_categories"])
            if not expected <= set(ISSUE_CATEGORIES):
                raise ValueError("unknown expected issue category")
            expected_fields = set(case.get("expected_differences", []))
            if not expected_fields <= configured_fields:
                raise ValueError("unknown expected difference field")
            group = case.get("applicant_group_id")
            split = case.get("split")
            if group is not None and split is not None:
                group_splits.setdefault(str(group), set()).add(str(split))
                grouped_cases += 1
            result = engine.review(case)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid precheck evaluation case at line %d: %s" % (line_number, exc)) from exc
        actual = {issue["category"] for issue in result["issues"]}
        actual_fields = {issue.get("details", {}).get("field") for issue in result["issues"]
                         if issue["category"] == "field_difference"}
        for category in ISSUE_CATEGORIES:
            counts[category]["tp"] += int(category in expected and category in actual)
            counts[category]["fp"] += int(category not in expected and category in actual)
            counts[category]["fn"] += int(category in expected and category not in actual)
        items.append({"query_id": case.get("query_id", "line-%d" % line_number),
                      "expected_categories": sorted(expected), "actual_categories": sorted(actual),
                      "expected_differences": sorted(expected_fields),
                      "actual_differences": sorted(actual_fields - {None}),
                      "exact_match": expected == actual})
        for field in configured_fields:
            field_counts[field]["tp"] += int(field in expected_fields and field in actual_fields)
            field_counts[field]["fp"] += int(field not in expected_fields and field in actual_fields)
            field_counts[field]["fn"] += int(field in expected_fields and field not in actual_fields)
    metrics = {}
    for category, values in counts.items():
        precision = values["tp"] / float(values["tp"] + values["fp"]) if values["tp"] + values["fp"] else 1.0
        recall = values["tp"] / float(values["tp"] + values["fn"]) if values["tp"] + values["fn"] else 1.0
        metrics[category] = {**values, "precision": round(precision, 4), "recall": round(recall, 4)}
    total_tp = sum(value["tp"] for value in counts.values())
    total_fp = sum(value["fp"] for value in counts.values())
    total_fn = sum(value["fn"] for value in counts.values())
    field_metrics = {}
    for field, values in sorted(field_counts.items()):
        predicted = values["tp"] + values["fp"]
        positives = values["tp"] + values["fn"]
        field_metrics[field] = {**values,
                                "precision": round(values["tp"] / predicted, 4) if predicted else None,
                                "recall": round(values["tp"] / positives, 4) if positives else None,
                                "positive_examples": positives}
    sample_adequacy = {}
    critical_fields = FIELD_CLASSES["identity"]
    alpha_per_critical_field = 0.05 / len(critical_fields)
    for risk_class, fields in FIELD_CLASSES.items():
        target = FIELD_CLASS_TARGETS[risk_class]
        alpha = alpha_per_critical_field if risk_class == "identity" else 0.05
        required_n = _zero_miss_sample_size(target, alpha)
        per_field = {field: {"positive_examples": field_metrics.get(field, {}).get("positive_examples", 0),
                             "required_n": required_n,
                             "zero_miss_observed": (field_metrics.get(field, {}).get("positive_examples", 0) > 0
                                                     and field_metrics.get(field, {}).get("fn", 0) == 0),
                             "adequate": field_metrics.get(field, {}).get("positive_examples", 0) >= required_n}
                     for field in fields}
        sample_adequacy[risk_class] = {"target_recall": target, "confidence": 0.95,
                                       "required_n_per_field": required_n, "fields": per_field,
                                       "adequate": all(value["adequate"] for value in per_field.values())}
    split_leakage = sorted(group for group, splits in group_splits.items() if len(splits) > 1)
    return {
        "cases": len(items),
        "exact_category_match_rate": round(sum(item["exact_match"] for item in items) / float(len(items)), 4) if items else 0.0,
        "micro_precision": round(total_tp / float(total_tp + total_fp), 4) if total_tp + total_fp else 1.0,
        "micro_recall": round(total_tp / float(total_tp + total_fn), 4) if total_tp + total_fn else 1.0,
        "category_metrics": metrics,
        "field_metrics": field_metrics,
        "sample_adequacy": sample_adequacy,
        "dataset_validation": {"split_leakage_groups": split_leakage,
                               "cases_with_group_and_split": grouped_cases,
                               "group_and_split_coverage": round(grouped_cases / float(len(items)), 4) if items else 0.0,
                               "metadata_complete": grouped_cases == len(items),
                               "leakage_free": grouped_cases == len(items) and not split_leakage},
        "items": items,
        "note": "Synthetic precheck metrics validate demo plumbing only. Small samples are insufficient for acceptance; use expert-reviewed, applicant-group-disjoint cases and meet each field-class sample target before making quality claims.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(prog="creditdoc-precheck-eval")
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    args = parser.parse_args()
    index = load_index(args.index)
    if index.get("environment") != "synthetic-demo":
        parser.error("this evaluator currently uses the demo-only simulated generator")
    result = evaluate_precheck(index, args.policy, args.cases, RAGService(index, SimulatedAnswerGenerator()))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
