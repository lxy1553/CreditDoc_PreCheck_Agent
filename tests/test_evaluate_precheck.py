import json
from pathlib import Path

from creditdoc_precheck_agent.evaluate_precheck import evaluate_precheck
from creditdoc_precheck_agent.generation import RAGService, SimulatedAnswerGenerator
from creditdoc_precheck_agent.rag import build_index


ROOT = Path(__file__).resolve().parents[1]


def test_precheck_eval_reports_category_precision_and_recall(tmp_path):
    cases = tmp_path / "cases.jsonl"
    cases.write_text(json.dumps({
        "query_id": "synthetic-1", "institution": "DEMO_ONLY",
        "product_line": "DEMO_PERSONAL_BUSINESS_LOAN", "as_of": "2026-10-03",
        "documents": [{"document_type": "application_form", "source_ref": "fake p1", "fields": {}}],
        "expected_categories": ["missing_material"],
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    index = build_index(ROOT / "knowledge/demo", ROOT / "config/manifest.demo.json", allow_demo=True)
    report = evaluate_precheck(index, ROOT / "config/precheck.demo.json", cases,
                               RAGService(index, SimulatedAnswerGenerator()))
    assert report["cases"] == 1
    assert report["category_metrics"]["missing_material"]["recall"] == 1.0
    assert report["note"].startswith("Synthetic")


def test_precheck_eval_reports_field_level_sample_adequacy_and_split_leakage(tmp_path):
    cases = tmp_path / "cases.jsonl"
    base = {
        "institution": "DEMO_ONLY", "product_line": "DEMO_PERSONAL_BUSINESS_LOAN",
        "as_of": "2026-10-03", "documents": [
            {"document_type": "application_form", "source_ref": "fake p1", "fields": {
                "applicant_name": {"value": "演示甲", "confidence": 0.99}}},
            {"document_type": "identity_document", "source_ref": "fake p2", "fields": {
                "applicant_name": {"value": "演示乙", "confidence": 0.99}}},
            {"document_type": "income_proof", "source_ref": "fake p3", "fields": {
                "monthly_income": {"value": "10000", "confidence": 0.99}}},
        ], "expected_categories": ["field_difference"],
        "expected_differences": ["applicant_name"], "applicant_group_id": "group-1", "split": "dev",
    }
    clean = {**base, "query_id": "clean", "expected_categories": [], "expected_differences": [],
             "applicant_group_id": "group-2", "split": "acceptance",
             "documents": [dict(d, fields={"applicant_name": {"value": "演示甲", "confidence": .99}})
                           for d in base["documents"][:2]]}
    positive = {**base, "query_id": "positive"}
    cases.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in [positive, clean]) + "\n", encoding="utf-8")
    index = build_index(ROOT / "knowledge/demo", ROOT / "config/manifest.demo.json", allow_demo=True)
    report = evaluate_precheck(index, ROOT / "config/precheck.demo.json", cases,
                               RAGService(index, SimulatedAnswerGenerator()))
    field = report["field_metrics"]["applicant_name"]
    assert (field["tp"], field["fp"], field["fn"], field["recall"]) == (1, 0, 0, 1.0)
    assert report["sample_adequacy"]["identity"]["required_n_per_field"] == 817
    assert report["sample_adequacy"]["identity"]["adequate"] is False
    assert report["sample_adequacy"]["identity"]["fields"]["id_number"]["zero_miss_observed"] is False
    assert report["dataset_validation"]["split_leakage_groups"] == []


def test_precheck_eval_detects_applicant_group_split_leakage(tmp_path):
    cases = tmp_path / "cases.jsonl"
    common = {"institution": "DEMO_ONLY", "product_line": "DEMO_PERSONAL_BUSINESS_LOAN",
              "as_of": "2026-10-03", "documents": [], "expected_categories": [],
              "expected_differences": [], "applicant_group_id": "same-applicant"}
    cases.write_text("\n".join(json.dumps({**common, "query_id": name, "split": split})
                           for name, split in [("a", "dev"), ("b", "acceptance")]) + "\n", encoding="utf-8")
    index = build_index(ROOT / "knowledge/demo", ROOT / "config/manifest.demo.json", allow_demo=True)
    report = evaluate_precheck(index, ROOT / "config/precheck.demo.json", cases,
                               RAGService(index, SimulatedAnswerGenerator()))
    assert report["dataset_validation"]["split_leakage_groups"] == ["same-applicant"]


def test_precheck_eval_requires_complete_group_split_metadata_for_validation(tmp_path):
    cases = tmp_path / "cases.jsonl"
    cases.write_text(json.dumps({"query_id": "missing-meta", "institution": "DEMO_ONLY",
        "product_line": "DEMO_PERSONAL_BUSINESS_LOAN", "as_of": "2026-10-03",
        "documents": [], "expected_categories": [], "expected_differences": []}) + "\n", encoding="utf-8")
    index = build_index(ROOT / "knowledge/demo", ROOT / "config/manifest.demo.json", allow_demo=True)
    report = evaluate_precheck(index, ROOT / "config/precheck.demo.json", cases,
                               RAGService(index, SimulatedAnswerGenerator()))
    assert report["dataset_validation"]["group_and_split_coverage"] == 0.0
    assert report["dataset_validation"]["metadata_complete"] is False
