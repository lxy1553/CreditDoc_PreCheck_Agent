from pathlib import Path

import pytest

from creditdoc_precheck_agent.generation import RAGService, SimulatedAnswerGenerator
from creditdoc_precheck_agent.precheck import PrecheckEngine
from creditdoc_precheck_agent.rag import build_index


ROOT = Path(__file__).resolve().parents[1]


def engine():
    index = build_index(ROOT / "knowledge/demo", ROOT / "config/manifest.demo.json", allow_demo=True,
                        keyword_dictionary_path=ROOT / "config/keyword_dict.demo.json")
    service = RAGService(index, SimulatedAnswerGenerator())
    return PrecheckEngine.from_file(ROOT / "config/precheck.demo.json", service)


def test_precheck_finds_missing_material_and_cross_document_difference():
    result = engine().review({
        "institution": "DEMO_ONLY",
        "product_line": "DEMO_PERSONAL_BUSINESS_LOAN",
        "as_of": "2026-10-03",
        "documents": [
            {"document_type": "application_form", "source_ref": "申请表第2页", "fields": {
                "applicant_name": {"value": "张某", "confidence": 0.99},
                "monthly_income": {"value": "12000", "confidence": 0.98},
            }},
            {"document_type": "identity_document", "source_ref": "身份证正面", "fields": {
                "applicant_name": {"value": "张某某", "confidence": 0.78},
            }},
        ],
    })

    assert result["status"] == "needs_human_review"
    assert result["decision"] == "no_credit_decision"
    assert any(item["category"] == "missing_material" for item in result["issues"])
    assert any(item["category"] == "field_difference" for item in result["issues"])
    assert any(item["category"] == "low_ocr_confidence" for item in result["issues"])
    assert all("rule_evidence" in item for item in result["issues"])



def test_precheck_does_not_flag_matching_normalized_values():
    result = engine().review({
        "institution": "DEMO_ONLY", "product_line": "DEMO_PERSONAL_BUSINESS_LOAN",
        "as_of": "2026-10-03",
        "documents": [
            {"document_type": "application_form", "source_ref": "申请表第2页", "fields": {
                "monthly_income": {"value": "12,000.00 元", "confidence": 0.99},
            }},
            {"document_type": "income_proof", "source_ref": "收入证明第1页", "fields": {
                "monthly_income": {"value": "12000 元", "confidence": 0.99},
            }},
        ],
    })
    assert not any(item["category"] == "field_difference" for item in result["issues"])



def test_precheck_reports_numeric_income_difference():
    result = engine().review({
        "institution": "DEMO_ONLY", "product_line": "DEMO_PERSONAL_BUSINESS_LOAN",
        "as_of": "2026-10-03", "documents": [
            {"document_type": "application_form", "source_ref": "申请表第2页", "fields": {
                "monthly_income": {"value": "12,000 元", "confidence": 0.99},
            }},
            {"document_type": "income_proof", "source_ref": "收入证明第1页", "fields": {
                "monthly_income": {"value": "10,000 元", "confidence": 0.99},
            }},
        ],
    })
    issue = next(item for item in result["issues"] if item["category"] == "field_difference")
    assert issue["details"]["numeric_differences"][0]["difference"] == "2000"


def test_identifier_comparison_preserves_leading_zeroes():
    result = engine().review({
        "institution": "DEMO_ONLY", "product_line": "DEMO_PERSONAL_BUSINESS_LOAN",
        "as_of": "2026-10-03", "documents": [
            {"document_type": "application_form", "source_ref": "申请表第1页", "fields": {
                "id_number": {"value": "001234", "confidence": 0.99},
            }},
            {"document_type": "identity_document", "source_ref": "身份证第1页", "fields": {
                "id_number": {"value": "1235", "confidence": 0.99},
            }},
        ],
    })

    issue = next(item for item in result["issues"]
                 if item["category"] == "field_difference" and item["details"]["field"] == "id_number")
    assert issue["details"]["observations"][0]["value"] == "001234"
    assert issue["details"]["observations"][1]["value"] == "1235"
    assert "numeric_differences" not in issue["details"]


def test_non_finite_numeric_value_is_reported_for_human_review_without_crashing():
    result = engine().review({
        "institution": "DEMO_ONLY", "product_line": "DEMO_PERSONAL_BUSINESS_LOAN",
        "as_of": "2026-10-03", "documents": [
            {"document_type": "application_form", "source_ref": "申请表第1页", "fields": {
                "monthly_income": {"value": "NaN", "confidence": 0.99},
            }},
            {"document_type": "income_proof", "source_ref": "收入证明第1页", "fields": {
                "monthly_income": {"value": "10000 元", "confidence": 0.99},
            }},
        ],
    })

    issue = next(item for item in result["issues"]
                 if item["category"] == "field_difference" and item["details"]["field"] == "monthly_income")
    assert issue["needs_human_review"] is True
    assert "numeric_differences" not in issue["details"]


def test_precheck_marks_missing_rule_evidence_for_human_review():
    class NoRuleEvidence:
        def get_evidence(self, *args, **kwargs):
            return []

    configured = engine()
    no_evidence_engine = PrecheckEngine(configured.policy, NoRuleEvidence())
    result = no_evidence_engine.review({"institution": "DEMO_ONLY", "product_line": "DEMO_PERSONAL_BUSINESS_LOAN",
                                        "as_of": "2026-10-03", "documents": []})
    assert result["needs_human_review"] is True
    assert all(issue["rule_evidence_status"] == "not_found" for issue in result["issues"])
    assert all("人工" in issue["rule_evidence_message"] for issue in result["issues"])


def test_precheck_rejects_boolean_as_ocr_confidence():
    from creditdoc_precheck_agent.precheck import PrecheckInputError
    with pytest.raises(PrecheckInputError, match="confidence"):
        engine().review({"institution": "DEMO_ONLY", "product_line": "DEMO_PERSONAL_BUSINESS_LOAN",
                         "documents": [{"document_type": "application_form", "source_ref": "fake", "fields": {
                             "monthly_income": {"value": "12000", "confidence": True},
                         }}]})


def test_precheck_policy_rejects_boolean_confidence_threshold(tmp_path):
    import json
    from creditdoc_precheck_agent.precheck import PrecheckInputError

    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps({"confidence_threshold": True,
                                      "required_any": [], "comparisons": []}), encoding="utf-8")
    with pytest.raises(PrecheckInputError, match="confidence_threshold"):
        PrecheckEngine.from_file(policy_path, None)


def test_precheck_policy_rejects_invalid_comparison_rule_at_load_time(tmp_path):
    import json
    from creditdoc_precheck_agent.precheck import PrecheckInputError

    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps({"confidence_threshold": 0.85,
                                      "required_any": [],
                                      "comparisons": [{"field": "monthly_income",
                                                       "document_types": ["income_proof"],
                                                       "numeric_tolerance": True}]}), encoding="utf-8")
    with pytest.raises(PrecheckInputError, match="numeric_tolerance"):
        PrecheckEngine.from_file(policy_path, None)
