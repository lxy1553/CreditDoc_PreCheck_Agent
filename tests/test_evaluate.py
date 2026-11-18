from pathlib import Path

from creditdoc_precheck_agent.evaluate import evaluate
from creditdoc_precheck_agent.evaluate_answers import evaluate_answers
from creditdoc_precheck_agent.generation import SimulatedAnswerGenerator
from creditdoc_precheck_agent.rag import build_index


ROOT = Path(__file__).resolve().parents[1]


def test_demo_qdoc_evaluation_reports_recall_and_mrr():
    index = build_index(ROOT / "knowledge" / "demo", ROOT / "config" / "manifest.demo.json", allow_demo=True)

    report = evaluate(index, ROOT / "eval" / "qdoc.demo.jsonl", top_k=5)

    assert report["queries"] == 9
    assert report["no_answer_queries"] == 3
    assert report["no_answer_false_positive_rate"] == 0.0
    assert 0.0 <= report["section_recall_at_k"] <= 1.0
    assert 0.0 <= report["mean_reciprocal_rank"] <= 1.0
    assert len(report["items"]) == 9
    by_id = {item["query_id"]: item for item in report["items"]}
    assert by_id["DEMO-Q-006"]["found_count"] == 0
    assert by_id["DEMO-Q-007"]["found_count"] == 0
    assert by_id["DEMO-Q-008"]["found_count"] == 1
    assert by_id["DEMO-Q-009"]["found_count"] == 2


def test_demo_answer_evaluation_checks_abstention_and_citation_grounding():
    index = build_index(ROOT / "knowledge" / "demo", ROOT / "config" / "manifest.demo.json", allow_demo=True)

    report = evaluate_answers(index, ROOT / "eval" / "qdoc-a.demo.jsonl", SimulatedAnswerGenerator())

    assert report["queries"] == 7
    assert report["answerability_accuracy"] == 1.0
    assert report["citation_grounding_rate"] == 1.0
    assert report["expected_source_coverage"] == 1.0
    assert report["expected_source_precision"] == 1.0
    assert report["human_review_rate"] == 1.0
    by_id = {item["query_id"]: item for item in report["items"]}
    assert by_id["DEMO-A-004"]["actual_status"] == "needs_human_review"
    assert by_id["DEMO-A-005"]["actual_status"] == "needs_human_review"
    assert by_id["DEMO-A-006"]["actual_status"] == "evidence_found"
    assert set(by_id["DEMO-A-007"]["cited_docs"]) == {"DEMO-CONFLICT-RULE-A", "DEMO-CONFLICT-RULE-B"}
    assert by_id["DEMO-A-007"]["needs_human_review"] is True
