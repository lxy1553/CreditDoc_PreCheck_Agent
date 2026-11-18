import json
import stat

from creditdoc_precheck_agent.audit import LocalJsonlAuditSink


def test_local_audit_sink_appends_jsonl_with_owner_only_mode(tmp_path):
    path = tmp_path / "private" / "audit.jsonl"
    sink = LocalJsonlAuditSink(path)
    sink.write({"event_id": "one", "status": "evidence_found"})
    sink.write({"event_id": "two", "status": "needs_human_review"})

    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [event["event_id"] for event in events] == ["one", "two"]
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
