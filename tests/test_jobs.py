import threading

import pytest

from creditdoc_precheck_agent.jobs import PrecheckJobCoordinator, SQLiteJobStore


def create(store, *, now=1000, key="idem-1"):
    return store.create(institution="BANK_A", product_line="LOAN_A", subject="svc-reviewer",
                        idempotency_key=key, now=now)


def test_queued_job_past_queue_cutoff_is_disposed_to_manual_review(tmp_path):
    store = SQLiteJobStore(tmp_path / "jobs.sqlite")
    job = create(store)
    claimed = store.claim(job["job_id"], now=1091, queue_timeout_seconds=90, deadline_seconds=180)
    assert claimed is None
    state = store.get(job["job_id"])
    assert state["status"] == "manual_review"
    assert state["reason"] == "queue_timeout"
    assert state["timely_disposition"] is True
    assert state["agent_result_slo_met"] is False


def test_late_worker_result_cannot_overwrite_manual_review(tmp_path):
    store = SQLiteJobStore(tmp_path / "jobs.sqlite")
    job = create(store)
    assert store.claim(job["job_id"], now=1010, queue_timeout_seconds=90, deadline_seconds=180)
    store.expire(now=1181, queue_timeout_seconds=90, deadline_seconds=180)
    assert store.finish(job["job_id"], now=1190, deadline_seconds=180) is False
    state = store.get(job["job_id"])
    assert state["status"] == "manual_review"
    assert state["reason"] == "hard_deadline"
    assert state["agent_result_slo_met"] is False


def test_job_at_hard_deadline_is_transferred_before_late_result(tmp_path):
    store = SQLiteJobStore(tmp_path / "jobs.sqlite")
    job = create(store)
    assert store.claim(job["job_id"], now=1010, queue_timeout_seconds=90, deadline_seconds=180)

    assert store.expire(now=1180, queue_timeout_seconds=90, deadline_seconds=180) == [job["job_id"]]
    assert store.finish(job["job_id"], now=1180, deadline_seconds=180) is False
    state = store.get(job["job_id"])
    assert state["status"] == "manual_review"
    assert state["reason"] == "hard_deadline"
    assert state["agent_result_slo_met"] is False


def test_worker_finishing_at_hard_deadline_is_transferred_to_manual_review(tmp_path):
    store = SQLiteJobStore(tmp_path / "jobs.sqlite")
    job = create(store)
    assert store.claim(job["job_id"], now=1010, queue_timeout_seconds=90, deadline_seconds=180)

    assert store.finish(job["job_id"], now=1180, deadline_seconds=180) is False
    state = store.get(job["job_id"])
    assert state["status"] == "manual_review"
    assert state["reason"] == "hard_deadline"
    assert state["timely_disposition"] is True
    assert state["agent_result_slo_met"] is False


def test_completed_agent_result_is_separate_from_disposition_slo(tmp_path):
    store = SQLiteJobStore(tmp_path / "jobs.sqlite")
    job = create(store)
    assert store.claim(job["job_id"], now=1070, queue_timeout_seconds=90, deadline_seconds=180)
    assert store.finish(job["job_id"], now=1160, deadline_seconds=180)
    state = store.get(job["job_id"])
    assert state["status"] == "completed"
    assert state["timely_disposition"] is True
    assert state["agent_result_slo_met"] is True


def test_idempotency_key_returns_existing_job(tmp_path):
    store = SQLiteJobStore(tmp_path / "jobs.sqlite")
    first = create(store)
    second = create(store, now=1010)
    assert second["job_id"] == first["job_id"]
    assert second["created"] is False


def test_idempotency_key_rejects_different_request_payload(tmp_path):
    from creditdoc_precheck_agent.jobs import IdempotencyConflict

    coordinator = PrecheckJobCoordinator(SQLiteJobStore(tmp_path / "jobs.sqlite"), lambda payload: payload)
    try:
        coordinator.submit({"amount": 100}, institution="BANK_A", product_line="LOAN_A",
                           subject="svc-reviewer", idempotency_key="same-key")
        with pytest.raises(IdempotencyConflict):
            coordinator.submit({"amount": 200}, institution="BANK_A", product_line="LOAN_A",
                               subject="svc-reviewer", idempotency_key="same-key")
    finally:
        coordinator.close()


def test_concurrent_duplicate_submissions_create_one_job_and_one_queue_event(tmp_path):
    store = SQLiteJobStore(tmp_path / "jobs.sqlite")
    start = threading.Barrier(12)
    results = []

    def submit_duplicate():
        start.wait()
        results.append(create(store))

    threads = [threading.Thread(target=submit_duplicate) for _ in range(12)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(2)

    assert all(not thread.is_alive() for thread in threads)
    assert len({item["job_id"] for item in results}) == 1
    assert sorted(item["created"] for item in results) == [False] * 11 + [True]
    assert [event["status"] for event in store.events(results[0]["job_id"])] == ["queued"]


@pytest.mark.parametrize(("queue_timeout", "deadline", "poll_interval"), [
    (0, 180, 0.1), (90, 0, 0.1), (90, 180, 0),
])
def test_coordinator_rejects_non_positive_timeouts(tmp_path, queue_timeout, deadline, poll_interval):
    with pytest.raises(ValueError, match="positive"):
        PrecheckJobCoordinator(SQLiteJobStore(tmp_path / "jobs.sqlite"), lambda payload: payload,
                               queue_timeout_seconds=queue_timeout, deadline_seconds=deadline,
                               poll_interval_seconds=poll_interval)


def test_tenant_cannot_read_another_job(tmp_path):
    store = SQLiteJobStore(tmp_path / "jobs.sqlite")
    job = create(store)
    assert store.get_for_scope(job["job_id"], institution="BANK_B", product_line="LOAN_A", subject="svc-reviewer") is None


def test_recent_jobs_are_ordered_and_scoped_to_allowed_products_and_subject(tmp_path):
    store = SQLiteJobStore(tmp_path / "jobs.sqlite")
    older = store.create(institution="BANK_A", product_line="LOAN_A", subject="reviewer-a",
                         idempotency_key="older", now=1000)
    store.create(institution="BANK_A", product_line="LOAN_B", subject="reviewer-a",
                 idempotency_key="other-product", now=1002)
    store.create(institution="BANK_A", product_line="LOAN_A", subject="reviewer-b",
                 idempotency_key="other-user", now=1003)
    newer = store.create(institution="BANK_A", product_line="LOAN_A", subject="reviewer-a",
                         idempotency_key="newer", now=1001)

    jobs = store.list_for_scope(institution="BANK_A", product_lines=["LOAN_A"],
                                subject="reviewer-a", limit=10)

    assert [item["job_id"] for item in jobs] == [newer["job_id"], older["job_id"]]
    assert all("idempotency_hash" not in item for item in jobs)


def test_coordinator_processes_job_and_exposes_result_to_owner(tmp_path):
    import time
    from creditdoc_precheck_agent.jobs import PrecheckJobCoordinator

    coordinator = PrecheckJobCoordinator(SQLiteJobStore(tmp_path / "jobs.sqlite"),
                                          lambda payload: {"safe_demo_result": payload["value"]},
                                          poll_interval_seconds=0.01)
    try:
        submitted = coordinator.submit({"value": "demo"}, institution="BANK_A", product_line="LOAN_A",
                                       subject="svc-reviewer", idempotency_key="job-1")
        deadline = time.time() + 1
        result = submitted
        while time.time() < deadline:
            result = coordinator.get(submitted["job_id"], institution="BANK_A", product_line="LOAN_A",
                                     subject="svc-reviewer")
            if result["status"] == "completed":
                break
            time.sleep(0.01)
        assert result["status"] == "completed"
        assert result["result"] == {"safe_demo_result": "demo"}
        assert coordinator.get(submitted["job_id"], institution="BANK_B", product_line="LOAN_A",
                               subject="svc-reviewer") is None
    finally:
        coordinator.close()


def test_coordinator_discards_result_that_finishes_after_deadline(tmp_path):
    import threading
    import time
    from creditdoc_precheck_agent.jobs import PrecheckJobCoordinator

    started = threading.Event()
    release = threading.Event()
    def processor(payload):
        started.set()
        release.wait(1)
        return {"late": True}
    coordinator = PrecheckJobCoordinator(SQLiteJobStore(tmp_path / "jobs.sqlite"), processor,
                                          deadline_seconds=0.15, poll_interval_seconds=0.01)
    try:
        submitted = coordinator.submit({"value": 1}, institution="BANK_A", product_line="LOAN_A",
                                       subject="svc-reviewer", idempotency_key="late-job")
        assert started.wait(1)
        deadline = time.time() + 1
        state = None
        while time.time() < deadline:
            state = coordinator.get(submitted["job_id"], institution="BANK_A", product_line="LOAN_A",
                                    subject="svc-reviewer")
            if state["status"] == "manual_review":
                break
            time.sleep(0.01)
        assert state["status"] == "manual_review"
        release.set()
        time.sleep(0.05)
        state = coordinator.get(submitted["job_id"], institution="BANK_A", product_line="LOAN_A",
                                subject="svc-reviewer")
        assert state["status"] == "manual_review"
        assert state["reason"] == "hard_deadline"
        assert "result" not in state
    finally:
        release.set()
        coordinator.close()


def test_slo_summary_is_scoped_to_institution_product_and_subject(tmp_path):
    store = SQLiteJobStore(tmp_path / "jobs.sqlite")
    job = create(store)
    assert store.claim(job["job_id"], now=1010, queue_timeout_seconds=90, deadline_seconds=180)
    assert store.finish(job["job_id"], now=1160, deadline_seconds=180)
    own = store.slo_summary(institution="BANK_A", product_lines=["LOAN_A"], subject="svc-reviewer")
    other = store.slo_summary(institution="BANK_B", product_lines=["LOAN_A"], subject="svc-reviewer")
    assert own["agent_result_slo_rate"] == 1.0
    assert own["disposition_latency_p50_seconds"] == 160.0
    assert own["agent_result_latency_p95_seconds"] == 160.0
    assert other["total_jobs"] == 0


def test_job_database_has_owner_only_permissions(tmp_path):
    import stat
    store = SQLiteJobStore(tmp_path / "private" / "jobs.sqlite")
    assert stat.S_IMODE(store.path.stat().st_mode) == 0o600


def test_completed_job_restart_keeps_original_timing_metrics(tmp_path):
    store = SQLiteJobStore(tmp_path / "jobs.sqlite")
    job = create(store)
    assert store.claim(job["job_id"], now=1010, queue_timeout_seconds=90, deadline_seconds=180)
    assert store.finish(job["job_id"], now=1160, deadline_seconds=180)

    assert store.recover_after_restart(now=1400, deadline_seconds=180) == [job["job_id"]]
    state = store.get(job["job_id"])
    assert state["status"] == "manual_review"
    assert state["reason"] == "result_unavailable_after_restart"
    assert state["timely_disposition"] is True
    assert state["disposition_at"] == 1160
    assert state["agent_result_slo_met"] is True
    assert state["agent_result_at"] == 1160


def test_sqlite_connections_are_closed_after_store_operations(tmp_path, monkeypatch):
    import sqlite3
    from creditdoc_precheck_agent import jobs as jobs_module

    store = SQLiteJobStore(tmp_path / "jobs.sqlite")
    real_connect = sqlite3.connect
    opened = []

    class TrackedConnection(sqlite3.Connection):
        def close(self):
            self.was_closed = True
            super().close()

    def tracked_connect(*args, **kwargs):
        kwargs["factory"] = TrackedConnection
        connection = real_connect(*args, **kwargs)
        connection.was_closed = False
        opened.append(connection)
        return connection

    monkeypatch.setattr(jobs_module.sqlite3, "connect", tracked_connect)
    store.get("missing-job")

    assert opened
    assert all(connection.was_closed for connection in opened)


def test_get_never_observes_completed_job_before_result_is_available(tmp_path):
    import threading
    import time
    from creditdoc_precheck_agent.jobs import PrecheckJobCoordinator

    store = SQLiteJobStore(tmp_path / "jobs.sqlite")
    original_finish = store.finish
    persisted = threading.Event()
    release_finish = threading.Event()

    def gated_finish(*args, **kwargs):
        completed = original_finish(*args, **kwargs)
        if completed:
            persisted.set()
            release_finish.wait(5)
        return completed

    store.finish = gated_finish
    coordinator = PrecheckJobCoordinator(store, lambda payload: {"value": payload["value"]},
                                          poll_interval_seconds=0.01)
    try:
        submitted = coordinator.submit({"value": "result"}, institution="BANK_A", product_line="LOAN_A",
                                       subject="svc-reviewer", idempotency_key="atomic-result")
        assert persisted.wait(1)
        observed = {}
        read_done = threading.Event()

        def read_job():
            observed.update(coordinator.get(submitted["job_id"], institution="BANK_A",
                                            product_line="LOAN_A", subject="svc-reviewer"))
            read_done.set()

        reader = threading.Thread(target=read_job)
        reader.start()
        time.sleep(0.03)
        assert not read_done.is_set()
        release_finish.set()
        assert read_done.wait(1)
        reader.join(1)
        assert observed["status"] == "completed"
        assert observed["result"] == {"value": "result"}
    finally:
        release_finish.set()
        coordinator.close()


def test_state_transitions_have_durable_metadata_only_event_history(tmp_path):
    store = SQLiteJobStore(tmp_path / "jobs.sqlite")
    job = create(store)
    assert store.claim(job["job_id"], now=1010, queue_timeout_seconds=90, deadline_seconds=180)
    assert store.finish(job["job_id"], now=1160, deadline_seconds=180)
    events = store.events(job["job_id"])
    assert [event["status"] for event in events] == ["queued", "running", "completed"]
    assert all("payload" not in event and "result" not in event for event in events)


def test_restart_routes_memory_only_results_to_manual_without_rewriting_result_metric(tmp_path):
    store = SQLiteJobStore(tmp_path / "jobs.sqlite")
    job = create(store)
    assert store.claim(job["job_id"], now=1010, queue_timeout_seconds=90, deadline_seconds=180)
    assert store.finish(job["job_id"], now=1160, deadline_seconds=180)
    assert store.recover_after_restart(now=1170, deadline_seconds=180) == [job["job_id"]]
    state = store.get(job["job_id"])
    assert state["status"] == "manual_review"
    assert state["reason"] == "result_unavailable_after_restart"
    assert state["agent_result_slo_met"] is True
    assert state["timely_disposition"] is True


def test_p95_latency_uses_nearest_rank_for_small_samples(tmp_path):
    store = SQLiteJobStore(tmp_path / "jobs.sqlite")
    for index, latency in enumerate((10, 20, 30, 40)):
        job = create(store, now=1000, key="p95-%d" % index)
        assert store.claim(job["job_id"], now=1001, queue_timeout_seconds=90, deadline_seconds=180)
        assert store.finish(job["job_id"], now=1000 + latency, deadline_seconds=180)
    summary = store.slo_summary(institution="BANK_A", product_lines=["LOAN_A"], subject="svc-reviewer")
    assert summary["disposition_latency_p95_seconds"] == 40.0


def test_backlog_cap_routes_new_work_directly_to_manual_review(tmp_path):
    import threading
    import time
    from creditdoc_precheck_agent.jobs import PrecheckJobCoordinator

    started = threading.Event()
    release = threading.Event()
    def processor(payload):
        started.set()
        release.wait(1)
        return {"ok": True}
    coordinator = PrecheckJobCoordinator(SQLiteJobStore(tmp_path / "jobs.sqlite"), processor,
                                          workers=1, max_queue_size=1, poll_interval_seconds=0.01)
    try:
        first = coordinator.submit({"n": 1}, institution="BANK_A", product_line="LOAN_A",
                                   subject="svc-reviewer", idempotency_key="first")
        assert started.wait(1)
        second = coordinator.submit({"n": 2}, institution="BANK_A", product_line="LOAN_A",
                                    subject="svc-reviewer", idempotency_key="second")
        third = coordinator.submit({"n": 3}, institution="BANK_A", product_line="LOAN_A",
                                   subject="svc-reviewer", idempotency_key="third")
        assert third["status"] == "manual_review"
        assert third["reason"] == "backlog_limit"
        assert third["timely_disposition"] is True
        release.set()
    finally:
        release.set()
        coordinator.close()
