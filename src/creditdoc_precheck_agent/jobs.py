"""SQLite metadata state machine for asynchronous precheck work.

Applicant payloads and results are deliberately excluded from SQLite. The demo
coordinator retains them in process memory; production deployments must wire an
approved encrypted case store before expecting results to survive restarts.
"""

from __future__ import annotations

import hashlib
import base64
import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


class IdempotencyConflict(ValueError):
    """An idempotency key was reused for a different request body."""


class SQLiteJobStore:
    TERMINAL = {"completed", "manual_review"}

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(str(self.path.parent), 0o700)
        self._initialize()
        os.chmod(str(self.path), 0o600)
        for suffix in ("-wal", "-shm"):
            sidecar = Path(str(self.path) + suffix)
            if sidecar.exists():
                os.chmod(str(sidecar), 0o600)

    def _connect(self):
        connection = sqlite3.connect(str(self.path), timeout=10, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=10000")
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA synchronous=FULL")
            db.execute("""CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                institution TEXT NOT NULL,
                product_line TEXT NOT NULL,
                subject TEXT NOT NULL,
                idempotency_hash TEXT NOT NULL,
                request_hash TEXT,
                status TEXT NOT NULL,
                reason TEXT,
                created_at REAL NOT NULL,
                started_at REAL,
                disposition_at REAL,
                agent_result_at REAL,
                timely_disposition INTEGER,
                agent_result_slo_met INTEGER,
                UNIQUE(institution, product_line, subject, idempotency_hash)
            )""")
            columns = {row["name"] for row in db.execute("PRAGMA table_info(jobs)")}
            if "request_hash" not in columns:
                db.execute("ALTER TABLE jobs ADD COLUMN request_hash TEXT")
            db.execute("""CREATE TABLE IF NOT EXISTS job_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL,
                status TEXT NOT NULL,
                reason TEXT,
                occurred_at REAL NOT NULL
            )""")
            db.execute("CREATE TRIGGER IF NOT EXISTS job_events_no_update BEFORE UPDATE ON job_events BEGIN SELECT RAISE(ABORT, 'job events are append-only'); END")
            db.execute("CREATE TRIGGER IF NOT EXISTS job_events_no_delete BEFORE DELETE ON job_events BEGIN SELECT RAISE(ABORT, 'job events are append-only'); END")

    @staticmethod
    def _add_event(db, job_id: str, status: str, reason: Optional[str], now: float) -> None:
        db.execute("INSERT INTO job_events(job_id,status,reason,occurred_at) VALUES(?,?,?,?)",
                   (job_id, status, reason, now))

    @staticmethod
    def _key_hash(key: str) -> str:
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    @staticmethod
    def _record(row: Optional[sqlite3.Row]) -> Optional[Dict[str, Any]]:
        if row is None:
            return None
        record = {key: row[key] for key in row.keys() if key not in {"idempotency_hash", "request_hash"}}
        for key in ("timely_disposition", "agent_result_slo_met"):
            if record[key] is not None:
                record[key] = bool(record[key])
        return record

    def create(self, *, institution: str, product_line: str, subject: str,
               idempotency_key: str, now: float, request_hash: Optional[str] = None) -> Dict[str, Any]:
        if not idempotency_key or len(idempotency_key) > 128:
            raise ValueError("idempotency key must contain 1 to 128 characters")
        digest = self._key_hash(idempotency_key)
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM jobs WHERE institution=? AND product_line=? AND subject=? AND idempotency_hash=?",
                             (institution, product_line, subject, digest)).fetchone()
            if row:
                if request_hash is not None and row["request_hash"] is not None and row["request_hash"] != request_hash:
                    db.execute("COMMIT")
                    raise IdempotencyConflict("idempotency key was already used with a different request")
                record = self._record(row)
                record["created"] = False
                db.execute("COMMIT")
                return record
            job_id = str(uuid.uuid4())
            db.execute("INSERT INTO jobs(job_id,institution,product_line,subject,idempotency_hash,request_hash,status,created_at) VALUES(?,?,?,?,?,?,?,?)",
                       (job_id, institution, product_line, subject, digest, request_hash, "queued", now))
            self._add_event(db, job_id, "queued", None, now)
            row = db.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            db.execute("COMMIT")
            record = self._record(row)
            record["created"] = True
            return record

    def claim(self, job_id: str, *, now: float, queue_timeout_seconds: float,
              deadline_seconds: float) -> Optional[Dict[str, Any]]:
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if row is None or row["status"] != "queued":
                db.execute("COMMIT")
                return None
            elapsed = now - row["created_at"]
            if elapsed >= deadline_seconds or elapsed >= queue_timeout_seconds:
                reason = "hard_deadline" if elapsed >= deadline_seconds else "queue_timeout"
                self._manual_update(db, job_id, reason, now, elapsed <= deadline_seconds)
                db.execute("COMMIT")
                return None
            db.execute("UPDATE jobs SET status='running',started_at=? WHERE job_id=? AND status='queued'", (now, job_id))
            self._add_event(db, job_id, "running", None, now)
            result = db.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            db.execute("COMMIT")
            return self._record(result)

    @staticmethod
    def _manual_update(db, job_id: str, reason: str, now: float, timely: bool) -> None:
        db.execute("""UPDATE jobs SET status='manual_review', reason=?, disposition_at=?,
                   timely_disposition=?, agent_result_slo_met=0
                   WHERE job_id=? AND status IN ('queued','running')""",
                   (reason, now, int(timely), job_id))
        SQLiteJobStore._add_event(db, job_id, "manual_review", reason, now)

    def expire(self, *, now: float, queue_timeout_seconds: float, deadline_seconds: float) -> List[str]:
        disposed: List[str] = []
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            rows = db.execute("SELECT job_id,status,created_at FROM jobs WHERE status IN ('queued','running')").fetchall()
            for row in rows:
                elapsed = now - row["created_at"]
                reason = None
                if elapsed >= deadline_seconds:
                    reason = "hard_deadline"
                elif row["status"] == "queued" and elapsed >= queue_timeout_seconds:
                    reason = "queue_timeout"
                if reason:
                    self._manual_update(db, row["job_id"], reason, now, elapsed <= deadline_seconds)
                    disposed.append(row["job_id"])
            db.execute("COMMIT")
        return disposed

    def finish(self, job_id: str, *, now: float, deadline_seconds: float) -> bool:
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT status,created_at FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if row is None or row["status"] != "running":
                db.execute("COMMIT")
                return False
            elapsed = now - row["created_at"]
            if elapsed >= deadline_seconds:
                self._manual_update(db, job_id, "hard_deadline", now, elapsed <= deadline_seconds)
                db.execute("COMMIT")
                return False
            db.execute("UPDATE jobs SET status='completed', disposition_at=?,agent_result_at=?,timely_disposition=1,agent_result_slo_met=1 WHERE job_id=? AND status='running'",
                       (now, now, job_id))
            self._add_event(db, job_id, "completed", None, now)
            db.execute("COMMIT")
            return True

    def manual_review(self, job_id: str, *, reason: str, now: float, deadline_seconds: float) -> bool:
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT created_at,status FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if row is None or row["status"] not in {"queued", "running"}:
                db.execute("COMMIT")
                return False
            self._manual_update(db, job_id, reason, now, now - row["created_at"] <= deadline_seconds)
            db.execute("COMMIT")
            return True

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._connection() as db:
            return self._record(db.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone())

    def events(self, job_id: str) -> List[Dict[str, Any]]:
        with self._connection() as db:
            rows = db.execute("SELECT event_id,job_id,status,reason,occurred_at FROM job_events WHERE job_id=? ORDER BY event_id",
                              (job_id,)).fetchall()
        return [dict(row) for row in rows]

    def get_for_scope(self, job_id: str, *, institution: str, product_line: str,
                      subject: str) -> Optional[Dict[str, Any]]:
        with self._connection() as db:
            row = db.execute("SELECT * FROM jobs WHERE job_id=? AND institution=? AND product_line=? AND subject=?",
                             (job_id, institution, product_line, subject)).fetchone()
            return self._record(row)

    def list_for_scope(self, *, institution: str, product_lines: Any,
                       subject: str, limit: int = 20) -> List[Dict[str, Any]]:
        allowed = list(dict.fromkeys(product_lines))
        if not allowed:
            return []
        if not 1 <= limit <= 50:
            raise ValueError("history limit must be between 1 and 50")
        placeholders = ",".join("?" for _ in allowed)
        query = "SELECT * FROM jobs WHERE institution=? AND subject=? AND product_line IN (%s) ORDER BY created_at DESC, rowid DESC LIMIT ?" % placeholders
        with self._connection() as db:
            rows = db.execute(query, [institution, subject] + allowed + [limit]).fetchall()
        return [self._record(row) for row in rows]

    def recover_after_restart(self, *, now: float, deadline_seconds: float) -> List[str]:
        """Route orphaned jobs to a human; never replay payloads that were memory-only."""
        recovered: List[str] = []
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            rows = db.execute("SELECT job_id,created_at,status FROM jobs WHERE status IN ('queued','running','completed')").fetchall()
            for row in rows:
                elapsed = now - row["created_at"]
                reason = "result_unavailable_after_restart" if row["status"] == "completed" else "worker_restart"
                if row["status"] == "completed":
                    db.execute("UPDATE jobs SET status='manual_review',reason=? WHERE job_id=?",
                               (reason, row["job_id"]))
                else:
                    db.execute("UPDATE jobs SET status='manual_review',reason=?,disposition_at=?,timely_disposition=?,agent_result_slo_met=0 WHERE job_id=?",
                               (reason, now, int(elapsed <= deadline_seconds), row["job_id"]))
                self._add_event(db, row["job_id"], "manual_review", reason, now)
                recovered.append(row["job_id"])
            db.execute("COMMIT")
        return recovered


    def slo_summary(self, *, institution: str, product_lines: Any, subject: str) -> Dict[str, Any]:
        allowed = list(product_lines)
        if not allowed:
            return {"total_jobs": 0, "disposed_jobs": 0, "timely_disposition_count": 0,
                    "timely_agent_result_count": 0, "manual_review_count": 0,
                    "timely_disposition_rate": 0.0, "agent_result_slo_rate": 0.0}
        placeholders = ",".join("?" for _ in allowed)
        query = """SELECT COUNT(*) AS total,
                SUM(CASE WHEN status IN ('completed','manual_review') THEN 1 ELSE 0 END) AS disposed,
                SUM(CASE WHEN timely_disposition=1 THEN 1 ELSE 0 END) AS timely_dispositions,
                SUM(CASE WHEN agent_result_slo_met=1 THEN 1 ELSE 0 END) AS timely_agent_results,
                SUM(CASE WHEN status='manual_review' THEN 1 ELSE 0 END) AS manual_reviews
                FROM jobs WHERE institution=? AND subject=? AND product_line IN (%s)""" % placeholders
        with self._connection() as db:
            row = db.execute(query, [institution, subject] + allowed).fetchone()
            detail_query = "SELECT status,created_at,started_at,disposition_at,agent_result_at FROM jobs WHERE institution=? AND subject=? AND product_line IN (%s)" % placeholders
            detail_rows = db.execute(detail_query, [institution, subject] + allowed).fetchall()
        total, disposed, timely, agent, manual = (int(value or 0) for value in row)
        now = __import__("time").time()
        disposition_latencies = [item["disposition_at"] - item["created_at"] for item in detail_rows if item["disposition_at"] is not None]
        agent_latencies = [item["agent_result_at"] - item["created_at"] for item in detail_rows if item["agent_result_at"] is not None]
        queue_waits = [item["started_at"] - item["created_at"] for item in detail_rows if item["started_at"] is not None]
        pending_queue_waits = [now - item["created_at"] for item in detail_rows if item["status"] == "queued"]
        def percentile(values, quantile):
            import math
            if not values:
                return None
            ordered = sorted(values)
            return round(ordered[min(len(ordered) - 1, max(0, math.ceil(len(ordered) * quantile) - 1))], 3)
        return {"total_jobs": total, "disposed_jobs": disposed,
                "queued_jobs": sum(item["status"] == "queued" for item in detail_rows),
                "running_jobs": sum(item["status"] == "running" for item in detail_rows),
                "timely_disposition_count": timely, "timely_agent_result_count": agent,
                "manual_review_count": manual,
                "timely_disposition_rate": round(timely / float(disposed), 4) if disposed else 0.0,
                "agent_result_slo_rate": round(agent / float(disposed), 4) if disposed else 0.0,
                "disposition_latency_p50_seconds": percentile(disposition_latencies, 0.50),
                "disposition_latency_p95_seconds": percentile(disposition_latencies, 0.95),
                "agent_result_latency_p95_seconds": percentile(agent_latencies, 0.95),
                "queue_wait_p95_seconds": percentile(queue_waits + pending_queue_waits, 0.95)}

class PrecheckJobCoordinator:
    """Background worker pool with durable metadata and in-memory sensitive payloads."""

    def __init__(self, store: SQLiteJobStore, processor: Any, *, workers: int = 2,
                 queue_timeout_seconds: float = 90, deadline_seconds: float = 180,
                 max_queue_size: int = 200, poll_interval_seconds: float = 0.1) -> None:
        import queue
        import threading
        import math

        if workers < 1 or max_queue_size < 1:
            raise ValueError("workers and max_queue_size must be positive")
        timeouts = (queue_timeout_seconds, deadline_seconds, poll_interval_seconds)
        if any(not math.isfinite(float(value)) or float(value) <= 0 for value in timeouts):
            raise ValueError("queue timeout, deadline, and poll interval must be finite and positive")
        self.store = store
        self.processor = processor
        self.workers = workers
        self.queue_timeout_seconds = queue_timeout_seconds
        self.deadline_seconds = deadline_seconds
        self.poll_interval_seconds = poll_interval_seconds
        self._queue = queue.Queue(maxsize=max_queue_size)
        self._payloads: Dict[str, Dict[str, Any]] = {}
        self._results: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._started = False
        self._closed = False
        self._stop = threading.Event()
        # Payloads/results are memory-only, so stale tasks are routed to manual review.
        self.store.recover_after_restart(now=__import__("time").time(), deadline_seconds=deadline_seconds)

    def _start(self) -> None:
        import threading
        if self._started:
            return
        with self._lock:
            if self._started:
                return
            self._started = True
            for _ in range(self.workers):
                threading.Thread(target=self._worker, daemon=True, name="precheck-worker").start()
            threading.Thread(target=self._watchdog, daemon=True, name="precheck-watchdog").start()

    def submit(self, payload: Dict[str, Any], *, institution: str, product_line: str,
               subject: str, idempotency_key: str, received_at: Optional[float] = None) -> Dict[str, Any]:
        import queue
        import time
        if self._closed:
            raise RuntimeError("job coordinator is closed")
        def json_default(value):
            if isinstance(value, bytes):
                return {"__bytes_base64__": base64.b64encode(value).decode("ascii")}
            if isinstance(value, (date, datetime)):
                return value.isoformat()
            raise TypeError("unsupported request payload type")
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                               ensure_ascii=False, default=json_default).encode("utf-8")
        request_hash = hashlib.sha256(canonical).hexdigest()
        record = self.store.create(institution=institution, product_line=product_line,
                                   subject=subject, idempotency_key=idempotency_key,
                                   now=received_at if received_at is not None else time.time(),
                                   request_hash=request_hash)
        if not record["created"]:
            return self.get(record["job_id"], institution=institution,
                            product_line=product_line, subject=subject) or record
        with self._lock:
            self._payloads[record["job_id"]] = payload
        self._start()
        try:
            self._queue.put_nowait(record["job_id"])
        except queue.Full:
            self.store.manual_review(record["job_id"], reason="backlog_limit", now=time.time(),
                                     deadline_seconds=self.deadline_seconds)
            with self._lock:
                self._payloads.pop(record["job_id"], None)
        return self.get(record["job_id"], institution=institution,
                        product_line=product_line, subject=subject) or record

    def _worker(self) -> None:
        import queue
        import time
        while not self._stop.is_set():
            try:
                job_id = self._queue.get(timeout=self.poll_interval_seconds)
            except queue.Empty:
                continue
            try:
                claimed = self.store.claim(job_id, now=time.time(),
                                           queue_timeout_seconds=self.queue_timeout_seconds,
                                           deadline_seconds=self.deadline_seconds)
                if claimed is None:
                    with self._lock:
                        self._payloads.pop(job_id, None)
                    continue
                with self._lock:
                    payload = self._payloads.pop(job_id, None)
                if payload is None:
                    self.store.manual_review(job_id, reason="worker_payload_unavailable", now=time.time(),
                                             deadline_seconds=self.deadline_seconds)
                    continue
                try:
                    result = self.processor(payload)
                except Exception:
                    self.store.manual_review(job_id, reason="processing_error", now=time.time(),
                                             deadline_seconds=self.deadline_seconds)
                    continue
                with self._lock:
                    if self.store.finish(job_id, now=time.time(), deadline_seconds=self.deadline_seconds):
                        self._results[job_id] = result
            finally:
                self._queue.task_done()

    def _watchdog(self) -> None:
        import time
        while not self._stop.wait(self.poll_interval_seconds):
            expired = self.store.expire(now=time.time(), queue_timeout_seconds=self.queue_timeout_seconds,
                                        deadline_seconds=self.deadline_seconds)
            if expired:
                with self._lock:
                    for job_id in expired:
                        self._payloads.pop(job_id, None)
                        self._results.pop(job_id, None)

    def get(self, job_id: str, *, institution: str, product_line: str, subject: str) -> Optional[Dict[str, Any]]:
        record = self.store.get_for_scope(job_id, institution=institution,
                                          product_line=product_line, subject=subject)
        if record is None:
            return None
        if record["status"] == "completed":
            with self._lock:
                record["result"] = self._results.get(job_id)
        return record

    def close(self) -> None:
        self._closed = True
        self._stop.set()
