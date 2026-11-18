"""Metadata-only audit sinks for API requests; never include prompts or content."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, List


class MemoryAuditSink:
    """In-memory sink for tests and ephemeral development."""

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def write(self, event: Dict[str, Any]) -> None:
        with self._lock:
            self.events.append(dict(event))


class LocalJsonlAuditSink:
    """Append metadata events to a local JSONL file with owner-only permissions."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def write(self, event: Dict[str, Any]) -> None:
        line = json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n"
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with self._lock:
            descriptor = os.open(str(self.path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            try:
                os.fchmod(descriptor, 0o600)
                with os.fdopen(descriptor, "a", encoding="utf-8") as stream:
                    stream.write(line)
                    stream.flush()
                    os.fsync(stream.fileno())
            except Exception:
                # fdopen owns descriptor after it succeeds.
                raise
