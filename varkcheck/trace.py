"""Structured tracing: every agent step goes to logs/*.jsonl.

The brief asks for "tool called, inputs, outputs, latency". One JSON object per
line means `jq` works, `grep` works, and nothing needs a database.
"""

from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class Trace:
    """Append-only JSONL trace for one run (a student session or a test replay)."""

    def __init__(self, log_dir: str | Path, run_id: str | None = None, echo: bool = False):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or f"run_{uuid.uuid4().hex[:12]}"
        self.path = self._path_for_today()
        self.echo = echo
        self.started = time.perf_counter()
        self.events: list[dict[str, Any]] = []

    def _path_for_today(self) -> Path:
        return self.log_dir / f"trace-{datetime.now().strftime('%Y%m%d')}.jsonl"

    def log(self, event: str, **fields: Any) -> dict[str, Any]:
        record = {
            "ts": _now_iso(),
            "run_id": self.run_id,
            "event": event,
            "elapsed_ms": round((time.perf_counter() - self.started) * 1000, 1),
            **fields,
        }
        # Keep the in-memory copy small; the file is the source of truth.
        if len(self.events) < 500:
            self.events.append(record)
        try:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except OSError:
            pass  # tracing must never break the demo
        if self.echo:
            print(json.dumps(record, ensure_ascii=False, default=str))
        return record

    @contextmanager
    def span(self, event: str, **fields: Any) -> Iterator[dict[str, Any]]:
        """Context manager that records latency even when the body raises."""
        started = time.perf_counter()
        payload: dict[str, Any] = {"ok": True}
        try:
            yield payload
        except Exception as exc:  # noqa: BLE001 - re-raised on purpose
            payload["ok"] = False
            payload["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self.log(
                event,
                latency_ms=round((time.perf_counter() - started) * 1000, 1),
                **payload,
                **fields,
            )

    def tail(self, n: int = 20) -> list[dict[str, Any]]:
        return self.events[-n:]

    def to_jsonl(self) -> str:
        return "\n".join(json.dumps(e, ensure_ascii=False, default=str) for e in self.events)
