"""Structured memory: SQLite, not chat history.

Tables (see the brief):
  students   id + declared VARK self-report
  attempts   one row per student/topic/format with the graded score
  flags      concepts the agent handed to the lecturer

Extra tables that make the prototype actually work:
  quiz_items answer keys live ONLY here, server-side, never in a prompt that
             produces student-facing text
  reviews    optional spaced-repetition dates
  cache      content-addressed artefacts, so re-running a demo costs 0 tokens
  messages   conversation history per session (truncated when reused)
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS students (
    id         TEXT PRIMARY KEY,
    declared   TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS attempts (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id TEXT NOT NULL,
    topic      TEXT NOT NULL,
    format     TEXT NOT NULL,
    score      REAL NOT NULL,
    max_score  REAL NOT NULL DEFAULT 3,
    ts         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_attempts_student ON attempts(student_id, topic);

CREATE TABLE IF NOT EXISTS flags (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    concept    TEXT NOT NULL,
    reason     TEXT,
    student_id TEXT,
    topic      TEXT,
    ts         TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS quiz_items (
    id             TEXT PRIMARY KEY,
    student_id     TEXT,
    topic          TEXT,
    format         TEXT,
    question       TEXT NOT NULL,
    answer_key     TEXT NOT NULL,
    concept        TEXT,
    source         TEXT,
    rubric         TEXT,
    max_score      REAL NOT NULL DEFAULT 2,
    created_at     TEXT NOT NULL,
    graded         INTEGER NOT NULL DEFAULT 0,
    student_answer TEXT,
    awarded        REAL,
    feedback       TEXT
);
CREATE INDEX IF NOT EXISTS idx_quiz_concept ON quiz_items(concept);

CREATE TABLE IF NOT EXISTS reviews (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id TEXT NOT NULL,
    topic      TEXT NOT NULL,
    due_date   TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS cache (
    key        TEXT PRIMARY KEY,
    payload    TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    student_id TEXT,
    role       TEXT NOT NULL,
    content    TEXT NOT NULL,
    ts         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
"""


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


class Store:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    # -- plumbing ---------------------------------------------------------- #
    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _init(self) -> None:
        with self._connect() as connection:
            connection.executescript(SCHEMA)

    def _execute(self, sql: str, params: tuple = ()) -> None:
        with self._connect() as connection:
            connection.execute(sql, params)

    def _query(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        with self._connect() as connection:
            return [dict(row) for row in connection.execute(sql, params).fetchall()]

    # -- students ---------------------------------------------------------- #
    def upsert_student(self, student_id: str, declared: dict[str, Any] | None = None) -> dict[str, Any]:
        existing = self.get_student(student_id)
        if existing is None:
            self._execute(
                "INSERT INTO students (id, declared, created_at) VALUES (?, ?, ?)",
                (student_id, json.dumps(declared or {}), _now()),
            )
        elif declared is not None:
            self._execute(
                "UPDATE students SET declared = ? WHERE id = ?",
                (json.dumps(declared), student_id),
            )
        return self.get_student(student_id) or {"id": student_id, "declared": {}}

    def get_student(self, student_id: str) -> dict[str, Any] | None:
        rows = self._query("SELECT * FROM students WHERE id = ?", (student_id,))
        if not rows:
            return None
        student = rows[0]
        try:
            student["declared"] = json.loads(student.get("declared") or "{}")
        except json.JSONDecodeError:
            student["declared"] = {}
        return student

    def all_students(self) -> list[dict[str, Any]]:
        return self._query("SELECT * FROM students ORDER BY created_at")

    # -- attempts ---------------------------------------------------------- #
    def add_attempt(self, student_id: str, topic: str, fmt: str, score: float, max_score: float = 3) -> None:
        self._execute(
            "INSERT INTO attempts (student_id, topic, format, score, max_score, ts) VALUES (?, ?, ?, ?, ?, ?)",
            (student_id, topic, fmt.upper(), float(score), float(max_score), _now()),
        )

    def attempts_for(self, student_id: str, topic: str | None = None) -> list[dict[str, Any]]:
        if topic:
            return self._query(
                "SELECT * FROM attempts WHERE student_id = ? AND topic = ? ORDER BY id",
                (student_id, topic),
            )
        return self._query("SELECT * FROM attempts WHERE student_id = ? ORDER BY id", (student_id,))

    def all_attempts(self) -> list[dict[str, Any]]:
        return self._query("SELECT * FROM attempts ORDER BY id")

    # -- flags ------------------------------------------------------------- #
    def add_flag(self, concept: str, reason: str = "", student_id: str | None = None, topic: str | None = None) -> None:
        self._execute(
            "INSERT INTO flags (concept, reason, student_id, topic, ts) VALUES (?, ?, ?, ?, ?)",
            (concept.strip(), reason.strip(), student_id, topic, _now()),
        )

    def flags(self) -> list[dict[str, Any]]:
        return self._query("SELECT * FROM flags ORDER BY id")

    # -- quiz items (answer keys stay here) -------------------------------- #
    def add_quiz_item(
        self,
        *,
        student_id: str,
        topic: str,
        fmt: str,
        question: str,
        answer_key: str,
        concept: str = "",
        source: str = "",
        rubric: str = "",
        max_score: float = 2,
    ) -> str:
        item_id = f"q_{uuid.uuid4().hex[:10]}"
        self._execute(
            """INSERT INTO quiz_items
               (id, student_id, topic, format, question, answer_key, concept, source, rubric,
                max_score, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (item_id, student_id, topic, fmt.upper(), question, answer_key, concept, source,
             rubric, float(max_score), _now()),
        )
        return item_id

    def get_quiz_item(self, item_id: str) -> dict[str, Any] | None:
        rows = self._query("SELECT * FROM quiz_items WHERE id = ?", (item_id,))
        return rows[0] if rows else None

    def quiz_items_for(self, student_id: str, topic: str, fmt: str | None = None) -> list[dict[str, Any]]:
        if fmt:
            return self._query(
                "SELECT * FROM quiz_items WHERE student_id = ? AND topic = ? AND format = ? ORDER BY created_at",
                (student_id, topic, fmt.upper()),
            )
        return self._query(
            "SELECT * FROM quiz_items WHERE student_id = ? AND topic = ? ORDER BY created_at",
            (student_id, topic),
        )

    def mark_graded(self, item_id: str, answer: str, awarded: float, feedback: str) -> None:
        self._execute(
            "UPDATE quiz_items SET graded = 1, student_answer = ?, awarded = ?, feedback = ? WHERE id = ?",
            (answer, float(awarded), feedback, item_id),
        )

    # -- lecturer digest --------------------------------------------------- #
    def concept_digest(self, top_n: int = 3) -> list[dict[str, Any]]:
        """Concepts the cohort has *not* grasped, ranked by weakness then flags.

        weakness  = 1 - (awarded / max_score), averaged over graded questions
        flags     = number of explicit flag_for_lecturer calls for that concept
        """
        concepts: dict[str, dict[str, Any]] = {}

        for row in self._query(
            """SELECT concept, AVG(COALESCE(awarded, 0) / max_score) AS pct,
                      COUNT(*) AS graded, MAX(source) AS source
               FROM quiz_items
               WHERE graded = 1 AND concept IS NOT NULL AND concept != ''
               GROUP BY concept"""
        ):
            concepts.setdefault(row["concept"], {"concept": row["concept"], "source": row["source"]})
            entry = concepts[row["concept"]]
            entry["avg_pct"] = float(row["pct"] or 0.0)
            entry["graded"] = int(row["graded"])
            entry["flags"] = 0

        for row in self._query(
            "SELECT concept, COUNT(*) AS n FROM flags GROUP BY concept"
        ):
            entry = concepts.setdefault(row["concept"], {"concept": row["concept"], "source": ""})
            entry["flags"] = int(row["n"])

        for row in self._query(
            "SELECT concept, MAX(source) AS source FROM quiz_items GROUP BY concept"
        ):
            entry = concepts.setdefault(row["concept"], {"concept": row["concept"], "source": ""})
            if not entry.get("source"):
                entry["source"] = row["source"]

        ranked = sorted(
            concepts.values(),
            key=lambda e: (-e.get("flags", 0), e.get("avg_pct", 1.0), e["concept"]),
        )
        for entry in ranked:
            entry.setdefault("avg_pct", 1.0)
            entry.setdefault("graded", 0)
            entry["weakness"] = round(1.0 - entry["avg_pct"], 3)
            entry["avg_pct"] = round(entry["avg_pct"], 3)
        return ranked[:top_n]

    # -- reviews ----------------------------------------------------------- #
    def add_review(self, student_id: str, topic: str, due_date: str) -> None:
        self._execute(
            "INSERT INTO reviews (student_id, topic, due_date, created_at) VALUES (?, ?, ?, ?)",
            (student_id, topic, due_date, _now()),
        )

    def reviews_for(self, student_id: str) -> list[dict[str, Any]]:
        return self._query(
            "SELECT * FROM reviews WHERE student_id = ? ORDER BY due_date", (student_id,)
        )

    # -- artefact cache ---------------------------------------------------- #
    def cache_get(self, key: str) -> dict[str, Any] | None:
        rows = self._query("SELECT payload FROM cache WHERE key = ?", (key,))
        if not rows:
            return None
        try:
            return json.loads(rows[0]["payload"])
        except json.JSONDecodeError:
            return None

    def cache_set(self, key: str, payload: dict[str, Any]) -> None:
        self._execute(
            "INSERT OR REPLACE INTO cache (key, payload, created_at) VALUES (?, ?, ?)",
            (key, json.dumps(payload, ensure_ascii=False), _now()),
        )

    # -- conversation history ---------------------------------------------- #
    def add_message(self, session_id: str, student_id: str, role: str, content: str) -> None:
        self._execute(
            "INSERT INTO messages (session_id, student_id, role, content, ts) VALUES (?, ?, ?, ?, ?)",
            (session_id, student_id, role, content, _now()),
        )

    def history(self, session_id: str, limit: int = 8) -> list[dict[str, Any]]:
        rows = self._query(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        )
        return list(reversed(rows))

    # -- demo utilities ---------------------------------------------------- #
    def stats(self) -> dict[str, int]:
        counts = {}
        for table in ("students", "attempts", "flags", "quiz_items", "reviews"):
            counts[table] = int(self._query(f"SELECT COUNT(*) AS n FROM {table}")[0]["n"])
        return counts

    def clear_demo_data(self, keep_cache: bool = True) -> None:
        tables = ["students", "attempts", "flags", "quiz_items", "reviews", "messages"]
        with self._connect() as connection:
            for table in tables:
                connection.execute(f"DELETE FROM {table}")
            if not keep_cache:
                connection.execute("DELETE FROM cache")
