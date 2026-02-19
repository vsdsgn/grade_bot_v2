from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from .models import Session, Turn


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Database:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def init(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    telegram_user_id INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    question_count INTEGER NOT NULL DEFAULT 0,
                    max_questions INTEGER NOT NULL DEFAULT 18,
                    warmup_index INTEGER NOT NULL DEFAULT 0,
                    profile_json TEXT NOT NULL DEFAULT '{}',
                    track_preference TEXT,
                    evidence_summary TEXT NOT NULL DEFAULT '',
                    summarized_turn_count INTEGER NOT NULL DEFAULT 0,
                    confidence_estimate REAL NOT NULL DEFAULT 0,
                    final_report_markdown TEXT,
                    final_report_json TEXT,
                    export_path TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_sessions_user_status
                    ON sessions (telegram_user_id, status);

                CREATE TABLE IF NOT EXISTS turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    turn_index INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    dimension TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_turns_session_idx
                    ON turns (session_id, turn_index);
                """
            )

    def _row_to_session(self, row: sqlite3.Row) -> Session:
        return Session(
            id=row["id"],
            telegram_user_id=row["telegram_user_id"],
            status=row["status"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            question_count=row["question_count"],
            max_questions=row["max_questions"],
            warmup_index=row["warmup_index"],
            profile_json=row["profile_json"],
            track_preference=row["track_preference"],
            evidence_summary=row["evidence_summary"],
            summarized_turn_count=row["summarized_turn_count"],
            confidence_estimate=row["confidence_estimate"],
            final_report_markdown=row["final_report_markdown"],
            final_report_json=row["final_report_json"],
            export_path=row["export_path"],
        )

    def _row_to_turn(self, row: sqlite3.Row) -> Turn:
        return Turn(
            id=row["id"],
            session_id=row["session_id"],
            turn_index=row["turn_index"],
            role=row["role"],
            content=row["content"],
            dimension=row["dimension"],
            created_at=row["created_at"],
        )

    def create_session(self, telegram_user_id: int, max_questions: int) -> Session:
        started_at = utc_now_iso()
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO sessions (
                    telegram_user_id, status, started_at, max_questions
                ) VALUES (?, 'in_progress', ?, ?)
                """,
                (telegram_user_id, started_at, max_questions),
            )
            session_id = int(cur.lastrowid)

            row = conn.execute(
                "SELECT * FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
            if row is None:
                raise RuntimeError("Failed to create session")
            return self._row_to_session(row)

    def get_active_session(self, telegram_user_id: int) -> Session | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM sessions
                WHERE telegram_user_id = ? AND status = 'in_progress'
                ORDER BY id DESC LIMIT 1
                """,
                (telegram_user_id,),
            ).fetchone()
            return self._row_to_session(row) if row else None

    def get_latest_session(self, telegram_user_id: int) -> Session | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM sessions
                WHERE telegram_user_id = ?
                ORDER BY id DESC LIMIT 1
                """,
                (telegram_user_id,),
            ).fetchone()
            return self._row_to_session(row) if row else None

    def get_session(self, session_id: int) -> Session | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
            return self._row_to_session(row) if row else None

    def reset_active_session(self, telegram_user_id: int) -> bool:
        session = self.get_active_session(telegram_user_id)
        if session is None:
            return False

        with self._connect() as conn:
            conn.execute("DELETE FROM turns WHERE session_id = ?", (session.id,))
            conn.execute(
                "DELETE FROM sessions WHERE id = ?",
                (session.id,),
            )
        return True

    def append_turn(
        self,
        session_id: int,
        role: str,
        content: str,
        dimension: str | None = None,
    ) -> Turn:
        created_at = utc_now_iso()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(turn_index), 0) AS max_idx FROM turns WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            next_idx = int(row["max_idx"]) + 1
            cur = conn.execute(
                """
                INSERT INTO turns (session_id, turn_index, role, content, dimension, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (session_id, next_idx, role, content, dimension, created_at),
            )
            turn_id = int(cur.lastrowid)
            turn_row = conn.execute("SELECT * FROM turns WHERE id = ?", (turn_id,)).fetchone()
            if turn_row is None:
                raise RuntimeError("Failed to create turn")
            return self._row_to_turn(turn_row)

    def list_turns(self, session_id: int) -> list[Turn]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM turns WHERE session_id = ? ORDER BY turn_index ASC",
                (session_id,),
            ).fetchall()
            return [self._row_to_turn(r) for r in rows]

    def list_user_turns(self, session_id: int) -> list[Turn]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM turns
                WHERE session_id = ? AND role = 'user'
                ORDER BY turn_index ASC
                """,
                (session_id,),
            ).fetchall()
            return [self._row_to_turn(r) for r in rows]

    def get_answered_dimensions(self, session_id: int) -> dict[str, int]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT dimension, COUNT(*) AS cnt
                FROM turns
                WHERE session_id = ? AND role = 'user' AND dimension IS NOT NULL
                GROUP BY dimension
                """,
                (session_id,),
            ).fetchall()
            return {row["dimension"]: int(row["cnt"]) for row in rows}

    def update_profile_field(self, session_id: int, field: str, value: str) -> None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT profile_json FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
            if row is None:
                raise RuntimeError("Session not found")
            profile = json.loads(row["profile_json"] or "{}")
            profile[field] = value
            conn.execute(
                "UPDATE sessions SET profile_json = ? WHERE id = ?",
                (json.dumps(profile, ensure_ascii=True), session_id),
            )

    def increment_warmup_index(self, session_id: int) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE sessions SET warmup_index = warmup_index + 1 WHERE id = ?",
                (session_id,),
            )

    def increment_question_count(self, session_id: int) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE sessions SET question_count = question_count + 1 WHERE id = ?",
                (session_id,),
            )

    def update_track_preference(self, session_id: int, track_preference: str | None) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE sessions SET track_preference = ? WHERE id = ?",
                (track_preference, session_id),
            )

    def update_confidence(self, session_id: int, confidence_estimate: float) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE sessions SET confidence_estimate = ? WHERE id = ?",
                (confidence_estimate, session_id),
            )

    def update_evidence_summary(
        self,
        session_id: int,
        evidence_summary: str,
        summarized_turn_count: int,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE sessions
                SET evidence_summary = ?, summarized_turn_count = ?
                WHERE id = ?
                """,
                (evidence_summary, summarized_turn_count, session_id),
            )

    def mark_completed(
        self,
        session_id: int,
        final_report_markdown: str,
        final_report_json: str,
        export_path: str,
        confidence_estimate: float,
    ) -> None:
        completed_at = utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE sessions
                SET status = 'completed',
                    completed_at = ?,
                    final_report_markdown = ?,
                    final_report_json = ?,
                    export_path = ?,
                    confidence_estimate = ?
                WHERE id = ?
                """,
                (
                    completed_at,
                    final_report_markdown,
                    final_report_json,
                    export_path,
                    confidence_estimate,
                    session_id,
                ),
            )
