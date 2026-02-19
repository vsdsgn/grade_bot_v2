from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(slots=True)
class Session:
    id: int
    telegram_user_id: int
    status: str
    started_at: str
    completed_at: str | None
    question_count: int
    max_questions: int
    warmup_index: int
    profile_json: str
    track_preference: str | None
    evidence_summary: str
    summarized_turn_count: int
    confidence_estimate: float
    final_report_markdown: str | None
    final_report_json: str | None
    export_path: str | None


@dataclass(slots=True)
class Turn:
    id: int
    session_id: int
    turn_index: int
    role: str
    content: str
    dimension: str | None
    created_at: str


@dataclass(slots=True)
class NextQuestion:
    question: str
    follow_up_probe: str | None


@dataclass(slots=True)
class ReportArtifacts:
    report_json: dict[str, Any]
    markdown: str
    export_path: str
    generated_at: datetime
