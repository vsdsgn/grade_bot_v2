from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .constants import DIMENSION_DISPLAY, HIGH_VARIANCE_PRIORITY
from .matrix import required_dimensions_for_track
from .models import NextQuestion, Session, Turn
from .openai_service import OpenAIService


@dataclass(slots=True)
class ProgressSnapshot:
    required_dimensions: list[str]
    covered_dimensions: list[str]
    remaining_dimensions: list[str]
    coverage_ratio: float
    question_count: int
    max_questions: int
    confidence_estimate: float


class DialogueManager:
    def __init__(self, matrix: dict[str, Any], openai_service: OpenAIService) -> None:
        self.matrix = matrix
        self.openai_service = openai_service

    @staticmethod
    def infer_track_preference(text: str) -> str | None:
        normalized = text.lower()

        manager_hits = [
            "manager",
            "management",
            "people manager",
            "head of",
            "director",
            "hiring",
            "менедж",
            "руковод",
            "управлен",
            "лид команды",
            "наним",
        ]
        ic_hits = [
            "ic",
            "individual contributor",
            "hands-on",
            "craft",
            "principal designer",
            "staff designer",
            "art director",
            "индивидуальный вклад",
            "индивидуальный трек",
            "сильный крафт",
            "хочу оставаться в продукте",
        ]

        m = any(keyword in normalized for keyword in manager_hits)
        ic = any(keyword in normalized for keyword in ic_hits)

        if m and not ic:
            return "M"
        if ic and not m:
            return "IC"
        return None

    @staticmethod
    def is_vague_answer(text: str) -> bool:
        token_count = len(text.split())
        if token_count < 12:
            return True

        vague_patterns = [
            r"\bit depends\b",
            r"\bnot sure\b",
            r"\busually\b",
            r"\bkind of\b",
            r"\bsomewhat\b",
            r"\bmaybe\b",
            r"\bзависит\b",
            r"\bне знаю\b",
            r"\bобычно\b",
            r"\bнаверное\b",
            r"\bпримерно\b",
            r"\bкак-то\b",
        ]
        return any(re.search(p, text.lower()) for p in vague_patterns)

    @staticmethod
    def build_recent_turns_payload(turns: list[Turn], limit: int = 10) -> list[dict[str, str]]:
        payload: list[dict[str, str]] = []
        for turn in turns[-limit:]:
            item = {
                "role": turn.role,
                "content": turn.content,
            }
            if turn.dimension:
                item["dimension"] = turn.dimension
            payload.append(item)
        return payload

    def choose_next_dimension(
        self,
        track: str | None,
        answered_dimensions: dict[str, int],
    ) -> str:
        required_dimensions = required_dimensions_for_track(track)

        for dim in HIGH_VARIANCE_PRIORITY:
            if dim in required_dimensions and answered_dimensions.get(dim, 0) == 0:
                return dim

        ranked = sorted(
            required_dimensions,
            key=lambda dim: (answered_dimensions.get(dim, 0), required_dimensions.index(dim)),
        )
        return ranked[0]

    def estimate_confidence(
        self,
        track: str | None,
        answered_dimensions: dict[str, int],
        user_turns: list[Turn],
    ) -> float:
        required_dimensions = required_dimensions_for_track(track)

        covered = [dim for dim in required_dimensions if answered_dimensions.get(dim, 0) > 0]
        coverage = len(covered) / max(1, len(required_dimensions))

        depth_points = sum(min(answered_dimensions.get(dim, 0), 2) for dim in required_dimensions)
        depth = depth_points / max(1, len(required_dimensions) * 2)

        avg_len = 0.0
        if user_turns:
            avg_len = sum(len(turn.content.split()) for turn in user_turns) / len(user_turns)
        clarity = min(avg_len / 60.0, 1.0)

        confidence = (0.5 * coverage) + (0.3 * depth) + (0.2 * clarity)
        return round(min(max(confidence, 0.0), 0.99), 3)

    def should_finish(
        self,
        session: Session,
        track: str | None,
        answered_dimensions: dict[str, int],
        confidence_estimate: float,
        confidence_threshold: float,
    ) -> bool:
        if session.question_count >= session.max_questions:
            return True

        required_dimensions = required_dimensions_for_track(track)
        covered = [dim for dim in required_dimensions if answered_dimensions.get(dim, 0) > 0]
        coverage_ratio = len(covered) / max(1, len(required_dimensions))

        return (
            session.question_count >= 8
            and coverage_ratio >= 0.75
            and confidence_estimate >= confidence_threshold
        )

    def progress_snapshot(
        self,
        session: Session,
        track: str | None,
        answered_dimensions: dict[str, int],
    ) -> ProgressSnapshot:
        required = required_dimensions_for_track(track)
        covered = [dim for dim in required if answered_dimensions.get(dim, 0) > 0]
        remaining = [dim for dim in required if answered_dimensions.get(dim, 0) == 0]
        ratio = len(covered) / max(1, len(required))

        return ProgressSnapshot(
            required_dimensions=required,
            covered_dimensions=covered,
            remaining_dimensions=remaining,
            coverage_ratio=ratio,
            question_count=session.question_count,
            max_questions=session.max_questions,
            confidence_estimate=session.confidence_estimate,
        )

    async def generate_assessment_question(
        self,
        session: Session,
        profile: dict[str, Any],
        turns: list[Turn],
        answered_dimensions: dict[str, int],
    ) -> tuple[str, NextQuestion]:
        target_dimension = self.choose_next_dimension(session.track_preference, answered_dimensions)
        dimension_matrix = self.matrix["dimensions"][target_dimension]
        recent_turns = self.build_recent_turns_payload(turns, limit=10)

        next_question = await self.openai_service.generate_next_question(
            target_dimension=target_dimension,
            profile=profile,
            evidence_summary=session.evidence_summary,
            recent_turns=recent_turns,
            dimension_matrix=dimension_matrix,
        )

        return target_dimension, next_question

    async def refresh_evidence_summary_if_needed(
        self,
        session: Session,
        turns: list[Turn],
    ) -> tuple[str, int] | None:
        unsummarized_turns = turns[session.summarized_turn_count :]
        if len(unsummarized_turns) < 6:
            return None

        payload = [
            {
                "role": turn.role,
                "dimension": turn.dimension or "",
                "content": turn.content,
            }
            for turn in unsummarized_turns
        ]

        updated_summary = await self.openai_service.summarize_evidence(
            existing_summary=session.evidence_summary,
            turns_to_summarize=payload,
        )

        return updated_summary, len(turns)

    @staticmethod
    def extract_profile(session: Session) -> dict[str, Any]:
        try:
            data = json.loads(session.profile_json or "{}")
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
        return {}

    @staticmethod
    def covered_dimension_names(dimensions: list[str]) -> str:
        if not dimensions:
            return "пока нет"
        return ", ".join(DIMENSION_DISPLAY.get(d, d) for d in dimensions)
