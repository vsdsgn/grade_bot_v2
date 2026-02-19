from __future__ import annotations

from typing import Any

from .constants import DIMENSIONS
from .models import Session, Turn
from .openai_service import OpenAIService


class GradeEngine:
    def __init__(self, matrix: dict[str, Any], openai_service: OpenAIService) -> None:
        self.matrix = matrix
        self.openai_service = openai_service

    def _matrix_excerpt(self) -> dict[str, Any]:
        return {
            "version": self.matrix.get("version", "2.0"),
            "levels": self.matrix.get("levels", []),
            "dimension_weights": self.matrix.get("dimension_weights", {}),
            "dimensions": self.matrix.get("dimensions", {}),
        }

    @staticmethod
    def _turns_payload(turns: list[Turn], limit: int = 16) -> list[dict[str, str]]:
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

    @staticmethod
    def _ensure_dimension_keys(payload: dict[str, Any]) -> dict[str, Any]:
        dimension_scores = payload.get("dimension_scores", {})
        evidence = payload.get("evidence", {})
        next_level_targets = payload.get("next_level_targets", {})

        for dim in DIMENSIONS:
            dimension_scores.setdefault(dim, 0)
            evidence.setdefault(dim, [])
            next_level_targets.setdefault(dim, "Gather more evidence and grow through deliberate practice.")

        payload["dimension_scores"] = dimension_scores
        payload["evidence"] = evidence
        payload["next_level_targets"] = next_level_targets
        return payload

    async def run_grade(
        self,
        session: Session,
        profile: dict[str, Any],
        turns: list[Turn],
    ) -> dict[str, Any]:
        evidence_summary = session.evidence_summary
        if not evidence_summary:
            evidence_summary = "No prior summary available. Use recent turns as primary evidence."

        grade_payload = await self.openai_service.grade_assessment(
            track_hint=session.track_preference,
            profile=profile,
            evidence_summary=evidence_summary,
            recent_turns=self._turns_payload(turns),
            matrix_excerpt=self._matrix_excerpt(),
        )

        grade_payload = self._ensure_dimension_keys(grade_payload)

        # Normalize confidence bounds if needed.
        confidence = float(grade_payload.get("confidence", 0))
        grade_payload["confidence"] = min(max(confidence, 0.0), 1.0)
        return grade_payload
