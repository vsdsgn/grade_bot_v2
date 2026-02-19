from __future__ import annotations

import logging
from typing import Any

from .constants import DIMENSIONS, DIMENSION_DISPLAY
from .models import Session, Turn
from .openai_service import OpenAIService, OpenAIServiceError

logger = logging.getLogger(__name__)


class GradeEngine:
    def __init__(self, matrix: dict[str, Any], openai_service: OpenAIService) -> None:
        self.matrix = matrix
        self.openai_service = openai_service

    def _matrix_excerpt_full(self) -> dict[str, Any]:
        return {
            "version": self.matrix.get("version", "2.0"),
            "levels": self.matrix.get("levels", []),
            "dimension_weights": self.matrix.get("dimension_weights", {}),
            "dimensions": self.matrix.get("dimensions", {}),
        }

    def _matrix_excerpt_compact(self) -> dict[str, Any]:
        dimensions = self.matrix.get("dimensions", {})
        compact_dims: dict[str, Any] = {}

        for dim, dim_payload in dimensions.items():
            compact_levels: dict[str, Any] = {}
            levels_payload = dim_payload.get("levels", {}) if isinstance(dim_payload, dict) else {}
            for level_name, level_data in levels_payload.items():
                signals = []
                anti_signals = []
                if isinstance(level_data, dict):
                    signals = list(level_data.get("signals", []))[:1]
                    anti_signals = list(level_data.get("anti_signals", []))[:1]

                compact_levels[level_name] = {
                    "signals": signals,
                    "anti_signals": anti_signals,
                }

            compact_dims[dim] = {
                "display_name": dim_payload.get("display_name", dim) if isinstance(dim_payload, dict) else dim,
                "levels": compact_levels,
            }

        return {
            "version": self.matrix.get("version", "2.0"),
            "levels": self.matrix.get("levels", []),
            "dimension_weights": self.matrix.get("dimension_weights", {}),
            "dimensions": compact_dims,
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
            next_level_targets.setdefault(dim, "Нужно больше evidence и системная практика для роста.")

        payload["dimension_scores"] = dimension_scores
        payload["evidence"] = evidence
        payload["next_level_targets"] = next_level_targets
        return payload

    @staticmethod
    def _clip_snippet(text: str, max_chars: int = 180) -> str:
        compact = " ".join(text.strip().split())
        if len(compact) <= max_chars:
            return compact
        return compact[: max_chars - 1].rstrip() + "…"

    @staticmethod
    def _infer_track(session: Session, profile: dict[str, Any], turns: list[Turn]) -> str:
        if session.track_preference in {"IC", "M"}:
            return session.track_preference

        haystack = " ".join(
            [
                str(profile.get("experience_and_track_goal", "")),
                str(profile.get("current_role", "")),
            ]
            + [t.content for t in turns if t.role == "user"]
        ).lower()

        manager_markers = ["менедж", "руковод", "head", "director", "people manager", "наним"]
        if any(marker in haystack for marker in manager_markers):
            return "M"

        return "IC"

    @staticmethod
    def _level_from_average(track: str, average_score: float) -> str:
        if track == "M":
            if average_score >= 8.4:
                return "DesignDirector_M"
            if average_score >= 7.4:
                return "Head_M"
            if average_score >= 6.4:
                return "Senior"
            if average_score >= 5.2:
                return "Middle"
            return "Junior"

        if average_score >= 8.4:
            return "ArtDirector_IC"
        if average_score >= 7.6:
            return "Lead_IC"
        if average_score >= 6.6:
            return "Senior"
        if average_score >= 5.4:
            return "Middle"
        return "Junior"

    def _build_fallback_grade(
        self,
        session: Session,
        profile: dict[str, Any],
        turns: list[Turn],
    ) -> dict[str, Any]:
        track = self._infer_track(session, profile, turns)

        evidence: dict[str, list[str]] = {dim: [] for dim in DIMENSIONS}
        for turn in turns:
            if turn.role != "user" or not turn.dimension or turn.dimension not in evidence:
                continue
            snippet = self._clip_snippet(turn.content)
            if snippet and snippet not in evidence[turn.dimension]:
                evidence[turn.dimension].append(snippet)

        for dim in evidence:
            evidence[dim] = evidence[dim][:3]

        dimension_scores: dict[str, float] = {}
        for dim in DIMENSIONS:
            snippets = evidence.get(dim, [])
            count = len(snippets)
            base = 4.2 + (count * 1.4)
            if count > 0:
                avg_words = sum(len(s.split()) for s in snippets) / count
                base += min(avg_words / 80.0, 1.0)

            score = max(3.5, min(base, 9.0))
            if dim == "management" and track == "IC":
                score = min(score, 5.5)
            dimension_scores[dim] = round(score, 1)

        scoring_dims = [d for d in DIMENSIONS if not (track == "IC" and d == "management")]
        average_score = sum(dimension_scores[d] for d in scoring_dims) / max(1, len(scoring_dims))

        overall_level = self._level_from_average(track, average_score)

        ranked = sorted(dimension_scores.items(), key=lambda kv: kv[1])
        top_dims = [d for d, _ in ranked[-5:]][::-1]
        low_dims = [d for d, _ in ranked[:5]]

        strengths = [
            f"Хорошая опора на «{DIMENSION_DISPLAY.get(dim, dim)}»."
            for dim in top_dims[:5]
        ]
        growth_areas = [
            f"Усилить «{DIMENSION_DISPLAY.get(dim, dim)}» через более системные кейсы и метрики."
            for dim in low_dims[:5]
        ]

        next_level_targets = {
            dim: (
                f"Показать 2-3 кейса с более высоким масштабом в зоне «{DIMENSION_DISPLAY.get(dim, dim)}», "
                "где явно видны личная роль, решение и измеримый результат."
            )
            for dim in DIMENSIONS
        }

        recommended_actions = [
            "Для каждого ключевого кейса фиксируйте: контекст, гипотезу, вашу личную роль, итоговые метрики.",
            "Усильте связь между дизайн-решениями и бизнес-результатом в презентации кейсов.",
            "Показывайте, как вы принимали решения в условиях неопределенности и рисков.",
            "Формализуйте подход к планированию: квартальная цель, фазы, риски, критерии успеха.",
            "Выберите 1-2 зоны роста и планомерно соберите evidence по ним в ближайшие 6-8 недель.",
            "Регулярно проводите ретроспективу по крупным решениям и фиксируйте выводы в процесс команды.",
        ]

        recommended_learning = [
            {
                "title": "Storytelling для продуктовых кейсов",
                "why": "Поможет яснее доносить влияние ваших решений и личный вклад.",
            },
            {
                "title": "Продуктовые метрики для дизайнеров",
                "why": "Усилит аргументацию через измеримый эффект, а не только качество интерфейса.",
            },
            {
                "title": "Принятие решений в неопределенности",
                "why": "Позволит увереннее вести сложные задачи с неполными данными.",
            },
            {
                "title": "Системный дизайн и дизайн-системы",
                "why": "Повысит масштабируемость решений и качество на уровне продукта/линейки.",
            },
            {
                "title": "Коммуникация и влияние без формальной власти",
                "why": "Поможет быстрее выравнивать кросс-функциональные решения.",
            },
        ]

        confidence = float(session.confidence_estimate or 0.55)
        confidence = max(0.45, min(confidence, 0.85))

        return {
            "overall_level": overall_level,
            "track": track,
            "confidence": round(confidence, 2),
            "dimension_scores": dimension_scores,
            "evidence": evidence,
            "strengths": strengths,
            "growth_areas": growth_areas,
            "next_level_targets": next_level_targets,
            "recommended_actions": recommended_actions[:8],
            "recommended_learning": recommended_learning,
        }

    async def run_grade(
        self,
        session: Session,
        profile: dict[str, Any],
        turns: list[Turn],
    ) -> dict[str, Any]:
        evidence_summary = session.evidence_summary
        if not evidence_summary:
            evidence_summary = "Предыдущего summary нет, используем последние реплики как основной evidence-контекст."

        attempts = [
            {
                "matrix_excerpt": self._matrix_excerpt_full(),
                "recent_turns": self._turns_payload(turns, limit=16),
                "label": "full",
            },
            {
                "matrix_excerpt": self._matrix_excerpt_compact(),
                "recent_turns": self._turns_payload(turns, limit=10),
                "label": "compact",
            },
        ]

        last_error: Exception | None = None
        for attempt in attempts:
            try:
                grade_payload = await self.openai_service.grade_assessment(
                    track_hint=session.track_preference,
                    profile=profile,
                    evidence_summary=evidence_summary,
                    recent_turns=attempt["recent_turns"],
                    matrix_excerpt=attempt["matrix_excerpt"],
                )
                grade_payload = self._ensure_dimension_keys(grade_payload)

                confidence = float(grade_payload.get("confidence", 0))
                grade_payload["confidence"] = min(max(confidence, 0.0), 1.0)
                return grade_payload
            except OpenAIServiceError as exc:
                last_error = exc
                logger.warning("Grading attempt '%s' failed: %s", attempt["label"], exc)
            except Exception as exc:  # pragma: no cover - defensive fallback
                last_error = exc
                logger.exception("Unexpected grading failure on '%s'", attempt["label"])

        logger.error("All grading attempts failed, using deterministic fallback. Last error: %s", last_error)
        fallback_payload = self._build_fallback_grade(session=session, profile=profile, turns=turns)
        return self._ensure_dimension_keys(fallback_payload)
