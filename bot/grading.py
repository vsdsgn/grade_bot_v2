from __future__ import annotations

import logging
from typing import Any

from .constants import DIMENSIONS, DIMENSION_DISPLAY, LEVELS
from .models import Session, Turn
from .openai_service import OpenAIService, OpenAIServiceError

logger = logging.getLogger(__name__)


IC_LEVEL_ORDER = ["Junior", "Middle", "Senior", "Lead_IC", "ArtDirector_IC"]
M_LEVEL_ORDER = ["Junior", "Middle", "Senior", "Head_M", "DesignDirector_M"]

DEFAULT_ROLE_FLOORS: dict[str, list[str]] = {
    "Junior": ["intern", "стажер", "junior", "trainee"],
    "Middle": ["middle", "mid-level", "дизайнер продукта", "product designer"],
    "Senior": ["senior", "старший дизайнер", "senior designer"],
    "Lead_IC": ["lead designer", "staff designer", "principal designer", "ведущий дизайнер"],
    "Head_M": [
        "head of design",
        "design manager",
        "руководитель дизайна",
        "руководитель группы",
        "head of ux",
        "head of product design",
    ],
    "ArtDirector_IC": ["art director", "арт-директор"],
    "DesignDirector_M": ["design director", "директор по дизайну", "vp design"],
}

DEFAULT_TRACK_HINTS: dict[str, list[str]] = {
    "IC": ["ic", "individual contributor", "ведущий дизайнер", "арт-директор"],
    "M": ["manager", "head", "director", "руководитель", "менеджер", "управление командой"],
}

MANAGER_BEHAVIOR_KEYWORDS = [
    "наним",
    "увольн",
    "ревью",
    "1:1",
    "one-on-one",
    "performance review",
    "план развития",
    "развитие команды",
    "управляю командой",
    "управление командой",
    "ресурсное планирование",
    "бюджет",
    "организационн",
]

DIRECTOR_BEHAVIOR_KEYWORDS = [
    "дизайн-стратег",
    "design strategy",
    "стратегия дизайна",
    "оргдизайн",
    "нескольких команд",
    "группа продуктов",
    "портфель продуктов",
    "cross-functional leadership",
    "директор",
    "vp design",
]


class GradeEngine:
    def __init__(self, matrix: dict[str, Any], openai_service: OpenAIService) -> None:
        self.matrix = matrix
        self.openai_service = openai_service

    def _supplemental_matrices(self) -> list[dict[str, Any]]:
        raw = self.matrix.get("supplemental_matrices", [])
        if isinstance(raw, list):
            return [m for m in raw if isinstance(m, dict)]
        return []

    @staticmethod
    def _canonical_level(raw_level: str | None) -> str | None:
        if not raw_level:
            return None

        level = str(raw_level).strip()
        if level in LEVELS:
            return level

        normalized = level.lower().replace(" ", "").replace("-", "").replace("_", "")

        if any(key in normalized for key in ["designdirector", "директорподизайну", "vpdesign"]):
            return "DesignDirector_M"
        if any(
            key in normalized
            for key in [
                "headofdesign",
                "designmanager",
                "руководительдизайна",
                "17менеджер",
                "18руководитель",
            ]
        ):
            return "Head_M"
        if any(key in normalized for key in ["artdirector", "артдиректор", "18специалист"]):
            return "ArtDirector_IC"
        if any(key in normalized for key in ["lead", "staff", "principal", "17специалист", "ведущий"]):
            return "Lead_IC"
        if "senior" in normalized or "син" in normalized or "стар" in normalized or normalized == "16":
            return "Senior"
        if "middle" in normalized or "mid" in normalized or "мид" in normalized or normalized == "15":
            return "Middle"
        if "junior" in normalized or "джун" in normalized or "нович" in normalized or normalized == "14":
            return "Junior"

        return None

    @staticmethod
    def _track_for_level(level: str) -> str:
        if level in {"Head_M", "DesignDirector_M"}:
            return "M"
        return "IC"

    @staticmethod
    def _role_text(profile: dict[str, Any], turns: list[Turn]) -> str:
        parts = [
            str(profile.get("current_role", "")),
            str(profile.get("experience_and_track_goal", "")),
            str(profile.get("domain_and_users", "")),
        ]
        parts.extend(t.content for t in turns if t.role == "user")
        return " ".join(parts).lower()

    def _merged_role_floors(self) -> dict[str, list[str]]:
        merged: dict[str, list[str]] = {level: list(markers) for level, markers in DEFAULT_ROLE_FLOORS.items()}
        for matrix in self._supplemental_matrices():
            role_floors = matrix.get("role_floors", {})
            if not isinstance(role_floors, dict):
                continue

            for raw_level, raw_markers in role_floors.items():
                canonical_level = self._canonical_level(str(raw_level))
                if not canonical_level:
                    continue

                markers: list[str] = []
                if isinstance(raw_markers, list):
                    markers = [str(m).lower() for m in raw_markers if str(m).strip()]
                elif isinstance(raw_markers, str) and raw_markers.strip():
                    markers = [raw_markers.lower()]

                if markers:
                    merged.setdefault(canonical_level, [])
                    merged[canonical_level].extend(markers)

        for level, markers in merged.items():
            seen: set[str] = set()
            deduped: list[str] = []
            for marker in markers:
                if marker in seen:
                    continue
                seen.add(marker)
                deduped.append(marker)
            merged[level] = deduped

        return merged

    def _merged_track_hints(self) -> dict[str, list[str]]:
        merged: dict[str, list[str]] = {track: list(markers) for track, markers in DEFAULT_TRACK_HINTS.items()}
        for matrix in self._supplemental_matrices():
            hints = matrix.get("track_hints", {})
            if not isinstance(hints, dict):
                continue

            for track, raw_markers in hints.items():
                track_key = str(track).upper()
                if track_key not in {"IC", "M"}:
                    continue

                markers: list[str] = []
                if isinstance(raw_markers, list):
                    markers = [str(m).lower() for m in raw_markers if str(m).strip()]
                elif isinstance(raw_markers, str) and raw_markers.strip():
                    markers = [raw_markers.lower()]

                if markers:
                    merged.setdefault(track_key, [])
                    merged[track_key].extend(markers)

        for track, markers in merged.items():
            seen: set[str] = set()
            deduped: list[str] = []
            for marker in markers:
                if marker in seen:
                    continue
                seen.add(marker)
                deduped.append(marker)
            merged[track] = deduped

        return merged

    def _merged_behavior_keywords(self, key: str, default_markers: list[str]) -> list[str]:
        merged = [str(marker).lower() for marker in default_markers if str(marker).strip()]

        for matrix in self._supplemental_matrices():
            raw = matrix.get(key, [])
            if isinstance(raw, list):
                markers = [str(marker).lower() for marker in raw if str(marker).strip()]
            elif isinstance(raw, str) and raw.strip():
                markers = [raw.lower()]
            else:
                markers = []
            merged.extend(markers)

        deduped: list[str] = []
        seen: set[str] = set()
        for marker in merged:
            if marker in seen:
                continue
            seen.add(marker)
            deduped.append(marker)
        return deduped

    @staticmethod
    def _adapt_level_to_track(level: str, track: str) -> str:
        if track == "M":
            if level == "Lead_IC":
                return "Senior"
            if level == "ArtDirector_IC":
                return "Head_M"
            return level

        if level == "Head_M":
            return "Lead_IC"
        if level == "DesignDirector_M":
            return "ArtDirector_IC"
        return level

    @staticmethod
    def _level_index_for_track(level: str, track: str) -> int:
        order = M_LEVEL_ORDER if track == "M" else IC_LEVEL_ORDER
        if level in order:
            return order.index(level)
        return 0

    def _pick_higher_level(self, current: str, floor: str, track: str) -> str:
        current_norm = self._adapt_level_to_track(current, track)
        floor_norm = self._adapt_level_to_track(floor, track)

        if self._level_index_for_track(current_norm, track) < self._level_index_for_track(floor_norm, track):
            return floor_norm
        return current_norm

    @staticmethod
    def _count_keyword_hits(text: str, keywords: list[str]) -> int:
        return sum(1 for keyword in keywords if keyword and keyword in text)

    def _infer_track(self, session: Session, profile: dict[str, Any], turns: list[Turn]) -> str:
        if session.track_preference in {"IC", "M"}:
            return session.track_preference

        text = self._role_text(profile, turns)
        hints = self._merged_track_hints()

        ic_hits = sum(1 for marker in hints.get("IC", []) if marker and marker in text)
        m_hits = sum(1 for marker in hints.get("M", []) if marker and marker in text)

        if m_hits > ic_hits:
            return "M"
        if ic_hits > m_hits:
            return "IC"

        manager_markers = self._merged_behavior_keywords("manager_behavior_keywords", MANAGER_BEHAVIOR_KEYWORDS)
        behavior_m_hits = self._count_keyword_hits(text, manager_markers)
        if behavior_m_hits >= 2:
            return "M"

        role_floors = self._merged_role_floors()
        manager_levels = {"Head_M", "DesignDirector_M"}
        manager_floor_hit = any(
            marker in text
            for level, markers in role_floors.items()
            if level in manager_levels
            for marker in markers
        )
        return "M" if manager_floor_hit else "IC"

    def _infer_role_floor(self, text: str, track: str) -> str | None:
        role_floors = self._merged_role_floors()
        candidates: list[str] = []

        for level, markers in role_floors.items():
            for marker in markers:
                if marker and marker in text:
                    candidates.append(level)
                    break

        if not candidates:
            return None

        best = self._adapt_level_to_track(candidates[0], track)
        for candidate in candidates[1:]:
            normalized = self._adapt_level_to_track(candidate, track)
            if self._level_index_for_track(normalized, track) > self._level_index_for_track(best, track):
                best = normalized

        return best

    def _infer_behavioral_floor(self, text: str, track: str) -> str | None:
        manager_markers = self._merged_behavior_keywords("manager_behavior_keywords", MANAGER_BEHAVIOR_KEYWORDS)
        director_markers = self._merged_behavior_keywords("director_behavior_keywords", DIRECTOR_BEHAVIOR_KEYWORDS)

        manager_hits = self._count_keyword_hits(text, manager_markers)
        director_hits = self._count_keyword_hits(text, director_markers)

        if track == "M":
            if director_hits >= 2:
                return "DesignDirector_M"
            if manager_hits >= 2:
                return "Head_M"
            if manager_hits >= 1:
                return "Senior"
            return None

        if manager_hits >= 3:
            return "Lead_IC"
        return None

    @staticmethod
    def _clip_snippet(text: str, max_chars: int = 180) -> str:
        compact = " ".join(text.strip().split())
        if len(compact) <= max_chars:
            return compact
        return compact[: max_chars - 1].rstrip() + "…"

    @staticmethod
    def _normalize_string_list(raw: Any, limit: int, fallback: list[str]) -> list[str]:
        if isinstance(raw, list):
            items = [str(item).strip() for item in raw if str(item).strip()]
        elif isinstance(raw, str) and raw.strip():
            items = [raw.strip()]
        else:
            items = []

        deduped: list[str] = []
        seen: set[str] = set()
        for item in items:
            if item in seen:
                continue
            seen.add(item)
            deduped.append(item)
            if len(deduped) >= limit:
                break

        if deduped:
            return deduped
        return fallback[:limit]

    @classmethod
    def _normalize_learning_items(cls, raw: Any) -> list[dict[str, str]]:
        fallback = [
            {
                "title": "Storytelling для дизайн-кейсов",
                "why": "Чтобы яснее показывать влияние и ваш личный вклад.",
            },
            {
                "title": "Продуктовые метрики для дизайнеров",
                "why": "Чтобы связывать дизайн-решения с бизнес-результатом.",
            },
            {
                "title": "Influence в кросс-функциональных командах",
                "why": "Чтобы быстрее проводить сложные решения.",
            },
        ]

        if not isinstance(raw, list):
            return fallback

        items: list[dict[str, str]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", "")).strip()
            why = str(item.get("why", "")).strip()
            if not title or not why:
                continue
            items.append({"title": title, "why": why})
            if len(items) >= 5:
                break

        if items:
            return items
        return fallback

    @staticmethod
    def _ensure_dimension_keys(payload: dict[str, Any]) -> dict[str, Any]:
        dimension_scores = payload.get("dimension_scores", {})
        if not isinstance(dimension_scores, dict):
            dimension_scores = {}

        evidence = payload.get("evidence", {})
        if not isinstance(evidence, dict):
            evidence = {}

        next_level_targets = payload.get("next_level_targets", {})
        if not isinstance(next_level_targets, dict):
            next_level_targets = {}

        for dim in DIMENSIONS:
            try:
                score = float(dimension_scores.get(dim, 0))
            except (TypeError, ValueError):
                score = 0.0
            dimension_scores[dim] = min(max(score, 0.0), 10.0)

            raw_evidence = evidence.get(dim, [])
            if isinstance(raw_evidence, str):
                snippets = [raw_evidence]
            elif isinstance(raw_evidence, list):
                snippets = [str(item) for item in raw_evidence]
            else:
                snippets = []
            cleaned_snippets = [
                GradeEngine._clip_snippet(snippet, max_chars=180)
                for snippet in snippets
                if str(snippet).strip()
            ][:4]
            evidence[dim] = cleaned_snippets

            raw_target = str(next_level_targets.get(dim, "")).strip()
            next_level_targets[dim] = (
                raw_target if raw_target else "Нужно больше evidence и системная практика для роста."
            )

        payload["overall_level"] = GradeEngine._canonical_level(str(payload.get("overall_level", ""))) or "Middle"
        track = str(payload.get("track", "")).upper()
        payload["track"] = track if track in {"IC", "M"} else "IC"

        try:
            confidence = float(payload.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        payload["confidence"] = min(max(confidence, 0.0), 1.0)

        payload["strengths"] = GradeEngine._normalize_string_list(
            payload.get("strengths"),
            limit=4,
            fallback=["Сильная база по ключевым зонам ответственности."],
        )
        payload["growth_areas"] = GradeEngine._normalize_string_list(
            payload.get("growth_areas"),
            limit=4,
            fallback=["Усилить измеримость влияния и системность решений."],
        )
        payload["recommended_actions"] = GradeEngine._normalize_string_list(
            payload.get("recommended_actions"),
            limit=4,
            fallback=[
                "Соберите 2-3 кейса с метриками до/после и явным личным вкладом.",
                "Зафиксируйте план роста на 90 дней по двум ключевым зонам.",
                "Регулярно синхронизируйте ожидания по уровню с руководителем.",
            ],
        )
        payload["recommended_learning"] = GradeEngine._normalize_learning_items(
            payload.get("recommended_learning")
        )

        payload["dimension_scores"] = dimension_scores
        payload["evidence"] = evidence
        payload["next_level_targets"] = next_level_targets
        return payload

    def _apply_role_priors(
        self,
        payload: dict[str, Any],
        session: Session,
        profile: dict[str, Any],
        turns: list[Turn],
    ) -> dict[str, Any]:
        text = self._role_text(profile, turns)

        inferred_track = self._infer_track(session, profile, turns)
        payload_track = str(payload.get("track", "")).upper()
        track = payload_track if payload_track in {"IC", "M"} else inferred_track

        role_floor = self._infer_role_floor(text, track)
        behavioral_floor = self._infer_behavioral_floor(text, track)

        current_level = self._canonical_level(str(payload.get("overall_level", ""))) or "Middle"
        if role_floor:
            current_level = self._pick_higher_level(current_level, role_floor, track)
        if behavioral_floor:
            current_level = self._pick_higher_level(current_level, behavioral_floor, track)

        explicit_manager_title = any(
            marker in text
            for marker in ["head of design", "design director", "руководитель дизайна", "директор по дизайну"]
        )
        if explicit_manager_title:
            track = "M"
            manager_floor = "Head_M"
            if "design director" in text or "директор по дизайну" in text:
                manager_floor = "DesignDirector_M"
            current_level = self._pick_higher_level(current_level, manager_floor, track)

        payload["track"] = track
        payload["overall_level"] = current_level

        dimension_scores = payload.get("dimension_scores", {})
        if isinstance(dimension_scores, dict):
            if current_level in {"Head_M", "DesignDirector_M"}:
                dimension_scores["management"] = max(float(dimension_scores.get("management", 0)), 7.2)
                dimension_scores["scope_responsibility"] = max(float(dimension_scores.get("scope_responsibility", 0)), 7.0)
                dimension_scores["impact"] = max(float(dimension_scores.get("impact", 0)), 6.8)
                dimension_scores["soft_communication_influence"] = max(
                    float(dimension_scores.get("soft_communication_influence", 0)),
                    7.0,
                )

            if current_level in {"Lead_IC", "ArtDirector_IC"}:
                dimension_scores["hard_craft"] = max(float(dimension_scores.get("hard_craft", 0)), 6.8)
                dimension_scores["hard_systems"] = max(float(dimension_scores.get("hard_systems", 0)), 6.5)

            for dim in DIMENSIONS:
                try:
                    score = float(dimension_scores.get(dim, 0))
                except (TypeError, ValueError):
                    score = 0.0
                dimension_scores[dim] = min(max(score, 0.0), 10.0)

            payload["dimension_scores"] = dimension_scores

        try:
            confidence = float(payload.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0

        if (role_floor or behavioral_floor) and confidence < 0.58:
            confidence = 0.58
        payload["confidence"] = min(max(confidence, 0.0), 1.0)

        return payload

    def _matrix_excerpt_full(self) -> dict[str, Any]:
        return {
            "version": self.matrix.get("version", "2.0"),
            "levels": self.matrix.get("levels", []),
            "dimension_weights": self.matrix.get("dimension_weights", {}),
            "dimensions": self.matrix.get("dimensions", {}),
            "supplemental_matrices": self.matrix.get("supplemental_matrices", []),
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

        supplemental = []
        for matrix in self._supplemental_matrices()[:8]:
            supplemental.append(
                {
                    "name": matrix.get("name", "matrix"),
                    "description": matrix.get("description", ""),
                    "role_floors": matrix.get("role_floors", {}),
                    "track_hints": matrix.get("track_hints", {}),
                    "manager_behavior_keywords": matrix.get("manager_behavior_keywords", []),
                    "director_behavior_keywords": matrix.get("director_behavior_keywords", []),
                }
            )

        return {
            "version": self.matrix.get("version", "2.0"),
            "levels": self.matrix.get("levels", []),
            "dimension_weights": self.matrix.get("dimension_weights", {}),
            "dimensions": compact_dims,
            "supplemental_matrices": supplemental,
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
    def _level_from_average(track: str, average_score: float) -> str:
        if track == "M":
            if average_score >= 8.5:
                return "DesignDirector_M"
            if average_score >= 7.3:
                return "Head_M"
            if average_score >= 6.4:
                return "Senior"
            if average_score >= 5.3:
                return "Middle"
            return "Junior"

        if average_score >= 8.5:
            return "ArtDirector_IC"
        if average_score >= 7.5:
            return "Lead_IC"
        if average_score >= 6.5:
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
            evidence[dim] = evidence[dim][:2]

        dimension_scores: dict[str, float] = {}
        for dim in DIMENSIONS:
            snippets = evidence.get(dim, [])
            count = len(snippets)
            base = 4.8 + (count * 1.5)
            if count > 0:
                avg_words = sum(len(s.split()) for s in snippets) / count
                base += min(avg_words / 90.0, 1.0)

            score = max(4.0, min(base, 9.2))
            if dim == "management" and track == "IC":
                score = min(score, 6.0)
            dimension_scores[dim] = round(score, 1)

        scoring_dims = [d for d in DIMENSIONS if not (track == "IC" and d == "management")]
        average_score = sum(dimension_scores[d] for d in scoring_dims) / max(1, len(scoring_dims))

        overall_level = self._level_from_average(track, average_score)

        ranked = sorted(dimension_scores.items(), key=lambda kv: kv[1])
        top_dims = [d for d, _ in ranked[-3:]][::-1]
        low_dims = [d for d, _ in ranked[:3]]

        strengths = [f"Сильная зона: {DIMENSION_DISPLAY.get(dim, dim)}." for dim in top_dims]
        growth_areas = [f"Зона роста: {DIMENSION_DISPLAY.get(dim, dim)}." for dim in low_dims]

        next_level_targets = {
            dim: (
                f"Покажите более масштабный кейс по зоне «{DIMENSION_DISPLAY.get(dim, dim)}» "
                "с явным личным вкладом и измеримым эффектом."
            )
            for dim in DIMENSIONS
        }

        recommended_actions = [
            "Соберите 3 кейса с четкими метриками до/после и акцентом на вашей роли.",
            "Усильте связь решений с бизнес-результатом и рисками, которые вы закрывали.",
            "Сформируйте 90-дневный план роста по 2 самым слабым зонам.",
        ]

        recommended_learning = [
            {
                "title": "Storytelling для дизайн-кейсов",
                "why": "Чтобы убедительнее показывать масштаб и влияние ваших решений.",
            },
            {
                "title": "Продуктовые метрики для дизайнеров",
                "why": "Чтобы привязывать дизайн-выбор к измеримому результату.",
            },
            {
                "title": "Influence без формальной власти",
                "why": "Чтобы быстрее выравнивать сложные кросс-функциональные решения.",
            },
        ]

        payload = {
            "overall_level": overall_level,
            "track": track,
            "confidence": max(0.5, min(float(session.confidence_estimate or 0.6), 0.85)),
            "dimension_scores": dimension_scores,
            "evidence": evidence,
            "strengths": strengths,
            "growth_areas": growth_areas,
            "next_level_targets": next_level_targets,
            "recommended_actions": recommended_actions,
            "recommended_learning": recommended_learning,
        }

        return self._apply_role_priors(payload, session=session, profile=profile, turns=turns)

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
                grade_payload = self._apply_role_priors(
                    grade_payload,
                    session=session,
                    profile=profile,
                    turns=turns,
                )
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
