from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .constants import DIMENSION_DISPLAY
from .models import ReportArtifacts, Session


class ReportBuilder:
    def __init__(self, exports_dir: Path) -> None:
        self.exports_dir = exports_dir

    @staticmethod
    def _coerce_scores(raw_scores: Any) -> dict[str, float]:
        if not isinstance(raw_scores, dict):
            return {}

        cleaned: dict[str, float] = {}
        for key, value in raw_scores.items():
            try:
                score = float(value)
            except (TypeError, ValueError):
                score = 0.0
            cleaned[str(key)] = min(max(score, 0.0), 10.0)
        return cleaned

    @staticmethod
    def _coerce_text_list(raw: Any, limit: int) -> list[str]:
        if isinstance(raw, list):
            items = [str(item).strip() for item in raw if str(item).strip()]
        elif isinstance(raw, str) and raw.strip():
            items = [raw.strip()]
        else:
            items = []

        return items[:limit]

    @staticmethod
    def _top_gap_dims(dimension_scores: dict[str, float], n: int = 3) -> list[str]:
        ranked = sorted(dimension_scores.items(), key=lambda item: item[1])
        return [dim for dim, _ in ranked[:n]]

    @staticmethod
    def _evidence_line(evidence: dict[str, list[str]], dim: str) -> str | None:
        raw_snippets = evidence.get(dim, []) if isinstance(evidence, dict) else []
        snippets = raw_snippets if isinstance(raw_snippets, list) else []
        if not snippets:
            return None

        snippet = str(snippets[0]).strip()
        if not snippet:
            return None

        if len(snippet) > 130:
            snippet = snippet[:129].rstrip() + "…"

        return f"{DIMENSION_DISPLAY.get(dim, dim)}: {snippet}"

    def build_markdown(self, grade: dict[str, Any]) -> str:
        overall_level = str(grade.get("overall_level", "Неизвестно"))

        try:
            confidence = float(grade.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = min(max(confidence, 0.0), 1.0)

        dimension_scores = self._coerce_scores(grade.get("dimension_scores", {}))
        evidence = grade.get("evidence", {})
        growth_areas = self._coerce_text_list(grade.get("growth_areas", []), limit=3)
        actions = self._coerce_text_list(grade.get("recommended_actions", []), limit=3)

        why_lines: list[str] = []
        for dim, _ in sorted(dimension_scores.items(), key=lambda item: item[1], reverse=True):
            line = self._evidence_line(evidence, dim)
            if line:
                why_lines.append(line)
            if len(why_lines) >= 3:
                break

        if not why_lines:
            why_lines = ["Оценка построена по вашим кейсам, масштабу задач и личному вкладу."]

        if not growth_areas:
            gap_dims = self._top_gap_dims(dimension_scores, n=3)
            growth_areas = [f"Усилить: {DIMENSION_DISPLAY.get(dim, dim)}." for dim in gap_dims]

        if not actions:
            actions = [
                "Соберите 2-3 кейса с метриками до/после и четким описанием личного вклада.",
                "Для каждой ключевой инициативы фиксируйте влияние на бизнес-результат.",
                "Сформируйте план развития на 90 дней по 2 главным зонам роста.",
            ]

        lines: list[str] = []
        lines.append("## Ваш результат")
        lines.append("")
        lines.append(f"**Грейд:** {overall_level}")
        lines.append(f"**Уверенность оценки:** {confidence:.2f}")
        lines.append("")

        lines.append("### Почему такая оценка")
        for item in why_lines[:3]:
            lines.append(f"- {item}")
        lines.append("")

        lines.append("### Зоны роста")
        for item in growth_areas[:3]:
            lines.append(f"- {item}")
        lines.append("")

        lines.append("### Что делать дальше")
        for idx, item in enumerate(actions[:3], start=1):
            lines.append(f"{idx}. {item}")

        return "\n".join(lines).strip()

    def build_report_json(
        self,
        session: Session,
        profile: dict[str, Any],
        grade: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "version": "2.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "telegram_user_id": session.telegram_user_id,
            "session_id": session.id,
            "profile": profile,
            "assessment": grade,
        }

    def export_report(
        self,
        session: Session,
        profile: dict[str, Any],
        grade: dict[str, Any],
    ) -> ReportArtifacts:
        generated_at = datetime.now(timezone.utc)
        timestamp = generated_at.strftime("%Y%m%d_%H%M%S")
        export_path = self.exports_dir / f"{session.telegram_user_id}_{timestamp}.json"

        report_json = self.build_report_json(session, profile, grade)
        markdown = self.build_markdown(grade)

        with export_path.open("w", encoding="utf-8") as f:
            json.dump(report_json, f, indent=2, ensure_ascii=True)

        return ReportArtifacts(
            report_json=report_json,
            markdown=markdown,
            export_path=str(export_path),
            generated_at=generated_at,
        )
