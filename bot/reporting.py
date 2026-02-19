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
    def _fmt_dimension_scores(dimension_scores: dict[str, float]) -> list[str]:
        lines: list[str] = []
        for dim, value in dimension_scores.items():
            title = DIMENSION_DISPLAY.get(dim, dim)
            lines.append(f"- {title}: {value:.1f}/10")
        return lines

    @staticmethod
    def _roadmap_milestones(
        dimension_scores: dict[str, float],
        next_level_targets: dict[str, str],
    ) -> list[tuple[str, str]]:
        ranked = sorted(dimension_scores.items(), key=lambda item: item[1])
        milestones: list[tuple[str, str]] = []
        for dim, _ in ranked:
            title = DIMENSION_DISPLAY.get(dim, dim)
            target = next_level_targets.get(dim, "")
            if target:
                milestones.append((title, target))
            if len(milestones) == 3:
                break
        return milestones

    def build_markdown(self, grade: dict[str, Any]) -> str:
        overall_level = grade.get("overall_level", "Unknown")
        track = grade.get("track", "Unknown")
        confidence = float(grade.get("confidence", 0.0))
        dimension_scores = grade.get("dimension_scores", {})
        evidence = grade.get("evidence", {})
        strengths = grade.get("strengths", [])
        growth_areas = grade.get("growth_areas", [])
        actions = grade.get("recommended_actions", [])
        learning = grade.get("recommended_learning", [])
        next_targets = grade.get("next_level_targets", {})

        milestones = self._roadmap_milestones(dimension_scores, next_targets)

        lines: list[str] = []
        lines.append("## Product Designer Self-Assessment Report")
        lines.append("")
        lines.append(f"**Suggested level:** {overall_level}")
        lines.append(f"**Track:** {track}")
        lines.append(f"**Confidence:** {confidence:.2f}")
        lines.append("")
        lines.append("### Per-layer scores")
        lines.extend(self._fmt_dimension_scores(dimension_scores))
        lines.append("")

        lines.append("### Evidence snippets")
        for dim, snippets in evidence.items():
            title = DIMENSION_DISPLAY.get(dim, dim)
            if snippets:
                formatted = "; ".join(f'"{s}"' for s in snippets[:2])
            else:
                formatted = "No direct evidence captured"
            lines.append(f"- {title}: {formatted}")
        lines.append("")

        lines.append("### Strengths")
        for item in strengths[:5]:
            lines.append(f"- {item}")
        lines.append("")

        lines.append("### Growth areas")
        for item in growth_areas[:5]:
            lines.append(f"- {item}")
        lines.append("")

        lines.append("### Next-step recommendations")
        for item in actions[:8]:
            lines.append(f"- {item}")
        lines.append("")

        lines.append("### Next level roadmap (3 milestones)")
        for i, (title, target) in enumerate(milestones, start=1):
            lines.append(f"{i}. {title}: {target}")
        lines.append("")

        lines.append("### Suggested learning")
        for item in learning[:10]:
            title = item.get("title", "Resource")
            why = item.get("why", "")
            lines.append(f"- {title}: {why}")

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
