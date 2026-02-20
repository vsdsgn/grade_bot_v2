from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from openai import APIError, APITimeoutError, AsyncOpenAI, RateLimitError

from .constants import DIMENSIONS, DIMENSION_DISPLAY, LEVELS
from .models import NextQuestion

logger = logging.getLogger(__name__)


class OpenAIServiceError(RuntimeError):
    pass


class OpenAIService:
    def __init__(self, api_key: str, model: str) -> None:
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model

    async def _responses_create_with_retry(self, **kwargs: Any) -> Any:
        delay = 1.0
        last_error: Exception | None = None

        for attempt in range(5):
            try:
                return await self.client.responses.create(**kwargs)
            except (RateLimitError, APITimeoutError) as exc:
                last_error = exc
                logger.warning(
                    "OpenAI transient error (%s), retry %s/5",
                    exc.__class__.__name__,
                    attempt + 1,
                )
                if attempt == 4:
                    break
                await asyncio.sleep(delay)
                delay *= 2
            except APIError as exc:
                last_error = exc
                retriable = getattr(exc, "status_code", 500) >= 500
                if not retriable or attempt == 4:
                    break
                logger.warning("OpenAI APIError retry %s/5: %s", attempt + 1, exc)
                await asyncio.sleep(delay)
                delay *= 2

        raise OpenAIServiceError(f"OpenAI request failed after retries: {last_error}")

    @staticmethod
    def _extract_text(response: Any) -> str:
        output_text = getattr(response, "output_text", None)
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()

        parts: list[str] = []
        output = getattr(response, "output", None)
        if output:
            for item in output:
                for content in getattr(item, "content", []) or []:
                    text = getattr(content, "text", None)
                    if isinstance(text, str) and text.strip():
                        parts.append(text.strip())
                        continue

                    text_value = getattr(text, "value", None)
                    if isinstance(text_value, str) and text_value.strip():
                        parts.append(text_value.strip())
                        continue

                    if isinstance(content, dict):
                        maybe_text = content.get("text")
                        if isinstance(maybe_text, str) and maybe_text.strip():
                            parts.append(maybe_text.strip())
                        elif isinstance(maybe_text, dict):
                            maybe_value = maybe_text.get("value")
                            if isinstance(maybe_value, str) and maybe_value.strip():
                                parts.append(maybe_value.strip())

        if parts:
            return "\n".join(parts)

        return str(response)

    @staticmethod
    def _extract_json(raw_text: str) -> dict[str, Any]:
        raw_text = raw_text.strip()
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if not match:
            raise OpenAIServiceError("Model output was not valid JSON")

        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise OpenAIServiceError(f"Model output JSON parse failed: {exc}") from exc

    @staticmethod
    def _extract_parsed_json(response: Any) -> dict[str, Any] | None:
        output_parsed = getattr(response, "output_parsed", None)
        if isinstance(output_parsed, dict):
            return output_parsed

        output = getattr(response, "output", None)
        if output:
            for item in output:
                for content in getattr(item, "content", []) or []:
                    parsed = getattr(content, "parsed", None)
                    if isinstance(parsed, dict):
                        return parsed

                    if isinstance(content, dict):
                        parsed_dict = content.get("parsed")
                        if isinstance(parsed_dict, dict):
                            return parsed_dict

        return None

    def _extract_payload(self, response: Any) -> dict[str, Any]:
        parsed = self._extract_parsed_json(response)
        if parsed is not None:
            return parsed
        return self._extract_json(self._extract_text(response))

    async def generate_next_question(
        self,
        target_dimension: str,
        profile: dict[str, Any],
        evidence_summary: str,
        recent_turns: list[dict[str, str]],
        dimension_matrix: dict[str, Any],
    ) -> NextQuestion:
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["question", "follow_up_probe"],
            "properties": {
                "question": {"type": "string"},
                # Empty string means no probe.
                "follow_up_probe": {"type": "string"},
            },
        }

        system_prompt = (
            "Ты опытный интервьюер по продуктовому дизайну. "
            "Веди разговор как живой диалог на русском языке. "
            "Сделай вопрос простым и коротким: одна мысль, одно предложение, одна точка фокуса. "
            "Не задавай сложных, составных и перечислительных вопросов."
        )

        user_prompt = {
            "target_dimension": target_dimension,
            "target_dimension_display": DIMENSION_DISPLAY.get(target_dimension, target_dimension),
            "profile": profile,
            "evidence_summary": evidence_summary,
            "recent_turns": recent_turns,
            "dimension_definition": dimension_matrix,
            "constraints": {
                "question_max_words": 14,
                "one_question_only": True,
                "follow_up_probe_optional": True,
                "follow_up_probe_empty_string_if_not_needed": True,
                "no_scoring": True,
                "language": "ru",
                "style": "natural_human_dialogue",
                "no_multi_part_question": True,
                "no_complex_clauses": True,
            },
        }

        response = await self._responses_create_with_retry(
            model=self.model,
            input=[
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": system_prompt}],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": json.dumps(user_prompt, ensure_ascii=True)}],
                },
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "next_question_payload",
                    "schema": schema,
                    "strict": True,
                }
            },
            temperature=0.65,
        )

        payload = self._extract_payload(response)
        question = str(payload.get("question", "")).strip()
        probe = str(payload.get("follow_up_probe", "")).strip()
        follow_up_probe = probe if probe else None

        if not question:
            raise OpenAIServiceError("Model returned an empty question")

        return NextQuestion(question=question, follow_up_probe=follow_up_probe)

    async def summarize_evidence(
        self,
        existing_summary: str,
        turns_to_summarize: list[dict[str, str]],
    ) -> str:
        if not turns_to_summarize:
            return existing_summary

        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["updated_summary"],
            "properties": {
                "updated_summary": {"type": "string"},
            },
        }

        system_prompt = (
            "Ты сжимаешь доказательства из интервью в рабочее summary для финального грейдинга. "
            "Пиши на русском, сохраняй только факты: результат, масштаб, личную роль и ограничения."
        )

        user_prompt = {
            "existing_summary": existing_summary,
            "new_turns": turns_to_summarize,
            "instructions": [
                "Сделай summary компактным и удобным для чтения.",
                "Сохраняй привязку evidence к конкретным измерениям.",
                "Ничего не выдумывай.",
                "Пиши только на русском языке.",
            ],
        }

        response = await self._responses_create_with_retry(
            model=self.model,
            input=[
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": system_prompt}],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": json.dumps(user_prompt, ensure_ascii=True)}],
                },
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "evidence_summary_payload",
                    "schema": schema,
                    "strict": True,
                }
            },
            temperature=0.25,
        )

        payload = self._extract_payload(response)
        updated_summary = str(payload.get("updated_summary", "")).strip()
        if not updated_summary:
            raise OpenAIServiceError("Model returned an empty summary")
        return updated_summary

    async def grade_assessment(
        self,
        track_hint: str | None,
        profile: dict[str, Any],
        evidence_summary: str,
        recent_turns: list[dict[str, str]],
        matrix_excerpt: dict[str, Any],
    ) -> dict[str, Any]:
        dimension_score_properties: dict[str, Any] = {
            dim: {"type": "number", "minimum": 0, "maximum": 10} for dim in DIMENSIONS
        }
        evidence_properties: dict[str, Any] = {
            dim: {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 0,
                "maxItems": 4,
            }
            for dim in DIMENSIONS
        }
        target_properties: dict[str, Any] = {dim: {"type": "string"} for dim in DIMENSIONS}

        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "overall_level",
                "track",
                "confidence",
                "dimension_scores",
                "evidence",
                "strengths",
                "growth_areas",
                "next_level_targets",
                "recommended_actions",
                "recommended_learning",
            ],
            "properties": {
                "overall_level": {"type": "string", "enum": LEVELS},
                "track": {"type": "string", "enum": ["IC", "M"]},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "dimension_scores": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": dimension_score_properties,
                },
                "evidence": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": evidence_properties,
                },
                "strengths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 2,
                    "maxItems": 4,
                },
                "growth_areas": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 2,
                    "maxItems": 4,
                },
                "next_level_targets": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": target_properties,
                },
                "recommended_actions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 3,
                    "maxItems": 4,
                },
                "recommended_learning": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["title", "why"],
                        "properties": {
                            "title": {"type": "string"},
                            "why": {"type": "string"},
                        },
                    },
                    "minItems": 3,
                    "maxItems": 5,
                },
            },
        }

        system_prompt = (
            "Ты калиброванный ассессор компетенций продуктовых дизайнеров по матрице. "
            "Определи наиболее вероятный уровень и трек (IC/M), оцени каждое измерение по шкале 0-10. "
            "Опирайся только на evidence из ответов кандидата. "
            "Если данных мало, снижай confidence и отражай это в зонах роста. "
            "Все текстовые поля верни только на русском языке."
        )

        user_prompt = {
            "track_hint": track_hint,
            "profile": profile,
            "evidence_summary": evidence_summary,
            "recent_turns": recent_turns,
            "matrix": matrix_excerpt,
            "instructions": {
                "evidence_quotes_short": True,
                "grounded_in_user_answers_only": True,
                "recommended_actions_count": "3-4",
                "recommended_learning_count": "3-5",
                "language": "ru",
            },
        }

        response = await self._responses_create_with_retry(
            model=self.model,
            input=[
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": system_prompt}],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": json.dumps(user_prompt, ensure_ascii=True)}],
                },
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "assessment_grade_payload",
                    "schema": schema,
                    "strict": True,
                }
            },
            temperature=0.2,
        )

        payload = self._extract_payload(response)
        return payload
