from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from .constants import DIMENSION_DISPLAY, HIGH_VARIANCE_PRIORITY
from .matrix import required_dimensions_for_track
from .models import CorrectionReply, NextQuestion, Session, Turn
from .openai_service import OpenAIService, OpenAIServiceError

logger = logging.getLogger(__name__)


FALLBACK_QUESTION_BANK: dict[str, list[str]] = {
    "scope_responsibility": [
        "Коротко: за какой кусок продукта вы отвечаете в ключевом проекте?",
        "В каком проекте у вас был самый широкий scope?",
        "Где ваша личная ответственность была максимальной?",
    ],
    "impact": [
        "Где ваш дизайн-ход дал самый заметный результат?",
        "Какой кейс лучше всего показывает ваш impact?",
        "Что из вашего решения заметно изменило метрику или поведение пользователей?",
    ],
    "uncertainty_tolerance": [
        "В каком кейсе вы принимали решение при нехватке данных?",
        "Где вы двигались в неопределенности и как выбрали направление?",
        "Какой риск вы осознанно приняли в недавнем кейсе?",
    ],
    "planning_horizon": [
        "На какой горизонт вы реально планируете решения в продукте?",
        "В каком кейсе у вас был план на несколько шагов вперед?",
        "Как вы держите курс, когда план начинает плыть?",
    ],
    "hard_craft": [
        "Какой кейс лучше всего показывает ваш craft?",
        "Где вы заметно подняли качество интерфейса?",
        "Какая работа сильнее всего показывает ваш уровень дизайна?",
    ],
    "hard_systems": [
        "Что из ваших решений стало системным и переиспользуемым?",
        "Какое улучшение в дизайн-системе вы реально продвинули?",
        "Где ваше решение помогло нескольким командам сразу?",
    ],
    "hard_product_business": [
        "Когда пришлось выбирать между UX и бизнес-целью, как решили?",
        "В каком кейсе вы меняли решение ради бизнес-результата?",
        "Как вы связывали дизайн с продуктовой метрикой на практике?",
    ],
    "soft_communication_influence": [
        "Кого в сложном кейсе вам пришлось убеждать и как?",
        "Где вы повлияли на решение без формальной власти?",
        "В каком примере вам удалось развернуть позицию стейкхолдеров?",
    ],
    "management": [
        "Как вы развивали людей в команде на недавнем примере?",
        "Где ваши управленческие действия заметно подняли команду?",
        "Какой people-management кейс для вас самый показательный?",
    ],
    "culture_ownership": [
        "Где вы взяли ownership вне формальной зоны?",
        "В каком кейсе вы стали owner проблемы целиком?",
        "Когда вы довели задачу до результата, хотя это был не ваш участок?",
    ],
    "culture_proactivity": [
        "Что вы улучшили по собственной инициативе за последний год?",
        "Какую инициативу вы запустили без прямого запроса сверху?",
        "Где вы сами увидели проблему и начали действовать?",
    ],
    "culture_quality_bar": [
        "Как вы держите планку качества под дедлайн?",
        "Когда сроки давили, как вы не уронили quality bar?",
        "В каком кейсе вы отстояли качество решения?",
    ],
    "culture_collaboration": [
        "Как вы решали конфликт между функциями в реальном кейсе?",
        "Где вы выстроили рабочее партнерство с продуктом и разработкой?",
        "Как вы синхронизировали людей вокруг одного решения?",
    ],
    "culture_learning": [
        "Какой недавний фейл вы превратили в улучшение?",
        "Как вы превращаете ошибки в изменения процесса?",
        "Где вы показали сильную обучаемость в последнем проекте?",
    ],
    "culture_integrity_safety": [
        "Был ли кейс, где вы тормознули решение из-за риска для пользователя?",
        "Когда вы защищали пользователя несмотря на бизнес-давление?",
        "Как вы учитываете этику и безопасность в спорных решениях?",
    ],
}

FALLBACK_PROBE_BANK: dict[str, list[str]] = {
    "scope_responsibility": [
        "Что сделали именно вы?",
        "Какой масштаб был у задачи?",
    ],
    "impact": [
        "Какой результат получили?",
        "Как поняли, что решение сработало?",
    ],
    "uncertainty_tolerance": [
        "Какой риск был главным?",
        "Почему выбрали именно это решение?",
    ],
    "planning_horizon": [
        "Как проверяли, что план реалистичен?",
        "Какие риски учли заранее?",
    ],
    "hard_craft": [
        "Что в этом было самым сложным по craft?",
        "Как проверяли качество?",
    ],
    "hard_systems": [
        "Что из этого переиспользуют другие?",
        "За счет чего решение масштабировалось?",
    ],
    "hard_product_business": [
        "Какой бизнес-аргумент был ключевым?",
        "Какая метрика изменилась?",
    ],
    "soft_communication_influence": [
        "Какой аргумент в итоге сработал?",
        "Что сняло основное сопротивление?",
    ],
    "management": [
        "Что изменилось у людей после ваших действий?",
        "Как измеряли эффект?",
    ],
    "culture_ownership": [
        "Что вы взяли на себя лично?",
    ],
    "culture_proactivity": [
        "Что подтолкнуло вас начать это без запроса?",
    ],
    "culture_quality_bar": [
        "Что вы точно не готовы были упростить по качеству?",
    ],
    "culture_collaboration": [
        "Какой шаг реально снял напряжение между функциями?",
    ],
    "culture_learning": [
        "Какое изменение вы внедрили после этого урока?",
    ],
    "culture_integrity_safety": [
        "Какой риск для пользователя был ключевым?",
    ],
}

FRUSTRATION_PATTERNS = [
    r"заеб",
    r"достал",
    r"хватит",
    r"надоел",
    r"бесит",
    r"отстан",
    r"stop",
    r"enough",
]

NON_ASSESSMENT_SHORT_REPLIES = {
    "да",
    "нет",
    "ага",
    "ок",
    "окей",
    "ясно",
    "пон",
    "норм",
    "лол",
    "хз",
    "idk",
    "yes",
    "no",
}

OBSCENE_PATTERNS = [
    r"\bх[ую]й",
    r"\bпизд",
    r"\bеб[а-я]*",
    r"\bбля",
    r"\bсук",
    r"\bfuck",
    r"\bshit",
]

QUESTION_SIMILARITY_STOPWORDS = {
    "как",
    "что",
    "какой",
    "какая",
    "какие",
    "когда",
    "где",
    "почему",
    "зачем",
    "это",
    "для",
    "в",
    "на",
    "по",
    "и",
    "но",
    "или",
    "а",
    "the",
    "a",
    "an",
    "is",
    "are",
    "to",
    "of",
    "in",
}


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

        manager = any(keyword in normalized for keyword in manager_hits)
        ic = any(keyword in normalized for keyword in ic_hits)

        if manager and not ic:
            return "M"
        if ic and not manager:
            return "IC"
        return None

    @staticmethod
    def is_frustrated_answer(text: str) -> bool:
        normalized = text.lower()
        return any(re.search(pattern, normalized) for pattern in FRUSTRATION_PATTERNS)

    @classmethod
    def is_non_assessment_answer(cls, text: str) -> bool:
        normalized = text.lower().strip()
        if not normalized:
            return True

        if any(re.search(pattern, normalized) for pattern in OBSCENE_PATTERNS):
            return True

        if re.fullmatch(r"(ха)+", normalized) or re.fullmatch(r"(ah)+", normalized):
            return True

        words = [w for w in re.split(r"\s+", normalized) if w]
        if not words:
            return True

        if len(words) <= 2 and all(word in NON_ASSESSMENT_SHORT_REPLIES for word in words):
            return True

        has_letters_or_digits = bool(re.search(r"[a-zа-я0-9]", normalized))
        if not has_letters_or_digits:
            return True

        return False

    @staticmethod
    def _has_specificity_markers(text: str) -> bool:
        normalized = text.lower()
        if re.search(r"\d", normalized):
            return True

        specificity_markers = [
            "команда",
            "метрик",
            "конверс",
            "выруч",
            "задач",
            "релиз",
            "спринт",
            "а/б",
            "ab-test",
            "stakeholder",
            "roadmap",
            "пользовател",
            "kpi",
            "%",
        ]
        return any(marker in normalized for marker in specificity_markers)

    @classmethod
    def is_vague_answer(cls, text: str) -> bool:
        token_count = len([w for w in text.split() if w.strip()])
        if token_count < 4:
            return True

        vague_patterns = [
            r"\bit depends\b",
            r"\bnot sure\b",
            r"\bmaybe\b",
            r"\busually\b",
            r"\bkind of\b",
            r"\bзависит\b",
            r"\bне знаю\b",
            r"\bнаверное\b",
            r"\bобычно\b",
            r"\bпримерно\b",
            r"\bкак-то\b",
        ]
        if any(re.search(pattern, text.lower()) for pattern in vague_patterns):
            return token_count < 14

        if token_count < 7 and not cls._has_specificity_markers(text):
            return True

        return False

    @classmethod
    def is_low_signal_answer(cls, text: str) -> bool:
        if cls.is_non_assessment_answer(text):
            return True

        words = [w for w in re.split(r"\s+", text.lower().strip()) if w]
        if len(words) < 4:
            return True

        if cls.is_vague_answer(text) and not cls._has_specificity_markers(text):
            return True

        return False

    @classmethod
    def has_minimum_signal(cls, turns: list[Turn]) -> bool:
        meaningful = 0
        for turn in turns:
            if turn.role != "user" or turn.dimension is None:
                continue
            if cls.is_low_signal_answer(turn.content):
                continue
            meaningful += 1

        return meaningful >= 4

    @staticmethod
    def answer_format_hint_for_dimension(dimension: str | None) -> str:
        hints = {
            "scope_responsibility": "Нужен один кейс: контекст, масштаб и ваша личная зона ответственности.",
            "impact": "Нужен один кейс: ваши действия и конкретный результат в метриках или поведении пользователей.",
            "uncertainty_tolerance": "Нужен один кейс: неопределенность, ваше решение и итог.",
            "planning_horizon": "Нужен один кейс: горизонт планирования, ключевые этапы и результат.",
            "hard_craft": "Нужен один кейс: что именно вы сделали руками и как это повысило качество решения.",
            "hard_systems": "Нужен один кейс: системное улучшение и как оно масштабировалось.",
            "hard_product_business": "Нужен один кейс: как вы балансировали UX и бизнес-цели и к чему пришли.",
            "soft_communication_influence": "Нужен один кейс: кого убеждали, каким аргументом и какой был итог.",
            "management": "Нужен один кейс: как вы развивали людей и какой эффект получили в команде.",
        }
        return hints.get(
            dimension or "",
            "Нужен один реальный кейс: контекст, ваш личный вклад, результат.",
        )

    @classmethod
    def _extract_topic_phrase(cls, text: str, max_words: int = 7) -> str:
        if cls.is_non_assessment_answer(text):
            return ""

        cleaned = re.sub(r"\s+", " ", text).strip()
        cleaned = re.sub(r"[^\w\s-]", "", cleaned)
        words = [word for word in cleaned.split() if len(word) > 1]
        if len(words) < 2:
            return ""
        return " ".join(words[:max_words])

    @classmethod
    def build_correction_message(
        cls,
        dimension: str | None,
        last_user_answer: str,
        attempt_index: int = 0,
    ) -> str:
        topic = cls._extract_topic_phrase(last_user_answer)
        prefix = f"Принял, вы говорите про «{topic}». " if topic else ""

        if attempt_index <= 0:
            return prefix + cls.answer_format_hint_for_dimension(dimension)

        if attempt_index == 1:
            return prefix + "Давайте короче и конкретнее: где это было, что сделали вы, какой итог."

        return (
            prefix
            + "Чтобы оценка была точной, дайте один кейс в 2-3 предложениях: контекст -> ваш вклад -> результат."
        )

    @classmethod
    def build_reflective_bridge(cls, last_user_answer: str) -> str | None:
        topic = cls._extract_topic_phrase(last_user_answer)
        if not topic:
            return None
        return f"Принял, кейс про «{topic}»."

    @staticmethod
    def build_recent_turns_payload(turns: list[Turn], limit: int = 12) -> list[dict[str, str]]:
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

    @classmethod
    def build_memory_points(cls, turns: list[Turn], limit: int = 8) -> list[str]:
        points: list[str] = []
        seen: set[str] = set()

        for turn in reversed(turns):
            if turn.role != "user" or turn.dimension is None:
                continue
            if cls.is_low_signal_answer(turn.content):
                continue

            words = cls._normalize_text_line(turn.content).split()
            if len(words) < 6:
                continue

            snippet = " ".join(words[:24])
            key = f"{turn.dimension}:{cls._normalize_for_compare(snippet)}"
            if key in seen:
                continue

            seen.add(key)
            dimension_name = DIMENSION_DISPLAY.get(turn.dimension, turn.dimension)
            points.append(f"{dimension_name}: {snippet}")
            if len(points) >= limit:
                break

        points.reverse()
        return points

    def choose_next_dimension(
        self,
        track: str | None,
        answered_dimensions: dict[str, int],
    ) -> str:
        required_dimensions = required_dimensions_for_track(track)

        for dimension in HIGH_VARIANCE_PRIORITY:
            if dimension in required_dimensions and answered_dimensions.get(dimension, 0) == 0:
                return dimension

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

        good_turns = [
            turn
            for turn in user_turns
            if turn.dimension is not None and not self.is_low_signal_answer(turn.content)
        ]
        quality = len(good_turns) / max(1, len(user_turns)) if user_turns else 0.0

        confidence = (0.5 * coverage) + (0.3 * depth) + (0.2 * quality)
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

    @staticmethod
    def interview_phase(question_count: int) -> str:
        if question_count <= 3:
            return "discovery"
        if question_count <= 8:
            return "core"
        return "deep"

    @staticmethod
    def _normalize_text_line(text: str) -> str:
        return " ".join(text.strip().split())

    @classmethod
    def _normalize_for_compare(cls, text: str) -> str:
        normalized = cls._normalize_text_line(text).lower()
        normalized = re.sub(r"[^\w\s]", "", normalized)
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized.strip()

    @classmethod
    def _tokenize_for_similarity(cls, text: str) -> set[str]:
        normalized = cls._normalize_for_compare(text)
        tokens = {token for token in normalized.split() if len(token) > 2}
        return {token for token in tokens if token not in QUESTION_SIMILARITY_STOPWORDS}

    @classmethod
    def _question_similarity(cls, left: str, right: str) -> float:
        left_tokens = cls._tokenize_for_similarity(left)
        right_tokens = cls._tokenize_for_similarity(right)
        if not left_tokens or not right_tokens:
            return 0.0

        union = left_tokens | right_tokens
        if not union:
            return 0.0

        overlap = left_tokens & right_tokens
        return len(overlap) / len(union)

    @classmethod
    def _is_complex_question(cls, question: str) -> bool:
        q = cls._normalize_text_line(question)
        if not q:
            return True

        if len(q.replace("?", "").split()) > 16:
            return True
        if q.count("?") > 1:
            return True
        if q.count(",") >= 2 or ";" in q:
            return True

        return False

    @classmethod
    def _is_repeated_question(cls, question: str, turns: list[Turn], lookback: int = 10) -> bool:
        normalized = cls._normalize_for_compare(question)
        if not normalized:
            return False

        recent_questions = [
            turn.content
            for turn in turns[-lookback:]
            if turn.role == "assistant" and turn.dimension is not None
        ]

        for recent in recent_questions:
            recent_normalized = cls._normalize_for_compare(recent)
            if not recent_normalized:
                continue
            if normalized == recent_normalized:
                return True
            if normalized in recent_normalized or recent_normalized in normalized:
                return True
            if cls._question_similarity(question, recent) >= 0.72:
                return True

        return False

    @classmethod
    def _sanitize_probe(cls, probe: str | None) -> str | None:
        if not probe:
            return None

        p = cls._normalize_text_line(probe)
        if not p:
            return None

        words = p.split()
        if len(words) > 12:
            p = " ".join(words[:12]).rstrip(".!") + "?"

        if not p.endswith("?"):
            p = p.rstrip(".!") + "?"

        return p

    @classmethod
    def _fallback_probe_for_dimension(cls, target_dimension: str | None, variant_index: int = 0) -> str:
        probes = FALLBACK_PROBE_BANK.get(target_dimension or "", [])
        if probes:
            return probes[variant_index % len(probes)]
        return "Уточните, пожалуйста: что сделали лично вы и к какому результату это привело?"

    @classmethod
    def _fallback_question_for_dimension(
        cls,
        target_dimension: str,
        turns: list[Turn],
        variant_index: int = 0,
    ) -> NextQuestion:
        questions = FALLBACK_QUESTION_BANK.get(target_dimension, [])
        if not questions:
            questions = ["Коротко расскажите о недавнем кейсе: что сделали вы и какой был результат?"]

        for offset in range(len(questions)):
            candidate = questions[(variant_index + offset) % len(questions)]
            if not cls._is_repeated_question(candidate, turns):
                return NextQuestion(
                    question=candidate,
                    follow_up_probe=cls._fallback_probe_for_dimension(target_dimension, variant_index + offset),
                )

        return NextQuestion(
            question=questions[variant_index % len(questions)],
            follow_up_probe=cls._fallback_probe_for_dimension(target_dimension, variant_index),
        )

    @classmethod
    def _sanitize_question(
        cls,
        question: str,
        target_dimension: str,
        turns: list[Turn],
        variant_index: int = 0,
    ) -> str:
        q = cls._normalize_text_line(question)
        if not q:
            return cls._fallback_question_for_dimension(target_dimension, turns, variant_index).question

        if "?" in q:
            q = q.split("?", 1)[0].strip()

        if not q:
            return cls._fallback_question_for_dimension(target_dimension, turns, variant_index).question

        if not q.endswith("?"):
            q = q.rstrip(".!") + "?"

        if cls._is_complex_question(q) or cls._is_repeated_question(q, turns):
            return cls._fallback_question_for_dimension(target_dimension, turns, variant_index).question

        return q

    @classmethod
    def _last_user_assessment_answer(cls, turns: list[Turn]) -> str:
        for turn in reversed(turns):
            if turn.role != "user" or turn.dimension is None:
                continue
            if cls.is_non_assessment_answer(turn.content):
                continue
            content = turn.content.strip()
            if content:
                return content
        return ""

    def build_follow_up_probe(
        self,
        target_dimension: str | None,
        last_user_answer: str,
        attempt_index: int = 0,
    ) -> str:
        topic = self._extract_topic_phrase(last_user_answer)
        if topic:
            contextual = [
                f"Если берем «{topic}», что сделали именно вы?",
                f"В кейсе «{topic}» какой был результат одним фактом?",
            ]
            return contextual[attempt_index % len(contextual)]

        return self._fallback_probe_for_dimension(target_dimension, attempt_index)

    async def generate_contextual_correction(
        self,
        target_dimension: str | None,
        last_user_answer: str,
        turns: list[Turn],
        attempt_index: int = 0,
    ) -> CorrectionReply:
        format_hint = self.answer_format_hint_for_dimension(target_dimension)
        recent_turns = self.build_recent_turns_payload(turns, limit=10)

        try:
            reply = await self.openai_service.generate_correction_reply(
                target_dimension=target_dimension,
                last_user_answer=last_user_answer,
                answer_format_hint=format_hint,
                recent_turns=recent_turns,
                attempt_index=attempt_index,
            )
            response = self._normalize_text_line(reply.response)
            probe = self._sanitize_probe(reply.follow_up_probe)
            if response:
                return CorrectionReply(response=response, follow_up_probe=probe)
        except OpenAIServiceError as exc:
            logger.warning("Contextual correction generation failed, using fallback: %s", exc)

        fallback_response = self.build_correction_message(
            dimension=target_dimension,
            last_user_answer=last_user_answer,
            attempt_index=attempt_index,
        )
        fallback_probe = self.build_follow_up_probe(
            target_dimension=target_dimension,
            last_user_answer=last_user_answer,
            attempt_index=attempt_index,
        )
        return CorrectionReply(response=fallback_response, follow_up_probe=fallback_probe)

    async def generate_assessment_question(
        self,
        session: Session,
        profile: dict[str, Any],
        turns: list[Turn],
        answered_dimensions: dict[str, int],
    ) -> tuple[str, NextQuestion]:
        target_dimension = self.choose_next_dimension(session.track_preference, answered_dimensions)
        dimension_matrix = self.matrix["dimensions"][target_dimension]
        recent_turns = self.build_recent_turns_payload(turns, limit=8)
        memory_points = self.build_memory_points(turns, limit=5)
        recent_assistant_questions = [
            turn.content
            for turn in turns
            if turn.role == "assistant" and turn.dimension is not None
        ][-8:]

        asked_for_dimension = sum(
            1
            for turn in turns
            if turn.role == "assistant" and turn.dimension == target_dimension
        )

        last_user_answer = self._last_user_assessment_answer(turns)
        interview_phase = self.interview_phase(session.question_count)

        try:
            raw_question = await self.openai_service.generate_next_question(
                target_dimension=target_dimension,
                profile=profile,
                evidence_summary=session.evidence_summary,
                recent_turns=recent_turns,
                recent_assistant_questions=recent_assistant_questions,
                dimension_matrix=dimension_matrix,
                memory_points=memory_points,
                last_user_answer=last_user_answer,
                interview_phase=interview_phase,
                question_index=session.question_count + 1,
                asked_for_dimension=asked_for_dimension,
            )
        except OpenAIServiceError as exc:
            logger.warning("Question generation failed for %s, using fallback: %s", target_dimension, exc)
            fallback = self._fallback_question_for_dimension(target_dimension, turns, asked_for_dimension)
            return target_dimension, fallback

        question = self._sanitize_question(
            raw_question.question,
            target_dimension,
            turns,
            asked_for_dimension,
        )
        probe = self._sanitize_probe(raw_question.follow_up_probe)

        if not probe:
            probe = self.build_follow_up_probe(
                target_dimension=target_dimension,
                last_user_answer=last_user_answer,
                attempt_index=0,
            )

        return target_dimension, NextQuestion(question=question, follow_up_probe=probe)

    async def refresh_evidence_summary_if_needed(
        self,
        session: Session,
        turns: list[Turn],
    ) -> tuple[str, int] | None:
        unsummarized_turns = turns[session.summarized_turn_count :]
        if len(unsummarized_turns) < 8:
            return None

        payload = [
            {
                "role": turn.role,
                "dimension": turn.dimension or "",
                "content": turn.content,
            }
            for turn in unsummarized_turns
        ]

        try:
            updated_summary = await self.openai_service.summarize_evidence(
                existing_summary=session.evidence_summary,
                turns_to_summarize=payload,
            )
        except OpenAIServiceError as exc:
            logger.warning("Evidence summary refresh skipped due to model error: %s", exc)
            return None

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
