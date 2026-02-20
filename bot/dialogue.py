from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from .constants import DIMENSION_DISPLAY, HIGH_VARIANCE_PRIORITY
from .matrix import required_dimensions_for_track
from .models import NextQuestion, Session, Turn
from .openai_service import OpenAIService, OpenAIServiceError

logger = logging.getLogger(__name__)


FALLBACK_QUESTION_BANK: dict[str, list[str]] = {
    "scope_responsibility": [
        "Какой проект за последний год был самым масштабным в вашей зоне?",
        "Где зона вашей ответственности заметно расширилась за последний год?",
        "В какой задаче вы отвечали за самый большой кусок продукта?",
    ],
    "impact": [
        "Какое ваше решение дало самый заметный эффект для продукта?",
        "Где ваш вклад сильнее всего сдвинул бизнес-метрику?",
        "Какой кейс лучше всего показывает влияние вашей работы на результат?",
    ],
    "uncertainty_tolerance": [
        "Как вы действовали, когда данных было недостаточно?",
        "Какой риск вы осознанно приняли в условиях неопределенности?",
        "Когда фактов не хватало, на что вы опирались при выборе решения?",
    ],
    "planning_horizon": [
        "На какой горизонт вы обычно планируете дизайн-работу?",
        "Как вы держите фокус команды на цели дальше ближайшего спринта?",
        "Как выглядит ваш подход к планированию на квартал?",
    ],
    "hard_craft": [
        "Какой кейс лучше всего показывает ваш уровень крафта?",
        "В каком проекте вы сильнее всего прокачали качество интерфейса?",
        "Где ваше визуальное и UX-решение заметно улучшило продукт?",
    ],
    "hard_systems": [
        "Что вы лично улучшили в дизайн-системе за последний год?",
        "Какое системное решение вы внедрили, чтобы команда работала быстрее?",
        "Где вы выстроили процесс, который масштабируется на несколько команд?",
    ],
    "hard_product_business": [
        "Когда приходилось выбирать между UX и бизнес-целями?",
        "Как вы связываете дизайн-решения с метриками продукта?",
        "В каком кейсе вы пересобрали решение ради бизнес-результата?",
    ],
    "soft_communication_influence": [
        "Как вы убеждали команду, когда мнения расходились?",
        "Как вы проводите сложные решения без формальной власти?",
        "В каком кейсе вам удалось изменить позицию стейкхолдеров?",
    ],
    "management": [
        "Как вы развиваете дизайнеров в команде?",
        "Как вы даете обратную связь и растите уровень команды?",
        "Как вы принимаете кадровые решения в своей функции?",
    ],
    "culture_ownership": [
        "Когда вы брали на себя проблему вне формальной зоны ответственности?",
        "Где вы довели задачу до результата, хотя это не входило в ваш scope?",
        "В каком кейсе вы стали owner проблемы, а не только своего участка?",
    ],
    "culture_proactivity": [
        "Что вы улучшили по собственной инициативе?",
        "Какую инициативу вы запустили без прямого запроса сверху?",
        "Что вы поменяли в процессе, чтобы команде стало проще работать?",
    ],
    "culture_quality_bar": [
        "Как вы удерживаете качество, когда сроки сжаты?",
        "В каком кейсе вы не опустили планку качества под дедлайн?",
        "Как вы определяете минимально допустимый quality bar?",
    ],
    "culture_collaboration": [
        "Как вы разруливали сложный кросс-функциональный конфликт?",
        "Где вам удалось выстроить сильное партнерство с продуктом и разработкой?",
        "Как вы синхронизируете команды вокруг общего решения?",
    ],
    "culture_learning": [
        "Как вы превращаете ошибки в улучшения процесса?",
        "Какой ваш недавний фейл стал полезным уроком для команды?",
        "Как вы внедряете системное обучение в команде?",
    ],
    "culture_integrity_safety": [
        "Когда вы отстаивали решение в пользу этики или безопасности пользователя?",
        "Был ли кейс, где вы остановили запуск из-за риска для пользователей?",
        "Как вы балансируете бизнес-давление и безопасность пользователя?",
    ],
}

FALLBACK_PROBE_BANK: dict[str, list[str]] = {
    "scope_responsibility": [
        "Что в этом кейсе было вашей личной зоной ответственности?",
        "Какой масштаб был у задачи: команда, продукт, бизнес?",
    ],
    "impact": [
        "Какой конкретный эффект это дало в цифрах или поведении пользователей?",
        "По каким метрикам вы поняли, что решение сработало?",
    ],
    "uncertainty_tolerance": [
        "Какой главный риск вы видели и как его снижали?",
        "Что стало решающим аргументом в вашем выборе?",
    ],
    "planning_horizon": [
        "Как вы проверяете, что план остается реалистичным?",
        "Какие риски вы закладываете в план заранее?",
    ],
    "hard_craft": [
        "Что в этом решении было самым сложным на уровне крафта?",
        "Как вы проверяли качество и консистентность решения?",
    ],
    "hard_systems": [
        "Что из этого теперь переиспользуется другими командами?",
        "Как вы масштабировали это решение за пределы одного кейса?",
    ],
    "hard_product_business": [
        "Как вы объяснили этот выбор с точки зрения бизнеса?",
        "Какая метрика изменилась после вашего решения?",
    ],
    "soft_communication_influence": [
        "Кого было сложнее всего убедить и как вы это сделали?",
        "Какая аргументация в итоге сработала лучше всего?",
    ],
    "management": [
        "Как именно вы растили людей в этом кейсе?",
        "Какие изменения в команде произошли после ваших действий?",
    ],
}

FRUSTRATION_PATTERNS = [
    r"заеб",
    r"достал",
    r"хватит",
    r"надоел",
    r"бесит",
    r"stop",
    r"enough",
]

NON_ASSESSMENT_SHORT_REPLIES = {
    "да",
    "нет",
    "ага",
    "ок",
    "окей",
    "пон",
    "ясно",
    "норм",
    "лол",
    "кек",
    "много",
    "хз",
    "idk",
    "yes",
    "no",
    "haha",
    "hah",
    "xd",
}

OBSCENE_PATTERNS = [
    r"\bх[ую]й",
    r"\bпизд",
    r"\bеб[а-я]*",
    r"\bбля",
    r"\bсук",
    r"\bдроч",
    r"\bпис[кю]",
    r"\bfuck",
    r"\bshit",
    r"\bdick",
    r"\bcock",
]


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
        has_numbers = bool(re.search(r"\d", text))

        if token_count < 4:
            return True
        if token_count < 7 and not has_numbers:
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

        if any(re.search(p, text.lower()) for p in vague_patterns):
            return token_count < 18

        return False

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

        if len(words) == 1 and words[0] in NON_ASSESSMENT_SHORT_REPLIES:
            return True

        has_letters_or_digits = bool(re.search(r"[a-zа-я0-9]", normalized))
        if not has_letters_or_digits:
            return True

        return False

    @classmethod
    def is_low_signal_answer(cls, text: str) -> bool:
        if cls.is_non_assessment_answer(text):
            return True

        normalized = text.lower().strip()
        words = [w for w in re.split(r"\s+", normalized) if w]
        if not words:
            return True

        has_numbers = bool(re.search(r"\d", normalized))
        if len(words) < 5:
            return True
        if len(words) < 8 and not has_numbers:
            return True

        if cls.is_vague_answer(text):
            if has_numbers and len(words) >= 5:
                return False
            return True

        return False

    @classmethod
    def has_minimum_signal(cls, turns: list[Turn]) -> bool:
        meaningful = 0
        for turn in turns:
            if turn.role != "user" or turn.dimension is None:
                continue
            if cls.is_non_assessment_answer(turn.content):
                continue
            if len(turn.content.split()) < 5:
                continue
            meaningful += 1

        return meaningful >= 4

    @staticmethod
    def answer_format_hint_for_dimension(dimension: str | None) -> str:
        hints = {
            "scope_responsibility": "Нужен пример в формате: масштаб задачи, ваша зона ответственности, результат.",
            "impact": "Нужен пример в формате: что сделали и какой эффект в метриках или поведении пользователей.",
            "uncertainty_tolerance": "Нужен пример: в чем была неопределенность, какое решение приняли, какой вышел итог.",
            "planning_horizon": "Нужен пример: как планировали, какие этапы/риски выделяли, что получили в итоге.",
            "hard_craft": "Нужен пример: какая дизайн-задача, что сделали руками, как это улучшило качество.",
            "hard_systems": "Нужен пример: какое системное улучшение вы внедрили и как оно масштабировалось.",
            "hard_product_business": "Нужен пример: где балансировали UX и бизнес-цель, и что получилось.",
            "soft_communication_influence": "Нужен пример: кого убеждали, какой аргумент сработал, к какому решению пришли.",
            "management": "Нужен пример: кого и как развивали, какие изменения это дало команде.",
        }
        return hints.get(
            dimension or "",
            "Нужен реальный кейс: контекст, ваш личный вклад и конкретный результат.",
        )

    @classmethod
    def build_correction_message(
        cls,
        dimension: str | None,
        last_user_answer: str,
        attempt_index: int = 0,
    ) -> str:
        topic = cls._extract_topic_phrase(last_user_answer)
        prefix = f"Понял, вы говорите про «{topic}». " if topic else ""

        if attempt_index <= 0:
            return (
                prefix
                + cls.answer_format_hint_for_dimension(dimension)
                + " Один конкретный кейс, без общих фраз."
            )

        if attempt_index == 1:
            return (
                prefix
                + "Давайте проще: где это происходило, что сделали лично вы и какой был результат."
            )

        return (
            prefix
            + "Без конкретики оценка будет неточной. Нужен один кейс в 2-4 предложениях: контекст -> ваш вклад -> результат."
        )

    @classmethod
    def build_reflective_bridge(cls, last_user_answer: str) -> str | None:
        topic = cls._extract_topic_phrase(last_user_answer)
        if not topic:
            return None
        return f"Понял, кейс про «{topic}»."

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
    def _is_complex_question(cls, question: str) -> bool:
        q = cls._normalize_text_line(question)
        if not q:
            return True

        word_count = len(q.replace("?", "").split())
        if word_count > 22:
            return True

        if q.count("?") > 1:
            return True
        if q.count(",") >= 3 or ";" in q:
            return True

        return False

    @classmethod
    def _fallback_probe_for_dimension(cls, target_dimension: str | None, variant_index: int = 0) -> str:
        probes = FALLBACK_PROBE_BANK.get(target_dimension or "", [])
        if probes:
            return probes[variant_index % len(probes)]
        return "Какой у вас был личный вклад и чем это закончилось для продукта?"

    @classmethod
    def _contextual_question_for_dimension(
        cls,
        target_dimension: str,
        last_user_answer: str | None,
    ) -> str | None:
        topic = cls._extract_topic_phrase(last_user_answer or "")
        if not topic:
            return None

        templates = {
            "scope_responsibility": f"Вы упомянули «{topic}». Какой масштаб ответственности был на вашей стороне?",
            "impact": f"В кейсе «{topic}» какой результат вы считаете главным?",
            "uncertainty_tolerance": f"В ситуации «{topic}» где была главная неопределенность и как вы ее закрывали?",
            "planning_horizon": f"Для кейса «{topic}» как вы планировали этапы вперед?",
            "hard_product_business": f"Если взять «{topic}», как вы балансировали пользовательскую ценность и бизнес-цель?",
            "soft_communication_influence": f"В кейсе «{topic}» как вы повлияли на решение команды?",
            "management": f"В контексте «{topic}» как вы управляли людьми и развитием команды?",
        }

        return templates.get(target_dimension)

    @classmethod
    def _fallback_question_for_dimension(
        cls,
        target_dimension: str,
        variant_index: int = 0,
        last_user_answer: str | None = None,
    ) -> NextQuestion:
        contextual_question = cls._contextual_question_for_dimension(target_dimension, last_user_answer)
        if contextual_question:
            return NextQuestion(
                question=contextual_question,
                follow_up_probe=cls._fallback_probe_for_dimension(target_dimension, variant_index),
            )

        questions = FALLBACK_QUESTION_BANK.get(target_dimension, [])
        if questions:
            question = questions[variant_index % len(questions)]
        else:
            question = "Расскажите о недавнем сложном кейсе и вашей роли в нем."

        return NextQuestion(
            question=question,
            follow_up_probe=cls._fallback_probe_for_dimension(target_dimension, variant_index),
        )

    @classmethod
    def _extract_topic_phrase(cls, text: str, max_words: int = 6) -> str:
        if cls.is_non_assessment_answer(text):
            return ""

        cleaned = re.sub(r"\s+", " ", text).strip()
        cleaned = re.sub(r"[^\w\s-]", "", cleaned)
        words = [w for w in cleaned.split() if len(w) > 1]
        if len(words) < 2:
            return ""
        return " ".join(words[:max_words])

    def build_follow_up_probe(
        self,
        target_dimension: str | None,
        last_user_answer: str,
        attempt_index: int = 0,
    ) -> str:
        topic = self._extract_topic_phrase(last_user_answer)
        if topic and target_dimension in {"impact", "scope_responsibility", "uncertainty_tolerance", "management"}:
            contextual_templates = [
                f"Если взять кейс «{topic}», что сделали лично вы и что изменилось в результате?",
                f"В примере «{topic}» какое решение было именно вашим?",
            ]
            return contextual_templates[attempt_index % len(contextual_templates)]

        return self._fallback_probe_for_dimension(target_dimension, attempt_index)

    @classmethod
    def _sanitize_question(
        cls,
        question: str,
        target_dimension: str,
        variant_index: int = 0,
        last_user_answer: str | None = None,
    ) -> str:
        q = cls._normalize_text_line(question)

        if "?" in q:
            q = q.split("?", 1)[0].strip()
        if not q:
            return cls._fallback_question_for_dimension(
                target_dimension,
                variant_index,
                last_user_answer=last_user_answer,
            ).question

        if cls._is_complex_question(q):
            return cls._fallback_question_for_dimension(
                target_dimension,
                variant_index,
                last_user_answer=last_user_answer,
            ).question

        if not q.endswith("?"):
            q = q.rstrip(".!") + "?"

        return q

    @classmethod
    def _sanitize_probe(cls, probe: str | None) -> str | None:
        if not probe:
            return None
        p = cls._normalize_text_line(probe)
        if not p:
            return None

        words = p.split()
        if len(words) > 18:
            p = " ".join(words[:18]).rstrip(".!") + "?"

        if not p.endswith("?"):
            p = p.rstrip(".!") + "?"

        return p

    @classmethod
    def _is_repeated_question(cls, question: str, turns: list[Turn], lookback: int = 8) -> bool:
        normalized = cls._normalize_for_compare(question)
        if not normalized:
            return False

        recent_assistant_questions = [
            cls._normalize_for_compare(turn.content)
            for turn in turns[-lookback:]
            if turn.role == "assistant"
        ]

        return normalized in recent_assistant_questions

    @classmethod
    def _last_user_assessment_answer(cls, turns: list[Turn]) -> str:
        for turn in reversed(turns):
            if turn.role != "user":
                continue
            if turn.dimension is None:
                continue
            if cls.is_non_assessment_answer(turn.content):
                continue
            content = turn.content.strip()
            if content:
                return content
        return ""

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

        asked_for_dimension = sum(
            1
            for turn in turns
            if turn.role == "assistant" and turn.dimension == target_dimension
        )

        last_user_answer = self._last_user_assessment_answer(turns)

        try:
            raw_question = await self.openai_service.generate_next_question(
                target_dimension=target_dimension,
                profile=profile,
                evidence_summary=session.evidence_summary,
                recent_turns=recent_turns,
                dimension_matrix=dimension_matrix,
                last_user_answer=last_user_answer,
            )
        except OpenAIServiceError as exc:
            logger.warning(
                "Question generation failed for %s, using fallback: %s",
                target_dimension,
                exc,
            )
            fallback = self._fallback_question_for_dimension(
                target_dimension,
                asked_for_dimension,
                last_user_answer=last_user_answer,
            )
            return target_dimension, fallback

        question = self._sanitize_question(
            raw_question.question,
            target_dimension,
            asked_for_dimension,
            last_user_answer=last_user_answer,
        )
        probe = self._sanitize_probe(raw_question.follow_up_probe)

        if self._is_complex_question(question) or self._is_repeated_question(question, turns):
            fallback = self._fallback_question_for_dimension(
                target_dimension,
                asked_for_dimension,
                last_user_answer=last_user_answer,
            )
            question = fallback.question
            probe = probe or fallback.follow_up_probe

        if not probe:
            probe = self.build_follow_up_probe(
                target_dimension,
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
