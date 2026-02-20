from __future__ import annotations

import json
import logging
from typing import Any

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ContextTypes

from .config import MATRIX_PATH
from .constants import DIMENSION_DISPLAY, PROFILE_FIELDS, WARMUP_QUESTIONS
from .database import Database
from .dialogue import DialogueManager
from .grading import GradeEngine
from .matrix import MatrixError, load_matrix, required_dimensions_for_track
from .models import Session, Turn
from .openai_service import OpenAIServiceError
from .reporting import ReportBuilder

logger = logging.getLogger(__name__)


NEW_COMMANDS_HINT = "Используйте команды: /start, /status, /result, /matrices, /reload_matrix, /reset, /help"


def _service(context: ContextTypes.DEFAULT_TYPE, key: str) -> Any:
    return context.application.bot_data[key]


def _chunk_text(text: str, limit: int = 3800) -> list[str]:
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    cursor = 0
    while cursor < len(text):
        next_cursor = min(cursor + limit, len(text))
        if next_cursor < len(text):
            split = text.rfind("\n", cursor, next_cursor)
            if split > cursor:
                next_cursor = split
        chunks.append(text[cursor:next_cursor].strip())
        cursor = next_cursor
    return [c for c in chunks if c]


async def _send_report(update: Update, report_markdown: str) -> None:
    if update.effective_message is None:
        return
    for chunk in _chunk_text(report_markdown):
        await update.effective_message.reply_text(chunk)


def _latest_assistant_dimension(turns: list[Turn]) -> str | None:
    for turn in reversed(turns):
        if turn.role == "assistant" and turn.dimension:
            return turn.dimension
    return None


def _build_warmup_question(index: int, profile: dict[str, Any]) -> str:
    if index <= 0:
        return "Коротко: какая у вас сейчас роль и зона ответственности?"
    if index == 1:
        return "В каком продукте вы работаете и кто ваш ключевой пользователь?"
    if index == 2:
        return "Сколько лет в продукте и какой трек вам ближе: IC или менеджмент?"

    return "Добавьте, пожалуйста, немного контекста по текущей роли."


def _short_ack(text: str) -> str:
    tokens = len(text.split())
    if tokens <= 5:
        return "Понял, спасибо."
    if tokens <= 14:
        return "Отлично, понял контекст."
    return "Спасибо, это полезный контекст."


def _bridge_for_dimension(dimension: str | None) -> str:
    bridge_map = {
        "scope_responsibility": "Дальше про масштаб.",
        "impact": "Хорошо, теперь про результат.",
        "uncertainty_tolerance": "Понял. Теперь про решения в неопределенности.",
        "planning_horizon": "Отлично. Следующий фокус - планирование.",
        "hard_craft": "Перейдем к вашему крафту.",
        "hard_systems": "Теперь про системность в работе.",
        "hard_product_business": "Хочу понять продуктовый угол.",
        "soft_communication_influence": "Супер, теперь про влияние на команду.",
        "management": "Давайте про управление людьми.",
    }
    if not dimension:
        return "Продолжаем."
    return bridge_map.get(dimension, "Продолжаем.")


def _compose_question_message(question: str, dimension: str | None) -> str:
    return f"{_bridge_for_dimension(dimension)}\n{question}"


def _supplemental_names(matrix: dict[str, Any]) -> list[str]:
    supplemental = matrix.get("supplemental_matrices", [])
    if not isinstance(supplemental, list):
        return []

    names: list[str] = []
    for item in supplemental:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if name:
            names.append(name)
    return names


async def legacy_command_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_message is None:
        return

    await update.effective_message.reply_text(
        "Эта команда больше не используется в версии v2.\n" + NEW_COMMANDS_HINT
    )


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_message is None or update.effective_user is None:
        return

    db: Database = _service(context, "db")
    active_session = db.get_active_session(update.effective_user.id)

    welcome_lines = [
        "Живое интервью по грейду продуктового дизайнера.",
        "Формат: один короткий вопрос за раз (обычно 10-15 минут).",
        "В конце: предполагаемый уровень, сильные стороны, зоны роста и план на 90 дней.",
    ]

    if active_session:
        welcome_lines.append("У вас уже есть активная сессия: /status для продолжения или /reset для перезапуска.")

    keyboard = InlineKeyboardMarkup(
        [[InlineKeyboardButton("Начать ассессмент", callback_data="start_assessment")]]
    )

    await update.effective_message.reply_text("\n".join(welcome_lines), reply_markup=keyboard)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_message is None:
        return
    await update.effective_message.reply_text(
        "Команды:\n"
        "/start - запуск и кнопка старта\n"
        "/reset - сброс активной сессии\n"
        "/status - текущий прогресс\n"
        "/result - итоговый отчет\n"
        "/matrices - загруженные матрицы калибровки\n"
        "/reload_matrix - перечитать матрицы из data/matrices\n"
        "/help - подсказка"
    )


async def matrices_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_message is None:
        return

    dialogue: DialogueManager = _service(context, "dialogue")
    matrix = dialogue.matrix
    names = _supplemental_names(matrix)

    if names:
        await update.effective_message.reply_text(
            "Загружены supplemental-матрицы:\n- " + "\n- ".join(names)
        )
        return

    await update.effective_message.reply_text(
        "Сейчас supplemental-матриц нет. Добавьте JSON-файлы в data/matrices и вызовите /reload_matrix."
    )


async def reload_matrix_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_message is None:
        return

    dialogue: DialogueManager = _service(context, "dialogue")
    grader: GradeEngine = _service(context, "grader")

    try:
        matrix = load_matrix(MATRIX_PATH)
    except MatrixError as exc:
        await update.effective_message.reply_text(f"Не удалось перечитать матрицу: {exc}")
        return

    dialogue.matrix = matrix
    grader.matrix = matrix

    names = _supplemental_names(matrix)
    summary = f"Матрица перезагружена. supplemental: {len(names)}"
    if names:
        summary += "\n- " + "\n- ".join(names)
    await update.effective_message.reply_text(summary)


async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_message is None or update.effective_user is None:
        return

    db: Database = _service(context, "db")
    pending_probes: dict[int, str | None] = _service(context, "pending_probes")

    active = db.get_active_session(update.effective_user.id)
    deleted = db.reset_active_session(update.effective_user.id)
    if active:
        pending_probes.pop(active.id, None)

    if deleted:
        await update.effective_message.reply_text("Текущая сессия сброшена. Напишите /start, чтобы начать заново.")
    else:
        await update.effective_message.reply_text("Активная сессия не найдена.")


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_message is None or update.effective_user is None:
        return

    db: Database = _service(context, "db")
    dialogue: DialogueManager = _service(context, "dialogue")

    session = db.get_active_session(update.effective_user.id)
    if session is None:
        latest = db.get_latest_session(update.effective_user.id)
        if latest and latest.status == "completed":
            await update.effective_message.reply_text("Сейчас активной сессии нет. Последняя уже завершена, посмотрите /result.")
        else:
            await update.effective_message.reply_text("Активной сессии нет. Напишите /start, чтобы начать.")
        return

    if session.warmup_index < len(WARMUP_QUESTIONS):
        await update.effective_message.reply_text(
            f"Идет вводный блок: отвечено {session.warmup_index}/{len(WARMUP_QUESTIONS)}. Ответьте на текущий вопрос."
        )
        return

    answered = db.get_answered_dimensions(session.id)
    snapshot = dialogue.progress_snapshot(session, session.track_preference, answered)
    covered_names = [DIMENSION_DISPLAY.get(d, d) for d in snapshot.covered_dimensions]
    remaining_names = [DIMENSION_DISPLAY.get(d, d) for d in snapshot.remaining_dimensions]

    approx_remaining_questions = max(0, session.max_questions - session.question_count)

    lines = [
        f"Прогресс: покрыто {len(snapshot.covered_dimensions)}/{len(snapshot.required_dimensions)} слоев ({snapshot.coverage_ratio:.0%}).",
        f"Задано вопросов: {session.question_count}/{session.max_questions}.",
        f"Оценка уверенности: {snapshot.confidence_estimate:.2f}.",
        f"Ориентировочно осталось вопросов: {approx_remaining_questions}.",
        "",
        f"Уже покрыто: {', '.join(covered_names) if covered_names else 'пока нет'}",
        f"Дальше фокус: {', '.join(remaining_names[:5]) if remaining_names else 'финализация отчета'}",
    ]
    await update.effective_message.reply_text("\n".join(lines))


async def result_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_message is None or update.effective_user is None:
        return

    db: Database = _service(context, "db")

    active = db.get_active_session(update.effective_user.id)
    if active is not None:
        if active.warmup_index < len(WARMUP_QUESTIONS):
            await update.effective_message.reply_text(
                "Ассессмент еще не завершен. Сначала пройдите вводные и интервью-вопросы."
            )
            return

        answered = db.get_answered_dimensions(active.id)
        required = required_dimensions_for_track(active.track_preference)
        missing = [DIMENSION_DISPLAY.get(d, d) for d in required if answered.get(d, 0) == 0]

        if missing:
            await update.effective_message.reply_text(
                "Ассессмент еще идет. Не хватает evidence по слоям: " + ", ".join(missing[:6])
            )
        else:
            await update.effective_message.reply_text(
                "Ассессмент еще в процессе. Нужно еще немного данных, чтобы повысить уверенность оценки."
            )
        return

    latest = db.get_latest_session(update.effective_user.id)
    if latest is None or latest.status != "completed" or not latest.final_report_markdown:
        await update.effective_message.reply_text("Готового результата пока нет. Запустите ассессмент через /start.")
        return

    await _send_report(update, latest.final_report_markdown)
    if latest.export_path:
        await update.effective_message.reply_text(f"JSON-экспорт: {latest.export_path}")


async def _start_assessment(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_message is None or update.effective_user is None:
        return

    db: Database = _service(context, "db")
    max_questions: int = _service(context, "max_questions")

    active = db.get_active_session(update.effective_user.id)
    if active:
        await update.effective_message.reply_text(
            "У вас уже есть активный ассессмент. Продолжайте отвечать или используйте /reset."
        )
        return

    session = db.create_session(update.effective_user.id, max_questions=max_questions)

    first_q = _build_warmup_question(0, profile={})
    db.append_turn(session.id, role="assistant", content=first_q, dimension=None)
    await update.effective_message.reply_text(f"Отлично, начинаем.\n{first_q}")


async def start_assessment_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.callback_query is None:
        return

    await update.callback_query.answer()
    await _start_assessment(update, context)


async def _ask_first_assessment_question(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    session: Session,
) -> None:
    if update.effective_message is None:
        return

    db: Database = _service(context, "db")
    dialogue: DialogueManager = _service(context, "dialogue")
    pending_probes: dict[int, str | None] = _service(context, "pending_probes")

    turns = db.list_turns(session.id)
    profile = dialogue.extract_profile(session)
    answered = db.get_answered_dimensions(session.id)

    target_dimension, next_question = await dialogue.generate_assessment_question(
        session=session,
        profile=profile,
        turns=turns,
        answered_dimensions=answered,
    )

    db.append_turn(session.id, role="assistant", content=next_question.question, dimension=target_dimension)
    db.increment_question_count(session.id)
    pending_probes[session.id] = next_question.follow_up_probe

    await update.effective_message.reply_text(_compose_question_message(next_question.question, target_dimension))


async def _finalize_assessment(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    session: Session,
) -> None:
    if update.effective_message is None:
        return

    db: Database = _service(context, "db")
    dialogue: DialogueManager = _service(context, "dialogue")
    grader: GradeEngine = _service(context, "grader")
    reporter: ReportBuilder = _service(context, "reporter")
    pending_probes: dict[int, str | None] = _service(context, "pending_probes")

    await update.effective_message.reply_text("Спасибо. Формирую короткий итог...")

    session = db.get_session(session.id) or session
    turns = db.list_turns(session.id)
    profile = dialogue.extract_profile(session)

    try:
        grade = await grader.run_grade(session=session, profile=profile, turns=turns)
        artifacts = reporter.export_report(session=session, profile=profile, grade=grade)
    except Exception as exc:
        logger.exception("Final report generation failed: %s", exc)
        await update.effective_message.reply_text(
            "Не удалось собрать итоговый отчет с первого раза. Напишите любое сообщение, и я повторю финализацию."
        )
        return

    db.mark_completed(
        session_id=session.id,
        final_report_markdown=artifacts.markdown,
        final_report_json=json.dumps(artifacts.report_json, ensure_ascii=True),
        export_path=artifacts.export_path,
        confidence_estimate=float(grade.get("confidence", 0.0)),
    )
    pending_probes.pop(session.id, None)

    await _send_report(update, artifacts.markdown)
    await update.effective_message.reply_text(f"JSON-экспорт сохранен: {artifacts.export_path}")


async def text_message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_message is None or update.effective_user is None:
        return

    text = (update.effective_message.text or "").strip()
    if not text:
        return

    db: Database = _service(context, "db")
    dialogue: DialogueManager = _service(context, "dialogue")
    pending_probes: dict[int, str | None] = _service(context, "pending_probes")
    confidence_threshold: float = _service(context, "confidence_threshold")

    session = db.get_active_session(update.effective_user.id)
    if session is None:
        await update.effective_message.reply_text("Активной сессии нет. Напишите /start и нажмите «Начать ассессмент».")
        return

    try:
        if session.warmup_index < len(WARMUP_QUESTIONS):
            warmup_idx = session.warmup_index
            db.append_turn(session.id, role="user", content=text, dimension=None)
            db.update_profile_field(session.id, PROFILE_FIELDS[warmup_idx], text)
            db.increment_warmup_index(session.id)

            if warmup_idx == len(WARMUP_QUESTIONS) - 1:
                track_pref = dialogue.infer_track_preference(text)
                db.update_track_preference(session.id, track_pref)

            session = db.get_session(session.id)
            if session is None:
                raise RuntimeError("Session disappeared")

            if session.warmup_index < len(WARMUP_QUESTIONS):
                profile = dialogue.extract_profile(session)
                next_warmup = _build_warmup_question(session.warmup_index, profile)
                db.append_turn(session.id, role="assistant", content=next_warmup, dimension=None)
                await update.effective_message.reply_text(f"{_short_ack(text)}\n{next_warmup}")
                return

            await update.effective_message.reply_text(f"{_short_ack(text)}\nПерейдем к основному интервью.")
            await _ask_first_assessment_question(update, context, session)
            return

        turns = db.list_turns(session.id)
        current_dimension = _latest_assistant_dimension(turns)
        db.append_turn(session.id, role="user", content=text, dimension=current_dimension)

        turns = db.list_turns(session.id)
        session = db.get_session(session.id) or session

        refreshed = await dialogue.refresh_evidence_summary_if_needed(session, turns)
        if refreshed:
            new_summary, summarized_turn_count = refreshed
            db.update_evidence_summary(
                session_id=session.id,
                evidence_summary=new_summary,
                summarized_turn_count=summarized_turn_count,
            )
            session = db.get_session(session.id) or session

        answered = db.get_answered_dimensions(session.id)
        user_turns = [t for t in turns if t.role == "user" and t.dimension is not None]
        confidence = dialogue.estimate_confidence(session.track_preference, answered, user_turns)
        db.update_confidence(session.id, confidence)
        session = db.get_session(session.id) or session

        if dialogue.should_finish(
            session=session,
            track=session.track_preference,
            answered_dimensions=answered,
            confidence_estimate=confidence,
            confidence_threshold=confidence_threshold,
        ):
            await _finalize_assessment(update, context, session)
            return

        use_probe = dialogue.is_vague_answer(text)
        if use_probe:
            probe = pending_probes.get(session.id) or (
                "Можете привести один конкретный пример: контекст, ваш вклад и результат?"
            )
            db.append_turn(session.id, role="assistant", content=probe, dimension=current_dimension)
            db.increment_question_count(session.id)
            pending_probes[session.id] = None
            await update.effective_message.reply_text(f"{_short_ack(text)}\n{probe}")
            return

        profile = dialogue.extract_profile(session)
        target_dimension, next_question = await dialogue.generate_assessment_question(
            session=session,
            profile=profile,
            turns=turns,
            answered_dimensions=answered,
        )

        db.append_turn(
            session.id,
            role="assistant",
            content=next_question.question,
            dimension=target_dimension,
        )
        db.increment_question_count(session.id)
        pending_probes[session.id] = next_question.follow_up_probe
        await update.effective_message.reply_text(_compose_question_message(next_question.question, target_dimension))

    except OpenAIServiceError as exc:
        logger.exception("OpenAIServiceError: %s", exc)
        await update.effective_message.reply_text(
            "Временная ошибка модели. Пожалуйста, повторите последнее сообщение через несколько секунд."
        )
    except Exception as exc:  # pragma: no cover - defensive branch
        logger.exception("Unexpected error in message handler: %s", exc)
        await update.effective_message.reply_text(
            "Произошла ошибка при обработке ответа. Попробуйте еще раз."
        )
