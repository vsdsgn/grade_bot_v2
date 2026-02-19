from __future__ import annotations

import json
import logging
from typing import Any

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ContextTypes

from .constants import DIMENSION_DISPLAY, PROFILE_FIELDS, WARMUP_QUESTIONS
from .database import Database
from .dialogue import DialogueManager
from .grading import GradeEngine
from .matrix import required_dimensions_for_track
from .models import Session, Turn
from .openai_service import OpenAIServiceError
from .reporting import ReportBuilder

logger = logging.getLogger(__name__)


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


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_message is None or update.effective_user is None:
        return

    db: Database = _service(context, "db")
    active_session = db.get_active_session(update.effective_user.id)

    welcome_lines = [
        "Hi! I run a live product design self-assessment interview using competency matrix v2.0.",
        "I will ask one open-ended question at a time, adapt to your answers, and then generate:",
        "1) suggested level, 2) per-layer scores with evidence, 3) strengths/growth areas, 4) next-level roadmap.",
        "",
        "Privacy: your data stays local in SQLite and local JSON exports in /data/exports.",
    ]

    if active_session:
        welcome_lines.append("\nYou already have an active assessment. Use /status to continue or /reset to restart.")

    keyboard = InlineKeyboardMarkup(
        [[InlineKeyboardButton("Start assessment", callback_data="start_assessment")]]
    )

    await update.effective_message.reply_text("\n".join(welcome_lines), reply_markup=keyboard)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_message is None:
        return
    await update.effective_message.reply_text(
        "Commands:\n"
        "/start - show intro and start button\n"
        "/reset - clear current in-progress session\n"
        "/status - show progress and approximate remaining questions\n"
        "/result - return final report (if completed)\n"
        "/help - show this help"
    )


async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_message is None or update.effective_user is None:
        return

    db: Database = _service(context, "db")
    pending_probes: dict[int, str | None] = _service(context, "pending_probes")

    deleted = db.reset_active_session(update.effective_user.id)
    pending_probes.pop(update.effective_user.id, None)

    if deleted:
        await update.effective_message.reply_text("Current session was reset. Use /start to begin again.")
    else:
        await update.effective_message.reply_text("No active session to reset.")


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_message is None or update.effective_user is None:
        return

    db: Database = _service(context, "db")
    dialogue: DialogueManager = _service(context, "dialogue")

    session = db.get_active_session(update.effective_user.id)
    if session is None:
        latest = db.get_latest_session(update.effective_user.id)
        if latest and latest.status == "completed":
            await update.effective_message.reply_text("No active assessment. Your last one is completed; use /result to view it.")
        else:
            await update.effective_message.reply_text("No active assessment. Use /start to begin.")
        return

    if session.warmup_index < len(WARMUP_QUESTIONS):
        await update.effective_message.reply_text(
            f"Warm-up in progress: {session.warmup_index}/{len(WARMUP_QUESTIONS)} answered. "
            "Please answer the current question."
        )
        return

    answered = db.get_answered_dimensions(session.id)
    snapshot = dialogue.progress_snapshot(session, session.track_preference, answered)
    covered_names = [DIMENSION_DISPLAY.get(d, d) for d in snapshot.covered_dimensions]
    remaining_names = [DIMENSION_DISPLAY.get(d, d) for d in snapshot.remaining_dimensions]

    approx_remaining_questions = max(0, session.max_questions - session.question_count)

    lines = [
        f"Progress: {len(snapshot.covered_dimensions)}/{len(snapshot.required_dimensions)} layers covered ({snapshot.coverage_ratio:.0%}).",
        f"Assessment questions asked: {session.question_count}/{session.max_questions}.",
        f"Estimated confidence: {snapshot.confidence_estimate:.2f}.",
        f"Approx questions remaining: {approx_remaining_questions}.",
        "",
        f"Covered: {', '.join(covered_names) if covered_names else 'none yet'}",
        f"Remaining focus: {', '.join(remaining_names[:5]) if remaining_names else 'finalizing'}",
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
                "Assessment is not finished yet. Complete the warm-up and interview questions first."
            )
            return

        answered = db.get_answered_dimensions(active.id)
        required = required_dimensions_for_track(active.track_preference)
        missing = [DIMENSION_DISPLAY.get(d, d) for d in required if answered.get(d, 0) == 0]

        if missing:
            await update.effective_message.reply_text(
                "Assessment is still in progress. Missing evidence in: " + ", ".join(missing[:6])
            )
        else:
            await update.effective_message.reply_text(
                "Assessment is still in progress. I need a few more answers for confidence before finalizing."
            )
        return

    latest = db.get_latest_session(update.effective_user.id)
    if latest is None or latest.status != "completed" or not latest.final_report_markdown:
        await update.effective_message.reply_text("No completed result found yet. Use /start to begin an assessment.")
        return

    await _send_report(update, latest.final_report_markdown)
    if latest.export_path:
        await update.effective_message.reply_text(f"JSON export: {latest.export_path}")


async def _start_assessment(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_message is None or update.effective_user is None:
        return

    db: Database = _service(context, "db")
    max_questions: int = _service(context, "max_questions")

    active = db.get_active_session(update.effective_user.id)
    if active:
        await update.effective_message.reply_text(
            "You already have an active assessment. Continue answering or use /reset to restart."
        )
        return

    session = db.create_session(update.effective_user.id, max_questions=max_questions)

    intro = (
        "Great, let’s begin. I’ll ask 3 quick warm-up questions first, then move into the adaptive assessment.\n"
        "Please answer with concrete examples where possible."
    )
    await update.effective_message.reply_text(intro)

    first_q = WARMUP_QUESTIONS[0]
    db.append_turn(session.id, role="assistant", content=first_q, dimension=None)
    await update.effective_message.reply_text(first_q)


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

    await update.effective_message.reply_text(next_question.question)


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

    await update.effective_message.reply_text("Thanks. Generating your final assessment report...")

    session = db.get_session(session.id) or session
    turns = db.list_turns(session.id)
    profile = dialogue.extract_profile(session)

    grade = await grader.run_grade(session=session, profile=profile, turns=turns)
    artifacts = reporter.export_report(session=session, profile=profile, grade=grade)

    db.mark_completed(
        session_id=session.id,
        final_report_markdown=artifacts.markdown,
        final_report_json=json.dumps(artifacts.report_json, ensure_ascii=True),
        export_path=artifacts.export_path,
        confidence_estimate=float(grade.get("confidence", 0.0)),
    )
    pending_probes.pop(session.id, None)

    await _send_report(update, artifacts.markdown)
    await update.effective_message.reply_text(f"JSON export saved: {artifacts.export_path}")


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
        await update.effective_message.reply_text("No active assessment. Use /start and press Start assessment.")
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
                next_warmup = WARMUP_QUESTIONS[session.warmup_index]
                db.append_turn(session.id, role="assistant", content=next_warmup, dimension=None)
                await update.effective_message.reply_text(next_warmup)
                return

            await update.effective_message.reply_text("Perfect, now let’s move into the competency interview.")
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
            probe = pending_probes.get(session.id) or "Can you share one concrete example, your role, and the outcome?"
            db.append_turn(session.id, role="assistant", content=probe, dimension=current_dimension)
            db.increment_question_count(session.id)
            pending_probes[session.id] = None
            await update.effective_message.reply_text(probe)
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
        await update.effective_message.reply_text(next_question.question)

    except OpenAIServiceError as exc:
        logger.exception("OpenAIServiceError: %s", exc)
        await update.effective_message.reply_text(
            "I hit a temporary model issue. Please resend your last answer in a moment."
        )
    except Exception as exc:  # pragma: no cover - defensive branch
        logger.exception("Unexpected error in message handler: %s", exc)
        await update.effective_message.reply_text(
            "Something went wrong while processing your answer. Please try again."
        )
