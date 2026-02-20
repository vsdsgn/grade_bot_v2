from __future__ import annotations

import logging
import sys

from telegram import BotCommand, BotCommandScopeAllPrivateChats, BotCommandScopeDefault
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, MessageHandler, filters

from bot.config import ConfigError, DB_PATH, EXPORTS_DIR, MATRIX_PATH, ensure_data_dirs, load_config
from bot.database import Database
from bot.dialogue import DialogueManager
from bot.grading import GradeEngine
from bot.handlers import (
    help_command,
    legacy_command_handler,
    matrices_command,
    reload_matrix_command,
    reset_command,
    result_command,
    start_assessment_callback,
    start_command,
    status_command,
    text_message_handler,
)
from bot.matrix import MatrixError, load_matrix
from bot.openai_service import OpenAIService
from bot.reporting import ReportBuilder

logger = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


async def _post_init_set_commands(app: Application) -> None:
    commands = [
        BotCommand("start", "Запуск и старт ассессмента"),
        BotCommand("status", "Текущий прогресс"),
        BotCommand("result", "Итоговый отчет"),
        BotCommand("reset", "Сбросить текущую сессию"),
        BotCommand("matrices", "Список загруженных матриц"),
        BotCommand("reload_matrix", "Перезагрузить матрицы"),
        BotCommand("help", "Справка по командам"),
    ]

    scopes = [BotCommandScopeDefault(), BotCommandScopeAllPrivateChats()]
    language_codes: list[str | None] = [None, "ru", "en"]

    for scope in scopes:
        for language_code in language_codes:
            await app.bot.delete_my_commands(
                scope=scope,
                language_code=language_code,
            )
            await app.bot.set_my_commands(
                commands,
                scope=scope,
                language_code=language_code,
            )

    logger.info("Telegram command menu updated for default/private scopes")


def build_application() -> Application:
    ensure_data_dirs()
    config = load_config()
    matrix = load_matrix(MATRIX_PATH)

    db = Database(DB_PATH)
    db.init()

    openai_service = OpenAIService(api_key=config.openai_api_key, model=config.openai_model)
    dialogue = DialogueManager(matrix=matrix, openai_service=openai_service)
    grader = GradeEngine(matrix=matrix, openai_service=openai_service)
    reporter = ReportBuilder(exports_dir=EXPORTS_DIR)

    app = Application.builder().token(config.telegram_bot_token).post_init(_post_init_set_commands).build()

    app.bot_data["db"] = db
    app.bot_data["dialogue"] = dialogue
    app.bot_data["grader"] = grader
    app.bot_data["reporter"] = reporter
    app.bot_data["pending_probes"] = {}
    app.bot_data["probe_attempts"] = {}
    app.bot_data["invalid_attempts"] = {}
    app.bot_data["quality_attempts"] = {}
    app.bot_data["max_questions"] = config.max_questions
    app.bot_data["confidence_threshold"] = config.confidence_threshold

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("reset", reset_command))
    app.add_handler(CommandHandler("status", status_command))
    app.add_handler(CommandHandler("result", result_command))
    app.add_handler(CommandHandler("matrices", matrices_command))
    app.add_handler(CommandHandler("reload_matrix", reload_matrix_command))

    # Graceful migration of old command names if users still have them cached.
    app.add_handler(CommandHandler(["grade", "feedback", "language"], legacy_command_handler))

    app.add_handler(CallbackQueryHandler(start_assessment_callback, pattern=r"^start_assessment$"))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_message_handler))

    return app


def main() -> None:
    configure_logging()

    try:
        app = build_application()
    except (ConfigError, MatrixError) as exc:
        logging.error("Startup failed: %s", exc)
        sys.exit(1)

    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
