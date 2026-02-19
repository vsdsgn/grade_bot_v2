from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = BASE_DIR / "data"
DEFAULT_MATRIX_TEMPLATE_PATH = DEFAULT_DATA_DIR / "matrix_v2.json"


def _resolve_path(raw_path: str | None, default_path: Path) -> Path:
    if not raw_path:
        return default_path
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    return (BASE_DIR / path).resolve()


# Load .env early so path env vars are available for module-level constants.
load_dotenv()

DATA_DIR = _resolve_path(os.getenv("DATA_DIR"), DEFAULT_DATA_DIR)
EXPORTS_DIR = DATA_DIR / "exports"
DB_PATH = _resolve_path(os.getenv("DB_PATH"), DATA_DIR / "assessments.db")
MATRIX_PATH = _resolve_path(os.getenv("MATRIX_PATH"), DATA_DIR / "matrix_v2.json")
MATRIX_TEMPLATE_PATH = _resolve_path(
    os.getenv("MATRIX_TEMPLATE_PATH"),
    DEFAULT_MATRIX_TEMPLATE_PATH,
)


@dataclass(frozen=True)
class AppConfig:
    telegram_bot_token: str
    openai_api_key: str
    openai_model: str = "gpt-4.1-mini"
    max_questions: int = 18
    confidence_threshold: float = 0.82


class ConfigError(RuntimeError):
    pass


def load_config() -> AppConfig:
    load_dotenv()

    telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
    openai_model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip() or "gpt-4.1-mini"
    max_questions_raw = os.getenv("MAX_QUESTIONS", "18").strip()
    confidence_threshold_raw = os.getenv("CONFIDENCE_THRESHOLD", "0.82").strip()

    if not telegram_bot_token:
        raise ConfigError("Missing TELEGRAM_BOT_TOKEN in environment/.env")
    if not openai_api_key:
        raise ConfigError("Missing OPENAI_API_KEY in environment/.env")

    try:
        max_questions = int(max_questions_raw)
        if max_questions < 6 or max_questions > 40:
            raise ValueError
    except ValueError as exc:
        raise ConfigError("MAX_QUESTIONS must be an integer in range [6, 40]") from exc

    try:
        confidence_threshold = float(confidence_threshold_raw)
        if confidence_threshold <= 0 or confidence_threshold >= 1:
            raise ValueError
    except ValueError as exc:
        raise ConfigError("CONFIDENCE_THRESHOLD must be a float in range (0, 1)") from exc

    return AppConfig(
        telegram_bot_token=telegram_bot_token,
        openai_api_key=openai_api_key,
        openai_model=openai_model,
        max_questions=max_questions,
        confidence_threshold=confidence_threshold,
    )


def ensure_data_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    MATRIX_PATH.parent.mkdir(parents=True, exist_ok=True)

    # If MATRIX_PATH points to an empty mounted volume, seed it from template.
    if not MATRIX_PATH.exists():
        if MATRIX_TEMPLATE_PATH.exists():
            shutil.copy2(MATRIX_TEMPLATE_PATH, MATRIX_PATH)
        else:
            raise ConfigError(
                f"Matrix file missing at {MATRIX_PATH} and template not found at {MATRIX_TEMPLATE_PATH}"
            )
