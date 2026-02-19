# Telegram Product Designer Self-Assessment Bot

A local end-to-end Telegram bot that runs a live adaptive interview for product designers using a competency matrix (v2.0), then produces:

1. Suggested level (`Junior`, `Middle`, `Senior`, `Lead_IC`, `Head_M`, `ArtDirector_IC`, `DesignDirector_M`)
2. Per-layer scores with evidence snippets
3. Strengths, growth areas, and recommended actions
4. A short roadmap for the next level

Data is stored locally in SQLite and JSON exports in `data/exports/`.

## Tech Stack

- Python 3.11+
- [python-telegram-bot v20+](https://github.com/python-telegram-bot/python-telegram-bot)
- Official [OpenAI Python SDK](https://github.com/openai/openai-python)
- OpenAI Responses API
  - Adaptive question generation
  - Structured output grading with `json_schema`
- SQLite (local)

## Project Structure

```text
/Users/vsdsgn/Documents/grade bot
├── main.py
├── requirements.txt
├── .env.example
├── README.md
├── bot/
│   ├── __init__.py
│   ├── config.py
│   ├── constants.py
│   ├── matrix.py
│   ├── models.py
│   ├── database.py
│   ├── openai_service.py
│   ├── dialogue.py
│   ├── grading.py
│   ├── reporting.py
│   └── handlers.py
└── data/
    ├── matrix_v2.json
    ├── assessments.db        # created on first run
    └── exports/
```

## Setup

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create `.env` from `.env.example`:

```bash
cp .env.example .env
```

4. Set real values in `.env`:

- `TELEGRAM_BOT_TOKEN`
- `OPENAI_API_KEY`
- `OPENAI_MODEL` (optional, default `gpt-4.1-mini`)
- `MAX_QUESTIONS` (optional, default `18`)
- `CONFIDENCE_THRESHOLD` (optional, default `0.82`)

## Run

```bash
python3 main.py
```

On startup the app will:

- Ensure `data/` and `data/exports/` exist
- Initialize SQLite schema in `data/assessments.db`
- Load `data/matrix_v2.json`
- Start Telegram long polling

## Deploy to Railway (GitHub + your OpenAI key)

The bot uses long polling, so no webhook/public URL is required.

1. Push this folder to a GitHub repository (existing or new).
2. In Railway, create a new project and connect that GitHub repo.
3. Railway will build from `Dockerfile` and run `python main.py`.
4. In Railway service variables, set:
   - `TELEGRAM_BOT_TOKEN`
   - `OPENAI_API_KEY`
   - `OPENAI_MODEL` (optional, e.g. `gpt-4.1-mini`)
   - `MAX_QUESTIONS` (optional)
   - `CONFIDENCE_THRESHOLD` (optional)
5. Add a Railway Volume and mount it to `/data` (important for persistent SQLite).
6. Set `DATA_DIR=/data` in Railway variables.
7. Deploy.

After deploy, your DB and JSON exports persist in the mounted volume under:

```text
/data/assessments.db
/data/exports/
```

## Telegram Commands

- `/start` - welcome, process/privacy explanation, and **Start assessment** button
- `/reset` - clears current in-progress session
- `/status` - progress: covered layers, confidence estimate, approximate remaining questions
- `/result` - returns final report if finished; otherwise explains what is missing
- `/help` - command help

## Interview Flow

- Warm-up (3 questions): role context, domain/users, years + IC/M preference
- Adaptive interview across matrix layers (one question at a time)
- Follow-up probes when answers are vague
- Stops when confidence threshold is met (with sufficient coverage) or max question limit is reached

## Persistence & Exports

- All sessions and turns are stored in SQLite (`data/assessments.db`)
- As conversation grows, older turns are summarized into a running `evidence_summary`
- Final reports are exported as JSON:

```text
data/exports/{telegram_user_id}_{timestamp}.json
```

## Notes

- No secrets are hardcoded
- OpenAI failures and rate limits are retried with exponential backoff
- If OpenAI fails during chat, user gets a graceful retry message
