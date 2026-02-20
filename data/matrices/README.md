# Supplemental Matrices

Drop additional grading calibrations here as JSON files.

Each file is optional; use any subset of fields below:

```json
{
  "name": "my_company_matrix_2026",
  "description": "Internal matrix for product design grades",
  "role_floors": {
    "Senior": ["senior product designer"],
    "Lead_IC": ["lead designer", "staff designer", "ведущий дизайнер"],
    "Head_M": ["head of design", "design manager", "руководитель дизайна"],
    "DesignDirector_M": ["design director", "директор по дизайну"]
  },
  "track_hints": {
    "IC": ["ic", "individual contributor", "арт-директор"],
    "M": ["manager", "head", "director", "руководитель", "менеджер"]
  },
  "manager_behavior_keywords": ["найм", "ревью", "1:1", "бюджет"],
  "director_behavior_keywords": ["дизайн-стратегия", "оргдизайн", "портфель продуктов"],
  "notes": "Any extra context for calibration"
}
```

The bot automatically loads all `*.json` files from this folder on startup.
