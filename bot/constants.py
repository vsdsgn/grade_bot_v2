from __future__ import annotations

LEVELS = [
    "Junior",
    "Middle",
    "Senior",
    "Lead_IC",
    "Head_M",
    "ArtDirector_IC",
    "DesignDirector_M",
]

DIMENSIONS = [
    "scope_responsibility",
    "impact",
    "uncertainty_tolerance",
    "planning_horizon",
    "hard_craft",
    "hard_systems",
    "hard_product_business",
    "soft_communication_influence",
    "management",
    "culture_ownership",
    "culture_proactivity",
    "culture_quality_bar",
    "culture_collaboration",
    "culture_learning",
    "culture_integrity_safety",
]

DIMENSION_DISPLAY = {
    "scope_responsibility": "Зона ответственности и масштаб",
    "impact": "Влияние на продукт и бизнес",
    "uncertainty_tolerance": "Решения в условиях неопределенности",
    "planning_horizon": "Горизонт планирования",
    "hard_craft": "Хард-скиллы: мастерство (craft)",
    "hard_systems": "Хард-скиллы: системность",
    "hard_product_business": "Хард-скиллы: продукт и бизнес",
    "soft_communication_influence": "Софт-скиллы: коммуникация и влияние",
    "management": "Управление людьми и процессом",
    "culture_ownership": "Культура: ownership",
    "culture_proactivity": "Культура: проактивность",
    "culture_quality_bar": "Культура: планка качества",
    "culture_collaboration": "Культура: сотрудничество",
    "culture_learning": "Культура: обучение",
    "culture_integrity_safety": "Культура: этика и безопасность",
}

HIGH_VARIANCE_PRIORITY = [
    "scope_responsibility",
    "uncertainty_tolerance",
    "impact",
    "planning_horizon",
]

# Used for warm-up count/status; actual text is generated in handlers for a more natural flow.
WARMUP_QUESTIONS = [
    "Роль и ответственность",
    "Продукт и пользователи",
    "Опыт и желаемый трек",
]

PROFILE_FIELDS = [
    "current_role",
    "domain_and_users",
    "experience_and_track_goal",
]
