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
    "scope_responsibility": "Scope & Responsibility",
    "impact": "Impact",
    "uncertainty_tolerance": "Decision-making Under Uncertainty",
    "planning_horizon": "Planning Horizon",
    "hard_craft": "Hard Skills: Craft",
    "hard_systems": "Hard Skills: Systems",
    "hard_product_business": "Hard Skills: Product & Business",
    "soft_communication_influence": "Soft Skills: Communication & Influence",
    "management": "Management",
    "culture_ownership": "Culture: Ownership",
    "culture_proactivity": "Culture: Proactivity",
    "culture_quality_bar": "Culture: Quality Bar",
    "culture_collaboration": "Culture: Collaboration",
    "culture_learning": "Culture: Learning",
    "culture_integrity_safety": "Culture: Integrity & Safety",
}

HIGH_VARIANCE_PRIORITY = [
    "scope_responsibility",
    "uncertainty_tolerance",
    "impact",
    "planning_horizon",
]

WARMUP_QUESTIONS = [
    "To ground us: what is your current role/title, and what team setup do you work in?",
    "What product domain are you in, and who are your main users?",
    "How many years of product design experience do you have, and are you aiming for an IC or manager path next?",
]

PROFILE_FIELDS = [
    "current_role",
    "domain_and_users",
    "experience_and_track_goal",
]
