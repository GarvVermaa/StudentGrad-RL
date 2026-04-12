"""Grader functions for the StudentGrad OpenEnv hackathon submission.

Each grader:
  - Accepts a solution string and a reference string
  - Returns a float STRICTLY between 0 and 1 (exclusive)
  - Uses clamp_score() to guarantee the (0, 1) constraint

Graders defined here:
  1. academic_performance_grader  — scores academic knowledge/attendance quality
  2. project_completion_grader    — scores project tier and skill coverage
  3. employability_grader         — scores overall employability from trajectory summary
  4. efficiency_grader            — scores time/energy efficiency of the solution
  5. balanced_strategy_grader     — scores whether the agent balanced study + projects
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict


# ── Utility ───────────────────────────────────────────────────────────────────

def clamp_score(score: float) -> float:
    """Ensure score is STRICTLY between 0 and 1 (exclusive).

    The validator rejects 0.0 and 1.0 — use this on every grader return value.
    """
    return max(1e-6, min(1.0 - 1e-6, float(score)))


def _safe_parse(text: str) -> Dict[str, Any]:
    """Try to parse JSON from text; return empty dict on failure."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    # Try extracting a JSON-like block from prose
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except (json.JSONDecodeError, TypeError):
            pass
    return {}


# ── Grader 1: Academic Performance ───────────────────────────────────────────

def academic_performance_grader(solution: str, reference: str) -> float:
    """Score how well the agent achieved academic targets.

    Looks for knowledge scores and attendance rates in the solution JSON.
    Reference contains target thresholds (default: knowledge >= 40, attendance >= 0.40).

    Returns a float strictly in (0, 1).
    """
    sol = _safe_parse(solution)
    ref = _safe_parse(reference)

    # Extract knowledge dict (e.g. {"dsa": 72.5, "dbms": 55.0, ...})
    knowledge: Dict[str, float] = sol.get("knowledge", {})
    attendance: Dict[str, float] = sol.get("attendance", {})

    # Targets from reference (with sensible defaults)
    knowledge_target: float = float(ref.get("knowledge_target", 40.0))
    attendance_target: float = float(ref.get("attendance_target", 0.40))

    # Score knowledge: fraction of subjects at or above target
    if knowledge:
        knowledge_score = sum(
            1.0 for v in knowledge.values() if v >= knowledge_target
        ) / len(knowledge)
    else:
        knowledge_score = 0.1  # partial credit for no data rather than hard 0

    # Score attendance: fraction of subjects at or above target
    if attendance:
        attendance_score = sum(
            1.0 for v in attendance.values() if v >= attendance_target
        ) / len(attendance)
    else:
        attendance_score = 0.1

    # Weighted combination: 60% knowledge, 40% attendance
    raw = 0.6 * knowledge_score + 0.4 * attendance_score
    return clamp_score(raw)


# ── Grader 2: Project Completion ──────────────────────────────────────────────

# Tier values aligned with models.py PROJECT_VALUE
_TIER_VALUES: Dict[str, float] = {
    "basic": 20.0,
    "fullstack": 50.0,
    "cloud": 100.0,
}
_MAX_PROJECT_VALUE = sum(_TIER_VALUES.values())  # 170.0


def project_completion_grader(solution: str, reference: str) -> float:
    """Score project completion quality.

    Checks which project tiers appear in the solution's completed_projects list,
    plus whether required skills were acquired.

    Returns a float strictly in (0, 1).
    """
    sol = _safe_parse(solution)
    ref = _safe_parse(reference)

    completed: list = sol.get("completed_projects", [])
    skills: Dict[str, float] = sol.get("skills", {})

    required_skills: list = ref.get("required_skills", ["js", "html", "css"])

    # Project score: sum of tier values / max possible
    project_value = sum(
        _TIER_VALUES.get(str(t).lower(), 0.0) for t in completed
    )
    project_score = min(1.0, project_value / _MAX_PROJECT_VALUE)

    # Skill score: fraction of required skills acquired (level > 0)
    if required_skills:
        skill_score = sum(
            1.0 for s in required_skills if skills.get(s, 0.0) > 0.0
        ) / len(required_skills)
    else:
        skill_score = 0.5  # neutral if no skills required

    # 70% projects, 30% skills
    raw = 0.7 * project_score + 0.3 * skill_score
    return clamp_score(raw)


# ── Grader 3: Employability Score ────────────────────────────────────────────

def employability_grader(solution: str, reference: str) -> float:
    """Score the overall employability outcome.

    Reads the final_score / employability_score field from the solution,
    normalised against the reference maximum (default 1.0).

    Returns a float strictly in (0, 1).
    """
    sol = _safe_parse(solution)
    ref = _safe_parse(reference)

    # Accept several field names agents might use
    raw_score = (
        sol.get("employability_score")
        or sol.get("final_score")
        or sol.get("score")
        or 0.0
    )
    try:
        raw_score = float(raw_score)
    except (TypeError, ValueError):
        raw_score = 0.0

    max_score = float(ref.get("max_score", 1.0))
    normalised = raw_score / max(max_score, 1e-9)

    return clamp_score(normalised)


# ── Grader 4: Efficiency ──────────────────────────────────────────────────────

def efficiency_grader(solution: str, reference: str) -> float:
    """Score how efficiently the agent used its time budget.

    A solution that meets all success criteria early scores higher.
    Uses days_remaining / day_total as the efficiency signal.

    Returns a float strictly in (0, 1).
    """
    sol = _safe_parse(solution)
    ref = _safe_parse(reference)

    days_remaining: float = float(sol.get("days_remaining", 0))
    day_total: float = float(sol.get("day_total", ref.get("day_total", 365)))

    # Base: fraction of days remaining (0 = used all days, 1 = done immediately)
    time_efficiency = days_remaining / max(day_total, 1.0)

    # Penalise burnout
    fatigue: float = float(sol.get("fatigue", 0.0))
    fatigue_threshold: float = float(ref.get("fatigue_threshold", 80.0))
    burnout_penalty = 0.2 if fatigue >= fatigue_threshold else 0.0

    raw = max(0.05, time_efficiency - burnout_penalty)
    return clamp_score(raw)


# ── Grader 5: Balanced Strategy ───────────────────────────────────────────────

_EXPECTED_ACTION_TYPES = {
    "full_academic",
    "balanced_life",
    "skill_deep_dive",
    "project_sprint",
    "cram_mode",
    "rest_recovery",
    "submit_outcome",
}


def balanced_strategy_grader(solution: str, reference: str) -> float:
    """Score whether the agent used a diverse, balanced set of actions.

    An agent that only uses one action type (e.g. always CRAM_MODE) scores low.
    An agent that uses most action types appropriately scores high.

    Returns a float strictly in (0, 1).
    """
    sol = _safe_parse(solution)

    # Accept either a list of action strings or a history list of dicts
    history = sol.get("session_history", sol.get("action_history", []))

    action_types_seen: set = set()
    for entry in history:
        if isinstance(entry, str):
            action_types_seen.add(entry.lower())
        elif isinstance(entry, dict):
            at = entry.get("action_type", "")
            if at:
                action_types_seen.add(str(at).lower())

    if not action_types_seen:
        # No history info — give minimal partial credit
        return clamp_score(0.05)

    diversity = len(action_types_seen & _EXPECTED_ACTION_TYPES) / len(_EXPECTED_ACTION_TYPES)
    return clamp_score(diversity)


# ── Registry ──────────────────────────────────────────────────────────────────

GRADERS = {
    "academic_performance": academic_performance_grader,
    "project_completion": project_completion_grader,
    "employability": employability_grader,
    "efficiency": efficiency_grader,
    "balanced_strategy": balanced_strategy_grader,
}


def get_grader(name: str):
    """Return a grader function by name. Raises KeyError if not found."""
    if name not in GRADERS:
        raise KeyError(
            f"Unknown grader '{name}'. Available: {list(GRADERS.keys())}"
        )
    return GRADERS[name]
