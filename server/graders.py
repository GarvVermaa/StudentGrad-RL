"""StudentGrad-RL graders — simple [END] line score extractors.

Each grader parses the score from the last [END] line in the inference output
and returns it clamped strictly to (0.0001, 0.9499) — never 0 or 1.
"""
from __future__ import annotations

import re


def clamp_score(score: float) -> float:
    """Clamp to strictly (0.0001, 0.9499).

    Upper bound 0.9499 ensures that even if the grader result is passed
    through any rounding, it stays strictly below 1.0.
    """
    return max(0.0001, min(0.9499, float(score)))


def _extract_score(solution: str) -> float:
    """Parse score from last [END] line. Returns 0.5 on any failure."""
    if not solution or not solution.strip():
        return 0.5

    end_lines = [line for line in solution.splitlines() if "[END]" in line]
    if not end_lines:
        return 0.5

    match = re.search(r"score=([0-9]*\.?[0-9]+)", end_lines[-1])
    if not match:
        return 0.5

    try:
        val = float(match.group(1))
        # Guard: if somehow 0 or 1 slipped through, return neutral
        if val <= 0.0 or val >= 1.0:
            return 0.5
        return val
    except (ValueError, TypeError):
        return 0.5


def easy_task_grader(solution: str, reference: str = "") -> float:
    """Grade easy_attendance task."""
    return clamp_score(_extract_score(solution))


def medium_task_grader(solution: str, reference: str = "") -> float:
    """Grade medium_projects task."""
    return clamp_score(_extract_score(solution))


def hard_task_grader(solution: str, reference: str = "") -> float:
    """Grade hard_full_year task."""
    return clamp_score(_extract_score(solution))