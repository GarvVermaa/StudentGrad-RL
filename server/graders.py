"""StudentGrad-RL — Grader functions for OpenEnv hackathon validator.

Each grader:
  - Accepts solution_str (inference.py stdout) and reference_str (unused)
  - Parses the [END] line to extract the score
  - Returns float strictly in (0.0001, 0.9999)
  - Returns 0.5 on any parse failure (neutral, never 0 or 1)
"""

from __future__ import annotations

import re


def clamp_score(score: float) -> float:
    """Clamp to strictly (0.0001, 0.9999) — validator rejects 0.0 and 1.0."""
    return max(0.0001, min(0.9999, float(score)))


def _extract_score(solution: str) -> float:
    """Parse the score from the last [END] line in solution string.

    Handles both formats:
      score=0.94        (2 decimal places — new format)
      score=0.9435      (4 decimal places — old format)
    Returns 0.5 if not found.
    """
    if not solution:
        return 0.5

    # Find the last [END] line
    end_lines = [line for line in solution.splitlines() if "[END]" in line]
    if not end_lines:
        return 0.5

    last_end = end_lines[-1]
    match = re.search(r"score=([0-9]+\.?[0-9]*)", last_end)
    if not match:
        return 0.5

    try:
        return float(match.group(1))
    except (ValueError, TypeError):
        return 0.5


def easy_task_grader(solution: str, reference: str = "") -> float:
    """Grade the easy_attendance task (30-day DSA scenario)."""
    return clamp_score(_extract_score(solution))


def medium_task_grader(solution: str, reference: str = "") -> float:
    """Grade the medium_projects task (60-day 3-subject scenario)."""
    return clamp_score(_extract_score(solution))


def hard_task_grader(solution: str, reference: str = "") -> float:
    """Grade the hard_full_year task (100-day full year scenario)."""
    return clamp_score(_extract_score(solution))