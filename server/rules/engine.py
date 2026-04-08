"""Rule engine for StudentGrad — prerequisite DAG and violation checks."""

from __future__ import annotations

from typing import List, Tuple

from models import ActionType, PROJECT_PREREQS, ProjectTier, StudentAction
from server.simulator.latent_state import FullLatentState


class RuleEngine:
    """Checks an action against the current latent state.

    Returns a flat list of violation strings.
    Hard violations (prefix 'HARD:') block the action entirely.
    Soft violations (prefix 'SOFT:') reduce output quality.
    """

    def check(
        self, action: StudentAction, state: FullLatentState
    ) -> List[str]:
        violations: List[str] = []
        violations.extend(self._check_prerequisites(action, state))
        violations.extend(self._check_resource_constraints(action, state))
        violations.extend(self._check_causal_validity(action, state))
        violations.extend(self._check_soft_warnings(action, state))
        return violations

    def hard_violations(self, violations: List[str]) -> List[str]:
        return [v for v in violations if v.startswith("HARD:")]

    def soft_violations(self, violations: List[str]) -> List[str]:
        return [v for v in violations if v.startswith("SOFT:")]

    # ── prerequisite checks ──────────────────────────────────────────────

    def _check_prerequisites(
        self, action: StudentAction, s: FullLatentState
    ) -> List[str]:
        violations: List[str] = []
        at = action.action_type
        p = s.progress

        # CRAM_MODE only legal within 15 days of exam (day 285-300)
        exam_day = 300
        cram_window = 15
        if at == ActionType.CRAM_MODE:
            days_to_exam = exam_day - s.resources.day_current
            if days_to_exam > cram_window:
                violations.append(
                    f"HARD:cram_too_early — CRAM_MODE only available within "
                    f"{cram_window} days of exam (currently {days_to_exam} days away)."
                )

        # PROJECT_SPRINT requires skill prerequisites
        if at == ActionType.PROJECT_SPRINT:
            if not action.project_target:
                violations.append(
                    "HARD:no_project_target — PROJECT_SPRINT requires project_target to be set."
                )
            else:
                tier = action.project_target
                prereqs = PROJECT_PREREQS.get(tier, {})
                missing = [
                    f"{skill.value}≥{needed} (have {round(s.true_skills.get(skill.value, 0), 1)})"
                    for skill, needed in prereqs.items()
                    if s.true_skills.get(skill.value, 0.0) < needed
                ]
                if missing:
                    violations.append(
                        f"HARD:missing_skill_prereqs — Cannot start {tier.value} project. "
                        f"Missing: {', '.join(missing)}."
                    )

        # SKILL_DEEP_DIVE requires skill_target
        if at == ActionType.SKILL_DEEP_DIVE and not action.skill_target:
            violations.append(
                "HARD:no_skill_target — SKILL_DEEP_DIVE requires skill_target to be set."
            )

        # SUBMIT_OUTCOME requires exam day reached
        if at == ActionType.SUBMIT_OUTCOME:
            if s.resources.day_current < exam_day:
                violations.append(
                    f"HARD:too_early_to_submit — Cannot SUBMIT_OUTCOME before Day {exam_day} "
                    f"(currently Day {s.resources.day_current})."
                )

        return violations

    # ── resource constraints ─────────────────────────────────────────────

    def _check_resource_constraints(
        self, action: StudentAction, s: FullLatentState
    ) -> List[str]:
        violations: List[str] = []
        if s.resources.budget_exhausted:
            violations.append("HARD:days_exhausted — No days remaining in the year.")
        return violations

    # ── causal validity ──────────────────────────────────────────────────

    def _check_causal_validity(
        self, action: StudentAction, s: FullLatentState
    ) -> List[str]:
        violations: List[str] = []
        at = action.action_type
        p = s.progress

        # Cannot SUBMIT_OUTCOME without having studied anything
        if at == ActionType.SUBMIT_OUTCOME and not p.attended_first_class:
            violations.append(
                "HARD:no_study_done — Cannot submit outcome without studying."
            )

        return violations

    # ── soft warnings ────────────────────────────────────────────────────

    def _check_soft_warnings(
        self, action: StudentAction, s: FullLatentState
    ) -> List[str]:
        violations: List[str] = []
        at = action.action_type
        p = s.progress

        # Warn if doing project sprint with very low attendance (risks academic failure)
        if at == ActionType.PROJECT_SPRINT:
            low_attendance_subjects = [
                subj for subj, att in s.true_attendance.items() if att < 0.40
            ]
            if low_attendance_subjects and s.resources.day_current > 200:
                violations.append(
                    f"SOFT:attendance_risk — Attendance critically low in "
                    f"{', '.join(low_attendance_subjects)}. Academic failure risk."
                )

        # Warn if cramming while already burned out
        if at == ActionType.CRAM_MODE and s.resources.fatigue_current > 70:
            violations.append(
                "SOFT:burnout_cram — Cramming while fatigued >70. Effectiveness severely reduced."
            )

        # Warn if resting when exams are imminent and not studied enough
        if at == ActionType.REST:
            days_to_exam = 300 - s.resources.day_current
            if 0 < days_to_exam <= 10 and not p.above_75_attendance_all:
                violations.append(
                    "SOFT:rest_near_exam — Resting with <10 days to exam and insufficient attendance."
                )

        return violations