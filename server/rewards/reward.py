"""StudentGrad reward function.

Keeps the exact RewardBreakdown dataclass interface from the original repo
so hackathon_environment.py works without modification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from models import (
    ActionType,
    DailyOutput,
    META_ACTIONS,
    PROJECT_VALUE,
    ProjectTier,
    StudentAction,
)
from server.simulator.latent_state import FullLatentState


def _clamp(score: float) -> float:
    """Return score strictly between 0 and 1 (exclusive).

    The OpenEnv validator rejects exactly 0.0 and exactly 1.0.
    """
    return max(1e-6, min(1.0 - 1e-6, float(score)))


@dataclass
class RewardBreakdown:
    validity: float = 0.0
    ordering: float = 0.0
    info_gain: float = 0.0
    efficiency: float = 0.0
    novelty: float = 0.0
    penalty: float = 0.0
    shaping: float = 0.0
    terminal: float = 0.0
    components: Dict[str, float] = field(default_factory=dict)

    @property
    def total(self) -> float:
        return (
            self.validity
            + self.ordering
            + self.info_gain
            + self.efficiency
            + self.novelty
            + self.penalty
            + self.shaping
            + self.terminal
        )

    def to_dict(self) -> Dict[str, float]:
        d = {
            "validity": self.validity,
            "ordering": self.ordering,
            "info_gain": self.info_gain,
            "efficiency": self.efficiency,
            "novelty": self.novelty,
            "penalty": self.penalty,
            "shaping": self.shaping,
            "terminal": self.terminal,
            "total": self.total,
        }
        d.update(self.components)
        return d


class RewardComputer:
    def __init__(
        self,
        efficiency_weight: float = 0.3,
        info_gain_weight: float = 0.4,
        validity_weight: float = 0.3,
    ):
        self.w_eff = efficiency_weight
        self.w_ig = info_gain_weight
        self.w_val = validity_weight

    # ── step reward ──────────────────────────────────────────────────────

    def step_reward(
        self,
        action: StudentAction,
        prev_state: FullLatentState,
        next_state: FullLatentState,
        output: DailyOutput,
        hard_violations: List[str],
        soft_violations: List[str],
    ) -> RewardBreakdown:
        rb = RewardBreakdown()

        if hard_violations:
            rb.validity = -0.999   # was -1.0
            rb.penalty = -0.5 * len(hard_violations)
            rb.components["hard_violations"] = len(hard_violations)
            return rb

        rb.validity = self.w_val * (0.999 if output.success else 0.001)

        ordering_score = self._ordering_score(action, prev_state)
        rb.ordering = 0.2 * ordering_score

        rb.info_gain = self.w_ig * output.quality_score * (1.0 - output.uncertainty)

        # Efficiency: days-remaining fraction
        days_frac = 1.0 / max(next_state.resources.day_total, 1)
        rb.efficiency = self.w_eff * max(0.0, 1.0 - 5.0 * days_frac)

        # Novelty bonus for non-redundant steps
        if not soft_violations:
            rb.novelty = 0.1

        # Academic safety: positive if any subject above 75%, penalty if any below 40%
        attendance = next_state.true_attendance
        above_75 = sum(1 for a in attendance.values() if a >= 0.75)
        below_40 = sum(1 for a in attendance.values() if a < 0.40)

        academic_safety = 0.05 * above_75 - 0.1 * below_40
        rb.components["academic_safety"] = academic_safety
        rb.validity += academic_safety

        # Skill shaping: small reward per skill point gained
        skill_delta = sum(
            next_state.true_skills.get(k, 0) - prev_state.true_skills.get(k, 0)
            for k in next_state.true_skills
        )
        rb.components["skill_shaping"] = 0.01 * skill_delta
        rb.info_gain += 0.01 * skill_delta

        # Project milestone bonus
        prev_projects = len(prev_state.completed_projects)
        next_projects = len(next_state.completed_projects)
        if next_projects > prev_projects:
            # Reward for completing a project
            new_tier = next_state.completed_projects[-1]
            proj_reward = PROJECT_VALUE.get(ProjectTier(new_tier), 0.0) * 0.1
            rb.components["project_milestone"] = proj_reward
            rb.info_gain += proj_reward

        # Burnout penalty
        if next_state.resources.fatigue_current >= next_state.latent.true_fatigue_threshold:
            rb.penalty -= 0.5
            rb.components["burnout_penalty"] = -0.5

        # Soft violation penalty
        rb.penalty -= 0.15 * len(soft_violations)

        # Potential-based shaping
        phi_prev = self._potential(prev_state)
        phi_next = self._potential(next_state)
        rb.shaping = phi_next - phi_prev

        return rb

    # ── terminal reward ──────────────────────────────────────────────────

    def terminal_reward(
        self,
        state: FullLatentState,
        conclusions: list,            # unused in student env — kept for interface compat
        task_success_criteria: List[str],
        discovered_markers: Optional[List[str]] = None,
        candidate_mechanisms: Optional[List[str]] = None,
    ) -> RewardBreakdown:
        rb = RewardBreakdown()
        p = state.progress

        # Gatekeeper: if academic failed → strong negative (not -10 to avoid unbounded grader)
        if p.academic_failed or not p.exam_eligible:
            rb.terminal = -5.0
            rb.components["academic_failed_penalty"] = -5.0
            return rb

        # Academic score: average knowledge adjusted by exam difficulty
        subjects = list(state.true_knowledge.keys())
        raw_scores = [
            state.true_knowledge.get(s, 0) / state.latent.true_exam_difficulty.get(s, 1.0)
            for s in subjects
        ]
        academic_score = sum(raw_scores) / max(len(raw_scores), 1)
        # Clamp norm to strictly (0, 1) exclusive
        academic_score_norm = _clamp(min(0.999, academic_score / 100.0))

        # Project value
        project_value = sum(
            PROJECT_VALUE.get(ProjectTier(t), 0.0)
            for t in state.completed_projects
        )
        # Normalize project value (max possible is basic+fullstack+cloud = 170)
        project_value_norm = _clamp(min(0.999, project_value / 170.0))

        # Attendance bonus: clamp away from exact 0.5 and 1.0
        avg_attendance = sum(state.true_attendance.values()) / max(len(state.true_attendance), 1)
        attendance_bonus = 0.999 if avg_attendance >= 0.75 else 0.5

        # Final formula: 0.3 × academic + 0.7 × project
        base_score = _clamp(
            (0.3 * academic_score_norm + 0.7 * project_value_norm) * attendance_bonus
        )

        rb.components["academic_score_norm"] = academic_score_norm
        rb.components["project_value_norm"] = project_value_norm
        rb.components["attendance_bonus"] = attendance_bonus
        rb.components["base_score"] = base_score

        # Scale terminal to be dominant signal (≈ 5-10× step rewards)
        rb.terminal = base_score * 10.0

        # Efficiency bonus: days remaining
        days_remaining_frac = state.resources.days_remaining / max(state.resources.day_total, 1)
        eff_bonus = _clamp(days_remaining_frac) * 1.0
        rb.terminal += eff_bonus
        rb.components["efficiency_bonus"] = eff_bonus

        return rb

    # ── helpers ──────────────────────────────────────────────────────────

    def _ordering_score(
        self, action: StudentAction, s: FullLatentState
    ) -> float:
        """1.0 = ideal next action, 0.3 = acceptable, -1.0 = harmful.
        
        All returns are clamped away from exact 0 and 1 boundaries.
        """
        at = action.action_type
        p = s.progress
        day = s.resources.day_current
        exam_day = 300
        days_to_exam = exam_day - day

        # Early game (days 1-100): prioritise attendance
        if day <= 100:
            if at == ActionType.FULL_ACADEMIC:
                return 0.999
            if at == ActionType.BALANCED_LIFE:
                return 0.7
            if at == ActionType.PROJECT_SPRINT and not p.skill_prereqs_basic_met:
                return 0.001

        # Mid game (days 101-250): skill acquisition + projects
        elif day <= 250:
            if at == ActionType.SKILL_DEEP_DIVE and not p.any_skill_above_10:
                return 0.999
            if at == ActionType.PROJECT_SPRINT and p.skill_prereqs_basic_met:
                return 0.999
            if at == ActionType.FULL_ACADEMIC and not p.above_75_attendance_all:
                return 0.8

        # Exam window (days 251-300): study or cram
        elif day <= exam_day:
            if at == ActionType.CRAM_MODE:
                return 0.999
            if at == ActionType.FULL_ACADEMIC:
                return 0.8
            if at == ActionType.PROJECT_SPRINT and not p.above_75_attendance_all:
                return 0.001

        # Post-exam: submit
        if day >= exam_day and at == ActionType.SUBMIT_OUTCOME:
            return 0.999

        return 0.3

    def _potential(self, s: FullLatentState) -> float:
        """Progress potential φ(s) — milestones completed."""
        p = s.progress
        milestones = [
            p.attended_first_class,
            p.above_75_attendance_any,
            p.above_75_attendance_all,
            p.first_skill_acquired,
            p.any_skill_above_10,
            p.skill_prereqs_basic_met,
            p.basic_project_done,
            p.skill_prereqs_fullstack_met,
            p.fullstack_project_done,
            p.skill_prereqs_cloud_met,
            p.cloud_project_done,
            p.exam_eligible,
            p.exam_taken,
            p.passed_all_subjects,
        ]
        return sum(milestones) / len(milestones)