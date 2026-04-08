"""Transition dynamics for the StudentGrad environment."""

from __future__ import annotations

import random
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from models import (
    ActionType,
    ACTION_ENERGY_COST,
    ACTION_FATIGUE_DELTA,
    DailyOutput,
    PROJECT_PREREQS,
    PROJECT_VALUE,
    ProjectTier,
    SUBJECT_SKILL_SYNERGY,
    SubjectType,
    SkillType,
    StudentAction,
)

from .latent_state import FullLatentState
from .noise import NoiseModel
from .output_generator import OutputGenerator


# (days_cost, energy_cost) — days is always 1 for student env
_BASE_ACTION_COSTS: Dict[ActionType, Tuple[float, float]] = {
    ActionType.FULL_ACADEMIC:   (1, 10.0),
    ActionType.SKILL_DEEP_DIVE: (1,  8.0),
    ActionType.PROJECT_SPRINT:  (1,  9.0),
    ActionType.BALANCED_LIFE:   (1,  7.0),
    ActionType.CRAM_MODE:       (1, 12.0),
    ActionType.REST:            (1,  0.0),
    ActionType.SUBMIT_OUTCOME:  (1,  0.0),
}

ACTION_COSTS = _BASE_ACTION_COSTS


def compute_action_cost(action: StudentAction) -> Tuple[float, float]:
    """Return (days, energy) cost for an action."""
    return _BASE_ACTION_COSTS.get(action.action_type, (1, 0.0))


@dataclass
class TransitionResult:
    next_state: FullLatentState
    output: DailyOutput
    reward_components: Dict[str, float] = field(default_factory=dict)
    hard_violations: List[str] = field(default_factory=list)
    soft_violations: List[str] = field(default_factory=list)
    done: bool = False


class TransitionEngine:
    def __init__(self, noise: NoiseModel):
        self.noise = noise
        self.output_gen = OutputGenerator(noise)

    def step(
        self,
        state: FullLatentState,
        action: StudentAction,
        *,
        hard_violations: Optional[List[str]] = None,
        soft_violations: Optional[List[str]] = None,
    ) -> TransitionResult:
        s = deepcopy(state)
        s.step_count += 1
        s.resources.day_current += 1
        step_idx = s.step_count

        hard_v = hard_violations or []
        soft_v = soft_violations or []

        if hard_v:
            output = DailyOutput(
                success=False,
                summary=f"Action blocked: {'; '.join(hard_v)}",
                energy_after=s.resources.energy_current,
                fatigue_after=s.resources.fatigue_current,
            )
            return TransitionResult(
                next_state=s, output=output,
                hard_violations=hard_v, soft_violations=soft_v,
            )

        # Random events first (sick day, surprise quiz)
        self._apply_random_events(s)

        self._apply_energy_and_fatigue(s, action)
        self._update_academic_state(s, action)
        self._update_skill_state(s, action)
        self._update_project_state(s, action)
        self._update_progress_flags(s)

        output = self.output_gen.generate(action, s, step_idx)

        if soft_v:
            output.quality_score *= 0.5
            output.warnings.extend(soft_v)

        done = (
            action.action_type == ActionType.SUBMIT_OUTCOME
            or s.resources.budget_exhausted
        )

        return TransitionResult(
            next_state=s,
            output=output,
            soft_violations=soft_v,
            done=done,
        )

    # ── internals ────────────────────────────────────────────────────────

    def _apply_random_events(self, s: FullLatentState) -> None:
        rng = random.Random(s.rng_seed + s.step_count)

        # Sick day
        if s.progress.sick_days_remaining > 0:
            s.progress.sick_days_remaining -= 1
            s.last_sick_triggered = True
            s.resources.energy_current = s.resources.energy_max * 0.5
        elif rng.random() < s.context.sick_probability:
            s.progress.sick_days_remaining = s.context.sick_duration_days - 1
            s.last_sick_triggered = True
            s.resources.energy_current = s.resources.energy_max * 0.5
        else:
            s.last_sick_triggered = False
            s.resources.energy_current = s.resources.energy_max  # daily reset

        # Surprise quiz
        s.last_quiz_triggered = rng.random() < s.context.quiz_probability

    def _apply_energy_and_fatigue(
        self, s: FullLatentState, action: StudentAction
    ) -> None:
        _, energy_cost = compute_action_cost(action)
        fatigue_delta = ACTION_FATIGUE_DELTA.get(action.action_type, 0.0)

        if s.last_sick_triggered:
            energy_cost *= 0.5  # sick reduces effectiveness

        # Burnout: if fatigue > threshold, clamp energy effectiveness
        if s.resources.fatigue_current >= s.latent.true_fatigue_threshold:
            s.progress.burnout_occurred = True
            energy_cost *= 0.3  # almost nothing gets done

        # REST recovers fatigue
        if action.action_type == ActionType.REST:
            s.resources.fatigue_current = max(
                0.0,
                s.resources.fatigue_current + fatigue_delta * s.context.energy_recovery_rate
            )
        else:
            s.resources.fatigue_current = min(
                100.0,
                s.resources.fatigue_current + fatigue_delta
            )

    def _update_academic_state(
        self, s: FullLatentState, action: StudentAction
    ) -> None:
        at = action.action_type
        subjects = list(s.true_attendance.keys())

        if at in (ActionType.FULL_ACADEMIC, ActionType.CRAM_MODE):
            attend_delta = 1.0 / s.latent.true_exam_difficulty.get("dsa", 1.0)
            for subj in subjects:
                # attendance increases with full academic or cram
                total_classes_possible = s.resources.day_current
                current_attended = s.true_attendance[subj] * max(total_classes_possible - 1, 1)
                new_attended = current_attended + (1.0 if at == ActionType.FULL_ACADEMIC else 0.5)
                s.true_attendance[subj] = new_attended / max(total_classes_possible, 1)

                # knowledge gain
                rate = s.latent.true_learning_rates.get(subj, 0.8)
                noise_factor = self.noise.coin_flip(0.1)  # small random variance
                gain = rate * (12.0 if at == ActionType.CRAM_MODE else 3.0) * (1 - noise_factor * 0.2)
                s.true_knowledge[subj] = min(100.0, s.true_knowledge[subj] + gain)

                # Synergy bonuses from attending class
                subj_enum = SubjectType(subj) if subj in SubjectType._value2member_map_ else None
                if subj_enum and subj_enum in SUBJECT_SKILL_SYNERGY:
                    for skill_enum, bonus in SUBJECT_SKILL_SYNERGY[subj_enum].items():
                        s.true_skills[skill_enum.value] = min(
                            30.0, s.true_skills[skill_enum.value] + bonus
                        )

        elif at == ActionType.BALANCED_LIFE:
            for subj in subjects:
                total_classes_possible = s.resources.day_current
                current_attended = s.true_attendance[subj] * max(total_classes_possible - 1, 1)
                new_attended = current_attended + 0.5
                s.true_attendance[subj] = new_attended / max(total_classes_possible, 1)
                rate = s.latent.true_learning_rates.get(subj, 0.8)
                s.true_knowledge[subj] = min(100.0, s.true_knowledge[subj] + rate * 1.5)

        # Surprise quiz penalty
        if s.last_quiz_triggered:
            for subj in subjects:
                if s.true_attendance[subj] < 0.6:
                    s.true_knowledge[subj] = max(0.0, s.true_knowledge[subj] - 5.0)

    def _update_skill_state(
        self, s: FullLatentState, action: StudentAction
    ) -> None:
        at = action.action_type
        if at == ActionType.SKILL_DEEP_DIVE and action.skill_target:
            skill = action.skill_target.value
            rate = s.latent.true_skill_rates.get(skill, 1.0)
            noise = self.noise.quality_degradation(0.1)
            gain = rate * 2.0 * (1 - noise)
            s.true_skills[skill] = min(30.0, s.true_skills.get(skill, 0.0) + gain)

        elif at == ActionType.BALANCED_LIFE and action.skill_target:
            skill = action.skill_target.value
            rate = s.latent.true_skill_rates.get(skill, 1.0)
            s.true_skills[skill] = min(30.0, s.true_skills.get(skill, 0.0) + rate * 1.0)

    def _update_project_state(
        self, s: FullLatentState, action: StudentAction
    ) -> None:
        if action.action_type != ActionType.PROJECT_SPRINT:
            return
        if not action.project_target:
            return

        tier = action.project_target
        tier_str = tier.value

        # Check if already completed (diminishing returns handled in rules)
        times_done = s.completed_projects.count(tier_str)
        effectiveness = 1.0 / (times_done + 1)

        if s.active_project_tier != tier_str:
            # Switching project resets progress
            s.active_project_tier = tier_str
            s.true_project_progress = 0.0

        noise = self.noise.quality_degradation(0.1)
        progress_gain = 0.25 * effectiveness * (1 - noise)

        # Burnout slows project work
        if s.resources.fatigue_current >= s.latent.true_fatigue_threshold:
            progress_gain *= 0.3

        s.true_project_progress = min(1.0, s.true_project_progress + progress_gain)

        if s.true_project_progress >= 1.0:
            s.completed_projects.append(tier_str)
            s.true_project_progress = 0.0
            s.active_project_tier = None

    def _update_progress_flags(self, s: FullLatentState) -> None:
        p = s.progress
        attendance = s.true_attendance
        skills = s.true_skills

        p.attended_first_class = any(a > 0 for a in attendance.values())
        p.above_75_attendance_any = any(a >= 0.75 for a in attendance.values())
        p.above_75_attendance_all = all(a >= 0.75 for a in attendance.values())
        p.studied_all_subjects = all(a > 0 for a in attendance.values())
        p.exam_eligible = all(a >= 0.40 for a in attendance.values())

        p.first_skill_acquired = any(v > 0 for v in skills.values())
        p.any_skill_above_10 = any(v >= 10.0 for v in skills.values())

        # Project prereqs
        basic_prereqs = PROJECT_PREREQS[ProjectTier.BASIC]
        p.skill_prereqs_basic_met = all(
            skills.get(k.value, 0) >= v for k, v in basic_prereqs.items()
        )
        fullstack_prereqs = PROJECT_PREREQS[ProjectTier.FULLSTACK]
        p.skill_prereqs_fullstack_met = all(
            skills.get(k.value, 0) >= v for k, v in fullstack_prereqs.items()
        )
        cloud_prereqs = PROJECT_PREREQS[ProjectTier.CLOUD]
        p.skill_prereqs_cloud_met = all(
            skills.get(k.value, 0) >= v for k, v in cloud_prereqs.items()
        )

        p.basic_project_done = "basic" in s.completed_projects
        p.fullstack_project_done = "fullstack" in s.completed_projects
        p.cloud_project_done = "cloud" in s.completed_projects

        # Exam logic: only on/after exam day
        exam_day = 300
        if s.resources.day_current >= exam_day and p.exam_eligible:
            p.exam_taken = True
            # Pass if knowledge > 40 in all subjects (adjusted by difficulty)
            p.passed_all_subjects = all(
                s.true_knowledge.get(subj, 0) / s.latent.true_exam_difficulty.get(subj, 1.0) >= 40.0
                for subj in s.true_attendance.keys()
            )
            if not p.exam_eligible or not p.passed_all_subjects:
                p.academic_failed = True

        if s.resources.fatigue_current >= s.latent.true_fatigue_threshold:
            p.burnout_occurred = True

        # Counters
        # (incremented outside, just ensure flags are consistent)