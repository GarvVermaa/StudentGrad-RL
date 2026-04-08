"""Output generator — produces the noisy DailyOutput the agent observes."""

from __future__ import annotations

from typing import Any, Dict

from models import (
    ActionType,
    DailyOutput,
    PROJECT_VALUE,
    ProjectTier,
    StudentAction,
)

from .latent_state import FullLatentState
from .noise import NoiseModel


class OutputGenerator:
    def __init__(self, noise: NoiseModel):
        self.noise = noise

    def generate(
        self,
        action: StudentAction,
        state: FullLatentState,
        step_index: int,
    ) -> DailyOutput:
        handler = self._HANDLERS.get(action.action_type, self._default)
        return handler(action, state, step_index)

    # ── handlers ────────────────────────────────────────────────────────

    def _full_academic(
        self, action: StudentAction, s: FullLatentState, step: int
    ) -> DailyOutput:
        noise = self.noise.quality_degradation(s.context.base_noise)
        sick_note = " (sick — reduced effectiveness)" if s.last_sick_triggered else ""
        knowledge_gained = {
            subj: round(
                s.latent.true_learning_rates.get(subj, 0.8) * 3.0 * (1 - noise), 2
            )
            for subj in s.true_attendance
        }
        skill_gained = {
            skill: round(0.5 * (1 - noise), 2)
            for skill in ["js", "docker", "node"]  # synergy skills
        }
        quality = max(0.1, 1.0 - noise - (0.3 if s.last_sick_triggered else 0.0))
        return DailyOutput(
            success=True,
            summary=f"Day {s.resources.day_current}: Attended all classes{sick_note}. "
                    f"Attendance improving across all subjects.",
            knowledge_gained=knowledge_gained,
            skill_gained=skill_gained,
            energy_after=max(0.0, s.resources.energy_current - 10.0),
            fatigue_after=min(100.0, s.resources.fatigue_current + 5.0),
            quality_score=quality,
            uncertainty=noise,
            data={"attendance_delta": round(1.0 / max(s.resources.day_current, 1), 4)},
        )

    def _skill_deep_dive(
        self, action: StudentAction, s: FullLatentState, step: int
    ) -> DailyOutput:
        noise = self.noise.quality_degradation(s.context.base_noise)
        skill = action.skill_target.value if action.skill_target else "js"
        rate = s.latent.true_skill_rates.get(skill, 1.0)
        gained = round(rate * 2.0 * (1 - noise), 2)
        current = s.true_skills.get(skill, 0.0)
        quality = max(0.1, 1.0 - noise - (0.4 if s.resources.fatigue_current > 60 else 0.0))
        return DailyOutput(
            success=True,
            summary=f"Day {s.resources.day_current}: Deep-dived into {skill}. "
                    f"Current {skill} points ≈ {round(current + gained, 1)}.",
            skill_gained={skill: gained},
            energy_after=max(0.0, s.resources.energy_current - 8.0),
            fatigue_after=min(100.0, s.resources.fatigue_current + 4.0),
            quality_score=quality,
            uncertainty=noise,
            data={"skill": skill, "gained": gained, "total_approx": round(current + gained, 1)},
        )

    def _project_sprint(
        self, action: StudentAction, s: FullLatentState, step: int
    ) -> DailyOutput:
        noise = self.noise.quality_degradation(s.context.base_noise)
        tier = action.project_target.value if action.project_target else "basic"
        progress = s.true_project_progress
        completed = tier in s.completed_projects
        quality = max(0.1, 1.0 - noise - (0.5 if s.resources.fatigue_current > 70 else 0.0))

        if completed:
            msg = f"Day {s.resources.day_current}: {tier.capitalize()} project already done. Building again for portfolio (diminishing returns)."
        else:
            msg = f"Day {s.resources.day_current}: Working on {tier.capitalize()} project. Progress ≈ {round(progress * 100)}%."

        return DailyOutput(
            success=True,
            summary=msg,
            project_progress=round(0.25 * (1 - noise), 2),
            energy_after=max(0.0, s.resources.energy_current - 9.0),
            fatigue_after=min(100.0, s.resources.fatigue_current + 5.0),
            quality_score=quality,
            uncertainty=noise,
            data={"tier": tier, "progress_approx": round(progress, 2)},
        )

    def _balanced_life(
        self, action: StudentAction, s: FullLatentState, step: int
    ) -> DailyOutput:
        noise = self.noise.quality_degradation(s.context.base_noise)
        skill = action.skill_target.value if action.skill_target else None
        knowledge_gained = {
            subj: round(s.latent.true_learning_rates.get(subj, 0.8) * 1.5 * (1 - noise), 2)
            for subj in s.true_attendance
        }
        skill_gained: Dict[str, float] = {}
        if skill:
            rate = s.latent.true_skill_rates.get(skill, 1.0)
            skill_gained[skill] = round(rate * 1.0 * (1 - noise), 2)

        return DailyOutput(
            success=True,
            summary=f"Day {s.resources.day_current}: Balanced day — half study, half skill. "
                    + (f"Skill focused: {skill}." if skill else "No skill target set."),
            knowledge_gained=knowledge_gained,
            skill_gained=skill_gained,
            energy_after=max(0.0, s.resources.energy_current - 7.0),
            fatigue_after=min(100.0, s.resources.fatigue_current + 3.0),
            quality_score=max(0.1, 1.0 - noise),
            uncertainty=noise,
        )

    def _cram_mode(
        self, action: StudentAction, s: FullLatentState, step: int
    ) -> DailyOutput:
        noise = self.noise.quality_degradation(s.context.base_noise * 1.5)
        knowledge_gained = {
            subj: round(s.latent.true_learning_rates.get(subj, 0.8) * 12.0 * (1 - noise), 2)
            for subj in s.true_attendance
        }
        quality = max(0.1, 1.0 - noise - (0.4 if s.resources.fatigue_current > 50 else 0.0))
        return DailyOutput(
            success=True,
            summary=f"Day {s.resources.day_current}: CRAM MODE — massive knowledge boost. "
                    f"Fatigue will spike significantly.",
            knowledge_gained=knowledge_gained,
            energy_after=max(0.0, s.resources.energy_current - 12.0),
            fatigue_after=min(100.0, s.resources.fatigue_current + 15.0),
            quality_score=quality,
            uncertainty=noise * 1.5,
            data={"warning": "High fatigue cost. Use sparingly."},
        )

    def _rest(
        self, action: StudentAction, s: FullLatentState, step: int
    ) -> DailyOutput:
        recovery = 20.0 * s.context.energy_recovery_rate
        new_fatigue = max(0.0, s.resources.fatigue_current - recovery)
        return DailyOutput(
            success=True,
            summary=f"Day {s.resources.day_current}: Rest day. Fatigue reduced by ≈{round(recovery)}. "
                    f"Fatigue now ≈ {round(new_fatigue)}.",
            energy_after=s.resources.energy_max,
            fatigue_after=new_fatigue,
            quality_score=1.0,
            uncertainty=0.0,
        )

    def _submit_outcome(
        self, action: StudentAction, s: FullLatentState, step: int
    ) -> DailyOutput:
        p = s.progress
        projects = s.completed_projects
        project_value = sum(
            PROJECT_VALUE.get(ProjectTier(t), 0.0) for t in projects
        )
        avg_knowledge = sum(s.true_knowledge.values()) / max(len(s.true_knowledge), 1)
        academic_score = avg_knowledge * (1.0 if p.exam_eligible else 0.0)
        final_score = 0.3 * academic_score + 0.7 * project_value if p.exam_eligible else 0.0

        return DailyOutput(
            success=p.exam_eligible,
            summary=(
                f"EPISODE COMPLETE (Day {s.resources.day_current}). "
                f"Academic Score: {round(academic_score, 1)}, "
                f"Project Value: {round(project_value, 1)}, "
                f"Final Employability Score: {round(final_score, 2)}. "
                + ("PASSED all subjects." if p.passed_all_subjects else "FAILED — did not meet academic threshold.")
            ),
            quality_score=1.0,
            uncertainty=0.0,
            data={
                "academic_score": academic_score,
                "project_value": project_value,
                "final_score": final_score,
                "completed_projects": projects,
                "exam_eligible": p.exam_eligible,
                "passed": p.passed_all_subjects,
            },
        )

    def _default(
        self, action: StudentAction, s: FullLatentState, step: int
    ) -> DailyOutput:
        return DailyOutput(
            success=False,
            summary=f"Unknown action type: {action.action_type}",
            quality_score=0.0,
        )

    _HANDLERS = {
        ActionType.FULL_ACADEMIC:   _full_academic,
        ActionType.SKILL_DEEP_DIVE: _skill_deep_dive,
        ActionType.PROJECT_SPRINT:  _project_sprint,
        ActionType.BALANCED_LIFE:   _balanced_life,
        ActionType.CRAM_MODE:       _cram_mode,
        ActionType.REST:            _rest,
        ActionType.SUBMIT_OUTCOME:  _submit_outcome,
    }