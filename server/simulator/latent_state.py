"""Hidden student world state — never directly visible to the agent."""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class LatentStudentState(BaseModel):
    """Ground truth the agent cannot observe directly."""

    # True learning rates per subject (how much knowledge per study hour)
    true_learning_rates: Dict[str, float] = Field(
        default_factory=lambda: {
            "dsa": 0.8, "dbms": 0.9, "os": 0.75, "maths": 0.7, "coa": 0.85
        }
    )
    # True fatigue threshold — agent doesn't know exactly when burnout kicks in
    true_fatigue_threshold: float = 80.0

    # True exam difficulty multiplier per subject (higher = harder)
    true_exam_difficulty: Dict[str, float] = Field(
        default_factory=lambda: {
            "dsa": 1.2, "dbms": 1.0, "os": 1.1, "maths": 1.3, "coa": 1.0
        }
    )
    # True skill acquisition rates
    true_skill_rates: Dict[str, float] = Field(
        default_factory=lambda: {
            "js": 1.0, "node": 0.9, "docker": 0.85, "html": 1.2, "css": 1.1
        }
    )


class StudentContextState(BaseModel):
    """Noise / context parameters shaping daily variation."""
    base_noise: float = 0.1          # Gaussian noise on all outputs
    sick_probability: float = 0.05   # daily probability of sick day
    quiz_probability: float = 0.03   # daily probability of surprise quiz
    sick_duration_days: int = 2
    energy_recovery_rate: float = 1.0  # multiplier on REST recovery


class StudentProgress(BaseModel):
    """Boolean milestone flags — shared across engine, rules, reward."""

    # Academic milestones
    attended_first_class: bool = False
    above_75_attendance_any: bool = False       # any subject ≥ 75%
    above_75_attendance_all: bool = False       # all subjects ≥ 75%
    studied_all_subjects: bool = False
    exam_eligible: bool = False                 # all attendance ≥ 40%
    exam_taken: bool = False
    passed_all_subjects: bool = False

    # Skill milestones
    first_skill_acquired: bool = False
    any_skill_above_10: bool = False
    skill_prereqs_basic_met: bool = False
    skill_prereqs_fullstack_met: bool = False
    skill_prereqs_cloud_met: bool = False

    # Project milestones
    basic_project_done: bool = False
    fullstack_project_done: bool = False
    cloud_project_done: bool = False

    # Safety
    burnout_occurred: bool = False
    academic_failed: bool = False               # attendance or score < 40

    # Counters
    total_days_studied: int = 0
    total_skill_days: int = 0
    total_rest_days: int = 0
    total_cram_days: int = 0
    sick_days_remaining: int = 0                # countdown for sick event


class ResourceState(BaseModel):
    """Day/energy tracking (mirrors the bio ResourceState interface)."""

    day_total: int = 365
    day_current: int = 0
    energy_max: float = 10.0
    energy_current: float = 10.0
    fatigue_current: float = 0.0

    @property
    def days_remaining(self) -> int:
        return max(0, self.day_total - self.day_current)

    @property
    def budget_remaining(self) -> float:
        """Alias so reward.py can use the same interface."""
        return float(self.days_remaining)

    @property
    def budget_total(self) -> float:
        return float(self.day_total)

    @property
    def budget_used(self) -> float:
        return float(self.day_current)

    @property
    def time_remaining_days(self) -> float:
        return float(self.days_remaining)

    @property
    def time_limit_days(self) -> float:
        return float(self.day_total)

    @property
    def budget_exhausted(self) -> bool:
        return self.day_current >= self.day_total

    @property
    def time_exhausted(self) -> bool:
        return self.budget_exhausted


class FullLatentState(BaseModel):
    """Complete hidden state of the student simulation world."""

    latent: LatentStudentState = Field(default_factory=LatentStudentState)
    context: StudentContextState = Field(default_factory=StudentContextState)
    progress: StudentProgress = Field(default_factory=StudentProgress)
    resources: ResourceState = Field(default_factory=ResourceState)

    # True internal academic state (noisy version exposed in observation)
    true_attendance: Dict[str, float] = Field(
        default_factory=lambda: {
            "dsa": 0.0, "dbms": 0.0, "os": 0.0, "maths": 0.0, "coa": 0.0
        }
    )
    true_knowledge: Dict[str, float] = Field(
        default_factory=lambda: {
            "dsa": 10.0, "dbms": 10.0, "os": 10.0, "maths": 10.0, "coa": 10.0
        }
    )
    true_skills: Dict[str, float] = Field(
        default_factory=lambda: {
            "js": 0.0, "node": 0.0, "docker": 0.0, "html": 0.0, "css": 0.0
        }
    )
    true_project_progress: float = 0.0         # progress on current active project
    active_project_tier: Optional[str] = None  # tier being worked on
    completed_projects: List[str] = Field(default_factory=list)

    hidden_failure_conditions: List[str] = Field(default_factory=list)
    rng_seed: int = 42
    step_count: int = 0

    # Transient (not serialized) — used within a single step
    last_sick_triggered: Optional[bool] = Field(None, exclude=True)
    last_quiz_triggered: Optional[bool] = Field(None, exclude=True)