"""Concrete scenario definitions for StudentGrad."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from models import StudentTaskSpec
from server.simulator.latent_state import (
    FullLatentState,
    LatentStudentState,
    StudentContextState,
    StudentProgress,
    ResourceState,
)


@dataclass
class Scenario:
    name: str
    task: StudentTaskSpec
    latent: LatentStudentState
    context: StudentContextState
    hidden_failure_conditions: List[str] = field(default_factory=list)
    difficulty: str = "hard"
    tags: List[str] = field(default_factory=list)

    def build_full_state(self) -> FullLatentState:
        return FullLatentState(
            latent=self.latent,
            context=self.context,
            progress=StudentProgress(),
            resources=ResourceState(
                day_total={"easy": 30, "medium": 180, "hard": 365}.get(self.difficulty, 365)
            ),
            hidden_failure_conditions=self.hidden_failure_conditions,
        )


SCENARIOS = [
    Scenario(
        name="easy_single_subject",
        difficulty="easy",
        task=StudentTaskSpec(
            problem_statement="Pass DSA in 30 days. Attend class and study consistently.",
            difficulty="easy",
            target_subjects=["dsa"],
            target_skills=["js"],
            success_criteria=["attendance_above_40", "exam_score_above_40"],
            exam_day=25,
            cram_window_days=5,
        ),
        latent=LatentStudentState(
            true_learning_rates={"dsa": 1.0, "dbms": 1.0, "os": 1.0, "maths": 1.0, "coa": 1.0},
            true_fatigue_threshold=90.0,
            true_exam_difficulty={"dsa": 0.8, "dbms": 1.0, "os": 1.0, "maths": 1.0, "coa": 1.0},
        ),
        context=StudentContextState(
            sick_probability=0.02,
            quiz_probability=0.01,
            base_noise=0.05,
        ),
        hidden_failure_conditions=["skip_too_many_classes"],
        tags=["easy", "single-subject"],
    ),
    Scenario(
        name="medium_three_subjects_basic_project",
        difficulty="medium",
        task=StudentTaskSpec(
            problem_statement=(
                "Pass DSA, DBMS, and OS in 180 days. "
                "Also complete a Basic frontend project."
            ),
            difficulty="medium",
            target_subjects=["dsa", "dbms", "os"],
            target_skills=["js", "html", "css"],
            success_criteria=[
                "attendance_above_40_dsa_dbms_os",
                "exam_score_above_40",
                "basic_project_done",
            ],
            exam_day=150,
            cram_window_days=10,
        ),
        latent=LatentStudentState(
            true_learning_rates={"dsa": 0.9, "dbms": 1.0, "os": 0.85, "maths": 0.7, "coa": 0.85},
            true_fatigue_threshold=85.0,
            true_exam_difficulty={"dsa": 1.1, "dbms": 1.0, "os": 1.0, "maths": 1.2, "coa": 1.0},
        ),
        context=StudentContextState(
            sick_probability=0.04,
            quiz_probability=0.02,
            base_noise=0.08,
        ),
        hidden_failure_conditions=["ignore_projects", "burn_out_early"],
        tags=["medium", "multi-subject", "basic-project"],
    ),
    Scenario(
        name="hard_full_year",
        difficulty="hard",
        task=StudentTaskSpec(
            problem_statement=(
                "Navigate the full 365-day academic year. "
                "Pass all 5 subjects AND build the highest-tier projects to maximise Employability Score."
            ),
            difficulty="hard",
            target_subjects=["dsa", "dbms", "os", "maths", "coa"],
            target_skills=["js", "node", "docker", "html", "css"],
            success_criteria=[
                "attendance_all_above_40",
                "exam_score_all_above_40",
                "cloud_project_done",
            ],
            exam_day=300,
            cram_window_days=15,
        ),
        latent=LatentStudentState(
            true_learning_rates={"dsa": 0.8, "dbms": 0.9, "os": 0.75, "maths": 0.7, "coa": 0.85},
            true_fatigue_threshold=80.0,
            true_exam_difficulty={"dsa": 1.2, "dbms": 1.0, "os": 1.1, "maths": 1.3, "coa": 1.0},
            true_skill_rates={"js": 1.0, "node": 0.9, "docker": 0.85, "html": 1.2, "css": 1.1},
        ),
        context=StudentContextState(
            sick_probability=0.05,
            quiz_probability=0.03,
            base_noise=0.10,
        ),
        hidden_failure_conditions=[
            "ignore_attendance_entirely",
            "burn_out_by_day_100",
            "no_projects_at_all",
        ],
        tags=["hard", "full-year", "all-subjects", "cloud-project"],
    ),
]

SCENARIO_MAP = {s.name: s for s in SCENARIOS}


def get_scenario(name: str) -> Scenario:
    if name not in SCENARIO_MAP:
        raise ValueError(f"Unknown scenario: {name}. Available: {list(SCENARIO_MAP.keys())}")
    return SCENARIO_MAP[name]