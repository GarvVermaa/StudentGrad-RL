"""StudentGrad — Action, Observation and supporting types.

Drop-in replacement for the bio-experiment models.py.
Inherits from openenv.core.env_server.types exactly as the original did.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from openenv.core.env_server.types import Action, Observation


# ── Action Space ────────────────────────────────────────────────────────

class ActionType(str, Enum):
    FULL_ACADEMIC    = "full_academic"      # 100% attendance/knowledge grind
    SKILL_DEEP_DIVE  = "skill_deep_dive"    # grind one tech skill
    PROJECT_SPRINT   = "project_sprint"     # build a project (needs skill prereqs)
    BALANCED_LIFE    = "balanced_life"      # 50% study / 50% skill
    CRAM_MODE        = "cram_mode"          # high-intensity study, only near exams
    REST             = "rest"               # recover energy, reduce fatigue
    SUBMIT_OUTCOME   = "submit_outcome"     # terminal action — end the episode


# Skills the agent can learn
class SkillType(str, Enum):
    JS      = "js"
    NODE    = "node"
    DOCKER  = "docker"
    HTML    = "html"
    CSS     = "css"


# College subjects
class SubjectType(str, Enum):
    DSA   = "dsa"
    DBMS  = "dbms"
    OS    = "os"
    MATHS = "maths"
    COA   = "coa"


# Project tiers
class ProjectTier(str, Enum):
    BASIC     = "basic"      # HTML/CSS frontend
    FULLSTACK = "fullstack"  # JS + DBMS backend
    CLOUD     = "cloud"      # Node + Docker DevOps


# ── Supporting models ────────────────────────────────────────────────────

class SessionRecord(BaseModel):
    """One day's activity log — visible to the agent."""
    day: int
    action_type: ActionType
    skill_target: Optional[SkillType] = None
    project_target: Optional[ProjectTier] = None
    energy_spent: float = 0.0
    reward: float = 0.0
    summary: str = ""


class DailyOutput(BaseModel):
    """Simulated result of one day's routine — visible to the agent."""
    success: bool = True
    summary: str = ""
    knowledge_gained: Dict[str, float] = Field(default_factory=dict)   # subject → delta
    skill_gained: Dict[str, float] = Field(default_factory=dict)        # skill → delta
    project_progress: float = 0.0                                        # 0-1 delta
    energy_after: float = 10.0
    fatigue_after: float = 0.0
    quality_score: float = 1.0
    uncertainty: float = 0.1
    warnings: List[str] = Field(default_factory=list)
    data: Dict[str, Any] = Field(default_factory=dict)


class ResourceUsage(BaseModel):
    """Agent-visible resource summary."""
    days_used: int = 0
    days_remaining: int = 365
    energy_current: float = 10.0
    fatigue_current: float = 0.0


class StudentTaskSpec(BaseModel):
    """The problem statement shown to the agent at episode start."""
    problem_statement: str = (
        "Navigate a 365-day academic year. Pass all 5 subjects (≥40% score, "
        "≥40% attendance). Build the highest-tier projects possible to maximise "
        "your Employability Score = 0.3 × AcademicScore + 0.7 × ProjectValue."
    )
    difficulty: str = "hard"
    target_subjects: List[str] = Field(
        default_factory=lambda: ["dsa", "dbms", "os", "maths", "coa"]
    )
    target_skills: List[str] = Field(
        default_factory=lambda: ["js", "node", "docker", "html", "css"]
    )
    success_criteria: List[str] = Field(
        default_factory=lambda: [
            "attendance_all_above_40",
            "exam_score_all_above_40",
            "at_least_one_project_completed",
        ]
    )
    exam_day: int = 300         # day when final exams are evaluated
    cram_window_days: int = 15  # days before exam where CRAM_MODE is legal


# ── Action model ─────────────────────────────────────────────────────────


class StudentAction(Action):
    """One daily routine chosen by the agent."""

    action_type: ActionType
    skill_target: Optional[SkillType] = None
    project_target: Optional[ProjectTier] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)
    justification: Optional[str] = None
    confidence: float = Field(0.5, ge=0.0, le=1.0)

    @field_validator("action_type", mode="before")
    @classmethod
    def to_lowercase(cls, v: Any) -> Any:
        """Normalise 'REST' or 'Rest' to 'rest' to match the ActionType Enum."""
        if isinstance(v, str):
            return v.lower()
        return v

# ── Observation model ────────────────────────────────────────────────────

class StudentObservation(Observation):
    """Everything the agent can see after each step.

    Inherits ``done``, ``reward``, ``metadata`` from the SDK base class.
    """

    task: StudentTaskSpec
    day: int                                        # current day (1-365)
    step_index: int

    # Academic state (noisy estimates — not the hidden truth)
    attendance: Dict[str, float] = Field(default_factory=dict)   # subject → %
    knowledge: Dict[str, float] = Field(default_factory=dict)    # subject → 0-100

    # Skill & project state
    skills: Dict[str, float] = Field(default_factory=dict)       # skill → points
    completed_projects: List[str] = Field(default_factory=list)  # tier names
    active_project_progress: float = 0.0                         # 0-1

    # Energy & fatigue
    energy: float = 10.0    # 0-10
    fatigue: float = 0.0    # 0-100 (burnout at 80+)

    # History & resources
    session_history: List[SessionRecord] = Field(default_factory=list)
    resource_usage: ResourceUsage = Field(default_factory=ResourceUsage)
    latest_output: Optional[DailyOutput] = None

    # Rule feedback
    rule_violations: List[str] = Field(default_factory=list)
    step_reward_breakdown: Dict[str, float] = Field(default_factory=dict)

    # Random events (visible to agent)
    sick_today: bool = False
    surprise_quiz_today: bool = False


# ── Skill-Project prerequisite registry ─────────────────────────────────

PROJECT_PREREQS: Dict[ProjectTier, Dict[SkillType, float]] = {
    ProjectTier.BASIC:     {SkillType.HTML: 5.0, SkillType.CSS: 5.0},
    ProjectTier.FULLSTACK: {SkillType.JS: 10.0},
    ProjectTier.CLOUD:     {SkillType.NODE: 10.0, SkillType.DOCKER: 10.0},
}

PROJECT_VALUE: Dict[ProjectTier, float] = {
    ProjectTier.BASIC:     20.0,
    ProjectTier.FULLSTACK: 50.0,
    ProjectTier.CLOUD:     100.0,
}

# Synergy: attending these subjects gives bonus skill points
SUBJECT_SKILL_SYNERGY: Dict[SubjectType, Dict[SkillType, float]] = {
    SubjectType.DBMS:  {SkillType.NODE: 0.5},
    SubjectType.DSA:   {SkillType.JS: 0.5},
    SubjectType.OS:    {SkillType.DOCKER: 0.5},
}

# Energy costs per routine
ACTION_ENERGY_COST: Dict[ActionType, float] = {
    ActionType.FULL_ACADEMIC:   10.0,
    ActionType.SKILL_DEEP_DIVE:  8.0,
    ActionType.PROJECT_SPRINT:   9.0,
    ActionType.BALANCED_LIFE:    7.0,
    ActionType.CRAM_MODE:       12.0,
    ActionType.REST:             0.0,
    ActionType.SUBMIT_OUTCOME:   0.0,
}

# Fatigue added per routine (recovered by REST)
ACTION_FATIGUE_DELTA: Dict[ActionType, float] = {
    ActionType.FULL_ACADEMIC:    5.0,
    ActionType.SKILL_DEEP_DIVE:  4.0,
    ActionType.PROJECT_SPRINT:   5.0,
    ActionType.BALANCED_LIFE:    3.0,
    ActionType.CRAM_MODE:       15.0,
    ActionType.REST:           -20.0,   # negative = recovery
    ActionType.SUBMIT_OUTCOME:   0.0,
}

META_ACTIONS = {ActionType.SUBMIT_OUTCOME}


def build_agent_system_prompt() -> str:
    return """You are an RL agent managing a student's 365-day academic year.

GOAL: Maximise Employability Score = 0.3 × AcademicScore + 0.7 × ProjectValue
HARD CONSTRAINT: All 5 subjects must have attendance ≥ 40% AND exam score ≥ 40 by Day 300.
                 Failing this gives reward = 0 regardless of projects.

ACTIONS (choose one per day):
  full_academic    — attend all classes. Cost: 10 energy. Boosts attendance + knowledge.
  skill_deep_dive  — grind one tech skill (set skill_target). Cost: 8 energy.
  project_sprint   — build a project (set project_target). Cost: 9 energy. Requires skill prereqs.
  balanced_life    — 50/50 study+skill split. Cost: 7 energy.
  cram_mode        — emergency study. Cost: 12 energy + high fatigue. Only legal within 15 days of exam (Day 285-300).
  rest             — recover energy and reduce fatigue. Cost: 0 energy.
  submit_outcome   — end the episode. Use only when Day >= 300 AND exams are done.

SKILL PREREQS FOR PROJECTS:
  basic     → html ≥ 5, css ≥ 5
  fullstack → js ≥ 10
  cloud     → node ≥ 10, docker ≥ 10

SYNERGIES (free bonus skill points from attending class):
  dbms class → +0.5 node skill
  dsa class  → +0.5 js skill
  os class   → +0.5 docker skill

ENERGY: Resets to 10 each day. If you try to exceed your energy budget, the action is downgraded.
FATIGUE: Accumulates. Above 80 = burnout penalty. REST reduces it significantly.

Respond ONLY with valid JSON:
{"action_type": "skill_deep_dive", "skill_target": "js", "project_target": null, "justification": "...", "confidence": 0.8}

For submit_outcome:
{"action_type": "submit_outcome", "justification": "Day 300 reached, exams done.", "confidence": 0.9}
"""