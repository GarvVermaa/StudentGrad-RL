"""Integration tests for StudentEnvironment."""
import pytest
from models import ActionType, StudentAction, SkillType, ProjectTier
from server.student_environment import StudentEnvironment


def make_env() -> StudentEnvironment:
    env = StudentEnvironment(scenario_name="easy_single_subject")
    env.reset(seed=42)
    return env


def test_reset_returns_observation():
    env = StudentEnvironment()
    obs = env.reset(seed=0)
    assert obs.day == 0 or obs.day >= 0
    assert obs.energy > 0


def test_step_full_academic():
    env = make_env()
    obs = env.step(StudentAction(action_type=ActionType.FULL_ACADEMIC))
    assert obs is not None
    assert isinstance(obs.reward, float)


def test_step_skill_deep_dive():
    env = make_env()
    obs = env.step(StudentAction(
        action_type=ActionType.SKILL_DEEP_DIVE,
        skill_target=SkillType.JS,
    ))
    assert obs is not None


def test_step_rest():
    env = make_env()
    obs = env.step(StudentAction(action_type=ActionType.REST))
    assert obs.fatigue <= 10.0  # rest should reduce fatigue


def test_multiple_steps():
    env = make_env()
    for _ in range(10):
        obs = env.step(StudentAction(action_type=ActionType.BALANCED_LIFE))
    assert obs.step_index == 10


def test_submit_outcome_terminates():
    env = make_env()
    for _ in range(5):
        env.step(StudentAction(action_type=ActionType.FULL_ACADEMIC))
    obs = env.step(StudentAction(action_type=ActionType.SUBMIT_OUTCOME))
    assert obs.done is True
