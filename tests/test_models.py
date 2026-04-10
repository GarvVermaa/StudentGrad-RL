"""Tests for StudentGrad models."""
import pytest
from models import (
    ActionType,
    StudentAction,
    StudentObservation,
    StudentTaskSpec,
    SkillType,
    ProjectTier,
)


def test_action_type_values():
    assert ActionType.FULL_ACADEMIC == "full_academic"
    assert ActionType.SUBMIT_OUTCOME == "submit_outcome"


def test_student_action_basic():
    action = StudentAction(action_type=ActionType.REST)
    assert action.action_type == ActionType.REST


def test_student_action_lowercase_coercion():
    """Validator must coerce uppercase strings to lowercase ActionType."""
    action = StudentAction(action_type="REST")
    assert action.action_type == ActionType.REST


def test_student_action_with_skill():
    action = StudentAction(
        action_type=ActionType.SKILL_DEEP_DIVE,
        skill_target=SkillType.JS,
        justification="Need JS for fullstack project",
        confidence=0.9,
    )
    assert action.skill_target == SkillType.JS


def test_student_action_with_project():
    action = StudentAction(
        action_type=ActionType.PROJECT_SPRINT,
        project_target=ProjectTier.FULLSTACK,
    )
    assert action.project_target == ProjectTier.FULLSTACK


def test_task_spec_defaults():
    spec = StudentTaskSpec()
    assert "dsa" in spec.target_subjects
    assert spec.exam_day == 300


def test_confidence_bounds():
    with pytest.raises(Exception):
        StudentAction(action_type=ActionType.REST, confidence=1.5)
