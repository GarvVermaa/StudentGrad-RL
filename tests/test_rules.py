"""Tests for the biological rule engine."""

from models import ActionType, ExperimentAction
from server.rules.engine import RuleEngine, Severity
from server.simulator.latent_state import (
    ExperimentProgress,
    FullLatentState,
    ResourceState,
)


def _state(**progress_flags) -> FullLatentState:
    return FullLatentState(
        progress=ExperimentProgress(**progress_flags),
        resources=ResourceState(budget_total=100_000, time_limit_days=180),
    )


class TestPrerequisites:
    def test_sequence_without_library_blocked(self):
        engine = RuleEngine()
        violations = engine.check(
            ExperimentAction(action_type=ActionType.SEQUENCE_CELLS),
            _state(samples_collected=True),
        )
        hard = engine.hard_violations(violations)
        assert any("library" in m.lower() for m in hard)

    def test_sequence_with_library_allowed(self):
        engine = RuleEngine()
        violations = engine.check(
            ExperimentAction(action_type=ActionType.SEQUENCE_CELLS),
            _state(samples_collected=True, library_prepared=True),
        )
        hard = engine.hard_violations(violations)
        assert not hard

    def test_de_without_normalization_blocked(self):
        engine = RuleEngine()
        violations = engine.check(
            ExperimentAction(action_type=ActionType.DIFFERENTIAL_EXPRESSION),
            _state(cells_sequenced=True, qc_performed=True, data_filtered=True),
        )
        hard = engine.hard_violations(violations)
        assert any("normalis" in m.lower() or "normaliz" in m.lower() for m in hard)

    def test_validate_marker_without_discovery_blocked(self):
        engine = RuleEngine()
        violations = engine.check(
            ExperimentAction(action_type=ActionType.VALIDATE_MARKER),
            _state(de_performed=True),
        )
        hard = engine.hard_violations(violations)
        assert any("marker" in m.lower() for m in hard)


class TestRedundancy:
    def test_double_qc_is_soft(self):
        engine = RuleEngine()
        violations = engine.check(
            ExperimentAction(action_type=ActionType.RUN_QC),
            _state(cells_sequenced=True, qc_performed=True),
        )
        hard = engine.hard_violations(violations)
        soft = engine.soft_violations(violations)
        assert not hard
        assert any("redundant" in m.lower() for m in soft)


class TestResourceConstraints:
    def test_exhausted_budget_blocked(self):
        s = _state()
        s.resources.budget_used = 100_000
        engine = RuleEngine()
        violations = engine.check(
            ExperimentAction(action_type=ActionType.COLLECT_SAMPLE), s,
        )
        hard = engine.hard_violations(violations)
        assert any("budget" in m.lower() for m in hard)
