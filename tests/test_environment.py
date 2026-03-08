"""Integration tests for the full BioExperimentEnvironment."""

from models import ActionType, ExperimentAction
from server.hackathon_environment import BioExperimentEnvironment


class TestEnvironmentLifecycle:
    def test_reset_returns_valid_observation(self):
        env = BioExperimentEnvironment()
        obs = env.reset()
        assert obs.step_index == 0
        assert obs.done is False
        assert obs.task.problem_statement != ""

    def test_step_increments_step_count(self):
        env = BioExperimentEnvironment()
        env.reset()
        obs = env.step(ExperimentAction(action_type=ActionType.COLLECT_SAMPLE))
        assert obs.step_index == 1
        assert env.state.step_count == 1

    def test_valid_pipeline_trajectory(self):
        env = BioExperimentEnvironment()
        env.reset()

        actions = [
            ExperimentAction(action_type=ActionType.COLLECT_SAMPLE,
                             parameters={"n_samples": 6}),
            ExperimentAction(action_type=ActionType.PREPARE_LIBRARY,
                             method="10x_chromium"),
            ExperimentAction(action_type=ActionType.SEQUENCE_CELLS),
            ExperimentAction(action_type=ActionType.RUN_QC),
            ExperimentAction(action_type=ActionType.FILTER_DATA),
            ExperimentAction(action_type=ActionType.NORMALIZE_DATA),
            ExperimentAction(action_type=ActionType.CLUSTER_CELLS),
            ExperimentAction(action_type=ActionType.DIFFERENTIAL_EXPRESSION,
                             parameters={"comparison": "disease_vs_healthy"}),
        ]

        for a in actions:
            obs = env.step(a)
            assert obs.latest_output is not None
            assert obs.latest_output.success is True, (
                f"Step {a.action_type} failed: {obs.rule_violations}"
            )

        assert obs.step_index == len(actions)
        assert obs.resource_usage.budget_used > 0

    def test_premature_de_blocked(self):
        env = BioExperimentEnvironment()
        env.reset()
        obs = env.step(ExperimentAction(
            action_type=ActionType.DIFFERENTIAL_EXPRESSION,
        ))
        assert obs.latest_output is not None
        assert obs.latest_output.success is False

    def test_conclusion_ends_episode(self):
        env = BioExperimentEnvironment()
        env.reset()

        quick_pipeline = [
            ExperimentAction(action_type=ActionType.COLLECT_SAMPLE),
            ExperimentAction(action_type=ActionType.PREPARE_LIBRARY),
            ExperimentAction(action_type=ActionType.SEQUENCE_CELLS),
            ExperimentAction(action_type=ActionType.RUN_QC),
            ExperimentAction(action_type=ActionType.FILTER_DATA),
            ExperimentAction(action_type=ActionType.NORMALIZE_DATA),
            ExperimentAction(action_type=ActionType.CLUSTER_CELLS),
            ExperimentAction(action_type=ActionType.DIFFERENTIAL_EXPRESSION,
                             parameters={"comparison": "disease_vs_healthy"}),
            ExperimentAction(
                action_type=ActionType.SYNTHESIZE_CONCLUSION,
                parameters={"claims": [
                    {"claim": "Test conclusion", "confidence": 0.7,
                     "claim_type": "correlational"},
                ]},
            ),
        ]
        for a in quick_pipeline:
            obs = env.step(a)

        assert obs.done is True
        assert obs.reward != 0.0
