"""Tests for the decomposable reward function."""

from models import ActionType, ConclusionClaim, ExperimentAction, IntermediateOutput, OutputType
from server.rewards.reward import RewardComputer
from server.simulator.latent_state import (
    ExperimentProgress,
    FullLatentState,
    LatentBiologicalState,
    ResourceState,
)


def _states(
    prev_flags: dict | None = None,
    next_flags: dict | None = None,
    budget_used: float = 0.0,
):
    prev = FullLatentState(
        progress=ExperimentProgress(**(prev_flags or {})),
        resources=ResourceState(budget_total=100_000, budget_used=budget_used),
    )
    nf = dict(prev_flags or {})
    nf.update(next_flags or {})
    nxt = FullLatentState(
        progress=ExperimentProgress(**nf),
        resources=ResourceState(budget_total=100_000, budget_used=budget_used + 5000),
    )
    return prev, nxt


class TestStepReward:
    def test_valid_step_positive(self):
        rc = RewardComputer()
        prev, nxt = _states(
            prev_flags={"samples_collected": True, "library_prepared": True},
            next_flags={"cells_sequenced": True},
        )
        output = IntermediateOutput(
            output_type=OutputType.SEQUENCING_RESULT,
            step_index=1,
            quality_score=0.85,
            uncertainty=0.15,
        )
        rb = rc.step_reward(
            ExperimentAction(action_type=ActionType.SEQUENCE_CELLS),
            prev, nxt, output, [], [],
        )
        assert rb.total > 0

    def test_hard_violation_negative(self):
        rc = RewardComputer()
        prev, nxt = _states()
        output = IntermediateOutput(
            output_type=OutputType.FAILURE_REPORT,
            step_index=1,
            success=False,
        )
        rb = rc.step_reward(
            ExperimentAction(action_type=ActionType.SEQUENCE_CELLS),
            prev, nxt, output, ["blocked"], [],
        )
        assert rb.total < 0


class TestTerminalReward:
    def test_correct_conclusion_rewarded(self):
        rc = RewardComputer()
        state = FullLatentState(
            biology=LatentBiologicalState(
                causal_mechanisms=["TGF-beta-driven fibrosis"],
                true_markers=["NPPA"],
            ),
            progress=ExperimentProgress(
                samples_collected=True, cells_sequenced=True,
                qc_performed=True, data_filtered=True,
                data_normalized=True, de_performed=True,
                conclusion_reached=True,
            ),
            resources=ResourceState(budget_total=100_000, budget_used=40_000),
        )
        claims = [
            ConclusionClaim(
                claim="TGF-beta-driven fibrosis observed",
                confidence=0.9,
                claim_type="causal",
            ),
        ]
        rb = rc.terminal_reward(state, claims, [])
        assert rb.terminal > 0

    def test_overconfident_wrong_claim_penalised(self):
        rc = RewardComputer()
        state = FullLatentState(
            biology=LatentBiologicalState(causal_mechanisms=["real_mechanism"]),
            progress=ExperimentProgress(conclusion_reached=True),
        )
        claims = [
            ConclusionClaim(
                claim="completely_wrong_mechanism",
                confidence=0.95,
                claim_type="causal",
            ),
        ]
        rb = rc.terminal_reward(state, claims, [])
        assert rb.components.get("overconfidence_penalty", 0) < 0
