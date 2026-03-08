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

    def test_premature_meta_action_gets_penalized(self):
        rc = RewardComputer()
        prev, nxt = _states(
            prev_flags={"data_normalized": True},
            next_flags={"followup_designed": True},
            budget_used=2_000,
        )
        output = IntermediateOutput(
            output_type=OutputType.FOLLOWUP_DESIGN,
            step_index=2,
            quality_score=1.0,
            uncertainty=0.0,
        )
        rb = rc.step_reward(
            ExperimentAction(action_type=ActionType.DESIGN_FOLLOWUP),
            prev,
            nxt,
            output,
            [],
            [],
        )
        assert rb.components.get("premature_meta_action_penalty", 0.0) < 0.0


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
        rb = rc.terminal_reward(
            state,
            claims,
            [],
            discovered_markers=["NPPA"],
            candidate_mechanisms=["TGF-beta-driven fibrosis"],
        )
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

    def test_discovery_error_penalizes_wrong_markers_and_mechanisms(self):
        rc = RewardComputer()
        state = FullLatentState(
            biology=LatentBiologicalState(
                true_markers=["NPPA", "NPPB"],
                causal_mechanisms=["TGF-beta-driven fibrosis"],
            ),
            progress=ExperimentProgress(
                samples_collected=True,
                cells_sequenced=True,
                qc_performed=True,
                data_filtered=True,
                data_normalized=True,
                de_performed=True,
                markers_discovered=True,
                conclusion_reached=True,
            ),
            resources=ResourceState(budget_total=100_000, budget_used=40_000),
        )

        aligned = rc.terminal_reward(
            state,
            [],
            [],
            discovered_markers=["NPPA", "NPPB"],
            candidate_mechanisms=["TGF-beta-driven fibrosis"],
        )
        misaligned = rc.terminal_reward(
            state,
            [],
            [],
            discovered_markers=["WRONG1", "WRONG2"],
            candidate_mechanisms=["unrelated inflammatory process"],
        )

        assert aligned.components["discovery_alignment"] > misaligned.components["discovery_alignment"]
        assert aligned.components["discovery_error_penalty"] > misaligned.components["discovery_error_penalty"]
        assert aligned.terminal > misaligned.terminal

    def test_conclusion_error_penalizes_wrong_structured_claims(self):
        rc = RewardComputer()
        state = FullLatentState(
            biology=LatentBiologicalState(
                true_markers=["NPPA", "NPPB"],
                causal_mechanisms=["TGF-beta-driven fibrosis"],
            ),
            progress=ExperimentProgress(
                data_normalized=True,
                de_performed=True,
                markers_discovered=True,
                pathways_analyzed=True,
                conclusion_reached=True,
            ),
            resources=ResourceState(budget_total=100_000, budget_used=40_000),
        )
        aligned = rc.terminal_reward(
            state,
            [
                ConclusionClaim(
                    top_markers=["NPPA", "NPPB"],
                    causal_mechanisms=["TGF-beta-driven fibrosis"],
                    confidence=0.8,
                ),
            ],
            [],
        )
        misaligned = rc.terminal_reward(
            state,
            [
                ConclusionClaim(
                    top_markers=["WRONG1"],
                    causal_mechanisms=["unrelated process"],
                    confidence=0.8,
                ),
            ],
            [],
        )

        assert aligned.components["conclusion_alignment"] > misaligned.components["conclusion_alignment"]
        assert aligned.components["conclusion_error_penalty"] > misaligned.components["conclusion_error_penalty"]
        assert aligned.terminal > misaligned.terminal
