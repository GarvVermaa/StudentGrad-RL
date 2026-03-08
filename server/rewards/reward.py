"""Decomposable reward function for the bio-experiment planning POMDP.

Reward components
─────────────────
  r_validity      — biological validity of the chosen action
  r_ordering      — correct ordering of experiment steps
  r_info_gain     — information gain from the step's output
  r_efficiency    — resource efficiency (budget & time normalised)
  r_novelty       — bonus for non-redundant, non-trivial actions
  r_penalty       — penalties for violations, redundancy, waste
  r_terminal      — terminal quality & calibration against hidden truth

Potential-based shaping
  φ(s)            — progress potential used for dense shaping signal

The final step reward is:
  R_t = r_validity + r_ordering + r_info_gain + r_efficiency
        + r_novelty + r_penalty + γ[φ(s_{t+1}) − φ(s_t)]

The terminal reward adds:
  R_T += r_terminal
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from models import (
    ActionType,
    ConclusionClaim,
    ExperimentAction,
    IntermediateOutput,
    META_ACTIONS,
    WET_LAB_ACTIONS,
)

from server.simulator.latent_state import FullLatentState


@dataclass
class RewardBreakdown:
    validity: float = 0.0
    ordering: float = 0.0
    info_gain: float = 0.0
    efficiency: float = 0.0
    novelty: float = 0.0
    penalty: float = 0.0
    shaping: float = 0.0
    terminal: float = 0.0
    components: Dict[str, float] = field(default_factory=dict)

    @property
    def total(self) -> float:
        return (
            self.validity
            + self.ordering
            + self.info_gain
            + self.efficiency
            + self.novelty
            + self.penalty
            + self.shaping
            + self.terminal
        )

    def to_dict(self) -> Dict[str, float]:
        d = {
            "validity": self.validity,
            "ordering": self.ordering,
            "info_gain": self.info_gain,
            "efficiency": self.efficiency,
            "novelty": self.novelty,
            "penalty": self.penalty,
            "shaping": self.shaping,
            "terminal": self.terminal,
            "total": self.total,
        }
        d.update(self.components)
        return d


class RewardComputer:
    """Computes step-wise and terminal rewards.

    Parameters
    ----------
    gamma : float
        Discount factor for potential-based shaping (default 0.99).
    efficiency_weight : float
        Relative importance of resource efficiency.
    """

    def __init__(
        self,
        gamma: float = 0.99,
        efficiency_weight: float = 0.3,
        info_gain_weight: float = 0.4,
        validity_weight: float = 0.3,
    ):
        self.gamma = gamma
        self.w_eff = efficiency_weight
        self.w_ig = info_gain_weight
        self.w_val = validity_weight

    # ── step reward ─────────────────────────────────────────────────────

    def step_reward(
        self,
        action: ExperimentAction,
        prev_state: FullLatentState,
        next_state: FullLatentState,
        output: IntermediateOutput,
        hard_violations: List[str],
        soft_violations: List[str],
    ) -> RewardBreakdown:
        rb = RewardBreakdown()

        # validity
        if hard_violations:
            rb.validity = -1.0
            rb.penalty = -0.5 * len(hard_violations)
            rb.components["hard_violations"] = len(hard_violations)
            return rb

        rb.validity = self.w_val * (1.0 if output.success else 0.0)

        # ordering bonus: +0.2 if the step was a natural next step
        rb.ordering = 0.2 * self._ordering_score(action, prev_state)

        # information gain proxy: quality × (1 - uncertainty)
        rb.info_gain = self.w_ig * output.quality_score * (1.0 - output.uncertainty)

        # efficiency: normalised cost relative to budget
        budget_frac = (
            (next_state.resources.budget_used - prev_state.resources.budget_used)
            / max(next_state.resources.budget_total, 1)
        )
        rb.efficiency = self.w_eff * max(0.0, 1.0 - 5.0 * budget_frac)

        # novelty: small bonus for non-redundant steps
        if not soft_violations:
            rb.novelty = 0.1

        # penalties
        rb.penalty = -0.15 * len(soft_violations)

        # potential-based shaping
        phi_prev = self._potential(prev_state)
        phi_next = self._potential(next_state)
        rb.shaping = self.gamma * phi_next - phi_prev

        return rb

    # ── terminal reward ─────────────────────────────────────────────────

    def terminal_reward(
        self,
        state: FullLatentState,
        conclusions: List[ConclusionClaim],
        task_success_criteria: List[str],
    ) -> RewardBreakdown:
        rb = RewardBreakdown()

        # pipeline completeness (0-1)
        completeness = self._completeness(state)
        rb.components["completeness"] = completeness

        # calibration: how well conclusions align with hidden ground truth
        calibration = self._calibration(state, conclusions)
        rb.components["calibration"] = calibration

        # efficiency bonus at terminal
        budget_eff = state.resources.budget_remaining / max(
            state.resources.budget_total, 1
        )
        time_eff = state.resources.time_remaining_days / max(
            state.resources.time_limit_days, 1
        )
        rb.components["budget_efficiency"] = budget_eff
        rb.components["time_efficiency"] = time_eff

        # over-confidence penalty
        overconf = self._overconfidence_penalty(state, conclusions)
        rb.components["overconfidence_penalty"] = overconf

        rb.terminal = (
            3.0 * completeness
            + 4.0 * calibration
            + 1.0 * (budget_eff + time_eff) / 2.0
            + overconf
        )
        return rb

    # ── helpers ─────────────────────────────────────────────────────────

    def _ordering_score(
        self, action: ExperimentAction, s: FullLatentState
    ) -> float:
        """Heuristic: 1.0 if this step naturally follows the current progress."""
        at = action.action_type
        p = s.progress
        NATURAL_NEXT = {
            ActionType.COLLECT_SAMPLE: not p.samples_collected,
            ActionType.PREPARE_LIBRARY: p.samples_collected and not p.library_prepared,
            ActionType.SEQUENCE_CELLS: p.library_prepared and not p.cells_sequenced,
            ActionType.RUN_QC: p.cells_sequenced and not p.qc_performed,
            ActionType.FILTER_DATA: p.qc_performed and not p.data_filtered,
            ActionType.NORMALIZE_DATA: p.data_filtered and not p.data_normalized,
            ActionType.CLUSTER_CELLS: p.data_normalized and not p.cells_clustered,
            ActionType.DIFFERENTIAL_EXPRESSION: p.data_normalized and not p.de_performed,
            ActionType.PATHWAY_ENRICHMENT: p.de_performed and not p.pathways_analyzed,
            ActionType.MARKER_SELECTION: p.de_performed and not p.markers_discovered,
            ActionType.VALIDATE_MARKER: p.markers_discovered and not p.markers_validated,
            ActionType.SYNTHESIZE_CONCLUSION: (
                p.de_performed or p.cells_clustered
            ) and not p.conclusion_reached,
        }
        return 1.0 if NATURAL_NEXT.get(at, False) else 0.3

    def _potential(self, s: FullLatentState) -> float:
        """Progress potential φ(s) — counts completed milestones."""
        p = s.progress
        milestones = [
            p.samples_collected,
            p.library_prepared,
            p.cells_sequenced,
            p.qc_performed,
            p.data_filtered,
            p.data_normalized,
            p.cells_clustered,
            p.de_performed,
            p.pathways_analyzed,
            p.markers_discovered,
            p.markers_validated,
            p.conclusion_reached,
        ]
        return sum(milestones) / len(milestones)

    def _completeness(self, s: FullLatentState) -> float:
        p = s.progress
        core = [
            p.samples_collected,
            p.cells_sequenced,
            p.qc_performed,
            p.data_filtered,
            p.data_normalized,
            p.de_performed or p.cells_clustered,
            p.conclusion_reached,
        ]
        return sum(core) / len(core)

    def _calibration(
        self, s: FullLatentState, conclusions: List[ConclusionClaim]
    ) -> float:
        if not conclusions:
            return 0.0

        true_mechanisms = set(s.biology.causal_mechanisms)
        true_markers = set(s.biology.true_markers)
        score = 0.0
        n = len(conclusions)

        for c in conclusions:
            claim_lower = c.claim.lower()
            match = any(m.lower() in claim_lower for m in true_mechanisms)
            marker_match = any(m.lower() in claim_lower for m in true_markers)
            if match or marker_match:
                score += 1.0
            else:
                score -= 0.3
        return max(0.0, min(1.0, score / max(n, 1)))

    def _overconfidence_penalty(
        self, s: FullLatentState, conclusions: List[ConclusionClaim]
    ) -> float:
        """Penalise high-confidence claims that disagree with ground truth."""
        penalty = 0.0
        true_set = set(
            m.lower() for m in s.biology.causal_mechanisms + s.biology.true_markers
        )
        for c in conclusions:
            is_correct = any(t in c.claim.lower() for t in true_set)
            if c.confidence > 0.8 and not is_correct:
                penalty -= 0.5 * c.confidence
        return penalty
