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
    TOOL_REGISTRY,
    WET_LAB_ACTIONS,
)

from server.biology.gene_index import (
    marker_set_score,
    mechanism_set_score,
    score_pathways,
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

        ordering_score = self._ordering_score(action, prev_state)
        rb.ordering = 0.2 * ordering_score
        if ordering_score < 0:
            rb.penalty += ordering_score * 0.3

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

        # tool-modality fit bonus/penalty
        tool_fit = self._tool_fit_score(action, prev_state)
        rb.components["tool_fit"] = tool_fit
        rb.validity += 0.15 * tool_fit

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
        """Heuristic: 1.0 if natural next, 0.3 if acceptable, -1.0 if premature."""
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
        if NATURAL_NEXT.get(at, False):
            return 1.0

        has_evidence = any([
            p.cells_clustered, p.de_performed, p.trajectories_inferred,
            p.pathways_analyzed, p.networks_inferred, p.markers_discovered,
        ])
        if at in META_ACTIONS and not has_evidence:
            return -1.0

        return 0.3

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
        """Structured set-similarity calibration against hidden ground truth.

        Uses pathway-weighted Gaussian similarity for markers, semantic
        similarity for mechanisms, and activity-weighted matching for pathways.
        Falls back to legacy substring matching when structured fields are empty.
        """
        if not conclusions:
            return 0.0

        pred_markers = [g for c in conclusions for g in c.top_markers]
        pred_mechs = [m for c in conclusions for m in c.causal_mechanisms]
        pred_pathways = {
            p: v for c in conclusions for p, v in c.predicted_pathways.items()
        }

        has_structured = bool(pred_markers or pred_mechs or pred_pathways)

        if has_structured:
            m_score = marker_set_score(pred_markers, s.biology.true_markers)
            mech_score = mechanism_set_score(
                pred_mechs, s.biology.causal_mechanisms
            )
            pw_score = score_pathways(pred_pathways, s.biology.true_pathways)
            return 0.50 * m_score + 0.35 * mech_score + 0.15 * pw_score

        return self._legacy_calibration(s, conclusions)

    @staticmethod
    def _legacy_calibration(
        s: FullLatentState, conclusions: List[ConclusionClaim]
    ) -> float:
        """Substring-based calibration kept for backward compatibility."""
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

    _METHOD_TO_TOOL: Dict[str, str] = {
        "scanpy.pp.calculate_qc_metrics": "Scanpy",
        "scanpy.pp.filter_cells": "Scanpy",
        "scanpy.pp.filter_genes": "Scanpy",
        "scanpy.pp.normalize_total": "Scanpy",
        "scanpy.pp.log1p": "Scanpy",
        "scanpy.pp.highly_variable_genes": "Scanpy",
        "scanpy.pp.neighbors": "Scanpy",
        "scanpy.tl.leiden": "Leiden",
        "scanpy.tl.louvain": "Louvain",
        "scanpy.tl.rank_genes_groups": "Scanpy",
        "scanpy.tl.paga": "PAGA",
        "scanpy.tl.umap": "UMAP",
        "gseapy.prerank": "Scanpy",
        "gseapy.gsea": "Scanpy",
        "10x_chromium": "CellRanger",
        "NovaSeq": "CellRanger",
    }

    @staticmethod
    def _tool_fit_score(
        action: ExperimentAction, s: FullLatentState
    ) -> float:
        """Score how well the chosen tool matches the task modality.

        Returns +1.0 for a perfect match, 0.0 if no tool specified,
        -1.0 for a known tool used on an incompatible modality.
        """
        method = action.method
        if not method:
            return 0.0
        resolved = RewardComputer._METHOD_TO_TOOL.get(method, method)
        tool_spec = TOOL_REGISTRY.get(resolved)
        if tool_spec is None:
            return -0.5
        modality = getattr(s, "task_modality", None)
        if not modality or not tool_spec.modalities:
            return 0.0
        if modality in tool_spec.modalities:
            return 1.0
        return -1.0

    def _overconfidence_penalty(
        self, s: FullLatentState, conclusions: List[ConclusionClaim]
    ) -> float:
        """Penalise high-confidence claims that disagree with ground truth.

        Checks structured fields (top_markers, causal_mechanisms) first;
        falls back to claim substring matching for backward compatibility.
        """
        penalty = 0.0
        true_markers_lower = {m.lower() for m in s.biology.true_markers}
        true_mechs_lower = {m.lower() for m in s.biology.causal_mechanisms}
        true_set = true_markers_lower | true_mechs_lower

        for c in conclusions:
            if c.confidence <= 0.8:
                continue

            has_structured = bool(c.top_markers or c.causal_mechanisms)
            if has_structured:
                marker_hit = any(
                    g.upper().strip() in {m.upper() for m in s.biology.true_markers}
                    for g in c.top_markers
                )
                mech_hit = any(
                    any(kw in m.lower() for kw in t.lower().split())
                    for m in c.causal_mechanisms
                    for t in s.biology.causal_mechanisms
                )
                is_correct = marker_hit or mech_hit
            else:
                is_correct = any(t in c.claim.lower() for t in true_set)

            if not is_correct:
                penalty -= 0.5 * c.confidence

        return penalty
