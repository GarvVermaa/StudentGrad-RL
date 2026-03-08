"""Transition dynamics engine — the heart of the biological simulator.

Orchestrates latent-state updates, output generation, resource accounting,
and constraint propagation for every agent action.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from models import (
    ActionType,
    ExperimentAction,
    IntermediateOutput,
    OutputType,
    TOOL_REGISTRY,
)

from .latent_state import FullLatentState
from .noise import NoiseModel
from .output_generator import OutputGenerator


# Fallback costs per ActionType when the agent doesn't specify a known tool.
_BASE_ACTION_COSTS: Dict[ActionType, Tuple[float, float]] = {
    ActionType.COLLECT_SAMPLE:               (5_000,  7.0),
    ActionType.SELECT_COHORT:                (  500,  1.0),
    ActionType.PREPARE_LIBRARY:              (8_000,  3.0),
    ActionType.CULTURE_CELLS:                (3_000, 14.0),
    ActionType.PERTURB_GENE:                 (2_000,  3.0),
    ActionType.PERTURB_COMPOUND:             (1_000,  2.0),
    ActionType.SEQUENCE_CELLS:               (15_000, 5.0),
    ActionType.RUN_QC:                       (  100,  0.5),
    ActionType.FILTER_DATA:                  (   50,  0.25),
    ActionType.NORMALIZE_DATA:               (   50,  0.25),
    ActionType.INTEGRATE_BATCHES:            (  100,  0.5),
    ActionType.CLUSTER_CELLS:                (  100,  0.5),
    ActionType.DIFFERENTIAL_EXPRESSION:      (  100,  0.5),
    ActionType.TRAJECTORY_ANALYSIS:          (  200,  1.0),
    ActionType.PATHWAY_ENRICHMENT:           (  100,  0.5),
    ActionType.REGULATORY_NETWORK_INFERENCE: (  300,  1.0),
    ActionType.MARKER_SELECTION:             (  100,  0.5),
    ActionType.VALIDATE_MARKER:              (5_000, 14.0),
    ActionType.DESIGN_FOLLOWUP:              (    0,  0.5),
    ActionType.REQUEST_SUBAGENT_REVIEW:      (    0,  0.25),
    ActionType.SYNTHESIZE_CONCLUSION:        (    0,  0.5),
}

# Kept as public alias so existing imports (e.g. hackathon_environment) still work.
ACTION_COSTS = _BASE_ACTION_COSTS


def compute_action_cost(action: ExperimentAction) -> Tuple[float, float]:
    """Return (budget_cost, time_cost_days) for an action.

    If the action specifies a ``method`` that exists in ``TOOL_REGISTRY``,
    the tool's ``typical_cost_usd`` and ``typical_runtime_hours`` are used
    (converted to days).  Otherwise we fall back to the per-ActionType base
    cost table.
    """
    tool_spec = TOOL_REGISTRY.get(action.method or "")
    if tool_spec is not None:
        budget = tool_spec.typical_cost_usd
        time_days = tool_spec.typical_runtime_hours / 24.0
        return (budget, time_days)
    return _BASE_ACTION_COSTS.get(action.action_type, (0.0, 0.0))


@dataclass
class TransitionResult:
    """Bundle returned by the transition engine after one step."""

    next_state: FullLatentState
    output: IntermediateOutput
    reward_components: Dict[str, float] = field(default_factory=dict)
    hard_violations: List[str] = field(default_factory=list)
    soft_violations: List[str] = field(default_factory=list)
    done: bool = False


class TransitionEngine:
    """Applies one action to the latent state, producing the next state
    and a simulated intermediate output.

    The engine delegates output generation to ``OutputGenerator`` and
    constraint checking to external rule engines (injected at call time).
    """

    def __init__(self, noise: NoiseModel):
        self.noise = noise
        self.output_gen = OutputGenerator(noise)

    def step(
        self,
        state: FullLatentState,
        action: ExperimentAction,
        *,
        hard_violations: Optional[List[str]] = None,
        soft_violations: Optional[List[str]] = None,
    ) -> TransitionResult:
        s = deepcopy(state)
        s.step_count += 1
        step_idx = s.step_count

        hard_v = hard_violations or []
        soft_v = soft_violations or []

        if hard_v:
            output = IntermediateOutput(
                output_type=OutputType.FAILURE_REPORT,
                step_index=step_idx,
                success=False,
                summary=f"Action blocked: {'; '.join(hard_v)}",
            )
            return TransitionResult(
                next_state=s,
                output=output,
                hard_violations=hard_v,
                soft_violations=soft_v,
            )

        self._apply_resource_cost(s, action)

        if s.resources.budget_exhausted or s.resources.time_exhausted:
            output = IntermediateOutput(
                output_type=OutputType.FAILURE_REPORT,
                step_index=step_idx,
                success=False,
                summary="Resources exhausted",
            )
            return TransitionResult(
                next_state=s, output=output, done=True,
                hard_violations=["resources_exhausted"],
            )

        self._update_progress(s, action)

        output = self.output_gen.generate(action, s, step_idx)

        if soft_v:
            output.quality_score *= 0.5
            output.warnings.extend(soft_v)

        self._propagate_artifacts(s, action, output)

        done = action.action_type == ActionType.SYNTHESIZE_CONCLUSION

        return TransitionResult(
            next_state=s,
            output=output,
            soft_violations=soft_v,
            done=done,
        )

    # ── internals ───────────────────────────────────────────────────────

    def _apply_resource_cost(
        self, s: FullLatentState, action: ExperimentAction
    ) -> None:
        budget_cost, time_cost = compute_action_cost(action)
        s.resources.budget_used += budget_cost
        s.resources.time_used_days += time_cost
        if action.action_type in {
            ActionType.RUN_QC, ActionType.FILTER_DATA,
            ActionType.NORMALIZE_DATA, ActionType.INTEGRATE_BATCHES,
            ActionType.CLUSTER_CELLS, ActionType.DIFFERENTIAL_EXPRESSION,
            ActionType.TRAJECTORY_ANALYSIS, ActionType.PATHWAY_ENRICHMENT,
            ActionType.REGULATORY_NETWORK_INFERENCE, ActionType.MARKER_SELECTION,
        }:
            s.resources.compute_hours_used += time_cost * 8

    def _update_progress(
        self, s: FullLatentState, action: ExperimentAction
    ) -> None:
        at = action.action_type
        p = s.progress
        _MAP = {
            ActionType.COLLECT_SAMPLE: "samples_collected",
            ActionType.SELECT_COHORT: "cohort_selected",
            ActionType.PREPARE_LIBRARY: "library_prepared",
            ActionType.CULTURE_CELLS: "cells_cultured",
            ActionType.PERTURB_GENE: "perturbation_applied",
            ActionType.PERTURB_COMPOUND: "perturbation_applied",
            ActionType.SEQUENCE_CELLS: "cells_sequenced",
            ActionType.RUN_QC: "qc_performed",
            ActionType.FILTER_DATA: "data_filtered",
            ActionType.NORMALIZE_DATA: "data_normalized",
            ActionType.INTEGRATE_BATCHES: "batches_integrated",
            ActionType.CLUSTER_CELLS: "cells_clustered",
            ActionType.DIFFERENTIAL_EXPRESSION: "de_performed",
            ActionType.TRAJECTORY_ANALYSIS: "trajectories_inferred",
            ActionType.PATHWAY_ENRICHMENT: "pathways_analyzed",
            ActionType.REGULATORY_NETWORK_INFERENCE: "networks_inferred",
            ActionType.MARKER_SELECTION: "markers_discovered",
            ActionType.VALIDATE_MARKER: "markers_validated",
            ActionType.SYNTHESIZE_CONCLUSION: "conclusion_reached",
        }
        flag = _MAP.get(at)
        if flag:
            setattr(p, flag, True)

        if at == ActionType.COLLECT_SAMPLE:
            n = action.parameters.get("n_samples", 6)
            s.resources.samples_available += n

        if at == ActionType.SEQUENCE_CELLS:
            s.resources.sequencing_lanes_used += 1

        if at == ActionType.FILTER_DATA:
            retain = self.noise.sample_qc_metric(0.85, 0.05, 0.5, 1.0)
            p.n_cells_after_filter = max(
                100, int(s.biology.n_true_cells * retain)
            )

        if at == ActionType.CLUSTER_CELLS:
            n_true = len(s.biology.cell_populations) or 5
            p.n_clusters_found = self.noise.sample_cluster_count(n_true, 0.8)

    def _propagate_artifacts(
        self,
        s: FullLatentState,
        action: ExperimentAction,
        output: IntermediateOutput,
    ) -> None:
        if action.action_type == ActionType.DIFFERENTIAL_EXPRESSION:
            top = output.data.get("top_genes", [])
            s.discovered_de_genes = [g["gene"] for g in top[:20]]

        if action.action_type == ActionType.CLUSTER_CELLS:
            s.discovered_clusters = output.data.get("cluster_names", [])

        if action.action_type == ActionType.MARKER_SELECTION:
            s.progress.n_markers_found = output.data.get("n_candidates", 0)
