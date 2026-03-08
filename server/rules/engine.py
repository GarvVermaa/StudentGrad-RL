"""Biological rule engine — hard and soft constraint checking.

Hard constraints block action execution entirely.
Soft constraints allow execution but degrade output quality and incur penalties.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List

from models import ActionType, ExperimentAction, TOOL_REGISTRY

from server.simulator.latent_state import FullLatentState


class Severity(str, Enum):
    HARD = "hard"
    SOFT = "soft"


@dataclass
class RuleViolation:
    rule_id: str
    severity: Severity
    message: str


class RuleEngine:
    """Evaluates biological and resource constraints against the current
    latent state before each action is applied.
    """

    def check(
        self, action: ExperimentAction, state: FullLatentState
    ) -> List[RuleViolation]:
        violations: List[RuleViolation] = []
        violations.extend(self._check_prerequisites(action, state))
        violations.extend(self._check_resource_constraints(action, state))
        violations.extend(self._check_redundancy(action, state))
        violations.extend(self._check_causal_validity(action, state))
        violations.extend(self._check_tool_compatibility(action, state))
        return violations

    def hard_violations(self, violations: List[RuleViolation]) -> List[str]:
        return [v.message for v in violations if v.severity == Severity.HARD]

    def soft_violations(self, violations: List[RuleViolation]) -> List[str]:
        return [v.message for v in violations if v.severity == Severity.SOFT]

    # ── prerequisite rules ──────────────────────────────────────────────

    def _check_prerequisites(
        self, action: ExperimentAction, s: FullLatentState
    ) -> List[RuleViolation]:
        vs: List[RuleViolation] = []
        at = action.action_type
        p = s.progress

        REQUIRES = {
            ActionType.PREPARE_LIBRARY: [
                ("samples_collected", "Cannot prepare library without collected samples"),
            ],
            ActionType.SEQUENCE_CELLS: [
                ("library_prepared", "Cannot sequence without library preparation"),
            ],
            ActionType.RUN_QC: [
                ("cells_sequenced", "Cannot run QC before sequencing"),
            ],
            ActionType.FILTER_DATA: [
                ("qc_performed", "Cannot filter data before QC"),
            ],
            ActionType.NORMALIZE_DATA: [
                ("data_filtered", "Cannot normalise before filtering"),
            ],
            ActionType.INTEGRATE_BATCHES: [
                ("data_normalized", "Cannot integrate batches before normalisation"),
            ],
            ActionType.CLUSTER_CELLS: [
                ("data_normalized", "Cannot cluster before normalisation"),
            ],
            ActionType.DIFFERENTIAL_EXPRESSION: [
                ("data_normalized", "Cannot run DE before normalisation"),
            ],
            ActionType.TRAJECTORY_ANALYSIS: [
                ("data_normalized", "Cannot infer trajectories before normalisation"),
            ],
            ActionType.PATHWAY_ENRICHMENT: [
                ("de_performed", "Cannot run pathway enrichment without DE results"),
            ],
            ActionType.REGULATORY_NETWORK_INFERENCE: [
                ("data_normalized", "Cannot infer networks before normalisation"),
            ],
            ActionType.MARKER_SELECTION: [
                ("de_performed", "Cannot select markers without DE results"),
            ],
            ActionType.VALIDATE_MARKER: [
                ("markers_discovered", "Cannot validate markers before discovery"),
            ],
            ActionType.PERTURB_GENE: [
                ("samples_collected", "Cannot perturb without samples"),
            ],
            ActionType.PERTURB_COMPOUND: [
                ("samples_collected", "Cannot perturb without samples"),
            ],
            ActionType.CULTURE_CELLS: [
                ("samples_collected", "Cannot culture without samples"),
            ],
        }

        for flag, msg in REQUIRES.get(at, []):
            if not getattr(p, flag, False):
                vs.append(RuleViolation(
                    rule_id=f"prereq_{at.value}_{flag}",
                    severity=Severity.HARD,
                    message=msg,
                ))
        return vs

    # ── resource constraints ────────────────────────────────────────────

    def _check_resource_constraints(
        self, action: ExperimentAction, s: FullLatentState
    ) -> List[RuleViolation]:
        vs: List[RuleViolation] = []
        if s.resources.budget_exhausted:
            vs.append(RuleViolation(
                rule_id="budget_exhausted",
                severity=Severity.HARD,
                message="Budget exhausted — no further actions possible",
            ))
        if s.resources.time_exhausted:
            vs.append(RuleViolation(
                rule_id="time_exhausted",
                severity=Severity.HARD,
                message="Time limit reached — no further actions possible",
            ))

        remaining = s.resources.budget_remaining
        from server.simulator.transition import ACTION_COSTS
        cost, _ = ACTION_COSTS.get(action.action_type, (0, 0))
        if cost > remaining and remaining > 0:
            vs.append(RuleViolation(
                rule_id="budget_insufficient",
                severity=Severity.SOFT,
                message=f"Action costs ${cost:,.0f} but only ${remaining:,.0f} remains",
            ))
        return vs

    # ── redundancy checks ───────────────────────────────────────────────

    def _check_redundancy(
        self, action: ExperimentAction, s: FullLatentState
    ) -> List[RuleViolation]:
        vs: List[RuleViolation] = []
        at = action.action_type
        p = s.progress

        REDUNDANT = {
            ActionType.COLLECT_SAMPLE: "samples_collected",
            ActionType.PREPARE_LIBRARY: "library_prepared",
            ActionType.SEQUENCE_CELLS: "cells_sequenced",
            ActionType.RUN_QC: "qc_performed",
            ActionType.FILTER_DATA: "data_filtered",
            ActionType.NORMALIZE_DATA: "data_normalized",
        }
        flag = REDUNDANT.get(at)
        if flag and getattr(p, flag, False):
            vs.append(RuleViolation(
                rule_id=f"redundant_{at.value}",
                severity=Severity.SOFT,
                message=f"Step '{at.value}' already completed — redundant action",
            ))
        return vs

    # ── causal validity ─────────────────────────────────────────────────

    def _check_causal_validity(
        self, action: ExperimentAction, s: FullLatentState
    ) -> List[RuleViolation]:
        vs: List[RuleViolation] = []
        if action.action_type == ActionType.SYNTHESIZE_CONCLUSION:
            if not s.progress.de_performed and not s.progress.cells_clustered:
                vs.append(RuleViolation(
                    rule_id="premature_conclusion",
                    severity=Severity.SOFT,
                    message="Synthesising conclusion without substantive analysis",
                ))

            claims = action.parameters.get("claims", [])
            for claim in claims:
                if isinstance(claim, dict) and claim.get("claim_type") == "causal":
                    if not s.progress.markers_validated and not s.progress.networks_inferred:
                        vs.append(RuleViolation(
                            rule_id="unsupported_causal_claim",
                            severity=Severity.SOFT,
                            message="Causal claim without validation or network evidence",
                        ))
                        break

        if action.action_type == ActionType.PATHWAY_ENRICHMENT:
            if not s.progress.de_performed:
                vs.append(RuleViolation(
                    rule_id="pathway_without_de",
                    severity=Severity.SOFT,
                    message="Pathway enrichment without DE may yield unreliable results",
                ))
        return vs

    # ── tool / modality compatibility ────────────────────────────────────

    def _check_tool_compatibility(
        self, action: ExperimentAction, s: FullLatentState
    ) -> List[RuleViolation]:
        """Warn when the chosen tool is incompatible with the task modality."""
        vs: List[RuleViolation] = []
        method = action.method
        if not method:
            return vs

        tool_spec = TOOL_REGISTRY.get(method)
        if tool_spec is None:
            vs.append(RuleViolation(
                rule_id="unknown_tool",
                severity=Severity.SOFT,
                message=f"Tool '{method}' is not in the registry — results may be unreliable",
            ))
            return vs

        # Check modality compatibility (modality lives on the task, which is
        # stored in the latent state's associated TaskSpec — but the latent
        # state doesn't carry the TaskSpec directly.  We can still check via
        # the action's own context or fall back gracefully).
        task_modality = getattr(s, "task_modality", None)
        if task_modality and tool_spec.modalities:
            if task_modality not in tool_spec.modalities:
                vs.append(RuleViolation(
                    rule_id="tool_modality_mismatch",
                    severity=Severity.SOFT,
                    message=(
                        f"Tool '{method}' is designed for "
                        f"{', '.join(tool_spec.modalities)} but task modality "
                        f"is '{task_modality}'"
                    ),
                ))

        return vs
