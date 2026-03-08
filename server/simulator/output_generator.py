"""Generate simulated intermediate outputs conditioned on latent state."""

from __future__ import annotations

from typing import Any, Dict, List

from models import (
    ActionType,
    ExperimentAction,
    IntermediateOutput,
    OutputType,
)

from .latent_state import FullLatentState
from .noise import NoiseModel


class OutputGenerator:
    """Creates structured ``IntermediateOutput`` objects conditioned on the
    hidden latent state, the action taken, and a stochastic noise model.
    """

    def __init__(self, noise: NoiseModel):
        self.noise = noise

    def generate(
        self,
        action: ExperimentAction,
        state: FullLatentState,
        step_index: int,
    ) -> IntermediateOutput:
        handler = _HANDLERS.get(action.action_type, self._default)
        return handler(self, action, state, step_index)

    # ── wet-lab outputs ─────────────────────────────────────────────────

    def _collect_sample(
        self, action: ExperimentAction, s: FullLatentState, idx: int
    ) -> IntermediateOutput:
        n_samples = action.parameters.get("n_samples", 6)
        quality = self.noise.quality_degradation(
            s.technical.sample_quality, [s.technical.capture_efficiency]
        )
        return IntermediateOutput(
            output_type=OutputType.SAMPLE_COLLECTION_RESULT,
            step_index=idx,
            quality_score=quality,
            summary=f"Collected {n_samples} samples (quality={quality:.2f})",
            data={
                "n_samples": n_samples,
                "quality": quality,
                "organism": "human",
                "tissue": "blood",
            },
            artifacts_available=["raw_samples"],
        )

    def _select_cohort(
        self, action: ExperimentAction, s: FullLatentState, idx: int
    ) -> IntermediateOutput:
        criteria = action.parameters.get("criteria", {})
        n_selected = action.parameters.get("n_selected", 4)
        return IntermediateOutput(
            output_type=OutputType.COHORT_RESULT,
            step_index=idx,
            summary=f"Selected cohort of {n_selected} samples with criteria {criteria}",
            data={"n_selected": n_selected, "criteria": criteria},
            artifacts_available=["cohort_manifest"],
        )

    def _prepare_library(
        self, action: ExperimentAction, s: FullLatentState, idx: int
    ) -> IntermediateOutput:
        complexity = self.noise.quality_degradation(
            s.technical.library_complexity,
            [s.technical.sample_quality],
        )
        return IntermediateOutput(
            output_type=OutputType.LIBRARY_PREP_RESULT,
            step_index=idx,
            quality_score=complexity,
            summary=f"Library prepared (complexity={complexity:.2f})",
            data={
                "library_complexity": complexity,
                "method": action.method or "10x_chromium",
            },
            artifacts_available=["prepared_library"],
        )

    def _culture_cells(
        self, action: ExperimentAction, s: FullLatentState, idx: int
    ) -> IntermediateOutput:
        days = action.parameters.get("days", 7)
        viability = self.noise.sample_qc_metric(0.92, 0.05, 0.5, 1.0)
        return IntermediateOutput(
            output_type=OutputType.CULTURE_RESULT,
            step_index=idx,
            quality_score=viability,
            summary=f"Cultured for {days}d, viability={viability:.2f}",
            data={"days": days, "viability": viability},
            artifacts_available=["cultured_cells"],
        )

    def _perturb(
        self, action: ExperimentAction, s: FullLatentState, idx: int
    ) -> IntermediateOutput:
        target = action.parameters.get("target", "unknown")
        efficiency = self.noise.sample_qc_metric(0.75, 0.15, 0.0, 1.0)
        return IntermediateOutput(
            output_type=OutputType.PERTURBATION_RESULT,
            step_index=idx,
            quality_score=efficiency,
            summary=f"Perturbation of {target} (efficiency={efficiency:.2f})",
            data={
                "target": target,
                "efficiency": efficiency,
                "type": action.action_type.value,
            },
            artifacts_available=["perturbed_cells"],
        )

    def _sequence_cells(
        self, action: ExperimentAction, s: FullLatentState, idx: int
    ) -> IntermediateOutput:
        depth = s.technical.sequencing_depth_factor
        n_cells = self.noise.sample_count(
            s.biology.n_true_cells * s.technical.capture_efficiency
        )
        n_genes = self.noise.sample_count(18_000)
        median_umi = self.noise.sample_count(int(3000 * depth))
        quality = self.noise.quality_degradation(
            s.technical.sample_quality,
            [s.technical.library_complexity, s.technical.capture_efficiency],
        )
        return IntermediateOutput(
            output_type=OutputType.SEQUENCING_RESULT,
            step_index=idx,
            quality_score=quality,
            summary=(
                f"Sequenced {n_cells} cells, {n_genes} genes detected, "
                f"median UMI={median_umi}"
            ),
            data={
                "n_cells": n_cells,
                "n_genes": n_genes,
                "median_umi": median_umi,
                "sequencing_saturation": self.noise.sample_qc_metric(0.7, 0.1),
            },
            artifacts_available=["raw_count_matrix"],
        )

    # ── computational outputs ───────────────────────────────────────────

    def _run_qc(
        self, action: ExperimentAction, s: FullLatentState, idx: int
    ) -> IntermediateOutput:
        doublet_frac = self.noise.sample_qc_metric(
            s.technical.doublet_rate, 0.01, 0.0, 0.2
        )
        mito_frac = self.noise.sample_qc_metric(0.05, 0.02, 0.0, 0.3)
        ambient_frac = self.noise.sample_qc_metric(
            s.technical.ambient_rna_fraction, 0.01, 0.0, 0.2
        )
        warnings: List[str] = []
        if doublet_frac > 0.08:
            warnings.append(f"High doublet rate ({doublet_frac:.1%})")
        if mito_frac > 0.1:
            warnings.append(f"High mitochondrial fraction ({mito_frac:.1%})")
        quality = 1.0 - (doublet_frac + mito_frac + ambient_frac)
        return IntermediateOutput(
            output_type=OutputType.QC_METRICS,
            step_index=idx,
            quality_score=max(0.0, quality),
            summary="QC metrics computed",
            data={
                "doublet_fraction": doublet_frac,
                "mitochondrial_fraction": mito_frac,
                "ambient_rna_fraction": ambient_frac,
                "median_genes_per_cell": self.noise.sample_count(2500),
                "median_umi_per_cell": self.noise.sample_count(8000),
            },
            warnings=warnings,
            artifacts_available=["qc_report"],
        )

    def _filter_data(
        self, action: ExperimentAction, s: FullLatentState, idx: int
    ) -> IntermediateOutput:
        retain_frac = self.noise.sample_qc_metric(0.85, 0.05, 0.5, 1.0)
        n_before = s.biology.n_true_cells
        n_after = max(100, int(n_before * retain_frac))
        return IntermediateOutput(
            output_type=OutputType.COUNT_MATRIX_SUMMARY,
            step_index=idx,
            quality_score=retain_frac,
            summary=f"Filtered {n_before} → {n_after} cells ({retain_frac:.0%} retained)",
            data={
                "n_cells_before": n_before,
                "n_cells_after": n_after,
                "n_genes_retained": self.noise.sample_count(15_000),
                "retain_fraction": retain_frac,
            },
            artifacts_available=["filtered_count_matrix"],
        )

    def _normalize_data(
        self, action: ExperimentAction, s: FullLatentState, idx: int
    ) -> IntermediateOutput:
        method = action.method or "log_normalize"
        return IntermediateOutput(
            output_type=OutputType.COUNT_MATRIX_SUMMARY,
            step_index=idx,
            summary=f"Normalized with {method}",
            data={"method": method, "n_hvg": self.noise.sample_count(2000)},
            artifacts_available=["normalized_matrix", "hvg_list"],
        )

    def _integrate_batches(
        self, action: ExperimentAction, s: FullLatentState, idx: int
    ) -> IntermediateOutput:
        method = action.method or "harmony"
        residual = self.noise.sample_qc_metric(0.05, 0.03, 0.0, 0.3)
        return IntermediateOutput(
            output_type=OutputType.EMBEDDING_SUMMARY,
            step_index=idx,
            quality_score=1.0 - residual,
            summary=f"Batch integration ({method}), residual batch effect={residual:.2f}",
            data={
                "method": method,
                "residual_batch_effect": residual,
                "n_batches": len(s.technical.batch_effects) or 1,
            },
            artifacts_available=["integrated_embedding"],
        )

    def _cluster_cells(
        self, action: ExperimentAction, s: FullLatentState, idx: int
    ) -> IntermediateOutput:
        n_true = len(s.biology.cell_populations) or 5
        quality = self.noise.quality_degradation(0.8, [0.95])
        n_clusters = self.noise.sample_cluster_count(n_true, quality)
        cluster_names = [f"cluster_{i}" for i in range(n_clusters)]
        sizes = self._random_partition(s.biology.n_true_cells, n_clusters)
        return IntermediateOutput(
            output_type=OutputType.CLUSTER_RESULT,
            step_index=idx,
            quality_score=quality,
            summary=f"Found {n_clusters} clusters (ground-truth populations: {n_true})",
            data={
                "n_clusters": n_clusters,
                "cluster_names": cluster_names,
                "cluster_sizes": sizes,
                "silhouette_score": self.noise.sample_qc_metric(0.35, 0.1, -1.0, 1.0),
            },
            uncertainty=abs(n_clusters - n_true) / max(n_true, 1),
            artifacts_available=["cluster_assignments", "umap_embedding"],
        )

    def _differential_expression(
        self, action: ExperimentAction, s: FullLatentState, idx: int
    ) -> IntermediateOutput:
        comparison = action.parameters.get("comparison", "disease_vs_healthy")
        true_effects = s.biology.true_de_genes.get(comparison, {})

        n_cells = s.progress.n_cells_after_filter or s.biology.n_true_cells
        noise_level = s.technical.dropout_rate + 0.1 * (1.0 - s.technical.sample_quality)
        observed = self.noise.sample_effect_sizes(true_effects, n_cells, noise_level)

        fp_genes = self.noise.generate_false_positives(5000, 0.002 + noise_level * 0.01)
        for g in fp_genes:
            observed[g] = float(self.noise.rng.normal(0, 0.3))

        fn_genes = self.noise.generate_false_negatives(list(true_effects.keys()), 0.15)
        for g in fn_genes:
            observed.pop(g, None)

        top_genes = sorted(observed.items(), key=lambda kv: abs(kv[1]), reverse=True)[:50]
        return IntermediateOutput(
            output_type=OutputType.DE_RESULT,
            step_index=idx,
            quality_score=self.noise.quality_degradation(0.8, [1.0 - noise_level]),
            summary=f"DE analysis ({comparison}): {len(observed)} genes tested, {len(top_genes)} top hits",
            data={
                "comparison": comparison,
                "n_tested": len(observed),
                "top_genes": [
                    {"gene": g, "log2FC": round(fc, 3)} for g, fc in top_genes
                ],
                "n_significant": sum(1 for _, fc in observed.items() if abs(fc) > 0.5),
            },
            uncertainty=noise_level,
            artifacts_available=["de_table"],
        )

    def _trajectory_analysis(
        self, action: ExperimentAction, s: FullLatentState, idx: int
    ) -> IntermediateOutput:
        has_trajectory = s.biology.true_trajectory is not None
        quality = self.noise.quality_degradation(0.7 if has_trajectory else 0.3, [0.9])
        summary_data: Dict[str, Any] = {"method": action.method or "monocle3"}
        if has_trajectory:
            summary_data.update({
                "n_lineages": s.biology.true_trajectory.get("n_lineages", 1),
                "pseudotime_range": [0.0, 1.0],
                "branching_detected": s.biology.true_trajectory.get("branching", False),
            })
        else:
            summary_data["n_lineages"] = self.noise.sample_count(1) + 1
            summary_data["pseudotime_range"] = [0.0, 1.0]
            summary_data["branching_detected"] = self.noise.coin_flip(0.3)

        return IntermediateOutput(
            output_type=OutputType.TRAJECTORY_RESULT,
            step_index=idx,
            quality_score=quality,
            summary="Trajectory / pseudotime analysis complete",
            data=summary_data,
            uncertainty=0.2 if has_trajectory else 0.6,
            artifacts_available=["pseudotime_values", "lineage_graph"],
        )

    def _pathway_enrichment(
        self, action: ExperimentAction, s: FullLatentState, idx: int
    ) -> IntermediateOutput:
        true_pathways = s.biology.true_pathways
        noise_level = 0.15
        observed: Dict[str, float] = {}
        for pw, activity in true_pathways.items():
            observed[pw] = activity + float(self.noise.rng.normal(0, noise_level))

        for i in range(self.noise.sample_count(2)):
            observed[f"FP_PATHWAY_{i}"] = float(self.noise.rng.uniform(0.3, 0.6))

        top = sorted(observed.items(), key=lambda kv: kv[1], reverse=True)[:15]
        return IntermediateOutput(
            output_type=OutputType.PATHWAY_RESULT,
            step_index=idx,
            quality_score=self.noise.quality_degradation(0.8, [0.95]),
            summary=f"Pathway enrichment: {len(top)} significant pathways",
            data={
                "method": action.method or "GSEA",
                "top_pathways": [
                    {"pathway": p, "score": round(s, 3)} for p, s in top
                ],
            },
            uncertainty=noise_level,
            artifacts_available=["enrichment_table"],
        )

    def _regulatory_network(
        self, action: ExperimentAction, s: FullLatentState, idx: int
    ) -> IntermediateOutput:
        true_net = s.biology.true_regulatory_network
        n_edges_true = sum(len(v) for v in true_net.values())
        noise_edges = self.noise.sample_count(max(5, int(n_edges_true * 0.3)))
        return IntermediateOutput(
            output_type=OutputType.NETWORK_RESULT,
            step_index=idx,
            quality_score=self.noise.quality_degradation(0.6, [0.9]),
            summary=f"Regulatory network inferred: {n_edges_true + noise_edges} edges",
            data={
                "method": action.method or "SCENIC",
                "n_regulons": len(true_net) + self.noise.sample_count(3),
                "n_edges": n_edges_true + noise_edges,
                "top_regulators": list(true_net.keys())[:10],
            },
            uncertainty=0.35,
            artifacts_available=["regulon_table", "grn_adjacency"],
        )

    def _marker_selection(
        self, action: ExperimentAction, s: FullLatentState, idx: int
    ) -> IntermediateOutput:
        true_markers = list(s.biology.true_markers)
        noise_level = 0.2
        observed_markers = [
            m for m in true_markers if not self.noise.coin_flip(noise_level)
        ]
        fp = self.noise.generate_false_positives(200, 0.01)
        observed_markers.extend(fp)
        return IntermediateOutput(
            output_type=OutputType.MARKER_RESULT,
            step_index=idx,
            quality_score=self.noise.quality_degradation(0.75, [0.9]),
            summary=f"Selected {len(observed_markers)} candidate markers",
            data={
                "markers": observed_markers[:20],
                "n_candidates": len(observed_markers),
            },
            uncertainty=noise_level,
            artifacts_available=["marker_list"],
        )

    def _validate_marker(
        self, action: ExperimentAction, s: FullLatentState, idx: int
    ) -> IntermediateOutput:
        marker = action.parameters.get("marker", "unknown")
        is_true = marker in s.biology.true_markers
        validation_correct = not self.noise.coin_flip(0.1)
        validated = is_true == validation_correct
        return IntermediateOutput(
            output_type=OutputType.VALIDATION_RESULT,
            step_index=idx,
            quality_score=0.9 if validation_correct else 0.4,
            summary=f"Marker {marker}: {'validated' if validated else 'not validated'}",
            data={
                "marker": marker,
                "validated": validated,
                "assay": action.method or "qPCR",
                "effect_size": self.noise.sample_qc_metric(
                    1.5 if is_true else 0.2, 0.3, -0.5, 5.0
                ),
            },
            artifacts_available=["validation_data"],
        )

    def _design_followup(
        self, action: ExperimentAction, s: FullLatentState, idx: int
    ) -> IntermediateOutput:
        return IntermediateOutput(
            output_type=OutputType.FOLLOWUP_DESIGN,
            step_index=idx,
            summary="Follow-up experiment design proposed",
            data={"proposal": action.parameters},
            artifacts_available=["followup_proposal"],
        )

    def _subagent_review(
        self, action: ExperimentAction, s: FullLatentState, idx: int
    ) -> IntermediateOutput:
        return IntermediateOutput(
            output_type=OutputType.SUBAGENT_REPORT,
            step_index=idx,
            summary=f"Subagent review ({action.invoked_subagent or 'general'})",
            data={"subagent": action.invoked_subagent, "notes": "Review complete."},
            artifacts_available=["subagent_report"],
        )

    def _synthesize_conclusion(
        self, action: ExperimentAction, s: FullLatentState, idx: int
    ) -> IntermediateOutput:
        return IntermediateOutput(
            output_type=OutputType.CONCLUSION,
            step_index=idx,
            summary="Conclusion synthesised from pipeline evidence",
            data={"claims": action.parameters.get("claims", [])},
            artifacts_available=["conclusion_report"],
        )

    def _default(
        self, action: ExperimentAction, s: FullLatentState, idx: int
    ) -> IntermediateOutput:
        return IntermediateOutput(
            output_type=OutputType.FAILURE_REPORT,
            step_index=idx,
            success=False,
            summary=f"Unhandled action type: {action.action_type}",
            data={},
        )

    # ── helpers ─────────────────────────────────────────────────────────

    def _random_partition(self, total: int, k: int) -> List[int]:
        if k <= 0:
            return []
        fracs = self.noise.rng.dirichlet(alpha=[1.0] * k)
        sizes = [max(1, int(total * f)) for f in fracs]
        diff = total - sum(sizes)
        sizes[0] += diff
        return sizes


_HANDLERS = {
    ActionType.COLLECT_SAMPLE: OutputGenerator._collect_sample,
    ActionType.SELECT_COHORT: OutputGenerator._select_cohort,
    ActionType.PREPARE_LIBRARY: OutputGenerator._prepare_library,
    ActionType.CULTURE_CELLS: OutputGenerator._culture_cells,
    ActionType.PERTURB_GENE: OutputGenerator._perturb,
    ActionType.PERTURB_COMPOUND: OutputGenerator._perturb,
    ActionType.SEQUENCE_CELLS: OutputGenerator._sequence_cells,
    ActionType.RUN_QC: OutputGenerator._run_qc,
    ActionType.FILTER_DATA: OutputGenerator._filter_data,
    ActionType.NORMALIZE_DATA: OutputGenerator._normalize_data,
    ActionType.INTEGRATE_BATCHES: OutputGenerator._integrate_batches,
    ActionType.CLUSTER_CELLS: OutputGenerator._cluster_cells,
    ActionType.DIFFERENTIAL_EXPRESSION: OutputGenerator._differential_expression,
    ActionType.TRAJECTORY_ANALYSIS: OutputGenerator._trajectory_analysis,
    ActionType.PATHWAY_ENRICHMENT: OutputGenerator._pathway_enrichment,
    ActionType.REGULATORY_NETWORK_INFERENCE: OutputGenerator._regulatory_network,
    ActionType.MARKER_SELECTION: OutputGenerator._marker_selection,
    ActionType.VALIDATE_MARKER: OutputGenerator._validate_marker,
    ActionType.DESIGN_FOLLOWUP: OutputGenerator._design_followup,
    ActionType.REQUEST_SUBAGENT_REVIEW: OutputGenerator._subagent_review,
    ActionType.SYNTHESIZE_CONCLUSION: OutputGenerator._synthesize_conclusion,
}
