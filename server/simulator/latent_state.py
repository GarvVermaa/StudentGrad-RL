"""Latent biological and technical state — hidden from the agent."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class CellPopulation(BaseModel):
    """Ground-truth cell sub-population in the simulated tissue."""

    name: str
    proportion: float = Field(ge=0.0, le=1.0)
    marker_genes: List[str] = Field(default_factory=list)
    state: str = "quiescent"
    condition_response: Dict[str, float] = Field(default_factory=dict)


class GeneProgram(BaseModel):
    """A latent gene-regulatory programme."""

    name: str
    genes: List[str] = Field(default_factory=list)
    activity_level: float = Field(0.5, ge=0.0, le=1.0)
    condition_dependent: bool = False
    conditions_active: List[str] = Field(default_factory=list)


class LatentBiologicalState(BaseModel):
    """Hidden ground-truth biology the agent cannot directly observe."""

    cell_populations: List[CellPopulation] = Field(default_factory=list)
    true_de_genes: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="comparison_key → {gene: log2FC}",
    )
    true_pathways: Dict[str, float] = Field(
        default_factory=dict,
        description="pathway → activity level",
    )
    gene_programs: List[GeneProgram] = Field(default_factory=list)
    true_trajectory: Optional[Dict[str, Any]] = None
    true_regulatory_network: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="TF → target genes",
    )
    perturbation_effects: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="perturbation → {gene: effect_size}",
    )
    confounders: Dict[str, float] = Field(default_factory=dict)
    true_markers: List[str] = Field(default_factory=list)
    causal_mechanisms: List[str] = Field(default_factory=list)
    n_true_cells: int = 10_000


class TechnicalState(BaseModel):
    """Hidden technical parameters that shape experimental noise."""

    batch_effects: Dict[str, float] = Field(default_factory=dict)
    ambient_rna_fraction: float = 0.05
    doublet_rate: float = 0.04
    dropout_rate: float = 0.1
    sample_quality: float = Field(0.9, ge=0.0, le=1.0)
    library_complexity: float = Field(0.8, ge=0.0, le=1.0)
    sequencing_depth_factor: float = 1.0
    capture_efficiency: float = 0.6


class ExperimentProgress(BaseModel):
    """Flags tracking which experiment stages have been completed."""

    samples_collected: bool = False
    cohort_selected: bool = False
    cells_cultured: bool = False
    library_prepared: bool = False
    perturbation_applied: bool = False
    cells_sequenced: bool = False
    qc_performed: bool = False
    data_filtered: bool = False
    data_normalized: bool = False
    batches_integrated: bool = False
    cells_clustered: bool = False
    de_performed: bool = False
    trajectories_inferred: bool = False
    pathways_analyzed: bool = False
    networks_inferred: bool = False
    markers_discovered: bool = False
    markers_validated: bool = False
    conclusion_reached: bool = False

    n_cells_after_filter: Optional[int] = None
    n_clusters_found: Optional[int] = None
    n_de_genes_found: Optional[int] = None
    n_markers_found: Optional[int] = None


class ResourceState(BaseModel):
    """Full internal resource tracking (superset of agent-visible ResourceUsage)."""

    budget_total: float = 100_000.0
    budget_used: float = 0.0
    time_limit_days: float = 180.0
    time_used_days: float = 0.0
    samples_available: int = 0
    samples_consumed: int = 0
    compute_hours_used: float = 0.0
    sequencing_lanes_used: int = 0
    reagent_kits_used: int = 0

    @property
    def budget_remaining(self) -> float:
        return max(0.0, self.budget_total - self.budget_used)

    @property
    def time_remaining_days(self) -> float:
        return max(0.0, self.time_limit_days - self.time_used_days)

    @property
    def budget_exhausted(self) -> bool:
        return self.budget_remaining <= 0

    @property
    def time_exhausted(self) -> bool:
        return self.time_remaining_days <= 0


class FullLatentState(BaseModel):
    """Complete hidden state of the simulated biological world."""

    biology: LatentBiologicalState = Field(
        default_factory=LatentBiologicalState
    )
    technical: TechnicalState = Field(default_factory=TechnicalState)
    progress: ExperimentProgress = Field(default_factory=ExperimentProgress)
    resources: ResourceState = Field(default_factory=ResourceState)
    hidden_failure_conditions: List[str] = Field(default_factory=list)
    mechanism_confidence: Dict[str, float] = Field(default_factory=dict)
    discovered_de_genes: List[str] = Field(default_factory=list)
    discovered_clusters: List[str] = Field(default_factory=list)
    task_modality: str = "scRNA-seq"
    step_count: int = 0
    rng_seed: int = 42
