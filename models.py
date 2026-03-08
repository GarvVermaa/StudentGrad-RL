"""
Data models for the Bio-Experiment Planning RL Environment.

Defines the POMDP action and observation contracts for a scientific agent
that constructs biological experiment pipelines step-by-step.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from openenv.core.env_server.types import Action, Observation


# ── Action vocabulary ───────────────────────────────────────────────────────


class ActionType(str, Enum):
    COLLECT_SAMPLE = "collect_sample"
    SELECT_COHORT = "select_cohort"
    PREPARE_LIBRARY = "prepare_library"
    CULTURE_CELLS = "culture_cells"
    PERTURB_GENE = "perturb_gene"
    PERTURB_COMPOUND = "perturb_compound"
    SEQUENCE_CELLS = "sequence_cells"
    RUN_QC = "run_qc"
    FILTER_DATA = "filter_data"
    NORMALIZE_DATA = "normalize_data"
    INTEGRATE_BATCHES = "integrate_batches"
    CLUSTER_CELLS = "cluster_cells"
    DIFFERENTIAL_EXPRESSION = "differential_expression"
    TRAJECTORY_ANALYSIS = "trajectory_analysis"
    PATHWAY_ENRICHMENT = "pathway_enrichment"
    REGULATORY_NETWORK_INFERENCE = "regulatory_network_inference"
    MARKER_SELECTION = "marker_selection"
    VALIDATE_MARKER = "validate_marker"
    DESIGN_FOLLOWUP = "design_followup_experiment"
    REQUEST_SUBAGENT_REVIEW = "request_subagent_review"
    SYNTHESIZE_CONCLUSION = "synthesize_conclusion"


WET_LAB_ACTIONS = frozenset({
    ActionType.COLLECT_SAMPLE,
    ActionType.SELECT_COHORT,
    ActionType.PREPARE_LIBRARY,
    ActionType.CULTURE_CELLS,
    ActionType.PERTURB_GENE,
    ActionType.PERTURB_COMPOUND,
    ActionType.SEQUENCE_CELLS,
    ActionType.VALIDATE_MARKER,
})

COMPUTATIONAL_ACTIONS = frozenset({
    ActionType.RUN_QC,
    ActionType.FILTER_DATA,
    ActionType.NORMALIZE_DATA,
    ActionType.INTEGRATE_BATCHES,
    ActionType.CLUSTER_CELLS,
    ActionType.DIFFERENTIAL_EXPRESSION,
    ActionType.TRAJECTORY_ANALYSIS,
    ActionType.PATHWAY_ENRICHMENT,
    ActionType.REGULATORY_NETWORK_INFERENCE,
    ActionType.MARKER_SELECTION,
})

META_ACTIONS = frozenset({
    ActionType.DESIGN_FOLLOWUP,
    ActionType.REQUEST_SUBAGENT_REVIEW,
    ActionType.SYNTHESIZE_CONCLUSION,
})


class SubagentType(str, Enum):
    WET_LAB_PLANNER = "wet_lab_planner"
    COMPUTATIONAL_ANALYST = "computational_analyst"
    OMICS_QC_AGENT = "omics_qc_agent"
    CAUSAL_REASONING_AGENT = "causal_reasoning_agent"
    BUDGET_SCHEDULER = "budget_scheduler"
    BIOLOGICAL_RULE_CHECKER = "biological_rule_checker"
    TOOL_EXECUTOR = "tool_executor"
    RETROSPECTIVE_CRITIC = "retrospective_critic"
    REPORT_SYNTHESIZER = "report_synthesizer"


# ── Action schema ───────────────────────────────────────────────────────────


class ExperimentAction(Action):
    """Structured, compositional action for one experiment / analysis step.

    Hybrid representation: discrete *action_type* plus typed arguments,
    optional sub-agent / tool invocation, and calibration fields.
    """

    action_type: ActionType = Field(
        ..., description="Discrete experiment or analysis step type"
    )
    input_targets: List[str] = Field(
        default_factory=list,
        description="References to prior outputs, samples, or artifacts",
    )
    method: Optional[str] = Field(
        None, description="Specific method or tool (e.g. 'Seurat', 'CellRanger')"
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Method-specific parameters"
    )
    expected_output_type: Optional[str] = Field(
        None, description="What the agent expects this step to produce"
    )
    justification: Optional[str] = Field(
        None, description="Scientific rationale for this step"
    )
    invoked_subagent: Optional[SubagentType] = Field(
        None, description="Sub-agent to delegate to, if any"
    )
    tool_call_spec: Optional[Dict[str, Any]] = Field(
        None, description="Structured tool invocation specification"
    )
    confidence: float = Field(
        0.5, ge=0.0, le=1.0, description="Agent confidence in this step"
    )


# ── Intermediate outputs ────────────────────────────────────────────────────


class OutputType(str, Enum):
    QC_METRICS = "qc_metrics"
    COUNT_MATRIX_SUMMARY = "count_matrix_summary"
    EMBEDDING_SUMMARY = "embedding_summary"
    CLUSTER_RESULT = "cluster_result"
    DE_RESULT = "de_result"
    PATHWAY_RESULT = "pathway_result"
    TRAJECTORY_RESULT = "trajectory_result"
    VALIDATION_RESULT = "validation_result"
    NETWORK_RESULT = "network_result"
    SAMPLE_COLLECTION_RESULT = "sample_collection_result"
    LIBRARY_PREP_RESULT = "library_prep_result"
    SEQUENCING_RESULT = "sequencing_result"
    PERTURBATION_RESULT = "perturbation_result"
    CULTURE_RESULT = "culture_result"
    COHORT_RESULT = "cohort_result"
    FOLLOWUP_DESIGN = "followup_design"
    MARKER_RESULT = "marker_result"
    FAILURE_REPORT = "failure_report"
    SUBAGENT_REPORT = "subagent_report"
    CONCLUSION = "conclusion"


class IntermediateOutput(BaseModel):
    """A single simulated output from one pipeline step."""

    output_type: OutputType
    step_index: int
    success: bool = True
    quality_score: float = Field(1.0, ge=0.0, le=1.0)
    summary: str = ""
    data: Dict[str, Any] = Field(default_factory=dict)
    uncertainty: float = Field(0.0, ge=0.0, le=1.0)
    warnings: List[str] = Field(default_factory=list)
    artifacts_available: List[str] = Field(default_factory=list)


# ── Observable state components ─────────────────────────────────────────────


class ResourceUsage(BaseModel):
    budget_used: float = 0.0
    budget_remaining: float = 100_000.0
    time_used_days: float = 0.0
    time_remaining_days: float = 180.0
    samples_consumed: int = 0
    compute_hours_used: float = 0.0


class PipelineStepRecord(BaseModel):
    step_index: int
    action_type: ActionType
    method: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)
    output_summary: str = ""
    output_type: OutputType
    success: bool = True
    quality_score: float = 1.0
    resource_cost: float = 0.0
    time_cost_days: float = 0.0


class PaperReference(BaseModel):
    """Metadata for a literature source used to ground a task."""

    title: str
    citation: Optional[str] = None
    doi: Optional[str] = None
    pmid: Optional[str] = None
    url: Optional[str] = None


class ExpectedFinding(BaseModel):
    """A paper-backed result that the agent should try to recover."""

    finding: str
    category: str = "claim"
    keywords: List[str] = Field(default_factory=list)


class TaskSpec(BaseModel):
    """Specification of the biological problem to solve."""

    problem_statement: str = "Unspecified biological problem"
    modality: str = "scRNA-seq"
    organism: str = "human"
    tissue: str = "blood"
    conditions: List[str] = Field(default_factory=list)
    available_assays: List[str] = Field(default_factory=lambda: [
        "10x_chromium", "smart-seq2", "bulk_rna_seq",
        "atac-seq", "cite-seq", "spatial_transcriptomics",
    ])
    available_tools: List[str] = Field(default_factory=lambda: [
        "CellRanger", "Seurat", "Scanpy", "DESeq2", "GSEA",
        "Monocle", "scVelo", "CellChat", "SCENIC",
    ])
    budget_limit: float = 100_000.0
    time_limit_days: float = 180.0
    prior_observations: List[str] = Field(default_factory=list)
    success_criteria: List[str] = Field(default_factory=list)
    dataset_metadata: Dict[str, Any] = Field(default_factory=dict)
    paper_references: List[PaperReference] = Field(default_factory=list)
    expected_findings: List[ExpectedFinding] = Field(default_factory=list)


class ConclusionClaim(BaseModel):
    claim: str
    evidence_steps: List[int] = Field(default_factory=list)
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    claim_type: str = "correlational"
    supporting_data: Dict[str, Any] = Field(default_factory=dict)


# ── Observation schema ──────────────────────────────────────────────────────


class ExperimentObservation(Observation):
    """Full observable state returned to the agent at each timestep.

    Deliberately excludes hidden latent biological truth, hidden failure
    conditions, and ground-truth mechanisms.
    """

    task: TaskSpec = Field(default_factory=TaskSpec)
    step_index: int = 0
    pipeline_history: List[PipelineStepRecord] = Field(default_factory=list)
    available_assays: List[str] = Field(default_factory=list)
    available_tools: List[str] = Field(default_factory=list)
    resource_usage: ResourceUsage = Field(default_factory=ResourceUsage)
    latest_output: Optional[IntermediateOutput] = None
    all_outputs: List[IntermediateOutput] = Field(default_factory=list)
    discovered_markers: List[str] = Field(default_factory=list)
    candidate_mechanisms: List[str] = Field(default_factory=list)
    uncertainty_summary: Dict[str, float] = Field(default_factory=dict)
    subagent_outputs: List[Dict[str, Any]] = Field(default_factory=list)
    conclusions: List[ConclusionClaim] = Field(default_factory=list)
    rule_violations: List[str] = Field(default_factory=list)
    step_reward_breakdown: Dict[str, float] = Field(default_factory=dict)
