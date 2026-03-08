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


# ── Tool, Assay & Modality Registries ──────────────────────────────────────


class ToolCategory(str, Enum):
    ALIGNMENT = "alignment"
    PREPROCESSING = "preprocessing"
    NORMALIZATION = "normalization"
    DIMENSIONALITY_REDUCTION = "dimensionality_reduction"
    CLUSTERING = "clustering"
    DIFFERENTIAL_EXPRESSION = "differential_expression"
    TRAJECTORY = "trajectory"
    GENE_REGULATORY_NETWORK = "gene_regulatory_network"
    CELL_COMMUNICATION = "cell_communication"
    SPATIAL = "spatial"
    MULTIMODAL_INTEGRATION = "multimodal_integration"
    GENE_SET_ANALYSIS = "gene_set_analysis"
    VARIANT_CALLING = "variant_calling"
    PEAK_CALLING = "peak_calling"
    IMPUTATION = "imputation"
    BATCH_CORRECTION = "batch_correction"
    CELL_TYPE_ANNOTATION = "cell_type_annotation"
    SIMULATION = "simulation"
    VISUALIZATION = "visualization"
    QUALITY_CONTROL = "quality_control"
    PERTURBATION_ANALYSIS = "perturbation_analysis"


class ToolSpec(BaseModel):
    """Registry entry describing a bioinformatics tool."""

    name: str
    category: ToolCategory
    modalities: List[str] = Field(default_factory=list)
    description: str = ""
    input_types: List[str] = Field(default_factory=list)
    output_types: List[str] = Field(default_factory=list)
    typical_runtime_hours: float = 0.1
    typical_cost_usd: float = 0.0
    requires_gpu: bool = False
    open_source: bool = True


TOOL_REGISTRY: Dict[str, ToolSpec] = {
    # ── Alignment & quantification ──
    "CellRanger": ToolSpec(
        name="CellRanger",
        category=ToolCategory.ALIGNMENT,
        modalities=["scRNA-seq", "scATAC-seq", "CITE-seq", "scMultiome"],
        description="10x Genomics pipeline for alignment, barcode demux, and counting",
        input_types=["fastq"],
        output_types=["count_matrix", "bam"],
        typical_runtime_hours=4.0,
        open_source=False,
    ),
    "STARsolo": ToolSpec(
        name="STARsolo",
        category=ToolCategory.ALIGNMENT,
        modalities=["scRNA-seq", "scATAC-seq"],
        description="Drop-seq / 10x-compatible aligner built into STAR",
        input_types=["fastq"],
        output_types=["count_matrix", "bam"],
        typical_runtime_hours=3.0,
    ),
    "kallisto_bustools": ToolSpec(
        name="kallisto_bustools",
        category=ToolCategory.ALIGNMENT,
        modalities=["scRNA-seq"],
        description="Pseudoalignment-based lightweight quantification",
        input_types=["fastq"],
        output_types=["count_matrix"],
        typical_runtime_hours=1.0,
    ),
    "Salmon_alevin": ToolSpec(
        name="Salmon_alevin",
        category=ToolCategory.ALIGNMENT,
        modalities=["scRNA-seq"],
        description="Quasi-mapping quantification for single-cell RNA-seq",
        input_types=["fastq"],
        output_types=["count_matrix"],
        typical_runtime_hours=1.5,
    ),
    "spaceranger": ToolSpec(
        name="spaceranger",
        category=ToolCategory.ALIGNMENT,
        modalities=["spatial_transcriptomics"],
        description="10x Visium spatial alignment and quantification",
        input_types=["fastq", "image"],
        output_types=["count_matrix", "spatial_coords"],
        typical_runtime_hours=3.0,
        open_source=False,
    ),
    # ── Preprocessing / analysis frameworks ──
    "Scanpy": ToolSpec(
        name="Scanpy",
        category=ToolCategory.PREPROCESSING,
        modalities=["scRNA-seq", "scATAC-seq", "spatial_transcriptomics"],
        description="Python single-cell analysis framework",
        input_types=["count_matrix", "h5ad"],
        output_types=["h5ad", "embedding", "cluster_result"],
        typical_runtime_hours=0.5,
    ),
    "Seurat": ToolSpec(
        name="Seurat",
        category=ToolCategory.PREPROCESSING,
        modalities=["scRNA-seq", "CITE-seq", "spatial_transcriptomics", "scATAC-seq"],
        description="R single-cell analysis toolkit with multimodal support",
        input_types=["count_matrix", "h5seurat"],
        output_types=["h5seurat", "embedding", "cluster_result"],
        typical_runtime_hours=0.5,
    ),
    "Bioconductor_SingleCellExperiment": ToolSpec(
        name="Bioconductor_SingleCellExperiment",
        category=ToolCategory.PREPROCESSING,
        modalities=["scRNA-seq"],
        description="R/Bioconductor framework for single-cell experiments",
        input_types=["count_matrix"],
        output_types=["sce_object"],
        typical_runtime_hours=0.3,
    ),
    # ── Normalization ──
    "scran": ToolSpec(
        name="scran",
        category=ToolCategory.NORMALIZATION,
        modalities=["scRNA-seq"],
        description="Pool-based size-factor normalization",
        input_types=["count_matrix"],
        output_types=["normalized_matrix"],
    ),
    "sctransform": ToolSpec(
        name="sctransform",
        category=ToolCategory.NORMALIZATION,
        modalities=["scRNA-seq"],
        description="Variance-stabilizing transformation via regularized NB regression",
        input_types=["count_matrix"],
        output_types=["normalized_matrix"],
    ),
    # ── Dimensionality reduction ──
    "scVI": ToolSpec(
        name="scVI",
        category=ToolCategory.DIMENSIONALITY_REDUCTION,
        modalities=["scRNA-seq", "CITE-seq", "scATAC-seq"],
        description="Deep generative model for scRNA-seq (variational inference)",
        input_types=["count_matrix"],
        output_types=["latent_embedding"],
        requires_gpu=True,
    ),
    "UMAP": ToolSpec(
        name="UMAP",
        category=ToolCategory.DIMENSIONALITY_REDUCTION,
        modalities=["scRNA-seq", "scATAC-seq", "CITE-seq", "spatial_transcriptomics"],
        description="Uniform manifold approximation for 2D/3D visualization",
        input_types=["pca_embedding", "latent_embedding"],
        output_types=["2d_embedding"],
    ),
    # ── Clustering ──
    "Leiden": ToolSpec(
        name="Leiden",
        category=ToolCategory.CLUSTERING,
        modalities=["scRNA-seq", "scATAC-seq", "CITE-seq"],
        description="Community detection via the Leiden algorithm",
        input_types=["knn_graph"],
        output_types=["cluster_result"],
    ),
    "Louvain": ToolSpec(
        name="Louvain",
        category=ToolCategory.CLUSTERING,
        modalities=["scRNA-seq", "scATAC-seq"],
        description="Community detection via Louvain modularity optimization",
        input_types=["knn_graph"],
        output_types=["cluster_result"],
    ),
    # ── Differential expression ──
    "DESeq2": ToolSpec(
        name="DESeq2",
        category=ToolCategory.DIFFERENTIAL_EXPRESSION,
        modalities=["bulk_rna_seq", "scRNA-seq"],
        description="Negative binomial GLM-based differential expression",
        input_types=["count_matrix"],
        output_types=["de_result"],
    ),
    "MAST": ToolSpec(
        name="MAST",
        category=ToolCategory.DIFFERENTIAL_EXPRESSION,
        modalities=["scRNA-seq"],
        description="Two-part hurdle model for scRNA-seq DE testing",
        input_types=["count_matrix"],
        output_types=["de_result"],
    ),
    "edgeR": ToolSpec(
        name="edgeR",
        category=ToolCategory.DIFFERENTIAL_EXPRESSION,
        modalities=["bulk_rna_seq", "scRNA-seq"],
        description="Empirical Bayes quasi-likelihood DE testing",
        input_types=["count_matrix"],
        output_types=["de_result"],
    ),
    "Wilcoxon": ToolSpec(
        name="Wilcoxon",
        category=ToolCategory.DIFFERENTIAL_EXPRESSION,
        modalities=["scRNA-seq"],
        description="Rank-sum test for marker gene detection",
        input_types=["count_matrix"],
        output_types=["de_result"],
    ),
    # ── Trajectory & RNA velocity ──
    "Monocle3": ToolSpec(
        name="Monocle3",
        category=ToolCategory.TRAJECTORY,
        modalities=["scRNA-seq"],
        description="Reversed graph embedding for pseudotime trajectories",
        input_types=["count_matrix", "embedding"],
        output_types=["trajectory_result", "pseudotime"],
    ),
    "scVelo": ToolSpec(
        name="scVelo",
        category=ToolCategory.TRAJECTORY,
        modalities=["scRNA-seq"],
        description="RNA velocity estimation via spliced/unspliced dynamics",
        input_types=["count_matrix"],
        output_types=["velocity_result"],
    ),
    "CellRank": ToolSpec(
        name="CellRank",
        category=ToolCategory.TRAJECTORY,
        modalities=["scRNA-seq"],
        description="Fate probability estimation combining velocity and transcriptomics",
        input_types=["velocity_result", "count_matrix"],
        output_types=["fate_probabilities"],
    ),
    "Slingshot": ToolSpec(
        name="Slingshot",
        category=ToolCategory.TRAJECTORY,
        modalities=["scRNA-seq"],
        description="Minimum spanning tree-based trajectory inference",
        input_types=["embedding", "cluster_result"],
        output_types=["trajectory_result", "pseudotime"],
    ),
    "PAGA": ToolSpec(
        name="PAGA",
        category=ToolCategory.TRAJECTORY,
        modalities=["scRNA-seq"],
        description="Partition-based graph abstraction for topology estimation",
        input_types=["knn_graph", "cluster_result"],
        output_types=["trajectory_result"],
    ),
    # ── Gene regulatory networks ──
    "SCENIC": ToolSpec(
        name="SCENIC",
        category=ToolCategory.GENE_REGULATORY_NETWORK,
        modalities=["scRNA-seq"],
        description="Single-cell regulatory network inference and clustering",
        input_types=["count_matrix"],
        output_types=["regulon_result", "network_result"],
        typical_runtime_hours=6.0,
    ),
    "CellOracle": ToolSpec(
        name="CellOracle",
        category=ToolCategory.GENE_REGULATORY_NETWORK,
        modalities=["scRNA-seq", "scATAC-seq", "scMultiome"],
        description="GRN-based in-silico perturbation prediction",
        input_types=["count_matrix", "peak_matrix"],
        output_types=["network_result", "perturbation_prediction"],
        typical_runtime_hours=4.0,
    ),
    # ── Cell-cell communication ──
    "CellChat": ToolSpec(
        name="CellChat",
        category=ToolCategory.CELL_COMMUNICATION,
        modalities=["scRNA-seq", "spatial_transcriptomics"],
        description="Ligand-receptor interaction inference with communication patterns",
        input_types=["count_matrix", "cluster_result"],
        output_types=["communication_result"],
    ),
    "NicheNet": ToolSpec(
        name="NicheNet",
        category=ToolCategory.CELL_COMMUNICATION,
        modalities=["scRNA-seq"],
        description="Ligand-target link prediction using prior knowledge",
        input_types=["count_matrix", "de_result"],
        output_types=["communication_result"],
    ),
    "LIANA": ToolSpec(
        name="LIANA",
        category=ToolCategory.CELL_COMMUNICATION,
        modalities=["scRNA-seq", "spatial_transcriptomics"],
        description="Framework unifying multiple ligand-receptor methods",
        input_types=["count_matrix", "cluster_result"],
        output_types=["communication_result"],
    ),
    # ── Spatial analysis ──
    "squidpy": ToolSpec(
        name="squidpy",
        category=ToolCategory.SPATIAL,
        modalities=["spatial_transcriptomics"],
        description="Spatial omics analysis (neighborhood, co-occurrence, image features)",
        input_types=["count_matrix", "spatial_coords"],
        output_types=["spatial_result"],
    ),
    "cell2location": ToolSpec(
        name="cell2location",
        category=ToolCategory.SPATIAL,
        modalities=["spatial_transcriptomics"],
        description="Spatial deconvolution mapping cell types to tissue locations",
        input_types=["count_matrix", "spatial_coords", "reference_h5ad"],
        output_types=["deconvolution_result"],
        requires_gpu=True,
    ),
    "BANKSY": ToolSpec(
        name="BANKSY",
        category=ToolCategory.SPATIAL,
        modalities=["spatial_transcriptomics"],
        description="Spatially-aware clustering combining cell and neighbor features",
        input_types=["count_matrix", "spatial_coords"],
        output_types=["cluster_result"],
    ),
    # ── Multimodal integration ──
    "Harmony": ToolSpec(
        name="Harmony",
        category=ToolCategory.BATCH_CORRECTION,
        modalities=["scRNA-seq", "scATAC-seq", "CITE-seq"],
        description="Fast iterative batch correction on PCA embeddings",
        input_types=["pca_embedding"],
        output_types=["corrected_embedding"],
    ),
    "scanorama": ToolSpec(
        name="scanorama",
        category=ToolCategory.BATCH_CORRECTION,
        modalities=["scRNA-seq"],
        description="Panoramic stitching of scRNA-seq batches",
        input_types=["count_matrix"],
        output_types=["corrected_embedding", "corrected_matrix"],
    ),
    "BBKNN": ToolSpec(
        name="BBKNN",
        category=ToolCategory.BATCH_CORRECTION,
        modalities=["scRNA-seq"],
        description="Batch-balanced KNN graph construction",
        input_types=["pca_embedding"],
        output_types=["knn_graph"],
    ),
    "WNN": ToolSpec(
        name="WNN",
        category=ToolCategory.MULTIMODAL_INTEGRATION,
        modalities=["CITE-seq", "scMultiome"],
        description="Weighted nearest neighbors for multimodal integration (Seurat v4+)",
        input_types=["rna_embedding", "protein_embedding"],
        output_types=["multimodal_embedding"],
    ),
    "MOFA+": ToolSpec(
        name="MOFA+",
        category=ToolCategory.MULTIMODAL_INTEGRATION,
        modalities=["scMultiome", "CITE-seq"],
        description="Multi-omics factor analysis for unsupervised integration",
        input_types=["count_matrix", "peak_matrix"],
        output_types=["factor_result"],
    ),
    "ArchR": ToolSpec(
        name="ArchR",
        category=ToolCategory.PREPROCESSING,
        modalities=["scATAC-seq", "scMultiome"],
        description="Full-featured scATAC-seq analysis framework in R",
        input_types=["fragments", "bam"],
        output_types=["peak_matrix", "gene_activity_matrix"],
        typical_runtime_hours=2.0,
    ),
    "Signac": ToolSpec(
        name="Signac",
        category=ToolCategory.PREPROCESSING,
        modalities=["scATAC-seq", "scMultiome"],
        description="Seurat extension for chromatin accessibility analysis",
        input_types=["fragments", "peak_matrix"],
        output_types=["peak_matrix", "motif_result"],
    ),
    "chromVAR": ToolSpec(
        name="chromVAR",
        category=ToolCategory.PEAK_CALLING,
        modalities=["scATAC-seq", "scMultiome"],
        description="TF motif accessibility deviation scoring",
        input_types=["peak_matrix"],
        output_types=["motif_deviation_scores"],
    ),
    # ── Gene set / pathway analysis ──
    "GSEA": ToolSpec(
        name="GSEA",
        category=ToolCategory.GENE_SET_ANALYSIS,
        modalities=["bulk_rna_seq", "scRNA-seq"],
        description="Gene Set Enrichment Analysis (preranked or phenotype-based)",
        input_types=["de_result", "ranked_gene_list"],
        output_types=["pathway_result"],
    ),
    "clusterProfiler": ToolSpec(
        name="clusterProfiler",
        category=ToolCategory.GENE_SET_ANALYSIS,
        modalities=["bulk_rna_seq", "scRNA-seq"],
        description="ORA & GSEA with GO, KEGG, Reactome, and custom gene sets",
        input_types=["de_result", "gene_list"],
        output_types=["pathway_result"],
    ),
    "decoupleR": ToolSpec(
        name="decoupleR",
        category=ToolCategory.GENE_SET_ANALYSIS,
        modalities=["scRNA-seq", "bulk_rna_seq", "spatial_transcriptomics"],
        description="Unified framework for functional activity inference (TF, pathway)",
        input_types=["count_matrix", "de_result"],
        output_types=["activity_scores"],
    ),
    # ── Cell type annotation ──
    "celltypist": ToolSpec(
        name="celltypist",
        category=ToolCategory.CELL_TYPE_ANNOTATION,
        modalities=["scRNA-seq"],
        description="Automated cell type classification with pre-trained models",
        input_types=["count_matrix"],
        output_types=["annotation_result"],
    ),
    "SingleR": ToolSpec(
        name="SingleR",
        category=ToolCategory.CELL_TYPE_ANNOTATION,
        modalities=["scRNA-seq"],
        description="Reference-based cell type annotation using correlation",
        input_types=["count_matrix", "reference_dataset"],
        output_types=["annotation_result"],
    ),
    "scArches": ToolSpec(
        name="scArches",
        category=ToolCategory.CELL_TYPE_ANNOTATION,
        modalities=["scRNA-seq", "scATAC-seq", "CITE-seq"],
        description="Reference mapping and label transfer via deep learning",
        input_types=["count_matrix", "reference_model"],
        output_types=["annotation_result", "latent_embedding"],
        requires_gpu=True,
    ),
    # ── Imputation ──
    "MAGIC": ToolSpec(
        name="MAGIC",
        category=ToolCategory.IMPUTATION,
        modalities=["scRNA-seq"],
        description="Markov affinity-based graph imputation of dropout zeros",
        input_types=["count_matrix"],
        output_types=["imputed_matrix"],
    ),
    # ── Perturbation analysis ──
    "MILO": ToolSpec(
        name="MILO",
        category=ToolCategory.PERTURBATION_ANALYSIS,
        modalities=["scRNA-seq"],
        description="Differential abundance testing on KNN graph neighborhoods",
        input_types=["count_matrix", "knn_graph"],
        output_types=["da_result"],
    ),
    "Mixscape": ToolSpec(
        name="Mixscape",
        category=ToolCategory.PERTURBATION_ANALYSIS,
        modalities=["Perturb-seq", "CROP-seq"],
        description="Seurat extension for CRISPR screen perturbation analysis",
        input_types=["count_matrix", "guide_assignments"],
        output_types=["perturbation_result"],
    ),
    "MIMOSCA": ToolSpec(
        name="MIMOSCA",
        category=ToolCategory.PERTURBATION_ANALYSIS,
        modalities=["Perturb-seq", "CROP-seq"],
        description="Multi-input multi-output single-cell analysis for screens",
        input_types=["count_matrix", "guide_assignments"],
        output_types=["perturbation_result"],
    ),
    # ── Quality control ──
    "scrublet": ToolSpec(
        name="scrublet",
        category=ToolCategory.QUALITY_CONTROL,
        modalities=["scRNA-seq"],
        description="Computational doublet detection via synthetic doublets",
        input_types=["count_matrix"],
        output_types=["doublet_scores"],
    ),
    "DoubletFinder": ToolSpec(
        name="DoubletFinder",
        category=ToolCategory.QUALITY_CONTROL,
        modalities=["scRNA-seq"],
        description="Artificial nearest-neighbor doublet detection",
        input_types=["count_matrix"],
        output_types=["doublet_scores"],
    ),
    "SoupX": ToolSpec(
        name="SoupX",
        category=ToolCategory.QUALITY_CONTROL,
        modalities=["scRNA-seq"],
        description="Ambient RNA contamination estimation and removal",
        input_types=["count_matrix", "raw_count_matrix"],
        output_types=["corrected_matrix"],
    ),
    "DecontX": ToolSpec(
        name="DecontX",
        category=ToolCategory.QUALITY_CONTROL,
        modalities=["scRNA-seq"],
        description="Bayesian ambient RNA decontamination",
        input_types=["count_matrix"],
        output_types=["corrected_matrix"],
    ),
    # ── Simulation ──
    "Splatter": ToolSpec(
        name="Splatter",
        category=ToolCategory.SIMULATION,
        modalities=["scRNA-seq"],
        description="Flexible scRNA-seq data simulation framework",
        input_types=["simulation_params"],
        output_types=["simulated_count_matrix"],
    ),
}


class Modality(str, Enum):
    SCRNA_SEQ = "scRNA-seq"
    SCATAC_SEQ = "scATAC-seq"
    CITE_SEQ = "CITE-seq"
    SPATIAL_TRANSCRIPTOMICS = "spatial_transcriptomics"
    BULK_RNA_SEQ = "bulk_rna_seq"
    SCRNA_MULTIOME = "scMultiome"
    PERTURB_SEQ = "Perturb-seq"
    CROP_SEQ = "CROP-seq"
    SMART_SEQ2 = "Smart-seq2"
    SLIDE_SEQ = "Slide-seq"
    MERFISH = "MERFISH"
    SEQFISH = "seqFISH"
    PATCH_SEQ = "Patch-seq"
    SHARE_SEQ = "SHARE-seq"
    SNARE_SEQ = "SNARE-seq"
    SC_HI_C = "scHi-C"
    SCBS_SEQ = "scBS-seq"
    SCNMT_SEQ = "scNMT-seq"


class ModalitySpec(BaseModel):
    """Registry entry for a single-cell or bulk assay modality."""

    name: str
    modality: Modality
    measurement: str = ""
    resolution: str = "single-cell"
    multiplexable: bool = False
    typical_cells: str = "1k-20k"
    typical_cost_per_sample_usd: float = 5000.0
    compatible_tools: List[str] = Field(default_factory=list)
    description: str = ""


MODALITY_REGISTRY: Dict[str, ModalitySpec] = {
    "scRNA-seq": ModalitySpec(
        name="scRNA-seq",
        modality=Modality.SCRNA_SEQ,
        measurement="mRNA transcripts",
        typical_cells="5k-20k",
        typical_cost_per_sample_usd=5000.0,
        compatible_tools=[
            "CellRanger", "STARsolo", "kallisto_bustools", "Scanpy", "Seurat",
            "scVI", "Leiden", "DESeq2", "MAST", "Monocle3", "scVelo", "SCENIC",
            "CellChat", "GSEA", "celltypist", "scrublet",
        ],
        description="Droplet-based single-cell RNA sequencing (e.g. 10x Chromium)",
    ),
    "scATAC-seq": ModalitySpec(
        name="scATAC-seq",
        modality=Modality.SCATAC_SEQ,
        measurement="open chromatin regions",
        typical_cells="5k-15k",
        typical_cost_per_sample_usd=6000.0,
        compatible_tools=[
            "CellRanger", "ArchR", "Signac", "chromVAR", "Scanpy", "Leiden",
        ],
        description="Single-cell Assay for Transposase-Accessible Chromatin",
    ),
    "CITE-seq": ModalitySpec(
        name="CITE-seq",
        modality=Modality.CITE_SEQ,
        measurement="mRNA + surface proteins (ADT)",
        multiplexable=True,
        typical_cells="5k-20k",
        typical_cost_per_sample_usd=8000.0,
        compatible_tools=[
            "CellRanger", "Seurat", "WNN", "MOFA+", "Scanpy", "Leiden",
        ],
        description="Cellular Indexing of Transcriptomes and Epitopes by Sequencing",
    ),
    "spatial_transcriptomics": ModalitySpec(
        name="spatial_transcriptomics",
        modality=Modality.SPATIAL_TRANSCRIPTOMICS,
        measurement="spatially resolved transcripts",
        resolution="spot (55µm) or subcellular",
        typical_cells="1k-10k spots",
        typical_cost_per_sample_usd=7000.0,
        compatible_tools=[
            "spaceranger", "squidpy", "cell2location", "BANKSY", "Scanpy", "Seurat",
        ],
        description="Spatially resolved transcriptomics (Visium, MERFISH, Slide-seq, etc.)",
    ),
    "bulk_rna_seq": ModalitySpec(
        name="bulk_rna_seq",
        modality=Modality.BULK_RNA_SEQ,
        measurement="aggregate mRNA across cells",
        resolution="bulk",
        typical_cells="N/A",
        typical_cost_per_sample_usd=500.0,
        compatible_tools=["DESeq2", "edgeR", "GSEA", "clusterProfiler"],
        description="Standard bulk RNA sequencing",
    ),
    "scMultiome": ModalitySpec(
        name="scMultiome",
        modality=Modality.SCRNA_MULTIOME,
        measurement="mRNA + open chromatin (joint)",
        typical_cells="5k-15k",
        typical_cost_per_sample_usd=10000.0,
        compatible_tools=[
            "CellRanger", "ArchR", "Signac", "Seurat", "MOFA+", "CellOracle",
        ],
        description="10x Multiome (joint scRNA + scATAC from same cell)",
    ),
    "Perturb-seq": ModalitySpec(
        name="Perturb-seq",
        modality=Modality.PERTURB_SEQ,
        measurement="mRNA + CRISPR guide assignment",
        multiplexable=True,
        typical_cells="10k-100k",
        typical_cost_per_sample_usd=15000.0,
        compatible_tools=[
            "CellRanger", "Scanpy", "Seurat", "Mixscape", "MIMOSCA",
        ],
        description="Pooled CRISPR screens with single-cell RNA readout",
    ),
    "CROP-seq": ModalitySpec(
        name="CROP-seq",
        modality=Modality.CROP_SEQ,
        measurement="mRNA + CRISPR guide assignment",
        multiplexable=True,
        typical_cells="10k-50k",
        typical_cost_per_sample_usd=12000.0,
        compatible_tools=[
            "CellRanger", "Scanpy", "Seurat", "Mixscape", "MIMOSCA",
        ],
        description="CRISPR dropout screen with single-cell RNA readout",
    ),
    "Smart-seq2": ModalitySpec(
        name="Smart-seq2",
        modality=Modality.SMART_SEQ2,
        measurement="full-length mRNA transcripts",
        typical_cells="100-1000",
        typical_cost_per_sample_usd=10000.0,
        compatible_tools=["Scanpy", "Seurat", "DESeq2", "MAST", "Monocle3"],
        description="Plate-based full-length scRNA-seq with high sensitivity",
    ),
    "MERFISH": ModalitySpec(
        name="MERFISH",
        modality=Modality.MERFISH,
        measurement="in situ mRNA (imaging-based)",
        resolution="subcellular",
        typical_cells="10k-1M",
        typical_cost_per_sample_usd=20000.0,
        compatible_tools=["squidpy", "Scanpy", "BANKSY"],
        description="Multiplexed Error-Robust FISH for spatial transcriptomics",
    ),
    "Slide-seq": ModalitySpec(
        name="Slide-seq",
        modality=Modality.SLIDE_SEQ,
        measurement="spatially resolved mRNA (bead array)",
        resolution="10µm",
        typical_cells="10k-50k beads",
        typical_cost_per_sample_usd=8000.0,
        compatible_tools=["squidpy", "cell2location", "Scanpy"],
        description="Near-cellular spatial transcriptomics on bead arrays",
    ),
    "Patch-seq": ModalitySpec(
        name="Patch-seq",
        modality=Modality.PATCH_SEQ,
        measurement="mRNA + electrophysiology + morphology",
        typical_cells="10-500",
        typical_cost_per_sample_usd=50000.0,
        compatible_tools=["Scanpy", "Seurat"],
        description="Combined patch-clamp electrophysiology and scRNA-seq",
    ),
    "scHi-C": ModalitySpec(
        name="scHi-C",
        modality=Modality.SC_HI_C,
        measurement="3D chromatin contacts",
        typical_cells="1k-10k",
        typical_cost_per_sample_usd=15000.0,
        compatible_tools=["Scanpy"],
        description="Single-cell chromosome conformation capture",
    ),
    "scBS-seq": ModalitySpec(
        name="scBS-seq",
        modality=Modality.SCBS_SEQ,
        measurement="DNA methylation (CpG)",
        typical_cells="100-5k",
        typical_cost_per_sample_usd=12000.0,
        compatible_tools=["Scanpy"],
        description="Single-cell bisulfite sequencing for DNA methylation",
    ),
    "scNMT-seq": ModalitySpec(
        name="scNMT-seq",
        modality=Modality.SCNMT_SEQ,
        measurement="nucleosome + methylation + transcription (joint)",
        typical_cells="100-1k",
        typical_cost_per_sample_usd=25000.0,
        compatible_tools=["MOFA+", "Scanpy"],
        description="Joint single-cell nucleosome, methylation, and transcription",
    ),
}


class AssayCategory(str, Enum):
    SEQUENCING = "sequencing"
    IMAGING = "imaging"
    PERTURBATION = "perturbation"
    FUNCTIONAL = "functional"
    EPIGENOMICS = "epigenomics"
    PROTEOMICS = "proteomics"
    METABOLOMICS = "metabolomics"


class AssaySpec(BaseModel):
    """Registry entry for a laboratory assay or protocol."""

    name: str
    category: AssayCategory
    modalities: List[str] = Field(default_factory=list)
    description: str = ""
    typical_duration_days: float = 1.0
    typical_cost_usd: float = 1000.0
    requires_live_cells: bool = False
    requires_fresh_tissue: bool = False
    throughput: str = "medium"
    outputs: List[str] = Field(default_factory=list)


ASSAY_REGISTRY: Dict[str, AssaySpec] = {
    "10x_chromium": AssaySpec(
        name="10x_chromium",
        category=AssayCategory.SEQUENCING,
        modalities=["scRNA-seq", "scATAC-seq", "CITE-seq", "scMultiome"],
        description="10x Genomics Chromium droplet-based single-cell partitioning",
        typical_duration_days=2.0,
        typical_cost_usd=5000.0,
        requires_live_cells=True,
        throughput="high (500-20k cells)",
        outputs=["fastq", "count_matrix"],
    ),
    "smart-seq2": AssaySpec(
        name="smart-seq2",
        category=AssayCategory.SEQUENCING,
        modalities=["Smart-seq2"],
        description="Plate-based full-length cDNA scRNA-seq",
        typical_duration_days=3.0,
        typical_cost_usd=10000.0,
        requires_live_cells=True,
        throughput="low (96-384 cells)",
        outputs=["fastq", "count_matrix"],
    ),
    "smart-seq3": AssaySpec(
        name="smart-seq3",
        category=AssayCategory.SEQUENCING,
        modalities=["Smart-seq2"],
        description="Improved full-length scRNA-seq with UMIs",
        typical_duration_days=3.0,
        typical_cost_usd=10000.0,
        requires_live_cells=True,
        throughput="low (96-384 cells)",
        outputs=["fastq", "count_matrix"],
    ),
    "bulk_rna_seq": AssaySpec(
        name="bulk_rna_seq",
        category=AssayCategory.SEQUENCING,
        modalities=["bulk_rna_seq"],
        description="Standard bulk RNA sequencing with poly-A or ribo-depletion",
        typical_duration_days=3.0,
        typical_cost_usd=500.0,
        throughput="high",
        outputs=["fastq", "count_matrix"],
    ),
    "atac-seq": AssaySpec(
        name="atac-seq",
        category=AssayCategory.EPIGENOMICS,
        modalities=["scATAC-seq"],
        description="Assay for Transposase-Accessible Chromatin using sequencing",
        typical_duration_days=2.0,
        typical_cost_usd=6000.0,
        requires_live_cells=True,
        outputs=["fastq", "fragments", "peak_matrix"],
    ),
    "cite-seq": AssaySpec(
        name="cite-seq",
        category=AssayCategory.PROTEOMICS,
        modalities=["CITE-seq"],
        description="Simultaneous RNA + surface protein via DNA-barcoded antibodies",
        typical_duration_days=2.0,
        typical_cost_usd=8000.0,
        requires_live_cells=True,
        throughput="high (5k-20k cells)",
        outputs=["fastq", "count_matrix", "adt_matrix"],
    ),
    "10x_multiome": AssaySpec(
        name="10x_multiome",
        category=AssayCategory.SEQUENCING,
        modalities=["scMultiome"],
        description="Joint scRNA-seq + scATAC-seq from the same cell",
        typical_duration_days=2.0,
        typical_cost_usd=10000.0,
        requires_live_cells=True,
        throughput="high (5k-15k cells)",
        outputs=["fastq", "count_matrix", "fragments"],
    ),
    "visium": AssaySpec(
        name="visium",
        category=AssayCategory.SEQUENCING,
        modalities=["spatial_transcriptomics"],
        description="10x Visium spatially barcoded capture on tissue sections",
        typical_duration_days=3.0,
        typical_cost_usd=7000.0,
        requires_fresh_tissue=True,
        throughput="medium (1k-5k spots)",
        outputs=["fastq", "count_matrix", "spatial_coords", "image"],
    ),
    "visium_hd": AssaySpec(
        name="visium_hd",
        category=AssayCategory.SEQUENCING,
        modalities=["spatial_transcriptomics"],
        description="High-definition Visium with 2µm bin resolution",
        typical_duration_days=3.0,
        typical_cost_usd=10000.0,
        requires_fresh_tissue=True,
        throughput="high",
        outputs=["fastq", "count_matrix", "spatial_coords", "image"],
    ),
    "merfish": AssaySpec(
        name="merfish",
        category=AssayCategory.IMAGING,
        modalities=["MERFISH"],
        description="Multiplexed Error-Robust FISH imaging-based spatial",
        typical_duration_days=5.0,
        typical_cost_usd=20000.0,
        requires_fresh_tissue=True,
        throughput="high (100-1000 genes, millions of transcripts)",
        outputs=["transcript_coords", "cell_segmentation"],
    ),
    "seqfish_plus": AssaySpec(
        name="seqfish_plus",
        category=AssayCategory.IMAGING,
        modalities=["seqFISH"],
        description="Sequential FISH for imaging-based spatial transcriptomics",
        typical_duration_days=5.0,
        typical_cost_usd=15000.0,
        requires_fresh_tissue=True,
        outputs=["transcript_coords"],
    ),
    "slide-seq": AssaySpec(
        name="slide-seq",
        category=AssayCategory.SEQUENCING,
        modalities=["Slide-seq"],
        description="Near-cellular spatial transcriptomics on bead arrays",
        typical_duration_days=3.0,
        typical_cost_usd=8000.0,
        requires_fresh_tissue=True,
        outputs=["count_matrix", "spatial_coords"],
    ),
    "perturb-seq": AssaySpec(
        name="perturb-seq",
        category=AssayCategory.PERTURBATION,
        modalities=["Perturb-seq"],
        description="Pooled CRISPR screen + scRNA-seq readout",
        typical_duration_days=14.0,
        typical_cost_usd=15000.0,
        requires_live_cells=True,
        throughput="high (10k-100k cells)",
        outputs=["fastq", "count_matrix", "guide_assignments"],
    ),
    "crop-seq": AssaySpec(
        name="crop-seq",
        category=AssayCategory.PERTURBATION,
        modalities=["CROP-seq"],
        description="CRISPR dropout screening with scRNA-seq readout",
        typical_duration_days=14.0,
        typical_cost_usd=12000.0,
        requires_live_cells=True,
        throughput="high (10k-50k cells)",
        outputs=["fastq", "count_matrix", "guide_assignments"],
    ),
    "patch-seq": AssaySpec(
        name="patch-seq",
        category=AssayCategory.FUNCTIONAL,
        modalities=["Patch-seq"],
        description="Patch-clamp electrophysiology + scRNA-seq on same neuron",
        typical_duration_days=7.0,
        typical_cost_usd=50000.0,
        requires_live_cells=True,
        throughput="very low (10-100 cells)",
        outputs=["fastq", "count_matrix", "ephys_trace", "morphology"],
    ),
    "sc_hi_c": AssaySpec(
        name="sc_hi_c",
        category=AssayCategory.EPIGENOMICS,
        modalities=["scHi-C"],
        description="Single-cell chromosome conformation capture",
        typical_duration_days=5.0,
        typical_cost_usd=15000.0,
        outputs=["contact_matrix"],
    ),
    "sc_bisulfite": AssaySpec(
        name="sc_bisulfite",
        category=AssayCategory.EPIGENOMICS,
        modalities=["scBS-seq"],
        description="Single-cell bisulfite sequencing for DNA methylation profiling",
        typical_duration_days=5.0,
        typical_cost_usd=12000.0,
        outputs=["methylation_matrix"],
    ),
    "sc_nmt_seq": AssaySpec(
        name="sc_nmt_seq",
        category=AssayCategory.EPIGENOMICS,
        modalities=["scNMT-seq"],
        description="Joint nucleosome occupancy, methylation, and transcription",
        typical_duration_days=7.0,
        typical_cost_usd=25000.0,
        requires_live_cells=True,
        throughput="low (100-1k cells)",
        outputs=["count_matrix", "methylation_matrix", "accessibility_matrix"],
    ),
    "flow_cytometry": AssaySpec(
        name="flow_cytometry",
        category=AssayCategory.FUNCTIONAL,
        modalities=[],
        description="Fluorescence-based cell sorting and phenotyping",
        typical_duration_days=1.0,
        typical_cost_usd=500.0,
        requires_live_cells=True,
        throughput="very high (millions of cells)",
        outputs=["cell_counts", "sorted_cells"],
    ),
    "mass_cytometry_CyTOF": AssaySpec(
        name="mass_cytometry_CyTOF",
        category=AssayCategory.PROTEOMICS,
        modalities=[],
        description="Mass-tag cytometry for 40+ protein markers per cell",
        typical_duration_days=2.0,
        typical_cost_usd=3000.0,
        requires_live_cells=True,
        throughput="high (100k-1M cells)",
        outputs=["protein_expression_matrix"],
    ),
    "western_blot": AssaySpec(
        name="western_blot",
        category=AssayCategory.PROTEOMICS,
        modalities=[],
        description="Protein detection and semi-quantification by size separation",
        typical_duration_days=2.0,
        typical_cost_usd=200.0,
        outputs=["band_image", "relative_quantification"],
    ),
    "qPCR": AssaySpec(
        name="qPCR",
        category=AssayCategory.FUNCTIONAL,
        modalities=[],
        description="Quantitative PCR for targeted gene expression validation",
        typical_duration_days=1.0,
        typical_cost_usd=100.0,
        throughput="low (target genes)",
        outputs=["ct_values", "fold_change"],
    ),
    "immunofluorescence": AssaySpec(
        name="immunofluorescence",
        category=AssayCategory.IMAGING,
        modalities=[],
        description="Antibody-based fluorescence imaging of proteins in situ",
        typical_duration_days=2.0,
        typical_cost_usd=500.0,
        outputs=["fluorescence_image"],
    ),
    "elisa": AssaySpec(
        name="elisa",
        category=AssayCategory.PROTEOMICS,
        modalities=[],
        description="Enzyme-linked immunosorbent assay for secreted protein quantification",
        typical_duration_days=1.0,
        typical_cost_usd=300.0,
        throughput="medium (96-384 well)",
        outputs=["protein_concentration"],
    ),
    "cell_viability_assay": AssaySpec(
        name="cell_viability_assay",
        category=AssayCategory.FUNCTIONAL,
        modalities=[],
        description="MTT/CellTiter-Glo viability and proliferation measurement",
        typical_duration_days=1.0,
        typical_cost_usd=200.0,
        requires_live_cells=True,
        throughput="high (96-384 well)",
        outputs=["viability_scores"],
    ),
}


# ── Registry helper functions ──────────────────────────────────────────────


def tools_for_modality(modality: str) -> List[ToolSpec]:
    """Return all registered tools compatible with a given modality."""
    return [t for t in TOOL_REGISTRY.values() if modality in t.modalities]


def assays_for_modality(modality: str) -> List[AssaySpec]:
    """Return all registered assays that produce a given modality."""
    return [a for a in ASSAY_REGISTRY.values() if modality in a.modalities]


def tools_by_category(category: ToolCategory) -> List[ToolSpec]:
    """Return all registered tools in a given category."""
    return [t for t in TOOL_REGISTRY.values() if t.category == category]


# ── Sub-agents ─────────────────────────────────────────────────────────────


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
    available_assays: List[str] = Field(
        default_factory=lambda: list(ASSAY_REGISTRY.keys()),
    )
    available_tools: List[str] = Field(
        default_factory=lambda: list(TOOL_REGISTRY.keys()),
    )
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
