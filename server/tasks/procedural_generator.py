"""Procedural scenario generator.

Composes biologically coherent ``Scenario`` objects from the curated
palette in ``bio_palette``, producing fully populated
``LatentBiologicalState`` instances that drive every simulator tool
(clustering, DE, pathway enrichment, trajectory, regulatory networks,
marker validation) with realistic intermediate outputs.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from models import TaskSpec

from server.simulator.latent_state import (
    CellPopulation,
    LatentBiologicalState,
    TechnicalState,
)

from .bio_palette import (
    DISEASE_PROFILES,
    HIDDEN_FAILURE_TEMPLATES,
    PATHWAY_LIBRARY,
    PERTURBATION_TEMPLATES,
    REGULATORY_TEMPLATES,
    TISSUE_CELL_TYPES,
    TRAJECTORY_TEMPLATES,
    CellTypeTemplate,
    DiseaseProfile,
)
from .scenarios import Scenario

logger = logging.getLogger(__name__)

SCENARIO_TYPES = ("de", "trajectory", "perturbation", "biomarker")

_DIFFICULTY_PARAMS = {
    "easy": {
        "n_pops": (4, 5),
        "de_scale": (1.2, 1.6),
        "noise_dropout": (0.05, 0.10),
        "noise_doublet": (0.03, 0.06),
        "noise_ambient": (0.02, 0.05),
        "noise_batch_strength": (0.05, 0.12),
        "n_batches": (1, 2),
        "budget_range": (70_000, 100_000),
        "time_range": (100, 150),
        "sample_quality": (0.85, 0.95),
        "include_trajectory": False,
        "include_perturbation": False,
        "include_network": False,
        "include_failure_conditions": False,
    },
    "medium": {
        "n_pops": (5, 7),
        "de_scale": (0.9, 1.3),
        "noise_dropout": (0.08, 0.14),
        "noise_doublet": (0.04, 0.08),
        "noise_ambient": (0.03, 0.07),
        "noise_batch_strength": (0.08, 0.18),
        "n_batches": (1, 3),
        "budget_range": (80_000, 120_000),
        "time_range": (120, 180),
        "sample_quality": (0.78, 0.92),
        "include_trajectory": True,
        "include_perturbation": False,
        "include_network": True,
        "include_failure_conditions": False,
    },
    "hard": {
        "n_pops": (6, 8),
        "de_scale": (0.6, 1.0),
        "noise_dropout": (0.10, 0.20),
        "noise_doublet": (0.06, 0.12),
        "noise_ambient": (0.05, 0.10),
        "noise_batch_strength": (0.12, 0.25),
        "n_batches": (2, 4),
        "budget_range": (90_000, 140_000),
        "time_range": (140, 200),
        "sample_quality": (0.65, 0.85),
        "include_trajectory": True,
        "include_perturbation": True,
        "include_network": True,
        "include_failure_conditions": True,
    },
}


def generate_scenario(
    seed: int,
    difficulty: str = "medium",
    scenario_type: Optional[str] = None,
) -> Scenario:
    """Generate a single procedural scenario with complete latent state.

    Parameters
    ----------
    seed
        RNG seed for reproducibility.
    difficulty
        One of ``"easy"``, ``"medium"``, ``"hard"``.
    scenario_type
        One of ``"de"``, ``"trajectory"``, ``"perturbation"``,
        ``"biomarker"``, or ``None`` for random selection.
    """
    rng = np.random.default_rng(seed)
    params = _DIFFICULTY_PARAMS[difficulty]

    if scenario_type is None:
        scenario_type = rng.choice(SCENARIO_TYPES)

    disease_key = rng.choice(list(DISEASE_PROFILES.keys()))
    disease = DISEASE_PROFILES[disease_key]
    tissue = disease.tissue

    cell_templates = TISSUE_CELL_TYPES.get(tissue, [])
    if not cell_templates:
        tissue = rng.choice(list(TISSUE_CELL_TYPES.keys()))
        cell_templates = TISSUE_CELL_TYPES[tissue]

    populations = _sample_populations(rng, cell_templates, disease, params)
    de_genes = _build_de_genes(rng, disease, params)
    pathways = _build_pathways(rng, disease)
    markers = _derive_markers(rng, de_genes, disease)
    mechanisms = list(disease.mechanism_templates)
    n_cells = int(rng.integers(8_000, 22_000))

    trajectory = None
    if scenario_type == "trajectory" or (
        params["include_trajectory"] and rng.random() < 0.4
    ):
        trajectory = _build_trajectory(rng, tissue, populations)

    reg_network: Dict[str, List[str]] = {}
    if scenario_type == "trajectory" or (
        params["include_network"] and rng.random() < 0.5
    ):
        reg_network = _build_regulatory_network(rng, tissue, populations)

    perturbation_effects: Dict[str, Dict[str, float]] = {}
    if scenario_type == "perturbation" or (
        params["include_perturbation"] and rng.random() < 0.5
    ):
        perturbation_effects = _build_perturbation(rng, disease)

    technical = _build_technical(rng, params)

    hidden_failures: List[str] = []
    if params["include_failure_conditions"] and rng.random() < 0.6:
        n_failures = int(rng.integers(1, 3))
        indices = rng.choice(
            len(HIDDEN_FAILURE_TEMPLATES), size=min(n_failures, len(HIDDEN_FAILURE_TEMPLATES)), replace=False,
        )
        hidden_failures = [HIDDEN_FAILURE_TEMPLATES[i] for i in indices]

    task = _build_task(rng, disease, tissue, scenario_type, params, perturbation_effects)

    biology = LatentBiologicalState(
        cell_populations=populations,
        true_de_genes=de_genes,
        true_pathways=pathways,
        true_trajectory=trajectory,
        true_regulatory_network=reg_network,
        perturbation_effects=perturbation_effects,
        true_markers=markers,
        causal_mechanisms=mechanisms,
        n_true_cells=n_cells,
    )

    name = f"proc_{disease.name}_{scenario_type}_{seed}"

    tags = [scenario_type, "scRNA-seq", tissue, disease.name, difficulty]

    return Scenario(
        name=name,
        task=task,
        biology=biology,
        technical=technical,
        hidden_failure_conditions=hidden_failures,
        difficulty=difficulty,
        tags=tags,
    )


def generate_procedural_scenarios(
    n: int = 20,
    seed: int = 42,
) -> List[Scenario]:
    """Pre-generate a pool of procedural scenarios across difficulties."""
    rng = np.random.default_rng(seed)
    scenarios: List[Scenario] = []
    difficulties = ["easy", "medium", "hard"]

    for i in range(n):
        diff = difficulties[i % len(difficulties)]
        child_seed = int(rng.integers(0, 2**31))
        scenario = generate_scenario(
            seed=child_seed,
            difficulty=diff,
            scenario_type=None,
        )
        scenarios.append(scenario)

    logger.info("Generated %d procedural scenarios.", len(scenarios))
    return scenarios


# ── Internal builders ───────────────────────────────────────────────────────


def _sample_populations(
    rng: np.random.Generator,
    templates: List[CellTypeTemplate],
    disease: DiseaseProfile,
    params: dict,
) -> List[CellPopulation]:
    lo, hi = params["n_pops"]
    n_pops = int(rng.integers(lo, hi + 1))
    n_pops = min(n_pops, len(templates))

    indices = rng.choice(len(templates), size=n_pops, replace=False)
    selected = [templates[i] for i in sorted(indices)]

    responding_names = set(disease.responding_cell_types)

    populations: List[CellPopulation] = []
    for tmpl in selected:
        prop = float(rng.uniform(*tmpl.proportion_range))
        state = rng.choice(tmpl.states)

        condition_response: Dict[str, float] = {}
        if tmpl.disease_responsive and tmpl.name in responding_names:
            condition_response[disease.condition_name] = float(
                rng.uniform(*tmpl.response_range)
            )

        populations.append(CellPopulation(
            name=tmpl.name,
            proportion=prop,
            marker_genes=list(tmpl.marker_genes),
            state=state,
            condition_response=condition_response,
        ))

    total = sum(p.proportion for p in populations)
    if total > 0:
        for p in populations:
            p.proportion = round(p.proportion / total, 4)

    return populations


def _build_de_genes(
    rng: np.random.Generator,
    disease: DiseaseProfile,
    params: dict,
) -> Dict[str, Dict[str, float]]:
    comparison = f"{disease.condition_name}_vs_healthy"
    scale_lo, scale_hi = params["de_scale"]

    effects: Dict[str, float] = {}
    for gene, (lo, hi) in disease.de_genes.items():
        base = float(rng.uniform(lo, hi))
        scale = float(rng.uniform(scale_lo, scale_hi))
        if base > 0:
            effects[gene] = round(base * scale, 3)
        else:
            effects[gene] = round(base * scale, 3)

    return {comparison: effects}


def _build_pathways(
    rng: np.random.Generator,
    disease: DiseaseProfile,
) -> Dict[str, float]:
    pathways: Dict[str, float] = {}
    for pw, (lo, hi) in disease.pathways.items():
        pathways[pw] = round(float(rng.uniform(lo, hi)), 3)
    return pathways


def _derive_markers(
    rng: np.random.Generator,
    de_genes: Dict[str, Dict[str, float]],
    disease: DiseaseProfile,
) -> List[str]:
    markers = list(disease.markers)

    all_effects: Dict[str, float] = {}
    for effects in de_genes.values():
        all_effects.update(effects)

    for gene in markers:
        if gene not in all_effects:
            all_effects[gene] = float(rng.uniform(1.0, 2.5))
            for comp_effects in de_genes.values():
                comp_effects[gene] = all_effects[gene]

    n_markers = min(len(markers), int(rng.integers(3, 7)))
    return markers[:n_markers]


def _build_trajectory(
    rng: np.random.Generator,
    tissue: str,
    populations: List[CellPopulation],
) -> Optional[Dict[str, Any]]:
    pop_names = {p.name for p in populations}

    for tmpl in TRAJECTORY_TEMPLATES:
        if tmpl.tissue == tissue:
            valid_branches = [
                branch for branch in tmpl.branches
                if all(node in pop_names for node in branch)
            ]
            if valid_branches:
                return {
                    "root": tmpl.root_population,
                    "n_lineages": len(valid_branches),
                    "branching": len(valid_branches) > 1,
                    "branches": valid_branches,
                }

    if len(populations) >= 3:
        root = populations[0].name
        branches = [[root, p.name] for p in populations[1:]]
        selected = branches[:int(rng.integers(2, min(4, len(branches)) + 1))]
        return {
            "root": root,
            "n_lineages": len(selected),
            "branching": len(selected) > 1,
            "branches": selected,
        }

    return None


def _build_regulatory_network(
    rng: np.random.Generator,
    tissue: str,
    populations: List[CellPopulation],
) -> Dict[str, List[str]]:
    all_genes = set()
    for p in populations:
        all_genes.update(p.marker_genes)

    network: Dict[str, List[str]] = {}

    tissue_to_programs = {
        "bone_marrow": ["erythroid", "myeloid", "stem_cell"],
        "thymus": ["lymphoid"],
        "blood": ["lymphoid", "myeloid"],
        "spleen": ["lymphoid"],
        "brain": ["neuronal", "inflammatory"],
        "heart": ["fibrotic", "inflammatory"],
        "lung": ["fibrotic", "inflammatory"],
        "liver": ["fibrotic", "inflammatory"],
        "kidney": ["fibrotic", "inflammatory"],
        "colon": ["inflammatory", "stem_cell"],
        "pancreas": ["inflammatory"],
        "skin": ["inflammatory"],
        "breast": ["inflammatory"],
        "synovium": ["inflammatory", "lymphoid"],
        "aorta": ["inflammatory"],
    }

    programs = tissue_to_programs.get(tissue, ["inflammatory"])
    for prog_name in programs:
        prog = REGULATORY_TEMPLATES.get(prog_name, {})
        for tf, targets in prog.items():
            network[tf] = list(targets)

    if not network:
        for p in populations[:2]:
            if len(p.marker_genes) >= 2:
                tf = p.marker_genes[0]
                network[tf] = p.marker_genes[1:]

    return network


def _build_perturbation(
    rng: np.random.Generator,
    disease: DiseaseProfile,
) -> Dict[str, Dict[str, float]]:
    disease_pathways = set(disease.pathways.keys())

    matching = [
        (name, tmpl) for name, tmpl in PERTURBATION_TEMPLATES.items()
        if tmpl.target_pathway in disease_pathways
    ]

    if matching:
        name, tmpl = matching[int(rng.integers(0, len(matching)))]
    else:
        name = rng.choice(list(PERTURBATION_TEMPLATES.keys()))
        tmpl = PERTURBATION_TEMPLATES[name]

    scaled: Dict[str, float] = {}
    for gene, effect in tmpl.gene_effects.items():
        scale = float(rng.uniform(0.7, 1.3))
        scaled[gene] = round(effect * scale, 3)

    return {name: scaled}


def _build_technical(
    rng: np.random.Generator,
    params: dict,
) -> TechnicalState:
    n_batches = int(rng.integers(*params["n_batches"]))
    batch_effects: Dict[str, float] = {}
    for i in range(max(1, n_batches)):
        strength = float(rng.uniform(*params["noise_batch_strength"]))
        batch_effects[f"batch_{i}"] = round(strength, 3)

    return TechnicalState(
        batch_effects=batch_effects,
        dropout_rate=round(float(rng.uniform(*params["noise_dropout"])), 3),
        doublet_rate=round(float(rng.uniform(*params["noise_doublet"])), 3),
        ambient_rna_fraction=round(float(rng.uniform(*params["noise_ambient"])), 3),
        sample_quality=round(float(rng.uniform(*params["sample_quality"])), 3),
    )


def _build_task(
    rng: np.random.Generator,
    disease: DiseaseProfile,
    tissue: str,
    scenario_type: str,
    params: dict,
    perturbation_effects: Dict[str, Dict[str, float]],
) -> TaskSpec:
    budget = float(rng.integers(*params["budget_range"]))
    time_days = float(rng.integers(*params["time_range"]))

    if scenario_type == "de":
        problem = (
            f"Identify differentially expressed genes between "
            f"{disease.display_name} and healthy {tissue} tissue "
            f"using single-cell RNA sequencing."
        )
        criteria = [
            f"Identify DE genes between {disease.condition_name} and healthy",
            "Validate at least one candidate marker",
        ]
    elif scenario_type == "trajectory":
        problem = (
            f"Infer the developmental trajectory of cell populations "
            f"in {tissue} tissue in the context of {disease.display_name}."
        )
        criteria = [
            "Reconstruct branching lineage structure",
            "Identify key transcription factors driving fate decisions",
        ]
    elif scenario_type == "perturbation":
        pert_name = next(iter(perturbation_effects), "treatment")
        pert_tmpl = PERTURBATION_TEMPLATES.get(pert_name)
        pert_desc = pert_tmpl.description if pert_tmpl else pert_name
        problem = (
            f"Determine the effect of {pert_desc} on cell states "
            f"in {tissue} tissue affected by {disease.display_name}."
        )
        criteria = [
            "Quantify shift in cell activation states",
            f"Identify pathways modulated by {pert_name}",
            "Propose validation strategy",
        ]
    else:
        top_marker = disease.markers[0] if disease.markers else "candidate"
        problem = (
            f"Validate candidate biomarker {top_marker} for "
            f"{disease.display_name} in {tissue} tissue using "
            f"single-cell RNA sequencing."
        )
        criteria = [
            f"Validate {top_marker} as a disease marker",
            "Confirm expression specificity across cell types",
        ]

    conditions = ["healthy", disease.condition_name]
    if scenario_type == "perturbation" and perturbation_effects:
        pert_name = next(iter(perturbation_effects))
        conditions = [f"untreated_{disease.condition_name}", f"{pert_name}_treated"]

    return TaskSpec(
        problem_statement=problem,
        modality="scRNA-seq",
        organism="human",
        tissue=tissue,
        conditions=conditions,
        budget_limit=budget,
        time_limit_days=time_days,
        success_criteria=criteria,
    )
