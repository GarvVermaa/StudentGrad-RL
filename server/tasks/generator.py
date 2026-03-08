"""Task generator — produces (TaskSpec, FullLatentState) pairs for episodes.

Supports three modes:
  1. Select from the pre-defined scenario library.
  2. Randomly perturb a scenario for domain-randomisation.
  3. Compose a fully procedural scenario (tissue × modality × difficulty).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from models import TaskSpec

from server.simulator.latent_state import (
    CellPopulation,
    ExperimentProgress,
    FullLatentState,
    GeneProgram,
    LatentBiologicalState,
    ResourceState,
    TechnicalState,
)
from .scenarios import SCENARIO_LIBRARY, Scenario


class TaskGenerator:
    """Generates task + latent-state pairs for environment episodes."""

    def __init__(
        self,
        scenarios: Optional[List[Scenario]] = None,
        domain_randomise: bool = True,
    ):
        self.scenarios = scenarios or SCENARIO_LIBRARY
        self.domain_randomise = domain_randomise

    def generate(
        self,
        *,
        seed: Optional[int] = None,
        scenario_name: Optional[str] = None,
    ) -> Tuple[TaskSpec, FullLatentState]:
        rng = np.random.default_rng(seed)

        if scenario_name:
            scenario = self._find_scenario(scenario_name)
        else:
            idx = int(rng.integers(0, len(self.scenarios)))
            scenario = self.scenarios[idx]

        task = scenario.task.model_copy(deep=True)
        biology = scenario.biology.model_copy(deep=True)
        technical = scenario.technical.model_copy(deep=True)

        if self.domain_randomise:
            self._randomise(rng, task, biology, technical)

        latent = FullLatentState(
            biology=biology,
            technical=technical,
            progress=ExperimentProgress(),
            resources=ResourceState(
                budget_total=task.budget_limit,
                time_limit_days=task.time_limit_days,
            ),
            hidden_failure_conditions=list(scenario.hidden_failure_conditions),
            rng_seed=seed or 0,
        )
        return task, latent

    def list_scenarios(self) -> List[str]:
        return [s.name for s in self.scenarios]

    # ── internals ───────────────────────────────────────────────────────

    def _find_scenario(self, name: str) -> Scenario:
        for s in self.scenarios:
            if s.name == name:
                return s
        available = ", ".join(self.list_scenarios())
        raise ValueError(f"Unknown scenario '{name}'. Available: {available}")

    def _randomise(
        self,
        rng: np.random.Generator,
        task: TaskSpec,
        bio: LatentBiologicalState,
        tech: TechnicalState,
    ) -> None:
        budget_scale = float(rng.uniform(0.7, 1.3))
        task.budget_limit *= budget_scale
        task.time_limit_days *= float(rng.uniform(0.8, 1.2))

        tech.dropout_rate = float(np.clip(
            tech.dropout_rate + rng.normal(0, 0.02), 0.01, 0.3
        ))
        tech.doublet_rate = float(np.clip(
            tech.doublet_rate + rng.normal(0, 0.01), 0.01, 0.15
        ))
        tech.sample_quality = float(np.clip(
            tech.sample_quality + rng.normal(0, 0.05), 0.5, 1.0
        ))
        tech.ambient_rna_fraction = float(np.clip(
            tech.ambient_rna_fraction + rng.normal(0, 0.01), 0.01, 0.15
        ))
        for batch_id in list(tech.batch_effects.keys()):
            tech.batch_effects[batch_id] = float(np.clip(
                tech.batch_effects[batch_id] + rng.normal(0, 0.03), 0.0, 0.4
            ))

        for pop in bio.cell_populations:
            pop.proportion = float(np.clip(
                pop.proportion * rng.uniform(0.8, 1.2), 0.01, 0.8
            ))
        total = sum(p.proportion for p in bio.cell_populations) or 1.0
        for pop in bio.cell_populations:
            pop.proportion /= total

        for comparison, effects in bio.true_de_genes.items():
            for gene in list(effects.keys()):
                effects[gene] *= float(rng.uniform(0.8, 1.2))

        bio.n_true_cells = max(
            1000,
            int(bio.n_true_cells * rng.uniform(0.6, 1.4)),
        )
