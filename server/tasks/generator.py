"""Task generator for StudentGrad — with domain randomisation."""

from __future__ import annotations

import random
from typing import Optional, Tuple

from models import StudentTaskSpec
from server.simulator.latent_state import FullLatentState, StudentContextState, LatentStudentState
from server.tasks.scenarios import SCENARIOS, SCENARIO_MAP, get_scenario


class TaskGenerator:
    def __init__(self, domain_randomise: bool = True):
        self.domain_randomise = domain_randomise

    def generate(
        self,
        seed: int = 42,
        scenario_name: Optional[str] = None,
    ) -> Tuple[StudentTaskSpec, FullLatentState]:
        rng = random.Random(seed)

        if scenario_name and scenario_name in SCENARIO_MAP:
            scenario = get_scenario(scenario_name)
        else:
            # Default to hard scenario for hackathon
            scenario = SCENARIO_MAP.get("hard_full_year", SCENARIOS[-1])

        full_state = scenario.build_full_state()
        full_state.rng_seed = seed

        if self.domain_randomise:
            self._randomise(full_state, rng)

        return scenario.task, full_state

    def _randomise(self, state: FullLatentState, rng: random.Random) -> None:
        """Perturb parameters ±20% so agent can't memorise fixed numbers."""
        def jitter(v: float, frac: float = 0.2) -> float:
            return v * (1.0 + rng.uniform(-frac, frac))

        lat = state.latent
        lat.true_learning_rates = {k: min(2.0, max(0.1, jitter(v))) for k, v in lat.true_learning_rates.items()}
        lat.true_fatigue_threshold = min(95.0, max(60.0, jitter(lat.true_fatigue_threshold, 0.15)))
        lat.true_exam_difficulty = {k: min(2.0, max(0.5, jitter(v, 0.15))) for k, v in lat.true_exam_difficulty.items()}
        lat.true_skill_rates = {k: min(2.0, max(0.3, jitter(v))) for k, v in lat.true_skill_rates.items()}

        ctx = state.context
        ctx.sick_probability = min(0.15, max(0.01, jitter(ctx.sick_probability, 0.3)))
        ctx.quiz_probability = min(0.10, max(0.01, jitter(ctx.quiz_probability, 0.3)))
        ctx.base_noise = min(0.3, max(0.02, jitter(ctx.base_noise, 0.2)))