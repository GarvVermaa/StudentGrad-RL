"""Gymnasium-compatible wrapper around ``BioExperimentEnvironment``.

Provides ``BioExperimentGymEnv`` which wraps the OpenEnv environment for
local in-process RL training (no HTTP/WebSocket overhead).

Observation and action spaces are represented as ``gymnasium.spaces.Dict``
so that standard RL libraries (SB3, CleanRL, etc.) can ingest them.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from models import ActionType, ExperimentAction, ExperimentObservation
from server.hackathon_environment import BioExperimentEnvironment, MAX_STEPS


ACTION_TYPE_LIST = list(ActionType)
_N_ACTION_TYPES = len(ACTION_TYPE_LIST)

_MAX_OUTPUTS = MAX_STEPS
_MAX_HISTORY = MAX_STEPS
_VEC_DIM = 64


class BioExperimentGymEnv(gym.Env):
    """Gymnasium ``Env`` backed by the in-process simulator.

    Observations are flattened into a dictionary of NumPy arrays suitable
    for RL policy networks.  Actions are integer-indexed action types with
    a continuous confidence scalar.

    For LLM-based agents or planners that prefer structured
    ``ExperimentAction`` objects, use the underlying
    ``BioExperimentEnvironment`` directly instead.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        self._env = BioExperimentEnvironment()
        self.render_mode = render_mode

        self.action_space = spaces.Dict({
            "action_type": spaces.Discrete(_N_ACTION_TYPES),
            "confidence": spaces.Box(0.0, 1.0, shape=(), dtype=np.float32),
        })

        self.observation_space = spaces.Dict({
            "step_index": spaces.Discrete(MAX_STEPS + 1),
            "budget_remaining_frac": spaces.Box(0.0, 1.0, shape=(), dtype=np.float32),
            "time_remaining_frac": spaces.Box(0.0, 1.0, shape=(), dtype=np.float32),
            "progress_flags": spaces.MultiBinary(18),
            "latest_quality": spaces.Box(0.0, 1.0, shape=(), dtype=np.float32),
            "latest_uncertainty": spaces.Box(0.0, 1.0, shape=(), dtype=np.float32),
            "avg_quality": spaces.Box(0.0, 1.0, shape=(), dtype=np.float32),
            "avg_uncertainty": spaces.Box(0.0, 1.0, shape=(), dtype=np.float32),
            "n_violations": spaces.Discrete(20),
            "n_outputs": spaces.Discrete(_MAX_OUTPUTS + 1),
            "cumulative_reward": spaces.Box(-100.0, 100.0, shape=(), dtype=np.float32),
        })

        self._last_obs: Optional[ExperimentObservation] = None

    # ── Gymnasium interface ─────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        super().reset(seed=seed)
        obs = self._env.reset()
        self._last_obs = obs
        return self._vectorise(obs), self._info(obs)

    def step(
        self, action: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        action_idx = int(action["action_type"])
        confidence = float(action.get("confidence", 0.5))

        experiment_action = ExperimentAction(
            action_type=ACTION_TYPE_LIST[action_idx],
            confidence=confidence,
        )
        obs = self._env.step(experiment_action)
        self._last_obs = obs

        terminated = obs.done
        truncated = obs.step_index >= MAX_STEPS and not terminated
        reward = obs.reward

        return (
            self._vectorise(obs),
            reward,
            terminated,
            truncated,
            self._info(obs),
        )

    def render(self) -> Optional[str]:
        if self.render_mode != "human" or self._last_obs is None:
            return None
        obs = self._last_obs
        lines = [
            f"Step {obs.step_index}",
            f"  Task: {obs.task.problem_statement[:80]}",
            f"  Budget: ${obs.resource_usage.budget_remaining:,.0f} remaining",
            f"  Time: {obs.resource_usage.time_remaining_days:.0f} days remaining",
        ]
        if obs.latest_output:
            lines.append(f"  Latest: {obs.latest_output.summary}")
        if obs.rule_violations:
            lines.append(f"  Violations: {obs.rule_violations}")
        text = "\n".join(lines)
        print(text)
        return text

    # ── helpers ─────────────────────────────────────────────────────────

    def _vectorise(self, obs: ExperimentObservation) -> Dict[str, Any]:
        progress = self._env._latent.progress if self._env._latent else None
        flags = np.zeros(18, dtype=np.int8)
        if progress:
            flag_names = [
                "samples_collected", "cohort_selected", "cells_cultured",
                "library_prepared", "perturbation_applied", "cells_sequenced",
                "qc_performed", "data_filtered", "data_normalized",
                "batches_integrated", "cells_clustered", "de_performed",
                "trajectories_inferred", "pathways_analyzed",
                "networks_inferred", "markers_discovered",
                "markers_validated", "conclusion_reached",
            ]
            for i, f in enumerate(flag_names):
                flags[i] = int(getattr(progress, f, False))

        unc = obs.uncertainty_summary
        lo = obs.latest_output

        return {
            "step_index": obs.step_index,
            "budget_remaining_frac": np.float32(
                obs.resource_usage.budget_remaining
                / max(obs.task.budget_limit, 1)
            ),
            "time_remaining_frac": np.float32(
                obs.resource_usage.time_remaining_days
                / max(obs.task.time_limit_days, 1)
            ),
            "progress_flags": flags,
            "latest_quality": np.float32(lo.quality_score if lo else 0.0),
            "latest_uncertainty": np.float32(lo.uncertainty if lo else 0.0),
            "avg_quality": np.float32(unc.get("avg_quality", 0.0)),
            "avg_uncertainty": np.float32(unc.get("avg_uncertainty", 0.0)),
            "n_violations": min(len(obs.rule_violations), 19),
            "n_outputs": min(len(obs.all_outputs), _MAX_OUTPUTS),
            "cumulative_reward": np.float32(
                obs.metadata.get("cumulative_reward", 0.0)
                if obs.metadata else 0.0
            ),
        }

    def _info(self, obs: ExperimentObservation) -> Dict[str, Any]:
        return {
            "structured_obs": obs,
            "episode_id": obs.metadata.get("episode_id") if obs.metadata else None,
        }
