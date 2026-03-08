"""Evaluation suite for the bio-experiment planning environment.

Separates metrics into four families:
  - online RL metrics      (collected during training rollouts)
  - offline benchmark metrics (computed on a fixed held-out set)
  - expert review metrics  (for human-in-the-loop evaluation)
  - simulator fidelity metrics (how well the simulator matches reality)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .trajectory import Trajectory, TrajectoryDataset


@dataclass
class MetricResult:
    name: str
    value: float
    details: Dict[str, Any] = field(default_factory=dict)


class EvaluationSuite:
    """Computes and aggregates evaluation metrics over trajectory datasets."""

    # ── online RL metrics ───────────────────────────────────────────────

    @staticmethod
    def online_metrics(trajectories: List[Trajectory]) -> List[MetricResult]:
        if not trajectories:
            return []

        rewards = [t.total_reward for t in trajectories]
        lengths = [len(t.steps) for t in trajectories]
        successes = [t.success for t in trajectories]

        return [
            MetricResult("mean_return", float(np.mean(rewards))),
            MetricResult("median_return", float(np.median(rewards))),
            MetricResult("std_return", float(np.std(rewards))),
            MetricResult("mean_episode_length", float(np.mean(lengths))),
            MetricResult("success_rate", float(np.mean(successes))),
        ]

    # ── offline benchmark metrics ───────────────────────────────────────

    @staticmethod
    def benchmark_metrics(dataset: TrajectoryDataset) -> List[MetricResult]:
        results: List[MetricResult] = []
        if len(dataset) == 0:
            return results

        results.append(MetricResult(
            "pipeline_validity_rate",
            EvaluationSuite._pipeline_validity_rate(dataset),
        ))
        results.append(MetricResult(
            "ordering_score",
            EvaluationSuite._ordering_score(dataset),
        ))
        results.append(MetricResult(
            "action_diversity",
            EvaluationSuite._action_diversity(dataset),
        ))
        results.append(MetricResult(
            "mean_conclusion_confidence",
            EvaluationSuite._mean_conclusion_confidence(dataset),
        ))
        return results

    # ── expert review metrics (stubs) ───────────────────────────────────

    @staticmethod
    def expert_review_metrics(
        trajectories: List[Trajectory],
        expert_scores: Optional[Dict[str, float]] = None,
    ) -> List[MetricResult]:
        """Placeholder for human expert review scores.

        In practice, each trajectory would be scored by a domain expert
        on axes such as scientific validity, creativity, and efficiency.
        """
        if not expert_scores:
            return [MetricResult("expert_review", 0.0, {"note": "no scores provided"})]
        avg = float(np.mean(list(expert_scores.values())))
        return [MetricResult("expert_review_mean", avg, expert_scores)]

    # ── simulator fidelity metrics (stubs) ──────────────────────────────

    @staticmethod
    def simulator_fidelity_metrics(
        simulated: TrajectoryDataset,
        real: Optional[TrajectoryDataset] = None,
    ) -> List[MetricResult]:
        """Compare simulated trajectories against real experimental data.

        When ``real`` is provided, computes distributional distances
        between simulated and real output statistics.
        """
        if real is None or len(real) == 0:
            return [MetricResult("fidelity", 0.0, {"note": "no real data"})]

        sim_rewards = [t.total_reward for t in simulated.trajectories]
        real_rewards = [t.total_reward for t in real.trajectories]

        reward_gap = abs(float(np.mean(sim_rewards)) - float(np.mean(real_rewards)))
        return [MetricResult("reward_distribution_gap", reward_gap)]

    # ── internal helpers ────────────────────────────────────────────────

    @staticmethod
    def _pipeline_validity_rate(ds: TrajectoryDataset) -> float:
        valid = 0
        for t in ds.trajectories:
            violations = sum(
                1 for s in t.steps
                if not s.observation.get("rule_violations") == []
                and s.observation.get("rule_violations") is not None
            )
            if violations == 0:
                valid += 1
        return valid / max(len(ds), 1)

    @staticmethod
    def _ordering_score(ds: TrajectoryDataset) -> float:
        scores: List[float] = []
        for t in ds.trajectories:
            breakdown_scores = []
            for s in t.steps:
                bd = s.reward_breakdown
                if "ordering" in bd:
                    breakdown_scores.append(bd["ordering"])
            if breakdown_scores:
                scores.append(float(np.mean(breakdown_scores)))
        return float(np.mean(scores)) if scores else 0.0

    @staticmethod
    def _action_diversity(ds: TrajectoryDataset) -> float:
        all_types: set = set()
        for t in ds.trajectories:
            for s in t.steps:
                at = s.action.get("action_type")
                if at:
                    all_types.add(at)
        return len(all_types)

    @staticmethod
    def _mean_conclusion_confidence(ds: TrajectoryDataset) -> float:
        confs: List[float] = []
        for t in ds.trajectories:
            for s in t.steps:
                conclusions = s.observation.get("conclusions", [])
                for c in conclusions:
                    if isinstance(c, dict) and "confidence" in c:
                        confs.append(c["confidence"])
        return float(np.mean(confs)) if confs else 0.0
