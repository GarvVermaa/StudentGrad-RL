"""Stochastic noise models for the biological simulator."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


class NoiseModel:
    """Generates calibrated noise for simulated experimental outputs.

    All randomness is funnelled through a single ``numpy.Generator``
    so that episodes are reproducible given the same seed.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def reseed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    # ── expression-level noise ──────────────────────────────────────────

    def add_expression_noise(
        self,
        true_values: Dict[str, float],
        noise_level: float,
        dropout_rate: float,
    ) -> Dict[str, float]:
        noisy: Dict[str, float] = {}
        for gene, value in true_values.items():
            if self.rng.random() < dropout_rate:
                noisy[gene] = 0.0
            else:
                sigma = noise_level * abs(value) + 0.1
                noisy[gene] = float(value + self.rng.normal(0, sigma))
        return noisy

    # ── effect-size sampling ────────────────────────────────────────────

    def sample_effect_sizes(
        self,
        true_effects: Dict[str, float],
        sample_size: int,
        noise_level: float,
    ) -> Dict[str, float]:
        se = noise_level / max(np.sqrt(max(sample_size, 1)), 1e-6)
        return {
            gene: float(effect + self.rng.normal(0, se))
            for gene, effect in true_effects.items()
        }

    def sample_p_values(
        self,
        true_effects: Dict[str, float],
        sample_size: int,
        noise_level: float,
    ) -> Dict[str, float]:
        """Simulate approximate p-values from z-statistics."""
        from scipy import stats  # type: ignore[import-untyped]

        p_values: Dict[str, float] = {}
        se = noise_level / max(np.sqrt(max(sample_size, 1)), 1e-6)
        for gene, effect in true_effects.items():
            z = abs(effect) / max(se, 1e-8)
            p_values[gene] = float(2 * stats.norm.sf(z))
        return p_values

    # ── false discovery helpers ─────────────────────────────────────────

    def generate_false_positives(
        self, n_background_genes: int, fdr: float
    ) -> List[str]:
        n_fp = int(self.rng.binomial(n_background_genes, fdr))
        return [f"FP_GENE_{i}" for i in range(n_fp)]

    def generate_false_negatives(
        self, true_genes: List[str], fnr: float
    ) -> List[str]:
        """Return the subset of *true_genes* that are missed."""
        return [g for g in true_genes if self.rng.random() < fnr]

    # ── quality helpers ─────────────────────────────────────────────────

    def quality_degradation(
        self, base_quality: float, factors: List[float]
    ) -> float:
        q = base_quality
        for f in factors:
            q *= f
        return float(np.clip(q + self.rng.normal(0, 0.02), 0.0, 1.0))

    def sample_qc_metric(
        self, mean: float, std: float, clip_lo: float = 0.0, clip_hi: float = 1.0
    ) -> float:
        return float(np.clip(self.rng.normal(mean, std), clip_lo, clip_hi))

    def sample_count(self, lam: float) -> int:
        return int(self.rng.poisson(max(lam, 0)))

    def coin_flip(self, p: float) -> bool:
        return bool(self.rng.random() < p)

    def sample_cluster_count(
        self, n_true_populations: int, quality: float
    ) -> int:
        """Over- or under-clustering depending on preprocessing quality."""
        delta = self.rng.integers(-2, 3)
        noise_clusters = max(0, int(round((1.0 - quality) * 3)))
        return max(1, n_true_populations + delta + noise_clusters)

    def shuffle_ranking(
        self, items: List[str], noise_level: float
    ) -> List[str]:
        """Permute a ranking with Gaussian noise on ordinals."""
        n = len(items)
        if n == 0:
            return []
        scores = np.arange(n, dtype=float) + self.rng.normal(
            0, noise_level * n, size=n
        )
        order = np.argsort(scores)
        return [items[int(i)] for i in order]
