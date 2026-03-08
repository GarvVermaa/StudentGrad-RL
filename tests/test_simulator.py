"""Tests for the latent-state simulator modules."""

import pytest

from models import ActionType, ExperimentAction, OutputType
from server.simulator.latent_state import (
    CellPopulation,
    ExperimentProgress,
    FullLatentState,
    LatentBiologicalState,
    ResourceState,
    TechnicalState,
)
from server.simulator.noise import NoiseModel
from server.simulator.output_generator import OutputGenerator
from server.simulator.transition import TransitionEngine


def _make_state() -> FullLatentState:
    return FullLatentState(
        biology=LatentBiologicalState(
            cell_populations=[
                CellPopulation(name="A", proportion=0.6, marker_genes=["G1"]),
                CellPopulation(name="B", proportion=0.4, marker_genes=["G2"]),
            ],
            true_de_genes={"disease_vs_healthy": {"G1": 2.0, "G2": -1.5}},
            true_pathways={"apoptosis": 0.7},
            true_markers=["G1"],
            causal_mechanisms=["G1-driven apoptosis"],
            n_true_cells=5000,
        ),
        technical=TechnicalState(dropout_rate=0.1, doublet_rate=0.04),
        progress=ExperimentProgress(),
        resources=ResourceState(budget_total=50_000, time_limit_days=90),
    )


class TestNoiseModel:
    def test_deterministic_with_seed(self):
        n1 = NoiseModel(seed=42)
        n2 = NoiseModel(seed=42)
        assert n1.sample_qc_metric(0.5, 0.1) == n2.sample_qc_metric(0.5, 0.1)

    def test_false_positives(self):
        n = NoiseModel(seed=0)
        fps = n.generate_false_positives(1000, 0.01)
        assert all(g.startswith("FP_GENE_") for g in fps)

    def test_quality_degradation_bounded(self):
        n = NoiseModel(seed=0)
        for _ in range(100):
            q = n.quality_degradation(0.9, [0.8, 0.7])
            assert 0.0 <= q <= 1.0


class TestOutputGenerator:
    def test_collect_sample(self):
        noise = NoiseModel(seed=1)
        gen = OutputGenerator(noise)
        s = _make_state()
        action = ExperimentAction(
            action_type=ActionType.COLLECT_SAMPLE,
            parameters={"n_samples": 4},
        )
        out = gen.generate(action, s, 1)
        assert out.output_type == OutputType.SAMPLE_COLLECTION_RESULT
        assert out.data["n_samples"] == 4

    def test_de_includes_true_genes(self):
        noise = NoiseModel(seed=42)
        gen = OutputGenerator(noise)
        s = _make_state()
        s.progress.data_normalized = True
        action = ExperimentAction(
            action_type=ActionType.DIFFERENTIAL_EXPRESSION,
            parameters={"comparison": "disease_vs_healthy"},
        )
        out = gen.generate(action, s, 5)
        assert out.output_type == OutputType.DE_RESULT
        gene_names = [g["gene"] for g in out.data["top_genes"]]
        assert "G1" in gene_names or "G2" in gene_names


class TestTransitionEngine:
    def test_progress_flags_set(self):
        noise = NoiseModel(seed=0)
        engine = TransitionEngine(noise)
        s = _make_state()
        action = ExperimentAction(action_type=ActionType.COLLECT_SAMPLE)
        result = engine.step(s, action)
        assert result.next_state.progress.samples_collected is True

    def test_hard_violation_blocks(self):
        noise = NoiseModel(seed=0)
        engine = TransitionEngine(noise)
        s = _make_state()
        result = engine.step(
            s,
            ExperimentAction(action_type=ActionType.COLLECT_SAMPLE),
            hard_violations=["test_block"],
        )
        assert result.output.success is False
        assert result.output.output_type == OutputType.FAILURE_REPORT

    def test_resource_deduction(self):
        noise = NoiseModel(seed=0)
        engine = TransitionEngine(noise)
        s = _make_state()
        action = ExperimentAction(action_type=ActionType.SEQUENCE_CELLS)
        s.progress.library_prepared = True
        result = engine.step(s, action)
        assert result.next_state.resources.budget_used == 15_000

    def test_conclusion_ends_episode(self):
        noise = NoiseModel(seed=0)
        engine = TransitionEngine(noise)
        s = _make_state()
        s.progress.de_performed = True
        action = ExperimentAction(action_type=ActionType.SYNTHESIZE_CONCLUSION)
        result = engine.step(s, action)
        assert result.done is True
