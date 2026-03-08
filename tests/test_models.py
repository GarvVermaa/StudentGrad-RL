"""Tests for POMDP schema models."""

import pytest
from models import (
    ActionType,
    ConclusionClaim,
    ExpectedFinding,
    ExperimentAction,
    ExperimentObservation,
    IntermediateOutput,
    OutputType,
    PaperReference,
    PipelineStepRecord,
    ResourceUsage,
    TaskSpec,
)


def test_experiment_action_roundtrip():
    a = ExperimentAction(
        action_type=ActionType.COLLECT_SAMPLE,
        input_targets=["prior_cohort"],
        method="10x_chromium",
        parameters={"n_samples": 6},
        confidence=0.8,
    )
    d = a.model_dump()
    assert d["action_type"] == "collect_sample"
    assert d["confidence"] == 0.8
    reconstructed = ExperimentAction(**d)
    assert reconstructed.action_type == ActionType.COLLECT_SAMPLE


def test_experiment_observation_defaults():
    obs = ExperimentObservation(done=False, reward=0.0)
    assert obs.step_index == 0
    assert obs.pipeline_history == []
    assert obs.resource_usage.budget_remaining == 100_000.0


def test_intermediate_output_quality_bounds():
    with pytest.raises(Exception):
        IntermediateOutput(
            output_type=OutputType.QC_METRICS,
            step_index=1,
            quality_score=1.5,
        )


def test_task_spec_defaults():
    t = TaskSpec()
    assert "10x_chromium" in t.available_assays
    assert t.budget_limit == 100_000.0
    assert t.paper_references == []
    assert t.expected_findings == []


def test_paper_reference_and_expected_finding_roundtrip():
    task = TaskSpec(
        paper_references=[
            PaperReference(
                title="Example paper",
                doi="10.0000/example",
            )
        ],
        expected_findings=[
            ExpectedFinding(
                finding="Example marker is enriched",
                category="marker",
                keywords=["EXAMPLE"],
            )
        ],
    )
    dumped = task.model_dump()
    assert dumped["paper_references"][0]["title"] == "Example paper"
    assert dumped["expected_findings"][0]["category"] == "marker"


def test_conclusion_claim_serialization():
    c = ConclusionClaim(
        claim="NPPA is upregulated in disease",
        evidence_steps=[3, 5],
        confidence=0.85,
        claim_type="correlational",
    )
    d = c.model_dump()
    assert d["claim_type"] == "correlational"
    assert d["confidence"] == 0.85
