"""Tests for literature-grounded benchmark utilities."""

from training.literature_benchmark import (
    run_paper_benchmark,
    select_literature_scenario,
)


def test_select_literature_scenario_for_ipf_prompt():
    scenario = select_literature_scenario(
        "Validate SPP1-positive macrophage findings in idiopathic pulmonary fibrosis."
    )
    assert scenario.name == "biomarker_validation_lung"


def test_select_literature_scenario_for_trajectory_prompt():
    scenario = select_literature_scenario(
        "Recover branching hematopoietic lineages and branch point transcription factors."
    )
    assert scenario.name == "hematopoiesis_trajectory"


def test_run_paper_benchmark_matches_curated_findings():
    result = run_paper_benchmark(
        problem_statement=(
            "Design a follow-up validation experiment for candidate biomarker "
            "SPP1 in idiopathic pulmonary fibrosis."
        ),
        scenario_name="biomarker_validation_lung",
        domain_randomise=False,
    )

    assert result.total_steps >= 1
    assert result.matched_papers
    assert result.match_ratio >= (2 / 3)
    assert any("SPP1" in finding for finding in result.matched_findings)
