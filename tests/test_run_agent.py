"""Tests for run_agent parser and fallback helpers."""

from models import ActionType, ExperimentAction
from run_agent import ensure_conclusion_claims, extract_json_object, parse_action, should_block_failed_reattempt
from server.hackathon_environment import BioExperimentEnvironment


def test_parse_action_accepts_reasoning_variant():
    action = parse_action(
        '{"action_type":"run_qc","parameters":{},"Reasoning":"check quality","confidence":0.8}'
    )
    assert action is not None
    assert action.action_type == ActionType.RUN_QC
    assert action.justification == "check quality"


def test_parse_action_accepts_justifyement_typo():
    action = parse_action(
        '{"action_type":"collect_sample","parameters":{},"justifyement":"typo key","confidence":0.7}'
    )
    assert action is not None
    assert action.action_type == ActionType.COLLECT_SAMPLE
    assert action.justification == "typo key"


def test_extract_json_object_unwraps_quoted_json_string():
    parsed = extract_json_object(
        '"{\\"action_type\\": \\"run_qc\\", \\"method\\": \\"\\", \\"parameters\\": {}, \\"Justification\\": \\"check quality\\", \\"confidence\\": 0.8}"'
    )
    assert parsed is not None
    assert parsed["action_type"] == "run_qc"


def test_parse_action_falls_back_when_inner_object_lacks_action_type():
    action = parse_action(
        '"{\\"action_type\\": \\"design_followup_experiment\\", \\"method\\": \\"\\", \\"parameters\\": {\\"criterion_description\\": \\"\\"}, \\"Justification\\": \\"follow-up\\", \\"confidence\\": 0.6, \\"threshold_value\\": {\\"conditions\\": [], \\"gene_filter_criteria\\": \\"x\\", \\"sample_group_size\\": 3}}"'  # noqa: E501
    )
    assert action is not None
    assert action.action_type == ActionType.DESIGN_FOLLOWUP


def test_should_block_failed_reattempt_until_pipeline_progress():
    env = BioExperimentEnvironment(scenario_name="cardiac_disease_de", domain_randomise=False)
    obs = env.reset(seed=0)
    for action_type in (
        ActionType.COLLECT_SAMPLE,
        ActionType.PREPARE_LIBRARY,
        ActionType.SEQUENCE_CELLS,
    ):
        obs = env.step(ExperimentAction(action_type=action_type))
    assert should_block_failed_reattempt(obs.pipeline_history, ActionType.SEQUENCE_CELLS) is False
    assert should_block_failed_reattempt(obs.pipeline_history, ActionType.RUN_QC) is False


def test_ensure_conclusion_claims_infers_from_outputs_when_discoveries_empty():
    env = BioExperimentEnvironment(scenario_name="cardiac_disease_de", domain_randomise=False)
    obs = env.reset(seed=0)
    pipeline = [
        ExperimentAction(action_type=ActionType.COLLECT_SAMPLE),
        ExperimentAction(action_type=ActionType.PREPARE_LIBRARY),
        ExperimentAction(action_type=ActionType.SEQUENCE_CELLS),
        ExperimentAction(action_type=ActionType.RUN_QC),
        ExperimentAction(action_type=ActionType.FILTER_DATA),
        ExperimentAction(action_type=ActionType.NORMALIZE_DATA),
        ExperimentAction(action_type=ActionType.CLUSTER_CELLS),
        ExperimentAction(
            action_type=ActionType.DIFFERENTIAL_EXPRESSION,
            parameters={"comparison": "disease_vs_healthy"},
        ),
        ExperimentAction(action_type=ActionType.PATHWAY_ENRICHMENT),
    ]
    for action in pipeline:
        obs = env.step(action)

    sparse_obs = obs.model_copy(update={
        "discovered_markers": [],
        "candidate_mechanisms": [],
    })
    action = ensure_conclusion_claims(
        sparse_obs,
        ExperimentAction(
            action_type=ActionType.SYNTHESIZE_CONCLUSION,
            confidence=0.9,
            parameters={},
        ),
    )

    claims = action.parameters["claims"]
    assert claims[0]["top_markers"]
    assert claims[0]["causal_mechanisms"]
    assert claims[0]["predicted_pathways"]
