"""Tests for GRPO training helpers."""

from pathlib import Path

from models import ActionType
from training_script import (
    INVALID_ACTION_PENALTY,
    OpenEnvReward,
    available_numeric_log_keys,
    build_prompt_examples,
    completion_to_text,
    parse_action_completion,
    save_training_plots,
    select_metric_key,
    select_reward_key,
)


def test_completion_to_text_from_chat_messages():
    completion = [
        {"role": "assistant", "content": '{"action_type":"collect_sample"}'}
    ]
    assert completion_to_text(completion) == '{"action_type":"collect_sample"}'


def test_parse_action_completion_roundtrip():
    action = parse_action_completion(
        '{"action_type":"run_qc","method":"scanpy.pp.calculate_qc_metrics",'
        '"parameters":{"min_genes":200},"confidence":0.8}'
    )
    assert action is not None
    assert action.action_type == ActionType.RUN_QC
    assert action.method == "scanpy.pp.calculate_qc_metrics"
    assert action.parameters["min_genes"] == 200
    assert action.confidence == 0.8


def test_parse_action_completion_accepts_reasoning_alias():
    action = parse_action_completion(
        '{"action_type":"run_qc","reasoning":"Measure quality before filtering."}'
    )
    assert action is not None
    assert action.justification == "Measure quality before filtering."


def test_build_prompt_examples_contains_reference_action():
    examples = build_prompt_examples(
        dataset_episodes=1,
        rollout_steps=2,
        collection_policy="heuristic",
        scenario_names=["cardiac_disease_de"],
        seed=0,
        domain_randomise=False,
    )
    assert len(examples) == 2
    assert examples[0]["scenario_name"] == "cardiac_disease_de"
    assert '"action_type": "collect_sample"' in examples[0]["reference_action"]


def test_openenv_reward_penalizes_invalid_completion():
    reward_fn = OpenEnvReward(
        reward_backend="local",
        base_url="http://localhost:8000",
    )
    rewards = reward_fn(
        completions=[[{"role": "assistant", "content": "not valid json"}]],
        scenario_name=["cardiac_disease_de"],
        history_actions=["[]"],
    )
    assert rewards == [INVALID_ACTION_PENALTY]


def test_openenv_reward_scores_valid_completion_locally():
    examples = build_prompt_examples(
        dataset_episodes=1,
        rollout_steps=1,
        collection_policy="heuristic",
        scenario_names=["cardiac_disease_de"],
        seed=0,
        domain_randomise=False,
    )
    reward_fn = OpenEnvReward(
        reward_backend="local",
        base_url="http://localhost:8000",
    )
    sample = examples[0]
    rewards = reward_fn(
        completions=[[{"role": "assistant", "content": sample["reference_action"]}]],
        scenario_name=[sample["scenario_name"]],
        history_actions=[sample["history_actions"]],
    )
    assert len(rewards) == 1
    assert rewards[0] > 0.0


def test_log_key_selection_prefers_reward_and_metric_keys():
    log_history = [
        {"step": 1, "loss": 1.2, "rewards/open_env_reward": 0.4, "objective/kl": 0.05},
        {"step": 2, "loss": 1.0, "rewards/open_env_reward": 0.6, "objective/kl": 0.04},
    ]
    assert available_numeric_log_keys(log_history) == [
        "loss",
        "objective/kl",
        "rewards/open_env_reward",
    ]
    reward_key = select_reward_key(log_history)
    assert reward_key == "rewards/open_env_reward"
    assert select_metric_key(log_history, reward_key=reward_key) == "objective/kl"


def test_save_training_plots_writes_expected_files(tmp_path):
    log_history = [
        {"step": 1, "loss": 1.2, "reward": 0.4, "grad_norm": 0.8},
        {"step": 2, "loss": 0.9, "reward": 0.7, "grad_norm": 0.5},
    ]
    plot_paths = save_training_plots(log_history, tmp_path, metric_key="grad_norm")

    assert set(plot_paths) == {"loss", "reward", "metric", "dashboard"}
    for plot_path in plot_paths.values():
        assert Path(plot_path).exists()

    manifest_path = tmp_path / "training_plot_manifest.json"
    assert manifest_path.exists()
