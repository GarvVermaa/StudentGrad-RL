"""Train a planner with TRL GRPO and OpenEnv rewards."""

from __future__ import annotations

import argparse
import json
import random
from numbers import Real
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from client import BioExperimentEnv
from models import ActionType, ExperimentAction, ExperimentObservation
from server.hackathon_environment import BioExperimentEnvironment
from server.tasks.scenarios import SCENARIO_LIBRARY

DEFAULT_MODEL_ID = "Qwen/Qwen3.5-0.8B"
DEFAULT_OUTPUT_DIR = "training/grpo-output"
DEFAULT_BASE_URL = "http://localhost:8000"
INVALID_ACTION_PENALTY = -2.0
ENVIRONMENT_ERROR_PENALTY = -4.0

SYSTEM_PROMPT = """\
You are an expert biologist planning a single-cell experiment pipeline.

At each turn you see the experiment state and must pick the next step.

Action types (in typical order):
  collect_sample, prepare_library, sequence_cells, run_qc, filter_data,
  normalize_data, cluster_cells, differential_expression,
  pathway_enrichment, marker_selection, validate_marker, synthesize_conclusion

Other actions: select_cohort, culture_cells, perturb_gene, perturb_compound,
  integrate_batches, trajectory_analysis, regulatory_network_inference,
  design_followup_experiment, request_subagent_review

Respond with ONLY valid JSON, nothing else:
{"action_type": "...", "method": null, "parameters": {}, "justification": "...", "confidence": 0.8}
"""

HEURISTIC_SEQUENCE = [
    ActionType.COLLECT_SAMPLE,
    ActionType.PREPARE_LIBRARY,
    ActionType.SEQUENCE_CELLS,
    ActionType.RUN_QC,
    ActionType.FILTER_DATA,
    ActionType.NORMALIZE_DATA,
    ActionType.CLUSTER_CELLS,
    ActionType.DIFFERENTIAL_EXPRESSION,
    ActionType.PATHWAY_ENRICHMENT,
    ActionType.MARKER_SELECTION,
    ActionType.SYNTHESIZE_CONCLUSION,
]

VALID_ACTION_TYPES = {action.value for action in ActionType}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a GRPO policy against the OpenEnv bio experiment environment."
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset-episodes", type=int, default=8)
    parser.add_argument("--rollout-steps", type=int, default=6)
    parser.add_argument(
        "--collection-policy",
        choices=["random", "heuristic"],
        default="heuristic",
        help="Policy used to build prompt states for GRPO training.",
    )
    parser.add_argument(
        "--reward-backend",
        choices=["local", "remote"],
        default="local",
        help="Use local in-process scoring or a live OpenEnv server.",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="Base URL for the OpenEnv server when reward-backend=remote.",
    )
    parser.add_argument(
        "--scenario-name",
        action="append",
        default=None,
        help="Repeatable scenario selector. Defaults to all curated scenarios.",
    )
    parser.add_argument(
        "--domain-randomise",
        action="store_true",
        help="Enable domain randomisation while building prompts and local rewards.",
    )
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-completion-length", type=int, default=220)
    parser.add_argument("--max-prompt-length", type=int, default=768)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument(
        "--plot-metric-key",
        default=None,
        help="Optional extra metric key from trainer log history to plot.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--load-model-only",
        action="store_true",
        help="Download and load the selected model and tokenizer, then exit.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to model/tokenizer loading.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the prompt dataset and smoke-test the reward function without training.",
    )
    return parser.parse_args()


def format_observation(obs: ExperimentObservation) -> str:
    parts = [
        f"TASK: {obs.task.problem_statement}",
        f"Organism: {obs.task.organism} | Tissue: {obs.task.tissue}",
        f"Conditions: {', '.join(obs.task.conditions) or 'N/A'}",
        (
            "Step: "
            f"{obs.step_index} | Budget: ${obs.resource_usage.budget_remaining:,.0f} "
            f"| Time: {obs.resource_usage.time_remaining_days:.0f}d"
        ),
        f"Available tools: {', '.join(obs.available_tools[:8]) or 'N/A'}",
    ]
    if obs.pipeline_history:
        parts.append("History:")
        for step in obs.pipeline_history[-5:]:
            tag = "OK" if step.success else "FAIL"
            parts.append(f"  [{tag}] {step.action_type.value}: {step.output_summary[:100]}")
    if obs.rule_violations:
        parts.append(f"Violations: {obs.rule_violations}")
    if obs.discovered_markers:
        parts.append(f"Markers: {obs.discovered_markers[:5]}")
    if obs.candidate_mechanisms:
        parts.append(f"Mechanisms: {obs.candidate_mechanisms[:5]}")
    return "\n".join(parts)


def build_training_prompt(obs: ExperimentObservation) -> str:
    return f"{SYSTEM_PROMPT}\n\n{format_observation(obs)}"


def heuristic_next_action(history: Sequence[ActionType], step_index: int) -> ActionType:
    seen = set(history)
    for action in HEURISTIC_SEQUENCE:
        if action not in seen:
            return action
    if step_index >= 2 and ActionType.VALIDATE_MARKER not in seen:
        return ActionType.VALIDATE_MARKER
    return ActionType.SYNTHESIZE_CONCLUSION


def pick_action(policy: str, step_index: int, history: Sequence[ActionType]) -> ActionType:
    if policy == "random":
        return random.choice(list(ActionType))
    return heuristic_next_action(history, step_index)


def default_comparison_name(conditions: Sequence[str]) -> str:
    normalized = {condition.lower() for condition in conditions}
    if {"healthy", "ipf"} <= normalized:
        return "IPF_vs_healthy"
    if any("treated" in condition for condition in normalized) and any(
        "untreated" in condition for condition in normalized
    ):
        return "treated_vs_untreated"
    return "disease_vs_healthy"


def build_experiment_action(
    action_type: ActionType,
    discovered_markers: Sequence[str],
    conditions: Sequence[str],
) -> ExperimentAction:
    method = None
    parameters: Dict[str, object] = {}
    justification = f"Advance the experiment with {action_type.value}."

    if action_type == ActionType.COLLECT_SAMPLE:
        parameters = {"n_samples": 6}
        justification = "Collect enough samples to start the experiment."
    elif action_type == ActionType.PREPARE_LIBRARY:
        method = "10x_chromium"
        justification = "Prepare a single-cell library for sequencing."
    elif action_type == ActionType.SEQUENCE_CELLS:
        method = "NovaSeq"
        justification = "Generate reads for downstream single-cell analysis."
    elif action_type == ActionType.RUN_QC:
        method = "scanpy.pp.calculate_qc_metrics"
        justification = "Measure technical quality before filtering."
    elif action_type == ActionType.FILTER_DATA:
        method = "scanpy.pp.filter_cells"
        justification = "Remove low-quality cells and technical artifacts."
    elif action_type == ActionType.NORMALIZE_DATA:
        method = "scanpy.pp.normalize_total"
        justification = "Normalize counts for comparable expression profiles."
    elif action_type == ActionType.CLUSTER_CELLS:
        method = "scanpy.tl.leiden"
        justification = "Resolve cell states before interpretation."
    elif action_type == ActionType.DIFFERENTIAL_EXPRESSION:
        method = "scanpy.tl.rank_genes_groups"
        parameters = {"comparison": default_comparison_name(conditions)}
        justification = "Identify genes associated with the phenotype of interest."
    elif action_type == ActionType.TRAJECTORY_ANALYSIS:
        method = "scanpy.tl.dpt"
        justification = "Recover pseudotime and lineage structure."
    elif action_type == ActionType.PATHWAY_ENRICHMENT:
        method = "gseapy.prerank"
        justification = "Translate gene-level changes into pathway programs."
    elif action_type == ActionType.MARKER_SELECTION:
        method = "scanpy.tl.rank_genes_groups"
        justification = "Nominate marker genes for validation."
    elif action_type == ActionType.VALIDATE_MARKER:
        method = "qPCR"
        parameters = {"marker": discovered_markers[0] if discovered_markers else "SPP1"}
        justification = "Validate the strongest discovered marker."
    elif action_type == ActionType.SYNTHESIZE_CONCLUSION:
        parameters = {"claims": []}
        justification = "Summarize the current evidence into a conclusion."

    return ExperimentAction(
        action_type=action_type,
        method=method,
        parameters=parameters,
        justification=justification,
        confidence=0.75,
    )


def selected_scenarios(requested: Optional[Sequence[str]]) -> List[str]:
    available = [scenario.name for scenario in SCENARIO_LIBRARY]
    if not requested:
        return available
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise ValueError(f"Unknown scenarios requested: {', '.join(unknown)}")
    return list(requested)


def action_completion_json(action: ExperimentAction) -> str:
    payload = {
        "action_type": action.action_type.value,
        "method": action.method,
        "parameters": action.parameters,
        "justification": action.justification,
        "confidence": action.confidence,
    }
    return json.dumps(payload)


def build_prompt_examples(
    *,
    dataset_episodes: int,
    rollout_steps: int,
    collection_policy: str,
    scenario_names: Sequence[str],
    seed: int,
    domain_randomise: bool,
) -> List[Dict[str, str]]:
    rng = random.Random(seed)
    examples: List[Dict[str, str]] = []
    scenario_cycle = list(scenario_names)
    rng.shuffle(scenario_cycle)

    for episode_idx in range(dataset_episodes):
        scenario_name = scenario_cycle[episode_idx % len(scenario_cycle)]
        env = BioExperimentEnvironment(
            scenario_name=scenario_name,
            domain_randomise=domain_randomise,
        )
        obs = env.reset()
        history_actions: List[ExperimentAction] = []

        for step_idx in range(rollout_steps):
            if obs.done:
                break

            next_action = build_experiment_action(
                action_type=pick_action(
                    collection_policy,
                    step_idx,
                    [action.action_type for action in history_actions],
                ),
                discovered_markers=obs.discovered_markers,
                conditions=obs.task.conditions,
            )
            examples.append({
                "prompt": build_training_prompt(obs),
                "scenario_name": scenario_name,
                "history_actions": json.dumps(
                    [action.model_dump() for action in history_actions]
                ),
                "reference_action": action_completion_json(next_action),
                "problem_statement": obs.task.problem_statement,
            })

            history_actions.append(next_action)
            obs = env.step(next_action)

    return examples


def completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion.strip()
    if isinstance(completion, dict):
        return content_to_text(completion.get("content", ""))
    if isinstance(completion, list):
        for item in reversed(completion):
            if isinstance(item, dict) and "content" in item:
                text = content_to_text(item["content"])
                if text:
                    return text
            if isinstance(item, str) and item.strip():
                return item.strip()
    return str(completion).strip()


def content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                if isinstance(part.get("text"), str):
                    parts.append(part["text"])
                elif isinstance(part.get("content"), str):
                    parts.append(part["content"])
        return "".join(parts).strip()
    return str(content).strip()


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    stripped = text.strip()
    fence_prefix = "```"
    if stripped.startswith(fence_prefix) and stripped.endswith(fence_prefix):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            stripped = "\n".join(lines[1:-1]).strip()

    candidates = [stripped]
    start = stripped.find("{")
    while start != -1:
        depth = 0
        for idx in range(start, len(stripped)):
            char = stripped[idx]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(stripped[start:idx + 1])
                    break
        start = stripped.find("{", start + 1)

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def parse_action_completion(text: str) -> Optional[ExperimentAction]:
    payload = extract_json_object(text)
    if payload is None:
        return None

    action_type = payload.get("action_type")
    if action_type not in VALID_ACTION_TYPES:
        return None

    parameters = payload.get("parameters") or {}
    if not isinstance(parameters, dict):
        parameters = {}

    confidence = payload.get("confidence", 0.5)
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.5

    return ExperimentAction(
        action_type=ActionType(action_type),
        method=payload.get("method"),
        parameters=parameters,
        justification=payload.get("justification"),
        confidence=min(1.0, max(0.0, confidence)),
    )


def decode_history_actions(history_actions: Optional[str]) -> List[ExperimentAction]:
    if not history_actions:
        return []
    raw_actions = json.loads(history_actions)
    return [
        ExperimentAction(**action_payload)
        for action_payload in raw_actions
        if isinstance(action_payload, dict)
    ]


def normalise_column(values: Any, length: int) -> List[Any]:
    if values is None:
        return [None] * length
    if isinstance(values, list):
        if len(values) == length:
            return values
        if len(values) == 1:
            return values * length
        return values[:length] + [None] * max(0, length - len(values))
    return [values] * length


class OpenEnvReward:
    """Reward function compatible with TRL GRPOTrainer."""

    def __init__(
        self,
        *,
        reward_backend: str,
        base_url: str,
        invalid_action_penalty: float = INVALID_ACTION_PENALTY,
        environment_error_penalty: float = ENVIRONMENT_ERROR_PENALTY,
        domain_randomise: bool = False,
    ) -> None:
        self.reward_backend = reward_backend
        self.base_url = base_url
        self.invalid_action_penalty = invalid_action_penalty
        self.environment_error_penalty = environment_error_penalty
        self.domain_randomise = domain_randomise

    def __call__(
        self,
        completions: List[Any],
        scenario_name: Optional[List[str]] = None,
        history_actions: Optional[List[str]] = None,
        **_: Any,
    ) -> List[float]:
        scenario_names = normalise_column(scenario_name, len(completions))
        history_columns = normalise_column(history_actions, len(completions))
        rewards: List[float] = []

        for completion, current_scenario, current_history in zip(
            completions,
            scenario_names,
            history_columns,
        ):
            action = parse_action_completion(completion_to_text(completion))
            if action is None:
                rewards.append(self.invalid_action_penalty)
                continue

            try:
                if self.reward_backend == "remote":
                    reward = self._score_remote(action, current_history)
                else:
                    reward = self._score_local(action, current_scenario, current_history)
            except Exception:
                reward = self.environment_error_penalty

            rewards.append(float(reward))

        return rewards

    def _score_local(
        self,
        action: ExperimentAction,
        scenario_name: Optional[str],
        history_actions: Optional[str],
    ) -> float:
        env = BioExperimentEnvironment(
            scenario_name=scenario_name,
            domain_randomise=self.domain_randomise,
        )
        obs = env.reset()
        for previous_action in decode_history_actions(history_actions):
            obs = env.step(previous_action)
            if obs.done:
                return float(obs.reward)
        obs = env.step(action)
        return float(obs.reward)

    def _score_remote(
        self,
        action: ExperimentAction,
        history_actions: Optional[str],
    ) -> float:
        with BioExperimentEnv(base_url=self.base_url) as env:
            result = env.reset()
            for previous_action in decode_history_actions(history_actions):
                result = env.step(previous_action)
                if result.done:
                    return float(result.reward or 0.0)
            result = env.step(action)
            if result.reward is not None:
                return float(result.reward)
            return float(result.observation.reward)


def is_numeric_log_value(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def available_numeric_log_keys(log_history: Sequence[Dict[str, Any]]) -> List[str]:
    keys = {
        key
        for entry in log_history
        if isinstance(entry, dict)
        for key, value in entry.items()
        if key != "step" and is_numeric_log_value(value)
    }
    return sorted(keys)


def extract_log_series(
    log_history: Sequence[Dict[str, Any]],
    key: Optional[str],
) -> List[Tuple[float, float]]:
    if not key:
        return []

    series: List[Tuple[float, float]] = []
    synthetic_step = 0
    for entry in log_history:
        if not isinstance(entry, dict) or key not in entry:
            continue
        value = entry.get(key)
        if not is_numeric_log_value(value):
            continue

        raw_step = entry.get("step")
        if is_numeric_log_value(raw_step):
            step = float(raw_step)
        else:
            synthetic_step += 1
            step = float(synthetic_step)

        series.append((step, float(value)))

    return series


def select_reward_key(log_history: Sequence[Dict[str, Any]]) -> Optional[str]:
    numeric_keys = available_numeric_log_keys(log_history)
    reward_keys = [key for key in numeric_keys if "reward" in key.lower()]
    if not reward_keys:
        return None

    preferred = [
        "reward",
        "mean_reward",
        "reward_mean",
        "rewards/open_env_reward",
    ]
    lowered = {key.lower(): key for key in reward_keys}
    for key in preferred:
        if key in lowered:
            return lowered[key]

    reward_keys.sort(key=lambda key: ("/" in key, len(key), key))
    return reward_keys[0]


def select_metric_key(
    log_history: Sequence[Dict[str, Any]],
    *,
    reward_key: Optional[str],
    requested_key: Optional[str] = None,
) -> Optional[str]:
    numeric_keys = available_numeric_log_keys(log_history)
    if requested_key:
        if requested_key not in numeric_keys:
            available = ", ".join(numeric_keys) or "none"
            raise ValueError(
                f"Requested plot metric '{requested_key}' was not logged. "
                f"Available numeric keys: {available}"
            )
        return requested_key

    excluded = {
        "epoch",
        "loss",
        "learning_rate",
        "step",
        "total_flos",
        "train_loss",
        "train_runtime",
        "train_samples_per_second",
        "train_steps_per_second",
    }
    if reward_key:
        excluded.add(reward_key)

    preferred = [
        "kl",
        "objective/kl",
        "completion_length",
        "mean_completion_length",
        "grad_norm",
        "entropy",
        "accuracy",
        "learning_rate",
        "epoch",
    ]
    numeric_set = set(numeric_keys)
    for key in preferred:
        if key in numeric_set and key not in excluded:
            return key

    candidates = [
        key for key in numeric_keys
        if key not in excluded and "reward" not in key.lower()
    ]
    if candidates:
        return candidates[0]

    for fallback in ("learning_rate", "epoch"):
        if fallback in numeric_set:
            return fallback

    return None


def save_plot(
    path: Path,
    *,
    series: Sequence[Tuple[float, float]],
    title: str,
    ylabel: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    if series:
        x_values, y_values = zip(*series)
        ax.plot(x_values, y_values, marker="o", linewidth=1.8)
    else:
        ax.text(
            0.5,
            0.5,
            "No logged data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_training_plots(
    log_history: Sequence[Dict[str, Any]],
    output_dir: str | Path,
    metric_key: Optional[str] = None,
) -> Dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    reward_key = select_reward_key(log_history)
    selected_metric_key = select_metric_key(
        log_history,
        reward_key=reward_key,
        requested_key=metric_key,
    )

    loss_series = extract_log_series(log_history, "loss")
    reward_series = extract_log_series(log_history, reward_key)
    metric_series = extract_log_series(log_history, selected_metric_key)

    loss_path = output_path / "training_loss.png"
    reward_path = output_path / "training_reward.png"
    metric_path = output_path / "training_metric.png"
    dashboard_path = output_path / "training_dashboard.png"
    manifest_path = output_path / "training_plot_manifest.json"

    save_plot(loss_path, series=loss_series, title="Training Loss", ylabel="Loss")
    save_plot(
        reward_path,
        series=reward_series,
        title=f"Training Reward ({reward_key or 'not logged'})",
        ylabel="Reward",
    )
    save_plot(
        metric_path,
        series=metric_series,
        title=f"Training Metric ({selected_metric_key or 'not logged'})",
        ylabel=selected_metric_key or "Metric",
    )

    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    plot_specs = [
        (axes[0], loss_series, "Training Loss", "Loss"),
        (axes[1], reward_series, f"Training Reward ({reward_key or 'not logged'})", "Reward"),
        (
            axes[2],
            metric_series,
            f"Training Metric ({selected_metric_key or 'not logged'})",
            selected_metric_key or "Metric",
        ),
    ]
    for axis, series, title, ylabel in plot_specs:
        if series:
            x_values, y_values = zip(*series)
            axis.plot(x_values, y_values, marker="o", linewidth=1.8)
        else:
            axis.text(
                0.5,
                0.5,
                "No logged data available",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
        axis.set_title(title)
        axis.set_xlabel("Step")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(dashboard_path, dpi=150)
    plt.close(fig)

    manifest = {
        "available_numeric_keys": available_numeric_log_keys(log_history),
        "reward_key": reward_key,
        "metric_key": selected_metric_key,
        "plots": {
            "loss": str(loss_path),
            "reward": str(reward_path),
            "metric": str(metric_path),
            "dashboard": str(dashboard_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest["plots"]


def run_dry_run_preview(
    examples: Sequence[Dict[str, str]],
    reward_fn: OpenEnvReward,
    output_dir: str,
) -> None:
    if not examples:
        raise ValueError("No training prompts were generated for the dry run.")

    sample = examples[0]
    sample_reward = reward_fn(
        completions=[[{"role": "assistant", "content": sample["reference_action"]}]],
        scenario_name=[sample["scenario_name"]],
        history_actions=[sample["history_actions"]],
    )[0]

    print(f"Built {len(examples)} prompt states.")
    print(f"Output directory: {Path(output_dir)}")
    print(f"Sample scenario: {sample['scenario_name']}")
    print(f"Sample reward for reference action: {sample_reward:+.3f}")
    print("\nSample prompt:\n")
    print(sample["prompt"])


def resolve_torch_runtime() -> Dict[str, Any]:
    import torch

    use_cuda = torch.cuda.is_available()
    bf16 = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)()) if use_cuda else False
    dtype = torch.bfloat16 if bf16 else (
        torch.float16 if use_cuda else torch.float32
    )
    return {
        "use_cuda": use_cuda,
        "device": "cuda:0" if use_cuda else "cpu",
        "dtype": dtype,
        "bf16": bf16,
        "fp16": use_cuda and not bf16,
        "device_name": torch.cuda.get_device_name(0) if use_cuda else "cpu",
    }


def load_model_artifacts(
    model_id: str,
    *,
    trust_remote_code: bool,
):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    runtime = resolve_torch_runtime()
    print(f"Loading tokenizer for {model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model for {model_id} ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
        dtype=runtime["dtype"],
    )
    if runtime["use_cuda"]:
        model = model.to(runtime["device"])
    if not runtime["use_cuda"]:
        model = model.to("cpu")
    return tokenizer, model


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    runtime = resolve_torch_runtime()

    if args.load_model_only:
        tokenizer, model = load_model_artifacts(
            args.model_id,
            trust_remote_code=args.trust_remote_code,
        )
        device = getattr(model, "device", "unknown")
        print(f"Model ready: {args.model_id}")
        print(f"Tokenizer vocab size: {len(tokenizer)}")
        print(f"Model device: {device}")
        print(f"Runtime device name: {runtime['device_name']}")
        return

    scenario_names = selected_scenarios(args.scenario_name)
    examples = build_prompt_examples(
        dataset_episodes=args.dataset_episodes,
        rollout_steps=args.rollout_steps,
        collection_policy=args.collection_policy,
        scenario_names=scenario_names,
        seed=args.seed,
        domain_randomise=args.domain_randomise,
    )
    reward_fn = OpenEnvReward(
        reward_backend=args.reward_backend,
        base_url=args.base_url,
        domain_randomise=args.domain_randomise,
    )

    if args.dry_run:
        run_dry_run_preview(examples, reward_fn, args.output_dir)
        return

    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    train_dataset = Dataset.from_list(examples)
    tokenizer, model = load_model_artifacts(
        args.model_id,
        trust_remote_code=args.trust_remote_code,
    )
    config = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        bf16=runtime["bf16"],
        fp16=runtime["fp16"],
        report_to="none",
        remove_unused_columns=False,
    )

    print(
        f"Training runtime: device={runtime['device']} "
        f"name={runtime['device_name']} "
        f"dtype={runtime['dtype']}"
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    plot_paths = save_training_plots(
        trainer.state.log_history,
        args.output_dir,
        metric_key=args.plot_metric_key,
    )
    print("Saved training plots:")
    for plot_name, plot_path in plot_paths.items():
        print(f"  - {plot_name}: {plot_path}")


if __name__ == "__main__":
    main()
