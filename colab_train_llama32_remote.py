"""Minimal Colab entrypoint for Unsloth GRPO against a remote OpenEnv Space.

This keeps the repo's prompt formatting and action parsing logic, but builds
prompt states by interacting with a deployed OpenEnv Hugging Face Space instead
of the local in-process environment. That makes the Colab workflow match the
remote environment users actually want to train against.
"""

from __future__ import annotations

import argparse
import json
import random
from typing import Any, Dict, List, Optional, Sequence

from client import BioExperimentEnv
import training_script as base

DEFAULT_MODEL_ID = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
DEFAULT_OUTPUT_DIR = "artifacts/grpo-unsloth-llama32-3b-space"
DEFAULT_SPACE_REPO_ID = "Ev3Dev/hackathon"


def hf_space_repo_to_base_url(repo_id: str) -> str:
    """Convert `owner/space-name` to the standard `hf.space` URL."""
    owner, space_name = repo_id.split("/", 1)
    normalized_owner = owner.strip().lower().replace("_", "-")
    normalized_space = space_name.strip().lower().replace("_", "-")
    return f"https://{normalized_owner}-{normalized_space}.hf.space"


def require_unsloth_base():
    # Unsloth must be imported before trl / transformers / peft.
    import unsloth  # noqa: F401
    import training_unsloth as unsloth_base

    return unsloth_base


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train Unsloth Llama 3.2 3B on a remote OpenEnv Hugging Face Space."
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset-episodes", type=int, default=8)
    parser.add_argument("--rollout-steps", type=int, default=6)
    parser.add_argument(
        "--collection-policy",
        choices=["random", "heuristic"],
        default="heuristic",
    )
    parser.add_argument("--base-url", default="")
    parser.add_argument(
        "--space-repo-id",
        default=DEFAULT_SPACE_REPO_ID,
        help="Hugging Face Space repo id, for example `Ev3Dev/hackathon`.",
    )
    parser.add_argument("--num-generations", type=int, default=2)
    parser.add_argument("--max-completion-length", type=int, default=160)
    parser.add_argument("--max-prompt-length", type=int, default=1280)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument("--plot-metric-key", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--load-model-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--disable-4bit", action="store_true")
    parser.add_argument("--lora-r", type=int, default=unsloth_defaults()["lora_r"])
    parser.add_argument(
        "--lora-alpha", type=int, default=unsloth_defaults()["lora_alpha"]
    )
    parser.add_argument(
        "--lora-dropout", type=float, default=unsloth_defaults()["lora_dropout"]
    )
    return parser


def unsloth_defaults() -> Dict[str, float]:
    return {
        "lora_r": 16,
        "lora_alpha": 16,
        "lora_dropout": 0.0,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    args = build_argument_parser().parse_args(argv)
    if not args.base_url:
        args.base_url = hf_space_repo_to_base_url(args.space_repo_id)
    return args


def make_training_args(**overrides: Any) -> argparse.Namespace:
    parser = build_argument_parser()
    defaults = vars(parser.parse_args([]))
    unknown = sorted(set(overrides) - set(defaults))
    if unknown:
        raise ValueError(f"Unknown training args: {', '.join(unknown)}")
    defaults.update(overrides)
    args = argparse.Namespace(**defaults)
    if not getattr(args, "base_url", ""):
        args.base_url = hf_space_repo_to_base_url(args.space_repo_id)
    return args


def build_remote_prompt_examples(args: argparse.Namespace) -> List[Dict[str, str]]:
    """Collect prompt states directly from the remote OpenEnv server."""
    rng = random.Random(args.seed)
    examples: List[Dict[str, str]] = []

    for _episode_idx in range(args.dataset_episodes):
        with BioExperimentEnv(base_url=args.base_url) as env:
            result = env.reset()
            obs = result.observation
            history_actions: List[base.ExperimentAction] = []

            for step_idx in range(args.rollout_steps):
                if obs.done:
                    break

                next_action = base.build_experiment_action(
                    action_type=base.pick_action(
                        args.collection_policy,
                        step_idx,
                        [action.action_type for action in history_actions],
                    ),
                    discovered_markers=obs.discovered_markers,
                    candidate_mechanisms=obs.candidate_mechanisms,
                    conditions=obs.task.conditions,
                )
                examples.append(
                    {
                        "prompt": base.build_training_prompt(obs),
                        "history_actions": json.dumps(
                            [action.model_dump() for action in history_actions]
                        ),
                        "reference_action": base.action_completion_json(next_action),
                        "problem_statement": obs.task.problem_statement,
                        "episode_tag": f"remote-{rng.randrange(10**9):09d}",
                    }
                )

                history_actions.append(next_action)
                result = env.step(next_action)
                obs = result.observation
                if result.done:
                    break

    return examples


class RemoteSpaceReward:
    """Reward function that replays each candidate against the remote Space."""

    def __init__(
        self,
        *,
        base_url: str,
        invalid_action_penalty: float = base.INVALID_ACTION_PENALTY,
        environment_error_penalty: float = base.ENVIRONMENT_ERROR_PENALTY,
    ) -> None:
        self.__name__ = "remote_space_reward"
        self.base_url = base_url
        self.invalid_action_penalty = invalid_action_penalty
        self.environment_error_penalty = environment_error_penalty

    def __call__(
        self,
        completions: List[Any],
        history_actions: Optional[List[str]] = None,
        **_: Any,
    ) -> List[float]:
        history_columns = base.normalise_column(history_actions, len(completions))
        rewards: List[float] = []

        for completion, current_history in zip(completions, history_columns):
            action = base.parse_action_completion(base.completion_to_text(completion))
            if action is None:
                rewards.append(self.invalid_action_penalty)
                continue

            try:
                rewards.append(self._score_remote(action, current_history))
            except Exception:
                rewards.append(self.environment_error_penalty)

        return rewards

    def _score_remote(
        self,
        action: base.ExperimentAction,
        history_actions: Optional[str],
    ) -> float:
        with BioExperimentEnv(base_url=self.base_url) as env:
            result = env.reset()
            obs = result.observation

            for previous_action in base.decode_history_actions(history_actions):
                result = env.step(previous_action)
                obs = result.observation
                if result.done:
                    return float(result.reward or obs.reward or 0.0)

            action = base.ensure_conclusion_claims(obs, action)
            result = env.step(action)
            if result.reward is not None:
                return float(result.reward)
            return float(result.observation.reward)


def run_dry_run_preview(
    examples: Sequence[Dict[str, str]],
    reward_fn: RemoteSpaceReward,
    output_dir: str,
    base_url: str,
) -> None:
    if not examples:
        raise ValueError("No training prompts were generated for the dry run.")

    sample = examples[0]
    sample_reward = reward_fn(
        completions=[[{"role": "assistant", "content": sample["reference_action"]}]],
        history_actions=[sample["history_actions"]],
    )[0]

    print(f"Built {len(examples)} remote prompt states.")
    print(f"Remote OpenEnv Space: {base_url}")
    print(f"Output directory: {output_dir}")
    print(f"Sample reward for reference action: {sample_reward:+.3f}")
    print("\nSample prompt:\n")
    print(sample["prompt"])


def run_training(args: argparse.Namespace) -> Dict[str, Any]:
    random.seed(args.seed)
    runtime = base.resolve_torch_runtime()
    unsloth_base = require_unsloth_base()

    if args.load_model_only:
        tokenizer, model = unsloth_base.load_model_artifacts(
            args.model_id,
            trust_remote_code=args.trust_remote_code,
            max_seq_length=args.max_seq_length,
            load_in_4bit=not args.disable_4bit,
            fast_inference=False,
            prepare_for_inference=True,
        )
        return {
            "args": args,
            "runtime": runtime,
            "tokenizer": tokenizer,
            "model": model,
        }

    examples = build_remote_prompt_examples(args)
    reward_fn = RemoteSpaceReward(base_url=args.base_url)

    if args.dry_run:
        run_dry_run_preview(examples, reward_fn, args.output_dir, args.base_url)
        return {
            "args": args,
            "runtime": runtime,
            "examples": examples,
            "reward_fn": reward_fn,
        }

    from datasets import Dataset

    FastLanguageModel = unsloth_base.patch_unsloth_grpo()
    train_dataset = Dataset.from_list(examples)

    tokenizer, model = unsloth_base.load_model_artifacts(
        args.model_id,
        trust_remote_code=args.trust_remote_code,
        max_seq_length=args.max_seq_length,
        load_in_4bit=not args.disable_4bit,
        fast_inference=False,
    )
    model = unsloth_base.apply_lora_adapters(FastLanguageModel, model, args)

    print(
        f"Training runtime: device={runtime['device']} "
        f"name={runtime['device_name']} "
        f"dtype={runtime['dtype']} "
        f"load_in_4bit={not args.disable_4bit}"
    )
    print(f"Remote OpenEnv Space: {args.base_url}")
    print(f"Collected remote prompt states: {len(examples)}")

    trainer = unsloth_base.build_unsloth_grpo_trainer(
        model=model,
        tokenizer=tokenizer,
        reward_func=reward_fn,
        train_dataset=train_dataset,
        args=args,
        runtime=runtime,
    )
    for attr in ("image_token_id", "vision_start_token_id", "vision_end_token_id"):
        if not hasattr(trainer, attr):
            setattr(trainer, attr, None)

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    plot_paths = base.save_training_plots(
        trainer.state.log_history,
        args.output_dir,
        metric_key=args.plot_metric_key,
    )
    print("Saved training plots:")
    for plot_name, plot_path in plot_paths.items():
        print(f"  - {plot_name}: {plot_path}")

    return {
        "args": args,
        "runtime": runtime,
        "examples": examples,
        "reward_fn": reward_fn,
        "train_dataset": train_dataset,
        "tokenizer": tokenizer,
        "model": model,
        "trainer": trainer,
        "plot_paths": plot_paths,
    }


def main() -> None:
    run_training(parse_args())


if __name__ == "__main__":
    main()
