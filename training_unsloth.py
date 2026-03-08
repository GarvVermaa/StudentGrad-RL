"""Train and run quantized self-driving lab models with Unsloth.

This keeps the same OpenEnv prompt + reward wiring as `training_script.py`,
but arranges the Unsloth path in the more typical pattern:
1. patch GRPO support
2. load a quantized model
3. apply LoRA adapters
4. train with an explicit OpenEnv reward function

NOTE: Unsloth must be imported before trl, transformers, peft. Import this
module before training_script.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

# Unsloth must be imported before trl/transformers/peft for optimizations.
import unsloth  # noqa: F401

import training_script as base

DEFAULT_OUTPUT_DIR = "training/grpo-unsloth-output"
DEFAULT_MAX_SEQ_LENGTH = 2048
DEFAULT_LORA_R = 16
DEFAULT_LORA_ALPHA = 16
DEFAULT_LORA_DROPOUT = 0.0
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def require_unsloth():
    try:
        from unsloth import FastLanguageModel, PatchFastRL
    except ImportError as exc:
        msg = str(exc)
        if "vllm.lora" in msg or "vllm" in msg.lower():
            raise RuntimeError(
                f"Unsloth failed: {exc}. "
                "unsloth_zoo expects vllm.lora.models. Install a compatible vllm:\n"
                "  pip install 'vllm==0.8.2'   # requires torch 2.6\n"
                "  pip install 'vllm==0.7.3'    # alternative\n"
                "If torch>=2.10 conflicts, use a separate env with torch 2.6–2.8."
            ) from exc
        if "unsloth" in msg.lower():
            raise RuntimeError(
                "Unsloth is not installed. Run `uv sync` or `pip install unsloth`."
            ) from exc
        raise RuntimeError(f"Failed to import Unsloth: {exc}") from exc
    return FastLanguageModel, PatchFastRL


def _call_unsloth_from_pretrained(FastLanguageModel, **kwargs: Any):
    for optional_key in ("fast_inference", "trust_remote_code"):
        try:
            return FastLanguageModel.from_pretrained(**kwargs)
        except TypeError as exc:
            if optional_key in kwargs and optional_key in str(exc):
                kwargs = dict(kwargs)
                kwargs.pop(optional_key, None)
                continue
            raise
    return FastLanguageModel.from_pretrained(**kwargs)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = base.build_argument_parser()
    parser.description = (
        "Train a GRPO policy with Unsloth quantized loading for faster H100 runs."
    )
    parser.set_defaults(output_dir=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=DEFAULT_MAX_SEQ_LENGTH,
        help="Context length passed to Unsloth model loading.",
    )
    parser.add_argument(
        "--disable-4bit",
        action="store_true",
        help="Disable 4-bit quantized loading and use the wider base weights.",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=DEFAULT_LORA_R,
        help="LoRA rank used for the quantized GRPO policy.",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=DEFAULT_LORA_ALPHA,
        help="LoRA alpha used for the quantized GRPO policy.",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=DEFAULT_LORA_DROPOUT,
        help="LoRA dropout used for the quantized GRPO policy.",
    )
    parser.add_argument(
        "--save-merged-16bit",
        action="store_true",
        help="Also export a merged 16-bit model after training if supported.",
    )
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    return build_argument_parser().parse_args(argv)


def make_training_args(**overrides: Any) -> argparse.Namespace:
    parser = build_argument_parser()
    defaults = vars(parser.parse_args([]))
    unknown = sorted(set(overrides) - set(defaults))
    if unknown:
        raise ValueError(f"Unknown training args: {', '.join(unknown)}")
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def load_model_artifacts(
    model_id: str,
    *,
    trust_remote_code: bool,
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    load_in_4bit: bool = True,
    fast_inference: bool = False,
    prepare_for_inference: bool = False,
):
    FastLanguageModel, _ = require_unsloth()
    runtime = base.resolve_torch_runtime()

    print(f"Loading Unsloth tokenizer+model for {model_id} ...")
    model, tokenizer = _call_unsloth_from_pretrained(
        FastLanguageModel,
        model_name=model_id,
        max_seq_length=max_seq_length,
        dtype=runtime["dtype"],
        load_in_4bit=load_in_4bit,
        fast_inference=fast_inference,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    if prepare_for_inference:
        try:
            FastLanguageModel.for_inference(model)
        except AttributeError:
            pass

    device = getattr(model, "device", None)
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = runtime["device"]
    print(f"Loaded model on device: {device}")
    return tokenizer, model


def build_openenv_reward(args: argparse.Namespace) -> base.OpenEnvReward:
    """Return the OpenEnv-compatible reward callable used by GRPO."""
    return base.OpenEnvReward(
        reward_backend=args.reward_backend,
        base_url=args.base_url,
        domain_randomise=args.domain_randomise,
    )


def prepare_prompt_examples(args: argparse.Namespace) -> Dict[str, Any]:
    """Build the OpenEnv rollout states that seed GRPO prompts."""
    scenario_names = base.selected_scenarios(args.scenario_name)
    examples = base.build_prompt_examples(
        dataset_episodes=args.dataset_episodes,
        rollout_steps=args.rollout_steps,
        collection_policy=args.collection_policy,
        scenario_names=scenario_names,
        seed=args.seed,
        domain_randomise=args.domain_randomise,
    )
    return {
        "scenario_names": scenario_names,
        "examples": examples,
    }


def patch_unsloth_grpo():
    """Patch TRL GRPO to use Unsloth's optimized kernels."""
    FastLanguageModel, PatchFastRL = require_unsloth()
    PatchFastRL("GRPO", FastLanguageModel)
    return FastLanguageModel


def apply_lora_adapters(FastLanguageModel, model: Any, args: argparse.Namespace) -> Any:
    """Apply LoRA adapters in the usual Unsloth configuration style."""
    return FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=LORA_TARGET_MODULES,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=args.seed,
    )


def build_grpo_config(
    args: argparse.Namespace,
    runtime: Dict[str, Any],
):
    import inspect

    base._guard_invalid_torchao_version()
    base._guard_partial_vllm_install()
    from trl import GRPOConfig

    supported_params = set(inspect.signature(GRPOConfig.__init__).parameters)
    config_kwargs = {
        "output_dir": args.output_dir,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_generations": args.num_generations,
        "max_completion_length": args.max_completion_length,
        "num_train_epochs": args.num_train_epochs,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "bf16": runtime["bf16"],
        "fp16": runtime["fp16"],
        "report_to": "none",
        "remove_unused_columns": False,
    }
    # Only add max_prompt_length if this TRL version supports it; UnslothGRPOTrainer can
    # fail when passing it to parent, so we only pass when explicitly supported.
    if "max_prompt_length" in supported_params:
        config_kwargs["max_prompt_length"] = None  # text-only; avoids image_token_id crash
    if (
        "max_length" in supported_params
        and "max_prompt_length" not in supported_params
        and "max_completion_length" not in supported_params
    ):
        config_kwargs["max_length"] = getattr(args, "max_prompt_length", 1024) + args.max_completion_length
    filtered_kwargs = {k: v for k, v in config_kwargs.items() if k in supported_params}
    skipped = sorted(set(config_kwargs) - set(filtered_kwargs))
    if skipped:
        print(f"GRPOConfig compatibility: skipping unsupported fields {', '.join(skipped)}")
    return GRPOConfig(**filtered_kwargs)


def build_unsloth_grpo_trainer(
    *,
    model: Any,
    tokenizer: Any,
    reward_func: Any,
    train_dataset: Any,
    args: argparse.Namespace,
    runtime: Dict[str, Any],
):
    base._guard_invalid_torchao_version()
    base._guard_partial_vllm_install()
    from trl import GRPOTrainer

    config = build_grpo_config(args, runtime)
    return GRPOTrainer(
        model=model,
        reward_funcs=reward_func,
        args=config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )


def generate_action_with_model(
    model: Any,
    tokenizer: Any,
    prompt_or_observation: str | base.ExperimentObservation,
    *,
    max_new_tokens: int = base.DEFAULT_COMPLETION_TOKEN_BUDGET,
    temperature: float = 0.2,
    top_p: float = 0.9,
    do_sample: bool = True,
) -> Dict[str, Any]:
    import torch

    if isinstance(prompt_or_observation, base.ExperimentObservation):
        prompt = base.build_training_prompt(prompt_or_observation)
    else:
        prompt = str(prompt_or_observation)

    model_device = getattr(model, "device", None)
    if model_device is None:
        try:
            model_device = next(model.parameters()).device
        except StopIteration:
            model_device = base.resolve_torch_runtime()["device"]

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(model_device) for key, value in inputs.items()}
    prompt_tokens = inputs["input_ids"].shape[1]

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "temperature": temperature,
        "top_p": top_p,
        "pad_token_id": tokenizer.pad_token_id,
    }
    with torch.no_grad():
        output_ids = model.generate(**inputs, **generation_kwargs)

    new_tokens = output_ids[0][prompt_tokens:]
    response_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    action = base.parse_action_completion(response_text)
    if action is not None and isinstance(prompt_or_observation, base.ExperimentObservation):
        action = base.ensure_conclusion_claims(prompt_or_observation, action)
    return {
        "prompt": prompt,
        "response_text": response_text,
        "action": action,
    }


def run_training(args: argparse.Namespace) -> Dict[str, Any]:
    random.seed(args.seed)
    runtime = base.resolve_torch_runtime()

    if args.load_model_only:
        tokenizer, model = load_model_artifacts(
            args.model_id,
            trust_remote_code=args.trust_remote_code,
            max_seq_length=args.max_seq_length,
            load_in_4bit=not args.disable_4bit,
            fast_inference=False,
            prepare_for_inference=True,
        )
        device = getattr(model, "device", "unknown")
        print(f"Unsloth model ready: {args.model_id}")
        print(f"Tokenizer vocab size: {len(tokenizer)}")
        print(f"Model device: {device}")
        print(f"Runtime device name: {runtime['device_name']}")
        return {
            "args": args,
            "runtime": runtime,
            "tokenizer": tokenizer,
            "model": model,
        }

    prompt_data = prepare_prompt_examples(args)
    scenario_names = prompt_data["scenario_names"]
    examples = prompt_data["examples"]
    env_reward = build_openenv_reward(args)

    if args.dry_run:
        base.run_dry_run_preview(examples, env_reward, args.output_dir)
        return {
            "args": args,
            "runtime": runtime,
            "scenario_names": scenario_names,
            "examples": examples,
            "reward_fn": env_reward,
        }

    from datasets import Dataset

    FastLanguageModel = patch_unsloth_grpo()
    train_dataset = Dataset.from_list(examples)

    # 1. Load model with Unsloth quantized loading.
    tokenizer, model = load_model_artifacts(
        args.model_id,
        trust_remote_code=args.trust_remote_code,
        max_seq_length=args.max_seq_length,
        load_in_4bit=not args.disable_4bit,
        fast_inference=False,
    )
    # 2. Apply LoRA adapters.
    model = apply_lora_adapters(FastLanguageModel, model, args)

    print(
        f"Unsloth training runtime: device={runtime['device']} "
        f"name={runtime['device_name']} "
        f"dtype={runtime['dtype']} "
        f"load_in_4bit={not args.disable_4bit}"
    )
    print(
        "OpenEnv reward: "
        f"backend={args.reward_backend} scenarios={len(scenario_names)} "
        f"examples={len(examples)}"
    )

    # 3. Train with GRPO against the OpenEnv reward function.
    trainer = build_unsloth_grpo_trainer(
        model=model,
        tokenizer=tokenizer,
        reward_func=env_reward,
        train_dataset=train_dataset,
        args=args,
        runtime=runtime,
    )
    # Workaround: UnslothGRPOTrainer expects vision token IDs for max_prompt_length
    # truncation; text-only models don't have them. Set to None so protected=[].
    for attr in ("image_token_id", "vision_start_token_id", "vision_end_token_id"):
        if not hasattr(trainer, attr):
            setattr(trainer, attr, None)
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.save_merged_16bit:
        merged_dir = Path(args.output_dir) / "merged_16bit"
        try:
            model.save_pretrained_merged(
                str(merged_dir),
                tokenizer,
                save_method="merged_16bit",
            )
            print(f"Saved merged 16-bit model to {merged_dir}")
        except AttributeError:
            print("Merged 16-bit export is not available in this Unsloth build; skipping.")

    if args.push_to_hub:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(repo_id=args.push_to_hub, repo_type="model", exist_ok=True)
        print(f"Pushing model to HuggingFace Hub: {args.push_to_hub}")
        api.upload_folder(
            folder_path=args.output_dir,
            repo_id=args.push_to_hub,
            repo_type="model",
            create_pr=False,
        )
        print(f"Model pushed to https://huggingface.co/{args.push_to_hub}")

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
        "scenario_names": scenario_names,
        "examples": examples,
        "reward_fn": env_reward,
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
