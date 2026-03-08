"""Compare base vs trained model on the same prompts."""

from __future__ import annotations

import argparse
import json
import random
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from training_script import (
    SYSTEM_PROMPT,
    OpenEnvReward,
    build_prompt_examples,
    completion_to_text,
    parse_action_completion,
    selected_scenarios,
)


def generate_completions(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 220,
) -> List[str]:
    completions = []
    for prompt in prompts:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
        generated = output[0][inputs["input_ids"].shape[1]:]
        completions.append(tokenizer.decode(generated, skip_special_tokens=True))
    return completions


def evaluate_model(
    model,
    tokenizer,
    examples: List[Dict[str, str]],
    reward_fn: OpenEnvReward,
    label: str,
) -> Dict[str, float]:
    prompts = [ex["prompt"] for ex in examples]
    completions = generate_completions(model, tokenizer, prompts)

    rewards = []
    valid_actions = 0
    for comp, ex in zip(completions, examples):
        reward = reward_fn(
            completions=[comp],
            scenario_name=[ex.get("scenario_name")],
            history_actions=[ex.get("history_actions")],
        )[0]
        rewards.append(reward)
        if parse_action_completion(comp) is not None:
            valid_actions += 1

    avg_reward = sum(rewards) / len(rewards) if rewards else 0
    valid_pct = valid_actions / len(completions) * 100 if completions else 0

    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    print(f"  Samples:        {len(completions)}")
    print(f"  Avg reward:     {avg_reward:.4f}")
    print(f"  Min reward:     {min(rewards):.4f}")
    print(f"  Max reward:     {max(rewards):.4f}")
    print(f"  Valid actions:  {valid_actions}/{len(completions)} ({valid_pct:.1f}%)")
    print()

    # Show a few example completions
    for i, (comp, r) in enumerate(zip(completions[:3], rewards[:3])):
        print(f"  Example {i+1} (reward={r:.2f}):")
        print(f"    {comp[:200]}")
        print()

    return {"avg_reward": avg_reward, "valid_pct": valid_pct, "rewards": rewards}


def main():
    parser = argparse.ArgumentParser(description="Compare base vs trained model")
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-0.8B",
                        help="Base model ID from HuggingFace")
    parser.add_argument("--trained-model", default="./grpo-output",
                        help="Path to trained model (local dir or HF repo)")
    parser.add_argument("--num-samples", type=int, default=16,
                        help="Number of eval prompts")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)

    # Build eval prompts
    scenarios = selected_scenarios(None)
    examples = build_prompt_examples(
        dataset_episodes=args.num_samples,
        rollout_steps=1,  # one prompt per episode
        collection_policy="heuristic",
        scenario_names=scenarios,
        seed=args.seed,
        domain_randomise=False,
    )
    print(f"Built {len(examples)} eval prompts across {len(scenarios)} scenarios")

    reward_fn = OpenEnvReward(reward_backend="local", base_url="")

    # Evaluate base model
    print(f"\nLoading base model: {args.base_model}")
    base_tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, trust_remote_code=args.trust_remote_code
    )
    if base_tokenizer.pad_token is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    base_results = evaluate_model(
        base_model, base_tokenizer, examples, reward_fn, "BASE MODEL"
    )
    del base_model
    torch.cuda.empty_cache()

    # Evaluate trained model
    print(f"\nLoading trained model: {args.trained_model}")
    trained_tokenizer = AutoTokenizer.from_pretrained(
        args.trained_model, trust_remote_code=args.trust_remote_code
    )
    if trained_tokenizer.pad_token is None:
        trained_tokenizer.pad_token = trained_tokenizer.eos_token
    trained_model = AutoModelForCausalLM.from_pretrained(
        args.trained_model,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    trained_results = evaluate_model(
        trained_model, trained_tokenizer, examples, reward_fn, "TRAINED MODEL"
    )

    # Summary
    delta = trained_results["avg_reward"] - base_results["avg_reward"]
    print(f"{'='*50}")
    print(f"  COMPARISON SUMMARY")
    print(f"{'='*50}")
    print(f"  Base avg reward:    {base_results['avg_reward']:.4f}")
    print(f"  Trained avg reward: {trained_results['avg_reward']:.4f}")
    print(f"  Delta:              {delta:+.4f}")
    print(f"  Base valid actions: {base_results['valid_pct']:.1f}%")
    print(f"  Trained valid:      {trained_results['valid_pct']:.1f}%")
    print()


if __name__ == "__main__":
    main()
