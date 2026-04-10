"""Train a StudentGrad policy with TRL GRPO and OpenEnv rewards.

Usage:
    python training_script.py \
        --model-id Qwen/Qwen3.5-0.8B \
        --output-dir outputs/grpo-run \
        --dataset-episodes 16 \
        --rollout-steps 30 \
        --num-generations 4

Environment:
    ENV_SERVER_URL  — URL of the running StudentGrad server (default: http://localhost:8000)
    OPENAI_API_KEY  — API key if using an OpenAI-compatible inference server
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from models import StudentAction, StudentObservation, build_agent_system_prompt

SERVER_URL = os.environ.get("ENV_SERVER_URL", "http://localhost:8000")


# ── Environment helpers ──────────────────────────────────────────────────────

def reset_env(scenario: Optional[str] = None) -> Dict[str, Any]:
    payload = {}
    if scenario:
        payload["scenario_name"] = scenario
    resp = requests.post(f"{SERVER_URL}/reset", json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


def step_env(action: Dict[str, Any]) -> Dict[str, Any]:
    resp = requests.post(f"{SERVER_URL}/step", json=action, timeout=30)
    resp.raise_for_status()
    return resp.json()


# ── Rollout collection ───────────────────────────────────────────────────────

def collect_rollout(
    model_id: str,
    scenario: Optional[str] = None,
    max_steps: int = 365,
) -> List[Dict[str, Any]]:
    """Collect one episode of (prompt, completion, reward) triples for GRPO."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "local"))
    except ImportError:
        raise ImportError("openai package required. Install with: pip install openai")

    system_prompt = build_agent_system_prompt()
    obs_data = reset_env(scenario)
    obs = obs_data.get("observation", obs_data)

    samples: List[Dict[str, Any]] = []
    done = False
    step = 0

    while not done and step < max_steps:
        step += 1
        user_msg = _build_user_prompt(obs)

        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=256,
            temperature=0.8,
        )
        completion_text = response.choices[0].message.content or ""

        action = _parse_action(completion_text)
        step_data = step_env(action)
        obs = step_data.get("observation", step_data)
        reward = step_data.get("reward", obs.get("reward", 0.0))
        done = step_data.get("done", obs.get("done", False))

        samples.append({
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            "completion": completion_text,
            "reward": reward,
        })

    return samples


def _build_user_prompt(obs: Dict[str, Any]) -> str:
    return (
        f"DAY: {obs.get('day', 0)}/365  |  "
        f"ENERGY: {obs.get('energy', 10)}/10  |  "
        f"FATIGUE: {obs.get('fatigue', 0)}/100\n"
        f"ATTENDANCE: {json.dumps(obs.get('attendance', {}))}\n"
        f"KNOWLEDGE: {json.dumps(obs.get('knowledge', {}))}\n"
        f"SKILLS: {json.dumps(obs.get('skills', {}))}\n"
        f"PROJECTS: {obs.get('completed_projects', [])}\n"
        f"LAST REWARD: {round(obs.get('reward', 0), 3)}\n\n"
        "Choose your action for today. Respond ONLY with JSON."
    )


def _parse_action(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
    return {"action_type": "balanced_life", "justification": "parse_failure", "confidence": 0.3}


# ── GRPO training loop ───────────────────────────────────────────────────────

def run_grpo_training(
    model_id: str,
    output_dir: Path,
    dataset_episodes: int = 8,
    rollout_steps: int = 30,
    num_generations: int = 4,
    scenario: Optional[str] = None,
) -> None:
    """Full GRPO training loop using TRL."""
    try:
        from trl import GRPOConfig, GRPOTrainer
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        raise ImportError("trl and transformers required. Install train extras.")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[GRPO] Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)

    print(f"[GRPO] Collecting {dataset_episodes} rollout episodes ...")
    all_samples: List[Dict[str, Any]] = []
    for ep in range(dataset_episodes):
        samples = collect_rollout(model_id, scenario=scenario, max_steps=rollout_steps)
        all_samples.extend(samples)
        print(f"  Episode {ep + 1}/{dataset_episodes}: {len(samples)} steps")

    # Save collected data
    data_path = output_dir / "rollout_data.json"
    data_path.write_text(json.dumps(all_samples, indent=2))
    print(f"[GRPO] Saved {len(all_samples)} samples to {data_path}")

    config = GRPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=1,
        per_device_train_batch_size=1,
        num_generations=num_generations,
        logging_steps=1,
    )

    def reward_fn(completions: List[str], **kwargs) -> List[float]:
        """Simple reward: valid JSON with a known action_type gets +1."""
        rewards = []
        valid_actions = {a.value for a in __import__("models").ActionType}
        for c in completions:
            action = _parse_action(c)
            rewards.append(1.0 if action.get("action_type") in valid_actions else -1.0)
        return rewards

    trainer = GRPOTrainer(
        model=model,
        args=config,
        reward_funcs=reward_fn,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(output_dir / "final_model"))
    print(f"[GRPO] Training complete. Model saved to {output_dir / 'final_model'}")


# ── CLI entry point ──────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="StudentGrad GRPO training script")
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--output-dir", default="outputs/grpo-run")
    parser.add_argument("--dataset-episodes", type=int, default=8)
    parser.add_argument("--rollout-steps", type=int, default=30)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    run_grpo_training(
        model_id=args.model_id,
        output_dir=Path(args.output_dir),
        dataset_episodes=args.dataset_episodes,
        rollout_steps=args.rollout_steps,
        num_generations=args.num_generations,
        scenario=args.scenario,
    )


if __name__ == "__main__":
    main()
