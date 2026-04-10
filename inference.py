"""Inference script for StudentGrad — OpenEnv mandatory log format.

Mandatory environment variables (per hackathon spec):
  API_BASE_URL   The API endpoint for the LLM (e.g. https://api.openai.com/v1)
  MODEL_NAME     The model identifier (e.g. gpt-4o-mini)
  HF_TOKEN       Your Hugging Face / API key

Logs (strict format):
  [START] task=<n> env=student-optimizer model=<model>
  [STEP]  step=<n> action=<routine> reward=<val> done=<bool> error=<None|msg>
  [END]   success=<bool> steps=<n> score=<0.0-1.0> rewards=<list>

Usage:
  python inference.py
  python inference.py --scenario easy_single_subject --max-steps 30
  python inference.py --scenario medium_three_subjects_basic_project --max-steps 180
  python inference.py --scenario hard_full_year --max-steps 365
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

from models import build_agent_system_prompt

# ── Mandatory hackathon variables ────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN",     os.environ.get("OPENAI_API_KEY", ""))

SERVER_URL   = os.environ.get("ENV_SERVER_URL", "http://localhost:8000")
TASK_NAME    = "student_career_optimization"
BENCHMARK    = "student-optimizer"


# ── Mandatory structured log helpers ─────────────────────────────────────────

def log_start(*, task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(*, step: int, action: str, reward: float, done: bool, error: Any) -> None:
    print(f"[STEP] step={step} action={action} reward={round(reward, 4)} done={done} error={error}", flush=True)


def log_end(*, success: bool, steps: int, score: float, rewards: List[float]) -> None:
    print(f"[END] success={success} steps={steps} score={round(score, 4)} rewards={rewards}", flush=True)


# ── Environment HTTP helpers ──────────────────────────────────────────────────
# openenv-core's StepRequest wraps the action: POST /step with body {"action": {...}}
# openenv-core's ResetRequest accepts: POST /reset with body {} or {"seed": n}

def reset_env(scenario: Optional[str] = None, seed: Optional[int] = None) -> Dict[str, Any]:
    """POST /reset — body is ResetRequest: {seed?, episode_id?}
    scenario_name is passed as a query parameter since ResetRequest has no scenario field."""
    params: Dict[str, Any] = {}
    if scenario:
        params["scenario_name"] = scenario  # sent as query param
    body: Dict[str, Any] = {}
    if seed is not None:
        body["seed"] = seed
    resp = requests.post(f"{SERVER_URL}/reset", json=body, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def step_env(action: Dict[str, Any]) -> Dict[str, Any]:
    """POST /step — body MUST be StepRequest: {"action": {action fields}}"""
    body = {"action": action}   # <-- THE FIX: openenv wraps in {"action": ...}
    resp = requests.post(f"{SERVER_URL}/step", json=body, timeout=30)
    resp.raise_for_status()
    return resp.json()


# ── Prompt helpers ────────────────────────────────────────────────────────────

def build_user_prompt(obs: Dict[str, Any]) -> str:
    day        = obs.get("day", 0)
    energy     = obs.get("energy", 10)
    fatigue    = obs.get("fatigue", 0)
    attendance = obs.get("attendance", {})
    knowledge  = obs.get("knowledge", {})
    skills     = obs.get("skills", {})
    projects   = obs.get("completed_projects", [])
    progress   = obs.get("active_project_progress", 0)
    sick       = obs.get("sick_today", False)
    quiz       = obs.get("surprise_quiz_today", False)
    violations = obs.get("rule_violations", [])
    last_reward = obs.get("reward", 0)

    return (
        f"DAY: {day}/365  |  ENERGY: {energy}/10  |  FATIGUE: {fatigue}/100\n"
        + ("⚠️ SICK TODAY — reduced effectiveness\n" if sick else "")
        + ("⚠️ SURPRISE QUIZ TODAY\n" if quiz else "")
        + f"\nATTENDANCE (need ≥40% each subject):\n{json.dumps(attendance, indent=2)}"
        + f"\nKNOWLEDGE (0-100):\n{json.dumps(knowledge, indent=2)}"
        + f"\nSKILLS (points):\n{json.dumps(skills, indent=2)}"
        + f"\nCOMPLETED PROJECTS: {projects}"
        + f"\nACTIVE PROJECT PROGRESS: {round(progress * 100)}%"
        + f"\nLAST STEP REWARD: {round(last_reward, 3)}"
        + f"\nRULE VIOLATIONS: {violations}"
        + "\n\nChoose your action for today. Respond ONLY with JSON."
    )


def parse_action(text: str) -> Dict[str, Any]:
    text = text.strip()
    if "<think>" in text and "</think>" in text:
        text = text[text.rfind("</think>") + 8:].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
    return {"action_type": "balanced_life", "justification": "parse_failure", "confidence": 0.3}


# ── Episode runner ────────────────────────────────────────────────────────────

def run_episode(model: str, scenario: Optional[str], max_steps: int, verbose: bool = True) -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    system_prompt = build_agent_system_prompt()

    obs_data = reset_env(scenario)
    obs = obs_data.get("observation", obs_data)

    log_start(task=TASK_NAME, env=BENCHMARK, model=model)

    total_reward = 0.0
    rewards: List[float] = []
    step = 0
    done = False
    success = False

    while not done and step < max_steps:
        step += 1
        user_msg = build_user_prompt(obs)
        error = None

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_msg},
                ],
                max_tokens=512,
                temperature=0.7,
            )
            raw = response.choices[0].message.content or ""
        except Exception as exc:
            error = str(exc)
            print(f"[DEBUG] LLM error: {exc}", file=sys.stderr, flush=True)
            raw = '{"action_type": "balanced_life"}'

        action = parse_action(raw)

        try:
            step_data = step_env(action)
        except requests.HTTPError as exc:
            error = str(exc)
            print(f"[DEBUG] Step HTTP error: {exc}. Falling back to rest.", file=sys.stderr, flush=True)
            try:
                step_data = step_env({"action_type": "rest"})
            except Exception as exc2:
                print(f"[DEBUG] Fallback also failed: {exc2}", file=sys.stderr, flush=True)
                break

        obs         = step_data.get("observation", step_data)
        reward      = float(step_data.get("reward", obs.get("reward", 0.0)))
        done        = bool(step_data.get("done",   obs.get("done",   False)))
        total_reward += reward
        rewards.append(reward)

        log_step(step=step, action=action.get("action_type", "unknown"),
                 reward=reward, done=done, error=error)

        if verbose and step % 30 == 0:
            print(
                f"  Day {obs.get('day', step)} | Fatigue: {obs.get('fatigue', 0)} | "
                f"Projects: {obs.get('completed_projects', [])} | "
                f"Cumulative: {round(total_reward, 2)}",
                flush=True,
            )

        if done:
            break

    if isinstance(obs, dict):
        latest = obs.get("latest_output") or {}
        if isinstance(latest, dict):
            success = bool(latest.get("data", {}).get("passed", False))
        breakdown = obs.get("step_reward_breakdown", {})
        if not success and breakdown.get("term_academic_failed_penalty", 0) == 0 and total_reward > 5:
            success = True

    score = min(1.0, max(0.0, total_reward / 15.0))
    log_end(success=success, steps=step, score=score, rewards=rewards)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="StudentGrad inference runner")
    parser.add_argument("--model",     default=MODEL_NAME,       help="Model name")
    parser.add_argument("--scenario",  default="hard_full_year", help="Scenario name")
    parser.add_argument("--max-steps", type=int, default=365,    help="Max episode steps")
    parser.add_argument("--verbose",   action="store_true", default=True)
    args = parser.parse_args()

    if not HF_TOKEN:
        print("[WARN] HF_TOKEN not set.", file=sys.stderr, flush=True)

    run_episode(model=args.model, scenario=args.scenario,
                max_steps=args.max_steps, verbose=args.verbose)


if __name__ == "__main__":
    main()
