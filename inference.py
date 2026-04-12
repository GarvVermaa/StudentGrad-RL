"""StudentGrad-RL — Inference script for OpenEnv hackathon validator.

Runs 3 tasks in sequence, producing one [START]...[END] block per task.

Required environment variables:
  API_BASE_URL   LLM endpoint (e.g. https://api.groq.com/openai/v1)
  MODEL_NAME     Model identifier (e.g. llama-3.1-8b-instant)
  HF_TOKEN       API key — NO DEFAULT, must be set explicitly
  ENV_SERVER_URL Environment server URL (default: HF Space URL)
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

from models import build_agent_system_prompt

# ── Mandatory env vars ────────────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN")   # NO default — must be set

SERVER_URL   = os.environ.get("ENV_SERVER_URL", "https://GarvVermaa-StudentGrad-RL.hf.space")

# ── Task definitions — names MUST match openenv.yaml exactly ─────────────────
TASKS = [
    {"name": "easy_attendance",  "scenario": "easy_single_subject",                  "max_steps": 30},
    {"name": "medium_projects",  "scenario": "medium_three_subjects_basic_project",   "max_steps": 60},
    {"name": "hard_full_year",   "scenario": "hard_full_year",                        "max_steps": 100},
]


# ── Log helpers — strict format required by validator ─────────────────────────

def _bool(v: bool) -> str:
    return "true" if v else "false"

def _error(e: Any) -> str:
    return "null" if e is None else str(e)

def log_start(task: str, model: str) -> None:
    print(f"[START] task={task} env=student-optimizer model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Any) -> None:
    print(
        f"[STEP] step={step} action={action} "
        f"reward={reward:.2f} done={_bool(done)} error={_error(error)}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_csv = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={_bool(success)} steps={steps} "
        f"score={score:.2f} rewards={rewards_csv}",
        flush=True,
    )


# ── Environment HTTP helpers ──────────────────────────────────────────────────

def reset_env(scenario: str) -> Dict[str, Any]:
    resp = requests.post(
        f"{SERVER_URL}/reset",
        json={},
        params={"scenario_name": scenario},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def step_env(action: Dict[str, Any]) -> Dict[str, Any]:
    resp = requests.post(
        f"{SERVER_URL}/step",
        json={"action": action},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


# ── LLM action generation ─────────────────────────────────────────────────────

def _parse_action(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
    return {"action_type": "balanced_life", "justification": "parse_failure", "confidence": 0.3}


def get_action(client: OpenAI, obs: Dict[str, Any]) -> Dict[str, Any]:
    system = build_agent_system_prompt()
    day      = obs.get("day", 0)
    energy   = obs.get("energy", 10.0)
    fatigue  = obs.get("fatigue", 0.0)
    att      = obs.get("attendance", {})
    know     = obs.get("knowledge", {})
    skills   = obs.get("skills", {})
    projects = obs.get("completed_projects", [])

    user = (
        f"DAY {day}/365 | ENERGY: {energy:.1f}/10 | FATIGUE: {fatigue:.0f}/100\n"
        f"ATTENDANCE: {', '.join(f'{k}:{v:.0%}' for k,v in att.items())}\n"
        f"KNOWLEDGE: {', '.join(f'{k}:{v:.0f}' for k,v in know.items())}\n"
        f"SKILLS: {', '.join(f'{k}:{v:.1f}' for k,v in skills.items())}\n"
        f"PROJECTS DONE: {projects if projects else 'none'}\n\n"
        "Choose your action. Respond ONLY with JSON."
    )

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            max_tokens=200,
            temperature=0.7,
        )
        return _parse_action(resp.choices[0].message.content or "")
    except Exception:
        return {"action_type": "balanced_life", "justification": "llm_error", "confidence": 0.3}


# ── Single task runner ────────────────────────────────────────────────────────

def run_task(client: OpenAI, task: Dict[str, Any]) -> None:
    task_name = task["name"]
    scenario  = task["scenario"]
    max_steps = task["max_steps"]

    log_start(task=task_name, model=MODEL_NAME)

    rewards: List[float] = []
    obs: Dict[str, Any] = {}
    error: Any = None
    done = False
    step = 0

    try:
        reset_data = reset_env(scenario)
        obs = reset_data.get("observation", reset_data)
    except Exception as exc:
        print(f"[DEBUG] Reset failed: {exc}", file=sys.stderr, flush=True)
        # Still emit a valid END block so validator can parse it
        log_end(success=False, steps=0, score=0.0001, rewards=[])
        return

    while step < max_steps and not done:
        step += 1
        error = None

        action = get_action(client, obs)
        action_str = action.get("action_type", "balanced_life")

        try:
            step_data = step_env(action)
            obs    = step_data.get("observation", step_data)
            reward = float(step_data.get("reward", obs.get("reward", 0.0)))
            done   = bool(step_data.get("done",   obs.get("done", False)))
        except requests.HTTPError as exc:
            error = str(exc)
            print(f"[DEBUG] Step HTTP error: {exc}", file=sys.stderr, flush=True)
            # Fallback action
            try:
                step_data = step_env({"action_type": "rest"})
                obs    = step_data.get("observation", step_data)
                reward = float(step_data.get("reward", obs.get("reward", 0.0)))
                done   = bool(step_data.get("done",   obs.get("done", False)))
            except Exception as exc2:
                print(f"[DEBUG] Fallback failed: {exc2}", file=sys.stderr, flush=True)
                reward = 0.0

        rewards.append(reward)
        log_step(step=step, action=action_str, reward=reward, done=done, error=error)

    # Score: total_reward / (max_steps * 1.6), clamped strictly to (0.0001, 0.9999)
    total = sum(rewards)
    raw_score = total / (max_steps * 1.6)
    score = max(0.0001, min(0.9999, raw_score))

    # Determine success from observation
    success = False
    if isinstance(obs, dict):
        latest = obs.get("latest_output") or {}
        if isinstance(latest, dict):
            success = bool(latest.get("data", {}).get("passed", False))
        if not success:
            breakdown = obs.get("step_reward_breakdown", {})
            if breakdown.get("term_academic_failed_penalty", 0) == 0 and total > 5:
                success = True

    log_end(success=success, steps=step, score=score, rewards=rewards)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    if not HF_TOKEN:
        print("[WARN] HF_TOKEN not set — LLM calls will fail.", file=sys.stderr, flush=True)

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "no-key")

    for task in TASKS:
        run_task(client, task)


if __name__ == "__main__":
    main()