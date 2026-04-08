"""Inference script for StudentGrad — OpenEnv mandatory log format.

Logs:
  [START] task=hired env=student-optimizer model=<model>
  [STEP]  step=<n> action=<routine> reward=<val> done=<bool>
  [END]   success=<bool> steps=<n> score=<0-1.0>

Usage:
  python inference.py --model gpt-4o-mini --scenario hard_full_year --max-steps 365
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, Optional

import requests
from openai import OpenAI

from models import build_agent_system_prompt

SERVER_URL = os.environ.get("ENV_SERVER_URL", "http://localhost:8000")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")


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


def build_user_prompt(obs: Dict[str, Any]) -> str:
    day = obs.get("day", 0)
    energy = obs.get("energy", 10)
    fatigue = obs.get("fatigue", 0)
    attendance = obs.get("attendance", {})
    knowledge = obs.get("knowledge", {})
    skills = obs.get("skills", {})
    projects = obs.get("completed_projects", [])
    progress = obs.get("active_project_progress", 0)
    sick = obs.get("sick_today", False)
    quiz = obs.get("surprise_quiz_today", False)
    violations = obs.get("rule_violations", [])
    last_reward = obs.get("reward", 0)

    return f"""
DAY: {day}/365  |  ENERGY: {energy}/10  |  FATIGUE: {fatigue}/100
{"⚠️ SICK TODAY — reduced effectiveness" if sick else ""}
{"⚠️ SURPRISE QUIZ TODAY" if quiz else ""}

ATTENDANCE (need ≥40% for each subject to sit exam):
{json.dumps(attendance, indent=2)}

KNOWLEDGE (0-100):
{json.dumps(knowledge, indent=2)}

SKILLS (points):
{json.dumps(skills, indent=2)}

COMPLETED PROJECTS: {projects}
ACTIVE PROJECT PROGRESS: {round(progress * 100)}%

LAST STEP REWARD: {round(last_reward, 3)}
RULE VIOLATIONS: {violations}

Choose your action for today. Respond ONLY with JSON.
""".strip()


def parse_action(text: str) -> Dict[str, Any]:
    """Extract JSON from LLM output, with fallback."""
    text = text.strip()
    # Strip thinking blocks
    if "<think>" in text and "</think>" in text:
        text = text[text.rfind("</think>") + 8:].strip()
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Find first {...}
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
    # Hard fallback
    return {"action_type": "balanced_life", "justification": "parse_failure", "confidence": 0.3}


def run_episode(
    model: str,
    scenario: Optional[str],
    max_steps: int,
    verbose: bool = True,
) -> None:
    client = OpenAI(api_key=OPENAI_API_KEY)
    system_prompt = build_agent_system_prompt()

    # Reset
    obs_data = reset_env(scenario)
    obs = obs_data.get("observation", obs_data)

    print(f"[START] task=hired env=student-optimizer model={model}")

    total_reward = 0.0
    step = 0
    done = False
    success = False

    while not done and step < max_steps:
        step += 1
        user_msg = build_user_prompt(obs)

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=512,
                temperature=0.7,
            )
            raw = response.choices[0].message.content or ""
        except Exception as e:
            print(f"  [WARN] LLM error: {e}. Using fallback action.", file=sys.stderr)
            raw = '{"action_type": "balanced_life"}'

        action = parse_action(raw)

        try:
            step_data = step_env(action)
        except requests.HTTPError as e:
            print(f"  [WARN] Step HTTP error: {e}. Retrying with rest.", file=sys.stderr)
            step_data = step_env({"action_type": "rest"})

        obs = step_data.get("observation", step_data)
        reward = step_data.get("reward", obs.get("reward", 0.0))
        done = step_data.get("done", obs.get("done", False))
        total_reward += reward

        action_type = action.get("action_type", "unknown")
        print(f"[STEP] step={step} action={action_type} reward={round(reward, 4)} done={done}")

        if verbose and step % 30 == 0:
            print(
                f"  Day {obs.get('day', step)} | Fatigue: {obs.get('fatigue', 0)} | "
                f"Projects: {obs.get('completed_projects', [])} | "
                f"Cumulative reward: {round(total_reward, 2)}"
            )

    # Determine success
    if isinstance(obs, dict):
        latest = obs.get("latest_output") or {}
        if isinstance(latest, dict):
            success = latest.get("data", {}).get("passed", False)
        passed_all = obs.get("latest_output", {})
        # Also check via step reward breakdown
        breakdown = obs.get("step_reward_breakdown", {})
        if breakdown.get("term_academic_failed_penalty", 0) == 0 and total_reward > 5:
            success = True

    # Normalise score to 0-1
    score = min(1.0, max(0.0, total_reward / 15.0))  # 15 is approx max achievable

    print(f"[END] success={success} steps={step} score={round(score, 4)}")

def main():
    # This can just trigger your inference logic
    print("Inference engine starting...")
    # Add a call to your run function here if applicable

if __name__ == "__main__":
    main()