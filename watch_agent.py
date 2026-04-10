"""
watch_agent.py — Run a full episode WITHOUT an LLM (rule-based agent),
print rich per-step output, and write a JSON trajectory for the dashboard.

Usage (while server is running in another terminal):
  uv run python watch_agent.py
  uv run python watch_agent.py --scenario easy_single_subject --max-steps 30
  uv run python watch_agent.py --scenario medium_three_subjects_basic_project --steps 180

The agent uses a simple priority-rule policy so you can watch it learn
the state space without needing an API key.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import requests

SERVER_URL = os.environ.get("ENV_SERVER_URL", "http://localhost:8000")
TRAJECTORY_FILE = "trajectory.json"


# ── Rule-based agent policy ───────────────────────────────────────────────────

def rule_policy(obs: Dict[str, Any], step: int) -> Dict[str, Any]:
    """
    Priority rules (shows the decision logic you'd want an RL agent to learn):
      1. If fatigue > 70 → REST
      2. If day ≤ 60 → FULL_ACADEMIC (build attendance base)
      3. If 61 ≤ day ≤ 180 → alternate SKILL_DEEP_DIVE (build js/node/docker)
      4. If 181 ≤ day ≤ 250 → PROJECT_SPRINT (build projects)
      5. If 251 ≤ day ≤ 300 → FULL_ACADEMIC / CRAM near exam
      6. If day > 300 → SUBMIT_OUTCOME
    """
    day     = obs.get("day", 1)
    fatigue = obs.get("fatigue", 0)
    skills  = obs.get("skills", {})
    projects = obs.get("completed_projects", [])

    if fatigue >= 70:
        return {"action_type": "rest", "justification": f"fatigue={fatigue} >= 70, resting"}

    if day > 300:
        return {"action_type": "submit_outcome", "justification": "exam day passed"}

    if day <= 60:
        return {"action_type": "full_academic", "justification": "building attendance base"}

    if day <= 120:
        skill = _next_skill_target(skills, ["js", "html", "css"])
        return {"action_type": "skill_deep_dive", "skill_target": skill,
                "justification": f"grinding {skill} for basic project"}

    if day <= 160:
        if skills.get("html", 0) >= 5 and skills.get("css", 0) >= 5 and "basic" not in projects:
            return {"action_type": "project_sprint", "project_target": "basic",
                    "justification": "basic prereqs met, building project"}
        skill = _next_skill_target(skills, ["js", "node"])
        return {"action_type": "skill_deep_dive", "skill_target": skill,
                "justification": f"building {skill} toward fullstack"}

    if day <= 220:
        if skills.get("js", 0) >= 10 and "fullstack" not in projects:
            return {"action_type": "project_sprint", "project_target": "fullstack",
                    "justification": "js >= 10, building fullstack"}
        skill = _next_skill_target(skills, ["node", "docker"])
        return {"action_type": "skill_deep_dive", "skill_target": skill,
                "justification": f"grinding {skill} for cloud project"}

    if day <= 260:
        if skills.get("node", 0) >= 10 and skills.get("docker", 0) >= 10 and "cloud" not in projects:
            return {"action_type": "project_sprint", "project_target": "cloud",
                    "justification": "cloud prereqs met, building cloud project"}
        return {"action_type": "balanced_life", "justification": "balancing study and skills"}

    # 261–300: exam prep
    days_to_exam = 300 - day
    if days_to_exam <= 15:
        return {"action_type": "cram_mode", "justification": f"{days_to_exam} days to exam, cramming"}
    return {"action_type": "full_academic", "justification": "exam prep, attending all classes"}


def _next_skill_target(skills: Dict[str, float], priority: List[str]) -> str:
    for s in priority:
        if skills.get(s, 0) < 10:
            return s
    return priority[0]


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def reset_env(scenario: Optional[str] = None) -> Dict[str, Any]:
    params = {"scenario_name": scenario} if scenario else {}
    r = requests.post(f"{SERVER_URL}/reset", json={}, params=params, timeout=15)
    r.raise_for_status()
    return r.json()


def step_env(action: Dict[str, Any]) -> Dict[str, Any]:
    r = requests.post(f"{SERVER_URL}/step", json={"action": action}, timeout=15)
    r.raise_for_status()
    return r.json()


# ── Pretty printer ────────────────────────────────────────────────────────────

COLORS = {
    "reset":    "\033[0m",
    "bold":     "\033[1m",
    "green":    "\033[92m",
    "yellow":   "\033[93m",
    "red":      "\033[91m",
    "cyan":     "\033[96m",
    "blue":     "\033[94m",
    "magenta":  "\033[95m",
    "dim":      "\033[2m",
}
C = COLORS

def bar(value: float, max_val: float, width: int = 20, fill: str = "█") -> str:
    filled = int(round(value / max_val * width))
    return fill * filled + "░" * (width - filled)


def print_step(step: int, action: Dict, obs: Dict, reward: float) -> None:
    day     = obs.get("day", 0)
    energy  = obs.get("energy", 10)
    fatigue = obs.get("fatigue", 0)
    att     = obs.get("attendance", {})
    know    = obs.get("knowledge", {})
    skills  = obs.get("skills", {})
    projects = obs.get("completed_projects", [])
    violations = obs.get("rule_violations", [])

    fatigue_color = C["red"] if fatigue >= 70 else C["yellow"] if fatigue >= 40 else C["green"]
    reward_color  = C["green"] if reward >= 0 else C["red"]

    print(f"\n{C['bold']}{C['cyan']}{'─'*60}{C['reset']}")
    print(f"{C['bold']} Day {day:>3}/365  Step {step:>3}   Action: {C['magenta']}{action.get('action_type','?'):20}{C['reset']}")
    print(f"{C['bold']}{'─'*60}{C['reset']}")

    # Energy & Fatigue
    print(f"  Energy  {C['green']}{bar(energy, 10)}{C['reset']} {energy:.1f}/10")
    print(f"  Fatigue {fatigue_color}{bar(fatigue, 100)}{C['reset']} {fatigue:.1f}/100")

    # Reward
    print(f"  Reward  {reward_color}{reward:+.4f}{C['reset']}   Justification: {C['dim']}{action.get('justification','')[:50]}{C['reset']}")

    # Attendance
    print(f"\n  {C['bold']}Attendance (≥40% needed):{C['reset']}")
    for subj, pct in att.items():
        color = C["green"] if pct >= 0.75 else C["yellow"] if pct >= 0.40 else C["red"]
        print(f"    {subj:6}  {color}{bar(pct, 1.0, 15)}{C['reset']} {pct*100:.1f}%")

    # Knowledge
    print(f"\n  {C['bold']}Knowledge (0-100):{C['reset']}")
    for subj, kval in know.items():
        color = C["green"] if kval >= 60 else C["yellow"] if kval >= 40 else C["red"]
        print(f"    {subj:6}  {color}{bar(kval, 100, 15)}{C['reset']} {kval:.1f}")

    # Skills
    print(f"\n  {C['bold']}Skills (0-30):{C['reset']}")
    for skill, pts in skills.items():
        color = C["green"] if pts >= 10 else C["yellow"] if pts >= 5 else C["dim"]
        print(f"    {skill:6}  {color}{bar(pts, 30, 15)}{C['reset']} {pts:.1f}")

    # Projects
    if projects:
        print(f"\n  {C['bold']}Completed Projects:{C['reset']} {C['green']}{', '.join(projects)}{C['reset']}")

    # Violations
    if violations:
        for v in violations:
            vcolor = C["red"] if v.startswith("HARD") else C["yellow"]
            print(f"  {vcolor}⚠  {v[:80]}{C['reset']}")


# ── Main runner ───────────────────────────────────────────────────────────────

def run(scenario: Optional[str], max_steps: int, delay: float) -> None:
    print(f"\n{C['bold']}{C['cyan']}StudentGrad-RL — Live Agent Watch{C['reset']}")
    print(f"Scenario: {scenario or 'hard_full_year'}  |  Max steps: {max_steps}")
    print(f"Server: {SERVER_URL}\n")

    try:
        data = reset_env(scenario)
    except requests.ConnectionError:
        print(f"{C['red']}Cannot connect to {SERVER_URL}. Is the server running?{C['reset']}")
        print(f"Start it with:  uv run python server/app.py")
        sys.exit(1)

    obs = data.get("observation", data)
    trajectory = []
    total_reward = 0.0

    for step in range(1, max_steps + 1):
        action = rule_policy(obs, step)

        try:
            result = step_env(action)
        except requests.HTTPError as e:
            print(f"{C['red']}Step {step} HTTP error: {e}{C['reset']}")
            break

        prev_obs  = obs
        obs       = result.get("observation", result)
        reward    = float(result.get("reward", 0.0))
        done      = bool(result.get("done", False))
        total_reward += reward

        print_step(step, action, obs, reward)

        # Save for dashboard
        trajectory.append({
            "step": step,
            "day": obs.get("day", step),
            "action": action.get("action_type"),
            "reward": reward,
            "cumulative_reward": round(total_reward, 4),
            "fatigue": obs.get("fatigue", 0),
            "energy": obs.get("energy", 10),
            "attendance": dict(obs.get("attendance", {})),
            "knowledge": dict(obs.get("knowledge", {})),
            "skills": dict(obs.get("skills", {})),
            "projects": list(obs.get("completed_projects", [])),
            "violations": list(obs.get("rule_violations", [])),
        })

        if delay > 0:
            time.sleep(delay)

        if done:
            print(f"\n{C['bold']}{C['green']}Episode done at step {step}!{C['reset']}")
            break

    # Summary
    score = min(1.0, max(0.0, total_reward / 15.0))
    print(f"\n{C['bold']}{'='*60}{C['reset']}")
    print(f"{C['bold']} EPISODE SUMMARY{C['reset']}")
    print(f"{'='*60}")
    print(f"  Total steps   : {step}")
    print(f"  Total reward  : {round(total_reward, 4)}")
    print(f"  Score (0-1)   : {round(score, 4)}")
    print(f"  Final projects: {obs.get('completed_projects', [])}")
    att = obs.get("attendance", {})
    all_pass = all(v >= 0.40 for v in att.values()) if att else False
    print(f"  Attendance OK : {C['green'] if all_pass else C['red']}{all_pass}{C['reset']}")
    print(f"{'='*60}\n")

    # Write trajectory JSON for dashboard
    with open(TRAJECTORY_FILE, "w") as f:
        json.dump(trajectory, f, indent=2)
    print(f"Trajectory saved → {TRAJECTORY_FILE}")
    print(f"Open {C['cyan']}agent_dashboard.html{C['reset']} in your browser to visualise it.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch the rule-based agent live")
    parser.add_argument("--scenario",  default="hard_full_year")
    parser.add_argument("--max-steps", type=int, default=365)
    parser.add_argument("--delay",     type=float, default=0.05,
                        help="Seconds between steps (0 = as fast as possible)")
    args = parser.parse_args()
    run(args.scenario, args.max_steps, args.delay)


if __name__ == "__main__":
    main()
