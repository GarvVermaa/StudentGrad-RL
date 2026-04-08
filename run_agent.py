"""Run the student environment with Qwen3-0.6B as the planning agent."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# ── FIXED: only import what actually exists in models.py ─────────────────────
from models import (
    ActionType,
    StudentAction,
    StudentObservation,
    build_agent_system_prompt,
)
from server.student_environment import StudentEnvironment

DASHBOARD_STATE_PATH = Path(__file__).parent / "_dashboard_state.json"
DASHBOARD_CMD_PATH   = Path(__file__).parent / "_dashboard_cmd.json"

USE_PIPELINE = os.getenv("RUN_AGENT_USE_PIPELINE", "0").strip().lower() not in {"0", "false", "off"}


def _parse_thinking_flag() -> bool:
    import sys
    if "--no-thinking" in sys.argv:
        return False
    if "--thinking" in sys.argv:
        return True
    return os.getenv("RUN_AGENT_ENABLE_THINKING", "1").strip().lower() not in {"0", "false", "off"}


ENABLE_THINKING = _parse_thinking_flag()

MODEL_ID          = "Qwen/Qwen3-0.6B"
MAX_EPISODE_STEPS = int(os.getenv("RUN_AGENT_MAX_EPISODE_STEPS", "20"))
PIPELINE_TASK     = "text-generation"

# ── FIXED: aliases point only to ActionType values that exist ─────────────────
ACTION_TYPES = [a.value for a in ActionType]

ACTION_TYPE_ALIASES: Dict[str, str] = {
    "study":        ActionType.FULL_ACADEMIC.value,
    "academic":     ActionType.FULL_ACADEMIC.value,
    "attend":       ActionType.FULL_ACADEMIC.value,
    "skill":        ActionType.SKILL_DEEP_DIVE.value,
    "deep_dive":    ActionType.SKILL_DEEP_DIVE.value,
    "project":      ActionType.PROJECT_SPRINT.value,
    "sprint":       ActionType.PROJECT_SPRINT.value,
    "balanced":     ActionType.BALANCED_LIFE.value,
    "balance":      ActionType.BALANCED_LIFE.value,
    "cram":         ActionType.CRAM_MODE.value,
    "grind":        ActionType.CRAM_MODE.value,
    "sleep":        ActionType.REST.value,
    "recover":      ActionType.REST.value,
    "finish":       ActionType.SUBMIT_OUTCOME.value,
    "submit":       ActionType.SUBMIT_OUTCOME.value,
    "done":         ActionType.SUBMIT_OUTCOME.value,
}

# ── FIXED: pipeline order uses only real ActionType members ───────────────────
STANDARD_PIPELINE_ORDER = [
    ActionType.FULL_ACADEMIC,
    ActionType.BALANCED_LIFE,
    ActionType.SKILL_DEEP_DIVE,
    ActionType.PROJECT_SPRINT,
    ActionType.REST,
    ActionType.CRAM_MODE,
    ActionType.SUBMIT_OUTCOME,
]

SYSTEM_PROMPT = build_agent_system_prompt()

MODEL_RESPONSE_PREVIEW_CHARS = int(
    os.getenv("RUN_AGENT_MODEL_RESPONSE_PREVIEW_CHARS", "240")
)


def compact_preview(value: Any, max_chars: int = 160) -> str:
    try:
        text = json.dumps(value, ensure_ascii=True, sort_keys=True)
    except TypeError:
        text = str(value)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


# ── FIXED: format_observation uses only StudentObservation fields ─────────────
def format_observation(obs: StudentObservation) -> str:
    parts = [
        f"TASK: {obs.task.problem_statement}",
        f"Day: {obs.day} / 365  |  Energy: {obs.energy:.1f} / 10  |  Fatigue: {obs.fatigue:.1f}",
        f"Sick today: {obs.sick_today}  |  Surprise quiz: {obs.surprise_quiz_today}",
    ]

    att_str = "  ".join(f"{s}: {v*100:.0f}%" for s, v in obs.attendance.items())
    parts.append(f"Attendance: {att_str}")

    know_str = "  ".join(f"{s}: {v:.1f}" for s, v in obs.knowledge.items())
    parts.append(f"Knowledge:  {know_str}")

    if obs.skills:
        skill_str = "  ".join(f"{s}: {v:.1f}" for s, v in obs.skills.items())
        parts.append(f"Skills:     {skill_str}")

    if obs.completed_projects:
        parts.append(f"Completed projects: {', '.join(obs.completed_projects)}")
    if obs.active_project_progress > 0:
        parts.append(f"Active project progress: {obs.active_project_progress*100:.0f}%")

    if obs.session_history:
        parts.append("Recent history (last days):")
        for h in obs.session_history[-5:]:
            skill_info = f" -> {h.skill_target.value}" if h.skill_target else ""
            proj_info  = f" -> {h.project_target.value}" if h.project_target else ""
            parts.append(
                f"  Day {h.day}: {h.action_type.value}{skill_info}{proj_info}"
                f"  reward={h.reward:+.2f}  [{h.summary[:80]}]"
            )

    if obs.rule_violations:
        parts.append(f"VIOLATIONS THIS STEP: {obs.rule_violations}")

    if obs.step_reward_breakdown:
        rb_str = "  ".join(f"{k}={v:+.3f}" for k, v in obs.step_reward_breakdown.items())
        parts.append(f"Last reward breakdown: {rb_str}")

    if obs.latest_output:
        lo = obs.latest_output
        parts.append(
            f"Last action result: success={lo.success}  "
            f"quality={lo.quality_score:.2f}  uncertainty={lo.uncertainty:.2f}"
        )
        if lo.warnings:
            parts.append(f"  Warnings: {lo.warnings}")

    parts.append(
        'Output ONLY a single JSON object — no comments, no extra text:\n'
        '{"action_type": "<action>", "skill_target": null, "project_target": null, '
        '"justification": "<why>", "confidence": 0.8}'
    )
    return "\n".join(parts)


def _repair_truncated_json(text: str) -> Optional[str]:
    s = text.strip()
    if not s.startswith("{"):
        return None
    s = re.sub(r',\s*"[^\"\n]*$', '', s)
    s = re.sub(r',\s*"[^\"\n]*"\s*:\s*$', '', s)
    in_string = escape = False
    for ch in s:
        if escape:
            escape = False; continue
        if ch == "\\":
            escape = True; continue
        if ch == '"':
            in_string = not in_string
    if in_string:
        s += '"'
    s += "]" * max(0, s.count("[") - s.count("]"))
    s += "}" * max(0, s.count("{") - s.count("}"))
    s = re.sub(r',\s*([}\]])', r'\1', s)
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return s
    except json.JSONDecodeError:
        pass
    return None


def _strip_js_comments(text: str) -> str:
    text = re.sub(r'//[^\n]*', '', text)
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    return text


def _normalize_jsonish_text(text: str) -> str:
    text = _strip_js_comments(text)
    text = re.sub(r'(?<=:\s)\bNone\b', 'null', text)
    text = re.sub(r'(?<=:\s)\bTrue\b', 'true', text)
    text = re.sub(r'(?<=:\s)\bFalse\b', 'false', text)
    return text


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    stripped = _normalize_jsonish_text(text).strip()
    if stripped.startswith('"') and stripped.endswith('"'):
        try:
            unwrapped = json.loads(stripped)
            if isinstance(unwrapped, str):
                stripped = _normalize_jsonish_text(unwrapped).strip()
        except json.JSONDecodeError:
            pass
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            stripped = "\n".join(lines[1:-1]).strip()

    candidates: List[str] = [stripped]
    start = stripped.find("{")
    while start != -1:
        depth = 0
        for idx in range(start, len(stripped)):
            ch = stripped[idx]
            if ch == "{": depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(stripped[start: idx + 1])
                    break
        start = stripped.find("{", start + 1)

    first_brace = stripped.find("{")
    if first_brace != -1:
        repaired = _repair_truncated_json(stripped[first_brace:])
        if repaired is not None:
            candidates.append(repaired)

    candidates.sort(key=len, reverse=True)
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    return None


def _edit_distance(a: str, b: str) -> int:
    if len(a) < len(b):
        return _edit_distance(b, a)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (ca != cb)))
        prev = curr
    return prev[-1]


def get_payload_value(payload: Dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in payload:
            return payload[name]
    lowered = {str(k).lower(): v for k, v in payload.items()}
    for name in names:
        if name.lower() in lowered:
            return lowered[name.lower()]
    for key, value in lowered.items():
        for name in names:
            threshold = max(2, len(name) // 3)
            if _edit_distance(key, name.lower()) <= threshold:
                return value
    return None


def normalize_optional_string(value: Any) -> Optional[str]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        value = value.strip()
        return value or None
    if isinstance(value, (int, float)):
        return str(value)
    return compact_preview(value, 80)


def normalize_action_type(raw: Any) -> Optional[str]:
    if not isinstance(raw, str):
        return None
    candidate = raw.strip().lower()
    if candidate in ACTION_TYPES:
        return candidate
    if candidate in ACTION_TYPE_ALIASES:
        return ACTION_TYPE_ALIASES[candidate]
    candidate = re.sub(r"[^a-z0-9]+", "_", candidate).strip("_")
    if candidate in ACTION_TYPES:
        return candidate
    if candidate in ACTION_TYPE_ALIASES:
        return ACTION_TYPE_ALIASES[candidate]

    # ── FIXED: heuristics use only real StudentGrad ActionType values ─────────
    heuristics = [
        (("full", "academic"),  ActionType.FULL_ACADEMIC.value),
        (("attend",),           ActionType.FULL_ACADEMIC.value),
        (("study",),            ActionType.FULL_ACADEMIC.value),
        (("skill", "deep"),     ActionType.SKILL_DEEP_DIVE.value),
        (("skill",),            ActionType.SKILL_DEEP_DIVE.value),
        (("project", "sprint"), ActionType.PROJECT_SPRINT.value),
        (("project",),          ActionType.PROJECT_SPRINT.value),
        (("balanced",),         ActionType.BALANCED_LIFE.value),
        (("balance",),          ActionType.BALANCED_LIFE.value),
        (("cram",),             ActionType.CRAM_MODE.value),
        (("rest",),             ActionType.REST.value),
        (("sleep",),            ActionType.REST.value),
        (("submit",),           ActionType.SUBMIT_OUTCOME.value),
        (("finish",),           ActionType.SUBMIT_OUTCOME.value),
    ]
    for fragments, normalized in heuristics:
        if all(frag in candidate for frag in fragments):
            return normalized
    return None


# ── FIXED: parse_action returns StudentAction ─────────────────────────────────
def parse_action(text: str) -> Optional[StudentAction]:
    d = extract_json_object(text)
    if d is not None:
        action_type_str = normalize_action_type(get_payload_value(d, "action_type"))
        if action_type_str is not None:
            parameters = get_payload_value(d, "parameters", "params") or {}
            if not isinstance(parameters, dict):
                parameters = {}

            raw_conf = get_payload_value(d, "confidence")
            try:
                confidence = float(raw_conf) if raw_conf is not None else 0.5
            except (TypeError, ValueError):
                confidence = 0.5

            justification = get_payload_value(
                d, "justification", "justifyement", "reasoning", "rationale", "reason"
            )
            if justification is not None and not isinstance(justification, str):
                justification = compact_preview(justification, 200)

            skill_target_raw   = normalize_optional_string(get_payload_value(d, "skill_target"))
            project_target_raw = normalize_optional_string(get_payload_value(d, "project_target"))

            from models import SkillType, ProjectTier
            skill_target = None
            if skill_target_raw:
                try:
                    skill_target = SkillType(skill_target_raw.lower())
                except ValueError:
                    pass

            project_target = None
            if project_target_raw:
                try:
                    project_target = ProjectTier(project_target_raw.lower())
                except ValueError:
                    pass

            return StudentAction(
                action_type=ActionType(action_type_str),
                skill_target=skill_target,
                project_target=project_target,
                parameters=parameters,
                justification=justification,
                confidence=min(1.0, max(0.0, confidence)),
            )

    # Regex fallback
    action_match = re.search(
        r'["\']action_type["\']\s*:\s*["\']([^"\']+)', text, re.IGNORECASE
    )
    if not action_match:
        return None
    action_type_str = normalize_action_type(action_match.group(1))
    if action_type_str is None:
        return None

    confidence_match = re.search(
        r'["\']confidence["\']\s*:\s*([0-9]*\.?[0-9]+)', text, re.IGNORECASE
    )
    justification_match = re.search(
        r'["\'](?:justif\w*|reasoning|rationale|reason)["\']\s*:\s*"((?:[^"\\]|\\.)*)',
        text, re.DOTALL | re.IGNORECASE,
    )

    confidence = 0.5
    if confidence_match:
        try:
            confidence = float(confidence_match.group(1))
        except ValueError:
            pass

    justification = None
    if justification_match:
        try:
            justification = json.loads(f'"{justification_match.group(1)}"')
        except json.JSONDecodeError:
            justification = justification_match.group(1)

    return StudentAction(
        action_type=ActionType(action_type_str),
        parameters={},
        justification=justification,
        confidence=min(1.0, max(0.0, confidence)),
    )


def _unique_nonempty(items: List[str], limit: int = 5) -> List[str]:
    seen: set[str] = set()
    result: List[str] = []
    for raw in items:
        value = normalize_optional_string(raw)
        if not value:
            continue
        key = value.upper()
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
        if len(result) >= limit:
            break
    return result


def write_dashboard_state(
    env: StudentEnvironment,
    obs: StudentObservation,
    *,
    step: int,
    cumulative_reward: float,
    model_response: str = "",
    model_thinking: str = "",
    action: Optional[StudentAction] = None,
    gen_time: float = 0.0,
    episode_done: bool = False,
) -> None:
    latent = env._latent
    snapshot: Dict[str, Any] = {
        "timestamp": time.time(),
        "step": step,
        "episode_done": episode_done,
        "cumulative_reward": cumulative_reward,
        "gen_time_s": round(gen_time, 2),
        "model_response_raw": model_response[:600],
        "model_thinking": model_thinking[:800],
        "thinking_enabled": ENABLE_THINKING,
    }

    snapshot["task"] = {
        "problem_statement": obs.task.problem_statement,
        "organism": "Student Agent",
        "tissue": f"AKTU-CSE ({obs.task.difficulty})",
        "modality": "365-Day Sim",
        "conditions": obs.task.target_subjects,
        "budget_limit": 10.0,
        "time_limit_days": 365.0,
    }
    snapshot["resources"] = {
        "budget_used": round(10.0 - obs.energy, 2),
        "budget_remaining": round(obs.energy, 2),
        "time_used_days": float(obs.day),
        "time_remaining_days": float(365 - obs.day),
        "samples_consumed": len(obs.completed_projects),
        "compute_hours_used": round(obs.fatigue, 1),
    }
    snapshot["pipeline_history"] = [
        {
            "step_index": h.day,
            "action_type": h.action_type.value,
            "method": h.skill_target.value if h.skill_target else "N/A",
            "output_summary": h.summary[:120],
            "success": True,
            "quality_score": round(h.reward, 3),
            "resource_cost": h.energy_spent,
            "time_cost_days": 1,
        }
        for h in obs.session_history
    ]
    snapshot["discovered_markers"]  = obs.completed_projects
    snapshot["candidate_mechanisms"] = [f"{k}: {v:.1f}" for k, v in obs.skills.items()]
    snapshot["rule_violations"]      = obs.rule_violations
    snapshot["uncertainty_summary"]  = {"avg_fatigue": obs.fatigue}
    snapshot["reward_breakdown"]     = obs.step_reward_breakdown

    if action:
        snapshot["current_action"] = {
            "action_type": action.action_type.value,
            "method": action.skill_target.value if action.skill_target else None,
            "parameters": action.parameters,
            "justification": action.justification,
            "confidence": action.confidence,
        }
    if latent:
        snapshot["latent"] = {
            "knowledge": latent.true_knowledge,
            "attendance": latent.true_attendance,
            "stress_level": latent.resources.fatigue_current,
            "is_sick": latent.last_sick_triggered,
        }

    try:
        DASHBOARD_STATE_PATH.write_text(
            json.dumps(snapshot, indent=2, default=str), encoding="utf-8"
        )
    except Exception:
        pass


def log(msg: str) -> None:
    print(msg, flush=True)


def build_observation_prompt(obs: StudentObservation) -> str:
    return format_observation(obs)


def run_with_pipeline(pipe, prompt: str) -> str:
    try:
        result = pipe(prompt, max_new_tokens=2048 if ENABLE_THINKING else 300, return_full_text=False)
    except Exception:
        return ""
    if isinstance(result, list) and result:
        result = result[0]
    if isinstance(result, dict):
        text = result.get("generated_text") or result.get("text") or result.get("answer")
    elif isinstance(result, str):
        text = result
    else:
        text = ""
    return text.strip() if isinstance(text, str) else ""


def resolve_torch_runtime() -> Dict[str, Any]:
    use_cuda = torch.cuda.is_available()
    bf16 = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)()) if use_cuda else False
    dtype = torch.bfloat16 if bf16 else (torch.float16 if use_cuda else torch.float32)
    return {
        "use_cuda": use_cuda,
        "device": "cuda:0" if use_cuda else "cpu",
        "dtype": dtype,
        "device_map": "auto" if use_cuda else None,
        "device_name": torch.cuda.get_device_name(0) if use_cuda else "cpu",
    }


def main():
    tokenizer = None
    model = None
    eos_ids: List[int] = []
    active_pipeline = None

    runtime = resolve_torch_runtime()
    log(f"Runtime: device={runtime['device']} name={runtime['device_name']} dtype={runtime['dtype']}")

    if USE_PIPELINE:
        log(f"Loading pipeline for {MODEL_ID} ...")
        try:
            active_pipeline = pipeline(
                PIPELINE_TASK, model=MODEL_ID, trust_remote_code=True,
                dtype=runtime["dtype"], device=0 if runtime["use_cuda"] else -1,
            )
            log("Pipeline loaded.")
        except Exception as exc:
            log(f"Pipeline load failed ({exc}), falling back.")

    if active_pipeline is None:
        log(f"Loading tokenizer for {MODEL_ID} ...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        log("Loading model ...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, dtype=runtime["dtype"],
            device_map=runtime["device_map"], trust_remote_code=True,
        )
        log(f"Model loaded. Device: {model.device}")
        if tokenizer.eos_token_id is not None:
            eos_ids.append(tokenizer.eos_token_id)
        for tok in ["<|im_end|>", "<|endoftext|>"]:
            tid = tokenizer.convert_tokens_to_ids(tok)
            if isinstance(tid, int) and tid not in eos_ids:
                eos_ids.append(tid)
        log(f"EOS ids: {eos_ids}")

    def check_dashboard_command() -> Optional[Dict[str, Any]]:
        try:
            raw = DASHBOARD_CMD_PATH.read_text(encoding="utf-8")
            try:
                DASHBOARD_CMD_PATH.unlink(missing_ok=True)
            except OSError:
                pass
            return json.loads(raw)
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def run_episode(
        scenario_name: Optional[str] = None,
        custom_ground_truth: Optional[Dict[str, Any]] = None,
    ) -> None:
        env = StudentEnvironment(scenario_name=scenario_name)
        obs = env.reset()

        if custom_ground_truth and env._latent:
            gt = custom_ground_truth
            latent = env._latent
            if gt.get("target_projects"):
                latent.completed_projects = gt["target_projects"]
            if gt.get("initial_knowledge"):
                latent.true_knowledge = {k: float(v) for k, v in gt["initial_knowledge"].items()}

        log("\n" + "=" * 70)
        log(f"TASK: {obs.task.problem_statement}")
        log(f"Subjects: {obs.task.target_subjects}")
        log(f"Energy: {obs.energy:.1f} | Day: {obs.day} / 365")
        if ENABLE_THINKING:
            log("Reasoning mode: ENABLED")
        log("=" * 70)

        cumulative_reward = 0.0
        write_dashboard_state(env, obs, step=0, cumulative_reward=0.0)

        for step in range(MAX_EPISODE_STEPS):
            # ── FIXED: terminate immediately when done/energy/day hit limits ──
            if obs.done:
                log("  [done=True — ending episode]"); break
            if obs.energy <= 0:
                log("  [energy depleted — ending episode]"); break
            if obs.day >= 365:
                log("  [day 365 reached — ending episode]"); break

            cmd = check_dashboard_command()
            if cmd and cmd.get("action") == "restart":
                log("\n[DASHBOARD] Restart requested."); break

            user_msg = build_observation_prompt(obs)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ]

            if active_pipeline is not None:
                prompt = f"{SYSTEM_PROMPT}\n\n{user_msg}"
            else:
                try:
                    prompt = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True,
                        enable_thinking=ENABLE_THINKING,
                    )
                except TypeError:
                    prompt = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True,
                    )

            t0 = time.time()
            if active_pipeline is not None:
                response = run_with_pipeline(active_pipeline, prompt) or ""
            else:
                assert tokenizer is not None and model is not None
                inputs  = tokenizer(prompt, return_tensors="pt").to(model.device)
                n_input = inputs["input_ids"].shape[1]
                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=2048 if ENABLE_THINKING else 300,
                        do_sample=True, temperature=0.7, top_p=0.8,
                        top_k=20, repetition_penalty=1.3,
                        eos_token_id=eos_ids if eos_ids else None,
                    )
                response = tokenizer.decode(output_ids[0][n_input:], skip_special_tokens=True).strip()
            gen_time = time.time() - t0

            thinking = ""
            if ENABLE_THINKING:
                think_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
                if think_match:
                    thinking = think_match.group(1).strip()
                    response  = response[think_match.end():].strip()
                elif response.startswith("<think>"):
                    parts = response.split("</think>", 1)
                    if len(parts) == 2:
                        thinking = parts[0].replace("<think>", "").strip()
                        response  = parts[1].strip()

            is_last_step = (step == MAX_EPISODE_STEPS - 1)

            # ── FIXED: parse into StudentAction ──────────────────────────────
            action = parse_action(response)
            if action is None:
                if is_last_step:
                    log(f"\n  [!] Parse failed on final step — forcing submit_outcome.")
                    action = StudentAction(
                        action_type=ActionType.SUBMIT_OUTCOME,
                        justification="forced terminal — parse failed",
                        confidence=0.5,
                    )
                else:
                    log(f"\n  [!] Parse failed, skipping. Raw: {response[:150]}")
                    continue

            if is_last_step and action.action_type != ActionType.SUBMIT_OUTCOME:
                log(f"\n  [!] Max steps — overriding with submit_outcome.")
                action = StudentAction(
                    action_type=ActionType.SUBMIT_OUTCOME,
                    justification="forced terminal — max steps reached",
                    confidence=action.confidence,
                )

            log(f"\nStep {step + 1}: {action.action_type.value}  ({gen_time:.1f}s)")
            if thinking:
                log(f"  Thinking: {thinking[:200]}")
            if action.justification:
                log(f"  Rationale: {action.justification}")
            if action.skill_target:
                log(f"  Skill: {action.skill_target.value}")
            if action.project_target:
                log(f"  Project: {action.project_target.value}")

            obs = env.step(action)

            if obs.latest_output:
                lo = obs.latest_output
                log(f"  [{'OK' if lo.success else 'FAIL'}] {lo.summary}")
                if lo.warnings:
                    log(f"  Warnings: {lo.warnings}")

            step_reward        = obs.reward
            cumulative_reward += step_reward
            log(f"  Reward: {step_reward:+.3f}  (cum: {cumulative_reward:+.3f})")
            log(f"  Energy: {obs.energy:.1f} | Fatigue: {obs.fatigue:.1f} | Day: {obs.day}")

            write_dashboard_state(
                env, obs, step=step + 1,
                cumulative_reward=cumulative_reward,
                model_response=response, model_thinking=thinking,
                action=action, gen_time=gen_time, episode_done=obs.done,
            )
            if obs.rule_violations:
                log(f"  Violations: {obs.rule_violations}")

            # ── FIXED: check done flag after every step ───────────────────
            if obs.done:
                log("  [done=True — episode finished cleanly]"); break

        log(f"\n{'=' * 70}")
        log("EPISODE COMPLETE" if obs.done else f"MAX STEPS ({MAX_EPISODE_STEPS}) REACHED")
        log(f"  Days used: {obs.day} / 365  |  Total reward: {cumulative_reward:+.3f}")
        log(f"  Energy remaining: {obs.energy:.1f}")
        if obs.completed_projects:
            log(f"  Projects: {obs.completed_projects}")
        log("=" * 70)

    try:
        DASHBOARD_CMD_PATH.unlink(missing_ok=True)
    except OSError:
        pass

    run_episode()

    while True:
        log("\nWaiting for dashboard command ...")
        while True:
            cmd = check_dashboard_command()
            if cmd:
                break
            time.sleep(1.0)
        action_type = cmd.get("action", "restart")
        if action_type == "quit":
            log("Quit requested."); break
        log(f"\n[DASHBOARD] {action_type} — scenario={cmd.get('scenario_name')}")
        run_episode(
            scenario_name=cmd.get("scenario_name"),
            custom_ground_truth=cmd.get("ground_truth"),
        )


if __name__ == "__main__":
    main()