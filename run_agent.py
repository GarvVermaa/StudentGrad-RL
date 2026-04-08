"""Run the student environment with Qwen3.5-0.8B as the planning agent."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from models import (
    ActionType,
    ExperimentAction,
    ExperimentObservation,
    OutputType,
    build_agent_observation_context,
    build_agent_system_prompt,
)
from models import StudentAction, StudentObservation
from server.student_environment import StudentEnvironment

DASHBOARD_STATE_PATH = Path(__file__).parent / "_dashboard_state.json"
DASHBOARD_CMD_PATH = Path(__file__).parent / "_dashboard_cmd.json"

USE_PIPELINE = os.getenv("RUN_AGENT_USE_PIPELINE", "0").strip().lower() not in {"0", "false", "off"}

def _parse_thinking_flag() -> bool:
    import sys
    if "--no-thinking" in sys.argv:
        return False
    if "--thinking" in sys.argv:
        return True
    return os.getenv("RUN_AGENT_ENABLE_THINKING", "1").strip().lower() not in {"0", "false", "off"}

ENABLE_THINKING = _parse_thinking_flag()

MODEL_ID = "Qwen/Qwen3.5-2B"
MAX_EPISODE_STEPS = int(os.getenv("RUN_AGENT_MAX_EPISODE_STEPS", "20"))
PIPELINE_TASK = "text-generation"

ACTION_TYPES = [a.value for a in ActionType]
ACTION_TYPE_ALIASES = {
    "study": ActionType.FULL_ACADEMIC.value,
    "academic": ActionType.FULL_ACADEMIC.value,
    "skill": ActionType.SKILL_DEEP_DIVE.value,
    "project": ActionType.PROJECT_SPRINT.value,
    "balanced": ActionType.BALANCED_LIFE.value,
    "cram": ActionType.CRAM_MODE.value,
    "sleep": ActionType.REST.value,
    "finish": ActionType.SUBMIT_OUTCOME.value,
}

SYSTEM_PROMPT = build_agent_system_prompt()

STANDARD_PIPELINE_ORDER = [
    ActionType.FULL_ACADEMIC,
    ActionType.BALANCED_LIFE,
    ActionType.SKILL_DEEP_DIVE,
    ActionType.PROJECT_SPRINT,
    ActionType.REST,
    ActionType.CRAM_MODE,
    ActionType.SUBMIT_OUTCOME,
]

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


def format_observation(obs: StudentObservation) -> str:
    parts = [
        f"TASK: {obs.task.problem_statement}",
        f"Day: {obs.day} / 365 | Energy: {obs.energy:.1f} | Fatigue: {obs.fatigue:.1f}",
        f"Subjects: {', '.join(obs.attendance.keys())}",
    ]
    context = build_agent_observation_context(obs, max_tools=5, max_assays=2)
    if context:
        parts.append(context)
    if obs.pipeline_history:
        last5 = obs.pipeline_history[-5:]
        parts.append("Recent history:")
        for h in last5:
            tag = "OK" if h.success else "FAIL"
            line = f"  [{tag}] {h.action_type.value}"
            if h.method:
                line += f" ({h.method})"
            line += f": {h.output_summary[:80]}"
            parts.append(line)

        completed = {h.action_type for h in obs.pipeline_history if h.success}
        if completed:
            parts.append(f"Completed steps (do NOT repeat): {', '.join(sorted(a.value for a in completed))}")
        remaining = [a.value for a in STANDARD_PIPELINE_ORDER if a not in completed]
        if remaining:
            parts.append(f"Remaining steps (choose one): {', '.join(remaining)}")

    if obs.latest_output and obs.latest_output.data:
        parts.append(
            f"Latest data: {compact_preview(obs.latest_output.data, 200)}"
        )
    if obs.rule_violations:
        parts.append(f"VIOLATIONS: {obs.rule_violations}")
    if obs.discovered_markers:
        parts.append(f"Markers found so far: {obs.discovered_markers[:5]}")

    parts.append(
        'Output ONLY a single JSON object with these exact keys, no comments, no extra text:\n'
        '{"action_type": "<one of the remaining steps>", "method": null, "parameters": {}, "justification": "<why>", "confidence": 0.8}'
    )
    return "\n".join(parts)


def _repair_truncated_json(text: str) -> Optional[str]:
    """Try to repair JSON truncated mid-value (common with small LLMs)."""
    s = text.strip()
    if not s.startswith("{"):
        return None

    # Drop dangling partial keys or empty key/value stubs at the tail.
    s = re.sub(r',\s*"[^"\n]*$', '', s)
    s = re.sub(r',\s*"[^"\n]*"\s*:\s*$', '', s)

    in_string = False
    escape = False
    for ch in s:
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string

    if in_string:
        s += '"'

    open_braces = s.count("{") - s.count("}")
    open_brackets = s.count("[") - s.count("]")
    s += "]" * max(0, open_brackets)
    s += "}" * max(0, open_braces)

    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return s
    except json.JSONDecodeError:
        pass

    s = re.sub(r',\s*([}\]])', r'\1', s)
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return s
    except json.JSONDecodeError:
        pass
    return None


def _normalize_jsonish_text(text: str) -> str:
    """Normalize common near-JSON artifacts emitted by small local models."""
    text = _strip_js_comments(text)
    text = re.sub(r'(?<=:\s)\bNone\b', 'null', text)
    text = re.sub(r'(?<=:\s)\bTrue\b', 'true', text)
    text = re.sub(r'(?<=:\s)\bFalse\b', 'false', text)
    text = re.sub(r'"([^"\n]+?):"\s*,', r'"\1": "",', text)
    return text


def _strip_js_comments(text: str) -> str:
    """Remove // and /* */ comments that small LLMs inject into JSON."""
    text = re.sub(r'//[^\n]*', '', text)
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    return text


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    stripped = _normalize_jsonish_text(text).strip()
    if stripped.startswith('"') and stripped.endswith('"'):
        try:
            unwrapped = json.loads(stripped)
        except json.JSONDecodeError:
            unwrapped = None
        if isinstance(unwrapped, str):
            stripped = _normalize_jsonish_text(unwrapped).strip()
    fence_prefix = "```"
    if stripped.startswith(fence_prefix) and stripped.endswith(fence_prefix):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            stripped = "\n".join(lines[1:-1]).strip()

    candidates: List[str] = [stripped]
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

    repaired = None
    first_brace = stripped.find("{")
    if first_brace != -1:
        repaired = _repair_truncated_json(stripped[first_brace:])
        if repaired is not None:
            candidates.append(repaired)

    candidates.sort(key=len, reverse=True)

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

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

    lowered = {
        str(key).lower(): value
        for key, value in payload.items()
    }
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


def normalize_action_type(raw_action_type: Any) -> Optional[str]:
    if not isinstance(raw_action_type, str):
        return None

    candidate = raw_action_type.strip().lower()
    if candidate in ACTION_TYPES:
        return candidate
    if candidate in ACTION_TYPE_ALIASES:
        return ACTION_TYPE_ALIASES[candidate]

    candidate = re.sub(r"[^a-z0-9]+", "_", candidate).strip("_")
    if candidate in ACTION_TYPES:
        return candidate
    if candidate in ACTION_TYPE_ALIASES:
        return ACTION_TYPE_ALIASES[candidate]

    heuristics = [
        (("collect", "sample"), ActionType.COLLECT_SAMPLE.value),
        (("library",), ActionType.PREPARE_LIBRARY.value),
        (("sequence",), ActionType.SEQUENCE_CELLS.value),
        (("qc",), ActionType.RUN_QC.value),
        (("quality", "control"), ActionType.RUN_QC.value),
        (("filter",), ActionType.FILTER_DATA.value),
        (("normal",), ActionType.NORMALIZE_DATA.value),
        (("integrat", "batch"), ActionType.INTEGRATE_BATCHES.value),
        (("cluster",), ActionType.CLUSTER_CELLS.value),
        (("differential", "expression"), ActionType.DIFFERENTIAL_EXPRESSION.value),
        (("pathway",), ActionType.PATHWAY_ENRICHMENT.value),
        (("trajectory",), ActionType.TRAJECTORY_ANALYSIS.value),
        (("network",), ActionType.REGULATORY_NETWORK_INFERENCE.value),
        (("marker",), ActionType.MARKER_SELECTION.value),
        (("validat", "marker"), ActionType.VALIDATE_MARKER.value),
        (("followup",), ActionType.DESIGN_FOLLOWUP.value),
        (("review",), ActionType.REQUEST_SUBAGENT_REVIEW.value),
        (("conclusion",), ActionType.SYNTHESIZE_CONCLUSION.value),
    ]
    for fragments, normalized in heuristics:
        if all(fragment in candidate for fragment in fragments):
            return normalized
    return None


def should_block_failed_reattempt(
    history: List[Any], action_type: ActionType
) -> bool:
    last_failed_idx = None
    last_success_idx = None

    for idx, record in enumerate(history):
        if record.action_type != action_type:
            continue
        if record.success:
            last_success_idx = idx
        else:
            last_failed_idx = idx

    if last_failed_idx is None:
        return False

    if last_success_idx is not None and last_success_idx > last_failed_idx:
        return False
    for record in history[last_failed_idx + 1:]:
        if record.success and record.action_type != action_type:
            return False
    return True


def parse_action(text: str) -> Optional[ExperimentAction]:
    d = extract_json_object(text)
    if d is not None:
        action_type = normalize_action_type(get_payload_value(d, "action_type"))
        if action_type is None:
            pass
        else:
            parameters = get_payload_value(d, "parameters", "params") or {}
            if not isinstance(parameters, dict):
                parameters = {}

            confidence = get_payload_value(d, "confidence")
            if confidence is None:
                confidence = 0.5
            try:
                confidence = float(confidence)
            except (TypeError, ValueError):
                confidence = 0.5

            justification = get_payload_value(
                d, "justification", "justifyement", "reasoning", "rationale", "reason"
            )
            if justification is not None and not isinstance(justification, str):
                justification = compact_preview(justification, 200)
            method = normalize_optional_string(get_payload_value(d, "method"))

            return ExperimentAction(
                action_type=ActionType(action_type),
                method=method,
                parameters=parameters,
                justification=justification,
                confidence=min(1.0, max(0.0, confidence)),
            )

    action_match = re.search(
        r'["\']action_type["\']\s*:\s*["\']([^"\']+)',
        text,
        re.IGNORECASE,
    )
    if not action_match:
        return None

    action_type = normalize_action_type(action_match.group(1))
    if action_type is None:
        return None

    method_match = re.search(
        r'["\']method["\']\s*:\s*("((?:[^"\\]|\\.)*)"|null|none|true|false|-?\d+(?:\.\d+)?)',
        text,
        re.IGNORECASE,
    )
    confidence_match = re.search(
        r'["\']confidence["\']\s*:\s*([0-9]*\.?[0-9]+)',
        text,
        re.IGNORECASE,
    )
    justification_match = re.search(
        r'["\'](?:justif\w*|reasoning|rationale|reason)["\']\s*:\s*"((?:[^"\\]|\\.)*)',
        text,
        re.DOTALL | re.IGNORECASE,
    )

    confidence = 0.5
    if confidence_match:
        try:
            confidence = float(confidence_match.group(1))
        except ValueError:
            confidence = 0.5

    justification = None
    if justification_match:
        try:
            justification = json.loads(f'"{justification_match.group(1)}"')
        except json.JSONDecodeError:
            justification = justification_match.group(1)

    method = None
    if method_match:
        raw_method = method_match.group(1)
        if raw_method.startswith('"') and raw_method.endswith('"'):
            try:
                method = json.loads(raw_method)
            except json.JSONDecodeError:
                method = raw_method.strip('"')
        elif raw_method.lower() not in {"null", "none", "true", "false"}:
            method = raw_method
    method = normalize_optional_string(method)

    return ExperimentAction(
        action_type=ActionType(action_type),
        method=method,
        parameters={},
        justification=justification,
        confidence=min(1.0, max(0.0, confidence)),
    )


def should_force_terminal_conclusion(
    action: ExperimentAction,
    completed_types: set[ActionType],
) -> bool:
    meta_repeatables = {
        ActionType.DESIGN_FOLLOWUP,
        ActionType.REQUEST_SUBAGENT_REVIEW,
    }
    return (
        action.action_type in meta_repeatables
        and action.action_type in completed_types
        and ActionType.SYNTHESIZE_CONCLUSION not in completed_types
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


def _infer_conclusion_evidence(
    obs: ExperimentObservation,
) -> tuple[List[str], List[str], Dict[str, float]]:
    top_markers = _unique_nonempty(list(obs.discovered_markers), limit=5)
    causal_mechanisms = _unique_nonempty(list(obs.candidate_mechanisms), limit=5)
    predicted_pathways: Dict[str, float] = {}

    for output in reversed(obs.all_outputs):
        if not output.success:
            continue

        data = output.data or {}
        if not top_markers:
            if output.output_type == OutputType.MARKER_RESULT:
                top_markers = _unique_nonempty(list(data.get("markers", [])), limit=5)
            elif output.output_type == OutputType.DE_RESULT:
                top_markers = _unique_nonempty(
                    [item.get("gene") for item in data.get("top_genes", []) if isinstance(item, dict)],
                    limit=5,
                )

        if output.output_type == OutputType.PATHWAY_RESULT and not predicted_pathways:
            for item in data.get("top_pathways", []):
                if not isinstance(item, dict):
                    continue
                pathway = normalize_optional_string(item.get("pathway"))
                score = item.get("score")
                if pathway and isinstance(score, (int, float)):
                    predicted_pathways[pathway] = float(score)
                    if len(predicted_pathways) >= 5:
                        break

        if not causal_mechanisms:
            if output.output_type == OutputType.PATHWAY_RESULT:
                causal_mechanisms = _unique_nonempty(
                    [item.get("pathway") for item in data.get("top_pathways", []) if isinstance(item, dict)],
                    limit=5,
                )
            elif output.output_type == OutputType.NETWORK_RESULT:
                causal_mechanisms = _unique_nonempty(
                    list(data.get("top_regulators", [])),
                    limit=5,
                )

        if top_markers and causal_mechanisms and predicted_pathways:
            break

    return top_markers, causal_mechanisms, predicted_pathways


def ensure_conclusion_claims(
    obs: ExperimentObservation,
    action: ExperimentAction,
) -> ExperimentAction:
    if action.action_type != ActionType.SYNTHESIZE_CONCLUSION:
        return action

    parameters = dict(action.parameters or {})
    raw_claims = parameters.get("claims")
    if isinstance(raw_claims, list) and raw_claims:
        normalized_claims = [claim for claim in raw_claims if isinstance(claim, dict)]
        if normalized_claims:
            parameters["claims"] = normalized_claims
            if parameters != action.parameters:
                return action.model_copy(update={"parameters": parameters})
            return action

    top_markers, causal_mechanisms, predicted_pathways = _infer_conclusion_evidence(obs)
    claim_type = "causal" if causal_mechanisms else "correlational"
    conditions = " vs ".join(obs.task.conditions[:2]) if obs.task.conditions else "the task conditions"
    claim = action.justification or f"Final synthesis for {conditions}."

    parameters["claims"] = [{
        "top_markers": top_markers,
        "causal_mechanisms": causal_mechanisms,
        "predicted_pathways": predicted_pathways,
        "confidence": action.confidence,
        "claim_type": claim_type,
        "claim": claim,
    }]
    if not action.justification:
        action = action.model_copy(update={"justification": claim})
    return action.model_copy(update={"parameters": parameters})


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
    """Serialise student state into bio-shaped JSON for the dashboard UI."""
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

    # Map Student Task to Bio Task slots
    snapshot["task"] = {
        "problem_statement": obs.task.problem_statement,
        "organism": "Student Agent",          # Dummy for UI
        "tissue": f"AKTU-CSE ({obs.task.difficulty})",
        "modality": "365-Day Sim",
        "conditions": obs.task.target_subjects,
        "budget_limit": 10.0,                 # Energy Max
        "time_limit_days": 365.0,
    }

    # Map Resources
    snapshot["resources"] = {
        "budget_used": round(10.0 - obs.energy, 2),  # Energy spent
        "budget_remaining": round(obs.energy, 2),     # Energy bar
        "time_used_days": float(obs.day),
        "time_remaining_days": float(365 - obs.day),
        "samples_consumed": len(obs.completed_projects),
        "compute_hours_used": round(obs.fatigue, 1),  # Fatigue mapped to cost
    }

    # History Mapping
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

    # Map Student-specific findings to Bio UI elements
    snapshot["discovered_markers"] = obs.completed_projects   # Shows in 'Markers' list
    snapshot["candidate_mechanisms"] = [
        f"{k}: {v:.1f}" for k, v in obs.skills.items()
    ]  # Shows in 'Mechanisms' list

    snapshot["rule_violations"] = obs.rule_violations
    snapshot["uncertainty_summary"] = {"avg_fatigue": obs.fatigue}
    snapshot["reward_breakdown"] = obs.step_reward_breakdown

    if action:
        snapshot["current_action"] = {
            "action_type": action.action_type.value,
            "method": action.skill_target.value if action.skill_target else None,
            "parameters": action.parameters,
            "justification": action.justification,
            "confidence": action.confidence,
        }

    # Hidden Latent State (For Admin Debugging)
    if latent:
        snapshot["latent"] = {
            "knowledge": latent.knowledge,
            "attendance": latent.attendance,
            "stress_level": latent.fatigue_current,
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


def build_observation_prompt(obs: ExperimentObservation) -> str:
    return format_observation(obs)


def run_with_pipeline(pipe, prompt: str) -> str:
    try:
        _pipe_max = 2048 if ENABLE_THINKING else 300
        result = pipe(prompt, max_new_tokens=_pipe_max, return_full_text=False)
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
    dtype = torch.bfloat16 if bf16 else (
        torch.float16 if use_cuda else torch.float32
    )
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
    log(
        f"Using local model runtime: device={runtime['device']} "
        f"name={runtime['device_name']} dtype={runtime['dtype']}"
    )

    if USE_PIPELINE:
        log(f"Loading pipeline ({PIPELINE_TASK}) for {MODEL_ID} ...")
        try:
            active_pipeline = pipeline(
                PIPELINE_TASK,
                model=MODEL_ID,
                trust_remote_code=True,
                dtype=runtime["dtype"],
                device=0 if runtime["use_cuda"] else -1,
            )
            log("Pipeline loaded.")
        except Exception as exc:
            log(f"Pipeline load failed ({exc}), falling back to tokenizer+model.")

    if active_pipeline is None:
        log(f"Loading tokenizer for {MODEL_ID} ...")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID, trust_remote_code=True,
        )
        log("Tokenizer loaded. Loading model (this may download files on first run) ...")

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            dtype=runtime["dtype"],
            device_map=runtime["device_map"],
            trust_remote_code=True,
        )
        log(f"Model loaded. Device: {model.device}")

        if tokenizer.eos_token_id is not None:
            eos_ids.append(tokenizer.eos_token_id)
        extra = tokenizer.convert_tokens_to_ids(["<|im_end|>", "<|endoftext|>"])
        for tid in extra:
            if isinstance(tid, int) and tid not in eos_ids:
                eos_ids.append(tid)
        log(f"EOS token ids: {eos_ids}")

    def check_dashboard_command() -> Optional[Dict[str, Any]]:
        """Read and consume a command file written by the dashboard."""
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
    ):
        env = StudentEnvironment(scenario_name=scenario_name)
        obs = env.reset()

        if custom_ground_truth and env._latent:
            gt = custom_ground_truth
            latent = env._latent  # This is a FullLatentState object

            # 1. Map 'target_projects' to completed_projects
            if gt.get("target_projects"):
                latent.completed_projects = gt["target_projects"]

            # 2. Map 'subject_difficulty' to latent.true_exam_difficulty
            if gt.get("subject_difficulty"):
                latent.latent.true_exam_difficulty = {
                    k: float(v) for k, v in gt["subject_difficulty"].items()
                }

            # 3. Map 'initial_knowledge' to true_knowledge
            if gt.get("initial_knowledge"):
                latent.true_knowledge = {
                    k: float(v) for k, v in gt["initial_knowledge"].items()
                }

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
            cmd = check_dashboard_command()
            if cmd and cmd.get("action") == "restart":
                log("\n[DASHBOARD] Restart requested — ending episode early.")
                break

            user_msg = build_observation_prompt(obs)

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ]

            if active_pipeline is not None:
                prompt = f"{SYSTEM_PROMPT}\n\n{user_msg}"
            else:
                try:
                    prompt = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=ENABLE_THINKING,
                    )
                except TypeError:
                    prompt = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )

            t0 = time.time()
            if active_pipeline is not None:
                response = run_with_pipeline(active_pipeline, prompt)
                if not response:
                    response = format_observation(obs)
            else:
                assert tokenizer is not None and model is not None
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                n_input = inputs["input_ids"].shape[1]
                max_new = 2048 if ENABLE_THINKING else 300
                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=max_new,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.8,
                        top_k=20,
                        repetition_penalty=1.3,
                        eos_token_id=eos_ids if eos_ids else None,
                    )
                new_tokens = output_ids[0][n_input:]
                response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            gen_time = time.time() - t0

            thinking = ""
            if ENABLE_THINKING:
                think_match = re.search(
                    r"<think>(.*?)</think>", response, re.DOTALL
                )
                if think_match:
                    thinking = think_match.group(1).strip()
                    response = response[think_match.end():].strip()
                elif response.startswith("<think>"):
                    parts = response.split("</think>", 1)
                    if len(parts) == 2:
                        thinking = parts[0].replace("<think>", "").strip()
                        response = parts[1].strip()

            is_last_step = (step == MAX_EPISODE_STEPS - 1)

            action = parse_action(response)
            if action is None:
                if is_last_step:
                    log(f"\n  [!] Parse failed on final step — forcing synthesize_conclusion.")
                    action = ExperimentAction(
                        action_type=ActionType.SYNTHESIZE_CONCLUSION,
                        justification="forced terminal conclusion",
                        confidence=0.5,
                    )
                else:
                    log(f"\n  [!] Parse failed, skipping step. Raw: {response[:150]}")
                    continue

            completed_types = {
                r.action_type for r in obs.pipeline_history if r.success
            }
            failed_types = {
                r.action_type
                for r in obs.pipeline_history
                if not r.success
            }

            if should_force_terminal_conclusion(action, completed_types):
                log(
                    f"\n  [!] repeated completed meta step {action.action_type.value} "
                    f"— forcing synthesize_conclusion."
                )
                action = ExperimentAction(
                    action_type=ActionType.SYNTHESIZE_CONCLUSION,
                    justification="repeated completed meta step forced terminal conclusion",
                    confidence=action.confidence,
                )
                completed_types = {
                    r.action_type for r in obs.pipeline_history if r.success
                }

            skip_reason = None
            if action.action_type in completed_types:
                skip_reason = (
                    f"blocked repeat of completed step {action.action_type.value}"
                )
            elif action.action_type in failed_types:
                if should_block_failed_reattempt(
                    obs.pipeline_history, action.action_type
                ):
                    skip_reason = (
                        f"blocked re-attempt of failed step {action.action_type.value}"
                    )

            if skip_reason:
                if is_last_step:
                    log(f"\n  [!] {skip_reason} on final step — forcing synthesize_conclusion.")
                    action = ExperimentAction(
                        action_type=ActionType.SYNTHESIZE_CONCLUSION,
                        justification="forced terminal conclusion",
                        confidence=0.5,
                    )
                else:
                    log(f"\n  [!] {skip_reason}, skipping step.")
                    continue

            if is_last_step and action.action_type != ActionType.SYNTHESIZE_CONCLUSION:
                log(f"\n  [!] Final step — overriding {action.action_type.value} with synthesize_conclusion.")
                action = ExperimentAction(
                    action_type=ActionType.SYNTHESIZE_CONCLUSION,
                    justification="forced terminal conclusion",
                    confidence=action.confidence,
                )

            action = ensure_conclusion_claims(obs, action)

            log(f"\nStep {step + 1}: {action.action_type.value}  ({gen_time:.1f}s)")
            if thinking:
                log(f"  Thinking: {thinking[:200]}")
            if action.justification:
                log(f"  Rationale: {action.justification}")
            else:
                log("  Rationale: [model did not provide one]")
            if action.parameters:
                log(f"  Parameters: {compact_preview(action.parameters, 200)}")
            elif not action.justification and response:
                log(
                    f"  Model response: "
                    f"{compact_preview(response, MODEL_RESPONSE_PREVIEW_CHARS)}"
                )

            obs = env.step(action)

            if obs.latest_output:
                lo = obs.latest_output
                status = "OK" if lo.success else "FAIL"
                log(f"  [{status}] {lo.summary}")
                if lo.warnings:
                    log(f"  Warnings: {lo.warnings}")

            step_reward = obs.reward
            cumulative_reward += step_reward
            log(f"  Reward: {step_reward:+.3f}  (cum: {cumulative_reward:+.3f})")
            log(f"  Energy: {obs.energy:.1f} | Day: {obs.day} / 365")

            write_dashboard_state(
                env, obs,
                step=step + 1,
                cumulative_reward=cumulative_reward,
                model_response=response,
                model_thinking=thinking,
                action=action,
                gen_time=gen_time,
                episode_done=obs.done,
            )

            if obs.rule_violations:
                log(f"  Violations: {obs.rule_violations}")

            if obs.done:
                break

        log(f"\n{'=' * 70}")
        log("EPISODE COMPLETE" if obs.done else f"MAX STEPS ({MAX_EPISODE_STEPS})")
        log(f"  Steps: {obs.day}")
        log(f"  Total reward: {cumulative_reward:+.3f}")
        log(f"  Energy remaining: {obs.energy:.1f}")
        log(f"  Days used: {obs.day} / 365")
        if obs.completed_projects:
            log(f"  Completed projects: {obs.completed_projects}")
        log("=" * 70)

    try:
        DASHBOARD_CMD_PATH.unlink(missing_ok=True)
    except OSError:
        pass
    run_episode()

    while True:
        log("\nWaiting for dashboard command (restart / new task) ...")
        while True:
            cmd = check_dashboard_command()
            if cmd:
                break
            time.sleep(1.0)

        action_type = cmd.get("action", "restart")
        if action_type == "quit":
            log("Quit requested.")
            break

        scenario = cmd.get("scenario_name")
        ground_truth = cmd.get("ground_truth")
        log(f"\n[DASHBOARD] {action_type} — scenario={scenario}")
        run_episode(scenario_name=scenario, custom_ground_truth=ground_truth)


if __name__ == "__main__":
    main()