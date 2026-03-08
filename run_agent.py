"""Run the bio-experiment environment with Qwen3.5-0.8B as the planning agent."""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, List, Optional

from models import ActionType, ExperimentAction, ExperimentObservation
from server.hackathon_environment import BioExperimentEnvironment

USE_OPENAI = os.getenv("RUN_AGENT_USE_OPENAI", "1").strip().lower() not in {"0", "false", "off"}
USE_PIPELINE = os.getenv("RUN_AGENT_USE_PIPELINE", "0").strip().lower() not in {"0", "false", "off"}

if not USE_OPENAI:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if USE_PIPELINE:
        from transformers import pipeline

MODEL_ID = "Qwen/Qwen3.5-0.8B"
MAX_EPISODE_STEPS = int(os.getenv("RUN_AGENT_MAX_EPISODE_STEPS", "12"))
PIPELINE_TASK = "text-generation"
OPENAI_MODEL = os.getenv("RUN_AGENT_OPENAI_MODEL", "gpt-4o-mini")
OPENAI_TIMEOUT_SECONDS = float(os.getenv("RUN_AGENT_OPENAI_TIMEOUT_SECONDS", "60"))
OPENAI_MAX_TOKENS = int(os.getenv("RUN_AGENT_OPENAI_MAX_TOKENS", "220"))

ACTION_TYPES = [a.value for a in ActionType]
ACTION_TYPE_ALIASES = {
    "collect_samples": ActionType.COLLECT_SAMPLE.value,
    "collect_sample_from_bone_marrow": ActionType.COLLECT_SAMPLE.value,
    "collect_samples_from_bone_marrow": ActionType.COLLECT_SAMPLE.value,
    "prepare_sc_library": ActionType.PREPARE_LIBRARY.value,
    "sequence_single_cells": ActionType.SEQUENCE_CELLS.value,
    "qc": ActionType.RUN_QC.value,
    "run_quality_control": ActionType.RUN_QC.value,
    "cluster": ActionType.CLUSTER_CELLS.value,
    "de_analysis": ActionType.DIFFERENTIAL_EXPRESSION.value,
    "differential_expression_analysis": ActionType.DIFFERENTIAL_EXPRESSION.value,
    "trajectory_inference": ActionType.TRAJECTORY_ANALYSIS.value,
    "infer_trajectory": ActionType.TRAJECTORY_ANALYSIS.value,
    "network_inference": ActionType.REGULATORY_NETWORK_INFERENCE.value,
    "select_markers": ActionType.MARKER_SELECTION.value,
    "final_conclusion": ActionType.SYNTHESIZE_CONCLUSION.value,
}

SYSTEM_PROMPT = """\
You are an expert biologist planning a single-cell experiment pipeline.

At each turn you see the experiment state and must pick the next step.

Action types (in typical order):
  collect_sample, prepare_library, sequence_cells, run_qc, filter_data,
  normalize_data, cluster_cells, differential_expression,
  pathway_enrichment, marker_selection, validate_marker, synthesize_conclusion

Other actions: select_cohort, culture_cells, perturb_gene, perturb_compound,
  integrate_batches, trajectory_analysis, regulatory_network_inference,
  design_followup_experiment, request_subagent_review

Respond with ONLY valid JSON, nothing else:
{"action_type": "...", "method": null, "parameters": {}, "justification": "...", "confidence": 0.8}
"""


def format_observation(obs: ExperimentObservation) -> str:
    parts = [
        f"TASK: {obs.task.problem_statement}",
        f"Organism: {obs.task.organism} | Tissue: {obs.task.tissue}",
        f"Conditions: {', '.join(obs.task.conditions) or 'N/A'}",
        f"Step: {obs.step_index} | Budget: ${obs.resource_usage.budget_remaining:,.0f} | Time: {obs.resource_usage.time_remaining_days:.0f}d",
    ]
    if obs.pipeline_history:
        last5 = obs.pipeline_history[-5:]
        parts.append("History:")
        for h in last5:
            tag = "OK" if h.success else "FAIL"
            parts.append(f"  [{tag}] {h.action_type.value}: {h.output_summary[:80]}")
    if obs.rule_violations:
        parts.append(f"VIOLATIONS: {obs.rule_violations}")
    if obs.discovered_markers:
        parts.append(f"Markers: {obs.discovered_markers[:5]}")
    return "\n".join(parts)


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    stripped = text.strip()
    fence_prefix = "```"
    if stripped.startswith(fence_prefix) and stripped.endswith(fence_prefix):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            stripped = "\n".join(lines[1:-1]).strip()

    candidates = [stripped]
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

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


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


def parse_action(text: str) -> Optional[ExperimentAction]:
    d = extract_json_object(text)
    if d is not None:
        action_type = normalize_action_type(d.get("action_type"))
        if action_type is None:
            return None

        parameters = d.get("parameters") or {}
        if not isinstance(parameters, dict):
            parameters = {}

        confidence = d.get("confidence", 0.5)
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.5

        return ExperimentAction(
            action_type=ActionType(action_type),
            method=d.get("method"),
            parameters=parameters,
            justification=d.get("justification"),
            confidence=min(1.0, max(0.0, confidence)),
        )

    action_match = re.search(r'"action_type"\s*:\s*"([^"]+)"', text)
    if not action_match:
        return None

    action_type = normalize_action_type(action_match.group(1))
    if action_type is None:
        return None

    method_match = re.search(r'"method"\s*:\s*"([^"]+)"', text)
    confidence_match = re.search(r'"confidence"\s*:\s*([0-9]*\.?[0-9]+)', text)
    justification_match = re.search(
        r'"justification"\s*:\s*"((?:[^"\\]|\\.)*)"',
        text,
        re.DOTALL,
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

    return ExperimentAction(
        action_type=ActionType(action_type),
        method=method_match.group(1) if method_match else None,
        parameters={},
        justification=justification,
        confidence=min(1.0, max(0.0, confidence)),
    )


FALLBACK_SEQUENCE = [
    ActionType.COLLECT_SAMPLE,
    ActionType.PREPARE_LIBRARY,
    ActionType.SEQUENCE_CELLS,
    ActionType.RUN_QC,
    ActionType.FILTER_DATA,
    ActionType.NORMALIZE_DATA,
    ActionType.CLUSTER_CELLS,
    ActionType.DIFFERENTIAL_EXPRESSION,
    ActionType.PATHWAY_ENRICHMENT,
    ActionType.MARKER_SELECTION,
    ActionType.SYNTHESIZE_CONCLUSION,
]


def fallback_action(step: int) -> ExperimentAction:
    idx = min(step, len(FALLBACK_SEQUENCE) - 1)
    return ExperimentAction(
        action_type=FALLBACK_SEQUENCE[idx],
        justification="fallback",
        confidence=0.3,
    )


def log(msg: str) -> None:
    print(msg, flush=True)


def build_observation_prompt(obs: ExperimentObservation) -> str:
    return format_observation(obs)


def run_with_pipeline(pipe, prompt: str) -> str:
    try:
        result = pipe(prompt, max_new_tokens=220, return_full_text=False)
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


def run_with_openai(messages: List[Dict[str, str]]) -> str:
    from openai_oauth_client import run_openai_chat

    return run_openai_chat(
        messages=messages,
        model=OPENAI_MODEL,
        max_tokens=OPENAI_MAX_TOKENS,
        timeout_seconds=OPENAI_TIMEOUT_SECONDS,
    )


def resolve_torch_runtime() -> Dict[str, Any]:
    assert not USE_OPENAI
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
    if USE_OPENAI:
        log(f"Using OpenAI chat model ({OPENAI_MODEL}) with OAuth token auth.")
    else:
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

    env = BioExperimentEnvironment()
    obs = env.reset()

    log("\n" + "=" * 70)
    log(f"TASK: {obs.task.problem_statement}")
    log(f"Conditions: {obs.task.conditions}")
    log(f"Budget: ${obs.task.budget_limit:,.0f} | Time: {obs.task.time_limit_days:.0f} days")
    log("=" * 70)

    cumulative_reward = 0.0

    for step in range(MAX_EPISODE_STEPS):
        user_msg = build_observation_prompt(obs)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

        if tokenizer is None:
            # Pipeline path usually ignores chat templates.
            prompt = f"{SYSTEM_PROMPT}\n\n{user_msg}"
        else:
            try:
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )

        t0 = time.time()
        if USE_OPENAI:
            response = run_with_openai(messages)
        elif active_pipeline is not None:
            response = run_with_pipeline(active_pipeline, prompt)
            if not response:
                response = format_observation(obs)
        else:
            assert tokenizer is not None and model is not None
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            n_input = inputs["input_ids"].shape[1]
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=200,
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

        action = parse_action(response)
        used_fallback = False
        if action is None:
            log(f"\n  [!] Parse failed, using fallback. Raw: {response[:150]}")
            action = fallback_action(step)
            used_fallback = True

        tag = " [FALLBACK]" if used_fallback else ""
        log(f"\nStep {step + 1}: {action.action_type.value}{tag}  ({gen_time:.1f}s)")
        if action.justification:
            log(f"  Rationale: {action.justification}")

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
        log(f"  Budget: ${obs.resource_usage.budget_remaining:,.0f} | Time: {obs.resource_usage.time_remaining_days:.0f}d")

        if obs.rule_violations:
            log(f"  Violations: {obs.rule_violations}")

        if obs.done:
            break

    log(f"\n{'=' * 70}")
    log("EPISODE COMPLETE" if obs.done else f"MAX STEPS ({MAX_EPISODE_STEPS})")
    log(f"  Steps: {obs.step_index}")
    log(f"  Total reward: {cumulative_reward:+.3f}")
    log(f"  Budget used: ${obs.resource_usage.budget_used:,.0f}")
    log(f"  Time used: {obs.resource_usage.time_used_days:.0f} days")
    if obs.conclusions:
        log("  Conclusions:")
        for c in obs.conclusions:
            log(f"    [{c.claim_type}, conf={c.confidence:.2f}] {c.claim}")
    log("=" * 70)


if __name__ == "__main__":
    main()
