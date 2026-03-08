"""Run the bio-experiment environment with Qwen3.5-2B as the planning agent."""

from __future__ import annotations

import json
import re
import sys
import time
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from models import ActionType, ExperimentAction, ExperimentObservation
from server.hackathon_environment import BioExperimentEnvironment

MODEL_ID = "Qwen/Qwen3.5-0.8B"
MAX_EPISODE_STEPS = 12
PIPELINE_TASK = "image-text-to-text"
USE_PIPELINE = True

ACTION_TYPES = [a.value for a in ActionType]

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


def parse_action(text: str) -> Optional[ExperimentAction]:
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        d = json.loads(match.group())
    except json.JSONDecodeError:
        return None

    action_type = d.get("action_type")
    if action_type not in ACTION_TYPES:
        return None

    return ExperimentAction(
        action_type=ActionType(action_type),
        method=d.get("method"),
        parameters=d.get("parameters") or {},
        justification=d.get("justification"),
        confidence=min(1.0, max(0.0, float(d.get("confidence", 0.5)))),
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
    attempts = [
        {"text": prompt},
        {"text": prompt, "image": None},
        {"image": prompt},
    ]

    for payload in attempts:
        try:
            result = pipe(payload, max_new_tokens=220)
            if isinstance(result, list) and result:
                result = result[0]
            if isinstance(result, dict):
                text = result.get("generated_text") or result.get("text") or result.get("answer")
            elif isinstance(result, str):
                text = result
            else:
                text = ""
            if isinstance(text, str) and text.strip():
                return text.strip()
        except Exception:
            continue

    return ""


def main():
    tokenizer = None
    model = None
    eos_ids: List[int] = []
    active_pipeline = None

    if USE_PIPELINE:
        log(f"Loading pipeline ({PIPELINE_TASK}) for {MODEL_ID} ...")
        try:
            active_pipeline = pipeline(
                PIPELINE_TASK,
                model=MODEL_ID,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )
            log("Pipeline loaded.")
        except Exception as exc:
            log(f"Pipeline load failed ({exc}), falling back to tokenizer+model.")

    if active_pipeline is None:
        log(f"Loading tokenizer for {MODEL_ID} ...")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID, trust_remote_code=True,
        )
        log("Tokenizer loaded. Loading model (this downloads ~4 GB on first run) ...")

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
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
        if active_pipeline is not None:
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
