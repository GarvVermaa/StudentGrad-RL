"""Run the bio-experiment environment with a quantized Unsloth model."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional

from models import ActionType, ExperimentAction
from server.hackathon_environment import BioExperimentEnvironment
from training_unsloth import (
    DEFAULT_MAX_SEQ_LENGTH,
    generate_action_with_model,
    load_model_artifacts,
)
from training_script import DEFAULT_COMPLETION_TOKEN_BUDGET

import run_agent as base

MODEL_ID = os.getenv("RUN_AGENT_UNSLOTH_MODEL_ID", "unsloth/Qwen3.5-2B-GGUF")
MAX_EPISODE_STEPS = int(
    os.getenv("RUN_AGENT_UNSLOTH_MAX_EPISODE_STEPS", str(base.MAX_EPISODE_STEPS))
)
MAX_NEW_TOKENS = int(
    os.getenv(
        "RUN_AGENT_UNSLOTH_MAX_NEW_TOKENS",
        str(DEFAULT_COMPLETION_TOKEN_BUDGET),
    )
)
MAX_SEQ_LENGTH = int(
    os.getenv("RUN_AGENT_UNSLOTH_MAX_SEQ_LENGTH", str(DEFAULT_MAX_SEQ_LENGTH))
)
TRUST_REMOTE_CODE = (
    os.getenv("RUN_AGENT_UNSLOTH_TRUST_REMOTE_CODE", "1").strip().lower()
    not in {"0", "false", "off"}
)
LOAD_IN_4BIT = (
    os.getenv("RUN_AGENT_UNSLOTH_LOAD_IN_4BIT", "1").strip().lower()
    not in {"0", "false", "off"}
)
FAST_INFERENCE = (
    os.getenv("RUN_AGENT_UNSLOTH_FAST_INFERENCE", "0").strip().lower()
    not in {"0", "false", "off"}
)


def check_dashboard_command() -> Optional[Dict[str, Any]]:
    try:
        raw = base.DASHBOARD_CMD_PATH.read_text(encoding="utf-8")
        base.DASHBOARD_CMD_PATH.unlink(missing_ok=True)
        return json.loads(raw)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def run_episode(
    model: Any,
    tokenizer: Any,
    *,
    scenario_name: Optional[str] = None,
    custom_ground_truth: Optional[Dict[str, Any]] = None,
) -> None:
    env = BioExperimentEnvironment(scenario_name=scenario_name)
    obs = env.reset()

    if custom_ground_truth and env._latent:
        gt = custom_ground_truth
        bio = env._latent.biology
        if gt.get("true_markers"):
            bio.true_markers = gt["true_markers"]
        if gt.get("causal_mechanisms"):
            bio.causal_mechanisms = gt["causal_mechanisms"]
        if gt.get("true_pathways"):
            bio.true_pathways = {
                key: float(value) for key, value in gt["true_pathways"].items()
            }

    base.log("\n" + "=" * 70)
    base.log(f"TASK: {obs.task.problem_statement}")
    base.log(f"Conditions: {obs.task.conditions}")
    base.log(
        f"Budget: ${obs.task.budget_limit:,.0f} | "
        f"Time: {obs.task.time_limit_days:.0f} days"
    )
    base.log("Runtime: Unsloth quantized generation")
    base.log("=" * 70)

    cumulative_reward = 0.0
    base.write_dashboard_state(env, obs, step=0, cumulative_reward=0.0)

    for step in range(MAX_EPISODE_STEPS):
        cmd = check_dashboard_command()
        if cmd and cmd.get("action") == "restart":
            base.log("\n[DASHBOARD] Restart requested - ending episode early.")
            break

        t0 = time.time()
        result = generate_action_with_model(
            model,
            tokenizer,
            obs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.2,
            top_p=0.9,
            do_sample=True,
        )
        response = result["response_text"]
        action = result["action"]
        gen_time = time.time() - t0

        is_last_step = step == MAX_EPISODE_STEPS - 1
        if action is None:
            if is_last_step:
                base.log("\n  [!] Parse failed on final step - forcing synthesize_conclusion.")
                action = ExperimentAction(
                    action_type=ActionType.SYNTHESIZE_CONCLUSION,
                    justification="forced terminal conclusion",
                    confidence=0.5,
                )
            else:
                base.log(
                    f"\n  [!] Parse failed, skipping step. Raw: {response[:150]}"
                )
                continue

        completed_types = {
            record.action_type for record in obs.pipeline_history if record.success
        }
        failed_types = {
            record.action_type for record in obs.pipeline_history if not record.success
        }

        if base.should_force_terminal_conclusion(action, completed_types):
            base.log(
                f"\n  [!] repeated completed meta step {action.action_type.value} "
                f"- forcing synthesize_conclusion."
            )
            action = ExperimentAction(
                action_type=ActionType.SYNTHESIZE_CONCLUSION,
                justification="repeated completed meta step forced terminal conclusion",
                confidence=action.confidence,
            )
            completed_types = {
                record.action_type for record in obs.pipeline_history if record.success
            }

        skip_reason = None
        if action.action_type in completed_types:
            skip_reason = f"blocked repeat of completed step {action.action_type.value}"
        elif action.action_type in failed_types:
            if base.should_block_failed_reattempt(obs.pipeline_history, action.action_type):
                skip_reason = (
                    f"blocked re-attempt of failed step {action.action_type.value}"
                )

        if skip_reason:
            if is_last_step:
                base.log(
                    f"\n  [!] {skip_reason} on final step - forcing synthesize_conclusion."
                )
                action = ExperimentAction(
                    action_type=ActionType.SYNTHESIZE_CONCLUSION,
                    justification="forced terminal conclusion",
                    confidence=0.5,
                )
            else:
                base.log(f"\n  [!] {skip_reason}, skipping step.")
                continue

        if is_last_step and action.action_type != ActionType.SYNTHESIZE_CONCLUSION:
            base.log(
                f"\n  [!] Final step - overriding {action.action_type.value} "
                "with synthesize_conclusion."
            )
            action = ExperimentAction(
                action_type=ActionType.SYNTHESIZE_CONCLUSION,
                justification="forced terminal conclusion",
                confidence=action.confidence,
            )

        action = base.ensure_conclusion_claims(obs, action)

        base.log(f"\nStep {step + 1}: {action.action_type.value}  ({gen_time:.1f}s)")
        if action.justification:
            base.log(f"  Rationale: {action.justification}")
        else:
            base.log("  Rationale: [model did not provide one]")
        if action.parameters:
            base.log(f"  Parameters: {base.compact_preview(action.parameters, 200)}")
        elif response:
            base.log(
                "  Model response: "
                f"{base.compact_preview(response, base.MODEL_RESPONSE_PREVIEW_CHARS)}"
            )

        obs = env.step(action)

        if obs.latest_output:
            latest_output = obs.latest_output
            status = "OK" if latest_output.success else "FAIL"
            base.log(f"  [{status}] {latest_output.summary}")
            if latest_output.warnings:
                base.log(f"  Warnings: {latest_output.warnings}")

        step_reward = obs.reward
        cumulative_reward += step_reward
        base.log(f"  Reward: {step_reward:+.3f}  (cum: {cumulative_reward:+.3f})")
        base.log(
            f"  Budget: ${obs.resource_usage.budget_remaining:,.0f} | "
            f"Time: {obs.resource_usage.time_remaining_days:.0f}d"
        )

        base.write_dashboard_state(
            env,
            obs,
            step=step + 1,
            cumulative_reward=cumulative_reward,
            model_response=response,
            action=action,
            gen_time=gen_time,
            episode_done=obs.done,
        )

        if obs.rule_violations:
            base.log(f"  Violations: {obs.rule_violations}")
        if obs.done:
            break

    base.log(f"\n{'=' * 70}")
    base.log("EPISODE COMPLETE" if obs.done else f"MAX STEPS ({MAX_EPISODE_STEPS})")
    base.log(f"  Steps: {obs.step_index}")
    base.log(f"  Total reward: {cumulative_reward:+.3f}")
    base.log(f"  Budget used: ${obs.resource_usage.budget_used:,.0f}")
    base.log(f"  Time used: {obs.resource_usage.time_used_days:.0f} days")
    if obs.conclusions:
        base.log("  Conclusions:")
        for conclusion in obs.conclusions:
            base.log(
                f"    [{conclusion.claim_type}, conf={conclusion.confidence:.2f}] "
                f"{conclusion.claim}"
            )
            if conclusion.top_markers:
                base.log(f"      Markers: {conclusion.top_markers}")
            if conclusion.causal_mechanisms:
                base.log(f"      Mechanisms: {conclusion.causal_mechanisms}")
            if conclusion.predicted_pathways:
                base.log(f"      Pathways: {conclusion.predicted_pathways}")
    base.log("=" * 70)


def main() -> None:
    runtime = base.resolve_torch_runtime()
    base.log(
        f"Using Unsloth runtime: device={runtime['device']} "
        f"name={runtime['device_name']} dtype={runtime['dtype']}"
    )
    tokenizer, model = load_model_artifacts(
        MODEL_ID,
        trust_remote_code=TRUST_REMOTE_CODE,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=LOAD_IN_4BIT,
        fast_inference=FAST_INFERENCE,
        prepare_for_inference=True,
    )
    base.DASHBOARD_CMD_PATH.unlink(missing_ok=True)
    run_episode(model, tokenizer)

    while True:
        base.log("\nWaiting for dashboard command (restart / new task) ...")
        while True:
            cmd = check_dashboard_command()
            if cmd:
                break
            time.sleep(1.0)

        action_type = cmd.get("action", "restart")
        if action_type == "quit":
            base.log("Quit requested.")
            break

        scenario = cmd.get("scenario_name")
        ground_truth = cmd.get("ground_truth")
        base.log(f"\n[DASHBOARD] {action_type} - scenario={scenario}")
        run_episode(
            model,
            tokenizer,
            scenario_name=scenario,
            custom_ground_truth=ground_truth,
        )


if __name__ == "__main__":
    main()
