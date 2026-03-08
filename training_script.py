"""Train a self-driving lab planner with TRL GRPO and OpenEnv rewards."""

from __future__ import annotations

import argparse
import json
import random
import re
from numbers import Real
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from client import BioExperimentEnv
from models import (
    ActionType,
    ExperimentAction,
    ExperimentObservation,
    build_agent_observation_context,
    build_agent_system_prompt,
)
from server.hackathon_environment import BioExperimentEnvironment
from server.tasks.scenarios import SCENARIO_LIBRARY

DEFAULT_MODEL_ID = "Qwen/Qwen3.5-4B"
DEFAULT_OUTPUT_DIR = "training/grpo-output"
DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_COMPLETION_TOKEN_BUDGET = 160
INVALID_ACTION_PENALTY = -2.0
ENVIRONMENT_ERROR_PENALTY = -4.0

SYSTEM_PROMPT = build_agent_system_prompt()

ACTION_TYPES = [action.value for action in ActionType]
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

HEURISTIC_SEQUENCE = [
    ActionType.COLLECT_SAMPLE,
    ActionType.SELECT_COHORT,
    ActionType.PREPARE_LIBRARY,
    ActionType.SEQUENCE_CELLS,
    ActionType.RUN_QC,
    ActionType.FILTER_DATA,
    ActionType.NORMALIZE_DATA,
    ActionType.INTEGRATE_BATCHES,
    ActionType.CLUSTER_CELLS,
    ActionType.DIFFERENTIAL_EXPRESSION,
    ActionType.PATHWAY_ENRICHMENT,
    ActionType.MARKER_SELECTION,
    ActionType.TRAJECTORY_ANALYSIS,
    ActionType.REGULATORY_NETWORK_INFERENCE,
    ActionType.SYNTHESIZE_CONCLUSION,
]

VALID_ACTION_TYPES = set(ACTION_TYPES)


def compact_preview(value: Any, max_chars: int = 160) -> str:
    try:
        text = json.dumps(value, ensure_ascii=True, sort_keys=True)
    except TypeError:
        text = str(value)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


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


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a GRPO policy against the OpenEnv bio experiment environment."
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset-episodes", type=int, default=8)
    parser.add_argument("--rollout-steps", type=int, default=6)
    parser.add_argument(
        "--collection-policy",
        choices=["random", "heuristic"],
        default="heuristic",
        help="Policy used to build prompt states for GRPO training.",
    )
    parser.add_argument(
        "--reward-backend",
        choices=["local", "remote"],
        default="local",
        help="Use local in-process scoring or a live OpenEnv server.",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="Base URL for the OpenEnv server when reward-backend=remote.",
    )
    parser.add_argument(
        "--scenario-name",
        action="append",
        default=None,
        help="Repeatable scenario selector. Defaults to all curated scenarios.",
    )
    parser.add_argument(
        "--domain-randomise",
        action="store_true",
        help="Enable domain randomisation while building prompts and local rewards.",
    )
    parser.add_argument("--num-generations", type=int, default=2)
    parser.add_argument(
        "--max-completion-length",
        type=int,
        default=DEFAULT_COMPLETION_TOKEN_BUDGET,
    )
    parser.add_argument("--max-prompt-length", type=int, default=768)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument(
        "--plot-metric-key",
        default=None,
        help="Optional extra metric key from trainer log history to plot.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--load-model-only",
        action="store_true",
        help="Download and load the selected model and tokenizer, then exit.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to model/tokenizer loading.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the prompt dataset and smoke-test the reward function without training.",
    )
    parser.add_argument(
        "--push-to-hub",
        type=str,
        default=None,
        help="HuggingFace Hub repo id to push the trained model to (e.g. 'myuser/my-model').",
    )
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    return build_argument_parser().parse_args(argv)


def make_training_args(**overrides: Any) -> argparse.Namespace:
    """Build an argparse-style namespace for notebooks and scripts."""
    parser = build_argument_parser()
    defaults = vars(parser.parse_args([]))
    unknown = sorted(set(overrides) - set(defaults))
    if unknown:
        raise ValueError(f"Unknown training args: {', '.join(unknown)}")
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def format_observation(obs: ExperimentObservation) -> str:
    parts = [
        f"TASK: {obs.task.problem_statement}",
        f"Organism: {obs.task.organism} | Tissue: {obs.task.tissue}",
        f"Conditions: {', '.join(obs.task.conditions) or 'N/A'}",
        (
            "Step: "
            f"{obs.step_index} | Budget: ${obs.resource_usage.budget_remaining:,.0f} "
            f"| Time: {obs.resource_usage.time_remaining_days:.0f}d"
        ),
    ]
    context = build_agent_observation_context(obs, max_tools=5, max_assays=2)
    if context:
        parts.append(context)
    if obs.pipeline_history:
        last5 = obs.pipeline_history[-5:]
        parts.append("Recent history:")
        for step in last5:
            tag = "OK" if step.success else "FAIL"
            line = f"  [{tag}] {step.action_type.value}"
            if step.method:
                line += f" ({step.method})"
            line += f": {step.output_summary[:80]}"
            parts.append(line)
        completed = {
            step.action_type for step in obs.pipeline_history if step.success
        }
        if completed:
            parts.append(
                "Completed steps (do NOT repeat): "
                + ", ".join(sorted(action.value for action in completed))
            )
        remaining = [
            action.value for action in HEURISTIC_SEQUENCE if action not in completed
        ]
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
    if obs.candidate_mechanisms:
        parts.append(f"Candidate mechanisms: {obs.candidate_mechanisms[:5]}")
    parts.append(
        'Output ONLY a single JSON object with these exact keys, no comments, no extra text:\n'
        '{"action_type": "<one of the remaining steps>", "method": null, "parameters": {}, "justification": "<why>", "confidence": 0.8}'
    )
    return "\n".join(parts)


def build_training_prompt(obs: ExperimentObservation) -> str:
    return f"{SYSTEM_PROMPT}\n\n{format_observation(obs)}"


def heuristic_next_action(history: Sequence[ActionType], step_index: int) -> ActionType:
    seen = set(history)
    for action in HEURISTIC_SEQUENCE:
        if action not in seen:
            return action
    if step_index >= 2 and ActionType.VALIDATE_MARKER not in seen:
        return ActionType.VALIDATE_MARKER
    return ActionType.SYNTHESIZE_CONCLUSION


def pick_action(policy: str, step_index: int, history: Sequence[ActionType]) -> ActionType:
    if policy == "random":
        return random.choice(list(ActionType))
    return heuristic_next_action(history, step_index)


def default_comparison_name(conditions: Sequence[str]) -> str:
    normalized = {condition.lower() for condition in conditions}
    if {"healthy", "ipf"} <= normalized:
        return "IPF_vs_healthy"
    if any("treated" in condition for condition in normalized) and any(
        "untreated" in condition for condition in normalized
    ):
        return "treated_vs_untreated"
    return "disease_vs_healthy"


def build_experiment_action(
    action_type: ActionType,
    discovered_markers: Sequence[str],
    candidate_mechanisms: Sequence[str],
    conditions: Sequence[str],
) -> ExperimentAction:
    method = None
    parameters: Dict[str, object] = {}
    justification = f"Advance the experiment with {action_type.value}."

    if action_type == ActionType.COLLECT_SAMPLE:
        parameters = {"n_samples": 6}
        justification = "Collect enough samples to start the experiment."
    elif action_type == ActionType.SELECT_COHORT:
        parameters = {
            "comparison": default_comparison_name(conditions),
            "conditions": list(conditions[:2]) or ["disease", "healthy"],
        }
        justification = "Define the cohort split before committing to downstream analysis."
    elif action_type == ActionType.PREPARE_LIBRARY:
        method = "10x_chromium"
        justification = "Prepare a single-cell library for sequencing."
    elif action_type == ActionType.CULTURE_CELLS:
        method = "organoid_culture"
        parameters = {"duration_days": 7}
        justification = "Expand viable cells before a perturbation or profiling step."
    elif action_type == ActionType.PERTURB_GENE:
        method = "CRISPRi"
        parameters = {"target_gene": candidate_mechanisms[0] if candidate_mechanisms else "STAT3"}
        justification = "Test whether a candidate regulator causally shifts cell state."
    elif action_type == ActionType.PERTURB_COMPOUND:
        method = "small_molecule_screen"
        parameters = {"compound": candidate_mechanisms[0] if candidate_mechanisms else "TGFb_inhibitor"}
        justification = "Probe the pathway hypothesis with a targeted compound perturbation."
    elif action_type == ActionType.SEQUENCE_CELLS:
        method = "NovaSeq"
        justification = "Generate reads for downstream single-cell analysis."
    elif action_type == ActionType.RUN_QC:
        method = "scanpy.pp.calculate_qc_metrics"
        justification = "Measure technical quality before filtering."
    elif action_type == ActionType.FILTER_DATA:
        method = "scanpy.pp.filter_cells"
        justification = "Remove low-quality cells and technical artifacts."
    elif action_type == ActionType.NORMALIZE_DATA:
        method = "scanpy.pp.normalize_total"
        justification = "Normalize counts for comparable expression profiles."
    elif action_type == ActionType.INTEGRATE_BATCHES:
        method = "scanorama.integrate"
        justification = "Correct batch effects before comparing cellular programs."
    elif action_type == ActionType.CLUSTER_CELLS:
        method = "scanpy.tl.leiden"
        justification = "Resolve cell states before interpretation."
    elif action_type == ActionType.DIFFERENTIAL_EXPRESSION:
        method = "scanpy.tl.rank_genes_groups"
        parameters = {"comparison": default_comparison_name(conditions)}
        justification = "Identify genes associated with the phenotype of interest."
    elif action_type == ActionType.TRAJECTORY_ANALYSIS:
        method = "scanpy.tl.dpt"
        justification = "Recover pseudotime and lineage structure."
    elif action_type == ActionType.PATHWAY_ENRICHMENT:
        method = "gseapy.prerank"
        justification = "Translate gene-level changes into pathway programs."
    elif action_type == ActionType.MARKER_SELECTION:
        method = "scanpy.tl.rank_genes_groups"
        justification = "Nominate marker genes for validation."
    elif action_type == ActionType.REGULATORY_NETWORK_INFERENCE:
        method = "pySCENIC"
        justification = "Infer upstream regulators behind the observed state changes."
    elif action_type == ActionType.VALIDATE_MARKER:
        method = "qPCR"
        parameters = {"marker": discovered_markers[0] if discovered_markers else "SPP1"}
        justification = "Validate the strongest discovered marker."
    elif action_type == ActionType.DESIGN_FOLLOWUP:
        method = "followup_plan"
        parameters = {"priority_hypothesis": candidate_mechanisms[0] if candidate_mechanisms else "fibrotic_activation"}
        justification = "Propose the next experiment to disambiguate remaining uncertainty."
    elif action_type == ActionType.REQUEST_SUBAGENT_REVIEW:
        method = "peer_review"
        parameters = {"focus": "experimental_design"}
        justification = "Request a review of the current self-driving lab plan."
    elif action_type == ActionType.SYNTHESIZE_CONCLUSION:
        top = list(discovered_markers[:5]) if discovered_markers else []
        parameters = {
            "claims": [{
                "top_markers": top,
                "causal_mechanisms": list(candidate_mechanisms[:5]),
                "predicted_pathways": {
                    mechanism: 0.6
                    for mechanism in list(candidate_mechanisms[:3])
                },
                "confidence": 0.6,
                "claim_type": "causal" if candidate_mechanisms else "correlational",
                "claim": f"Synthesis for {default_comparison_name(conditions)}.",
            }],
        }
        justification = "Summarize the current evidence into a conclusion."

    return ExperimentAction(
        action_type=action_type,
        method=method,
        parameters=parameters,
        justification=justification,
        confidence=0.75,
    )


def selected_scenarios(requested: Optional[Sequence[str]]) -> List[str]:
    from server.tasks.procedural_generator import generate_procedural_scenarios
    all_scenarios = list(SCENARIO_LIBRARY) + generate_procedural_scenarios(n=20, seed=42)
    available = [scenario.name for scenario in all_scenarios]
    if not requested:
        return available
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise ValueError(f"Unknown scenarios requested: {', '.join(unknown)}")
    return list(requested)


def action_completion_json(action: ExperimentAction) -> str:
    payload = {
        "action_type": action.action_type.value,
        "method": action.method,
        "parameters": action.parameters,
        "justification": action.justification,
        "confidence": action.confidence,
    }
    return json.dumps(payload)


def build_prompt_examples(
    *,
    dataset_episodes: int,
    rollout_steps: int,
    collection_policy: str,
    scenario_names: Sequence[str],
    seed: int,
    domain_randomise: bool,
) -> List[Dict[str, str]]:
    rng = random.Random(seed)
    examples: List[Dict[str, str]] = []
    scenario_cycle = list(scenario_names)
    rng.shuffle(scenario_cycle)

    for episode_idx in range(dataset_episodes):
        scenario_name = scenario_cycle[episode_idx % len(scenario_cycle)]
        env = BioExperimentEnvironment(
            scenario_name=scenario_name,
            domain_randomise=domain_randomise,
        )
        obs = env.reset()
        history_actions: List[ExperimentAction] = []

        for step_idx in range(rollout_steps):
            if obs.done:
                break

            next_action = build_experiment_action(
                action_type=pick_action(
                    collection_policy,
                    step_idx,
                    [action.action_type for action in history_actions],
                ),
                discovered_markers=obs.discovered_markers,
                candidate_mechanisms=obs.candidate_mechanisms,
                conditions=obs.task.conditions,
            )
            examples.append({
                "prompt": build_training_prompt(obs),
                "scenario_name": scenario_name,
                "history_actions": json.dumps(
                    [action.model_dump() for action in history_actions]
                ),
                "rng_seed": str(env._latent.rng_seed),
                "reference_action": action_completion_json(next_action),
                "problem_statement": obs.task.problem_statement,
            })

            history_actions.append(next_action)
            obs = env.step(next_action)

    return examples


def completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion.strip()
    if isinstance(completion, dict):
        return content_to_text(completion.get("content", ""))
    if isinstance(completion, list):
        for item in reversed(completion):
            if isinstance(item, dict) and "content" in item:
                text = content_to_text(item["content"])
                if text:
                    return text
            if isinstance(item, str) and item.strip():
                return item.strip()
    return str(completion).strip()


def content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                if isinstance(part.get("text"), str):
                    parts.append(part["text"])
                elif isinstance(part.get("content"), str):
                    parts.append(part["content"])
        return "".join(parts).strip()
    return str(content).strip()


def _repair_truncated_json(text: str) -> Optional[str]:
    """Try to repair JSON truncated mid-value (common with small LLMs)."""
    s = text.strip()
    if not s.startswith("{"):
        return None

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
        (("cohort",), ActionType.SELECT_COHORT.value),
        (("library",), ActionType.PREPARE_LIBRARY.value),
        (("culture",), ActionType.CULTURE_CELLS.value),
        (("perturb", "gene"), ActionType.PERTURB_GENE.value),
        (("perturb", "compound"), ActionType.PERTURB_COMPOUND.value),
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


def _unique_nonempty(items: Sequence[Any], limit: int = 5) -> List[str]:
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
) -> Tuple[List[str], List[str], Dict[str, float]]:
    top_markers = _unique_nonempty(list(obs.discovered_markers), limit=5)
    causal_mechanisms = _unique_nonempty(list(obs.candidate_mechanisms), limit=5)
    predicted_pathways: Dict[str, float] = {}

    for output in reversed(obs.all_outputs):
        if not output.success:
            continue
        data = output.data or {}

        if not top_markers:
            markers = data.get("markers", [])
            if isinstance(markers, list):
                top_markers = _unique_nonempty(markers, limit=5)
        if not causal_mechanisms:
            regulators = data.get("top_regulators", [])
            if isinstance(regulators, list):
                causal_mechanisms = _unique_nonempty(regulators, limit=5)
        if not predicted_pathways:
            for item in data.get("top_pathways", []):
                if not isinstance(item, dict):
                    continue
                pathway = normalize_optional_string(item.get("pathway"))
                score = item.get("score")
                if pathway and isinstance(score, (int, float)):
                    predicted_pathways[pathway] = float(score)
                    if len(predicted_pathways) >= 5:
                        break
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
    if isinstance(raw_claims, list):
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


def parse_action_completion(text: str) -> Optional[ExperimentAction]:
    payload = extract_json_object(text)
    if payload is not None:
        action_type = normalize_action_type(get_payload_value(payload, "action_type"))
        if action_type is None:
            return None

        parameters = get_payload_value(payload, "parameters", "params") or {}
        if not isinstance(parameters, dict):
            parameters = {}

        confidence = get_payload_value(payload, "confidence")
        if confidence is None:
            confidence = 0.5
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.5

        justification = get_payload_value(
            payload, "justification", "reasoning", "rationale", "reason"
        )
        if justification is not None and not isinstance(justification, str):
            justification = compact_preview(justification, 200)

        return ExperimentAction(
            action_type=ActionType(action_type),
            method=normalize_optional_string(get_payload_value(payload, "method")),
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

    return ExperimentAction(
        action_type=ActionType(action_type),
        method=normalize_optional_string(method),
        parameters={},
        justification=justification,
        confidence=min(1.0, max(0.0, confidence)),
    )


def decode_history_actions(history_actions: Optional[str]) -> List[ExperimentAction]:
    if not history_actions:
        return []
    raw_actions = json.loads(history_actions)
    return [
        ExperimentAction(**action_payload)
        for action_payload in raw_actions
        if isinstance(action_payload, dict)
    ]


def normalise_column(values: Any, length: int) -> List[Any]:
    if values is None:
        return [None] * length
    if isinstance(values, list):
        if len(values) == length:
            return values
        if len(values) == 1:
            return values * length
        return values[:length] + [None] * max(0, length - len(values))
    return [values] * length


class OpenEnvReward:
    """Reward function compatible with TRL GRPOTrainer."""

    def __init__(
        self,
        *,
        reward_backend: str,
        base_url: str,
        invalid_action_penalty: float = INVALID_ACTION_PENALTY,
        environment_error_penalty: float = ENVIRONMENT_ERROR_PENALTY,
        domain_randomise: bool = False,
    ) -> None:
        self.__name__ = "openenv_reward"
        self.reward_backend = reward_backend
        self.base_url = base_url
        self.invalid_action_penalty = invalid_action_penalty
        self.environment_error_penalty = environment_error_penalty
        self.domain_randomise = domain_randomise

    def __call__(
        self,
        completions: List[Any],
        scenario_name: Optional[List[str]] = None,
        history_actions: Optional[List[str]] = None,
        rng_seed: Optional[List[str]] = None,
        **_: Any,
    ) -> List[float]:
        scenario_names = normalise_column(scenario_name, len(completions))
        history_columns = normalise_column(history_actions, len(completions))
        seed_columns = normalise_column(rng_seed, len(completions))
        rewards: List[float] = []

        for completion, current_scenario, current_history, current_seed in zip(
            completions,
            scenario_names,
            history_columns,
            seed_columns,
        ):
            action = parse_action_completion(completion_to_text(completion))
            if action is None:
                rewards.append(self.invalid_action_penalty)
                continue

            try:
                if self.reward_backend == "remote":
                    reward = self._score_remote(action, current_scenario, current_history)
                else:
                    reward = self._score_local(action, current_scenario, current_history, current_seed)
            except Exception:
                reward = self.environment_error_penalty

            rewards.append(float(reward))

        return rewards

    def _score_local(
        self,
        action: ExperimentAction,
        scenario_name: Optional[str],
        history_actions: Optional[str],
        rng_seed: Optional[str] = None,
    ) -> float:
        env = BioExperimentEnvironment(
            scenario_name=scenario_name,
            domain_randomise=self.domain_randomise,
        )
        seed = int(rng_seed) if rng_seed else None
        obs = env.reset(seed=seed)
        for previous_action in decode_history_actions(history_actions):
            obs = env.step(previous_action)
            if obs.done:
                return float(obs.reward)
        action = ensure_conclusion_claims(obs, action)
        obs = env.step(action)
        return float(obs.reward)

    def _score_remote(
        self,
        action: ExperimentAction,
        scenario_name: Optional[str],
        history_actions: Optional[str],
    ) -> float:
        with BioExperimentEnv(base_url=self.base_url) as env:
            # NOTE: scenario_name is accepted for API parity with _score_local
            # but the OpenEnv HTTP protocol does not yet support passing it
            # through reset(). The server will use its configured default.
            result = env.reset()
            for previous_action in decode_history_actions(history_actions):
                result = env.step(previous_action)
                if result.done:
                    return float(result.reward or 0.0)
            result = env.step(action)
            if result.reward is not None:
                return float(result.reward)
            return float(result.observation.reward)


def is_numeric_log_value(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def available_numeric_log_keys(log_history: Sequence[Dict[str, Any]]) -> List[str]:
    keys = {
        key
        for entry in log_history
        if isinstance(entry, dict)
        for key, value in entry.items()
        if key != "step" and is_numeric_log_value(value)
    }
    return sorted(keys)


def extract_log_series(
    log_history: Sequence[Dict[str, Any]],
    key: Optional[str],
) -> List[Tuple[float, float]]:
    if not key:
        return []

    series: List[Tuple[float, float]] = []
    synthetic_step = 0
    for entry in log_history:
        if not isinstance(entry, dict) or key not in entry:
            continue
        value = entry.get(key)
        if not is_numeric_log_value(value):
            continue

        raw_step = entry.get("step")
        if is_numeric_log_value(raw_step):
            step = float(raw_step)
        else:
            synthetic_step += 1
            step = float(synthetic_step)

        series.append((step, float(value)))

    return series


def select_reward_key(log_history: Sequence[Dict[str, Any]]) -> Optional[str]:
    numeric_keys = available_numeric_log_keys(log_history)
    reward_keys = [key for key in numeric_keys if "reward" in key.lower()]
    if not reward_keys:
        return None

    preferred = [
        "reward",
        "mean_reward",
        "reward_mean",
        "rewards/open_env_reward",
    ]
    lowered = {key.lower(): key for key in reward_keys}
    for key in preferred:
        if key in lowered:
            return lowered[key]

    reward_keys.sort(key=lambda key: ("/" in key, len(key), key))
    return reward_keys[0]


def select_metric_key(
    log_history: Sequence[Dict[str, Any]],
    *,
    reward_key: Optional[str],
    requested_key: Optional[str] = None,
) -> Optional[str]:
    numeric_keys = available_numeric_log_keys(log_history)
    if requested_key:
        if requested_key not in numeric_keys:
            available = ", ".join(numeric_keys) or "none"
            raise ValueError(
                f"Requested plot metric '{requested_key}' was not logged. "
                f"Available numeric keys: {available}"
            )
        return requested_key

    excluded = {
        "epoch",
        "loss",
        "learning_rate",
        "step",
        "total_flos",
        "train_loss",
        "train_runtime",
        "train_samples_per_second",
        "train_steps_per_second",
    }
    if reward_key:
        excluded.add(reward_key)

    preferred = [
        "kl",
        "objective/kl",
        "completion_length",
        "mean_completion_length",
        "grad_norm",
        "entropy",
        "accuracy",
        "learning_rate",
        "epoch",
    ]
    numeric_set = set(numeric_keys)
    for key in preferred:
        if key in numeric_set and key not in excluded:
            return key

    candidates = [
        key for key in numeric_keys
        if key not in excluded and "reward" not in key.lower()
    ]
    if candidates:
        return candidates[0]

    for fallback in ("learning_rate", "epoch"):
        if fallback in numeric_set:
            return fallback

    return None


def save_plot(
    path: Path,
    *,
    series: Sequence[Tuple[float, float]],
    title: str,
    ylabel: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    if series:
        x_values, y_values = zip(*series)
        ax.plot(x_values, y_values, marker="o", linewidth=1.8)
    else:
        ax.text(
            0.5,
            0.5,
            "No logged data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_training_plots(
    log_history: Sequence[Dict[str, Any]],
    output_dir: str | Path,
    metric_key: Optional[str] = None,
) -> Dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    reward_key = select_reward_key(log_history)
    selected_metric_key = select_metric_key(
        log_history,
        reward_key=reward_key,
        requested_key=metric_key,
    )

    loss_series = extract_log_series(log_history, "loss")
    reward_series = extract_log_series(log_history, reward_key)
    metric_series = extract_log_series(log_history, selected_metric_key)

    loss_path = output_path / "training_loss.png"
    reward_path = output_path / "training_reward.png"
    metric_path = output_path / "training_metric.png"
    dashboard_path = output_path / "training_dashboard.png"
    manifest_path = output_path / "training_plot_manifest.json"

    save_plot(loss_path, series=loss_series, title="Training Loss", ylabel="Loss")
    save_plot(
        reward_path,
        series=reward_series,
        title=f"Training Reward ({reward_key or 'not logged'})",
        ylabel="Reward",
    )
    save_plot(
        metric_path,
        series=metric_series,
        title=f"Training Metric ({selected_metric_key or 'not logged'})",
        ylabel=selected_metric_key or "Metric",
    )

    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    plot_specs = [
        (axes[0], loss_series, "Training Loss", "Loss"),
        (axes[1], reward_series, f"Training Reward ({reward_key or 'not logged'})", "Reward"),
        (
            axes[2],
            metric_series,
            f"Training Metric ({selected_metric_key or 'not logged'})",
            selected_metric_key or "Metric",
        ),
    ]
    for axis, series, title, ylabel in plot_specs:
        if series:
            x_values, y_values = zip(*series)
            axis.plot(x_values, y_values, marker="o", linewidth=1.8)
        else:
            axis.text(
                0.5,
                0.5,
                "No logged data available",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
        axis.set_title(title)
        axis.set_xlabel("Step")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(dashboard_path, dpi=150)
    plt.close(fig)

    manifest = {
        "available_numeric_keys": available_numeric_log_keys(log_history),
        "reward_key": reward_key,
        "metric_key": selected_metric_key,
        "plots": {
            "loss": str(loss_path),
            "reward": str(reward_path),
            "metric": str(metric_path),
            "dashboard": str(dashboard_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest["plots"]


def run_dry_run_preview(
    examples: Sequence[Dict[str, str]],
    reward_fn: OpenEnvReward,
    output_dir: str,
) -> None:
    if not examples:
        raise ValueError("No training prompts were generated for the dry run.")

    sample = examples[0]
    sample_reward = reward_fn(
        completions=[[{"role": "assistant", "content": sample["reference_action"]}]],
        scenario_name=[sample["scenario_name"]],
        history_actions=[sample["history_actions"]],
    )[0]

    print(f"Built {len(examples)} prompt states.")
    print(f"Output directory: {Path(output_dir)}")
    print(f"Sample scenario: {sample['scenario_name']}")
    print(f"Sample reward for reference action: {sample_reward:+.3f}")
    print("\nSample prompt:\n")
    print(sample["prompt"])


def resolve_torch_runtime() -> Dict[str, Any]:
    import torch

    use_cuda = torch.cuda.is_available()
    bf16 = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)()) if use_cuda else False
    dtype = torch.bfloat16 if bf16 else (
        torch.float16 if use_cuda else torch.float32
    )
    return {
        "use_cuda": use_cuda,
        "device": "cuda:0" if use_cuda else "cpu",
        "dtype": dtype,
        "bf16": bf16,
        "fp16": use_cuda and not bf16,
        "device_name": torch.cuda.get_device_name(0) if use_cuda else "cpu",
    }


def _guard_invalid_torchao_version() -> None:
    """Treat malformed torchao installs as unavailable for HF imports."""
    import importlib.metadata as importlib_metadata
    from packaging.version import InvalidVersion, Version

    if getattr(importlib_metadata, "_openenv_torchao_guard_installed", False):
        return

    original_version = importlib_metadata.version

    def guarded_version(distribution_name: str) -> str:
        version = original_version(distribution_name)
        if distribution_name.lower() == "torchao":
            try:
                Version(version)
            except InvalidVersion as exc:
                raise importlib_metadata.PackageNotFoundError(
                    f"Malformed torchao version metadata: {version!r}"
                ) from exc
        return version

    importlib_metadata.version = guarded_version
    importlib_metadata._openenv_torchao_guard_installed = True


def load_model_artifacts(
    model_id: str,
    *,
    trust_remote_code: bool,
):
    _guard_invalid_torchao_version()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    runtime = resolve_torch_runtime()
    print(f"Loading tokenizer for {model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model for {model_id} ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
        torch_dtype=runtime["dtype"],
    )
    if runtime["use_cuda"]:
        model = model.to(runtime["device"])
    else:
        model = model.to("cpu")
    return tokenizer, model


def build_openenv_reward(args: argparse.Namespace) -> OpenEnvReward:
    """Return the OpenEnv-compatible reward callable used by GRPO."""
    return OpenEnvReward(
        reward_backend=args.reward_backend,
        base_url=args.base_url,
        domain_randomise=args.domain_randomise,
    )


def prepare_prompt_examples(args: argparse.Namespace) -> Dict[str, Any]:
    """Build the OpenEnv rollout states that seed GRPO prompts."""
    scenario_names = selected_scenarios(args.scenario_name)
    examples = build_prompt_examples(
        dataset_episodes=args.dataset_episodes,
        rollout_steps=args.rollout_steps,
        collection_policy=args.collection_policy,
        scenario_names=scenario_names,
        seed=args.seed,
        domain_randomise=args.domain_randomise,
    )
    return {
        "scenario_names": scenario_names,
        "examples": examples,
    }


def build_grpo_config(
    args: argparse.Namespace,
    runtime: Dict[str, Any],
):
    _guard_invalid_torchao_version()
    from trl import GRPOConfig

    return GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        max_prompt_length=args.max_prompt_length,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        bf16=runtime["bf16"],
        fp16=runtime["fp16"],
        report_to="none",
        remove_unused_columns=False,
    )


def build_grpo_trainer(
    *,
    model: Any,
    tokenizer: Any,
    reward_func: Any,
    train_dataset: Any,
    args: argparse.Namespace,
    runtime: Dict[str, Any],
):
    _guard_invalid_torchao_version()
    from trl import GRPOTrainer

    config = build_grpo_config(args, runtime)
    return GRPOTrainer(
        model=model,
        reward_funcs=reward_func,
        args=config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )


def generate_action_with_model(
    model: Any,
    tokenizer: Any,
    prompt_or_observation: str | ExperimentObservation,
    *,
    max_new_tokens: int = DEFAULT_COMPLETION_TOKEN_BUDGET,
    temperature: float = 0.2,
    top_p: float = 0.9,
    do_sample: bool = True,
) -> Dict[str, Any]:
    import torch

    if isinstance(prompt_or_observation, ExperimentObservation):
        prompt = build_training_prompt(prompt_or_observation)
    else:
        prompt = str(prompt_or_observation)

    model_device = getattr(model, "device", None)
    if model_device is None:
        model_device = resolve_torch_runtime()["device"]

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(model_device) for key, value in inputs.items()}
    prompt_tokens = inputs["input_ids"].shape[1]

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "temperature": temperature,
        "top_p": top_p,
        "pad_token_id": tokenizer.pad_token_id,
    }
    with torch.no_grad():
        output_ids = model.generate(**inputs, **generation_kwargs)

    new_tokens = output_ids[0][prompt_tokens:]
    response_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    action = parse_action_completion(response_text)
    if action is not None and isinstance(prompt_or_observation, ExperimentObservation):
        action = ensure_conclusion_claims(prompt_or_observation, action)
    return {
        "prompt": prompt,
        "response_text": response_text,
        "action": action,
    }


def run_training(args: argparse.Namespace) -> Dict[str, Any]:
    random.seed(args.seed)
    runtime = resolve_torch_runtime()

    if args.load_model_only:
        tokenizer, model = load_model_artifacts(
            args.model_id,
            trust_remote_code=args.trust_remote_code,
        )
        device = getattr(model, "device", "unknown")
        print(f"Model ready: {args.model_id}")
        print(f"Tokenizer vocab size: {len(tokenizer)}")
        print(f"Model device: {device}")
        print(f"Runtime device name: {runtime['device_name']}")
        return {
            "args": args,
            "runtime": runtime,
            "tokenizer": tokenizer,
            "model": model,
        }

    prompt_data = prepare_prompt_examples(args)
    scenario_names = prompt_data["scenario_names"]
    examples = prompt_data["examples"]
    reward_fn = build_openenv_reward(args)

    if args.dry_run:
        run_dry_run_preview(examples, reward_fn, args.output_dir)
        return {
            "args": args,
            "runtime": runtime,
            "scenario_names": scenario_names,
            "examples": examples,
            "reward_fn": reward_fn,
        }

    from datasets import Dataset
    train_dataset = Dataset.from_list(examples)
    tokenizer, model = load_model_artifacts(
        args.model_id,
        trust_remote_code=args.trust_remote_code,
    )

    print(
        f"Training runtime: device={runtime['device']} "
        f"name={runtime['device_name']} "
        f"dtype={runtime['dtype']}"
    )
    print(
        "OpenEnv reward: "
        f"backend={args.reward_backend} scenarios={len(scenario_names)} "
        f"examples={len(examples)}"
    )

    trainer = build_grpo_trainer(
        model=model,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        reward_func=reward_fn,
        args=args,
        runtime=runtime,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    if args.push_to_hub:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(repo_id=args.push_to_hub, repo_type="model", exist_ok=True)
        print(f"Pushing model to HuggingFace Hub: {args.push_to_hub}")
        api.upload_folder(
            folder_path=args.output_dir,
            repo_id=args.push_to_hub,
            repo_type="model",
            create_pr=False,
        )
        print(f"Model pushed to https://huggingface.co/{args.push_to_hub}")
    plot_paths = save_training_plots(
        trainer.state.log_history,
        args.output_dir,
        metric_key=args.plot_metric_key,
    )
    print("Saved training plots:")
    for plot_name, plot_path in plot_paths.items():
        print(f"  - {plot_name}: {plot_path}")

    return {
        "args": args,
        "runtime": runtime,
        "scenario_names": scenario_names,
        "examples": examples,
        "reward_fn": reward_fn,
        "train_dataset": train_dataset,
        "tokenizer": tokenizer,
        "model": model,
        "trainer": trainer,
        "plot_paths": plot_paths,
    }


def main() -> None:
    run_training(parse_args())


if __name__ == "__main__":
    main()
