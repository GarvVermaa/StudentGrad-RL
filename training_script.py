"""Collect trajectories with direct OpenEnv environment access."""

from __future__ import annotations

import argparse
import random
import uuid
from pathlib import Path
from typing import Dict, List, Sequence

from models import ActionType, ExperimentAction
from server.hackathon_environment import BioExperimentEnvironment
from training.evaluation import EvaluationSuite
from training.trajectory import Trajectory, TrajectoryDataset


HEURISTIC_SEQUENCE = [
    ActionType.COLLECT_SAMPLE,
    ActionType.PREPARE_LIBRARY,
    ActionType.SEQUENCE_CELLS,
    ActionType.RUN_QC,
    ActionType.FILTER_DATA,
    ActionType.NORMALIZE_DATA,
    ActionType.CLUSTER_CELLS,
    ActionType.TRAJECTORY_ANALYSIS,
    ActionType.MARKER_SELECTION,
    ActionType.SYNTHESIZE_CONCLUSION,
]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run rollout episodes and persist trajectories."
    )
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes.")
    parser.add_argument(
        "--policy",
        choices=["random", "heuristic"],
        default="heuristic",
        help="Policy to use for rollouts.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional hard cutoff per episode (defaults to env limit).",
    )
    parser.add_argument(
        "--output-dir",
        default="training/rollouts",
        help="Directory for JSON trajectory outputs.",
    )
    parser.add_argument("--seed", type=int, default=None, help="RNG seed.")
    return parser.parse_args()


def heuristic_next_action(history: Sequence[ActionType], step_index: int) -> ActionType:
    seen = set(history)
    for action in HEURISTIC_SEQUENCE:
        if action not in seen:
            return action
    if step_index >= 2 and ActionType.VALIDATE_MARKER not in seen:
        return ActionType.VALIDATE_MARKER
    if ActionType.SYNTHESIZE_CONCLUSION in seen:
        return ActionType.SYNTHESIZE_CONCLUSION
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
    if any("healthy" in condition for condition in normalized):
        return "disease_vs_healthy"
    return "disease_vs_healthy"


def build_experiment_action(
    action_type: ActionType,
    discovered_markers: Sequence[str],
    conditions: Sequence[str],
) -> ExperimentAction:
    method = None
    parameters: Dict[str, object] = {}

    if action_type == ActionType.COLLECT_SAMPLE:
        parameters = {"n_samples": 6}
    elif action_type == ActionType.PREPARE_LIBRARY:
        method = "10x_chromium"
    elif action_type == ActionType.RUN_QC:
        method = "scanpy.pp.calculate_qc_metrics"
    elif action_type == ActionType.FILTER_DATA:
        method = "scanpy.pp.filter_cells"
    elif action_type == ActionType.NORMALIZE_DATA:
        method = "scanpy.pp.normalize_total"
    elif action_type == ActionType.CLUSTER_CELLS:
        method = "scanpy.tl.leiden"
    elif action_type == ActionType.DIFFERENTIAL_EXPRESSION:
        method = "scanpy.tl.rank_genes_groups"
        parameters = {"comparison": default_comparison_name(conditions)}
    elif action_type == ActionType.TRAJECTORY_ANALYSIS:
        method = "scanpy.tl.dpt"
    elif action_type == ActionType.MARKER_SELECTION:
        method = "scanpy.tl.rank_genes_groups"
    elif action_type == ActionType.VALIDATE_MARKER:
        method = "qPCR"
        parameters = {"marker": discovered_markers[0] if discovered_markers else "SPP1"}
    elif action_type == ActionType.SYNTHESIZE_CONCLUSION:
        parameters = {"claims": []}

    return ExperimentAction(
        action_type=action_type,
        method=method,
        parameters=parameters,
        confidence=0.75,
    )


def run_episode(
    env: BioExperimentEnvironment,
    episode_id: str,
    policy: str,
    max_steps: int | None = None,
) -> Trajectory:
    structured_obs = env.reset()
    traj = Trajectory(
        episode_id=episode_id,
        task=structured_obs.task.model_dump(),
        metadata={
            "task_problem": structured_obs.task.problem_statement,
            "policy": policy,
        },
    )

    done = structured_obs.done
    step_num = 0
    while not done:
        if max_steps is not None and step_num >= max_steps:
            break

        history = [rec.action_type for rec in structured_obs.pipeline_history]
        action_type = pick_action(policy, step_num, history)
        experiment_action = build_experiment_action(
            action_type=action_type,
            discovered_markers=structured_obs.discovered_markers,
            conditions=structured_obs.task.conditions,
        )

        structured_obs = env.step(experiment_action)
        reward = structured_obs.reward
        done = structured_obs.done
        step_num += 1

        traj.add_step(
            action=experiment_action,
            observation=structured_obs,
            reward=reward,
            done=done,
            reward_breakdown=structured_obs.step_reward_breakdown,
        )

        print(
            f"  step={structured_obs.step_index:02d} "
            f"action={action_type.value:>28} "
            f"reward={reward:+.3f}"
        )

    return traj


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = BioExperimentEnvironment()
    trajectories: List[Trajectory] = []

    print(
        f"Starting rollout collection: episodes={args.episodes}, policy={args.policy}"
    )
    for ep in range(args.episodes):
        print(f"Episode {ep + 1}/{args.episodes}")
        traj = run_episode(
            env=env,
            episode_id=str(uuid.uuid4()),
            policy=args.policy,
            max_steps=args.max_steps,
        )
        traj.save(out_dir / f"{traj.episode_id}.json")
        trajectories.append(traj)

    dataset = TrajectoryDataset(trajectories)
    stats = EvaluationSuite.online_metrics(trajectories)

    print("\nRun complete.")
    print(f"Saved trajectories to: {out_dir}")
    print("Online metrics:")
    for metric in stats:
        print(f"  - {metric.name}: {metric.value:.4f}")

    print(f"Summary: {dataset.summary()}")


if __name__ == "__main__":
    main()
