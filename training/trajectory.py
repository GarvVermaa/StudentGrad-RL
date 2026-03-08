"""Trajectory serialisation and dataset utilities.

A ``Trajectory`` stores the full history of one episode (task, actions,
observations, rewards, latent-state snapshots) in a format that supports:
  - offline RL training
  - imitation learning from expert demonstrations
  - evaluation / replay
  - simulator calibration
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from models import (
    ExperimentAction,
    ExperimentObservation,
    TaskSpec,
)


@dataclass
class TrajectoryStep:
    step_index: int
    action: Dict[str, Any]
    observation: Dict[str, Any]
    reward: float
    done: bool
    reward_breakdown: Dict[str, float] = field(default_factory=dict)
    latent_snapshot: Optional[Dict[str, Any]] = None


@dataclass
class Trajectory:
    """Complete record of one environment episode."""

    episode_id: str
    task: Dict[str, Any]
    steps: List[TrajectoryStep] = field(default_factory=list)
    total_reward: float = 0.0
    success: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ── construction helpers ────────────────────────────────────────────

    def add_step(
        self,
        action: ExperimentAction,
        observation: ExperimentObservation,
        reward: float,
        done: bool,
        reward_breakdown: Optional[Dict[str, float]] = None,
        latent_snapshot: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.steps.append(TrajectoryStep(
            step_index=len(self.steps),
            action=action.model_dump(),
            observation=observation.model_dump(),
            reward=reward,
            done=done,
            reward_breakdown=reward_breakdown or {},
            latent_snapshot=latent_snapshot,
        ))
        self.total_reward += reward
        if done:
            self.success = reward > 0

    # ── serialisation ───────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "task": self.task,
            "steps": [
                {
                    "step_index": s.step_index,
                    "action": s.action,
                    "observation": s.observation,
                    "reward": s.reward,
                    "done": s.done,
                    "reward_breakdown": s.reward_breakdown,
                    "latent_snapshot": s.latent_snapshot,
                }
                for s in self.steps
            ],
            "total_reward": self.total_reward,
            "success": self.success,
            "metadata": self.metadata,
        }

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load(cls, path: str | Path) -> "Trajectory":
        with open(path) as f:
            d = json.load(f)
        traj = cls(
            episode_id=d["episode_id"],
            task=d["task"],
            total_reward=d.get("total_reward", 0.0),
            success=d.get("success", False),
            metadata=d.get("metadata", {}),
        )
        for s in d.get("steps", []):
            traj.steps.append(TrajectoryStep(**s))
        return traj


class TrajectoryDataset:
    """In-memory collection of trajectories with convenience accessors."""

    def __init__(self, trajectories: Optional[List[Trajectory]] = None):
        self.trajectories: List[Trajectory] = trajectories or []

    def add(self, traj: Trajectory) -> None:
        self.trajectories.append(traj)

    def __len__(self) -> int:
        return len(self.trajectories)

    def __getitem__(self, idx: int) -> Trajectory:
        return self.trajectories[idx]

    def filter_successful(self) -> "TrajectoryDataset":
        return TrajectoryDataset([t for t in self.trajectories if t.success])

    def save_dir(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        for t in self.trajectories:
            t.save(d / f"{t.episode_id}.json")

    @classmethod
    def load_dir(cls, directory: str | Path) -> "TrajectoryDataset":
        d = Path(directory)
        trajs = [Trajectory.load(p) for p in sorted(d.glob("*.json"))]
        return cls(trajs)

    def summary(self) -> Dict[str, Any]:
        if not self.trajectories:
            return {"n": 0}
        rewards = [t.total_reward for t in self.trajectories]
        lengths = [len(t.steps) for t in self.trajectories]
        success_rate = sum(1 for t in self.trajectories if t.success) / len(self.trajectories)
        return {
            "n": len(self.trajectories),
            "success_rate": success_rate,
            "mean_reward": sum(rewards) / len(rewards),
            "mean_length": sum(lengths) / len(lengths),
            "max_reward": max(rewards),
            "min_reward": min(rewards),
        }
