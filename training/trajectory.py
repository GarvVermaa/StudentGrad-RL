"""Trajectory serialisation and dataset utilities for StudentGrad.

A ``Trajectory`` stores the full history of one episode (task, actions,
observations, rewards) in a format that supports:
  - offline RL training
  - imitation learning from expert demonstrations
  - evaluation / replay
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class TrajectoryStep:
    step_index: int
    action: Dict[str, Any]
    observation: Dict[str, Any]
    reward: float
    done: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Trajectory:
    """Full record of one StudentGrad episode."""

    episode_id: str
    task: Dict[str, Any]
    steps: List[TrajectoryStep] = field(default_factory=list)
    total_reward: float = 0.0
    success: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_step(
        self,
        action: Dict[str, Any],
        observation: Dict[str, Any],
        reward: float,
        done: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.steps.append(
            TrajectoryStep(
                step_index=len(self.steps),
                action=action,
                observation=observation,
                reward=reward,
                done=done,
                metadata=metadata or {},
            )
        )
        self.total_reward += reward
        if done:
            self.success = reward > 0

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
                    "metadata": s.metadata,
                }
                for s in self.steps
            ],
            "total_reward": self.total_reward,
            "success": self.success,
            "metadata": self.metadata,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> "Trajectory":
        data = json.loads(path.read_text())
        traj = cls(
            episode_id=data["episode_id"],
            task=data["task"],
            total_reward=data.get("total_reward", 0.0),
            success=data.get("success", False),
            metadata=data.get("metadata", {}),
        )
        for s in data.get("steps", []):
            traj.steps.append(
                TrajectoryStep(
                    step_index=s["step_index"],
                    action=s["action"],
                    observation=s["observation"],
                    reward=s["reward"],
                    done=s.get("done", False),
                    metadata=s.get("metadata", {}),
                )
            )
        return traj


class TrajectoryDataset:
    """Collection of trajectories with basic filtering utilities."""

    def __init__(self, trajectories: Optional[List[Trajectory]] = None) -> None:
        self.trajectories: List[Trajectory] = trajectories or []

    def add(self, traj: Trajectory) -> None:
        self.trajectories.append(traj)

    def successful(self) -> List[Trajectory]:
        return [t for t in self.trajectories if t.success]

    def mean_reward(self) -> float:
        if not self.trajectories:
            return 0.0
        return sum(t.total_reward for t in self.trajectories) / len(self.trajectories)

    def save_all(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        for t in self.trajectories:
            t.save(directory / f"{t.episode_id}.json")
