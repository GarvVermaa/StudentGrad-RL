"""StudentGrad Environment — drop-in replacement for hackathon_environment.py."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from models import (
    ActionType,
    DailyOutput,
    ResourceUsage,
    SessionRecord,
    StudentAction,
    StudentObservation,
    StudentTaskSpec,
)

from server.rules.engine import RuleEngine
from server.rewards.reward import RewardBreakdown, RewardComputer
from server.simulator.latent_state import FullLatentState
from server.simulator.noise import NoiseModel
from server.simulator.transition import ACTION_COSTS, TransitionEngine, compute_action_cost
from server.tasks.generator import TaskGenerator

MAX_STEPS = 365


class StudentEnvironment(Environment):
    """POMDP environment for 365-day student life optimisation."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(
        self,
        scenario_name: Optional[str] = None,
        *,
        domain_randomise: bool = True,
    ) -> None:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._latent: Optional[FullLatentState] = None
        self._task: Optional[StudentTaskSpec] = None
        self._scenario_name = scenario_name
        self._noise = NoiseModel()
        self._engine = TransitionEngine(self._noise)
        self._rules = RuleEngine()
        self._rewards = RewardComputer()
        self._task_gen = TaskGenerator(domain_randomise=domain_randomise)

        self._history: List[SessionRecord] = []
        self._outputs: List[DailyOutput] = []
        self._cumulative_reward: float = 0.0

    # ── Environment interface ────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None) -> StudentObservation:
        seed = seed if seed is not None else hash(uuid4()) % (2**31)
        self._noise.reseed(seed)
        self._state = State(episode_id=str(uuid4()), step_count=0)

        self._task, self._latent = self._task_gen.generate(
            seed=seed,
            scenario_name=self._scenario_name,
        )
        self._latent.rng_seed = seed

        self._history.clear()
        self._outputs.clear()
        self._cumulative_reward = 0.0

        return self._build_observation(reward=0.0, done=False)

    def step(self, action: StudentAction) -> StudentObservation:
        # Auto-reset if called without prior reset (openenv-core creates a fresh
        # instance per request when not using a singleton factory).
        if self._latent is None or self._task is None:
            self.reset()

        self._state.step_count += 1
        prev_state = self._latent.model_copy(deep=True)

        violations = self._rules.check(action, self._latent)
        hard_v = self._rules.hard_violations(violations)
        soft_v = self._rules.soft_violations(violations)

        result = self._engine.step(
            self._latent,
            action,
            hard_violations=hard_v,
            soft_violations=soft_v,
        )
        self._latent = result.next_state

        step_rb = self._rewards.step_reward(
            action, prev_state, self._latent, result.output, hard_v, soft_v,
        )

        self._history.append(SessionRecord(
            day=self._latent.resources.day_current,
            action_type=action.action_type,
            skill_target=action.skill_target,
            project_target=action.project_target,
            energy_spent=ACTION_COSTS.get(action.action_type, (1, 0.0))[1],
            reward=step_rb.total,
            summary=result.output.summary,
        ))
        self._outputs.append(result.output)

        done = result.done or self._state.step_count >= MAX_STEPS

        terminal_rb = RewardBreakdown()
        if done:
            terminal_rb = self._rewards.terminal_reward(
                self._latent,
                [],                              # no conclusion objects in student env
                self._task.success_criteria,
            )

        total_reward = step_rb.total + terminal_rb.total
        self._cumulative_reward += total_reward

        breakdown = step_rb.to_dict()
        breakdown.update({f"term_{k}": v for k, v in terminal_rb.to_dict().items()})

        return self._build_observation(
            reward=total_reward,
            done=done,
            latest_output=result.output,
            rule_violations=hard_v + soft_v,
            reward_breakdown=breakdown,
            metadata_extra={"reward_breakdown": breakdown},
        )

    @property
    def state(self) -> State:
        return self._state

    def set_scenario(self, scenario_name: Optional[str]) -> None:
        self._scenario_name = scenario_name

    # ── internal helpers ─────────────────────────────────────────────────

    def _build_observation(
        self,
        *,
        reward: float,
        done: bool,
        latest_output: Optional[DailyOutput] = None,
        rule_violations: Optional[List[str]] = None,
        reward_breakdown: Optional[Dict[str, float]] = None,
        metadata_extra: Optional[Dict[str, Any]] = None,
    ) -> StudentObservation:
        assert self._task is not None
        assert self._latent is not None

        res = self._latent.resources
        meta: Dict[str, Any] = {
            "episode_id": self._state.episode_id,
            "step": self._state.step_count,
            "cumulative_reward": self._cumulative_reward,
        }
        if metadata_extra:
            meta.update(metadata_extra)

        # Add small noise to exposed state (POMDP — agent doesn't see exact truth)
        noisy_attendance = {
            k: round(min(1.0, v + self._noise.quality_degradation(0.02) - 0.01), 3)
            for k, v in self._latent.true_attendance.items()
        }
        noisy_knowledge = {
            k: round(min(100.0, v + self._noise.quality_degradation(1.0) - 0.5), 1)
            for k, v in self._latent.true_knowledge.items()
        }
        noisy_skills = {
            k: round(max(0.0, v + self._noise.quality_degradation(0.1) - 0.05), 2)
            for k, v in self._latent.true_skills.items()
        }

        return StudentObservation(
            task=self._task,
            day=res.day_current,
            step_index=self._state.step_count,
            attendance=noisy_attendance,
            knowledge=noisy_knowledge,
            skills=noisy_skills,
            completed_projects=list(self._latent.completed_projects),
            active_project_progress=round(self._latent.true_project_progress, 3),
            energy=res.energy_current,
            fatigue=round(res.fatigue_current, 1),
            session_history=list(self._history[-10:]),   # last 10 days only
            resource_usage=ResourceUsage(
                days_used=res.day_current,
                days_remaining=res.days_remaining,
                energy_current=res.energy_current,
                fatigue_current=res.fatigue_current,
            ),
            latest_output=latest_output,
            rule_violations=rule_violations or [],
            step_reward_breakdown=reward_breakdown or {},
            sick_today=bool(self._latent.last_sick_triggered),
            surprise_quiz_today=bool(self._latent.last_quiz_triggered),
            done=done,
            reward=reward,
            metadata=meta,
        )