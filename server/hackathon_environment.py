"""Bio-Experiment Planning Environment.

Implements the OpenEnv ``Environment`` interface as a POMDP where the
agent proposes one structured experiment / analysis step at a time and
receives simulated intermediate outputs from a latent biological world.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from models import (
    ActionType,
    ConclusionClaim,
    ExperimentAction,
    ExperimentObservation,
    IntermediateOutput,
    PipelineStepRecord,
    ResourceUsage,
    TaskSpec,
)

from server.rules.engine import RuleEngine
from server.rewards.reward import RewardBreakdown, RewardComputer
from server.simulator.latent_state import FullLatentState
from server.simulator.noise import NoiseModel
from server.simulator.transition import ACTION_COSTS, TransitionEngine
from server.tasks.generator import TaskGenerator


MAX_STEPS = 30


class BioExperimentEnvironment(Environment):
    """POMDP environment for iterative biological experiment planning.

    The agent observes ``ExperimentObservation`` (partial view) while the
    environment maintains a ``FullLatentState`` (hidden ground truth).
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(
        self,
        scenario_name: Optional[str] = None,
        *,
        domain_randomise: bool = True,
    ) -> None:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._latent: Optional[FullLatentState] = None
        self._task: Optional[TaskSpec] = None
        self._scenario_name = scenario_name
        self._noise = NoiseModel()
        self._engine = TransitionEngine(self._noise)
        self._rules = RuleEngine()
        self._rewards = RewardComputer()
        self._task_gen = TaskGenerator(domain_randomise=domain_randomise)

        self._history: List[PipelineStepRecord] = []
        self._outputs: List[IntermediateOutput] = []
        self._conclusions: List[ConclusionClaim] = []
        self._subagent_outputs: List[Dict[str, Any]] = []
        self._discovered_markers: List[str] = []
        self._candidate_mechanisms: List[str] = []
        self._cumulative_reward: float = 0.0

    # ── Environment interface ───────────────────────────────────────────

    def reset(self) -> ExperimentObservation:
        seed = hash(uuid4()) % (2**31)
        self._noise.reseed(seed)
        self._state = State(episode_id=str(uuid4()), step_count=0)

        self._task, self._latent = self._task_gen.generate(
            seed=seed,
            scenario_name=self._scenario_name,
        )
        self._latent.rng_seed = seed

        self._history.clear()
        self._outputs.clear()
        self._conclusions.clear()
        self._subagent_outputs.clear()
        self._discovered_markers.clear()
        self._candidate_mechanisms.clear()
        self._cumulative_reward = 0.0

        return self._build_observation(reward=0.0, done=False)

    def step(  # type: ignore[override]
        self, action: ExperimentAction
    ) -> ExperimentObservation:
        assert self._latent is not None, "Call reset() before step()"
        assert self._task is not None

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

        cost_budget, cost_time = ACTION_COSTS.get(action.action_type, (0, 0))
        self._history.append(PipelineStepRecord(
            step_index=self._state.step_count,
            action_type=action.action_type,
            method=action.method,
            parameters=action.parameters,
            output_summary=result.output.summary,
            output_type=result.output.output_type,
            success=result.output.success,
            quality_score=result.output.quality_score,
            resource_cost=cost_budget,
            time_cost_days=cost_time,
        ))
        self._outputs.append(result.output)
        self._update_discoveries(action, result.output)

        if action.action_type == ActionType.SYNTHESIZE_CONCLUSION:
            raw_claims = action.parameters.get("claims", [])
            for c in raw_claims:
                if isinstance(c, dict):
                    self._conclusions.append(ConclusionClaim(**c))

        done = result.done or self._state.step_count >= MAX_STEPS

        terminal_rb = RewardBreakdown()
        if done:
            terminal_rb = self._rewards.terminal_reward(
                self._latent, self._conclusions, self._task.success_criteria,
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
        )

    @property
    def state(self) -> State:
        return self._state

    def set_scenario(self, scenario_name: Optional[str]) -> None:
        """Set the scenario used on the next reset."""

        self._scenario_name = scenario_name

    # ── internal helpers ────────────────────────────────────────────────

    def _build_observation(
        self,
        *,
        reward: float,
        done: bool,
        latest_output: Optional[IntermediateOutput] = None,
        rule_violations: Optional[List[str]] = None,
        reward_breakdown: Optional[Dict[str, float]] = None,
    ) -> ExperimentObservation:
        assert self._task is not None
        assert self._latent is not None
        res = self._latent.resources
        return ExperimentObservation(
            task=self._task,
            step_index=self._state.step_count,
            pipeline_history=list(self._history),
            available_assays=list(self._task.available_assays),
            available_tools=list(self._task.available_tools),
            resource_usage=ResourceUsage(
                budget_used=res.budget_used,
                budget_remaining=res.budget_remaining,
                time_used_days=res.time_used_days,
                time_remaining_days=res.time_remaining_days,
                samples_consumed=res.samples_consumed,
                compute_hours_used=res.compute_hours_used,
            ),
            latest_output=latest_output,
            all_outputs=list(self._outputs),
            discovered_markers=list(self._discovered_markers),
            candidate_mechanisms=list(self._candidate_mechanisms),
            uncertainty_summary=self._compute_uncertainty_summary(),
            subagent_outputs=list(self._subagent_outputs),
            conclusions=list(self._conclusions),
            rule_violations=rule_violations or [],
            step_reward_breakdown=reward_breakdown or {},
            done=done,
            reward=reward,
            metadata={
                "episode_id": self._state.episode_id,
                "step": self._state.step_count,
                "cumulative_reward": self._cumulative_reward,
            },
        )

    def _compute_uncertainty_summary(self) -> Dict[str, float]:
        if not self._outputs:
            return {}
        recent = self._outputs[-5:]
        avg_unc = sum(o.uncertainty for o in recent) / len(recent)
        avg_qual = sum(o.quality_score for o in recent) / len(recent)
        return {"avg_uncertainty": avg_unc, "avg_quality": avg_qual}

    def _update_discoveries(
        self, action: ExperimentAction, output: IntermediateOutput
    ) -> None:
        if action.action_type == ActionType.MARKER_SELECTION:
            markers = output.data.get("markers", [])
            self._discovered_markers.extend(markers)
        if action.action_type == ActionType.REGULATORY_NETWORK_INFERENCE:
            regs = output.data.get("top_regulators", [])
            self._candidate_mechanisms.extend(regs)
        if action.action_type == ActionType.PATHWAY_ENRICHMENT:
            pathways = output.data.get("top_pathways", [])
            self._candidate_mechanisms.extend(
                [p["pathway"] for p in pathways if isinstance(p, dict)]
            )
