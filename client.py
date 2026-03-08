"""Bio-Experiment Environment Client.

Provides the ``BioExperimentEnv`` class that communicates with the
environment server over WebSocket / HTTP using the OpenEnv protocol.
"""

from typing import Any, Dict, List

from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State
from openenv.core import EnvClient

try:  # pragma: no cover - package import path
    from .models import ExperimentAction, ExperimentObservation
except ImportError:  # pragma: no cover - direct module import path
    from models import ExperimentAction, ExperimentObservation


class BioExperimentEnv(
    EnvClient[ExperimentAction, ExperimentObservation, State]
):
    """Client for the Bio-Experiment Planning Environment.

    Example:
        >>> with BioExperimentEnv(base_url="http://localhost:8000") as env:
        ...     result = env.reset()
        ...     print(result.observation.task.problem_statement)
        ...     result = env.step(ExperimentAction(
        ...         action_type="collect_sample",
        ...         parameters={"n_samples": 6},
        ...     ))
        ...     print(result.observation.latest_output.summary)
    """

    def _step_payload(self, action: ExperimentAction) -> Dict:
        return action.model_dump()

    def _parse_result(
        self, payload: Dict
    ) -> StepResult[ExperimentObservation]:
        obs_data = payload.get("observation", {})
        observation = ExperimentObservation(**obs_data)
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
