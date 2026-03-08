"""FastAPI application for the Bio-Experiment Planning Environment.

Endpoints:
    - POST /reset:  Reset the environment
    - POST /step:   Execute an action
    - GET  /state:  Get current environment state
    - GET  /schema: Get action/observation schemas
    - WS   /ws:     WebSocket endpoint for persistent sessions
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. "
        "Install dependencies with 'uv sync'"
    ) from e

from models import ExperimentAction, ExperimentObservation
from .hackathon_environment import BioExperimentEnvironment

app = create_app(
    BioExperimentEnvironment,
    ExperimentAction,
    ExperimentObservation,
    env_name="bio_experiment",
    max_concurrent_envs=1,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.host == "0.0.0.0" and args.port == 8000:
        main()
    else:
        main(host=args.host, port=args.port)
