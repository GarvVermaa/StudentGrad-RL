"""FastAPI application for the Bio-Experiment Planning Environment.

Endpoints:
    - POST /reset:  Reset the environment
    - POST /step:   Execute an action
    - GET  /state:  Get current environment state
    - GET  /schema: Get action/observation schemas
    - WS   /ws:     WebSocket endpoint for persistent sessions
    - GET  /        Demo UI
"""

import os
from pathlib import Path

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. "
        "Install dependencies with 'uv sync'"
    ) from e

from fastapi.responses import HTMLResponse
from models import ExperimentAction, ExperimentObservation
from .hackathon_environment import BioExperimentEnvironment

app = create_app(
    BioExperimentEnvironment,
    ExperimentAction,
    ExperimentObservation,
    env_name="bio_experiment",
    max_concurrent_envs=int(os.environ.get("MAX_ENVS", "4")),
)

# Serve demo UI at root
DEMO_HTML = Path(__file__).resolve().parent.parent / "demo.html"


@app.get("/", response_class=HTMLResponse)
async def demo_ui():
    if DEMO_HTML.exists():
        return HTMLResponse(content=DEMO_HTML.read_text(), status_code=200)
    return HTMLResponse(content="<h1>BioEnv API</h1><p>Visit /docs for API documentation.</p>", status_code=200)


def main(host: str = "0.0.0.0", port: int = None):
    import uvicorn
    if port is None:
        port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()
    main(host=args.host, port=args.port)
