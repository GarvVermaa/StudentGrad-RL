"""FastAPI application for StudentGrad Environment."""

import os
from pathlib import Path

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError("openenv is required. Run 'uv sync'") from e

from fastapi.responses import HTMLResponse
from models import StudentAction, StudentObservation
from .student_environment import StudentEnvironment

app = create_app(
    StudentEnvironment,
    StudentAction,
    StudentObservation,
    env_name="student_optimizer",
    max_concurrent_envs=int(os.environ.get("MAX_ENVS", "4")),
)

DEMO_HTML = Path(__file__).resolve().parent.parent / "demo.html"


@app.get("/", response_class=HTMLResponse)
async def demo_ui():
    if DEMO_HTML.exists():
        return HTMLResponse(content=DEMO_HTML.read_text(), status_code=200)
    return HTMLResponse(
        content="<h1>StudentGrad API</h1><p>Visit /docs</p>",
        status_code=200,
    )


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