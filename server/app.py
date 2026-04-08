"""FastAPI application for StudentGrad Environment."""

import os
import sys
from pathlib import Path

# Force the current directory into sys.path to ensure relative imports work in multi-mode
sys.path.append(os.getcwd())

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError("openenv is required. Run 'uv sync'") from e

from fastapi.responses import HTMLResponse
from models import StudentAction, StudentObservation
from server.student_environment import StudentEnvironment

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

def main():
    """Main entry point for the environment server."""
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    # Using the full module path "server.app:app" is critical for multi-mode
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, log_level="info")

if __name__ == "__main__":
    main()