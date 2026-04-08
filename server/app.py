"""FastAPI application for StudentGrad Environment."""
import os
import sys
from pathlib import Path

# Add the project root to sys.path so 'server' is recognized as a package
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from fastapi.responses import HTMLResponse
try:
    from openenv.core.env_server.http_server import create_app
except ImportError:
    raise ImportError("openenv-core is missing. Install it with 'uv pip install openenv-core'")

# Use ABSOLUTE imports here
from models import StudentAction, StudentObservation
from server.student_environment import StudentEnvironment

app = create_app(
    StudentEnvironment,
    StudentAction,
    StudentObservation,
    env_name="student_optimizer",
    max_concurrent_envs=int(os.environ.get("MAX_ENVS", "4")),
)

@app.get("/", response_class=HTMLResponse)
async def demo_ui():
    return HTMLResponse(content="<h1>StudentGrad API</h1>", status_code=200)

def main():
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    # We use the full path so the validator knows exactly where to look
    uvicorn.run("server.app:app", host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()