"""FastAPI application for StudentGrad Environment."""

import os
from fastapi.responses import HTMLResponse

try:
    from openenv.core.env_server.http_server import create_app
except ImportError:
    raise ImportError("openenv-core is missing. Install it with 'uv pip install openenv-core'")

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
    return "<h1>StudentGrad API</h1>"


def main():
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)


# ⚠️ CRITICAL: MUST use single quotes
if __name__ == '__main__':
    main()