"""FastAPI application for StudentGrad Environment.

ARCHITECTURE NOTE — Why we use a singleton factory:
openenv-core calls _env_factory() on EVERY HTTP request (both /reset and /step).
By default this creates a fresh instance each time, so state from reset() is
lost before step() is ever called (AssertionError: "Call reset() before step()").

The fix: wrap StudentEnvironment in a singleton closure so every call to the
factory returns the same instance.  The environment's own reset()/step() logic
handles state transitions; the framework just needs a stable object to call.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from fastapi.responses import HTMLResponse

try:
    from openenv.core.env_server.http_server import create_app
except ImportError as exc:
    raise ImportError(
        "openenv-core is missing. Install with: pip install openenv-core"
    ) from exc

from models import StudentAction, StudentObservation
from server.student_environment import StudentEnvironment


# ── Singleton factory ─────────────────────────────────────────────────────────
# openenv-core calls this callable on every /reset and /step request.
# Returning the same instance keeps state alive across calls.
_singleton: StudentEnvironment | None = None

def _env_factory() -> StudentEnvironment:
    global _singleton
    if _singleton is None:
        _singleton = StudentEnvironment()
    return _singleton


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = create_app(
    _env_factory,                       # callable — returns the singleton
    StudentAction,
    StudentObservation,
    env_name="student_optimizer",
    max_concurrent_envs=1,              # singleton → only 1 concurrent session
)


@app.get("/", response_class=HTMLResponse)
async def demo_ui() -> HTMLResponse:
    return HTMLResponse("<h1>StudentGrad API</h1><p>POST /reset then POST /step</p>")


# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    """Zero-argument entry point for the OpenEnv multi-mode validator."""
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
