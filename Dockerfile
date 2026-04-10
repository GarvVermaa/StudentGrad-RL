# ── StudentGrad-RL  ──────────────────────────────────────────────────────────
# Root Dockerfile: builds and runs the OpenEnv FastAPI server.
# Used by: openenv validate, HF Space deployment, the pre-submission validator.
#
# The server/Dockerfile is the multi-stage version used by `openenv build`.
# This root Dockerfile is the simple, self-contained version for local builds
# and the HF Space /reset ping check.
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.12-slim

WORKDIR /app

# System deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl && \
    rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/uv && \
    mv /root/.local/bin/uvx /usr/local/bin/uvx

# Layer-cache deps before copying source
COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-editable

# Copy source
COPY . .

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-editable

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app:$PYTHONPATH"
ENV PORT=8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Run the OpenEnv FastAPI server
CMD ["sh", "-c", "cd /app && uvicorn server.app:app --host 0.0.0.0 --port ${PORT}"]
