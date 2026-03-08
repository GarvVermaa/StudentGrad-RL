FROM python:3.12-slim

WORKDIR /app

# System deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl && \
    rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency resolution
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/uv && \
    mv /root/.local/bin/uvx /usr/local/bin/uvx

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies (including train extras for GRPO training)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-editable --extra train

# Copy application code
COPY . .

# Install project
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-editable --extra train

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app:$PYTHONPATH"

# Training output directory
ENV OUTPUT_DIR="/app/training/grpo-output"

# Default: run GRPO training. Override CMD for other modes.
CMD ["python", "training_script.py", \
     "--model-id", "Qwen/Qwen3.5-0.8B", \
     "--output-dir", "/app/training/grpo-output", \
     "--dataset-episodes", "8", \
     "--rollout-steps", "6", \
     "--num-generations", "4", \
     "--trust-remote-code"]
