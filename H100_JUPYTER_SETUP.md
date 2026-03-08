# H100 Jupyter Notebook Setup

This guide walks you through setting up the OpenEnv Bio Experiment environment on an **NVIDIA H100** Jupyter notebook instance (e.g., Jupiter Labs, Lambda Labs, RunPod, or similar).

## Prerequisites

- **Python** ≥ 3.10 (3.10, 3.11, or 3.12 recommended)
- **uv** – fast Python package manager ([install instructions](#installing-uv))
- **NVIDIA driver** ≥ 535.104.05 (usually pre-installed on H100 instances)
- **CUDA** – H100 uses CUDA 12.x; PyTorch wheels bundle the runtime, so a separate CUDA Toolkit is not required

## Installing uv

If `uv` is not already installed:

```bash
# Unix/Linux (including Jupiter notebook terminals)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with pip
pip install uv
```

Verify:

```bash
uv --version
```

## Quick Setup (Recommended)

### 1. Clone and enter the project

```bash
git clone <repository-url> OpenENV-Hackathon
cd OpenENV-Hackathon
```

### 2. Use uv's auto PyTorch backend

uv can detect your GPU and pick the right PyTorch build. For H100 (CUDA 12.x):

```bash
# Install everything: core + training (TRL, transformers, torch, unsloth) + Jupyter
UV_TORCH_BACKEND=cu128 uv sync --extra train

# Add Jupyter kernel support
uv add ipykernel jupyter --extra train
```

If `UV_TORCH_BACKEND=cu128` fails (e.g., cu128 wheels not available yet), try:

```bash
UV_TORCH_BACKEND=cu126 uv sync --extra train
```

### 3. Register the environment as a Jupyter kernel

```bash
uv run python -m ipykernel install --user --name openenv-bio --display-name "OpenEnv Bio (H100)"
```

### 4. Verify CUDA

In a new Jupyter notebook, select the **"OpenEnv Bio (H100)"** kernel and run:

```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
```

Expected output (or similar):

```
PyTorch: 2.x.x+cu128
CUDA available: True
GPU: NVIDIA H100 ...
```

### 5. Sanity check the environment

```bash
uv run pytest tests/test_environment.py tests/test_literature_benchmark.py -q
```

## Manual PyTorch CUDA Configuration

If you need explicit control over the PyTorch index (e.g., for reproducibility), add the following to `pyproject.toml`:

### Add to `pyproject.toml`

```toml
# After [tool.uv], add:

[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[tool.uv.sources]
torch = [{ index = "pytorch-cu128" }]
torchvision = [{ index = "pytorch-cu128" }]
```

Then run:

```bash
uv sync --extra train
uv add ipykernel jupyter --extra train
```

For CUDA 12.6 instead of 12.8, use `cu126` in the index URL and source names.

## Dependency Groups

| uv sync flag      | Contents                                                                 |
|-------------------|--------------------------------------------------------------------------|
| *(default)*       | Core: `openenv-core`, `numpy`, `scipy`, `pydantic`                       |
| `--extra dev`     | Testing: `pytest`, `pytest-cov`                                          |
| `--extra train`   | Training: `torch`, `transformers`, `trl`, `accelerate`, `peft`, `unsloth`, etc. |
| `--extra bio`     | Bioinformatics: `scanpy`, `biopython`, `gseapy`                          |
| `--extra train --extra dev` | Combined for development + training                    |

## Preferred H100 Workflow

On H100, use the quantized Unsloth entrypoints:

```bash
uv run python training_unsloth.py --dry-run
uv run python training_unsloth.py --model-id Qwen/Qwen3.5-4B --output-dir training/grpo-unsloth-output
uv run python run_agent_unsloth.py
```

The checked-in `inference.ipynb` notebook uses `training_unsloth.py` helpers with 4-bit loading. vLLM fast inference is disabled to avoid dependency conflicts.

## Running Training in a Jupyter Notebook

Example cell:

```python
# In a notebook with the OpenEnv Bio (H100) kernel
!uv run python training_unsloth.py --model-id Qwen/Qwen3.5-4B --dry-run
```

Or run interactively from Python:

```python
import subprocess
subprocess.run([
    "uv", "run", "python", "training_unsloth.py",
    "--model-id", "Qwen/Qwen3.5-4B",
    "--output-dir", "training/grpo-unsloth-output",
], check=True)
```

## Requirements Summary

| Component      | Version / Notes                                      |
|----------------|------------------------------------------------------|
| Python         | ≥ 3.10                                               |
| uv             | ≥ 0.5.3 (for PyTorch index support)                  |
| torch          | ≥ 2.10.0 (cu128 or cu126 for H100)                   |
| transformers   | ≥ 5.3.0                                              |
| trl            | ≥ 0.29.0                                             |
| accelerate     | ≥ 1.13.0                                             |
| Jupyter        | Optional, for notebook workflows                     |

## Troubleshooting

### `torch.cuda.is_available()` is False

- Confirm the Jupyter kernel is the one where you ran `uv sync` (the one with `ipykernel`).
- Ensure no CPU-only PyTorch is overriding the CUDA build (e.g., from a different conda/pip env).
- Run `uv run python -c "import torch; print(torch.__file__)"` to verify PyTorch comes from your project venv.

### Flash Attention / causal-conv fallback warnings

These are common and usually harmless; execution continues with a slower path. For best H100 performance, ensure `transformers` and `torch` are recent versions that support Flash Attention 2.

### HuggingFace symlink warnings

Set:

```bash
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
```

### Out-of-memory during training

- Reduce `--num-generations` or `--rollout-steps`.
- Use a smaller model (e.g., `Qwen/Qwen3.5-0.8B`) for experiments.
- Keep `--disable-4bit` off unless you explicitly need wider weights.

### `ModuleNotFoundError: No module named 'vllm.lora.models'`

Unsloth's `unsloth_zoo` imports vLLM at load time and expects `vllm.lora.models`, which some vLLM versions don't have. Fix by installing a compatible vLLM:

```bash
pip install "vllm==0.8.2"
# or
pip install "vllm==0.7.3"
```

**Note:** vLLM 0.8.2 pins `torch==2.6.0`, which conflicts with this project's `torch>=2.10.0`. If you hit that conflict:

1. Use a **separate environment** with torch 2.6–2.8 + vllm 0.8.2 + unsloth.
2. Or use the non-Unsloth path (`training_script.py` / `train.ipynb`) which doesn't depend on vLLM.

### Unsloth import order warning

If you see "Unsloth should be imported before trl, transformers, peft", ensure `training_unsloth` is imported before `training_script` in your notebook:

```python
from training_unsloth import make_training_args, run_training  # first
import training_script as base
```

## See Also

- Main [README.md](README.md) for project overview, APIs, and usage
- [uv PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/) for advanced PyTorch configuration
