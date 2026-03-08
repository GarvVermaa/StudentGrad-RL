# H100 Jupyter Notebook Setup

This guide walks you through setting up the OpenEnv Bio Experiment environment on an **NVIDIA H100** Jupyter notebook instance (e.g., Jupiter Labs, Lambda Labs, RunPod, or similar).

## Prerequisites

- **Python** 3.10, 3.11, or **3.12** (3.12 recommended for H100; 3.13 is not supported—numba, vllm, and others require &lt;3.13)
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

The project uses Python 3.12 (see `.python-version`). uv will create a 3.12 venv. For H100 (CUDA 12.x):

```bash
# Install everything: core + training (TRL, transformers, torch) + Jupyter
UV_TORCH_BACKEND=cu128 uv sync --extra train

# Add Unsloth for training_unsloth.py (skips trl downgrade; Unsloth works with TRL 0.29)
uv pip install unsloth unsloth_zoo --no-deps

# (ipykernel is included in --extra train)
```

If `UV_TORCH_BACKEND=cu128` fails (e.g., cu128 wheels not available yet), try:

```bash
UV_TORCH_BACKEND=cu126 uv sync --extra train
```

### 3. Register the environment as a Jupyter kernel

```bash
uv run python -m ipykernel install --user --name openenv-bio-312 --display-name "OpenEnv Bio (Python 3.12)"
```

Or run the helper script (from project root):

```bash
bash scripts/register_kernel_312.sh
```

Then select **"OpenEnv Bio (Python 3.12)"** in the notebook kernel picker.

### 4. Verify CUDA

In a new Jupyter notebook, select the **"OpenEnv Bio (Python 3.12)"** kernel and run:

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
uv run python training_unsloth.py --model-id Qwen/Qwen3-4B-Base --output-dir training/grpo-unsloth-qwen3-4b --dry-run
uv run python training_unsloth.py --model-id Qwen/Qwen3-4B-Base --output-dir training/grpo-unsloth-qwen3-4b
uv run python run_agent_unsloth.py
```

The checked-in `inference.ipynb` notebook uses `training_unsloth.py` helpers with 4-bit loading. vLLM fast inference is disabled to avoid dependency conflicts.

## Running Training in a Jupyter Notebook

Example cell:

```python
# In a notebook with the OpenEnv Bio (Python 3.12) kernel
!uv run python training_unsloth.py --model-id Qwen/Qwen3-4B-Base --output-dir training/grpo-unsloth-qwen3-4b --dry-run
```

Or run interactively from Python:

```python
import subprocess
subprocess.run([
    "uv", "run", "python", "training_unsloth.py",
    "--model-id", "Qwen/Qwen3-4B-Base",
    "--output-dir", "training/grpo-unsloth-qwen3-4b",
], check=True)
```

## Requirements Summary

| Component      | Version / Notes                                      |
|----------------|------------------------------------------------------|
| Python         | 3.10–3.12 (3.12 recommended; 3.13 not supported)     |
| uv             | ≥ 0.5.3 (for PyTorch index support)                  |
| torch          | ≥ 2.10.0 (cu128 or cu126 for H100)                   |
| transformers   | ≥4.57 (with unsloth≥2025.10.14)                                 |
| trl            | ≥ 0.29.0                                             |
| accelerate     | ≥ 1.13.0                                             |
| Jupyter        | Optional, for notebook workflows                     |

## Troubleshooting

### `RuntimeError: Cannot install on Python version 3.13.x` or numba / setup.py errors

Python 3.13 is not supported (numba, vllm, and other deps require &lt;3.13). Use Python 3.12:

```bash
# With uv: ensure Python 3.12 is available, then sync
uv python install 3.12
uv sync --extra train

# Or create venv explicitly with 3.12
uv venv --python 3.12
UV_TORCH_BACKEND=cu128 uv sync --extra train
```

The project's `.python-version` file pins 3.12; uv will use it when creating the venv.

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

### `KeyError: 'qwen3_5'` / Qwen3.5 not supported

Qwen3.5 requires transformers 5.x. With transformers 4.57, use **Qwen2.5** instead:
- `unsloth/Qwen2.5-3B-Instruct-bnb-4bit`
- `unsloth/Qwen2.5-7B-Instruct-bnb-4bit`
- `Qwen/Qwen2.5-3B-Instruct`

### `NameError: name 'PreTrainedConfig' is not defined` / `check_model_inputs` ImportError

Use unsloth≥2025.10.14 (PreTrainedConfig fix) with transformers≥4.57 (check_model_inputs). Run `uv sync --extra train` to get compatible versions.

### `ImportError: cannot import name 'ConstantLengthDataset' from 'trl.trainer.utils'`

unsloth_zoo expects TRL &lt;0.20. The project pins `trl>=0.19.0,<0.20`. If you see this error, ensure you've run `uv sync --extra train` so the locked trl version is used. Alternatively, try:

```bash
pip install --upgrade --force-reinstall --no-cache-dir --no-deps unsloth unsloth_zoo
```

(A newer unsloth_zoo may fix this and allow TRL 0.20+.)

### Unsloth import order warning

If you see "Unsloth should be imported before trl, transformers, peft", ensure `training_unsloth` is imported before `training_script` in your notebook:

```python
from training_unsloth import make_training_args, run_training  # first
import training_script as base
```

## See Also

- Main [README.md](README.md) for project overview, APIs, and usage
- [uv PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/) for advanced PyTorch configuration
