---
title: Bio Experiment Environment Server
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - reinforcement-learning
  - bioinformatics
---

# Bio Experiment Environment

This repository implements an OpenEnv-compatible reinforcement learning environment for planning biological experiment pipelines. The agent does not directly see the true biological state. Instead, it proposes one structured experiment or analysis step at a time, receives a noisy simulated output, and is rewarded for valid, informative, efficient, well-calibrated plans.

The environment is designed as a partially observable Markov decision process (POMDP) with:

- hidden ground-truth biology
- hidden technical noise and failure conditions
- visible task metadata, resource usage, step history, and intermediate outputs
- dense step-wise reward plus terminal reward for conclusion quality

## How it works

At a high level, each episode looks like this:

1. `reset()` picks a biological scenario and seeds the simulator.
2. The agent receives an `ExperimentObservation` describing the task and current visible state.
3. The agent submits an `ExperimentAction` such as `collect_sample`, `run_qc`, or `differential_expression`.
4. The rule engine checks whether the action is valid at this point in the pipeline.
5. The transition engine updates hidden state, spends resources, and asks the output generator to simulate the result.
6. The reward computer scores the step for validity, ordering, information gain, efficiency, novelty, and penalties.
7. The environment returns a new observation with updated history, outputs, discoveries, violations, and reward.
8. The episode ends when the agent synthesizes a conclusion, exhausts resources, or reaches the step limit.

## The core mental model

### Hidden state

The simulator keeps a `FullLatentState` that the agent never directly sees. It contains:

- true cell populations and marker genes
- true DE genes, pathways, trajectories, and regulatory networks
- technical factors such as dropout, doublets, ambient RNA, and batch effects
- experiment progress flags
- remaining budget and time
- hidden failure conditions

### Visible state

The agent only sees `ExperimentObservation`, which includes:

- the current `TaskSpec`
- pipeline history
- available assays and tools
- resource usage
- the latest and cumulative intermediate outputs
- discovered markers and candidate mechanisms
- rule violations
- per-step reward breakdown

This separation is what makes the environment a POMDP rather than a fully observed simulator.

## Main building blocks

### `models.py`

Defines the contracts that all other modules use:

- `ActionType`: 21 discrete experiment steps, grouped into three frozensets — `WET_LAB_ACTIONS` (8), `COMPUTATIONAL_ACTIONS` (10), and `META_ACTIONS` (3)
- `SubagentType`: 9 sub-agent delegate roles (e.g. `wet_lab_planner`, `computational_analyst`, `causal_reasoning_agent`)
- `ExperimentAction`: one structured step chosen by the agent; fields include `action_type`, `method`, `parameters`, `justification`, `confidence` (clamped to `[0, 1]`), `invoked_subagent`, `tool_call_spec`, `input_targets`
- `ExperimentObservation`: what the agent can see after each step; includes `task`, `pipeline_history`, `resource_usage`, `latest_output`, `all_outputs`, `discovered_markers`, `candidate_mechanisms`, `conclusions`, `rule_violations`, `step_reward_breakdown`
- `TaskSpec`: the problem statement, organism, tissue, conditions, budget, time limit, assays, tools, paper references, and expected findings
- `IntermediateOutput`: the simulated artifact returned by a step; carries `output_type`, `success`, `quality_score`, `summary`, `data`, `uncertainty`, `warnings`, `artifacts_available`
- `ConclusionClaim`: structured claims used for final synthesis; carries `claim`, `evidence_steps`, `confidence`, `claim_type`, `supporting_data`
- `PipelineStepRecord`: compact observable record of one past step stored in history
- `ResourceUsage`: budget and time tracking visible to the agent

The action vocabulary is intentionally broad enough to mix wet-lab, computational, and meta-planning actions.

### `server/tasks/`

This is where episodes come from.

- `scenarios.py` defines a curated library of four biological scenarios as `Scenario` dataclass objects, each bundling a `TaskSpec`, a `LatentBiologicalState`, a `TechnicalState`, hidden failure conditions, and tags
- `generator.py` turns a scenario into a `(TaskSpec, FullLatentState)` pair via `TaskGenerator.generate()`; optional domain randomisation perturbs budget (±30%), time (±20%), technical noise, batch effects, cell proportions, and effect sizes

The four scenarios are:

| Name | Difficulty | Tissue | Problem | Budget | Time |
|---|---|---|---|---|---|
| `cardiac_disease_de` | easy | heart | Differential expression between healthy and dilated cardiomyopathy cardiomyocytes | $80 K | 120 days |
| `hematopoiesis_trajectory` | medium | bone marrow | Infer HSC → mature lineage trajectory with three branches | $100 K | 150 days |
| `perturbation_immune` | hard | synovial fluid | JAK inhibitor effect on T-cell states in rheumatoid arthritis | $120 K | 180 days |
| `biomarker_validation_lung` | medium | lung | Validate SPP1 as biomarker for pro-fibrotic macrophages in IPF | $90 K | 150 days |

Each scenario carries paper references with DOIs, true DE genes with log2FC values, true pathway activities, true regulatory networks, and ground-truth causal mechanisms used for terminal reward calibration.

### `server/simulator/`

This is the simulator itself.

- `latent_state.py` defines `FullLatentState`, the root aggregate of all hidden state. Key sub-structures are `LatentBiologicalState` (true DE genes, pathways, gene programs, trajectory, regulatory network, markers, causal mechanisms), `TechnicalState` (dropout, doublets, ambient RNA, sample quality), `ExperimentProgress` (18 boolean milestone flags plus counts), and `ResourceState` (internal budget and time tracking with exhaustion properties)
- `noise.py` centralises stochasticity in `NoiseModel`. All randomness flows through a single seeded `numpy.Generator`. Methods include `add_expression_noise`, `sample_effect_sizes`, `sample_p_values`, `generate_false_positives`, `generate_false_negatives`, `quality_degradation`, `sample_qc_metric`, `sample_cluster_count`, `shuffle_ranking`, and `coin_flip`
- `output_generator.py` turns an action plus hidden state into a realistic `IntermediateOutput`. Every action type has a dedicated handler conditioned on the latent state; noise is then injected — dropout in expression data, false positives and false negatives in DE and marker results, over/under-clustering, and pathway contamination
- `transition.py` applies action costs from `ACTION_COSTS`, updates progress flags, calls the output generator, degrades quality on soft violations, propagates discovered DE genes and cluster names back into latent state, and decides whether the episode is done

The output generator does not simply echo the action. It conditions outputs on the hidden state, then injects realistic noise.

### `server/rules/engine.py`

The rule engine enforces scientific and procedural constraints before each action is applied.

- hard violations block the action entirely
- soft violations allow the action, but reduce output quality and add reward penalties

The four rule families are:

1. **Prerequisites (HARD)** — each computational step requires the appropriate upstream milestone flag. For example: `normalize_data` requires `data_filtered`, `differential_expression` requires `data_normalized`, `validate_marker` requires `markers_discovered`
2. **Resource constraints (HARD/SOFT)** — budget or time exhausted is a hard block; action cost exceeding remaining budget (when budget > 0) is a soft warning
3. **Redundancy (SOFT)** — repeating an already-completed step such as `run_qc` or `normalize_data`
4. **Causal validity (SOFT)** — synthesizing conclusions without prior DE or clustering; making causal claims without validation evidence; pathway enrichment before DE

### `server/rewards/reward.py`

Rewards are decomposed rather than being a single opaque number.

Per-step reward formula:

```
R_t = r_validity + r_ordering + r_info_gain + r_efficiency + r_novelty + r_penalty + γ[φ(s_{t+1}) − φ(s_t)]
```

| Component | Weight | Description |
|---|---|---|
| `validity` | 0.3 | `1.0` if output succeeded, `−1.0` if hard violation |
| `ordering` | 0.2 | `1.0` if natural next step, `0.3` otherwise |
| `info_gain` | 0.4 | `quality_score × (1 − uncertainty)` |
| `efficiency` | 0.3 | `max(0, 1 − 5 × budget_fraction_used)` |
| `novelty` | +0.1 | Bonus when no soft violations |
| `penalty` | −0.15/violation | Per soft violation |
| `shaping` | γ = 0.99 | Potential-based over 12 progress milestones |

Terminal reward adds:

| Component | Weight | Description |
|---|---|---|
| Pipeline completeness | 3.0 | Fraction of 7 core milestones completed |
| Calibration | 4.0 | How well conclusions match hidden markers and mechanisms |
| Budget + time efficiency | 1.0 | Average fraction of budget and time remaining |
| Overconfidence penalty | −0.5/claim | For high-confidence claims (`> 0.8`) that are wrong |

This makes the environment easier to debug, benchmark, and train against.

### `server/hackathon_environment.py`

This is the orchestration layer that ties everything together.

On `reset()` it:

- seeds the noise model
- generates a task and latent state via `TaskGenerator`
- clears history, outputs, discoveries, conclusions, and cumulative reward

On `step()` it:

- checks rules
- calls the transition engine
- computes reward
- appends a `PipelineStepRecord`
- updates discovered markers and candidate mechanisms
- stores conclusion claims if the action is `synthesize_conclusion`
- builds the next `ExperimentObservation`

This file is the best place to read if you want the end-to-end control flow.

## What actually happens on one step

Here is the concrete order of operations for `env.step(action)`:

1. Increment the step counter.
2. Copy the previous latent state for reward comparison.
3. Run rule checks and split violations into hard vs soft.
4. If there is a hard violation, return a failure report without applying the action.
5. Otherwise deduct budget and time based on `ACTION_COSTS`.
6. Update latent progress flags like `samples_collected`, `qc_performed`, or `de_performed`.
7. Generate a structured simulated output for the chosen action.
8. If there were soft violations, degrade output quality (×0.5) and attach warnings.
9. Propagate artifacts back into latent state, such as discovered DE genes or cluster names.
10. Compute decomposed reward from state transition plus output quality.
11. If the episode is ending, compute terminal reward from completeness and conclusion calibration.
12. Return an observation that exposes the visible summary but not the hidden truth.

## Action costs

Each action deducts from the episode's budget and time. Computational steps also accrue compute hours.

| Action | Budget | Time (days) |
|---|---|---|
| `sequence_cells` | $15,000 | 5 |
| `prepare_library` | $8,000 | 3 |
| `collect_sample` | $5,000 | 7 |
| `validate_marker` | $5,000 | 14 |
| `culture_cells` | $3,000 | 14 |
| `perturb_gene` | $2,000 | 3 |
| `perturb_compound` | $1,000 | 2 |
| `select_cohort` | $500 | 1 |
| `run_qc` | $100 | 0.5 |
| `integrate_batches` | $300 | 1 |
| `regulatory_network_inference` | $200 | 1 |
| `cluster_cells` | $150 | 0.5 |
| `differential_expression`, `trajectory_analysis`, `pathway_enrichment` | $100–200 | 0.5–1 |
| `filter_data`, `normalize_data`, `marker_selection` | $50–100 | 0.25–0.5 |
| `synthesize_conclusion`, `design_followup_experiment`, `request_subagent_review` | $0 | 0.25–0.5 |

## Typical successful pipeline

Most scenarios reward a sensible experiment order similar to:

1. `collect_sample`
2. `prepare_library`
3. `sequence_cells`
4. `run_qc`
5. `filter_data`
6. `normalize_data`
7. `cluster_cells`
8. one or more of:
   `differential_expression`, `trajectory_analysis`, `pathway_enrichment`,
   `regulatory_network_inference`, `marker_selection`, `validate_marker`
9. `synthesize_conclusion`

The exact best sequence depends on the scenario:

- trajectory scenarios benefit from `trajectory_analysis` and regulatory inference
- biomarker scenarios benefit from DE, marker selection, and validation
- perturbation scenarios benefit from pathway-level interpretation

## Episode termination

An episode ends when one of the following happens:

- the agent chooses `synthesize_conclusion`
- resources are exhausted
- the environment reaches `MAX_STEPS` which is currently `30`

## Installation

Dependencies are managed with `uv`. The package requires Python ≥ 3.10.

> **H100 Jupyter notebook setup:** See [H100_JUPYTER_SETUP.md](H100_JUPYTER_SETUP.md) for environment setup on NVIDIA H100 instances with Jupyter.

```bash
# Core environment only
uv sync

# With dev/test tools
uv sync --extra dev

# With training dependencies (TRL, transformers, torch)
uv sync --extra train

# With bioinformatics extras (scanpy, biopython, gseapy)
uv sync --extra bio
```

Key dependency groups from `pyproject.toml`:

| Group | Key packages |
|---|---|
| core | `openenv-core[core]>=0.2.0`, `numpy`, `scipy`, `pydantic>=2.0` |
| train | `trl>=0.29`, `transformers>=5.3`, `accelerate`, `datasets`, `torch`, `matplotlib` |
| bio | `scanpy`, `biopython`, `gseapy` |
| dev | `pytest`, `pytest-cov` |

## Interfaces you can use

### 1. In-process environment

Use `BioExperimentEnvironment` when you want direct Python access with full structured observations:

```python
from models import ActionType, ExperimentAction
from server.hackathon_environment import BioExperimentEnvironment

env = BioExperimentEnvironment(scenario_name="biomarker_validation_lung")
obs = env.reset()

obs = env.step(ExperimentAction(
    action_type=ActionType.COLLECT_SAMPLE,
    parameters={"n_samples": 8},
    justification="Collect enough material for downstream single-cell analysis.",
    confidence=0.8,
))

print(obs.task.problem_statement)
print(obs.latest_output.summary if obs.latest_output else "No output yet")
print(obs.reward)
```

The constructor accepts:
- `scenario_name: Optional[str]` — pin to a specific scenario; `None` picks randomly each episode
- `domain_randomise: bool = True` — perturbs scenario parameters for generalization

### 2. OpenEnv client/server mode

Use the FastAPI app when you want to serve the environment over HTTP and WebSocket:

```bash
uv sync --extra dev
uv run uvicorn server.app:app --reload
```

The server exposes five endpoints:

| Method | Path | Description |
|---|---|---|
| `POST` | `/reset` | Start a new episode |
| `POST` | `/step` | Execute one action |
| `GET` | `/state` | Current environment state |
| `GET` | `/schema` | Action/observation JSON schemas |
| `WS` | `/ws` | WebSocket for persistent sessions |

Then connect with the client:

```python
from client import BioExperimentEnv
from models import ActionType, ExperimentAction

with BioExperimentEnv(base_url="http://localhost:8000") as env:
    result = env.reset()
    result = env.step(ExperimentAction(action_type=ActionType.COLLECT_SAMPLE))
    print(result.observation.latest_output.summary)
```

The environment class supports concurrent sessions, but the bundled server is currently configured with `max_concurrent_envs=1` in `server/app.py`.

### 3. Running a local agent

`run_agent.py` runs a single interactive episode using a local Hugging Face model:

```bash
uv run python run_agent.py
```

Configuration is via environment variables:

| Variable | Default | Description |
|---|---|---|
| `RUN_AGENT_USE_PIPELINE` | `0` | Use HF `pipeline()` path instead of direct generate |
| `RUN_AGENT_MAX_EPISODE_STEPS` | `12` | Maximum number of planning steps |

The local model defaults to `Qwen/Qwen3.5-0.8B` with sampling parameters `temperature=0.7`, `top_p=0.8`, `top_k=20`, `repetition_penalty=1.3`. The episode runs up to `MAX_EPISODE_STEPS = 12` steps. When action parsing fails, the script falls back to an observation-aware action that respects prerequisites.

PowerShell note: older PowerShell versions do not support `&&`. Run commands from the target directory directly, or use `;` as the command separator.

Windows runtime warnings:
- If you see HuggingFace symlink-cache warnings, functionality is unaffected; optionally set `HF_HUB_DISABLE_SYMLINKS_WARNING=1`.
- If you see flash attention / causal-conv fallback warnings, execution continues with a slower PyTorch path.

### 4. GRPO training

`training_script.py` follows the TRL GRPO pattern and uses OpenEnv rewards to score generated action JSON against this environment.

```bash
uv sync --extra train
uv run python training_script.py --dry-run
uv run python training_script.py --model-id Qwen/Qwen3.5-0.8B
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--model-id` | `Qwen/Qwen2.5-7B-Instruct` | Base model to fine-tune |
| `--output-dir` | `training/grpo-output` | Save directory |
| `--dataset-episodes` | `8` | Rollout episodes for prompt dataset |
| `--rollout-steps` | `6` | Steps per episode during collection |
| `--collection-policy` | `heuristic` | `random` or `heuristic` |
| `--reward-backend` | `local` | `local` (in-process) or `remote` (live server) |
| `--base-url` | `http://localhost:8000` | Server URL for remote backend |
| `--scenario-name` | all | Repeatable; restricts which scenarios are used |
| `--domain-randomise` | off | Enable domain randomisation |
| `--num-generations` | `4` | GRPO generations per prompt |
| `--max-completion-length` | `220` | Max tokens for model completions |
| `--max-prompt-length` | `768` | Max tokens for prompts |
| `--learning-rate` | `5e-6` | AdamW learning rate |
| `--dry-run` | off | Build data and test reward without training |

By default the reward function reconstructs prompt states locally so the prompt and reward stay aligned. Switch to a live server-backed reward loop with `--reward-backend remote --base-url http://localhost:8000`.

After training, the script saves plots to the output directory:

- `training_loss.png`
- `training_reward.png`
- `training_metric.png`
- `training_dashboard.png`
- `training_plot_manifest.json`

Use `--plot-metric-key <logged_key>` to force a specific extra metric on the third chart; otherwise the script auto-selects a useful logged metric such as KL or gradient norm.

### 5. Rollout collection

`training/rollout_collection.py` collects direct environment rollouts into trajectory files:

```bash
uv run python -m training.rollout_collection
```

This runs N episodes with a `random` or `heuristic` policy, saves JSON trajectories, and prints evaluation metrics.

### 6. Benchmark and scripted agents

- `training/literature_benchmark.py` runs paper-aligned action sequences and compares outcomes against curated expected findings
- `training/rollout_collection.py` collects direct environment rollouts into trajectory files
- `training_script.py` trains a GRPO policy with OpenEnv reward calls
- `run_agent.py` runs a local language model planner against the environment
- `training/trajectory.py` stores trajectories for offline RL, imitation learning, replay, and evaluation
- `training/evaluation.py` computes online, benchmark, expert-review, and fidelity-oriented metrics

## Training utilities

### `training/trajectory.py`

Provides `TrajectoryStep`, `Trajectory`, and `TrajectoryDataset` for episode serialization.

- `TrajectoryStep` stores `action`, `observation`, `reward`, `done`, `reward_breakdown`, and an optional `latent_snapshot`
- `Trajectory` accumulates steps with `add_step()`, computes `total_reward`, and exposes `save(path)` / `load(path)`
- `TrajectoryDataset` wraps a list of trajectories with `filter_successful()`, `save_dir()`, `load_dir()`, and `summary()` (n, success_rate, mean_reward, mean_length, max/min reward)

### `training/evaluation.py`

`EvaluationSuite` is a stateless class with four families of `@staticmethod` methods:

| Family | Method | Metrics |
|---|---|---|
| Online RL | `online_metrics(trajectories)` | `mean_return`, `median_return`, `std_return`, `mean_episode_length`, `success_rate` |
| Offline benchmark | `benchmark_metrics(dataset)` | `pipeline_validity_rate`, `ordering_score`, `action_diversity`, `mean_conclusion_confidence` |
| Expert review | `expert_review_metrics(...)` | Placeholder; averages provided scores |
| Simulator fidelity | `simulator_fidelity_metrics(sim, real)` | `reward_distribution_gap` |

### `training/literature_benchmark.py`

`run_paper_benchmark(problem_statement, scenario_name, domain_randomise)` runs a paper-aligned action pipeline and scores against `expected_findings` using keyword matching. Returns a `PaperBenchmarkResult` with `match_ratio`.

## Docker deployment

The server ships with a `server/Dockerfile`. It uses a multi-stage build based on `openenv-base`, installs dependencies via `uv`, and starts `uvicorn server.app:app` on port 8000.

```bash
docker build -f server/Dockerfile -t bio-experiment-env .
docker run -p 8000:8000 bio-experiment-env
```

The `openenv.yaml` file configures the deployment for the OpenEnv platform:

```yaml
spec_version: 1
name: hackathon
type: space
runtime: fastapi
app: server.app:app
port: 8000
```

## Why this is useful

This environment is trying to model a realistic scientific planning loop rather than a toy decision problem:

- actions have prerequisites
- outputs are noisy and imperfect
- budget and time matter
- not every correct-looking answer is well supported
- final conclusions are scored against hidden ground truth

That makes it suitable for:

- agent planning benchmarks
- RL experiments on long-horizon scientific reasoning
- literature-grounded evaluation
- comparing structured policies against LLM-driven planners

## Project map

```text
.
├── client.py                     # OpenEnv HTTP/WebSocket client
├── models.py                     # Shared action / observation / task schemas
├── openenv.yaml                  # OpenEnv platform deployment config
├── pyproject.toml                # Package metadata and dependency groups
├── run_agent.py                  # Single-episode interactive agent runner
├── server/
│   ├── app.py                    # FastAPI/OpenEnv server entry point
│   ├── Dockerfile                # Multi-stage Docker build
│   ├── hackathon_environment.py  # Main environment orchestration
│   ├── requirements.txt          # Minimal server dependencies
│   ├── rewards/
│   │   └── reward.py             # Decomposed reward model
│   ├── rules/
│   │   └── engine.py             # Biological constraint checking
│   ├── simulator/
│   │   ├── latent_state.py       # Hidden biological, technical, progress, resource state
│   │   ├── noise.py              # Seeded stochastic noise model
│   │   ├── output_generator.py   # Per-action simulated output generation
│   │   └── transition.py         # State transition engine and ACTION_COSTS table
│   ├── subagents/                # Placeholder for future sub-agent integration
│   └── tasks/
│       ├── generator.py          # TaskGenerator with domain randomisation
│       └── scenarios.py          # SCENARIO_LIBRARY with 4 curated scenarios
├── training/
│   ├── evaluation.py             # EvaluationSuite metrics
│   ├── literature_benchmark.py   # Paper-backed benchmark flow
│   ├── rollout_collection.py     # Direct rollout collection helper
│   └── trajectory.py             # Trajectory serialization and dataset utilities
├── training_script.py            # TRL GRPO training entry point
└── tests/
    ├── test_environment.py
    ├── test_literature_benchmark.py
    ├── test_models.py
    ├── test_rewards.py
    ├── test_rules.py
    ├── test_simulator.py
    └── test_training_script.py
```

## Quick sanity check

```bash
uv run pytest tests/test_environment.py tests/test_literature_benchmark.py -q
```

Those tests verify:

- reset and step lifecycle
- valid vs invalid pipeline behavior
- conclusion termination
- literature-backed scenario selection
- benchmark matching for curated expected findings

Run the full suite with coverage:

```bash
uv run pytest tests/ --cov -q
```
