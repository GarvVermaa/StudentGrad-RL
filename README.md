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

## What "how it works" means here

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

- `ExperimentAction`: one structured step chosen by the agent
- `ExperimentObservation`: what the agent can see after each step
- `TaskSpec`: the problem statement, budget, time limit, assays, tools, and expected findings
- `IntermediateOutput`: the simulated artifact returned by a step
- `ConclusionClaim`: structured claims used for final synthesis

The action vocabulary is intentionally broad enough to mix wet-lab, computational, and meta-planning actions.

### `server/tasks/`

This is where episodes come from.

- `scenarios.py` defines a small library of curated biological scenarios
- `generator.py` turns a scenario into a `(TaskSpec, FullLatentState)` pair
- optional domain randomization perturbs budget, time, noise, batch effects, cell proportions, and effect sizes

Right now the scenario library includes:

- `cardiac_disease_de`: disease vs healthy differential expression in heart tissue
- `hematopoiesis_trajectory`: developmental trajectory inference in bone marrow
- `perturbation_immune`: treatment response under JAK inhibition
- `biomarker_validation_lung`: follow-up validation of `SPP1` in IPF

### `server/simulator/`

This is the simulator itself.

- `latent_state.py` defines hidden biological, technical, progress, and resource state
- `noise.py` centralizes stochasticity so episodes are reproducible from a seed
- `output_generator.py` turns an action plus hidden state into a realistic `IntermediateOutput`
- `transition.py` applies action costs, updates progress flags, propagates artifacts, and decides whether the episode is done

The output generator does not simply echo the action. It conditions outputs on the hidden state, then injects realistic noise such as dropout, false positives, false negatives, and imperfect clustering.

### `server/rules/engine.py`

The rule engine enforces scientific and procedural constraints before each action is applied.

- hard violations block the action entirely
- soft violations allow the action, but reduce output quality and add reward penalties

Examples:

- sequencing before library prep is a hard violation
- running QC twice is a soft redundancy violation
- making causal claims without enough evidence is a soft validity violation

### `server/rewards/reward.py`

Rewards are decomposed rather than being a single opaque number.

Per-step reward includes:

- validity
- ordering
- information gain
- efficiency
- novelty
- penalties
- potential-based shaping

Terminal reward adds:

- pipeline completeness
- calibration of conclusions against hidden truth
- remaining budget and time efficiency
- overconfidence penalties for strong but incorrect claims

This makes the environment easier to debug, benchmark, and train against.

### `server/hackathon_environment.py`

This is the orchestration layer that ties everything together.

On `reset()` it:

- seeds the noise model
- generates a task and latent state
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
8. If there were soft violations, degrade output quality and attach warnings.
9. Propagate artifacts back into latent state, such as discovered DE genes or cluster names.
10. Compute decomposed reward from state transition plus output quality.
11. If the episode is ending, compute terminal reward from completeness and conclusion calibration.
12. Return an observation that exposes the visible summary but not the hidden truth.

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

The exact best sequence depends on the scenario. For example:

- trajectory scenarios benefit from `trajectory_analysis` and regulatory inference
- biomarker scenarios benefit from DE, marker selection, and validation
- perturbation scenarios benefit from pathway-level interpretation

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

### 2. OpenEnv client/server mode

Use the FastAPI app when you want to serve the environment over HTTP and WebSocket:

```bash
uv sync --extra dev
uv run uvicorn server.app:app --reload
```

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

### 3. Gymnasium wrapper

Use `training/gym_wrapper.py` when you want a classic RL interface:

```python
from training.gym_wrapper import BioExperimentGymEnv

env = BioExperimentGymEnv()
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step({
    "action_type": 0,
    "confidence": 0.7,
})
```

This wrapper vectorizes the structured observation into arrays and reduces the action interface to:

- a discrete action type index
- a scalar confidence value

### 4. Benchmark and scripted agents

- `training/literature_benchmark.py` runs paper-aligned action sequences and compares outcomes against curated expected findings
- `run_agent.py` runs a local language model planner against the environment
- `training/trajectory.py` stores trajectories for offline RL, imitation learning, replay, and evaluation
- `training/evaluation.py` computes online, benchmark, expert-review, and fidelity-oriented metrics

## Episode termination

An episode ends when one of the following happens:

- the agent chooses `synthesize_conclusion`
- resources are exhausted
- the environment reaches `MAX_STEPS` which is currently `30`

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

## Minimal project map

```text
.
├── client.py                     # OpenEnv client
├── models.py                     # Shared action / observation / task schemas
├── server/
│   ├── app.py                    # FastAPI/OpenEnv server
│   ├── hackathon_environment.py  # Main environment orchestration
│   ├── rewards/                  # Reward model
│   ├── rules/                    # Constraint checking
│   ├── simulator/                # Latent state, noise, outputs, transitions
│   └── tasks/                    # Scenario library and task generation
├── training/
│   ├── evaluation.py             # Metrics
│   ├── gym_wrapper.py            # Gymnasium wrapper
│   ├── literature_benchmark.py   # Paper-backed benchmark flow
│   └── trajectory.py             # Trajectory serialization
└── tests/                        # Unit and integration tests
```

## Quick sanity check

The current implementation was sanity-checked with:

```bash
uv run pytest tests/test_environment.py tests/test_literature_benchmark.py -q
```

Those tests verify:

- reset and step lifecycle
- valid vs invalid pipeline behavior
- conclusion termination
- literature-backed scenario selection
- benchmark matching for curated expected findings
