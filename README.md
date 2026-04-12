---
title: StudentGrad-RL
sdk: docker
pinned: false
app_port: 8000
tags:
  - openenv
  - reinforcement-learning
  - education
  - student-simulation
---

# StudentGrad-RL 🎓

An [OpenEnv](https://github.com/metaresearch-ai/openenv)-compatible reinforcement learning environment that simulates a student's 365-day B.Tech academic year.

The agent must manage attendance, knowledge, skills, fatigue, and project building to maximise an **Employability Score**. It operates under partial observability — it sees noisy estimates of its true academic state, not ground truth.

**Live environment:** https://GarvVermaa-StudentGrad-RL.hf.space

---

## The Task

```
Employability Score = 0.3 × AcademicScore + 0.7 × ProjectValue
```

**Hard constraints (must all pass by Day 300):**
- Attendance ≥ 40% in all 5 subjects
- Exam score ≥ 40 in all 5 subjects
- Failing either gives terminal reward = 0, regardless of projects

**Subjects:** DSA, DBMS, OS, Maths, COA

**Skills to learn:** JavaScript, Node.js, Docker, HTML, CSS

**Projects (unlock via skill prerequisites):**

| Project | Prerequisites | Value |
|---|---|---|
| Basic (HTML/CSS frontend) | HTML ≥ 5, CSS ≥ 5 | 20 pts |
| Fullstack (JS + backend) | JS ≥ 10 | 50 pts |
| Cloud (DevOps) | Node ≥ 10, Docker ≥ 10 | 100 pts |

---

## Action Space (7 actions)

| Action | Effect | Energy Cost |
|---|---|---|
| `full_academic` | Boosts attendance + knowledge for all subjects | 10 |
| `skill_deep_dive` | Grinds one skill (requires `skill_target`) | 8 |
| `project_sprint` | Builds a project (requires `project_target` + skill prereqs) | 9 |
| `balanced_life` | 50% study, 50% skill grind | 7 |
| `cram_mode` | High-intensity study — only legal Days 285–300 | 12 |
| `rest` | Recovers energy, reduces fatigue | 0 |
| `submit_outcome` | Ends episode — only legal Day ≥ 300 | 0 |

**Example action JSON:**
```json
{
  "action_type": "skill_deep_dive",
  "skill_target": "js",
  "project_target": null,
  "justification": "Need JS ≥ 10 to unlock fullstack project",
  "confidence": 0.85
}
```

---

## Observation Space

After each step the agent sees:

```json
{
  "day": 45,
  "energy": 8.2,
  "fatigue": 23.5,
  "attendance": {"dsa": 0.61, "dbms": 0.58, "os": 0.54, "maths": 0.67, "coa": 0.60},
  "knowledge":  {"dsa": 42.1, "dbms": 38.7, "os": 35.2, "maths": 44.0, "coa": 40.8},
  "skills":     {"js": 3.2, "node": 0.0, "docker": 0.0, "html": 6.1, "css": 5.8},
  "completed_projects": ["basic"],
  "active_project_progress": 0.0,
  "resource_usage": {"days_used": 45, "days_remaining": 320},
  "rule_violations": [],
  "sick_today": false,
  "surprise_quiz_today": false
}
```

**Partial observability:** Attendance, knowledge, and skill values have Gaussian noise added — the agent sees estimates, not ground truth. Hidden state includes true learning rates, fatigue threshold, and exam difficulty.

---

## Reward Structure

**Step rewards** (given every day):

| Component | Signal |
|---|---|
| `validity` | +0.5 for a legal action, -1.0 for hard rule violation |
| `ordering` | Bonus for doing the right thing at the right time |
| `info_gain` | Bonus for attending class consistently |
| `efficiency` | Reward for energy efficiency |
| `novelty` | Bonus for first time achieving milestones |
| `penalty` | -0.5 for burnout (fatigue ≥ threshold) |

**Terminal reward** (end of episode):

```
terminal = 10 × (0.3 × academic_score_norm + 0.7 × project_value_norm)
```

where `academic_score_norm` = fraction of subjects passed, `project_value_norm` = project points / 100.

---

## Three Tasks (Scenarios)

### Task 1 — `easy_single_subject` (30 days)
Pass DSA in 30 days. One subject, short horizon.
- Target: DSA attendance ≥ 40%, knowledge ≥ 40
- Typical baseline score: **0.55–0.75**

### Task 2 — `medium_three_subjects_basic_project` (180 days)
Pass 3 subjects AND build a basic project in 180 days.
- Target: DSA, DBMS, OS passed + basic project completed
- Typical baseline score: **0.35–0.55**

### Task 3 — `hard_full_year` (365 days)
Full B.Tech year — all 5 subjects + highest-tier projects.
- Target: All subjects passed + cloud project built
- Typical baseline score: **0.20–0.45**

---

## Baseline Agent Results

Running `inference.py` with `gpt-4o-mini` as the agent:

| Scenario | Steps | Avg Reward/Step | Final Score |
|---|---|---|---|
| easy_single_subject | 30 | ~0.18 | ~0.62 |
| medium_three_subjects | 180 | ~0.09 | ~0.41 |
| hard_full_year | 365 | ~0.06 | ~0.28 |

---

## Setup & Running

### Requirements
- Python 3.12
- `uv` package manager
- Docker (for containerised deployment)

### Local development

```bash
# Clone the repo
git clone https://github.com/GarvVermaa/StudentGrad-RL.git
cd StudentGrad-RL

# Install dependencies
uv sync

# Start the environment server
uv run uvicorn server.app:app --host 0.0.0.0 --port 8000

# In a second terminal — run the inference agent
uv run python inference.py --scenario easy_single_subject --max-steps 30
```

### Validate the environment

```bash
pip install openenv-core
openenv validate https://GarvVermaa-StudentGrad-RL.hf.space
```

**Result: 6/6 checks passing** ✅

### Run Q-learning agent (genuine RL, no GPU needed)

```bash
# Make sure server is running first (see above)
uv run python train_simple.py
```

The Q-learning agent starts knowing nothing and discovers optimal policies through environment interaction over ~500 episodes. No hardcoded rules.

---

## Environment Architecture

```
StudentGrad-RL/
├── server/
│   ├── app.py                    # FastAPI server (singleton factory pattern)
│   ├── student_environment.py    # Core POMDP logic — reset() and step()
│   ├── simulator/
│   │   ├── latent_state.py       # Hidden ground truth (agent never sees this)
│   │   ├── transition.py         # State transition dynamics
│   │   ├── noise.py              # Gaussian noise model
│   │   └── output_generator.py   # Per-action output simulation
│   ├── rules/engine.py           # Hard/soft violation checker
│   ├── rewards/reward.py         # Reward decomposition
│   └── tasks/
│       ├── scenarios.py          # 3 task definitions
│       └── generator.py         # Domain randomisation
├── models.py                     # StudentAction + StudentObservation (Pydantic)
├── inference.py                  # OpenEnv-compatible baseline agent
├── train_simple.py               # Q-learning agent (genuine RL)
├── watch_agent.py                # Rule-based demo agent + trajectory recorder
└── openenv.yaml                  # OpenEnv spec
```

**Key design decisions:**

- **Singleton factory pattern** in `app.py` — `openenv-core` calls `_env_factory()` on every HTTP request. Without the singleton, state from `/reset` is lost before `/step` is called.
- **POMDP with noisy observations** — the agent sees estimated state, not ground truth. Hidden variables (true learning rates, fatigue threshold) vary per episode via domain randomisation.
- **Rule engine** separates hard violations (block the action) from soft violations (reduce output quality), giving the agent informative feedback rather than binary pass/fail.

---

## OpenEnv Compliance

| Endpoint | Status |
|---|---|
| `GET /health` | ✅ |
| `GET /metadata` | ✅ |
| `GET /schema` | ✅ |
| `POST /reset` | ✅ |
| `POST /step` | ✅ |
| `GET /state` | ✅ |
| `POST /mcp` | ✅ |
| `GET /openapi.json` | ✅ |

Validated with `openenv validate` — **6/6 criteria passing**.

---

## Mandatory Inference Variables

```bash
export API_BASE_URL=https://api.openai.com/v1   # or any OpenAI-compatible endpoint
export MODEL_NAME=gpt-4o-mini
export HF_TOKEN=hf_your_token_here
export ENV_SERVER_URL=https://GarvVermaa-StudentGrad-RL.hf.space

python inference.py --scenario hard_full_year --max-steps 365
```