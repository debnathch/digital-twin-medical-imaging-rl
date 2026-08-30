# Digital Twin–Driven Reinforcement Learning for Dynamic Medical Imaging Workflow Optimization

> Master's Thesis Research Application — LJMU
>
> **⚠️ SYNTHETIC / RESEARCH DATA ONLY** — No real patient data is used or stored.

## Overview

This application implements a complete experimental framework for evaluating
whether **Hierarchical PPO/HAPPO** (a multi-agent reinforcement learning approach)
can outperform traditional scheduling baselines (**FIFO** and **Static Triage Priority**)
for optimizing medical imaging workflows in hospitals.

### Architecture

```
Medical Imaging Event
        ↓
  Digital Twin (Environment / Simulation)
        ↓
  Global State S_t = [Q_t, W_t, M_t, G_t, R_t, SLA_t, C_t]
        ↓
  Workflow Agent (Level 1)  →  Strategy context
        ↓
  Queue Agent / Resource Agent / AI Model Agent (Level 2)
        ↓
  Joint Action  →  Constraint Validator
        ↓
  Digital Twin Step  →  New State + Rewards
        ↓
  HAPPO Sequential Update
```

### Key Research Principle

> **The framework does NOT hard-code or fabricate results showing PPO is better.**
> PPO is declared "preferred" ONLY if predefined acceptance criteria are met.
> If the evidence doesn't support PPO, the system reports "NO SUPPORTED WINNER."

---

## Quick Start

### 1. Install Dependencies

```bash
cd /Users/debnathchatterjee/Documents/BR/Project/LJMU_MI_WF

# Create and activate a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install the project and its dependencies
pip install -e ".[dev]"
```

### 2. Run the Quick Experiment (~1-5 min)

```bash
python -m experiments.runners.full_experiment --config experiments/configs/quick.yaml
```

This runs the full 18-step pipeline with reduced parameters:
- 2 seeds, 50 training episodes, 2 scenarios
- Outputs to `results/quick/`

### 3. Run the Full Experiment (~30-60 min)

```bash
python -m experiments.runners.full_experiment --config config.yaml
```

This runs:
- 5 seeds × 3 training scenarios × 3 test scenarios
- 500 training episodes per seed
- Full statistical analysis and acceptance criteria
- Outputs to `results/`

---

## CLI Commands

```bash
# Run baselines only
python -m medical_imaging_rl.cli baseline --policy both --scenario normal

# Train PPO/HAPPO only
python -m medical_imaging_rl.cli train --config config.yaml --seed 42 --scenario normal

# Run full experiment pipeline
python -m medical_imaging_rl.cli run-experiment --config experiments/configs/quick.yaml
```

---

## Project Structure

```
LJMU_MI_WF/
├── config.yaml                          # Master experiment configuration
├── pyproject.toml                       # Python project metadata
├── README.md
│
├── simulator/                           # Digital Twin (Environment)
│   ├── digital_twin/environment.py      # MedicalImagingDigitalTwin
│   ├── events/                          # Event system (heap-based queue)
│   ├── state/                           # Study, GlobalState dataclasses
│   ├── resources/                       # Scanner, Radiologist, AIModel, GPU
│   └── scenarios/                       # 6 scenarios (A-F)
│
├── agents/                              # RL Agents
│   ├── observations.py                  # Heterogeneous observation builders
│   ├── workflow/agent.py                # Level 1: Workflow Agent (5 actions)
│   ├── queue/agent.py                   # Level 2: Queue Agent (8 actions)
│   ├── resource/agent.py                # Level 2: Resource Agent (60 actions)
│   ├── model_selection/agent.py         # Level 2: AI Model Agent (5 actions)
│   └── constraint_validator.py          # 9 safety constraints
│
├── rl/                                  # Reinforcement Learning Core
│   ├── ppo/                             # PPO: ActorCritic, Buffer, GAE
│   ├── happo/                           # HAPPO: Sequential update trainer
│   └── training/trainer.py              # Main training loop
│
├── baselines/                           # Non-learning baselines
│   ├── fifo.py                          # FIFO (arrival order)
│   └── triage.py                        # Static Triage (Critical > Urgent > Routine)
│
├── rewards/                             # Reward system
│   ├── global_reward.py                 # 7-component weighted global reward
│   ├── workflow_reward.py               # Local reward for Workflow Agent
│   ├── queue_reward.py                  # Local reward for Queue Agent
│   ├── resource_reward.py               # Local reward for Resource Agent
│   └── model_reward.py                  # Local reward for AI Model Agent
│
├── evaluation/                          # Evaluation framework
│   ├── metrics.py                       # 19 primary/secondary metrics
│   ├── convergence.py                   # Convergence detection
│   ├── statistics.py                    # Welch's t-test, Mann-Whitney U, bootstrap CI
│   ├── acceptance.py                    # Multi-criteria acceptance gate
│   └── ablation.py                      # Ablation study runner
│
├── experiments/                         # Experiment pipeline
│   ├── runners/full_experiment.py       # 18-step experiment pipeline
│   └── configs/                         # quick.yaml, default.yaml
│
├── reports/                             # Output generation
│   ├── chart_generator.py               # Plotly charts (HTML + PNG)
│   ├── table_generator.py               # Thesis tables (CSV, Excel, MD)
│   └── report_generator.py              # HTML/Markdown reports
│
├── medical_imaging_rl/                  # Package entry point
│   ├── cli.py                           # CLI with subcommands
│   └── logging.py                       # Structured logging
│
├── tests/                               # Test suite
└── results/                             # Output directory (generated)
```

---

## Experimental Pipeline (18 Steps)

| Step | Description |
|------|-------------|
| 1 | Validate configuration |
| 2 | Validate Digital Twin instantiation |
| 3 | Validate baselines (FIFO, Triage) |
| 4 | Run FIFO across all seeds and scenarios |
| 5 | Run Static Triage across all seeds and scenarios |
| 6 | Train Hierarchical PPO/HAPPO |
| 7 | Detect convergence |
| 8 | Save model checkpoints |
| 9 | Evaluate trained PPO policy |
| 10 | Test across unseen scenarios |
| 11 | Multi-seed aggregation |
| 12 | Ablation studies |
| 13 | Calculate confidence intervals |
| 14 | Statistical significance tests |
| 15 | Apply acceptance criteria |
| 16 | Generate charts |
| 17 | Generate thesis tables |
| 18 | Generate final report |

---

## Configuration

Edit `config.yaml` or create a new config. Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `training.num_episodes` | 500 | Training episodes |
| `experiment.seeds` | [42,123,456,789,1024] | Random seeds |
| `ppo.learning_rate` | 0.0003 | Learning rate |
| `ppo.clip_epsilon` | 0.20 | PPO clip range |
| `rewards.agent_alpha` | 0.6 | Local reward weight |
| `rewards.agent_beta` | 0.4 | Global reward weight |
| `convergence.min_episodes` | 200 | Min episodes before convergence |

---

## Scenarios

| Scenario | Arrival Rate | Description |
|----------|-------------|-------------|
| Normal (A) | 10/hr | Standard operating conditions |
| Peak (B) | 20/hr | Peak load |
| High Urgent (C) | 10/hr | 40% critical, 35% urgent |
| Low Compute (D) | 10/hr | 50% GPU capacity |
| Low Radiologist (E) | 10/hr | 50% radiologist availability |
| High Latency (F) | 10/hr | 3× inference latency |

---

## Output

After running an experiment, find results in the output directory:

```
results/
├── raw/                    # Raw JSON results per policy
├── plots/                  # Plotly charts (HTML + PNG)
├── tables/                 # Thesis tables (CSV, Excel, MD)
├── reports/                # Generated reports
└── experiment_manifest.json  # Reproducibility manifest
```

---

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0
- NumPy, SciPy, Pandas, Plotly, PyYAML

See `pyproject.toml` for the full dependency list.
