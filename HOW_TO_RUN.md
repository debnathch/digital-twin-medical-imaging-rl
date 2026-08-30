# How to Run the Application

> **Digital Twin–Driven RL for Medical Imaging Workflow Optimization**
> ⚠️ All data is synthetic / research only — no real patient data.

---

## Prerequisites

- **Python ≥ 3.10**
- **macOS / Linux**

---

## Step 1 — Set Up the Virtual Environment

```bash
cd /Users/debnathchatterjee/Documents/BR/Project/LJMU_MI_WF

# Create a virtual environment (skip if .venv already exists)
python3 -m venv .venv

# Activate it
source .venv/bin/activate
```

---

## Step 2 — Install Dependencies

```bash
pip install numpy torch fastapi "uvicorn[standard]" pydantic pyyaml \
  scipy pandas plotly kaleido openpyxl jinja2 structlog websockets
```

> **Note:** The `pyproject.toml` has a build-backend issue (`setuptools.backends._legacy`), so `pip install -e .` will fail. Install dependencies directly as shown above.

---

## Step 3 — Choose How to Run

### Option A: Launch the Interactive Dashboard (Web UI)

```bash
python dashboard/app.py
```

Then open **http://127.0.0.1:8000** in your browser.

| Flag | Default | Description |
|------|---------|-------------|
| `--port` | `8000` | Server port |
| `--host` | `127.0.0.1` | Bind address |
| `--results-dir` | `results/quick` | Directory to load pre-computed results from |

> **Important:** The first page load takes **10–30 seconds** because it runs a live experiment (FIFO + Triage baselines + 30 episodes of PPO training) on every request.

---

### Option B: Run the Quick Experiment (~1–5 min)

```bash
python -m experiments.runners.full_experiment --config experiments/configs/quick.yaml
```

This runs the full **18-step pipeline** with reduced parameters:
- 2 seeds, 50 training episodes, 2 scenarios
- Outputs to `results/quick/`

---

### Option C: Run the Full Experiment (~30–60 min)

```bash
python -m experiments.runners.full_experiment --config config.yaml
```

This runs:
- **5 seeds** × 3 training scenarios × 3 test scenarios
- **500 training episodes** per seed
- Full statistical analysis and acceptance criteria
- Outputs to `results/`

---

### Option D: CLI Commands (Individual Components)

```bash
# Run baselines only
python -m medical_imaging_rl.cli baseline --policy both --scenario normal

# Train PPO/HAPPO only
python -m medical_imaging_rl.cli train --config config.yaml --seed 42 --scenario normal

# Run the full experiment pipeline
python -m medical_imaging_rl.cli run-experiment --config experiments/configs/quick.yaml
```

---

## Step 4 — View Results

After an experiment completes, find outputs in:

```
results/
├── raw/                      # Raw JSON results per policy
├── plots/                    # Plotly charts (HTML + PNG)
├── tables/                   # Thesis tables (CSV, Excel, Markdown)
├── reports/                  # Generated HTML/Markdown reports
└── experiment_manifest.json  # Reproducibility manifest
```

---

## Experimental Pipeline (18 Steps)

| Step | Description |
|------|-------------|
| 1 | Validate configuration |
| 2 | Validate Digital Twin instantiation |
| 3 | Validate baselines (FIFO, Triage) |
| 4–5 | Run FIFO & Triage across all seeds/scenarios |
| 6 | Train Hierarchical PPO/HAPPO |
| 7 | Detect convergence |
| 8 | Save model checkpoints |
| 9–10 | Evaluate PPO & test on unseen scenarios |
| 11 | Multi-seed aggregation |
| 12 | Ablation studies |
| 13–14 | Confidence intervals & statistical tests |
| 15 | Apply acceptance criteria |
| 16–18 | Generate charts, tables & final report |

---

## Quick Reference

| What you want | Command |
|---|---|
| **Web Dashboard** | `python dashboard/app.py` → http://127.0.0.1:8000 |
| **Quick experiment** | `python -m experiments.runners.full_experiment --config experiments/configs/quick.yaml` |
| **Full experiment** | `python -m experiments.runners.full_experiment --config config.yaml` |
| **Baselines only** | `python -m medical_imaging_rl.cli baseline --policy both --scenario normal` |
| **Train PPO only** | `python -m medical_imaging_rl.cli train --config config.yaml --seed 42 --scenario normal` |
| **Run tests** | `python -m pytest tests/` |
