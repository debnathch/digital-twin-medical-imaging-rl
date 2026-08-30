#!/usr/bin/env python3
"""Interactive Web Dashboard for Thesis Results.

Launch:
    python3 dashboard/app.py
    python3 dashboard/app.py --port 8080
    python3 dashboard/app.py --results-dir results/quick

Then open http://localhost:8000 in your browser.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# ── Add project root to path ────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
import uvicorn

# ── Globals filled at startup ────────────────────────────────────────
RESULTS_DIR: Path = PROJECT_ROOT / "results" / "quick"
EXPERIMENT_DATA: dict = {}


# ── FastAPI App ──────────────────────────────────────────────────────
app = FastAPI(title="Medical Imaging RL — Thesis Dashboard")


def load_results():
    """Load all JSON result files."""
    global EXPERIMENT_DATA
    data = {}

    manifest = RESULTS_DIR / "experiment_manifest.json"
    if manifest.exists():
        data["manifest"] = json.loads(manifest.read_text())

    for name in ["fifo_results", "triage_results", "ppo_eval_results"]:
        p = RESULTS_DIR / "raw" / f"{name}.json"
        if p.exists():
            data[name] = json.loads(p.read_text())

    # Scan for training histories inside sub-dirs
    training_dirs = list(RESULTS_DIR.glob("**/training_history.json"))
    histories = []
    for td in training_dirs:
        histories.append(json.loads(td.read_text()))
    data["training_histories"] = histories

    EXPERIMENT_DATA = data


# ── Run a live experiment for the dashboard ──────────────────────────

def run_live_experiment() -> dict:
    """Run baselines + short PPO training and return structured results."""
    from simulator.digital_twin.environment import MedicalImagingDigitalTwin, JointAction
    from simulator.scenarios.scenario import ScenarioFactory
    from baselines.fifo import FIFOPolicy
    from baselines.triage import TriagePriorityPolicy
    from rl.training.trainer import ExperimentTrainer, TrainingConfig
    from evaluation.convergence import ConvergenceConfig
    from rewards.global_reward import RewardWeights

    results = {"fifo": {}, "triage": {}, "ppo": {}, "training_history": []}
    seeds = [42, 123]

    for seed in seeds:
        # FIFO
        sc = ScenarioFactory.create("normal")
        dt = MedicalImagingDigitalTwin({"max_steps_per_episode": 200}, sc, seed)
        r = FIFOPolicy().run_episode(dt)
        m = r["metrics"]
        results["fifo"][seed] = {
            "completed": r["completed_studies"],
            "tat": m.mean_turnaround_time,
            "sla": m.sla_compliance,
            "breaches": r["sla_breaches"],
            "queue_wait": m.mean_queue_waiting_time,
            "reward": m.mean_global_reward,
        }

        # Triage
        dt2 = MedicalImagingDigitalTwin({"max_steps_per_episode": 200}, sc, seed)
        r2 = TriagePriorityPolicy().run_episode(dt2)
        m2 = r2["metrics"]
        results["triage"][seed] = {
            "completed": r2["completed_studies"],
            "tat": m2.mean_turnaround_time,
            "sla": m2.sla_compliance,
            "breaches": r2["sla_breaches"],
            "queue_wait": m2.mean_queue_waiting_time,
            "reward": m2.mean_global_reward,
        }

    # PPO training (short)
    tc = TrainingConfig(
        num_episodes=30, eval_interval=15, eval_episodes=2,
        checkpoint_interval=50, hidden_dim=64,
        convergence=ConvergenceConfig(window_size=10, min_episodes=15, patience=3),
        output_dir=str(RESULTS_DIR / "dashboard_run"),
        experiment_name="dashboard",
    )
    sc = ScenarioFactory.create("normal")
    trainer = ExperimentTrainer(tc, sc, seed=42)
    train_result = trainer.train()

    results["training_history"] = train_result["training_history"]
    results["convergence"] = train_result["convergence_result"].message

    # PPO eval metrics
    completed = trainer.digital_twin.completed_studies
    tats = [s.turnaround_time for s in completed if s.turnaround_time]
    n_total = len(trainer.digital_twin.all_studies)
    n_breached = sum(1 for s in trainer.digital_twin.all_studies.values() if s.sla_breached)
    results["ppo"] = {
        "completed": len(completed),
        "tat": float(np.mean(tats)) if tats else 0,
        "sla": 1 - n_breached / max(1, n_total),
        "breaches": trainer.digital_twin.sla_breaches,
        "reward": train_result["training_history"][-1]["reward"] if train_result["training_history"] else 0,
    }

    # Scenario comparison
    scenario_results = {}
    for sname in ["normal", "peak", "high_urgent", "low_compute"]:
        sc2 = ScenarioFactory.create(sname)
        dt_f = MedicalImagingDigitalTwin({"max_steps_per_episode": 200}, sc2, 42)
        rf = FIFOPolicy().run_episode(dt_f)
        dt_t = MedicalImagingDigitalTwin({"max_steps_per_episode": 200}, sc2, 42)
        rt = TriagePriorityPolicy().run_episode(dt_t)
        scenario_results[sname] = {
            "fifo_completed": rf["completed_studies"],
            "fifo_tat": rf["metrics"].mean_turnaround_time,
            "fifo_sla": rf["metrics"].sla_compliance,
            "triage_completed": rt["completed_studies"],
            "triage_tat": rt["metrics"].mean_turnaround_time,
            "triage_sla": rt["metrics"].sla_compliance,
        }
    results["scenarios"] = scenario_results

    return results


# ── HTML Dashboard ───────────────────────────────────────────────────

def build_dashboard_html(results: dict) -> str:
    """Build complete interactive HTML dashboard."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.io as pio

    # ── Chart 1: Training Curves ─────────────────────────────────
    history = results.get("training_history", [])
    fig_train = make_subplots(rows=1, cols=2,
        subplot_titles=("Global Reward vs Episode", "SLA Compliance vs Episode"))
    if history:
        eps = [h["episode"] for h in history]
        rewards = [h["reward"] for h in history]
        sla = [h["sla_compliance"] for h in history]

        fig_train.add_trace(go.Scatter(x=eps, y=rewards, mode="lines",
            name="Reward", line=dict(color="#4CAF50", width=1), opacity=0.5), row=1, col=1)
        # Moving average
        w = min(5, len(rewards))
        if w > 1:
            ma = np.convolve(rewards, np.ones(w)/w, mode="valid")
            fig_train.add_trace(go.Scatter(x=eps[w-1:], y=ma.tolist(), mode="lines",
                name="Moving Avg", line=dict(color="#4CAF50", width=3)), row=1, col=1)

        fig_train.add_trace(go.Scatter(x=eps, y=sla, mode="lines+markers",
            name="SLA %", line=dict(color="#2196F3", width=2),
            marker=dict(size=4)), row=1, col=2)

    fig_train.update_layout(height=400, template="plotly_white",
        margin=dict(t=40, b=40), showlegend=False)
    chart_training = pio.to_html(fig_train, full_html=False, include_plotlyjs=False)

    # ── Chart 2: Policy Comparison Bars ──────────────────────────
    fifo = results.get("fifo", {})
    triage = results.get("triage", {})
    ppo = results.get("ppo", {})

    fifo_avg = _avg_dict(fifo) if isinstance(fifo, dict) and any(isinstance(v, dict) for v in fifo.values()) else fifo
    triage_avg = _avg_dict(triage) if isinstance(triage, dict) and any(isinstance(v, dict) for v in triage.values()) else triage
    ppo_flat = ppo if isinstance(ppo, dict) and "completed" in ppo else {}

    policies = ["FIFO", "Static Triage", "PPO/HAPPO"]
    colors = ["#2196F3", "#FF9800", "#4CAF50"]

    metrics_to_plot = [
        ("Completed Studies", "completed"),
        ("Mean TAT (min)", "tat"),
        ("SLA Compliance", "sla"),
        ("SLA Breaches", "breaches"),
    ]
    fig_compare = make_subplots(rows=2, cols=2, subplot_titles=[m[0] for m in metrics_to_plot])

    for i, (label, key) in enumerate(metrics_to_plot):
        r, c = (i // 2) + 1, (i % 2) + 1
        vals = [
            fifo_avg.get(key, 0),
            triage_avg.get(key, 0),
            ppo_flat.get(key, 0),
        ]
        fig_compare.add_trace(go.Bar(
            x=policies, y=vals,
            marker_color=colors,
            text=[f"{v:.1f}" if isinstance(v, float) else str(v) for v in vals],
            textposition="auto",
        ), row=r, col=c)

    fig_compare.update_layout(height=500, template="plotly_white",
        margin=dict(t=40, b=20), showlegend=False)
    chart_compare = pio.to_html(fig_compare, full_html=False, include_plotlyjs=False)

    # ── Chart 3: Scenario Heatmap ────────────────────────────────
    scenarios = results.get("scenarios", {})
    chart_scenario = ""
    if scenarios:
        snames = list(scenarios.keys())
        fig_sc = make_subplots(rows=1, cols=2,
            subplot_titles=("Completed Studies by Scenario", "SLA Compliance by Scenario"))

        fig_sc.add_trace(go.Bar(name="FIFO",
            x=snames, y=[scenarios[s]["fifo_completed"] for s in snames],
            marker_color="#2196F3"), row=1, col=1)
        fig_sc.add_trace(go.Bar(name="Triage",
            x=snames, y=[scenarios[s]["triage_completed"] for s in snames],
            marker_color="#FF9800"), row=1, col=1)

        fig_sc.add_trace(go.Bar(name="FIFO",
            x=snames, y=[scenarios[s]["fifo_sla"] for s in snames],
            marker_color="#2196F3", showlegend=False), row=1, col=2)
        fig_sc.add_trace(go.Bar(name="Triage",
            x=snames, y=[scenarios[s]["triage_sla"] for s in snames],
            marker_color="#FF9800", showlegend=False), row=1, col=2)

        fig_sc.update_layout(height=380, template="plotly_white", barmode="group",
            margin=dict(t=40, b=20))
        chart_scenario = pio.to_html(fig_sc, full_html=False, include_plotlyjs=False)

    # ── Chart 4: Training Loss & Entropy ─────────────────────────
    chart_losses = ""
    if history and "workflow_policy_loss" in history[0]:
        fig_loss = make_subplots(rows=1, cols=2,
            subplot_titles=("Policy Loss (Workflow Agent)", "Entropy (Workflow Agent)"))
        eps = [h["episode"] for h in history]
        fig_loss.add_trace(go.Scatter(x=eps,
            y=[h.get("workflow_policy_loss", 0) for h in history],
            mode="lines", line=dict(color="#E91E63", width=2)), row=1, col=1)
        fig_loss.add_trace(go.Scatter(x=eps,
            y=[h.get("workflow_entropy", 0) for h in history],
            mode="lines", line=dict(color="#9C27B0", width=2)), row=1, col=2)
        fig_loss.update_layout(height=350, template="plotly_white",
            margin=dict(t=40, b=20), showlegend=False)
        chart_losses = pio.to_html(fig_loss, full_html=False, include_plotlyjs=False)

    # ── Build comparison table ───────────────────────────────────
    def fmt(v, pct=False):
        if v is None: return "—"
        if pct: return f"{v:.1%}"
        if isinstance(v, float): return f"{v:.1f}"
        return str(v)

    rows_html = ""
    for label, key, pct in [
        ("Completed Studies", "completed", False),
        ("Mean TAT (min)", "tat", False),
        ("SLA Compliance", "sla", True),
        ("SLA Breaches", "breaches", False),
        ("Mean Queue Wait (min)", "queue_wait", False),
    ]:
        rows_html += f"""<tr>
            <td style="font-weight:600">{label}</td>
            <td>{fmt(fifo_avg.get(key), pct)}</td>
            <td>{fmt(triage_avg.get(key), pct)}</td>
            <td>{fmt(ppo_flat.get(key), pct)}</td>
        </tr>"""

    convergence_msg = results.get("convergence", "—")
    verdict = results.get("verdict", "")
    if not verdict:
        ppo_r = ppo_flat.get("reward", 0) or 0
        fifo_r = fifo_avg.get("reward", 0) or 0
        triage_r = triage_avg.get("reward", 0) or 0
        if ppo_r > fifo_r and ppo_r > triage_r:
            verdict = "PPO PREFERRED (pending full statistical validation)"
            verdict_color = "#4CAF50"
        else:
            verdict = "NO SUPPORTED WINNER — further calibration needed"
            verdict_color = "#FF9800"
    else:
        verdict_color = "#4CAF50" if "PREFERRED" in verdict.upper() else "#FF9800"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Digital Twin RL — Thesis Results Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: #f5f7fa; color: #1a1a2e; }}
  .header {{ background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
             color: white; padding: 32px 48px; }}
  .header h1 {{ font-size: 24px; font-weight: 700; margin-bottom: 6px; }}
  .header p {{ font-size: 14px; opacity: 0.8; }}
  .container {{ max-width: 1300px; margin: 0 auto; padding: 24px; }}
  .card {{ background: white; border-radius: 12px; padding: 24px;
           margin-bottom: 24px; box-shadow: 0 2px 12px rgba(0,0,0,0.06); }}
  .card h2 {{ font-size: 18px; font-weight: 700; margin-bottom: 16px;
              color: #1a1a2e; border-bottom: 2px solid #e8ecf1; padding-bottom: 8px; }}
  .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
  .grid-3 {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; }}
  .metric-card {{ background: #f8f9fc; border-radius: 10px; padding: 20px; text-align: center; }}
  .metric-card .value {{ font-size: 32px; font-weight: 800; }}
  .metric-card .label {{ font-size: 13px; color: #6b7280; margin-top: 4px; }}
  .fifo {{ color: #2196F3; }} .triage {{ color: #FF9800; }} .ppo {{ color: #4CAF50; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
  th {{ background: #f1f5f9; padding: 12px 16px; text-align: left; font-weight: 600;
       border-bottom: 2px solid #e2e8f0; }}
  td {{ padding: 10px 16px; border-bottom: 1px solid #f1f5f9; }}
  tr:hover {{ background: #f8fafc; }}
  .verdict {{ border-radius: 12px; padding: 24px; text-align: center; margin-bottom: 24px; }}
  .verdict h2 {{ font-size: 14px; text-transform: uppercase; letter-spacing: 2px;
                 margin-bottom: 8px; opacity: 0.7; }}
  .verdict .decision {{ font-size: 22px; font-weight: 800; }}
  .convergence-badge {{ display: inline-block; padding: 4px 14px; border-radius: 20px;
                        font-size: 13px; font-weight: 600; }}
  .badge-green {{ background: #dcfce7; color: #166534; }}
  .badge-yellow {{ background: #fef9c3; color: #854d0e; }}
  .synthetic-notice {{ background: #fef3c7; border-left: 4px solid #f59e0b;
                        padding: 12px 16px; border-radius: 6px; font-size: 13px;
                        margin-bottom: 24px; }}
  .footer {{ text-align: center; padding: 24px; color: #9ca3af; font-size: 12px; }}
  @media (max-width: 768px) {{ .grid-2, .grid-3 {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>

<div class="header">
  <h1>🏥 Digital Twin–Driven RL for Medical Imaging Workflow Optimization</h1>
  <p>Master's Thesis — Experimental Results Dashboard</p>
</div>

<div class="container">

  <div class="synthetic-notice">
    ⚠️ <strong>SYNTHETIC / RESEARCH DATA ONLY</strong> — All results are from simulated
    hospital workloads. No real patient data is used.
  </div>

  <!-- Verdict -->
  <div class="verdict" style="background: {verdict_color}22; border: 2px solid {verdict_color};">
    <h2 style="color: {verdict_color};">Experimental Conclusion</h2>
    <div class="decision" style="color: {verdict_color};">{verdict}</div>
    <p style="margin-top:8px; font-size:13px; color:#666;">
      Convergence: <span class="convergence-badge {'badge-green' if 'CONVERGED' in convergence_msg else 'badge-yellow'}">{convergence_msg}</span>
    </p>
  </div>

  <!-- KPI Cards -->
  <div class="grid-3">
    <div class="metric-card">
      <div class="value fifo">{fmt(fifo_avg.get('completed'))}</div>
      <div class="label">FIFO — Completed Studies</div>
    </div>
    <div class="metric-card">
      <div class="value triage">{fmt(triage_avg.get('completed'))}</div>
      <div class="label">Triage — Completed Studies</div>
    </div>
    <div class="metric-card">
      <div class="value ppo">{fmt(ppo_flat.get('completed'))}</div>
      <div class="label">PPO/HAPPO — Completed Studies</div>
    </div>
  </div>

  <!-- Comparison Table -->
  <div class="card">
    <h2>📊 Policy Comparison</h2>
    <table>
      <thead>
        <tr><th>Metric</th><th class="fifo">FIFO</th><th class="triage">Static Triage</th><th class="ppo">PPO/HAPPO</th></tr>
      </thead>
      <tbody>{rows_html}</tbody>
    </table>
  </div>

  <!-- Charts -->
  <div class="card">
    <h2>📈 Policy Comparison Charts</h2>
    {chart_compare}
  </div>

  <div class="card">
    <h2>🧠 PPO/HAPPO Training Curves</h2>
    {chart_training}
  </div>

  {"<div class='card'><h2>📉 Training Loss & Entropy</h2>" + chart_losses + "</div>" if chart_losses else ""}

  {"<div class='card'><h2>🗺️ Scenario Comparison</h2>" + chart_scenario + "</div>" if chart_scenario else ""}

  <!-- Architecture -->
  <div class="card">
    <h2>🏗️ System Architecture</h2>
    <div style="text-align:center; padding:20px; font-family:monospace; font-size:13px; line-height:2.2; background:#f8f9fc; border-radius:8px;">
      Medical Imaging Event<br>
      ↓<br>
      <strong>Digital Twin</strong> (Event-Driven Simulation)<br>
      ↓<br>
      Global State S<sub>t</sub> = [Q, W, M, G, R, SLA, C]<br>
      ↓<br>
      <span style="color:#4CAF50; font-weight:700;">Workflow Agent</span> (Level 1 — 5 strategies)<br>
      ↓ strategy context<br>
      <span style="color:#2196F3; font-weight:700;">Queue · Resource · AI Model</span> (Level 2 — 8 + 60 + 5 actions)<br>
      ↓<br>
      Joint Action → <span style="color:#E91E63;">Constraint Validator</span> → Digital Twin Step<br>
      ↓<br>
      R<sub>i</sub> = α · R<sub>local</sub> + β · R<sub>global</sub> → <span style="color:#9C27B0; font-weight:700;">HAPPO Sequential Update</span>
    </div>
  </div>

</div>

<div class="footer">
  Digital Twin–Driven RL for Medical Imaging Workflow Optimization · LJMU Master's Thesis
</div>

</body>
</html>"""


def _avg_dict(d: dict) -> dict:
    """Average metrics across seeds."""
    if not d:
        return {}
    # d is {seed: {metric: val, ...}, ...}
    keys = set()
    for v in d.values():
        if isinstance(v, dict):
            keys.update(v.keys())
    result = {}
    for k in keys:
        vals = [v.get(k, 0) for v in d.values() if isinstance(v, dict)]
        vals = [v for v in vals if isinstance(v, (int, float))]
        result[k] = float(np.mean(vals)) if vals else 0
    return result


# ── Routes ───────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    """Main dashboard — runs a live experiment and renders results."""
    print("🔬 Running live experiment for dashboard...")
    results = run_live_experiment()
    html = build_dashboard_html(results)
    return HTMLResponse(content=html)


@app.get("/api/health")
async def health():
    return {"status": "ok", "message": "Dashboard running"}


# ── Entry Point ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Launch thesis results dashboard")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--results-dir", default=None)
    args = parser.parse_args()

    global RESULTS_DIR
    if args.results_dir:
        RESULTS_DIR = Path(args.results_dir)

    print()
    print("=" * 56)
    print("  🏥 Medical Imaging RL — Thesis Dashboard")
    print("=" * 56)
    print(f"  Open in your browser:")
    print(f"  → http://{args.host}:{args.port}")
    print("=" * 56)
    print()

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
