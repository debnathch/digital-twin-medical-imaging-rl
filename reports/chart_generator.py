"""Chart Generator — Publication-quality Plotly charts for thesis."""

import numpy as np
from pathlib import Path

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


class ChartGenerator:
    def __init__(self, output_dir: str = "results/plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.colors = {"FIFO": "#2196F3", "STATIC_TRIAGE": "#FF9800", "HAPPO": "#4CAF50"}

    def generate_training_curves(self, training_history: list, convergence_episode: int = None) -> str:
        if not HAS_PLOTLY or not training_history:
            return ""
        episodes = [h.get("episode", i) for i, h in enumerate(training_history)]
        rewards = [h.get("reward", 0) for h in training_history]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=episodes, y=rewards, mode="lines", name="Reward", opacity=0.4))
        # Moving average
        window = min(20, len(rewards))
        if window > 1:
            ma = np.convolve(rewards, np.ones(window)/window, mode="valid")
            fig.add_trace(go.Scatter(x=episodes[window-1:], y=ma, mode="lines", name="Moving Avg", line=dict(width=3)))
        if convergence_episode:
            fig.add_vline(x=convergence_episode, line_dash="dash", line_color="red", annotation_text="Convergence")
        fig.update_layout(title="Training Curves", xaxis_title="Episode", yaxis_title="Global Reward")
        path = str(self.output_dir / "training_curves")
        fig.write_html(f"{path}.html")
        try:
            fig.write_image(f"{path}.png", width=1200, height=600)
        except Exception:
            pass
        return path

    def generate_policy_comparison_bars(self, metrics: dict) -> str:
        if not HAS_PLOTLY:
            return ""
        fig = go.Figure()
        policies = list(metrics.keys())
        metric_names = ["mean_turnaround_time", "sla_compliance", "mean_queue_waiting_time"]
        for mn in metric_names:
            vals = [metrics[p].get(mn, 0) for p in policies]
            fig.add_trace(go.Bar(name=mn, x=policies, y=vals))
        fig.update_layout(title="Policy Comparison", barmode="group")
        path = str(self.output_dir / "policy_comparison")
        fig.write_html(f"{path}.html")
        return path

    def generate_all(self, results: dict) -> list:
        paths = []
        if "training_history" in results:
            p = self.generate_training_curves(results["training_history"])
            if p:
                paths.append(p)
        return paths
