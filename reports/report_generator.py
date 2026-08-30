"""Report Generator — HTML/Markdown thesis-ready reports."""

from pathlib import Path
from datetime import datetime


class ReportGenerator:
    def __init__(self, output_dir: str = "results/reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, config, ppo_results, fifo_results, triage_results,
                 ppo_eval_results=None, statistical_results=None,
                 ablation_results=None, acceptance_result=None) -> str:
        md = self._build_markdown(config, ppo_results, fifo_results, triage_results)
        md_path = self.output_dir / "report.md"
        md_path.write_text(md)
        html_path = self.output_dir / "report.html"
        html_path.write_text(f"<html><body><pre>{md}</pre></body></html>")
        return str(md_path)

    def _build_markdown(self, config, ppo, fifo, triage) -> str:
        return f\"\"\"# Digital Twin-Driven RL — Experiment Report

**Generated**: {datetime.now().isoformat()}
**SYNTHETIC / RESEARCH DATA ONLY**

## Executive Summary

This report summarises the experimental comparison of three workflow
scheduling policies: FIFO, Static Triage Priority, and Hierarchical
PPO/HAPPO trained via a Digital Twin simulation.

## Configuration

- Seeds: {config.get('experiment', {}).get('seeds', [])}
- Training episodes: {config.get('training', {}).get('num_episodes', 'N/A')}
- Scenarios: {config.get('scenarios', {})}

## Results

Results are stored in the `results/` directory.

## Limitations

1. All experiments use synthetic workload data.
2. The simulation abstracts complex hospital workflows.
3. Results may not generalise to different hospital configurations.
\"\"\"
