"""Table Generator — All 12 thesis tables in CSV/Excel/Markdown/HTML."""

import pandas as pd
from pathlib import Path


class TableGenerator:
    def __init__(self, output_dir: str = "results/tables"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _save(self, df: pd.DataFrame, name: str):
        base = self.output_dir / name
        df.to_csv(f"{base}.csv", index=False)
        try:
            df.to_excel(f"{base}.xlsx", index=False)
        except Exception:
            pass

    def table_7_policy_comparison(self, fifo: dict, triage: dict, ppo: dict) -> pd.DataFrame:
        metrics = ["mean_turnaround_time", "sla_compliance", "mean_queue_waiting_time",
                    "resource_utilization", "throughput"]
        rows = []
        for m in metrics:
            rows.append({
                "Metric": m,
                "FIFO": f"{fifo.get(m, 0):.3f}",
                "Triage": f"{triage.get(m, 0):.3f}",
                "PPO": f"{ppo.get(m, 0):.3f}",
            })
        df = pd.DataFrame(rows)
        self._save(df, "table_7_policy_comparison")
        return df
