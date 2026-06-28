#!/usr/bin/env python3
"""Update README model results from saved metric files."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from energy_forecasting.config import METRICS_DIR
from energy_forecasting.evaluate import format_metric

README_PATH = PROJECT_ROOT / "README.md"
METRICS_JSON = METRICS_DIR / "model_metrics.json"
DRIFT_JSON = METRICS_DIR / "drift_report.json"

MODEL_LABELS = {
    "ridge": "Ridge",
    "hist_gradient_boosting": "HistGradientBoosting",
    "naive_persistence": "Naive persistence",
    "seasonal_naive": "Seasonal naive",
}


def _pct(value: float) -> str:
    return f"{format_metric(value)}%"


def build_results_section(metrics: dict, drift: dict | None) -> str:
    best = metrics["best_model"]
    best_label = MODEL_LABELS.get(best, best)
    val = metrics["validation_comparison"]
    test = metrics["test_best"]["overall"]

    lines = [
        "## Model results",
        "",
        "All metrics below are copied from `reports/metrics/model_metrics.json` "
        "after running the training pipeline on the default **synthetic** dataset.",
        "",
        f"Selected model: **{best_label}** (`{best}`, lowest validation MAE)",
        "",
        "### Validation metrics (all models)",
        "",
        "| Model | MAE | RMSE | MAPE | sMAPE |",
        "|-------|-----|------|------|-------|",
    ]

    for name, entry in sorted(val.items(), key=lambda x: x[1]["overall"]["mae"]):
        o = entry["overall"]
        label = MODEL_LABELS.get(name, name)
        lines.append(
            f"| {label} | {format_metric(o['mae'])} | {format_metric(o['rmse'])} | "
            f"{_pct(o['mape'])} | {_pct(o['smape'])} |"
        )

    lines.extend(
        [
            "",
            f"### Test metrics ({best_label}, held-out 15%; not used for model selection)",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| MAE | {format_metric(test['mae'])} |",
            f"| RMSE | {format_metric(test['rmse'])} |",
            f"| MAPE | {_pct(test['mape'])} |",
            f"| sMAPE | {_pct(test['smape'])} |",
            f"| Top-10% price MAE | {format_metric(metrics['test_best']['top_10_percent_prices']['mae'])} |",
            f"| Spike-period MAE (>= {format_metric(metrics['spike_threshold'], 2)}) | "
            f"{format_metric(metrics['test_best']['spike_period']['mae'])} |",
            "",
            "Test-set degradation relative to validation reflects the structural break "
            "injected in the synthetic data after the 85th percentile of the timeline.",
            "",
        ]
    )

    cv = metrics["time_series_cv"]
    lines.append(
        f"Expanding-window time-series CV ({best_label}): mean MAE "
        f"{format_metric(cv['mean_mae'])}, mean RMSE {format_metric(cv['mean_rmse'])} "
        f"across {len(cv['folds'])} folds."
    )

    if drift:
        summary = drift["summary"]
        lines.append(
            f"\nDrift analysis flagged {summary['n_features_flagged']} of "
            f"{summary['n_features_analyzed']} features during the post-break monitoring window."
        )
        if "mae_drift" in drift:
            md = drift["mae_drift"]
            lines.append(
                f"Monitoring-window MAE: {format_metric(md['current_mae'])} "
                f"(reference: {format_metric(md['reference_mae'])})."
            )

    return "\n".join(lines)


def sync_readme() -> None:
    if not METRICS_JSON.exists():
        raise FileNotFoundError(
            f"Metrics file not found: {METRICS_JSON}. Run scripts/run_training.py first."
        )

    metrics = json.loads(METRICS_JSON.read_text())
    drift = json.loads(DRIFT_JSON.read_text()) if DRIFT_JSON.exists() else None
    new_section = build_results_section(metrics, drift)

    readme = README_PATH.read_text()
    pattern = r"## Model results\n.*?(\n## Limitations\n)"
    match = re.search(pattern, readme, flags=re.DOTALL)
    if not match:
        raise ValueError("Could not locate '## Model results' section in README.md")

    updated = readme[: match.start()] + new_section + "\n" + match.group(1) + readme[match.end() :]
    README_PATH.write_text(updated)
    print(f"Updated {README_PATH}")


if __name__ == "__main__":
    sync_readme()
