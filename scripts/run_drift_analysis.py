#!/usr/bin/env python3
"""Run drift analysis on a structurally shifted monitoring period."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from energy_forecasting.config import (
    DEFAULT_MODEL_PATH,
    DEFAULT_RAW_DATA,
    FIGURES_DIR,
    METRICS_DIR,
    MONITORING_WINDOW_HOURS,
    REFERENCE_WINDOW_HOURS,
)
from energy_forecasting.data import load_energy_data
from energy_forecasting.drift import run_drift_analysis, save_drift_report
from energy_forecasting.features import build_features, prepare_xy
from energy_forecasting.pipeline import ensure_data, plot_drift_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run drift monitoring analysis")
    parser.add_argument("--data", type=Path, default=DEFAULT_RAW_DATA)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()

    raw_path = args.data if args.data.exists() else ensure_data(args.data)
    df, _ = load_energy_data(raw_path)
    featured = build_features(df)

    n = len(featured)
    ref_start = max(0, n - REFERENCE_WINDOW_HOURS - MONITORING_WINDOW_HOURS)
    ref_end = ref_start + REFERENCE_WINDOW_HOURS
    mon_start = ref_end
    mon_end = min(n, mon_start + MONITORING_WINDOW_HOURS)

    reference_df = featured.iloc[ref_start:ref_end].copy()
    monitoring_df = featured.iloc[mon_start:mon_end].copy()

    artifact_path = args.model
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {artifact_path}. Run scripts/run_training.py first."
        )
    artifact = joblib.load(artifact_path)
    model = artifact["model"]
    feature_cols = artifact["feature_columns"]

    x_ref, y_ref = prepare_xy(reference_df, feature_cols)
    x_mon, y_mon = prepare_xy(monitoring_df, feature_cols)

    ref_preds = model.predict(x_ref)
    mon_preds = model.predict(x_mon)

    report = run_drift_analysis(
        reference_df.loc[x_ref.index],
        monitoring_df.loc[x_mon.index],
        feature_cols,
        reference_preds=ref_preds,
        monitoring_preds=mon_preds,
        reference_labels=y_ref.values,
        monitoring_labels=y_mon.values,
    )

    drift_path = METRICS_DIR / "drift_report.json"
    save_drift_report(report, drift_path)
    plot_drift_report(report, FIGURES_DIR / "08_drift_report.png")

    flagged = report["summary"]["n_features_flagged"]
    total = report["summary"]["n_features_analyzed"]
    print(f"Drift analysis complete: {flagged}/{total} features flagged")
    if "mae_drift" in report:
        md = report["mae_drift"]
        print(f"Reference MAE: {md['reference_mae']:.4f}")
        print(f"Monitoring MAE: {md['current_mae']:.4f}")
        print(f"MAE drift flag: {md['drift_flag']}")
    print(f"Report saved -> {drift_path}")


if __name__ == "__main__":
    main()
