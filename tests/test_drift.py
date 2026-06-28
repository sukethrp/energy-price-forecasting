"""Tests for drift detection utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from energy_forecasting.drift import (
    analyze_feature_drift,
    compute_psi,
    detect_mae_drift,
    run_drift_analysis,
)
from energy_forecasting.features import build_features, get_feature_columns
from energy_forecasting.synthetic import generate_synthetic_data


def test_psi_identical_distributions_near_zero() -> None:
    rng = np.random.default_rng(42)
    data = rng.normal(50, 5, 1000)
    psi = compute_psi(data, data.copy())
    assert psi < 0.01


def test_psi_detects_shift() -> None:
    rng = np.random.default_rng(42)
    ref = rng.normal(50, 5, 1000)
    cur = rng.normal(70, 5, 1000)
    psi = compute_psi(ref, cur)
    assert psi > 0.1


def test_feature_drift_report_structure() -> None:
    rng = np.random.default_rng(0)
    ref = pd.Series(rng.normal(50, 5, 500))
    cur = pd.Series(rng.normal(65, 8, 500))
    report = analyze_feature_drift("price_lag_1", ref, cur)
    assert report["feature"] == "price_lag_1"
    assert "psi" in report
    assert "ks_statistic" in report
    assert "drift_flag" in report
    assert "severity" in report


def test_drift_analysis_stable_on_same_data() -> None:
    df = generate_synthetic_data(n_hours=800, output_path=None, seed=7)
    featured = build_features(df)
    feature_cols = get_feature_columns(featured)
    half = len(featured) // 2
    ref = featured.iloc[:half]
    cur = featured.iloc[half : half + 200]

    report1 = run_drift_analysis(ref, cur, feature_cols)
    report2 = run_drift_analysis(ref, cur, feature_cols)
    assert report1["summary"] == report2["summary"]


def test_mae_drift_detection() -> None:
    result = detect_mae_drift(reference_mae=5.0, current_mae=6.5, threshold=1.15)
    assert result["drift_flag"] is True
    result_ok = detect_mae_drift(reference_mae=5.0, current_mae=5.2, threshold=1.15)
    assert result_ok["drift_flag"] is False


def test_drift_detects_structural_break() -> None:
    df = generate_synthetic_data(n_hours=2000, output_path=None, seed=99)
    featured = build_features(df)
    feature_cols = get_feature_columns(featured)
    ref = featured.iloc[500:1200]
    mon = featured.iloc[1600:1900]
    report = run_drift_analysis(ref, mon, feature_cols)
    flagged = sum(r["drift_flag"] for r in report["feature_drift"])
    assert flagged > 0
