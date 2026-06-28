"""Distribution shift and drift monitoring."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from energy_forecasting.config import (
    KS_ALPHA,
    MAE_DRIFT_THRESHOLD,
    PSI_SEVERE_THRESHOLD,
    PSI_THRESHOLD,
)


def compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
) -> float:
    ref = reference[~np.isnan(reference)]
    cur = current[~np.isnan(current)]
    if len(ref) == 0 or len(cur) == 0:
        return float("nan")

    breakpoints = np.percentile(ref, np.linspace(0, 100, n_bins + 1))
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) < 2:
        return 0.0

    ref_counts, _ = np.histogram(ref, bins=breakpoints)
    cur_counts, _ = np.histogram(cur, bins=breakpoints)

    ref_pct = ref_counts / max(len(ref), 1)
    cur_pct = cur_counts / max(len(cur), 1)
    epsilon = 1e-6
    ref_pct = np.clip(ref_pct, epsilon, None)
    cur_pct = np.clip(cur_pct, epsilon, None)
    ref_pct = ref_pct / ref_pct.sum()
    cur_pct = cur_pct / cur_pct.sum()

    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return psi


def _severity(psi: float, ks_p: float, mean_change_pct: float) -> str:
    flags = 0
    if not np.isnan(psi) and psi >= PSI_SEVERE_THRESHOLD:
        return "severe"
    if not np.isnan(psi) and psi >= PSI_THRESHOLD:
        flags += 1
    if not np.isnan(ks_p) and ks_p < KS_ALPHA:
        flags += 1
    if abs(mean_change_pct) > 20:
        flags += 1
    if flags >= 2:
        return "moderate"
    if flags == 1:
        return "low"
    return "none"


def _drift_flag(psi: float, ks_p: float, mean_change_pct: float, std_change_pct: float) -> bool:
    if not np.isnan(psi) and psi >= PSI_THRESHOLD:
        return True
    if not np.isnan(ks_p) and ks_p < KS_ALPHA:
        return True
    if abs(mean_change_pct) > 15 or abs(std_change_pct) > 25:
        return True
    return False


def analyze_feature_drift(
    feature: str,
    reference: pd.Series,
    current: pd.Series,
) -> dict[str, Any]:
    ref_vals = reference.dropna().values.astype(float)
    cur_vals = current.dropna().values.astype(float)

    ref_mean = float(np.mean(ref_vals)) if len(ref_vals) else float("nan")
    cur_mean = float(np.mean(cur_vals)) if len(cur_vals) else float("nan")
    ref_std = float(np.std(ref_vals)) if len(ref_vals) else float("nan")
    cur_std = float(np.std(cur_vals)) if len(cur_vals) else float("nan")

    mean_change_pct = (
        ((cur_mean - ref_mean) / abs(ref_mean) * 100) if ref_mean != 0 else float("nan")
    )
    std_change_pct = (
        ((cur_std - ref_std) / abs(ref_std) * 100) if ref_std != 0 else float("nan")
    )

    ref_missing = float(reference.isna().mean())
    cur_missing = float(current.isna().mean())
    missing_rate_change = cur_missing - ref_missing

    psi = compute_psi(ref_vals, cur_vals)
    ks_stat, ks_p = stats.ks_2samp(ref_vals, cur_vals) if len(ref_vals) and len(cur_vals) else (float("nan"), float("nan"))

    flagged = _drift_flag(psi, ks_p, mean_change_pct, std_change_pct)
    severity = _severity(psi, ks_p, mean_change_pct)

    return {
        "feature": feature,
        "reference_statistics": {
            "mean": ref_mean,
            "std": ref_std,
            "missing_rate": ref_missing,
            "n": int(len(ref_vals)),
        },
        "current_statistics": {
            "mean": cur_mean,
            "std": cur_std,
            "missing_rate": cur_missing,
            "n": int(len(cur_vals)),
        },
        "psi": psi,
        "ks_statistic": float(ks_stat),
        "ks_p_value": float(ks_p),
        "mean_change_pct": mean_change_pct,
        "std_change_pct": std_change_pct,
        "missing_rate_change": missing_rate_change,
        "drift_flag": flagged,
        "severity": severity,
    }


def analyze_prediction_drift(
    reference_preds: np.ndarray,
    current_preds: np.ndarray,
) -> dict[str, Any]:
    return analyze_feature_drift(
        "predictions",
        pd.Series(reference_preds),
        pd.Series(current_preds),
    )


def analyze_residual_drift(
    reference_residuals: np.ndarray,
    current_residuals: np.ndarray,
) -> dict[str, Any]:
    return analyze_feature_drift(
        "residuals",
        pd.Series(reference_residuals),
        pd.Series(current_residuals),
    )


def rolling_mae_monitoring(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    window: int = 168,
) -> pd.DataFrame:
    errors = np.abs(np.asarray(y_true) - np.asarray(y_pred))
    series = pd.Series(errors)
    rolling = series.rolling(window=window, min_periods=window // 2).mean()
    return pd.DataFrame({"absolute_error": errors, "rolling_mae": rolling})


def detect_mae_drift(
    reference_mae: float,
    current_mae: float,
    threshold: float = MAE_DRIFT_THRESHOLD,
) -> dict[str, Any]:
    ratio = current_mae / reference_mae if reference_mae > 0 else float("nan")
    flagged = ratio >= threshold if not np.isnan(ratio) else False
    return {
        "reference_mae": reference_mae,
        "current_mae": current_mae,
        "mae_ratio": ratio,
        "drift_flag": flagged,
        "threshold": threshold,
    }


def run_drift_analysis(
    reference_df: pd.DataFrame,
    monitoring_df: pd.DataFrame,
    feature_columns: list[str],
    reference_preds: np.ndarray | None = None,
    monitoring_preds: np.ndarray | None = None,
    reference_labels: np.ndarray | None = None,
    monitoring_labels: np.ndarray | None = None,
) -> dict[str, Any]:
    feature_reports = []
    for col in feature_columns:
        if col in reference_df.columns and col in monitoring_df.columns:
            feature_reports.append(
                analyze_feature_drift(col, reference_df[col], monitoring_df[col])
            )

    report: dict[str, Any] = {
        "feature_drift": feature_reports,
        "summary": {
            "n_features_flagged": sum(r["drift_flag"] for r in feature_reports),
            "n_features_analyzed": len(feature_reports),
        },
        "note": (
            "Drift signals indicate distributional change and should trigger "
            "investigation. They do not automatically require model retraining."
        ),
    }

    if reference_preds is not None and monitoring_preds is not None:
        report["prediction_drift"] = analyze_prediction_drift(
            reference_preds, monitoring_preds
        )

    if (
        reference_labels is not None
        and monitoring_labels is not None
        and reference_preds is not None
        and monitoring_preds is not None
    ):
        ref_residuals = reference_labels - reference_preds
        mon_residuals = monitoring_labels - monitoring_preds
        report["residual_drift"] = analyze_residual_drift(ref_residuals, mon_residuals)

        ref_mae = float(np.mean(np.abs(ref_residuals)))
        mon_mae = float(np.mean(np.abs(mon_residuals)))
        report["mae_drift"] = detect_mae_drift(ref_mae, mon_mae)

    return report


def save_drift_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)
