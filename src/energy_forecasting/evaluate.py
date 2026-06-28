"""Model evaluation metrics and reporting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def safe_mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1.0) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), epsilon)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)


def smape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true) + np.abs(y_pred), epsilon)
    return float(np.mean(2 * np.abs(y_true - y_pred) / denom) * 100)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    errors = y_true - y_pred
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "mape": safe_mape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
    }


def metrics_by_hour(
    timestamps: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[int, dict[str, float]]:
    df = pd.DataFrame(
        {
            "hour": pd.to_datetime(timestamps).dt.hour,
            "y_true": y_true,
            "y_pred": y_pred,
        }
    )
    result: dict[int, dict[str, float]] = {}
    for hour, group in df.groupby("hour"):
        result[int(hour)] = compute_metrics(
            group["y_true"].values, group["y_pred"].values
        )
    return result


def metrics_top_price_decile(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    mask = y_true >= threshold
    if mask.sum() == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "mape": float("nan"), "smape": float("nan")}
    return compute_metrics(y_true[mask], y_pred[mask])


def metrics_spike_period(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    spike_threshold: float,
) -> dict[str, float]:
    mask = y_true >= spike_threshold
    if mask.sum() == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "mape": float("nan"), "smape": float("nan")}
    return compute_metrics(y_true[mask], y_pred[mask])


def build_full_metrics_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    timestamps: pd.Series,
    spike_threshold: float,
    top_decile_threshold: float,
    split_name: str,
    model_name: str,
) -> dict[str, Any]:
    overall = compute_metrics(y_true, y_pred)
    by_hour = metrics_by_hour(timestamps, y_true, y_pred)
    return {
        "split": split_name,
        "model": model_name,
        "overall": overall,
        "by_hour": {str(k): v for k, v in by_hour.items()},
        "top_10_percent_prices": metrics_top_price_decile(
            y_true, y_pred, top_decile_threshold
        ),
        "spike_period": metrics_spike_period(y_true, y_pred, spike_threshold),
        "spike_threshold": spike_threshold,
        "top_decile_threshold": top_decile_threshold,
        "n_samples": int(len(y_true)),
    }


def save_metrics_report(
    full_report: dict[str, Any],
    flat_entries: dict[str, Any],
    json_path: Path,
    csv_path: Path,
) -> None:
    """Persist the structured report to JSON and a flat model-comparison CSV."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(full_report, f, indent=2)

    rows = []
    for key, value in flat_entries.items():
        if not isinstance(value, dict) or "overall" not in value:
            continue
        row = {
            "model_comparison_key": key,
            "split": value.get("split", ""),
            "model": value.get("model", ""),
            "n_samples": value.get("n_samples", ""),
            **{f"overall_{k}": v for k, v in value["overall"].items()},
        }
        if "top_10_percent_prices" in value:
            for k, v in value["top_10_percent_prices"].items():
                row[f"top10_{k}"] = v
        if "spike_period" in value:
            for k, v in value["spike_period"].items():
                row[f"spike_{k}"] = v
        rows.append(row)

    pd.DataFrame(rows).to_csv(csv_path, index=False)


def format_metric(value: float, decimals: int = 4) -> str:
    """Format a numeric metric for display."""
    if value != value:  # NaN
        return "nan"
    return f"{value:.{decimals}f}"
