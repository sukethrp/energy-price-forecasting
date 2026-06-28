"""End-to-end training pipeline and visualization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from energy_forecasting.config import (
    DEFAULT_MODEL_PATH,
    DEFAULT_PREDICTIONS_PATH,
    DEFAULT_PROCESSED_DATA,
    DEFAULT_RAW_DATA,
    FIGURES_DIR,
    METRICS_DIR,
    PSI_THRESHOLD,
    SPIKE_PERCENTILE,
)
from energy_forecasting.data import (
    get_spike_threshold,
    load_energy_data,
    temporal_split,
)
from energy_forecasting.evaluate import build_full_metrics_report, save_metrics_report
from energy_forecasting.features import build_features, get_feature_columns, prepare_xy
from energy_forecasting.train import cross_validate_model, train_all_models


def ensure_data(raw_path: Path = DEFAULT_RAW_DATA) -> Path:
    """Return the raw data path, generating synthetic data when the file is absent."""
    if raw_path.exists():
        return raw_path
    from energy_forecasting.synthetic import generate_synthetic_data

    generate_synthetic_data(output_path=raw_path)
    return raw_path


def run_training_pipeline(
    data_path: Path | None = None,
    model_path: Path = DEFAULT_MODEL_PATH,
    predictions_path: Path = DEFAULT_PREDICTIONS_PATH,
    processed_path: Path = DEFAULT_PROCESSED_DATA,
) -> dict[str, Any]:
    """Execute load, feature engineering, training, evaluation, and artifact export."""
    raw_path = data_path if data_path and Path(data_path).exists() else ensure_data(
        data_path or DEFAULT_RAW_DATA
    )
    df, validation_report = load_energy_data(raw_path)
    featured = build_features(df)
    featured.to_csv(processed_path, index=False)

    train_df, val_df, test_df = temporal_split(featured)
    feature_cols = get_feature_columns(featured)

    x_train, y_train = prepare_xy(train_df, feature_cols)
    x_val, y_val = prepare_xy(val_df, feature_cols)
    x_test, y_test = prepare_xy(test_df, feature_cols)

    fitted, val_metrics, best_name = train_all_models(x_train, y_train, x_val, y_val)
    best_model = fitted[best_name]

    test_preds = best_model.predict(x_test)
    test_timestamps = test_df.loc[x_test.index, "timestamp"]

    train_prices = train_df["price"]
    spike_threshold = get_spike_threshold(train_prices, SPIKE_PERCENTILE)
    top_decile_threshold = float(np.percentile(train_prices.dropna(), 90))

    test_report = build_full_metrics_report(
        y_test.values,
        test_preds,
        test_timestamps,
        spike_threshold,
        top_decile_threshold,
        "test",
        best_name,
    )
    val_report_best = build_full_metrics_report(
        y_val.values,
        best_model.predict(x_val),
        val_df.loc[x_val.index, "timestamp"],
        spike_threshold,
        top_decile_threshold,
        "validation",
        best_name,
    )

    test_comparison: dict[str, Any] = {}
    for name, model in fitted.items():
        preds = model.predict(x_test)
        test_comparison[name] = {
            "split": "test",
            "model": name,
            "overall": build_full_metrics_report(
                y_test.values,
                preds,
                test_timestamps,
                spike_threshold,
                top_decile_threshold,
                "test",
                name,
            )["overall"],
        }

    cv_folds = cross_validate_model(fitted[best_name], x_train, y_train)

    data_validation = validation_report.__dict__.copy()
    for key, ts in data_validation.items():
        if isinstance(ts, pd.Timestamp):
            data_validation[key] = ts.isoformat()

    metrics_report: dict[str, Any] = {
        "dataset": "synthetic",
        "best_model": best_name,
        "data_validation": data_validation,
        "validation_comparison": {
            name: {"split": "validation", "model": name, "overall": metrics}
            for name, metrics in val_metrics.items()
        },
        "validation_best": val_report_best,
        "test_best": test_report,
        "test_comparison": test_comparison,
        "time_series_cv": {
            "model": best_name,
            "folds": cv_folds,
            "mean_mae": float(np.mean([f["mae"] for f in cv_folds])),
            "mean_rmse": float(np.mean([f["rmse"] for f in cv_folds])),
        },
        "spike_threshold": spike_threshold,
    }

    flat_for_csv = {
        **{f"validation_{k}": v for k, v in metrics_report["validation_comparison"].items()},
        "validation_best": val_report_best,
        "test_best": test_report,
        **{f"test_{k}": v for k, v in test_comparison.items()},
    }
    save_metrics_report(
        metrics_report,
        flat_for_csv,
        METRICS_DIR / "model_metrics.json",
        METRICS_DIR / "model_metrics.csv",
    )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": best_model,
        "model_name": best_name,
        "feature_columns": feature_cols,
        "spike_threshold": spike_threshold,
    }
    joblib.dump(artifact, model_path)

    pred_df = pd.DataFrame(
        {
            "timestamp": test_timestamps.values,
            "actual": y_test.values,
            "predicted": test_preds,
            "residual": y_test.values - test_preds,
        }
    )
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(predictions_path, index=False)

    generate_all_figures(
        pred_df,
        test_report,
        x_test,
        y_test.values,
        test_preds,
        best_model,
        feature_cols,
        val_metrics,
    )

    return metrics_report


def generate_all_figures(
    pred_df: pd.DataFrame,
    test_report: dict[str, Any],
    x_test: pd.DataFrame,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    model: Any,
    feature_columns: list[str],
    val_metrics: dict[str, dict[str, float]],
) -> None:
    """Write evaluation figures to the reports directory."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    _plot_full_test_period(pred_df, FIGURES_DIR / "01_actual_vs_predicted_full_test.png")
    _plot_seven_day_window(pred_df, FIGURES_DIR / "02_actual_vs_predicted_7day.png")
    _plot_residual_distribution(pred_df, FIGURES_DIR / "03_residual_distribution.png")
    _plot_error_by_hour(test_report, FIGURES_DIR / "04_error_by_hour.png")
    _plot_permutation_importance(
        model, x_test, y_test, feature_columns, FIGURES_DIR / "05_feature_importance.png"
    )
    _plot_rolling_mae(pred_df, FIGURES_DIR / "06_rolling_mae.png")
    _plot_model_comparison(val_metrics, FIGURES_DIR / "07_model_comparison_validation.png")


def _plot_full_test_period(pred_df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(pred_df["timestamp"], pred_df["actual"], label="Actual", alpha=0.8, linewidth=0.8)
    ax.plot(pred_df["timestamp"], pred_df["predicted"], label="Predicted", alpha=0.8, linewidth=0.8)
    ax.set_title("Actual vs Predicted Prices — Full Test Period")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Price")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_seven_day_window(pred_df: pd.DataFrame, path: Path) -> None:
    start_idx = max(0, len(pred_df) // 2 - 84)
    window = pred_df.iloc[start_idx : start_idx + 168]
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(window["timestamp"], window["actual"], label="Actual", marker=".", markersize=3)
    ax.plot(window["timestamp"], window["predicted"], label="Predicted", marker=".", markersize=3)
    ax.set_title("Actual vs Predicted Prices — Seven-Day Window")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Price")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_residual_distribution(pred_df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(pred_df["residual"], bins=50, edgecolor="black", alpha=0.7)
    ax.axvline(0, color="red", linestyle="--", label="Zero error")
    ax.set_title("Residual Distribution — Test Set")
    ax.set_xlabel("Residual (Actual − Predicted)")
    ax.set_ylabel("Frequency")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_error_by_hour(test_report: dict[str, Any], path: Path) -> None:
    by_hour = test_report.get("by_hour", {})
    hours = sorted(int(h) for h in by_hour.keys())
    maes = [by_hour[str(h)]["mae"] for h in hours]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(hours, maes, color="steelblue", edgecolor="black")
    ax.set_title("Mean Absolute Error by Hour of Day — Test Set")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("MAE")
    ax.set_xticks(hours)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_permutation_importance(
    model: Any,
    x_test: pd.DataFrame,
    y_test: np.ndarray,
    feature_columns: list[str],
    path: Path,
) -> None:
    if not hasattr(model, "predict"):
        return
    sample_size = min(500, len(x_test))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(x_test), size=sample_size, replace=False)
    x_sample = x_test.iloc[idx]
    y_sample = y_test[idx]

    result = permutation_importance(
        model, x_sample, y_sample, n_repeats=5, random_state=42, n_jobs=1
    )

    sorted_idx = result.importances_mean.argsort()[-15:]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        [feature_columns[i] for i in sorted_idx],
        result.importances_mean[sorted_idx],
        xerr=result.importances_std[sorted_idx],
        color="teal",
        edgecolor="black",
    )
    ax.set_title("Permutation Feature Importance — Test Sample")
    ax.set_xlabel("Mean Importance")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_rolling_mae(pred_df: pd.DataFrame, path: Path) -> None:
    errors = np.abs(pred_df["residual"].values)
    rolling = pd.Series(errors).rolling(window=168, min_periods=84).mean()
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(pred_df["timestamp"], rolling, color="darkorange", linewidth=1.2)
    ax.set_title("Rolling MAE (168-hour window) — Test Period")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Rolling MAE")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_model_comparison(val_metrics: dict[str, dict[str, float]], path: Path) -> None:
    models = list(val_metrics.keys())
    maes = [val_metrics[m]["mae"] for m in models]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(models, maes, color="slateblue", edgecolor="black")
    ax.set_title("Validation MAE by Model")
    ax.set_xlabel("Model")
    ax.set_ylabel("MAE")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_drift_report(drift_report: dict[str, Any], path: Path) -> None:
    """Render a PSI-based drift summary chart."""
    features = drift_report.get("feature_drift", [])
    if not features:
        return
    names = [f["feature"] for f in features]
    psi_vals = [f.get("psi", 0) for f in features]
    colors = ["crimson" if f.get("drift_flag") else "steelblue" for f in features]

    fig, ax = plt.subplots(figsize=(12, max(5, len(names) * 0.35)))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, psi_vals, color=colors, edgecolor="black")
    ax.axvline(
        PSI_THRESHOLD,
        color="orange",
        linestyle="--",
        label=f"PSI threshold ({PSI_THRESHOLD})",
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel("Population Stability Index (PSI)")
    ax.set_title("Feature Drift Report — PSI by Feature")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
