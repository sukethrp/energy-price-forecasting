"""Model training and baseline definitions."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from energy_forecasting.config import RANDOM_STATE, TS_CV_SPLITS
from energy_forecasting.evaluate import compute_metrics


class NaivePersistenceModel:
    """Forecast previous hour's price."""

    def fit(self, x: pd.DataFrame, y: pd.Series) -> NaivePersistenceModel:
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        if "price_lag_1" not in x.columns:
            raise ValueError("price_lag_1 required for naive persistence")
        return x["price_lag_1"].values


class SeasonalNaiveModel:
    """Forecast price from 24 hours earlier."""

    def fit(self, x: pd.DataFrame, y: pd.Series) -> SeasonalNaiveModel:
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        if "price_lag_24" not in x.columns:
            raise ValueError("price_lag_24 required for seasonal naive")
        return x["price_lag_24"].values


def build_model_registry() -> dict[str, Any]:
    registry: dict[str, Any] = {
        "naive_persistence": NaivePersistenceModel(),
        "seasonal_naive": SeasonalNaiveModel(),
        "ridge": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0, random_state=RANDOM_STATE)),
            ]
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.1,
            max_depth=6,
            random_state=RANDOM_STATE,
        ),
    }
    try:
        from xgboost import XGBRegressor

        registry["xgboost"] = XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    except ImportError:
        pass
    return registry


def train_all_models(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
) -> tuple[dict[str, Any], dict[str, dict[str, float]], str]:
    registry = build_model_registry()
    val_metrics: dict[str, dict[str, float]] = {}
    fitted: dict[str, Any] = {}

    for name, model in registry.items():
        model.fit(x_train, y_train)
        preds = model.predict(x_val)
        val_metrics[name] = compute_metrics(y_val.values, preds)
        fitted[name] = model

    best_name = min(val_metrics, key=lambda k: val_metrics[k]["mae"])
    return fitted, val_metrics, best_name


def cross_validate_model(
    model: Any,
    x: pd.DataFrame,
    y: pd.Series,
    n_splits: int = TS_CV_SPLITS,
) -> list[dict[str, float]]:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics: list[dict[str, float]] = []
    x_arr = x.values if hasattr(x, "values") else x
    y_arr = y.values if hasattr(y, "values") else y

    for train_idx, val_idx in tscv.split(x_arr):
        x_tr, x_va = x.iloc[train_idx], x.iloc[val_idx]
        y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]
        clone = _clone_model(model)
        clone.fit(x_tr, y_tr)
        preds = clone.predict(x_va)
        fold_metrics.append(compute_metrics(y_va.values, preds))
    return fold_metrics


def _clone_model(model: Any) -> Any:
    from sklearn.base import clone

    try:
        return clone(model)
    except Exception:
        if isinstance(model, (NaivePersistenceModel, SeasonalNaiveModel)):
            return type(model)()
        raise
