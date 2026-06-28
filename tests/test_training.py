"""Tests for training integrity and preprocessing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from energy_forecasting.features import build_features, get_feature_columns, prepare_xy
from energy_forecasting.synthetic import generate_synthetic_data
from energy_forecasting.train import build_model_registry, train_all_models


@pytest.fixture
def split_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    df = generate_synthetic_data(n_hours=1200, output_path=None, seed=11)
    featured = build_features(df)
    feature_cols = get_feature_columns(featured)
    n = len(featured)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    x_train, y_train = prepare_xy(featured.iloc[:train_end], feature_cols)
    x_val, y_val = prepare_xy(featured.iloc[train_end:val_end], feature_cols)
    return x_train, y_train, x_val, y_val


def test_model_selection_uses_validation_mae_only(split_data) -> None:
    x_train, y_train, x_val, y_val = split_data
    _, val_metrics, best_name = train_all_models(x_train, y_train, x_val, y_val)
    expected = min(val_metrics, key=lambda k: val_metrics[k]["mae"])
    assert best_name == expected


def test_ridge_scaler_fit_on_training_data_only(split_data) -> None:
    x_train, y_train, x_val, _y_val = split_data
    registry = build_model_registry()
    ridge = registry["ridge"]
    ridge.fit(x_train, y_train)
    scaler = ridge.named_steps["scaler"]
    train_means = x_train.mean().values
    assert np.allclose(scaler.mean_, train_means, rtol=0.01)
    assert not np.allclose(scaler.mean_, x_val.mean().values, rtol=0.01)


def test_train_all_models_ignores_test_split() -> None:
    """Model selection is driven by validation metrics only."""
    from energy_forecasting.train import train_all_models

    assert "x_test" not in train_all_models.__code__.co_varnames
