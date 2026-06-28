"""Tests for leakage-safe feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from energy_forecasting.features import build_features, get_feature_columns, prepare_xy
from energy_forecasting.synthetic import generate_synthetic_data


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return generate_synthetic_data(n_hours=500, output_path=None, seed=42)


def test_lag_features_use_past_only(sample_df: pd.DataFrame) -> None:
    featured = build_features(sample_df)
    for lag in [1, 2, 24, 48, 168]:
        col = f"price_lag_{lag}"
        assert col in featured.columns
        idx = 200
        expected = sample_df.loc[idx - lag, "price"]
        assert featured.loc[idx, col] == pytest.approx(expected)


def test_future_target_change_does_not_affect_earlier_features(sample_df: pd.DataFrame) -> None:
    featured_before = build_features(sample_df)
    modified = sample_df.copy()
    modified.loc[modified.index[-1], "price"] = 9999.0
    featured_after = build_features(modified)

    check_idx = modified.index[-50]
    cols = get_feature_columns(featured_before)
    before_row = featured_before.loc[check_idx, cols]
    after_row = featured_after.loc[check_idx, cols]
    pd.testing.assert_series_equal(before_row, after_row)


def test_rolling_features_shifted_before_window(sample_df: pd.DataFrame) -> None:
    featured = build_features(sample_df)
    idx = 200
    window = 24
    shifted = sample_df["price"].shift(1)
    expected_mean = shifted.iloc[idx - window + 1 : idx + 1].mean()
    assert featured.loc[idx, f"price_roll_mean_{window}"] == pytest.approx(expected_mean)


def test_rolling_std_no_future_leakage(sample_df: pd.DataFrame) -> None:
    featured = build_features(sample_df)
    modified = sample_df.copy()
    modified.loc[modified.index[250:], "price"] = 500.0
    featured_modified = build_features(modified)

    check_idx = 200
    assert featured.loc[check_idx, "price_roll_std_24"] == pytest.approx(
        featured_modified.loc[check_idx, "price_roll_std_24"]
    )


def test_optional_temperature_features(sample_df: pd.DataFrame) -> None:
    featured = build_features(sample_df)
    assert "temperature_sq" in featured.columns
    assert "heating_degree" in featured.columns
    assert "cooling_degree" in featured.columns
    assert "temperature_lag_24" in featured.columns


def test_optional_load_features(sample_df: pd.DataFrame) -> None:
    featured = build_features(sample_df)
    assert "load_lag_1" in featured.columns
    assert "load_lag_24" in featured.columns
    assert "load_roll_mean_24" in featured.columns


def test_missing_optional_columns() -> None:
    timestamps = pd.date_range("2023-01-01", periods=300, freq="h")
    df = pd.DataFrame({"timestamp": timestamps, "price": np.linspace(40, 60, 300)})
    featured = build_features(df)
    assert "load_lag_1" not in featured.columns
    assert "temperature_sq" not in featured.columns
    x, y = prepare_xy(featured)
    assert len(x) > 0


def test_prepare_xy_drops_incomplete_rows(sample_df: pd.DataFrame) -> None:
    featured = build_features(sample_df)
    x, y = prepare_xy(featured)
    assert x.isna().sum().sum() == 0
    assert y.isna().sum() == 0
