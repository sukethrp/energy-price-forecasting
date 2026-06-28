"""Leakage-safe time-series feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd

from energy_forecasting.config import COOLING_BASE_TEMP, HEATING_BASE_TEMP


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    ts = result["timestamp"]

    result["hour"] = ts.dt.hour
    result["day_of_week"] = ts.dt.dayofweek
    result["month"] = ts.dt.month
    result["day_of_year"] = ts.dt.dayofyear
    result["is_weekend"] = (result["day_of_week"] >= 5).astype(int)

    result["hour_sin"] = np.sin(2 * np.pi * result["hour"] / 24)
    result["hour_cos"] = np.cos(2 * np.pi * result["hour"] / 24)
    result["day_of_week_sin"] = np.sin(2 * np.pi * result["day_of_week"] / 7)
    result["day_of_week_cos"] = np.cos(2 * np.pi * result["day_of_week"] / 7)
    result["month_sin"] = np.sin(2 * np.pi * result["month"] / 12)
    result["month_cos"] = np.cos(2 * np.pi * result["month"] / 12)

    price = result["price"]
    for lag in [1, 2, 24, 48, 168]:
        result[f"price_lag_{lag}"] = price.shift(lag)

    shifted_price = price.shift(1)
    for window in [24, 48, 168]:
        result[f"price_roll_mean_{window}"] = shifted_price.rolling(
            window=window, min_periods=window
        ).mean()
    for window in [24, 168]:
        result[f"price_roll_std_{window}"] = shifted_price.rolling(
            window=window, min_periods=window
        ).std()
    result["price_roll_min_24"] = shifted_price.rolling(
        window=24, min_periods=24
    ).min()
    result["price_roll_max_24"] = shifted_price.rolling(
        window=24, min_periods=24
    ).max()

    if "load" in result.columns:
        load = result["load"]
        result["load_lag_1"] = load.shift(1)
        result["load_lag_24"] = load.shift(24)
        result["load_roll_mean_24"] = load.shift(1).rolling(
            window=24, min_periods=24
        ).mean()

    if "temperature" in result.columns:
        temp = result["temperature"]
        result["temperature_sq"] = temp ** 2
        result["heating_degree"] = np.maximum(HEATING_BASE_TEMP - temp, 0)
        result["cooling_degree"] = np.maximum(temp - COOLING_BASE_TEMP, 0)
        result["temperature_lag_24"] = temp.shift(24)

    return result


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = {"timestamp", "price"}
    return [c for c in df.columns if c not in exclude]


def prepare_xy(
    df: pd.DataFrame, feature_columns: list[str] | None = None
) -> tuple[pd.DataFrame, pd.Series]:
    features = feature_columns or get_feature_columns(df)
    xy = df[features + ["price"]].dropna()
    x = xy[features]
    y = xy["price"]
    return x, y
