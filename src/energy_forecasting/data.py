"""Data loading, validation, and temporal splitting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from energy_forecasting.config import (
    CANONICAL_COLUMNS,
    MAX_LAG,
    MIN_OBSERVATIONS,
    TEST_FRACTION,
    TRAIN_FRACTION,
    VAL_FRACTION,
)


@dataclass
class DataValidationReport:
    n_rows: int
    n_duplicates_removed: int
    missing_intervals: int
    expected_hourly_rows: int
    has_temperature: bool
    has_load: bool
    price_min: float
    price_max: float
    start_timestamp: pd.Timestamp
    end_timestamp: pd.Timestamp


def standardize_columns(
    df: pd.DataFrame, column_mapping: dict[str, str] | None = None
) -> pd.DataFrame:
    mapping = column_mapping or {}
    rename_map: dict[str, str] = {}
    for canonical, default_name in CANONICAL_COLUMNS.items():
        source = mapping.get(canonical, default_name)
        if source in df.columns:
            rename_map[source] = canonical
    result = df.rename(columns=rename_map)
    required = ["timestamp", "price"]
    missing = [col for col in required if col not in result.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return result


def load_energy_data(
    path: str | pd.PathLike,
    column_mapping: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, DataValidationReport]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path)
    df = standardize_columns(df, column_mapping)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False)
    df = df.sort_values("timestamp").reset_index(drop=True)

    n_before = len(df)
    df = df.drop_duplicates(subset=["timestamp"], keep="first")
    n_duplicates_removed = n_before - len(df)

    if not pd.api.types.is_numeric_dtype(df["price"]):
        raise ValueError("Price column must be numeric")

    if df["price"].isna().all():
        raise ValueError("Price column contains no valid values")

    if len(df) < MIN_OBSERVATIONS:
        raise ValueError(
            f"Dataset has {len(df)} rows; at least {MIN_OBSERVATIONS} required "
            f"for lag features up to {MAX_LAG} hours"
        )

    start_ts = df["timestamp"].min()
    end_ts = df["timestamp"].max()
    expected_hours = int((end_ts - start_ts).total_seconds() // 3600) + 1
    missing_intervals = max(0, expected_hours - len(df))

    df = _forward_fill_numerics(df)

    report = DataValidationReport(
        n_rows=len(df),
        n_duplicates_removed=n_duplicates_removed,
        missing_intervals=missing_intervals,
        expected_hourly_rows=expected_hours,
        has_temperature="temperature" in df.columns,
        has_load="load" in df.columns,
        price_min=float(df["price"].min()),
        price_max=float(df["price"].max()),
        start_timestamp=start_ts,
        end_timestamp=end_ts,
    )
    return df, report


def _forward_fill_numerics(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    numeric_cols = [c for c in result.columns if c != "timestamp"]
    for col in numeric_cols:
        result[col] = pd.to_numeric(result[col], errors="coerce")
        result[col] = result[col].ffill()
    return result


def temporal_split(
    df: pd.DataFrame,
    train_fraction: float = TRAIN_FRACTION,
    val_fraction: float = VAL_FRACTION,
    test_fraction: float = TEST_FRACTION,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    total = train_fraction + val_fraction + test_fraction
    if not np.isclose(total, 1.0):
        raise ValueError("Split fractions must sum to 1.0")

    n = len(df)
    train_end = int(n * train_fraction)
    val_end = int(n * (train_fraction + val_fraction))

    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()

    if len(train) == 0 or len(val) == 0 or len(test) == 0:
        raise ValueError("Temporal split produced an empty partition")

    return train, val, test


def split_indices(
    n: int,
    train_fraction: float = TRAIN_FRACTION,
    val_fraction: float = VAL_FRACTION,
    test_fraction: float = TEST_FRACTION,
) -> dict[str, slice]:
    train_end = int(n * train_fraction)
    val_end = int(n * (train_fraction + val_fraction))
    return {
        "train": slice(0, train_end),
        "validation": slice(train_end, val_end),
        "test": slice(val_end, n),
    }


def get_spike_threshold(prices: pd.Series, percentile: float = 90.0) -> float:
    return float(np.percentile(prices.dropna(), percentile))
