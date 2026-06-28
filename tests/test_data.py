"""Tests for data loading and validation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from energy_forecasting.data import load_energy_data, standardize_columns


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        load_energy_data(tmp_path / "missing.csv")


def test_duplicate_timestamps_removed(tmp_path: Path) -> None:
    timestamps = pd.date_range("2023-01-01", periods=250, freq="h")
    df = pd.DataFrame({"timestamp": timestamps, "price": range(250)})
    dup = pd.concat([df, df.iloc[[10, 20]]], ignore_index=True)
    path = tmp_path / "dup.csv"
    dup.to_csv(path, index=False)
    loaded, report = load_energy_data(path)
    assert report.n_duplicates_removed == 2
    assert loaded["timestamp"].is_unique


def test_invalid_price_raises(tmp_path: Path) -> None:
    timestamps = pd.date_range("2023-01-01", periods=250, freq="h")
    df = pd.DataFrame({"timestamp": timestamps, "price": ["bad"] * 250})
    path = tmp_path / "bad_price.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="numeric"):
        load_energy_data(path)


def test_insufficient_rows_raises(tmp_path: Path) -> None:
    timestamps = pd.date_range("2023-01-01", periods=50, freq="h")
    df = pd.DataFrame({"timestamp": timestamps, "price": range(50)})
    path = tmp_path / "short.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="at least"):
        load_energy_data(path)


def test_column_mapping(tmp_path: Path) -> None:
    timestamps = pd.date_range("2023-01-01", periods=250, freq="h")
    df = pd.DataFrame(
        {
            "ts": timestamps,
            "spot_price": range(250),
            "temp_c": range(250),
        }
    )
    path = tmp_path / "mapped.csv"
    df.to_csv(path, index=False)
    loaded, _ = load_energy_data(
        path,
        column_mapping={"timestamp": "ts", "price": "spot_price", "temperature": "temp_c"},
    )
    assert "timestamp" in loaded.columns
    assert "price" in loaded.columns
    assert "temperature" in loaded.columns
