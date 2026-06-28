"""Tests for chronological temporal splitting."""

from __future__ import annotations

import pandas as pd
import pytest

from energy_forecasting.data import split_indices, temporal_split
from energy_forecasting.synthetic import generate_synthetic_data


@pytest.fixture
def ordered_df() -> pd.DataFrame:
    df = generate_synthetic_data(n_hours=1000, output_path=None, seed=1)
    return df.sort_values("timestamp").reset_index(drop=True)


def test_temporal_order_preserved(ordered_df: pd.DataFrame) -> None:
    train, val, test = temporal_split(ordered_df)
    assert train["timestamp"].max() <= val["timestamp"].min()
    assert val["timestamp"].max() <= test["timestamp"].min()


def test_no_overlap_between_partitions(ordered_df: pd.DataFrame) -> None:
    train, val, test = temporal_split(ordered_df)
    train_idx = set(train.index)
    val_idx = set(val.index)
    test_idx = set(test.index)
    assert train_idx.isdisjoint(val_idx)
    assert val_idx.isdisjoint(test_idx)
    assert train_idx.isdisjoint(test_idx)


def test_split_covers_all_rows(ordered_df: pd.DataFrame) -> None:
    train, val, test = temporal_split(ordered_df)
    assert len(train) + len(val) + len(test) == len(ordered_df)


def test_split_fractions_approximate(ordered_df: pd.DataFrame) -> None:
    n = len(ordered_df)
    train, val, test = temporal_split(ordered_df)
    assert abs(len(train) / n - 0.70) < 0.02
    assert abs(len(val) / n - 0.15) < 0.02
    assert abs(len(test) / n - 0.15) < 0.02


def test_split_indices_boundaries() -> None:
    slices = split_indices(1000)
    assert slices["train"].stop == 700
    assert slices["validation"].start == 700
    assert slices["validation"].stop == 850
    assert slices["test"].start == 850


def test_invalid_fractions_raise() -> None:
    df = pd.DataFrame({"timestamp": pd.date_range("2023-01-01", periods=10, freq="h"), "price": range(10)})
    with pytest.raises(ValueError):
        temporal_split(df, train_fraction=0.5, val_fraction=0.3, test_fraction=0.1)
