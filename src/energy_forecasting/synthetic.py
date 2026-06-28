"""Reproducible synthetic hourly electricity price data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from energy_forecasting.config import DRIFT_BREAK_FRACTION, RANDOM_STATE


def generate_synthetic_data(
    n_hours: int = 17520,
    start: str = "2022-01-01 00:00:00",
    output_path: Path | None = None,
    structural_break_fraction: float = DRIFT_BREAK_FRACTION,
    seed: int = RANDOM_STATE,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range(start=start, periods=n_hours, freq="h")

    hour = timestamps.hour.values
    dow = timestamps.dayofweek.values
    doy = timestamps.dayofyear.values

    load_base = 45000 + 8000 * np.sin(2 * np.pi * doy / 365)
    load_daily = 6000 * np.sin(2 * np.pi * (hour - 8) / 24)
    load_weekly = 2000 * (dow >= 5).astype(float)
    load_noise = rng.normal(0, 800, n_hours)
    load = load_base + load_daily + load_weekly + load_noise
    load = np.clip(load, 25000, None)

    temperature = (
        12
        + 10 * np.sin(2 * np.pi * (doy - 80) / 365)
        + 4 * np.sin(2 * np.pi * hour / 24)
        + rng.normal(0, 2, n_hours)
    )

    daily_season = 15 * np.sin(2 * np.pi * (hour - 7) / 24)
    weekly_season = 8 * (dow >= 5).astype(float)
    annual_season = 12 * np.sin(2 * np.pi * doy / 365)
    load_effect = 0.0004 * (load - load.mean())
    temp_effect = 0.3 * (temperature - 18) ** 2 / 100
    noise = rng.normal(0, 3, n_hours)

    price = 45 + daily_season + weekly_season + annual_season + load_effect + temp_effect + noise

    spike_mask = rng.random(n_hours) < 0.008
    price[spike_mask] += rng.uniform(30, 80, spike_mask.sum())

    break_idx = int(n_hours * structural_break_fraction)
    price[break_idx:] *= 1.25
    price[break_idx:] += 8
    load[break_idx:] *= 1.08
    temperature[break_idx:] += 2.5

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "price": np.round(price, 2),
            "temperature": np.round(temperature, 2),
            "load": np.round(load, 1),
        }
    )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

    return df
