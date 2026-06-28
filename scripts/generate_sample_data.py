"""Generate reproducible synthetic electricity price data."""

from __future__ import annotations

import argparse
from pathlib import Path

from energy_forecasting.config import DEFAULT_RAW_DATA, RANDOM_STATE
from energy_forecasting.synthetic import generate_synthetic_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic energy price data")
    parser.add_argument("--output", type=Path, default=DEFAULT_RAW_DATA)
    parser.add_argument("--hours", type=int, default=17520)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    args = parser.parse_args()

    df = generate_synthetic_data(n_hours=args.hours, output_path=args.output, seed=args.seed)
    print(f"Generated {len(df)} rows -> {args.output}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print("Note: This is synthetic data for demonstration purposes.")


if __name__ == "__main__":
    main()
