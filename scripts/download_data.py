"""Download publicly available energy price data when a URL is provided."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import requests

from energy_forecasting.config import DATA_RAW_DIR, DEFAULT_RAW_DATA
from energy_forecasting.data import load_energy_data, standardize_columns


def download_csv(url: str, output_path: Path, column_mapping: dict[str, str] | None = None) -> Path:
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(response.content)

    df = pd.read_csv(output_path)
    df = standardize_columns(df, column_mapping)
    df.to_csv(output_path, index=False)
    load_energy_data(output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download energy price CSV from a public URL")
    parser.add_argument("url", help="Public CSV URL")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RAW_DATA,
        help="Output path for downloaded data",
    )
    args = parser.parse_args()
    path = download_csv(args.url, args.output)
    print(f"Downloaded and validated data -> {path}")


if __name__ == "__main__":
    main()
