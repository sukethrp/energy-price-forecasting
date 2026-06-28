#!/usr/bin/env python3
"""Run the full training and evaluation workflow."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from energy_forecasting.config import DEFAULT_RAW_DATA
from energy_forecasting.pipeline import run_training_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Train energy price forecasting models")
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_RAW_DATA,
        help="Path to input CSV (synthetic data generated if missing)",
    )
    args = parser.parse_args()

    reports = run_training_pipeline(data_path=args.data)
    test = reports["test_best"]["overall"]
    print(f"Best model: {reports['best_model']}")
    print(f"Validation MAE (best): {reports['validation_best']['overall']['mae']:.4f}")
    print(f"Test MAE:  {test['mae']:.4f}")
    print(f"Test RMSE: {test['rmse']:.4f}")
    print(f"Test MAPE: {test['mape']:.4f}%")
    print(f"Test sMAPE: {test['smape']:.4f}%")


if __name__ == "__main__":
    main()
