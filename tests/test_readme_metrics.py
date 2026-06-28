"""Ensure README metrics match saved pipeline output."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from energy_forecasting.config import METRICS_DIR, PROJECT_ROOT
from energy_forecasting.evaluate import format_metric

METRICS_JSON = METRICS_DIR / "model_metrics.json"
README_PATH = PROJECT_ROOT / "README.md"


@pytest.mark.skipif(not METRICS_JSON.exists(), reason="Run training pipeline first")
def test_readme_contains_saved_validation_metrics() -> None:
    metrics = json.loads(METRICS_JSON.read_text())
    readme = README_PATH.read_text()
    for _name, entry in metrics["validation_comparison"].items():
        overall = entry["overall"]
        assert format_metric(overall["mae"]) in readme
        assert format_metric(overall["rmse"]) in readme


@pytest.mark.skipif(not METRICS_JSON.exists(), reason="Run training pipeline first")
def test_readme_contains_saved_test_metrics() -> None:
    metrics = json.loads(METRICS_JSON.read_text())
    readme = README_PATH.read_text()
    test = metrics["test_best"]["overall"]
    for key in ("mae", "rmse", "mape", "smape"):
        assert format_metric(test[key]) in readme


@pytest.mark.skipif(not METRICS_JSON.exists(), reason="Run training pipeline first")
def test_readme_states_synthetic_dataset() -> None:
    readme = README_PATH.read_text()
    section = re.search(r"## Model results(.*?)## Limitations", readme, re.DOTALL)
    assert section is not None
    assert "synthetic" in section.group(1).lower()
