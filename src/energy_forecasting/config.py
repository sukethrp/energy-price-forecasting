"""Project configuration and paths."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
METRICS_DIR = PROJECT_ROOT / "reports" / "metrics"

DEFAULT_RAW_DATA = DATA_RAW_DIR / "energy_prices.csv"
DEFAULT_PROCESSED_DATA = DATA_PROCESSED_DIR / "energy_prices_processed.csv"
DEFAULT_MODEL_PATH = MODELS_DIR / "best_model.joblib"
DEFAULT_PREDICTIONS_PATH = DATA_PROCESSED_DIR / "test_predictions.csv"

CANONICAL_COLUMNS = {
    "timestamp": "timestamp",
    "price": "price",
    "temperature": "temperature",
    "load": "load",
}

TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15

MIN_OBSERVATIONS = 200
MAX_LAG = 168

SPIKE_PERCENTILE = 90
PSI_THRESHOLD = 0.1
PSI_SEVERE_THRESHOLD = 0.25
KS_ALPHA = 0.05
MAE_DRIFT_THRESHOLD = 1.15

RANDOM_STATE = 42

TS_CV_SPLITS = 5

HEATING_BASE_TEMP = 18.0
COOLING_BASE_TEMP = 22.0

DRIFT_BREAK_FRACTION = 0.85
REFERENCE_WINDOW_HOURS = 720
MONITORING_WINDOW_HOURS = 720
