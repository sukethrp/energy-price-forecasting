# Energy Price Forecasting and Drift Monitoring

Hourly electricity price forecasting with leakage-safe feature engineering, temporal validation, baseline comparison, and distribution-shift monitoring.

## Business context

Electricity prices vary with demand, weather, fuel costs, and grid conditions. Accurate short-term forecasts support trading, scheduling, and risk management. This project demonstrates a reproducible forecasting workflow and monitoring layer for detecting when model inputs or errors shift over time.

## Forecasting approach

1. Load hourly records with `timestamp` and `price` (optional: `temperature`, `load`)
2. Engineer calendar, cyclical, lag, and rolling features using past information only
3. Split data chronologically into train (70%), validation (15%), and test (15%)
4. Compare naive baselines, Ridge regression, and HistGradientBoostingRegressor
5. Select the best model by validation MAE and evaluate once on the test set
6. Monitor drift between reference and monitoring windows

## Dataset

**Default: synthetic data.** Run `python scripts/generate_sample_data.py` to create two years of hourly synthetic prices with daily, weekly, and annual seasonality, load and temperature effects, random spikes, and a structural break for drift testing.

Place your own CSV in `data/raw/energy_prices.csv` with columns:

| Column | Required |
|--------|----------|
| timestamp | Yes |
| price | Yes |
| temperature | No |
| load | No |

Column names can be remapped via the loader's `column_mapping` argument.

## Feature categories

- Calendar: hour, day of week, month, day of year, weekend flag
- Cyclical: sin/cos encodings for hour, day of week, month
- Lags: price at 1, 2, 24, 48, 168 hours
- Rolling (shifted): mean (24/48/168h), std (24/168h), min/max (24h)
- Optional load/temperature-derived features when columns are present

## Models compared

| Model | Description |
|-------|-------------|
| Naive persistence | Previous hour's price |
| Seasonal naive | Price from 24 hours ago |
| Ridge | Linear model with StandardScaler |
| HistGradientBoostingRegressor | Tree-based gradient boosting |

## Temporal validation

- Chronological 70/15/15 split without shuffling
- Expanding-window `TimeSeriesSplit` for supplementary comparison
- Preprocessing fit on training data only
- Model selection on validation MAE; single test-set evaluation

## Evaluation metrics

MAE, RMSE, MAPE (with zero guard), sMAPE, plus breakdowns by hour of day, top-10% prices, and spike periods.

## Drift monitoring

Compares reference and monitoring windows with PSI, Kolmogorov-Smirnov tests, mean/std shifts, missing-rate changes, prediction and residual distribution shifts, and rolling MAE. Drift flags indicate investigation is warranted; they do not automatically trigger retraining.

## Repository structure

```
energy-price-forecasting/
├── data/raw/              # Input CSV
├── data/processed/        # Feature-engineered data, predictions
├── models/                # Saved model artifact
├── reports/figures/       # Evaluation and drift plots
├── reports/metrics/       # JSON and CSV metrics
├── notebooks/             # Exploratory notebook
├── scripts/               # CLI entry points
├── src/energy_forecasting/# Core library
└── tests/                 # pytest suite
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Requires Python 3.11 or newer.

Or: `make install`

## Reproduce results

```bash
make data      # generate synthetic data
make train     # full training pipeline
make drift     # drift analysis and README metric sync
make test      # run pytest
```

Figures and metric files are written to `reports/` and are excluded from Git. After training and drift analysis, run `python scripts/sync_readme_metrics.py` to refresh README numbers from `reports/metrics/model_metrics.json`.

## Model results

All metrics below are copied from `reports/metrics/model_metrics.json` after running the training pipeline on the default **synthetic** dataset.

Selected model: **Ridge** (`ridge`, lowest validation MAE)

### Validation metrics (all models)

| Model | MAE | RMSE | MAPE | sMAPE |
|-------|-----|------|------|-------|
| Ridge | 2.8349 | 5.6077 | 7.1750% | 7.1341% |
| HistGradientBoosting | 3.1568 | 5.8282 | 8.0795% | 7.9656% |
| Naive persistence | 5.1167 | 8.6566 | 13.2195% | 12.6987% |
| Seasonal naive | 5.6950 | 9.1459 | 14.5723% | 14.0364% |

### Test metrics (Ridge, held-out 15%; not used for model selection)

| Metric | Value |
|--------|-------|
| MAE | 7.9781 |
| RMSE | 11.2363 |
| MAPE | 13.4616% |
| sMAPE | 14.7769% |
| Top-10% price MAE | 15.4352 |
| Spike-period MAE (>= 73.94) | 15.4352 |

Test-set degradation relative to validation reflects the structural break injected in the synthetic data after the 85th percentile of the timeline.

Expanding-window time-series CV (Ridge): mean MAE 4.6710, mean RMSE 7.8579 across 5 folds.

Drift analysis flagged 27 of 32 features during the post-break monitoring window.
Monitoring-window MAE: 8.4358 (reference: 8.0169).

## Limitations

- **Synthetic data by default** — does not reflect real market dynamics.
- Electricity markets are affected by market rules, generation mix, congestion, outages, fuel prices, and weather.
- Strong synthetic-data performance does not guarantee real-market performance.
- Real deployment would require market-specific data, rigorous backtesting, and operational monitoring.

## Potential improvements

- Integrate public market data (e.g., ENTSO-E, CAISO day-ahead prices)
- Probabilistic forecasts and quantile loss
- Holiday and event calendars
- Automated retraining policies tied to validated drift investigations
