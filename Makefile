.PHONY: install data train drift test notebook clean

install:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	.venv/bin/pip install -e .

data:
	.venv/bin/python scripts/generate_sample_data.py

train:
	.venv/bin/python scripts/run_training.py

drift:
	.venv/bin/python scripts/run_drift_analysis.py
	.venv/bin/python scripts/sync_readme_metrics.py

test:
	.venv/bin/pytest tests/ -v

notebook:
	.venv/bin/jupyter notebook notebooks/01_energy_price_forecasting.ipynb

clean:
	rm -rf .venv __pycache__ .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
