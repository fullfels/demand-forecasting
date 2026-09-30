# Retail Demand Forecasting

A small time-series project for forecasting daily retail demand. It builds calendar and lag features, trains a gradient-boosting regressor, and evaluates predictions on a chronological holdout period.

## What it demonstrates

- Time-aware feature engineering
- Lag and rolling-window features without target leakage
- Chronological train/test split
- MAE, RMSE, and MAPE evaluation
- Forecast chart and CSV export

## Quick start

```bash
python -m venv .venv
pip install -r requirements.txt
python forecast.py --output-dir artifacts
pytest -q
```

The script generates a realistic synthetic sales series, so the repository works immediately and does not rely on an external dataset.

## Outputs

- `artifacts/metrics.json`
- `artifacts/forecast.csv`
- `artifacts/forecast.png`

## License

MIT
