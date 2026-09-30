"""Forecast daily demand with calendar, lag, and rolling features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


FEATURES = [
    "day_of_week",
    "month",
    "day_of_year_sin",
    "day_of_year_cos",
    "lag_1",
    "lag_7",
    "lag_14",
    "rolling_7",
    "rolling_28",
]


def make_sales_series(days: int = 900, random_state: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    dates = pd.date_range("2023-01-01", periods=days, freq="D")
    trend = np.linspace(80, 116, days)
    weekly = 16 * np.sin(2 * np.pi * dates.dayofweek.to_numpy() / 7)
    yearly = 11 * np.sin(2 * np.pi * dates.dayofyear.to_numpy() / 365.25)
    promotions = rng.binomial(1, 0.08, days) * rng.uniform(12, 28, days)
    noise = rng.normal(0, 5.5, days)
    sales = np.maximum(0, trend + weekly + yearly + promotions + noise).round()
    return pd.DataFrame({"date": dates, "sales": sales.astype(int)})


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.sort_values("date").copy()
    data["day_of_week"] = data["date"].dt.dayofweek
    data["month"] = data["date"].dt.month
    data["day_of_year_sin"] = np.sin(2 * np.pi * data["date"].dt.dayofyear / 365.25)
    data["day_of_year_cos"] = np.cos(2 * np.pi * data["date"].dt.dayofyear / 365.25)
    for lag in (1, 7, 14):
        data[f"lag_{lag}"] = data["sales"].shift(lag)
    shifted = data["sales"].shift(1)
    data["rolling_7"] = shifted.rolling(7).mean()
    data["rolling_28"] = shifted.rolling(28).mean()
    return data.dropna().reset_index(drop=True)


def train_and_evaluate(test_days: int = 90, random_state: int = 7):
    data = build_features(make_sales_series(random_state=random_state))
    train, test = data.iloc[:-test_days], data.iloc[-test_days:]
    model = HistGradientBoostingRegressor(
        learning_rate=0.06, max_iter=220, max_leaf_nodes=20, l2_regularization=1.0, random_state=random_state
    )
    model.fit(train[FEATURES], train["sales"])
    predictions = model.predict(test[FEATURES])
    mae = mean_absolute_error(test["sales"], predictions)
    metrics = {
        "mae": round(float(mae), 3),
        "rmse": round(float(mean_squared_error(test["sales"], predictions) ** 0.5), 3),
        "mape_percent": round(float(np.mean(np.abs((test["sales"] - predictions) / test["sales"])) * 100), 3),
    }
    result = test[["date", "sales"]].copy()
    result["prediction"] = predictions.round(2)
    return model, metrics, result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--test-days", type=int, default=90)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    _, metrics, forecast = train_and_evaluate(args.test_days)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    forecast.to_csv(args.output_dir / "forecast.csv", index=False)

    plt.figure(figsize=(10, 4))
    plt.plot(forecast["date"], forecast["sales"], label="actual", linewidth=1.8)
    plt.plot(forecast["date"], forecast["prediction"], label="forecast", linewidth=1.8)
    plt.title("Daily demand: chronological holdout")
    plt.ylabel("units")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output_dir / "forecast.png", dpi=150)
    plt.close()
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
