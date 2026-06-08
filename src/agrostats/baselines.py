"""Compute baseline forecasts for comparison with tuned ML models."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from agrostats.modeling import TARGET_CROPS, compute_metrics


BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_FEATURES = BASE_DIR / "data" / "processed" / "agrostats_poltava_features.parquet"
REPORTS_DIR = BASE_DIR / "reports"
BASELINE_METRICS_PATH = REPORTS_DIR / "metrics_baselines.csv"
BASELINE_SUMMARY_PATH = REPORTS_DIR / "metrics_baselines_summary.csv"
ARIMA_PATH = REPORTS_DIR / "metrics_arima.csv"

TRAIN_END = 2018
VAL_YEARS = [2019, 2020, 2021]
TEST_YEARS = [2022, 2023, 2024]

BASELINES = ("naive_lag1", "forecast_linear", "linest_lag_only", "arima")

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="Non-invertible starting MA parameters found")
warnings.filterwarnings("ignore", message="Non-stationary starting autoregressive parameters found")


def load_features() -> pd.DataFrame:
    if not PROCESSED_FEATURES.exists():
        raise FileNotFoundError("Processed feature file not found. Run training pipeline first.")
    return pd.read_parquet(PROCESSED_FEATURES)


def lag_only_columns(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if c.endswith("_lag1") or c.startswith("ma5_")]
    return sorted([c for c in cols if df[c].notna().any()])


def forecast_naive(crop_df: pd.DataFrame, year: int) -> float | None:
    prev = crop_df[crop_df["year"] == year - 1]
    if prev.empty:
        return None
    return float(prev["Yield_t_ha"].iloc[0])


def forecast_linear_trend(crop_df: pd.DataFrame, year: int) -> float | None:
    train_df = crop_df[crop_df["year"] <= year - 1]
    if len(train_df) < 2:
        return None
    X_train = train_df[["year"]].to_numpy()
    y_train = train_df["Yield_t_ha"].to_numpy()
    model = LinearRegression()
    model.fit(X_train, y_train)
    return float(model.predict(np.array([[year]]))[0])


def forecast_linest_lag(crop_df: pd.DataFrame, year: int, lag_columns: List[str]) -> float | None:
    train_df = crop_df[crop_df["year"] <= year - 1]
    target_row = crop_df[crop_df["year"] == year]
    if target_row.empty or train_df.empty:
        return None

    X_train = train_df[lag_columns].dropna()
    y_train = train_df.loc[X_train.index, "Yield_t_ha"]
    if len(X_train) < 3:
        return None

    valid_cols = [col for col in X_train.columns if X_train[col].nunique(dropna=True) > 1]
    if not valid_cols:
        return None

    X_train = X_train[valid_cols]
    X_target = target_row[valid_cols]
    if X_target.isna().any().any():
        return None

    model = LinearRegression()
    model.fit(X_train, y_train)
    return float(model.predict(X_target)[0])


def _fit_best_arima(y: np.ndarray) -> tuple[object, tuple[int, int, int]] | tuple[None, None]:
    best_result = None
    best_order = None
    best_aic = np.inf

    for p in range(3):
        for d in range(3):
            for q in range(3):
                if p + d + q > 4:
                    continue
                try:
                    result = ARIMA(y, order=(p, d, q), trend="n").fit()
                except Exception:  # noqa: BLE001
                    continue
                if np.isfinite(result.aic) and result.aic < best_aic:
                    best_aic = float(result.aic)
                    best_result = result
                    best_order = (p, d, q)
    return best_result, best_order


def forecast_arima(crop_df: pd.DataFrame, year: int) -> tuple[float | None, tuple[int, int, int] | None]:
    train_df = crop_df[crop_df["year"] <= year - 1]
    if len(train_df) < 5:
        return None, None
    y_train = train_df["Yield_t_ha"].to_numpy(dtype=float)
    result, order = _fit_best_arima(y_train)
    if result is None or order is None:
        return None, None
    forecast = float(result.forecast(steps=1)[0])
    return forecast, order


def evaluate_baselines(features: pd.DataFrame) -> pd.DataFrame:
    lag_cols = lag_only_columns(features)
    records: List[Dict[str, float]] = []

    for crop in TARGET_CROPS:
        crop_df = features[features["group_or_crop"] == crop].sort_values("year").reset_index(drop=True)
        for year in crop_df["year"].tolist():
            if year <= TRAIN_END:
                continue
            split = "validation" if year in VAL_YEARS else "test" if year in TEST_YEARS else "train"
            y_true = float(crop_df.loc[crop_df["year"] == year, "Yield_t_ha"].iloc[0])

            for baseline in BASELINES:
                order = None
                if baseline == "naive_lag1":
                    y_pred = forecast_naive(crop_df, year)
                elif baseline == "forecast_linear":
                    y_pred = forecast_linear_trend(crop_df, year)
                elif baseline == "linest_lag_only":
                    y_pred = forecast_linest_lag(crop_df, year, lag_cols)
                elif baseline == "arima":
                    y_pred, order = forecast_arima(crop_df, year)
                else:
                    continue

                if y_pred is None:
                    continue
                metrics = compute_metrics(np.array([y_true]), np.array([y_pred]))
                records.append(
                    {
                        "year": year,
                        "crop": crop,
                        "baseline": baseline,
                        "scenario": "lag_only" if baseline in {"naive_lag1", "linest_lag_only"} else "trend",
                        "split": split,
                        "y_true": y_true,
                        "y_pred": y_pred,
                        "mae": metrics["mae"],
                        "rmse": metrics["rmse"],
                        "mape": metrics["mape"],
                        "order": "" if order is None else str(order),
                    }
                )

    df = pd.DataFrame(records)
    if df.empty:
        raise RuntimeError("No baseline predictions were generated.")
    return df


def summarise_metrics(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df[df["split"] == "test"]
        .groupby(["baseline", "crop"], as_index=False)
        .agg(
            mae=("mae", "mean"),
            rmse=("rmse", "mean"),
            mape=("mape", "mean"),
        )
    )


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    features = load_features()
    results = evaluate_baselines(features)
    results.to_csv(BASELINE_METRICS_PATH, index=False)
    summary = summarise_metrics(results)
    summary.to_csv(BASELINE_SUMMARY_PATH, index=False)
    results[results["baseline"] == "arima"].to_csv(ARIMA_PATH, index=False)
    print("Saved baseline metrics to", BASELINE_METRICS_PATH)
    print("Saved baseline summary to", BASELINE_SUMMARY_PATH)
    print("Saved ARIMA metrics to", ARIMA_PATH)


if __name__ == "__main__":
    main()
