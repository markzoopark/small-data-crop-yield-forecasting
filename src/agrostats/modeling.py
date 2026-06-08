"""Reusable modeling utilities for leakage-safe forecasting experiments."""

from __future__ import annotations

import itertools
import json
import math
import re
import unicodedata
import warnings
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMRegressor was fitted with feature names",
)


SPLIT_TRAIN_END = 2018
SPLIT_VAL_END = 2021
TEST_START = 2022
ROBUSTNESS_START = 2020
RANDOM_STATE = 42

TARGET_CROPS = ("Пшениця", "Кукурудза", "Соняшник")
SCENARIOS = ("lag_only", "in_season")
LAG_CONFIGS = ("L1", "L1_L2", "L1_L2_L3")

MANUAL_SLUGS = {
    "Пшениця": "pshenytsia",
    "Кукурудза": "kukurudza",
    "Соняшник": "sonyashnyk",
}

MODEL_LABELS = {
    "elasticnet": "ElasticNet",
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
}

FEATURE_GROUPS = {
    "yield_history": ("Yield_t_ha_lag1", "Yield_t_ha_lag2", "Yield_t_ha_lag3", "ma5_Yield"),
    "area": ("Area_ha_lag1", "Area_ha_lag2", "Area_ha_lag3"),
    "fertiliser_n": ("N_kg_ha_lag1", "N_kg_ha_lag2", "N_kg_ha_lag3", "ma5_N"),
    "fertiliser_p": ("P2O5_kg_ha_lag1", "P2O5_kg_ha_lag2", "P2O5_kg_ha_lag3", "ma5_P2O5"),
    "fertiliser_k": ("K_kg_ha_lag1", "K_kg_ha_lag2", "K_kg_ha_lag3", "ma5_K"),
    "mineral_share": (
        "Mineral_treated_share_lag1",
        "Mineral_treated_share_lag2",
        "Mineral_treated_share_lag3",
        "ma5_MineralShare",
    ),
    "organics": (
        "Org_kg_ha_or_share_lag1",
        "Org_kg_ha_or_share_lag2",
        "Org_kg_ha_or_share_lag3",
        "ma5_Org",
    ),
    "irrigation": (
        "Irrig_m3_ha_lag1",
        "Irrig_m3_ha_lag2",
        "Irrig_m3_ha_lag3",
        "Irrig_mm_lag1",
        "Irrig_mm_lag2",
        "Irrig_mm_lag3",
        "ma5_Irrig_m3",
        "ma5_Irrig_mm",
    ),
    "climate": (
        "climate_apr_sep_t2m_mean",
        "climate_apr_sep_t2m_max_mean",
        "climate_apr_sep_prectotcorr_total",
        "climate_apr_sep_hot_days_gt30",
    ),
}


@dataclass
class CropDataset:
    crop: str
    years: pd.Series
    features: pd.DataFrame
    target: pd.Series


def slugify(value: str) -> str:
    if value in MANUAL_SLUGS:
        return MANUAL_SLUGS[value]
    normalised = unicodedata.normalize("NFKD", value)
    ascii_value = normalised.encode("ascii", "ignore").decode("ascii")
    ascii_value = re.sub(r"[^A-Za-z0-9]+", "_", ascii_value)
    slug = ascii_value.strip("_").lower()
    return slug or f"crop_{abs(hash(value)) % 10000}"


def classify_split(year: int) -> str:
    if year <= SPLIT_TRAIN_END:
        return "train"
    if year <= SPLIT_VAL_END:
        return "validation"
    return "test"


def get_years_for_window(window: str) -> list[int]:
    if window == "validation":
        return list(range(SPLIT_TRAIN_END + 1, SPLIT_VAL_END + 1))
    if window == "test":
        return list(range(TEST_START, TEST_START + 3))
    if window == "robustness_2020_2024":
        return list(range(ROBUSTNESS_START, TEST_START + 3))
    raise ValueError(f"Unsupported evaluation window: {window}")


def parse_param_record(params: Mapping[str, object]) -> str:
    return json.dumps(dict(sorted(params.items())), ensure_ascii=False)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(math.sqrt(np.mean((y_true - y_pred) ** 2)))
    with np.errstate(divide="ignore", invalid="ignore"):
        mape_array = np.abs((y_true - y_pred) / y_true) * 100
        valid = mape_array[~np.isinf(mape_array) & ~np.isnan(mape_array)]
        mape = float(np.mean(valid)) if valid.size else float("nan")
    return {"mae": mae, "rmse": rmse, "mape": mape}


def prepare_crop_dataset(df: pd.DataFrame, crop: str) -> Optional[CropDataset]:
    subset = df[df["group_or_crop"] == crop].copy()
    if subset.empty:
        return None

    subset = subset.sort_values("year")
    subset["year"] = subset["year"].astype(int)
    numeric_cols = subset.select_dtypes(include=[np.number]).columns.tolist()
    for col in subset.columns:
        if col not in numeric_cols and col not in {"region", "group_or_crop"}:
            subset[col] = pd.to_numeric(subset[col], errors="coerce")

    feature_cols = [
        col
        for col in subset.columns
        if col not in {"region", "group_or_crop", "year", "Yield_t_ha", "Yield_anom"}
    ]
    X = subset[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna(axis=1, how="all")
    if X.shape[1] == 0:
        return None

    target = subset["Yield_t_ha"].astype(float)
    if target.isna().all():
        return None

    return CropDataset(
        crop=crop,
        years=subset["year"],
        features=X,
        target=target,
    )


def _lag_suffixes(lag_config: str) -> tuple[str, ...]:
    if lag_config == "L1":
        return ("_lag1",)
    if lag_config == "L1_L2":
        return ("_lag1", "_lag2")
    if lag_config == "L1_L2_L3":
        return ("_lag1", "_lag2", "_lag3")
    raise ValueError(f"Unsupported lag configuration: {lag_config}")


def build_scenario_dataset(
    dataset: CropDataset,
    scenario: str,
    *,
    lag_config: str = "L1",
    include_climate: bool = False,
    drop_feature_groups: Optional[Sequence[str]] = None,
) -> Optional[CropDataset]:
    features = dataset.features.copy()
    drop_feature_groups = tuple(drop_feature_groups or ())

    if scenario == "lag_only":
        suffixes = _lag_suffixes(lag_config)
        allowed_cols = [
            col
            for col in features.columns
            if col.startswith("ma5_") or col.endswith(suffixes)
        ]
        climate_cols = [col for col in features.columns if col.startswith("climate_")]
        if include_climate:
            allowed_cols.extend(climate_cols)
        features = features.loc[:, [col for col in allowed_cols if col in features.columns]]
    elif scenario == "in_season":
        if not include_climate:
            climate_cols = [col for col in features.columns if col.startswith("climate_")]
            features = features.drop(columns=climate_cols, errors="ignore")
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    if drop_feature_groups:
        blocked = set()
        for group_name in drop_feature_groups:
            blocked.update(FEATURE_GROUPS.get(group_name, ()))
        features = features.drop(columns=[col for col in blocked if col in features.columns], errors="ignore")

    features = features.dropna(axis=1, how="all")
    if features.empty:
        return None

    return CropDataset(
        crop=dataset.crop,
        years=dataset.years.copy(),
        features=features,
        target=dataset.target.copy(),
    )


def get_param_grid(model_name: str) -> list[dict[str, object]]:
    if model_name == "elasticnet":
        return [
            {"alpha": alpha, "l1_ratio": l1_ratio}
            for alpha, l1_ratio in itertools.product((0.01, 0.1, 1.0), (0.2, 0.5, 0.8))
        ]
    if model_name == "xgboost":
        return [
            {
                "n_estimators": n_estimators,
                "learning_rate": learning_rate,
                "max_depth": max_depth,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
            }
            for n_estimators, learning_rate, max_depth in itertools.product(
                (100, 300),
                (0.03, 0.1),
                (2, 4),
            )
        ]
    if model_name == "lightgbm":
        return [
            {
                "n_estimators": n_estimators,
                "learning_rate": learning_rate,
                "num_leaves": num_leaves,
                "max_depth": -1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
            }
            for n_estimators, learning_rate, num_leaves in itertools.product(
                (100, 300),
                (0.03, 0.1),
                (15, 31),
            )
        ]
    raise ValueError(f"Unknown model: {model_name}")


def build_model(model_name: str, params: Optional[Mapping[str, object]] = None):
    params = dict(params or {})
    if model_name == "elasticnet":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    ElasticNet(
                        alpha=float(params.get("alpha", 0.1)),
                        l1_ratio=float(params.get("l1_ratio", 0.5)),
                        random_state=RANDOM_STATE,
                        max_iter=10000,
                    ),
                ),
            ]
        )
    if model_name == "xgboost":
        from xgboost import XGBRegressor

        model_params = {
            "n_estimators": int(params.get("n_estimators", 300)),
            "learning_rate": float(params.get("learning_rate", 0.05)),
            "max_depth": int(params.get("max_depth", 4)),
            "min_child_weight": float(params.get("min_child_weight", 1)),
            "subsample": float(params.get("subsample", 0.8)),
            "colsample_bytree": float(params.get("colsample_bytree", 0.8)),
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "objective": "reg:squarederror",
            "verbosity": 0,
        }
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", XGBRegressor(**model_params)),
            ]
        )
    if model_name == "lightgbm":
        from lightgbm import LGBMRegressor

        model_params = {
            "n_estimators": int(params.get("n_estimators", 300)),
            "learning_rate": float(params.get("learning_rate", 0.05)),
            "num_leaves": int(params.get("num_leaves", 31)),
            "max_depth": int(params.get("max_depth", -1)),
            "subsample": float(params.get("subsample", 0.8)),
            "colsample_bytree": float(params.get("colsample_bytree", 0.8)),
            "random_state": RANDOM_STATE,
            "min_child_samples": 1,
            "min_data_in_leaf": 1,
            "min_data_in_bin": 1,
            "verbose": -1,
        }
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", LGBMRegressor(**model_params)),
            ]
        )
    raise ValueError(f"Unknown model: {model_name}")


def _select_train_columns(X_train: pd.DataFrame) -> list[str]:
    available = X_train.notna().any(axis=0)
    candidate_cols = X_train.columns[available].tolist()
    valid_cols = []
    for col in candidate_cols:
        if X_train[col].nunique(dropna=True) > 1:
            valid_cols.append(col)
    return valid_cols


def evaluate_expanding_window(
    dataset: CropDataset,
    *,
    model_name: str,
    scenario: str,
    params: Optional[Mapping[str, object]] = None,
    years: Sequence[int],
    lag_config: str = "L1",
    include_climate: bool = False,
    drop_feature_groups: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    scenario_dataset = build_scenario_dataset(
        dataset,
        scenario,
        lag_config=lag_config,
        include_climate=include_climate,
        drop_feature_groups=drop_feature_groups,
    )
    if scenario_dataset is None:
        return pd.DataFrame()

    records: list[dict[str, object]] = []
    for year in years:
        train_mask = scenario_dataset.years < year
        test_mask = scenario_dataset.years == year
        if not train_mask.any() or not test_mask.any():
            continue

        X_train = scenario_dataset.features.loc[train_mask].copy()
        y_train = scenario_dataset.target.loc[train_mask].copy()
        X_test = scenario_dataset.features.loc[test_mask].copy()
        y_test = scenario_dataset.target.loc[test_mask].copy()

        valid_cols = _select_train_columns(X_train)
        if not valid_cols:
            continue

        X_train = X_train[valid_cols]
        X_test = X_test[valid_cols]

        model = build_model(model_name, params=params)
        model.fit(X_train, y_train)
        predictions = np.asarray(model.predict(X_test), dtype=float)

        for actual, predicted in zip(y_test.to_numpy(dtype=float), predictions):
            metrics = compute_metrics(np.array([actual]), np.array([predicted]))
            records.append(
                {
                    "crop": dataset.crop,
                    "model": model_name,
                    "scenario": scenario,
                    "lag_config": lag_config,
                    "year": int(year),
                    "split": classify_split(int(year)),
                    "actual": float(actual),
                    "predicted": float(predicted),
                    "n_features": len(valid_cols),
                    "include_climate": include_climate,
                    "params_json": parse_param_record(params or {}),
                    **metrics,
                }
            )

    return pd.DataFrame(records)


def tune_model(
    dataset: CropDataset,
    *,
    model_name: str,
    scenario: str,
    lag_config: str = "L1",
    include_climate: bool = False,
) -> tuple[dict[str, object], pd.DataFrame]:
    validation_years = get_years_for_window("validation")
    candidates = get_param_grid(model_name)
    tuning_rows: list[dict[str, object]] = []

    for params in candidates:
        pred_df = evaluate_expanding_window(
            dataset,
            model_name=model_name,
            scenario=scenario,
            params=params,
            years=validation_years,
            lag_config=lag_config,
            include_climate=include_climate,
        )
        if pred_df.empty:
            continue
        aggregate = aggregate_predictions(
            pred_df,
            group_cols=["crop", "model", "scenario", "lag_config", "include_climate"],
        )
        if aggregate.empty:
            continue
        row = aggregate.iloc[0].to_dict()
        row["params_json"] = parse_param_record(params)
        tuning_rows.append(row)

    tuning_df = pd.DataFrame(tuning_rows)
    if tuning_df.empty:
        return {}, tuning_df

    best_row = tuning_df.sort_values(["mae", "rmse", "mape"]).iloc[0]
    best_params = json.loads(str(best_row["params_json"]))
    return best_params, tuning_df


def aggregate_predictions(predictions: pd.DataFrame, *, group_cols: Sequence[str]) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    grouped = (
        predictions.groupby(list(group_cols), as_index=False)
        .agg(
            mae=("mae", "mean"),
            rmse=("rmse", "mean"),
            mape=("mape", "mean"),
            n=("year", "count"),
        )
    )
    return grouped


def choose_best_by_crop(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    idx = summary.groupby("crop")["mae"].idxmin()
    return summary.loc[idx].reset_index(drop=True)
