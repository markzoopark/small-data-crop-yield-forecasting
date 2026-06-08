"""Generate supplementary analysis artefacts for the ERC revision."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from agrostats.climate import aggregate_power_apr_sep, merge_climate_features
from agrostats.diagnostics import run_feature_group_ablation, run_prediction_bands
from agrostats.reliability import run_reliability_outputs
from agrostats.modeling import (
    FEATURE_GROUPS,
    LAG_CONFIGS,
    TARGET_CROPS,
    aggregate_predictions,
    evaluate_expanding_window,
    get_years_for_window,
    prepare_crop_dataset,
    tune_model,
)


BASE_DIR = Path(__file__).resolve().parents[2]
FEATURES_PATH = BASE_DIR / "data" / "processed" / "agrostats_poltava_features.parquet"
REPORTS_DIR = BASE_DIR / "reports"

TUNED_PARAMS_PATH = REPORTS_DIR / "tuned_hyperparameters.csv"
BASELINE_RESULTS_PATH = REPORTS_DIR / "metrics_baselines.csv"
LEADERBOARD_PATH = REPORTS_DIR / "metrics_leaderboard.csv"

LAG_SENSITIVITY_PATH = REPORTS_DIR / "lag_sensitivity.csv"
ROBUSTNESS_PATH = REPORTS_DIR / "robustness_2020_2024.csv"
CLIMATE_SENSITIVITY_PATH = REPORTS_DIR / "climate_sensitivity.csv"
VARIABLE_SUMMARY_PATH = REPORTS_DIR / "variable_summary.csv"
MAIZE_DIAGNOSTICS_PATH = REPORTS_DIR / "maize_diagnostics.csv"
FEATURE_GROUP_ABLATION_PATH = REPORTS_DIR / "feature_group_ablation.csv"
PREDICTION_BANDS_PATH = REPORTS_DIR / "prediction_bands.csv"
PREDICTION_BAND_SUMMARY_PATH = REPORTS_DIR / "prediction_band_summary.csv"

MODELS = ("elasticnet", "xgboost", "lightgbm")

UNIT_MAP = {
    "Yield_t_ha_lag1": "t/ha",
    "Yield_t_ha_lag2": "t/ha",
    "Yield_t_ha_lag3": "t/ha",
    "Area_ha_lag1": "ha",
    "Area_ha_lag2": "ha",
    "Area_ha_lag3": "ha",
    "N_kg_ha_lag1": "kg/ha",
    "N_kg_ha_lag2": "kg/ha",
    "N_kg_ha_lag3": "kg/ha",
    "P2O5_kg_ha_lag1": "kg/ha",
    "P2O5_kg_ha_lag2": "kg/ha",
    "P2O5_kg_ha_lag3": "kg/ha",
    "K_kg_ha_lag1": "kg/ha",
    "K_kg_ha_lag2": "kg/ha",
    "K_kg_ha_lag3": "kg/ha",
    "Mineral_treated_share_lag1": "share",
    "Mineral_treated_share_lag2": "share",
    "Mineral_treated_share_lag3": "share",
    "Org_kg_ha_or_share_lag1": "mixed",
    "Org_kg_ha_or_share_lag2": "mixed",
    "Org_kg_ha_or_share_lag3": "mixed",
    "Irrig_m3_ha_lag1": "m3/ha",
    "Irrig_m3_ha_lag2": "m3/ha",
    "Irrig_m3_ha_lag3": "m3/ha",
    "Irrig_mm_lag1": "mm",
    "Irrig_mm_lag2": "mm",
    "Irrig_mm_lag3": "mm",
    "ma5_Yield": "t/ha",
    "ma5_N": "kg/ha",
    "ma5_P2O5": "kg/ha",
    "ma5_K": "kg/ha",
    "ma5_MineralShare": "share",
    "ma5_Org": "mixed",
    "ma5_Irrig_m3": "m3/ha",
    "ma5_Irrig_mm": "mm",
    "climate_apr_sep_t2m_mean": "C",
    "climate_apr_sep_t2m_max_mean": "C",
    "climate_apr_sep_prectotcorr_total": "mm",
    "climate_apr_sep_hot_days_gt30": "days",
}

CATEGORY_MAP = {
    "yield_history": "crop_history",
    "area": "crop_area",
    "fertiliser_n": "fertiliser",
    "fertiliser_p": "fertiliser",
    "fertiliser_k": "fertiliser",
    "mineral_share": "fertiliser",
    "organics": "fertiliser",
    "irrigation": "water_management",
    "climate": "climate",
}

TIMING_MAP = {
    "_lag1": "t-1",
    "_lag2": "t-2",
    "_lag3": "t-3",
    "ma5_": "historical_5y_mean",
    "climate_": "Apr-Sep current season",
}


def load_features_with_climate() -> pd.DataFrame:
    features = pd.read_parquet(FEATURES_PATH)
    climate = aggregate_power_apr_sep()
    return merge_climate_features(features, climate)


def _read_tuned_params() -> pd.DataFrame:
    return pd.read_csv(TUNED_PARAMS_PATH)


def _best_lag_only_models() -> pd.DataFrame:
    leaderboard = pd.read_csv(LEADERBOARD_PATH)
    return leaderboard


def _feature_category(feature: str) -> str:
    for group_name, feature_names in FEATURE_GROUPS.items():
        if feature in feature_names:
            return CATEGORY_MAP.get(group_name, "other")
    return "other"


def _feature_timing(feature: str) -> str:
    for prefix, timing in TIMING_MAP.items():
        if prefix.startswith("_") and feature.endswith(prefix):
            return timing
        if feature.startswith(prefix):
            return timing
    return "derived"


def build_variable_summary(features_with_climate: pd.DataFrame) -> pd.DataFrame:
    feature_columns = [
        col
        for col in features_with_climate.columns
        if col not in {"region", "group_or_crop", "year", "Yield_t_ha", "Yield_anom"}
    ]
    rows = []
    for feature in feature_columns:
        series = pd.to_numeric(features_with_climate[feature], errors="coerce").dropna()
        if series.empty:
            continue
        mean_value = float(series.mean())
        rows.append(
            {
                "variable": feature,
                "category": _feature_category(feature),
                "unit": UNIT_MAP.get(feature, ""),
                "timing": _feature_timing(feature),
                "mean": mean_value,
                "min": float(series.min()),
                "max": float(series.max()),
                "coefficient_of_variation": float(series.std(ddof=0) / mean_value) if mean_value else float("nan"),
            }
        )
    summary = pd.DataFrame(rows).sort_values(["category", "variable"])
    summary.to_csv(VARIABLE_SUMMARY_PATH, index=False)
    return summary


def run_lag_sensitivity(features_with_climate: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for crop in TARGET_CROPS:
        dataset = prepare_crop_dataset(features_with_climate, crop)
        if dataset is None:
            continue
        for lag_config in LAG_CONFIGS:
            for model_name in MODELS:
                best_params, tuning_df = tune_model(dataset, model_name=model_name, scenario="lag_only", lag_config=lag_config)
                if tuning_df.empty:
                    continue
                pred_df = evaluate_expanding_window(
                    dataset,
                    model_name=model_name,
                    scenario="lag_only",
                    params=best_params,
                    years=get_years_for_window("test"),
                    lag_config=lag_config,
                )
                summary = aggregate_predictions(pred_df, group_cols=["crop", "model", "scenario", "lag_config"])
                if summary.empty:
                    continue
                row = summary.iloc[0].to_dict()
                best_val = tuning_df.sort_values(["mae", "rmse", "mape"]).iloc[0]
                row["validation_mae"] = float(best_val["mae"])
                row["validation_rmse"] = float(best_val["rmse"])
                row["validation_mape"] = float(best_val["mape"])
                row["params_json"] = json.dumps(best_params, ensure_ascii=False)
                rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(LAG_SENSITIVITY_PATH, index=False)
    return result


def run_robustness_window(features_with_climate: pd.DataFrame, tuned_df: pd.DataFrame, baseline_df: pd.DataFrame) -> pd.DataFrame:
    leaderboard = _best_lag_only_models()
    rows = []
    robustness_years = get_years_for_window("robustness_2020_2024")

    for crop in TARGET_CROPS:
        crop_leader = leaderboard[leaderboard["crop"] == crop]
        if crop_leader.empty:
            continue
        best_model = crop_leader.iloc[0]["model"]
        tuned_row = tuned_df[
            (tuned_df["crop"] == crop) & (tuned_df["scenario"] == "lag_only") & (tuned_df["model"] == best_model)
        ]
        if tuned_row.empty:
            continue
        params = json.loads(tuned_row.iloc[0]["params_json"])
        dataset = prepare_crop_dataset(features_with_climate, crop)
        if dataset is None:
            continue
        pred_df = evaluate_expanding_window(
            dataset,
            model_name=best_model,
            scenario="lag_only",
            params=params,
            years=robustness_years,
            lag_config="L1",
        )
        ml_summary = aggregate_predictions(pred_df, group_cols=["crop", "model", "scenario", "lag_config"])
        if not ml_summary.empty:
            row = ml_summary.iloc[0].to_dict()
            row["approach_type"] = "ml"
            row["approach_name"] = best_model
            row["window"] = "2020_2024"
            rows.append(row)

        baseline_subset = baseline_df[(baseline_df["crop"] == crop) & (baseline_df["year"] >= 2020)]
        if not baseline_subset.empty:
            baseline_summary = (
                baseline_subset.groupby(["baseline", "crop"], as_index=False)
                .agg(mae=("mae", "mean"), rmse=("rmse", "mean"), mape=("mape", "mean"), n=("year", "count"))
                .sort_values(["mae", "rmse", "mape"])
            )
            best_baseline = baseline_summary.iloc[0].to_dict()
            best_baseline["approach_type"] = "baseline"
            best_baseline["approach_name"] = best_baseline["baseline"]
            best_baseline["window"] = "2020_2024"
            rows.append(best_baseline)

    result = pd.DataFrame(rows)
    result.to_csv(ROBUSTNESS_PATH, index=False)
    return result


def run_climate_sensitivity(features_with_climate: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for crop in TARGET_CROPS:
        dataset = prepare_crop_dataset(features_with_climate, crop)
        if dataset is None:
            continue

        base_candidates = []
        climate_candidates = []
        for model_name in MODELS:
            base_params, base_tuning = tune_model(dataset, model_name=model_name, scenario="lag_only", lag_config="L1", include_climate=False)
            if not base_tuning.empty:
                base_eval = evaluate_expanding_window(
                    dataset,
                    model_name=model_name,
                    scenario="lag_only",
                    params=base_params,
                    years=get_years_for_window("test"),
                    lag_config="L1",
                    include_climate=False,
                )
                base_summary = aggregate_predictions(base_eval, group_cols=["crop", "model", "scenario", "lag_config"])
                if not base_summary.empty:
                    row = base_summary.iloc[0].to_dict()
                    row["params_json"] = json.dumps(base_params, ensure_ascii=False)
                    base_candidates.append(row)

            climate_params, climate_tuning = tune_model(dataset, model_name=model_name, scenario="lag_only", lag_config="L1", include_climate=True)
            if not climate_tuning.empty:
                climate_eval = evaluate_expanding_window(
                    dataset,
                    model_name=model_name,
                    scenario="lag_only",
                    params=climate_params,
                    years=get_years_for_window("test"),
                    lag_config="L1",
                    include_climate=True,
                )
                climate_summary = aggregate_predictions(climate_eval, group_cols=["crop", "model", "scenario", "lag_config", "include_climate"])
                if not climate_summary.empty:
                    row = climate_summary.iloc[0].to_dict()
                    row["params_json"] = json.dumps(climate_params, ensure_ascii=False)
                    climate_candidates.append(row)

        if not base_candidates or not climate_candidates:
            continue

        best_base = pd.DataFrame(base_candidates).sort_values(["mae", "rmse", "mape"]).iloc[0]
        best_climate = pd.DataFrame(climate_candidates).sort_values(["mae", "rmse", "mape"]).iloc[0]
        rows.append(
            {
                "crop": crop,
                "window": "2022_2024",
                "base_model": best_base["model"],
                "base_mae": best_base["mae"],
                "base_rmse": best_base["rmse"],
                "base_mape": best_base["mape"],
                "climate_model": best_climate["model"],
                "climate_mae": best_climate["mae"],
                "climate_rmse": best_climate["rmse"],
                "climate_mape": best_climate["mape"],
                "delta_mae": best_climate["mae"] - best_base["mae"],
            }
        )

    result = pd.DataFrame(rows)
    result.to_csv(CLIMATE_SENSITIVITY_PATH, index=False)
    return result


def run_maize_diagnostics(features_with_climate: pd.DataFrame, tuned_df: pd.DataFrame, baseline_df: pd.DataFrame) -> pd.DataFrame:
    crop = "Кукурудза"
    leaderboard = _best_lag_only_models()
    crop_leader = leaderboard[leaderboard["crop"] == crop]
    if crop_leader.empty:
        result = pd.DataFrame()
        result.to_csv(MAIZE_DIAGNOSTICS_PATH, index=False)
        return result

    best_model = crop_leader.iloc[0]["model"]
    tuned_row = tuned_df[
        (tuned_df["crop"] == crop) & (tuned_df["scenario"] == "lag_only") & (tuned_df["model"] == best_model)
    ]
    if tuned_row.empty:
        result = pd.DataFrame()
        result.to_csv(MAIZE_DIAGNOSTICS_PATH, index=False)
        return result

    params = json.loads(tuned_row.iloc[0]["params_json"])
    dataset = prepare_crop_dataset(features_with_climate, crop)
    if dataset is None:
        result = pd.DataFrame()
        result.to_csv(MAIZE_DIAGNOSTICS_PATH, index=False)
        return result

    rows = []
    ml_eval = evaluate_expanding_window(
        dataset,
        model_name=best_model,
        scenario="lag_only",
        params=params,
        years=get_years_for_window("test"),
        lag_config="L1",
    )
    if not ml_eval.empty:
        for _, row in ml_eval.iterrows():
            rows.append(
                {
                    "analysis_type": "yearly_error",
                    "model": best_model,
                    "year": int(row["year"]),
                    "abs_error": abs(float(row["actual"]) - float(row["predicted"])),
                    "feature_group": "",
                    "mae": row["mae"],
                    "delta_mae": np.nan,
                }
            )

    for baseline_name in ("linest_lag_only", "arima"):
        subset = baseline_df[
            (baseline_df["crop"] == crop)
            & (baseline_df["baseline"] == baseline_name)
            & (baseline_df["split"] == "test")
        ]
        for _, row in subset.iterrows():
            rows.append(
                {
                    "analysis_type": "yearly_error",
                    "model": baseline_name,
                    "year": int(row["year"]),
                    "abs_error": abs(float(row["y_true"]) - float(row["y_pred"])),
                    "feature_group": "",
                    "mae": row["mae"],
                    "delta_mae": np.nan,
                }
            )

    full_summary = aggregate_predictions(ml_eval, group_cols=["crop", "model", "scenario", "lag_config"])
    full_mae = float(full_summary.iloc[0]["mae"]) if not full_summary.empty else np.nan
    for group_name in FEATURE_GROUPS:
        if group_name == "climate":
            continue
        ablated = evaluate_expanding_window(
            dataset,
            model_name=best_model,
            scenario="lag_only",
            params=params,
            years=get_years_for_window("test"),
            lag_config="L1",
            drop_feature_groups=[group_name],
        )
        summary = aggregate_predictions(ablated, group_cols=["crop", "model", "scenario", "lag_config"])
        if summary.empty:
            continue
        mae = float(summary.iloc[0]["mae"])
        rows.append(
            {
                "analysis_type": "feature_group_ablation",
                "model": best_model,
                "year": np.nan,
                "abs_error": np.nan,
                "feature_group": group_name,
                "mae": mae,
                "delta_mae": mae - full_mae,
            }
        )

    result = pd.DataFrame(rows)
    result.to_csv(MAIZE_DIAGNOSTICS_PATH, index=False)
    return result


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    features_with_climate = load_features_with_climate()
    tuned_df = _read_tuned_params()
    baseline_df = pd.read_csv(BASELINE_RESULTS_PATH)
    leaderboard = _best_lag_only_models()

    build_variable_summary(features_with_climate)
    run_lag_sensitivity(features_with_climate)
    run_robustness_window(features_with_climate, tuned_df, baseline_df)
    run_climate_sensitivity(features_with_climate)
    run_maize_diagnostics(features_with_climate, tuned_df, baseline_df)
    run_feature_group_ablation(features_with_climate, tuned_df, leaderboard, FEATURE_GROUP_ABLATION_PATH)
    run_prediction_bands(
        features_with_climate,
        tuned_df,
        leaderboard,
        PREDICTION_BANDS_PATH,
        PREDICTION_BAND_SUMMARY_PATH,
    )
    run_reliability_outputs(
        leaderboard_path=LEADERBOARD_PATH,
        baseline_summary_path=REPORTS_DIR / "metrics_baselines_summary.csv",
        band_summary_path=PREDICTION_BAND_SUMMARY_PATH,
        feature_ablation_path=FEATURE_GROUP_ABLATION_PATH,
    )
    print("Saved revision analysis artefacts to", REPORTS_DIR)


if __name__ == "__main__":
    main()
