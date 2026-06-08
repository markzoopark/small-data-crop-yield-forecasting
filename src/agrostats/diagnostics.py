"""Additional small-sample diagnostics for the mini article."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from agrostats.modeling import (
    FEATURE_GROUPS,
    aggregate_predictions,
    evaluate_expanding_window,
    get_years_for_window,
    prepare_crop_dataset,
)


BAND_QUANTILE = 0.8
BAND_NOTE = (
    "Empirical validation-residual diagnostic band; small-sample uncertainty "
    "screen, not a formal confidence interval."
)


def build_band_summary(
    validation_predictions: pd.DataFrame,
    *,
    quantile: float = BAND_QUANTILE,
) -> pd.DataFrame:
    """Summarise validation residuals into empirical absolute-error bands."""
    if validation_predictions.empty:
        return pd.DataFrame()

    rows = []
    group_cols = ["crop", "model", "scenario", "lag_config"]
    for keys, group in validation_predictions.groupby(group_cols, dropna=False):
        residuals = group["actual"].astype(float) - group["predicted"].astype(float)
        abs_errors = residuals.abs()
        rows.append(
            {
                "crop": keys[0],
                "model": keys[1],
                "scenario": keys[2],
                "lag_config": keys[3],
                "residual_n": int(abs_errors.count()),
                "validation_mae": float(abs_errors.mean()),
                "validation_rmse": float(np.sqrt(np.mean(np.square(residuals)))),
                "empirical_abs_error_q80": float(abs_errors.quantile(quantile)),
                "band_quantile": quantile,
                "band_note": BAND_NOTE,
            }
        )
    return pd.DataFrame(rows)


def apply_prediction_bands(test_predictions: pd.DataFrame, band_summary: pd.DataFrame) -> pd.DataFrame:
    """Attach symmetric empirical validation-error bands to test predictions."""
    if test_predictions.empty or band_summary.empty:
        return pd.DataFrame()

    merge_cols = ["crop", "model", "scenario", "lag_config"]
    merged = test_predictions.merge(
        band_summary[merge_cols + ["empirical_abs_error_q80", "band_quantile", "band_note"]],
        on=merge_cols,
        how="left",
    )
    merged = merged.dropna(subset=["empirical_abs_error_q80"]).copy()
    if merged.empty:
        return pd.DataFrame()

    width = merged["empirical_abs_error_q80"].astype(float)
    merged["band_lower"] = merged["predicted"].astype(float) - width
    merged["band_upper"] = merged["predicted"].astype(float) + width
    merged["actual_within_band"] = (
        (merged["actual"].astype(float) >= merged["band_lower"])
        & (merged["actual"].astype(float) <= merged["band_upper"])
    )
    return merged[
        [
            "crop",
            "model",
            "scenario",
            "lag_config",
            "year",
            "actual",
            "predicted",
            "band_lower",
            "band_upper",
            "empirical_abs_error_q80",
            "band_quantile",
            "actual_within_band",
            "band_note",
        ]
    ].reset_index(drop=True)


def summarise_band_coverage(prediction_bands: pd.DataFrame, band_summary: pd.DataFrame) -> pd.DataFrame:
    """Add test-window coverage diagnostics to validation-residual summaries."""
    if band_summary.empty:
        return pd.DataFrame()
    if prediction_bands.empty:
        result = band_summary.copy()
        result["test_n"] = 0
        result["test_coverage"] = np.nan
        result["mean_band_width"] = np.nan
        return result

    coverage = (
        prediction_bands.groupby(["crop", "model", "scenario", "lag_config"], as_index=False)
        .agg(
            test_n=("year", "count"),
            test_coverage=("actual_within_band", "mean"),
            mean_band_width=("empirical_abs_error_q80", lambda values: float(2 * values.mean())),
        )
    )
    return band_summary.merge(
        coverage,
        on=["crop", "model", "scenario", "lag_config"],
        how="left",
    )


def _best_params_for_crop(tuned_df: pd.DataFrame, crop: str, model: str) -> dict[str, object] | None:
    subset = tuned_df[
        (tuned_df["crop"] == crop)
        & (tuned_df["model"] == model)
        & (tuned_df["scenario"] == "lag_only")
    ]
    if subset.empty:
        return None
    return json.loads(str(subset.iloc[0]["params_json"]))


def run_feature_group_ablation(
    features_with_climate: pd.DataFrame,
    tuned_df: pd.DataFrame,
    leaderboard: pd.DataFrame,
    output_path: Path,
    *,
    feature_groups: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Evaluate how removing each feature group changes MAE for every crop."""
    rows = []
    feature_groups = tuple(feature_groups or [group for group in FEATURE_GROUPS if group != "climate"])
    test_years = get_years_for_window("test")

    for _, leader in leaderboard.iterrows():
        crop = str(leader["crop"])
        model = str(leader["model"])
        params = _best_params_for_crop(tuned_df, crop, model)
        dataset = prepare_crop_dataset(features_with_climate, crop)
        if dataset is None or params is None:
            continue

        full_eval = evaluate_expanding_window(
            dataset,
            model_name=model,
            scenario="lag_only",
            params=params,
            years=test_years,
            lag_config="L1",
        )
        full_summary = aggregate_predictions(full_eval, group_cols=["crop", "model", "scenario", "lag_config"])
        if full_summary.empty:
            continue
        full_mae = float(full_summary.iloc[0]["mae"])

        for group_name in feature_groups:
            ablated = evaluate_expanding_window(
                dataset,
                model_name=model,
                scenario="lag_only",
                params=params,
                years=test_years,
                lag_config="L1",
                drop_feature_groups=[group_name],
            )
            summary = aggregate_predictions(ablated, group_cols=["crop", "model", "scenario", "lag_config"])
            if summary.empty:
                continue
            row = summary.iloc[0].to_dict()
            rows.append(
                {
                    "crop": crop,
                    "model": model,
                    "scenario": "lag_only",
                    "lag_config": "L1",
                    "feature_group": group_name,
                    "full_mae": full_mae,
                    "ablated_mae": float(row["mae"]),
                    "delta_mae": float(row["mae"]) - full_mae,
                    "n": int(row["n"]),
                }
            )

    result = pd.DataFrame(rows)
    result.to_csv(output_path, index=False)
    return result


def run_prediction_bands(
    features_with_climate: pd.DataFrame,
    tuned_df: pd.DataFrame,
    leaderboard: pd.DataFrame,
    bands_path: Path,
    summary_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create row-level test prediction bands and a compact coverage summary."""
    validation_frames = []
    test_frames = []
    for _, leader in leaderboard.iterrows():
        crop = str(leader["crop"])
        model = str(leader["model"])
        params = _best_params_for_crop(tuned_df, crop, model)
        dataset = prepare_crop_dataset(features_with_climate, crop)
        if dataset is None or params is None:
            continue

        validation_frames.append(
            evaluate_expanding_window(
                dataset,
                model_name=model,
                scenario="lag_only",
                params=params,
                years=get_years_for_window("validation"),
                lag_config="L1",
            )
        )
        test_frames.append(
            evaluate_expanding_window(
                dataset,
                model_name=model,
                scenario="lag_only",
                params=params,
                years=get_years_for_window("test"),
                lag_config="L1",
            )
        )

    validation_predictions = pd.concat(validation_frames, ignore_index=True) if validation_frames else pd.DataFrame()
    test_predictions = pd.concat(test_frames, ignore_index=True) if test_frames else pd.DataFrame()
    band_summary = build_band_summary(validation_predictions)
    prediction_bands = apply_prediction_bands(test_predictions, band_summary)
    coverage_summary = summarise_band_coverage(prediction_bands, band_summary)

    prediction_bands.to_csv(bands_path, index=False)
    coverage_summary.to_csv(summary_path, index=False)
    return prediction_bands, coverage_summary
