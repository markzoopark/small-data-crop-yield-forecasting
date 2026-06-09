"""Run the external multi-region reliability check."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agrostats import baselines, features, io, normalize, train
from agrostats.diagnostics import run_feature_group_ablation, run_prediction_bands
from agrostats.modeling import aggregate_predictions, choose_best_by_crop
from agrostats.multiregion import (
    EXPECTED_REGIONS,
    REGION_LABELS,
    build_data_inventory,
    build_novelty_evidence_table,
    build_region_comparison_summary,
    build_threshold_sensitivity,
    write_multi_region_summary,
)
from agrostats.reliability import build_forecast_cards, select_recommended_methods


RAW_ROOT = ROOT / "data" / "raw" / "agrostats"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
MULTI_REPORTS = REPORTS / "multi_region"


def _add_region(df: pd.DataFrame, slug: str) -> pd.DataFrame:
    result = df.copy()
    result.insert(0, "region_slug", slug)
    result.insert(1, "region_label", REGION_LABELS.get(slug, slug))
    return result


def _save_region_outputs(region_dir: Path, outputs: dict[str, pd.DataFrame]) -> None:
    region_dir.mkdir(parents=True, exist_ok=True)
    for name, df in outputs.items():
        df.to_csv(region_dir / f"{name}.csv", index=False)


def _restore_default_poltava_features() -> None:
    raw_df = io.load_folder(RAW_ROOT / "poltava")
    norm_df = normalize.normalize_units(raw_df)
    features.build_features(norm_df)


def _best_baseline_lookup(baseline_summary: pd.DataFrame) -> pd.DataFrame:
    return (
        baseline_summary.sort_values(["crop", "mae", "rmse", "mape"])
        .groupby("crop", as_index=False)
        .first()[["crop", "baseline"]]
        .rename(columns={"baseline": "best_baseline_raw"})
    )


def _best_model_lookup(leaderboard: pd.DataFrame) -> pd.DataFrame:
    return leaderboard[["crop", "model"]].rename(columns={"model": "best_ml_model_raw"})


def run_region(slug: str) -> dict[str, pd.DataFrame]:
    print(f"Running multi-region check for {REGION_LABELS.get(slug, slug)}")
    raw_df = io.load_folder(RAW_ROOT / slug)
    norm_df = normalize.normalize_units(raw_df)
    features_df = features.build_features(norm_df)

    feature_csv = PROCESSED / f"agrostats_{slug}_features.csv"
    feature_parquet = PROCESSED / f"agrostats_{slug}_features.parquet"
    features_df.to_csv(feature_csv, index=False)
    features_df.to_parquet(feature_parquet, index=False)

    predictions_df, tuned_df = train.train_models(features_df, languages=())
    if predictions_df.empty or tuned_df.empty:
        raise RuntimeError(f"No model outputs were generated for {slug}")

    predictions_export = predictions_df[
        [
            "year",
            "crop",
            "model",
            "scenario",
            "lag_config",
            "split",
            "actual",
            "predicted",
            "n_features",
            "params_json",
        ]
    ].rename(columns={"actual": "y_true", "predicted": "y_pred"})

    metrics_export = predictions_df[
        ["year", "crop", "model", "scenario", "lag_config", "split", "mae", "rmse", "mape", "n_features", "params_json"]
    ].copy()
    test_predictions = predictions_df[predictions_df["split"] == "test"].copy()
    metrics_by_scenario = aggregate_predictions(
        test_predictions,
        group_cols=["scenario", "model", "crop", "lag_config"],
    )
    leaderboard = choose_best_by_crop(metrics_by_scenario[metrics_by_scenario["scenario"] == "lag_only"].copy())

    baseline_metrics = baselines.evaluate_baselines(features_df)
    baseline_summary = baselines.summarise_metrics(baseline_metrics)

    region_dir = MULTI_REPORTS / slug
    region_dir.mkdir(parents=True, exist_ok=True)
    prediction_bands, band_summary = run_prediction_bands(
        features_df,
        tuned_df,
        leaderboard,
        region_dir / "prediction_bands.csv",
        region_dir / "prediction_band_summary.csv",
    )
    ablation = run_feature_group_ablation(
        features_df,
        tuned_df,
        leaderboard,
        region_dir / "feature_group_ablation.csv",
    )
    recommended = select_recommended_methods(leaderboard, baseline_summary, band_summary)
    recommended = recommended.merge(_best_baseline_lookup(baseline_summary), on="crop", how="left")
    recommended = recommended.merge(_best_model_lookup(leaderboard), on="crop", how="left")
    recommended["model"] = recommended["best_ml_model_raw"]
    cards = build_forecast_cards(recommended, ablation, band_summary)

    outputs = {
        "predictions": predictions_export,
        "metrics": metrics_export,
        "metrics_by_scenario": metrics_by_scenario,
        "metrics_leaderboard": leaderboard,
        "metrics_baselines": baseline_metrics,
        "metrics_baselines_summary": baseline_summary,
        "tuned_hyperparameters": tuned_df,
        "prediction_bands": prediction_bands,
        "prediction_band_summary": band_summary,
        "feature_group_ablation": ablation,
        "recommended_methods": recommended,
        "forecast_cards": cards,
    }
    _save_region_outputs(region_dir, outputs)
    return outputs


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    MULTI_REPORTS.mkdir(parents=True, exist_ok=True)

    inventory = build_data_inventory(RAW_ROOT)
    inventory.to_csv(REPORTS / "data_inventory.csv", index=False)
    incomplete = inventory[~inventory["complete"]]
    if not incomplete.empty:
        print(incomplete.to_string(index=False))
        raise SystemExit("Data inventory is incomplete; see reports/data_inventory.csv")

    combined: dict[str, list[pd.DataFrame]] = {
        "recommended": [],
        "cards": [],
        "bands": [],
        "band_summary": [],
        "ablation": [],
    }
    for slug in EXPECTED_REGIONS:
        outputs = run_region(slug)
        combined["recommended"].append(_add_region(outputs["recommended_methods"], slug))
        combined["cards"].append(_add_region(outputs["forecast_cards"], slug))
        combined["bands"].append(_add_region(outputs["prediction_bands"], slug))
        combined["band_summary"].append(_add_region(outputs["prediction_band_summary"], slug))
        combined["ablation"].append(_add_region(outputs["feature_group_ablation"], slug))

    recommended_all = pd.concat(combined["recommended"], ignore_index=True)
    cards_all = pd.concat(combined["cards"], ignore_index=True)
    bands_all = pd.concat(combined["bands"], ignore_index=True)
    band_summary_all = pd.concat(combined["band_summary"], ignore_index=True)
    ablation_all = pd.concat(combined["ablation"], ignore_index=True)

    recommended_all.to_csv(REPORTS / "multi_region_recommended_methods.csv", index=False)
    cards_all.to_csv(REPORTS / "multi_region_forecast_cards.csv", index=False)
    bands_all.to_csv(REPORTS / "multi_region_prediction_bands.csv", index=False)
    band_summary_all.to_csv(REPORTS / "multi_region_prediction_band_summary.csv", index=False)
    ablation_all.to_csv(REPORTS / "multi_region_feature_group_ablation.csv", index=False)

    region_summary = build_region_comparison_summary(cards_all)
    region_summary.to_csv(REPORTS / "region_comparison_summary.csv", index=False)

    threshold_sensitivity = build_threshold_sensitivity(recommended_all)
    threshold_sensitivity.to_csv(REPORTS / "decision_threshold_sensitivity.csv", index=False)

    novelty = build_novelty_evidence_table(cards_all)
    novelty.to_csv(REPORTS / "novelty_evidence_table.csv", index=False)

    write_multi_region_summary(cards_all, REPORTS / "multi_region_reliability_summary.md")
    _restore_default_poltava_features()
    print("Saved multi-region reliability reports to reports/")


if __name__ == "__main__":
    main()
