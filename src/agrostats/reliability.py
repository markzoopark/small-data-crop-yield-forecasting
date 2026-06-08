"""Baseline-first model selection and reliability summaries."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PRACTICAL_MAE_MARGIN = 0.05
RECOMMENDED_METHODS_PATH = Path("reports/recommended_methods.csv")
FORECAST_CARDS_PATH = Path("reports/forecast_cards.csv")
RELIABILITY_SUMMARY_PATH = Path("reports/reliability_summary.md")

MODEL_LABELS = {
    "elasticnet": "ElasticNet",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
}

BASELINE_LABELS = {
    "forecast_linear": "FORECAST.LINEAR",
    "naive_lag1": "Naive lag-1",
    "linest_lag_only": "LINEST + lags",
    "arima": "ARIMA",
}

GROUP_LABELS = {
    "yield_history": "yield history",
    "area": "crop area",
    "fertiliser_n": "nitrogen fertiliser",
    "fertiliser_p": "phosphorus fertiliser",
    "fertiliser_k": "potassium fertiliser",
    "mineral_share": "mineral-treated share",
    "organics": "organic input",
    "irrigation": "irrigation",
}

CROP_LABELS = {
    "Пшениця": "wheat",
    "Кукурудза": "maize",
    "Соняшник": "sunflower",
}


def select_recommended_methods(
    leaderboard: pd.DataFrame,
    baseline_summary: pd.DataFrame,
    band_summary: pd.DataFrame,
    *,
    practical_margin: float = PRACTICAL_MAE_MARGIN,
) -> pd.DataFrame:
    """Choose ML only when it beats the best baseline by a practical margin."""
    rows = []
    for _, ml_row in leaderboard.iterrows():
        crop = ml_row["crop"]
        crop_baselines = baseline_summary[baseline_summary["crop"] == crop]
        if crop_baselines.empty:
            continue
        best_baseline = crop_baselines.sort_values(["mae", "rmse", "mape"]).iloc[0]
        crop_band = band_summary[band_summary["crop"] == crop]
        test_coverage = float(crop_band.iloc[0]["test_coverage"]) if not crop_band.empty else float("nan")

        ml_mae = float(ml_row["mae"])
        baseline_mae = float(best_baseline["mae"])
        ml_gain = baseline_mae - ml_mae
        use_ml = ml_gain >= practical_margin
        if use_ml:
            recommended_type = "machine_learning"
            recommended_method = MODEL_LABELS.get(str(ml_row["model"]), str(ml_row["model"]))
            recommended_mae = ml_mae
            if test_coverage >= 2 / 3:
                warning_label = "within expected error"
            else:
                warning_label = "outside validation error scale"
        else:
            recommended_type = "baseline"
            recommended_method = BASELINE_LABELS.get(str(best_baseline["baseline"]), str(best_baseline["baseline"]))
            recommended_mae = baseline_mae
            warning_label = "baseline safer"

        best_baseline_label = BASELINE_LABELS.get(str(best_baseline["baseline"]), str(best_baseline["baseline"]))
        rows.append(
            {
                "crop": crop,
                "crop_en": CROP_LABELS.get(crop, crop),
                "recommended_type": recommended_type,
                "recommended_method": recommended_method,
                "recommended_mae": recommended_mae,
                "best_ml_model": MODEL_LABELS.get(str(ml_row["model"]), str(ml_row["model"])),
                "best_ml_mae": ml_mae,
                "best_baseline": best_baseline_label,
                "best_baseline_method": best_baseline_label,
                "best_baseline_mae": baseline_mae,
                "ml_gain_vs_baseline": ml_gain,
                "practical_margin": practical_margin,
                "test_coverage": test_coverage,
                "warning_label": warning_label,
            }
        )
    return pd.DataFrame(rows).sort_values("crop_en").reset_index(drop=True)


def _top_feature_group(feature_ablation: pd.DataFrame, crop: str) -> tuple[str, float]:
    subset = feature_ablation[feature_ablation["crop"] == crop]
    if subset.empty:
        return "", float("nan")
    row = subset.sort_values("delta_mae", ascending=False).iloc[0]
    return GROUP_LABELS.get(str(row["feature_group"]), str(row["feature_group"])), float(row["delta_mae"])


def _interpretation(row: pd.Series, top_group: str) -> str:
    crop = row["crop_en"]
    if row["recommended_type"] == "baseline":
        return (
            f"For {crop}, the transparent baseline is recommended because the selected ML model "
            "does not clear the practical improvement margin. Treat this crop as the negative/control case."
        )
    if row["warning_label"] == "outside validation error scale":
        return (
            f"For {crop}, ML improves MAE, but the test errors fall outside the small validation-residual "
            f"scale too often. Use the forecast with caution; the strongest diagnostic group is {top_group}."
        )
    return (
        f"For {crop}, ML clears the baseline-first rule and the test errors are mostly within the empirical "
        f"validation-residual band. The strongest diagnostic group is {top_group}."
    )


def build_forecast_cards(
    recommended_methods: pd.DataFrame,
    feature_ablation: pd.DataFrame,
    band_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Create one compact decision-support row per crop."""
    rows = []
    for _, row in recommended_methods.iterrows():
        crop = row["crop"]
        top_group, top_delta = _top_feature_group(feature_ablation, crop)
        crop_band = band_summary[band_summary["crop"] == crop]
        band_width = float(crop_band.iloc[0]["mean_band_width"]) if not crop_band.empty else float("nan")
        card = row.to_dict()
        card["top_feature_group"] = top_group
        card["top_feature_group_delta_mae"] = top_delta
        card["mean_empirical_band_width"] = band_width
        card["interpretation"] = _interpretation(row, top_group)
        rows.append(card)
    return pd.DataFrame(rows)


def write_reliability_summary(forecast_cards: pd.DataFrame, output_path: Path = RELIABILITY_SUMMARY_PATH) -> None:
    """Write a simple human-readable Markdown summary."""
    lines = [
        "# Reliability summary",
        "",
        "This is a small-data benchmark, not a production forecasting system.",
        "The workflow recommends ML only when it beats the best simple baseline by the configured practical MAE margin.",
        "",
    ]
    for _, row in forecast_cards.iterrows():
        lines.extend(
            [
                f"## {row['crop_en'].title()}",
                "",
                f"- Recommended method: {row['recommended_method']} ({row['recommended_type']})",
                f"- Recommended MAE: {row['recommended_mae']:.2f} t/ha",
                f"- Best ML MAE: {row['best_ml_mae']:.2f} t/ha",
                f"- Best baseline MAE: {row['best_baseline_mae']:.2f} t/ha",
                f"- ML gain vs baseline: {row['ml_gain_vs_baseline']:.2f} t/ha",
                f"- Test coverage inside validation-residual band: {row['test_coverage']:.1%}",
                f"- Warning label: {row['warning_label']}",
                f"- Top useful feature group: {row['top_feature_group']}",
                "",
                str(row["interpretation"]),
                "",
            ]
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_reliability_outputs(
    leaderboard_path: Path = Path("reports/metrics_leaderboard.csv"),
    baseline_summary_path: Path = Path("reports/metrics_baselines_summary.csv"),
    band_summary_path: Path = Path("reports/prediction_band_summary.csv"),
    feature_ablation_path: Path = Path("reports/feature_group_ablation.csv"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate recommended methods, forecast cards, and Markdown summary."""
    leaderboard = pd.read_csv(leaderboard_path)
    baselines = pd.read_csv(baseline_summary_path)
    bands = pd.read_csv(band_summary_path)
    ablation = pd.read_csv(feature_ablation_path)
    recommended = select_recommended_methods(leaderboard, baselines, bands)
    cards = build_forecast_cards(recommended, ablation, bands)
    RECOMMENDED_METHODS_PATH.parent.mkdir(parents=True, exist_ok=True)
    recommended.to_csv(RECOMMENDED_METHODS_PATH, index=False)
    cards.to_csv(FORECAST_CARDS_PATH, index=False)
    write_reliability_summary(cards, RELIABILITY_SUMMARY_PATH)
    return recommended, cards
