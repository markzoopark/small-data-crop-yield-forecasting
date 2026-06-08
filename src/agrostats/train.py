"""Leakage-safe model training, tuning, evaluation, and interpretability."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional
import warnings

import matplotlib.pyplot as plt
from matplotlib.text import Text
import numpy as np
import pandas as pd
import shap
import typer
from rich.console import Console

from agrostats import eda, utils
from agrostats.climate import aggregate_power_apr_sep, merge_climate_features
from agrostats.modeling import (
    MODEL_LABELS,
    SCENARIOS,
    TARGET_CROPS,
    aggregate_predictions,
    build_model,
    build_scenario_dataset,
    choose_best_by_crop,
    evaluate_expanding_window,
    get_years_for_window,
    prepare_crop_dataset,
    slugify,
    tune_model,
)


console = Console()
app = typer.Typer(help="Training and evaluation commands for agrostats features.")

warnings.filterwarnings(
    "ignore",
    message="The NumPy global RNG was seeded by calling `np.random.seed`.",
    category=FutureWarning,
)

FEATURES_PATH = Path("data/processed/agrostats_poltava_features.parquet")
METRICS_PATH = Path("reports/metrics.csv")
FIGURES_DIR = Path("reports/figures")
REPORTS_DIR = Path("reports")
TUNED_PARAMS_PATH = REPORTS_DIR / "tuned_hyperparameters.csv"
DISPLAY_SHAP_FEATURES = 5

COMPACT_FEATURE_LABELS = {
    "Yield_t_ha_lag1": "Yield (t-1)",
    "Area_ha_lag1": "Crop area (t-1)",
    "N_kg_ha_lag1": "N fertiliser (t-1)",
    "P2O5_kg_ha_lag1": "P2O5 (t-1)",
    "K_kg_ha_lag1": "K fertiliser (t-1)",
    "Irrig_mm_lag1": "Irrigation (t-1)",
    "Irrig_m3_ha_lag1": "Irrigation (t-1)",
    "Org_kg_ha_or_share_lag1": "Organic input (t-1)",
    "Mineral_treated_share_lag1": "Mineral share (t-1)",
    "ma5_Yield": "Yield MA5",
    "ma5_Org": "Organic MA5",
    "ma5_K": "K MA5",
    "ma5_N": "N MA5",
    "ma5_P2O5": "P2O5 MA5",
    "ma5_Irrig_m3": "Irrigation MA5",
    "ma5_Irrig_mm": "Irrigation MA5",
    "ma5_MineralShare": "Mineral share MA5",
}

TRAIN_LANG = {
    "uk": {
        "legend_actual": "Факт",
        "legend_pred": "Прогноз",
        "year_label": "Рік",
        "yield_label": "Урожайність, т/га",
        "actual_title": "Тестовий прогноз — {crop} ({model})",
        "scatter_x": "Факт (т/га)",
        "scatter_y": "Прогноз (т/га)",
        "scatter_title": "{crop} ({model}) — MAE={mae:.2f}, MAPE={mape:.1f}%",
        "shap_title": "{crop} ({model})",
    },
    "en": {
        "legend_actual": "Actual",
        "legend_pred": "Forecast",
        "year_label": "Year",
        "yield_label": "Yield, t/ha",
        "actual_title": "Test forecast — {crop} ({model})",
        "scatter_x": "Actual (t/ha)",
        "scatter_y": "Forecast (t/ha)",
        "scatter_title": "{crop} ({model}) — MAE={mae:.2f}, MAPE={mape:.1f}%",
        "shap_title": "{crop} ({model})",
    },
}

CROP_NAME_TRANSLATIONS = {
    "uk": {
        "Пшениця": "Пшениця",
        "Кукурудза": "Кукурудза",
        "Соняшник": "Соняшник",
    },
    "en": {
        "Пшениця": "Wheat",
        "Кукурудза": "Corn",
        "Соняшник": "Sunflower",
    },
}


def load_features(path: Path = FEATURES_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Feature file not found: {path}. Generate features before training.")
    return pd.read_parquet(path)


def _crop_label(crop: str, language: str) -> str:
    return CROP_NAME_TRANSLATIONS.get(language, {}).get(crop, crop)


def _compact_feature_label(feature: str) -> str:
    return COMPACT_FEATURE_LABELS.get(feature, feature.replace("_", " "))


def plot_actual_vs_predicted(
    predictions_df: pd.DataFrame,
    crop: str,
    model: str,
    scenario: str,
    language: str,
) -> None:
    subset = predictions_df[
        (predictions_df["crop"] == crop)
        & (predictions_df["model"] == model)
        & (predictions_df["scenario"] == scenario)
        & (predictions_df["split"] == "test")
    ].sort_values("year")
    if subset.empty:
        return

    cfg = TRAIN_LANG[language]
    crop_label = _crop_label(crop, language)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(subset["year"], subset["actual"], marker="o", linewidth=2.0, label=cfg["legend_actual"])
    ax.plot(
        subset["year"],
        subset["predicted"],
        marker="o",
        linewidth=2.0,
        linestyle="--",
        label=cfg["legend_pred"],
    )
    ax.set_title(cfg["actual_title"].format(crop=crop_label, model=MODEL_LABELS.get(model, model)))
    ax.set_xlabel(cfg["year_label"])
    ax.set_ylabel(cfg["yield_label"])
    ax.set_xticks(subset["year"].tolist())
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig_dir = FIGURES_DIR / language
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / f"{model}_{slugify(crop)}_{scenario}_actual_vs_pred.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_scatter_actual_pred(
    predictions_df: pd.DataFrame,
    crop: str,
    model: str,
    scenario: str,
    language: str,
) -> None:
    subset = predictions_df[
        (predictions_df["crop"] == crop)
        & (predictions_df["model"] == model)
        & (predictions_df["scenario"] == scenario)
        & (predictions_df["split"] == "test")
    ]
    if subset.empty:
        return

    cfg = TRAIN_LANG[language]
    crop_label = _crop_label(crop, language)
    y_true = subset["actual"].to_numpy()
    y_pred = subset["predicted"].to_numpy()
    mae = float(np.mean(np.abs(y_true - y_pred)))
    mape = float((np.abs((y_true - y_pred) / y_true) * 100).mean())
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    line = np.linspace(min_val, max_val, 100)

    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    ax.scatter(y_true, y_pred, color="#1f77b4", edgecolor="black", s=80)
    ax.plot(line, line, color="#d62728", linestyle="--", linewidth=1.5, label="y = x")
    ax.set_xlabel(cfg["scatter_x"])
    ax.set_ylabel(cfg["scatter_y"])
    ax.set_title(cfg["scatter_title"].format(crop=crop_label, model=MODEL_LABELS.get(model, model), mae=mae, mape=mape))
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig_dir = FIGURES_DIR / language
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / f"scatter_{model}_{slugify(crop)}_{scenario}_test.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _prepare_shap_artifacts(
    features_df: pd.DataFrame,
    crop: str,
    model_name: str,
    scenario: str,
    params: dict[str, object],
) -> tuple[object, pd.DataFrame, pd.DataFrame] | None:
    crop_dataset = prepare_crop_dataset(features_df, crop)
    if crop_dataset is None:
        return None
    scenario_dataset = build_scenario_dataset(crop_dataset, scenario, lag_config="L1", include_climate=False)
    if scenario_dataset is None:
        return None

    years = scenario_dataset.years
    train_mask = years < 2022
    test_mask = years >= 2022
    if not train_mask.any() or not test_mask.any():
        return None

    X_train = scenario_dataset.features.loc[train_mask].copy()
    y_train = scenario_dataset.target.loc[train_mask].copy()
    X_test = scenario_dataset.features.loc[test_mask].copy()
    valid_cols = [col for col in X_train.columns if X_train[col].notna().any() and X_train[col].nunique(dropna=True) > 1]
    if not valid_cols:
        return None

    X_train = X_train[valid_cols]
    X_test = X_test[valid_cols]
    model = build_model(model_name, params=params)
    model.fit(X_train, y_train)
    return model, X_train, X_test


def plot_shap_summary(
    model: object,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    crop: str,
    model_name: str,
    scenario: str,
    language: str,
) -> None:
    crop_label = _crop_label(crop, language)
    title = TRAIN_LANG[language]["shap_title"].format(crop=crop_label, model=MODEL_LABELS.get(model_name, model_name))

    explainer = shap.Explainer(model.predict, X_train)
    shap_values = explainer(X_test)
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    feature_importance = (
        pd.DataFrame({"feature": X_test.columns, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .head(10)
    )
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    feature_importance.to_csv(
        REPORTS_DIR / f"shap_top_{model_name}_{slugify(crop)}_{scenario}.csv",
        index=False,
    )

    display_features = feature_importance[feature_importance["mean_abs_shap"] > 1e-6]["feature"].tolist()
    if not display_features:
        display_features = feature_importance["feature"].tolist()
    display_features = display_features[:DISPLAY_SHAP_FEATURES]
    shap_subset = shap.Explanation(
        values=shap_values.values[:, [X_test.columns.get_loc(col) for col in display_features]],
        base_values=shap_values.base_values,
        data=X_test[display_features].to_numpy(),
        feature_names=[_compact_feature_label(col) for col in display_features],
    )

    plt.figure(figsize=(16.0, 11.6))
    shap.summary_plot(shap_subset, show=False, plot_size=None)
    fig = plt.gcf()
    fig.set_size_inches(28.5, 20.8)
    axes = fig.axes
    main_ax = axes[0]
    main_ax.tick_params(axis="x", labelsize=56)
    main_ax.tick_params(axis="y", pad=22)
    main_ax.xaxis.label.set_size(64)
    main_ax.yaxis.label.set_size(52)
    main_ax.set_xlabel("SHAP value (impact on forecast)", fontsize=64)
    main_ax.set_title(title, fontsize=68, pad=44)
    for label in main_ax.get_yticklabels():
        label.set_fontsize(74)
        label.set_fontweight("medium")
    for label in main_ax.get_xticklabels():
        label.set_fontsize(56)
    for collection in main_ax.collections:
        offsets = getattr(collection, "get_offsets", lambda: None)()
        if offsets is not None and len(offsets):
            collection.set_sizes(np.full(len(offsets), 760.0))
    for line in main_ax.lines:
        line.set_linewidth(4.6)
        line.set_alpha(0.9)
    if len(axes) > 1:
        cbar_ax = axes[1]
        cbar_ax.tick_params(labelsize=54)
        cbar_ax.yaxis.label.set_size(54)
        for label in cbar_ax.get_yticklabels():
            label.set_fontsize(54)
    for text in fig.findobj(match=Text):
        value = text.get_text().strip()
        if not value:
            continue
        if value in shap_subset.feature_names:
            text.set_fontsize(78)
            text.set_fontweight("medium")
        elif value == title:
            text.set_fontsize(72)
        elif value == "SHAP value (impact on forecast)":
            text.set_fontsize(68)
        elif value == "Feature value":
            text.set_fontsize(60)
        elif value in {"High", "Low"}:
            text.set_fontsize(60)
        else:
            text.set_fontsize(max(float(text.get_fontsize()), 58))
    plt.tight_layout(pad=1.35)
    fig_dir = FIGURES_DIR / language
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_dir / f"shap_{model_name}_{slugify(crop)}_{scenario}.png", dpi=360, bbox_inches="tight")
    plt.close()


def train_models(
    features_df: pd.DataFrame,
    *,
    languages: Iterable[str] = ("uk", "en"),
    models: Iterable[str] = ("elasticnet", "xgboost", "lightgbm"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_predictions: list[pd.DataFrame] = []
    tuned_rows: list[dict[str, object]] = []
    all_years = get_years_for_window("validation") + get_years_for_window("test")

    for crop in TARGET_CROPS:
        dataset = prepare_crop_dataset(features_df, crop)
        if dataset is None:
            console.log(f"[yellow]No usable dataset for {crop}; skipping.[/yellow]")
            continue

        for scenario in SCENARIOS:
            for model_name in models:
                console.print(f"[cyan]Tuning {model_name} — {crop} — {scenario}[/cyan]")
                best_params, tuning_df = tune_model(dataset, model_name=model_name, scenario=scenario, lag_config="L1")
                if tuning_df.empty:
                    console.log(f"[yellow]No tuning results for {model_name}/{crop}/{scenario}[/yellow]")
                    continue
                best_validation = tuning_df.sort_values(["mae", "rmse", "mape"]).iloc[0].to_dict()
                tuned_rows.append(
                    {
                        "crop": crop,
                        "scenario": scenario,
                        "lag_config": "L1",
                        "model": model_name,
                        "params_json": json.dumps(best_params, ensure_ascii=False),
                        "validation_mae": best_validation["mae"],
                        "validation_rmse": best_validation["rmse"],
                        "validation_mape": best_validation["mape"],
                    }
                )
                pred_df = evaluate_expanding_window(
                    dataset,
                    model_name=model_name,
                    scenario=scenario,
                    params=best_params,
                    years=all_years,
                    lag_config="L1",
                )
                if pred_df.empty:
                    continue
                all_predictions.append(pred_df)

                for language in languages:
                    plot_actual_vs_predicted(pred_df, crop, model_name, scenario, language)
                    plot_scatter_actual_pred(pred_df, crop, model_name, scenario, language)

                shap_artifacts = _prepare_shap_artifacts(features_df, crop, model_name, scenario, best_params)
                if shap_artifacts is not None:
                    model_obj, X_train, X_test = shap_artifacts
                    for language in languages:
                        plot_shap_summary(model_obj, X_train, X_test, crop, model_name, scenario, language)

    predictions_df = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    tuned_df = pd.DataFrame(tuned_rows)
    return predictions_df, tuned_df


@app.command("poltava")
def poltava_command(
    features_path: Path = typer.Option(FEATURES_PATH, exists=True, help="Path to the feature parquet file."),
    languages: str = typer.Option("uk,en", help="Comma-separated list of languages for plot generation (uk,en)."),
) -> None:
    """Run tuned model training/evaluation for the Poltava dataset."""
    language_list = [lang.strip() for lang in languages.split(",") if lang.strip()]
    if not language_list:
        language_list = ["uk"]
    for language in language_list:
        if language not in TRAIN_LANG:
            raise ValueError(f"Unsupported language: {language}")

    features_df = load_features(features_path)
    climate_summary = aggregate_power_apr_sep()
    features_with_climate = merge_climate_features(features_df, climate_summary)
    predictions_df, tuned_df = train_models(features_with_climate, languages=language_list)
    if predictions_df.empty:
        console.print("[red]Unable to compute metrics — please verify the data.[/red]")
        return

    utils.ensure_directories([REPORTS_DIR, METRICS_PATH.parent])

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
    predictions_path = REPORTS_DIR / "predictions.csv"
    predictions_export.to_csv(predictions_path, index=False)
    console.print(f"[green]Predictions saved to {predictions_path}[/green]")

    metrics_export = predictions_df[
        ["year", "crop", "model", "scenario", "lag_config", "split", "mae", "rmse", "mape", "n_features", "params_json"]
    ].copy()
    metrics_export.to_csv(METRICS_PATH, index=False)
    console.print(f"[green]Metrics saved to {METRICS_PATH}[/green]")

    tuned_df.to_csv(TUNED_PARAMS_PATH, index=False)
    console.print(f"[green]Tuned hyperparameters saved to {TUNED_PARAMS_PATH}[/green]")

    test_predictions = predictions_df[predictions_df["split"] == "test"].copy()
    summary = aggregate_predictions(
        test_predictions,
        group_cols=["scenario", "model", "crop", "lag_config"],
    )
    summary_path = REPORTS_DIR / "metrics_by_scenario.csv"
    summary.to_csv(summary_path, index=False)
    console.print(f"[green]Scenario summary saved to {summary_path}[/green]")

    lag_only_summary = summary[summary["scenario"] == "lag_only"].copy()
    leaderboard = choose_best_by_crop(lag_only_summary)
    leaderboard_path = REPORTS_DIR / "metrics_leaderboard.csv"
    leaderboard.to_csv(leaderboard_path, index=False)
    console.print(f"[green]Leaderboard saved to {leaderboard_path}[/green]")

    try:
        eda.plot_trends(features_df, eda.TARGET_CROPS, language_list)
    except Exception as exc:  # noqa: BLE001
        console.log(f"[yellow]Failed to render trend plots: {exc}[/yellow]")


if __name__ == "__main__":
    app()
