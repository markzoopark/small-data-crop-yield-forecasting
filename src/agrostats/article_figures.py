"""Generate publication-ready figures for the revised manuscript."""

from __future__ import annotations

import shutil
import textwrap
from pathlib import Path
from typing import Dict, Iterable

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
from string import ascii_lowercase


BASE_DIR = Path(__file__).resolve().parents[2]
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures_article"
FIGURES_BASE_DIR = REPORTS_DIR / "figures"
ARTICLE_DIR = BASE_DIR / "paper"
SUBMISSION_FIGURES_DIR = ARTICLE_DIR / "submission_figures_en"
PROCESSED_FEATURES = BASE_DIR / "data" / "processed" / "agrostats_poltava_features.parquet"
PREDICTIONS_CSV = REPORTS_DIR / "predictions.csv"
METRICS_BY_SCENARIO_CSV = REPORTS_DIR / "metrics_by_scenario.csv"
BASELINE_SUMMARY_CSV = REPORTS_DIR / "metrics_baselines_summary.csv"
LAG_SENSITIVITY_CSV = REPORTS_DIR / "lag_sensitivity.csv"
CLIMATE_SENSITIVITY_CSV = REPORTS_DIR / "climate_sensitivity.csv"
CORR_TEMPLATE = REPORTS_DIR / "correlations_{crop}.csv"

TARGET_CROPS = ("Пшениця", "Кукурудза", "Соняшник")
CROP_SLUG = {
    "Пшениця": "pshenytsia",
    "Кукурудза": "kukurudza",
    "Соняшник": "sonyashnyk",
}

LANG_CONFIG: Dict[str, Dict[str, str | dict[str, str]]] = {
    "uk": {
        "crop_names": {"Пшениця": "Пшениця", "Кукурудза": "Кукурудза", "Соняшник": "Соняшник"},
        "year": "Рік",
        "yield": "Урожайність, т/га",
        "actual": "Факт",
        "pred": "Прогноз",
        "scatter_x": "Факт (т/га)",
        "scatter_y": "Прогноз (т/га)",
        "heatmap_title": "Кореляції лагових факторів – {crop}",
        "heatmap_cb": "Pearson r",
        "baseline_title": "Порівняння базових прогнозних методів і ML",
        "baseline_ylabel": "MAE, т/га",
        "lag_title": "Чутливість до структури лагів",
        "lag_ylabel": "MAE, т/га",
        "climate_title": "Експеримент чутливості до кліматичних ознак",
        "climate_ylabel": "MAE, т/га",
        "climate_labels": ("Лише агростатистика", "Агростатистика + клімат"),
        "legend": "Культура",
        "trends": {
            "Yield_t_ha": ("Урожайність", "т/га"),
            "Area_ha": ("Площа культури", "га"),
            "N_kg_ha": ("Азотні добрива", "кг/га"),
            "P2O5_kg_ha": ("Фосфорні добрива", "кг/га"),
            "K_kg_ha": ("Калійні добрива", "кг/га"),
            "Irrig_mm": ("Зрошення", "мм"),
        },
    },
    "en": {
        "crop_names": {"Пшениця": "Wheat", "Кукурудза": "Corn", "Соняшник": "Sunflower"},
        "year": "Year",
        "yield": "Yield, t/ha",
        "actual": "Actual",
        "pred": "Forecast",
        "scatter_x": "Actual (t/ha)",
        "scatter_y": "Forecast (t/ha)",
        "heatmap_title": "Lag-factor correlations – {crop}",
        "heatmap_cb": "Pearson r",
        "baseline_title": "Forecasting baselines versus tuned ML",
        "baseline_ylabel": "MAE, t/ha",
        "lag_title": "Sensitivity to lag structure",
        "lag_ylabel": "MAE, t/ha",
        "climate_title": "Climate sensitivity experiment",
        "climate_ylabel": "MAE, t/ha",
        "climate_labels": ("Agro-only", "Agro+climate"),
        "legend": "Crop",
        "trends": {
            "Yield_t_ha": ("Yield", "t/ha"),
            "Area_ha": ("Crop area", "ha"),
            "N_kg_ha": ("Nitrogen fertilisers", "kg/ha"),
            "P2O5_kg_ha": ("Phosphorus fertilisers", "kg/ha"),
            "K_kg_ha": ("Potassium fertilisers", "kg/ha"),
            "Irrig_mm": ("Irrigation", "mm"),
        },
    },
}

TREND_COLUMNS = ["Yield_t_ha", "Area_ha", "N_kg_ha", "P2O5_kg_ha", "K_kg_ha", "Irrig_mm"]
BASELINE_LABELS = {
    "naive_lag1": {"uk": "Naive (t-1)", "en": "Naive (t-1)"},
    "forecast_linear": {"uk": "Лінійний тренд", "en": "Linear trend"},
    "linest_lag_only": {"uk": "LINEST + лаги", "en": "LINEST + lags"},
    "arima": {"uk": "ARIMA", "en": "ARIMA"},
}
MODEL_LABELS = {"elasticnet": "ElasticNet", "xgboost": "XGBoost", "lightgbm": "LightGBM"}
PANEL_LABELS = ("a", "b", "c")
ALL_PANEL_LABELS = tuple(ascii_lowercase)
MANUSCRIPT_IMAGE_WIDTH_IN = 6.3
A4_PAGE_SIZE_IN = (8.27, 11.69)

TITLE_SIZE = 30
LABEL_SIZE = 25
TICK_SIZE = 23
LEGEND_SIZE = 23
PANEL_LABEL_SIZE = 30
COMPOSITE_PANEL_LABEL_SIZE = 32
ANNOTATION_SIZE = 21
BASELINE_XTICK_SIZE = 25
BASELINE_VALUE_SIZE = 21

FIGURE_CAPTIONS = {
    1: "Figure 1. Long-term trends of yield, crop area, mineral fertilizer use (N, P2O5, K2O), and irrigation for wheat, maize, and sunflower in Poltava region (2010-2024). Each panel shows one indicator; coloured lines correspond to crops.",
    2: "Figure 2. Pearson correlation between yield and lagged agronomic factors for (a) wheat, (b) maize, and (c) sunflower. Colour scale shows the correlation coefficient; numbers in cells are r values.",
    3: "Figure 3. Actual and forecast yield for the best machine-learning models under the lag-only scenario: (a) wheat (ElasticNet), (b) maize (LightGBM), and (c) sunflower (LightGBM) over the 2022-2024 test period.",
    4: "Figure 4. Scatter plots of observed versus forecast yield for the best machine-learning models in 2022-2024: (a) wheat (ElasticNet), (b) maize (LightGBM), and (c) sunflower (LightGBM). Points closer to the y = x line indicate higher accuracy.",
    5: "Figure 5. Comparison of MAE for machine-learning models and baseline forecasting methods (Naive, FORECAST.LINEAR, LINEST, ARIMA) on the 2022-2024 test period for wheat, maize, and sunflower. Lower values indicate more accurate forecasts.",
    6: "Figure 6. Colour SHAP plots for the best machine-learning models under the lag-only scenario: (a) wheat (ElasticNet), (b) maize (LightGBM), and (c) sunflower (LightGBM). Colour indicates low versus high feature values, while the position on the x-axis indicates negative or positive contribution to the forecast.",
}

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


def ensure_dirs(languages: Iterable[str]) -> dict[str, Path]:
    result = {}
    for lang in languages:
        out_dir = FIGURES_DIR / lang
        out_dir.mkdir(parents=True, exist_ok=True)
        result[lang] = out_dir
    return result


def style_matplotlib() -> None:
    style_name = "seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "classic"
    plt.style.use(style_name)
    plt.rcParams.update(
        {
            "axes.titlesize": TITLE_SIZE,
            "axes.labelsize": LABEL_SIZE,
            "xtick.labelsize": TICK_SIZE,
            "ytick.labelsize": TICK_SIZE,
            "legend.fontsize": LEGEND_SIZE,
            "figure.dpi": 240,
            "axes.titlepad": 14,
        }
    )


def add_panel_labels(
    axes: Iterable[plt.Axes],
    *,
    start_index: int = 0,
    x: float = -0.1,
    y: float = 1.06,
    fontsize: int = PANEL_LABEL_SIZE,
) -> None:
    for offset, ax in enumerate(axes):
        label = ALL_PANEL_LABELS[start_index + offset]
        ax.text(
            x,
            y,
            f"{label})",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=fontsize,
            fontweight="bold",
            clip_on=False,
        )


def crop_label(crop: str, lang: str) -> str:
    return str(LANG_CONFIG[lang]["crop_names"][crop])


def compact_feature_label(label: str) -> str:
    return COMPACT_FEATURE_LABELS.get(label, label.replace("_", " "))


def trim_image_whitespace(image: np.ndarray, *, threshold: float = 0.985, padding: int = 6) -> np.ndarray:
    rgb = image[..., :3] if image.ndim == 3 else image
    mask = np.any(rgb < threshold, axis=-1) if rgb.ndim == 3 else rgb < threshold
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return image
    r0 = max(int(rows[0]) - padding, 0)
    r1 = min(int(rows[-1]) + padding + 1, image.shape[0])
    c0 = max(int(cols[0]) - padding, 0)
    c1 = min(int(cols[-1]) + padding + 1, image.shape[1])
    return image[r0:r1, c0:c1]


def render_a4_preview(image_path: Path, preview_path: Path, caption: str) -> None:
    page_w, page_h = A4_PAGE_SIZE_IN
    image = mpimg.imread(image_path)
    img_h, img_w = image.shape[:2]
    target_w_frac = MANUSCRIPT_IMAGE_WIDTH_IN / page_w
    target_h_frac = target_w_frac * (img_h / img_w) * (page_w / page_h)

    fig = plt.figure(figsize=A4_PAGE_SIZE_IN, dpi=220, facecolor="white")
    top_margin = 0.08
    left_margin = 0.12
    caption_gap = 0.035
    caption_height = 0.165
    max_image_h_frac = 1 - top_margin - caption_gap - caption_height - 0.08
    image_h_frac = min(target_h_frac, max_image_h_frac)

    ax = fig.add_axes([left_margin, 1 - top_margin - image_h_frac, target_w_frac, image_h_frac])
    ax.imshow(image)
    ax.axis("off")

    wrapped_caption = textwrap.fill(caption, width=105)
    fig.text(
        left_margin,
        1 - top_margin - image_h_frac - caption_gap,
        wrapped_caption,
        ha="left",
        va="top",
        fontsize=13.5,
        color="black",
    )
    fig.savefig(preview_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_trends(features: pd.DataFrame, languages: Iterable[str]) -> None:
    dirs = ensure_dirs(languages)
    subset = features[features["group_or_crop"].isin(TARGET_CROPS)].sort_values("year")
    tick_years = [2010, 2014, 2018, 2022, 2024]
    for lang in languages:
        cfg = LANG_CONFIG[lang]
        fig, axes = plt.subplots(2, 3, figsize=(24.2, 15.8))
        axes = axes.flatten()
        for ax, column in zip(axes, TREND_COLUMNS):
            title, unit = cfg["trends"][column]
            for crop in TARGET_CROPS:
                crop_df = subset[subset["group_or_crop"] == crop]
                ax.plot(crop_df["year"], crop_df[column], marker="o", linewidth=4.2, markersize=11.0, label=crop_label(crop, lang))
            ax.set_title(title, fontsize=TITLE_SIZE + 2)
            ax.set_xlabel(str(cfg["year"]), fontsize=LABEL_SIZE + 2)
            ax.set_ylabel(unit, fontsize=LABEL_SIZE + 2)
            ax.set_xticks(tick_years)
            ax.tick_params(axis="both", labelsize=TICK_SIZE + 2)
            ax.tick_params(axis="x", labelrotation=25)
            for tick_label in ax.get_xticklabels():
                tick_label.set_horizontalalignment("right")
            ax.grid(alpha=0.3)
        add_panel_labels(axes, x=-0.16, y=1.06, fontsize=PANEL_LABEL_SIZE)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.01),
            frameon=False,
            ncol=3,
            title=str(cfg["legend"]),
            handlelength=2.3,
            columnspacing=1.8,
            fontsize=LEGEND_SIZE + 2,
            title_fontsize=LEGEND_SIZE + 2,
        )
        fig.tight_layout(rect=(0.02, 0.045, 0.985, 0.91), w_pad=1.0, h_pad=1.0)
        fig.savefig(dirs[lang] / "poltava_trends.png", bbox_inches="tight")
        plt.close(fig)


def plot_prediction_series(predictions: pd.DataFrame, crop: str, model: str, lang: str) -> None:
    cfg = LANG_CONFIG[lang]
    subset = predictions[
        (predictions["crop"] == crop)
        & (predictions["model"] == model)
        & (predictions["scenario"] == "lag_only")
        & (predictions["split"] == "test")
    ].sort_values("year")
    if subset.empty:
        return
    fig, ax = plt.subplots(figsize=(18.4, 14.8))
    ax.plot(subset["year"], subset["y_true"], marker="o", linewidth=5.8, markersize=17.0, label=str(cfg["actual"]))
    ax.plot(subset["year"], subset["y_pred"], marker="o", linewidth=5.8, markersize=17.0, linestyle="--", label=str(cfg["pred"]))
    ax.set_title(f"{crop_label(crop, lang)} ({MODEL_LABELS.get(model, model)})", pad=20, fontsize=58)
    ax.set_xlabel(str(cfg["year"]), fontsize=48)
    ax.set_ylabel(str(cfg["yield"]), fontsize=48)
    ax.set_xticks(subset["year"].tolist())
    ax.tick_params(axis="both", labelsize=42)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, loc="upper left", handlelength=3.0, fontsize=40)
    fig.tight_layout(pad=1.6)
    fig.savefig(FIGURES_DIR / lang / f"line_{model}_{CROP_SLUG[crop]}_lag_only.png", bbox_inches="tight")
    plt.close(fig)


def plot_prediction_scatter(predictions: pd.DataFrame, crop: str, model: str, lang: str) -> None:
    cfg = LANG_CONFIG[lang]
    subset = predictions[
        (predictions["crop"] == crop)
        & (predictions["model"] == model)
        & (predictions["scenario"] == "lag_only")
        & (predictions["split"] == "test")
    ]
    if subset.empty:
        return
    y_true = subset["y_true"].to_numpy()
    y_pred = subset["y_pred"].to_numpy()
    mae = float(np.abs(y_true - y_pred).mean())
    mape = float((np.abs((y_true - y_pred) / y_true) * 100).mean())
    line = np.linspace(min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max()), 100)
    fig, ax = plt.subplots(figsize=(12.8, 12.6))
    ax.scatter(y_true, y_pred, color="#1f77b4", edgecolor="black", s=460)
    ax.plot(line, line, color="#d62728", linestyle="--", linewidth=3.4, label="y = x")
    ax.set_xlabel(str(cfg["scatter_x"]), fontsize=46)
    ax.set_ylabel(str(cfg["scatter_y"]), fontsize=46)
    ax.set_title(crop_label(crop, lang), y=1.10, fontsize=54)
    ax.text(
        0.5,
        1.035,
        f"{MODEL_LABELS.get(model, model)}; MAE={mae:.2f}, MAPE={mape:.1f}%",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=34.0,
        color="#4a4a4a",
        clip_on=False,
    )
    ax.tick_params(axis="both", labelsize=40)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, loc="upper left", fontsize=36)
    fig.tight_layout(pad=1.4, rect=(0.02, 0.02, 0.98, 0.90))
    fig.savefig(FIGURES_DIR / lang / f"scatter_{model}_{CROP_SLUG[crop]}_lag_only.png", bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(corr_df: pd.DataFrame, crop: str, lang: str) -> None:
    cfg = LANG_CONFIG[lang]
    pearson = corr_df.set_index("factor")["pearson_yield"].sort_values(key=lambda s: np.abs(s), ascending=False)
    fig, ax = plt.subplots(figsize=(11.2, max(10.2, len(pearson) * 1.3)))
    matrix = pearson.values[:, None]
    cax = ax.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks([0])
    ax.set_xticklabels([str(cfg["heatmap_cb"])], fontsize=LABEL_SIZE + 4)
    ax.set_yticks(range(len(pearson)))
    ax.set_yticklabels([compact_feature_label(label) for label in pearson.index], fontsize=TICK_SIZE + 9)
    for i, value in enumerate(pearson.values):
        ax.text(0, i, f"{value:.2f}", ha="center", va="center", color="black", fontsize=ANNOTATION_SIZE + 6)
    ax.tick_params(axis="x", labelsize=LABEL_SIZE + 4)
    ax.tick_params(axis="y", pad=8)
    ax.set_title(crop_label(crop, lang), fontsize=TITLE_SIZE + 5)
    cbar = fig.colorbar(cax, ax=ax, fraction=0.06, pad=0.045)
    cbar.ax.tick_params(labelsize=TICK_SIZE + 6)
    cbar.set_label("Correlation", fontsize=LABEL_SIZE + 5)
    fig.tight_layout(pad=1.15)
    fig.savefig(FIGURES_DIR / lang / f"correlation_heatmap_{CROP_SLUG[crop]}.png", bbox_inches="tight")
    plt.close(fig)


def plot_baseline_vs_ml(baselines: pd.DataFrame, leaderboard: pd.DataFrame, languages: Iterable[str]) -> None:
    dirs = ensure_dirs(languages)
    for lang in languages:
        cfg = LANG_CONFIG[lang]
        fig, axes = plt.subplots(1, 3, figsize=(23.0, 13.8), sharey=True)
        max_val = 0.0
        for crop in TARGET_CROPS:
            crop_base = baselines[baselines["crop"] == crop]
            crop_ml = leaderboard[leaderboard["crop"] == crop]
            max_val = max(max_val, crop_base["mae"].max(), crop_ml["mae"].max())
        max_val *= 1.25

        for ax, crop in zip(axes, TARGET_CROPS):
            crop_base = baselines[baselines["crop"] == crop].copy()
            crop_base["label"] = crop_base["baseline"].map(lambda x: BASELINE_LABELS.get(x, {}).get(lang, x))
            crop_ml = leaderboard[leaderboard["crop"] == crop].copy()
            labels = crop_base["label"].tolist() + [f"Best ML ({MODEL_LABELS.get(crop_ml.iloc[0]['model'], crop_ml.iloc[0]['model'])})"]
            values = crop_base["mae"].tolist() + [float(crop_ml.iloc[0]["mae"])]
            colors = ["#9ecae1"] * len(crop_base) + ["#f28e2b"]
            x = np.arange(len(values))
            bars = ax.bar(x, values, color=colors, width=0.72)
            for bar, value in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + max_val * 0.02,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=BASELINE_VALUE_SIZE + 4,
                )
            ax.set_title(crop_label(crop, lang), pad=12, fontsize=TITLE_SIZE + 3)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=20, ha="right", rotation_mode="anchor")
            ax.tick_params(axis="x", labelsize=BASELINE_XTICK_SIZE + 4, pad=6)
            ax.tick_params(axis="y", labelsize=TICK_SIZE + 5)
            ax.set_ylim(0, max_val)
            ax.grid(axis="y", alpha=0.3)
        add_panel_labels(axes, x=-0.09, y=1.04, fontsize=PANEL_LABEL_SIZE)
        axes[0].set_ylabel(str(cfg["baseline_ylabel"]), fontsize=LABEL_SIZE + 5)
        fig.tight_layout(rect=(0.02, 0.09, 0.98, 0.98))
        fig.savefig(dirs[lang] / "mae_baselines_vs_ml.png", bbox_inches="tight")
        plt.close(fig)


def plot_lag_sensitivity(lag_df: pd.DataFrame, languages: Iterable[str]) -> None:
    best_per_lag = lag_df.sort_values(["crop", "lag_config", "validation_mae", "mae"]).groupby(["crop", "lag_config"], as_index=False).first()
    dirs = ensure_dirs(languages)
    lag_label_map = {"L1": "L1", "L1_L2": "L1+L2", "L1_L2_L3": "L1+L2+L3"}
    for lang in languages:
        cfg = LANG_CONFIG[lang]
        fig, axes = plt.subplots(1, 3, figsize=(14.4, 5.0), sharey=True)
        for ax, crop in zip(axes, TARGET_CROPS):
            subset = best_per_lag[best_per_lag["crop"] == crop]
            x = np.arange(len(subset))
            labels = [lag_label_map[item] for item in subset["lag_config"]]
            bars = ax.bar(x, subset["mae"], color="#4e79a7", width=0.65)
            for bar, value, model_name in zip(bars, subset["mae"], subset["model"]):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.02,
                    f"{value:.2f}\n{MODEL_LABELS.get(model_name, model_name)}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
            ax.set_title(crop_label(crop, lang))
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.grid(axis="y", alpha=0.3)
        axes[0].set_ylabel(str(cfg["lag_ylabel"]))
        add_panel_labels(axes, x=-0.08, y=1.03)
        fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.98))
        fig.savefig(dirs[lang] / "lag_sensitivity.png", bbox_inches="tight")
        plt.close(fig)


def plot_climate_sensitivity(climate_df: pd.DataFrame, languages: Iterable[str]) -> None:
    dirs = ensure_dirs(languages)
    for lang in languages:
        cfg = LANG_CONFIG[lang]
        fig, axes = plt.subplots(1, 3, figsize=(14.2, 5.0), sharey=True)
        for ax, crop in zip(axes, TARGET_CROPS):
            subset = climate_df[climate_df["crop"] == crop]
            if subset.empty:
                ax.set_visible(False)
                continue
            row = subset.iloc[0]
            labels = list(cfg["climate_labels"])
            values = [float(row["base_mae"]), float(row["climate_mae"])]
            colors = ["#9ecae1", "#59a14f"]
            bars = ax.bar(np.arange(2), values, color=colors, width=0.62)
            for bar, value in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.02,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=ANNOTATION_SIZE,
                )
            ax.set_title(crop_label(crop, lang))
            ax.set_xticks(np.arange(2))
            ax.set_xticklabels(labels, rotation=15, ha="right")
            ax.grid(axis="y", alpha=0.3)
        axes[0].set_ylabel(str(cfg["climate_ylabel"]))
        add_panel_labels(axes, x=-0.08, y=1.03)
        fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.98))
        fig.savefig(dirs[lang] / "climate_sensitivity.png", bbox_inches="tight")
        plt.close(fig)


def copy_best_shap_figures(leaderboard: pd.DataFrame, languages: Iterable[str]) -> None:
    for lang in languages:
        out_dir = FIGURES_DIR / lang
        out_dir.mkdir(parents=True, exist_ok=True)
        for _, row in leaderboard.iterrows():
            crop = row["crop"]
            model = row["model"]
            src = FIGURES_BASE_DIR / lang / f"shap_{model}_{CROP_SLUG[crop]}_lag_only.png"
            dst = out_dir / f"shap_{model}_{CROP_SLUG[crop]}_lag_only.png"
            if src.exists():
                shutil.copy2(src, dst)


def _panel_paths_for_figure(leaderboard: pd.DataFrame, lang: str, figure_no: int) -> list[Path]:
    figure_paths: list[Path] = []
    for crop in TARGET_CROPS:
        row = leaderboard[leaderboard["crop"] == crop].iloc[0]
        model = row["model"]
        crop_slug = CROP_SLUG[crop]
        if figure_no == 2:
            figure_paths.append(FIGURES_DIR / lang / f"correlation_heatmap_{crop_slug}.png")
        elif figure_no == 3:
            figure_paths.append(FIGURES_DIR / lang / f"line_{model}_{crop_slug}_lag_only.png")
        elif figure_no == 4:
            figure_paths.append(FIGURES_DIR / lang / f"scatter_{model}_{crop_slug}_lag_only.png")
        elif figure_no == 6:
            figure_paths.append(FIGURES_DIR / lang / f"shap_{model}_{crop_slug}_lag_only.png")
        else:
            raise ValueError(f"Unsupported composite figure number: {figure_no}")
    return figure_paths


def build_composite_figure(
    input_paths: list[Path],
    output_path: Path,
    *,
    ncols: int = 3,
    figsize: tuple[float, float] = (18, 5),
    panel_label_y: float = 1.015,
    image_padding: int = 4,
    tight_pad: float = 1.2,
    w_pad: float = 0.8,
    h_pad: float = 0.9,
    left: float = 0.015,
    right: float = 0.99,
    top: float = 0.982,
    bottom: float = 0.02,
) -> None:
    fig, axes = plt.subplots(int(np.ceil(len(input_paths) / ncols)), ncols, figsize=figsize)
    axes_array = np.atleast_1d(axes).flatten()
    for ax, panel_label, path in zip(axes_array, PANEL_LABELS, input_paths):
        image = trim_image_whitespace(mpimg.imread(path), threshold=0.99, padding=image_padding)
        ax.imshow(image)
        ax.axis("off")
        ax.text(
            0.01,
            panel_label_y,
            f"{panel_label})",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=COMPOSITE_PANEL_LABEL_SIZE,
            fontweight="bold",
            clip_on=False,
        )
    for ax in axes_array[len(input_paths):]:
        ax.axis("off")
    fig.tight_layout(pad=tight_pad, w_pad=w_pad, h_pad=h_pad)
    fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def build_composite_figure_2plus1(
    input_paths: list[Path],
    output_path: Path,
    *,
    figsize: tuple[float, float] = (26.0, 30.0),
    top_row_ratio: float = 1.22,
    bottom_row_ratio: float = 1.44,
    top_padding: int = 0,
    bottom_padding: int = 1,
    hspace: float = -0.16,
) -> None:
    if len(input_paths) != 3:
        raise ValueError("2+1 composite layout expects exactly three input panels")

    fig = plt.figure(figsize=figsize)
    grid = fig.add_gridspec(2, 2, height_ratios=(top_row_ratio, bottom_row_ratio), hspace=hspace, wspace=0.03)
    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[1, :]),
    ]

    for idx, (ax, panel_label, path) in enumerate(zip(axes, PANEL_LABELS, input_paths)):
        padding = top_padding if idx < 2 else bottom_padding
        image = trim_image_whitespace(mpimg.imread(path), threshold=0.99, padding=padding)
        ax.imshow(image)
        ax.axis("off")
        ax.text(
            0.01,
            1.001,
            f"{panel_label})",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=COMPOSITE_PANEL_LABEL_SIZE + 4,
            fontweight="bold",
            clip_on=False,
        )

    fig.subplots_adjust(left=0.004, right=0.997, top=0.997, bottom=0.006)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def export_submission_versions(
    lang: str = "en",
    suffix: str = ".4",
    figure_indices: Iterable[int] | None = None,
    include_previews: bool = True,
) -> None:
    SUBMISSION_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_dir = FIGURES_DIR / lang
    indices = tuple(figure_indices) if figure_indices is not None else tuple(range(1, 7))
    for index in indices:
        src = out_dir / f"manuscript_figure{index}.png"
        dst = SUBMISSION_FIGURES_DIR / f"Figure_{index}{suffix}.png"
        shutil.copy2(src, dst)
        if include_previews:
            render_a4_preview(dst, SUBMISSION_FIGURES_DIR / f"Figure_{index}{suffix}_A4_preview.png", FIGURE_CAPTIONS[index])


def export_manuscript_figure_set(
    leaderboard: pd.DataFrame,
    languages: Iterable[str],
    *,
    export_default_submission: bool = True,
) -> None:
    for lang in languages:
        out_dir = FIGURES_DIR / lang
        shutil.copy2(out_dir / "poltava_trends.png", out_dir / "manuscript_figure1.png")
        shutil.copy2(out_dir / "mae_baselines_vs_ml.png", out_dir / "manuscript_figure5.png")
        build_composite_figure(
            _panel_paths_for_figure(leaderboard, lang, 2),
            out_dir / "manuscript_figure2.png",
            ncols=3,
            figsize=(24.6, 13.8),
            panel_label_y=1.01,
            image_padding=5,
            tight_pad=1.35,
            w_pad=1.75,
            h_pad=0.9,
            left=0.02,
            right=0.985,
        )
        build_composite_figure(
            _panel_paths_for_figure(leaderboard, lang, 3),
            out_dir / "manuscript_figure3.png",
            ncols=3,
            figsize=(31.0, 20.6),
            panel_label_y=1.01,
            image_padding=4,
            tight_pad=1.35,
            w_pad=1.8,
            h_pad=1.0,
            left=0.02,
            right=0.985,
        )
        build_composite_figure(
            _panel_paths_for_figure(leaderboard, lang, 4),
            out_dir / "manuscript_figure4.png",
            ncols=3,
            figsize=(28.0, 16.8),
            panel_label_y=1.01,
            image_padding=4,
            tight_pad=1.35,
            w_pad=1.55,
            h_pad=1.0,
            left=0.02,
            right=0.985,
        )
        build_composite_figure_2plus1(
            _panel_paths_for_figure(leaderboard, lang, 6),
            out_dir / "manuscript_figure6.png",
            figsize=(28.5, 30.5),
            top_row_ratio=1.78,
            bottom_row_ratio=1.02,
            top_padding=0,
            bottom_padding=1,
        )

    if export_default_submission:
        export_submission_versions("en", ".4")


def main() -> None:
    languages = ("uk", "en")
    style_matplotlib()
    ensure_dirs(languages)

    features = pd.read_parquet(PROCESSED_FEATURES)
    predictions = pd.read_csv(PREDICTIONS_CSV)
    leaderboard = pd.read_csv(METRICS_BY_SCENARIO_CSV)
    leaderboard = leaderboard[leaderboard["scenario"] == "lag_only"].sort_values(["crop", "mae"]).groupby("crop", as_index=False).first()
    baselines = pd.read_csv(BASELINE_SUMMARY_CSV)
    lag_df = pd.read_csv(LAG_SENSITIVITY_CSV)
    climate_df = pd.read_csv(CLIMATE_SENSITIVITY_CSV)

    plot_trends(features, languages)
    plot_baseline_vs_ml(baselines, leaderboard, languages)
    plot_lag_sensitivity(lag_df, languages)
    plot_climate_sensitivity(climate_df, languages)

    for _, row in leaderboard.iterrows():
        crop = row["crop"]
        model = row["model"]
        for lang in languages:
            plot_prediction_series(predictions, crop, model, lang)
            plot_prediction_scatter(predictions, crop, model, lang)
            corr_path = str(CORR_TEMPLATE).format(crop=CROP_SLUG[crop])
            corr_df = pd.read_csv(corr_path)
            plot_heatmap(corr_df, crop, lang)

    copy_best_shap_figures(leaderboard, languages)
    export_manuscript_figure_set(leaderboard, languages)


if __name__ == "__main__":
    main()
