"""Build compact local figures for the reliability mini-article.

The output directory is intentionally under paper/ and is ignored by Git. The
script makes the local article figures reproducible without publishing the
article assets in the repository.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FEATURES = ROOT / "data" / "processed" / "agrostats_poltava_features.csv"
OUT = ROOT / "paper" / "figures"

TARGET_CROPS = ["Пшениця", "Кукурудза", "Соняшник"]
CROP_EN = {"Пшениця": "Wheat", "Кукурудза": "Maize", "Соняшник": "Sunflower"}
CROP_EN_SHORT = {"Пшениця": "wheat", "Кукурудза": "maize", "Соняшник": "sunflower"}
MODEL_LABELS = {"elasticnet": "ElasticNet", "xgboost": "XGBoost", "lightgbm": "LightGBM"}
BASELINE_LABELS = {
    "arima": "ARIMA",
    "forecast_linear": "Linear trend",
    "linest_lag_only": "LINEST + lags",
    "naive_lag1": "Naive (t-1)",
}
TREND_COLUMNS = [
    ("Yield_t_ha", "Yield", "t/ha"),
    ("Area_ha", "Crop area", "ha"),
    ("N_kg_ha", "Nitrogen fertilisers", "kg/ha"),
    ("P2O5_kg_ha", "Phosphorus fertilisers", "kg/ha"),
    ("K_kg_ha", "Potassium fertilisers", "kg/ha"),
    ("Irrig_mm", "Irrigation", "mm"),
]
GROUP_LABELS = {
    "yield_history": "yield history",
    "area": "crop area",
    "fertiliser_n": "nitrogen",
    "fertiliser_p": "phosphorus",
    "fertiliser_k": "potassium",
    "mineral_share": "mineral share",
    "organic": "organic input",
    "irrigation": "irrigation",
}


def require(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(path)


def setup_style() -> None:
    style = "seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "classic"
    plt.style.use(style)
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.titlepad": 5,
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, bbox_inches="tight", pad_inches=0.03, facecolor="white")
    plt.close(fig)


def best_lag_only_models() -> pd.DataFrame:
    metrics = pd.read_csv(REPORTS / "metrics_by_scenario.csv")
    return (
        metrics[metrics["scenario"].eq("lag_only")]
        .sort_values(["crop", "mae"])
        .groupby("crop", as_index=False)
        .first()
    )


def plot_trends() -> None:
    require(FEATURES)
    features = pd.read_csv(FEATURES)
    fig, axes = plt.subplots(2, 3, figsize=(7.15, 4.55))
    colors = {"Пшениця": "#1f77b4", "Кукурудза": "#ff7f0e", "Соняшник": "#2ca02c"}
    for ax, (column, title, ylabel), label in zip(axes.ravel(), TREND_COLUMNS, "abcdef"):
        for crop in TARGET_CROPS:
            subset = features[features["group_or_crop"].eq(crop)].sort_values("year")
            ax.plot(subset["year"], subset[column], marker="o", markersize=2.4, linewidth=1.2, color=colors[crop], label=CROP_EN[crop])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Year")
        ax.set_xlim(2009.6, 2024.4)
        ax.set_xticks([2010, 2014, 2018, 2022, 2024])
        ax.tick_params(axis="x", rotation=35)
        ax.text(-0.12, 1.04, f"{label})", transform=ax.transAxes, fontweight="bold", ha="left", va="bottom")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.subplots_adjust(left=0.075, right=0.99, top=0.88, bottom=0.11, wspace=0.34, hspace=0.62)
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.99))
    save(fig, "manuscript_figure1.png")


def plot_baseline_vs_ml() -> None:
    baselines = pd.read_csv(REPORTS / "metrics_baselines_summary.csv")
    leaderboard = best_lag_only_models()
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 3.55), sharey=True, constrained_layout=True)
    max_val = 0.0
    for crop in TARGET_CROPS:
        crop_base = baselines[baselines["crop"].eq(crop)]
        crop_ml = leaderboard[leaderboard["crop"].eq(crop)]
        max_val = max(max_val, float(crop_base["mae"].max()), float(crop_ml["mae"].max()))
    max_val *= 1.22

    for ax, crop, panel in zip(axes, TARGET_CROPS, "abc"):
        crop_base = baselines[baselines["crop"].eq(crop)].copy()
        crop_base["label"] = crop_base["baseline"].map(BASELINE_LABELS)
        crop_ml = leaderboard[leaderboard["crop"].eq(crop)].iloc[0]
        labels = crop_base["label"].tolist() + [f"Best ML\n({MODEL_LABELS.get(crop_ml['model'], crop_ml['model'])})"]
        values = crop_base["mae"].astype(float).tolist() + [float(crop_ml["mae"])]
        colors = ["#9ecae1"] * len(crop_base) + ["#f28e2b"]
        x = np.arange(len(values))
        bars = ax.bar(x, values, color=colors, width=0.68)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + max_val * 0.018, f"{value:.2f}", ha="center", va="bottom", fontsize=7)
        ax.set_title(CROP_EN[crop])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right", rotation_mode="anchor")
        ax.set_ylim(0, max_val)
        ax.grid(axis="y", alpha=0.25)
        ax.text(-0.11, 1.03, f"{panel})", transform=ax.transAxes, fontweight="bold", ha="left", va="bottom")
    axes[0].set_ylabel("MAE, t/ha")
    save(fig, "mae_baselines_vs_ml.png")


def plot_actual_vs_forecast() -> None:
    predictions = pd.read_csv(REPORTS / "predictions.csv")
    leaderboard = best_lag_only_models()
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.45), constrained_layout=True)
    for ax, crop, panel in zip(axes, TARGET_CROPS, "abc"):
        model = leaderboard[leaderboard["crop"].eq(crop)].iloc[0]["model"]
        subset = predictions[
            predictions["crop"].eq(crop)
            & predictions["model"].eq(model)
            & predictions["scenario"].eq("lag_only")
            & predictions["split"].eq("test")
        ].sort_values("year")
        ax.plot(subset["year"], subset["y_true"], marker="o", markersize=2.8, linewidth=1.2, label="Actual")
        ax.plot(subset["year"], subset["y_pred"], marker="o", markersize=2.8, linewidth=1.2, linestyle="--", label="Forecast")
        ax.set_title(f"{CROP_EN[crop]} ({MODEL_LABELS.get(model, model)})")
        ax.set_xlabel("Year")
        ax.set_ylabel("Yield, t/ha")
        ax.set_xticks(subset["year"].tolist())
        ax.grid(alpha=0.2)
        ax.text(-0.1, 1.03, f"{panel})", transform=ax.transAxes, fontweight="bold", ha="left", va="bottom")
    axes[0].legend(frameon=False, loc="best")
    save(fig, "manuscript_figure3.png")


def plot_feature_group_ablation() -> None:
    ablation = pd.read_csv(REPORTS / "feature_group_ablation.csv")
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.4), constrained_layout=True)
    x_limits = {"Пшениця": (-0.04, 0.20), "Кукурудза": (-0.08, 0.92), "Соняшник": (-0.03, 0.20)}
    x_ticks = {
        "Пшениця": [0.0, 0.1, 0.2],
        "Кукурудза": [0.0, 0.4, 0.8],
        "Соняшник": [0.0, 0.1, 0.2],
    }
    for ax, crop, panel in zip(axes, TARGET_CROPS, "abc"):
        subset = ablation[ablation["crop"].eq(crop)].copy()
        subset["label"] = subset["feature_group"].map(GROUP_LABELS).fillna(subset["feature_group"])
        subset = subset.sort_values("delta_mae", ascending=True)
        colors = ["#bdbdbd" if value < 0 else "#4e79a7" for value in subset["delta_mae"]]
        ax.barh(subset["label"], subset["delta_mae"], color=colors, height=0.62)
        ax.axvline(0, color="#555555", linewidth=0.8)
        ax.set_title(CROP_EN[crop])
        ax.set_xlim(*x_limits[crop])
        ax.set_xticks(x_ticks[crop])
        ax.set_xlabel("Delta MAE, t/ha")
        ax.tick_params(axis="y", labelsize=6.5)
        ax.grid(axis="x", alpha=0.25)
        ax.text(-0.08, 1.03, f"{panel})", transform=ax.transAxes, fontweight="bold", ha="left", va="bottom")
    save(fig, "mini_feature_group_ablation.png")


def main() -> None:
    for path in [
        FEATURES,
        REPORTS / "metrics_baselines_summary.csv",
        REPORTS / "metrics_by_scenario.csv",
        REPORTS / "predictions.csv",
        REPORTS / "feature_group_ablation.csv",
    ]:
        require(path)
    setup_style()
    plot_trends()
    plot_baseline_vs_ml()
    plot_actual_vs_forecast()
    plot_feature_group_ablation()
    print(f"Built mini-article figures under {OUT}")


if __name__ == "__main__":
    main()
