"""Exploratory data analysis toolkit for agrostats."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from statsmodels.tsa.stattools import acf as sm_acf, pacf as sm_pacf

FEATURES_PATH = Path("data/processed/agrostats_poltava_features.parquet")
FIGURES_DIR = Path("reports/figures")
REPORTS_DIR = Path("reports")
TARGET_CROPS = ("Пшениця", "Кукурудза", "Соняшник")
LAG_FEATURES = [
    "N_kg_ha_lag1",
    "P2O5_kg_ha_lag1",
    "K_kg_ha_lag1",
    "Mineral_treated_share_lag1",
    "Org_kg_ha_or_share_lag1",
    "Irrig_mm_lag1",
    "Area_ha_lag1",
]
TREND_COLUMNS = [
    "Yield_t_ha",
    "Area_ha",
    "N_kg_ha",
    "P2O5_kg_ha",
    "K_kg_ha",
    "Irrig_mm",
]

LANG_CONFIG = {
    "uk": {
        "acf_title": "ACF урожайності — {crop}",
        "pacf_title": "PACF урожайності — {crop}",
        "lag_label": "Лаг (років)",
        "corr_title": "Кореляції лагових факторів — {crop}",
        "heatmap_colorbar": "Коефіцієнт Пірсона",
        "heatmap_xtick": "Кореляція",
        "year_label": "Рік",
        "pearson_label": "Кореляція з урожайністю",
        "legend_title": "Культура",
        "trend_labels": {
            "Yield_t_ha": ("Урожайність", "т/га"),
            "Area_ha": ("Посівна площа", "га"),
            "N_kg_ha": ("Азотні добрива", "кг/га"),
            "P2O5_kg_ha": ("Фосфорні добрива", "кг/га"),
            "K_kg_ha": ("Калійні добрива", "кг/га"),
            "Irrig_mm": ("Зрошення", "мм"),
        },
    },
    "en": {
        "acf_title": "Yield ACF — {crop}",
        "pacf_title": "Yield PACF — {crop}",
        "lag_label": "Lag (years)",
        "corr_title": "Lag-factor correlations — {crop}",
        "heatmap_colorbar": "Pearson r",
        "heatmap_xtick": "Correlation",
        "year_label": "Year",
        "pearson_label": "Correlation with yield",
        "legend_title": "Crop",
        "trend_labels": {
            "Yield_t_ha": ("Yield", "t/ha"),
            "Area_ha": ("Sown area", "ha"),
            "N_kg_ha": ("Nitrogen fertilisers", "kg/ha"),
            "P2O5_kg_ha": ("Phosphorus fertilisers", "kg/ha"),
            "K_kg_ha": ("Potassium fertilisers", "kg/ha"),
            "Irrig_mm": ("Irrigation", "mm"),
        },
    },
}

CROP_NAMES = {
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


def slugify(value: str) -> str:
    mapping = {
        "Пшениця": "pshenytsia",
        "Кукурудза": "kukurudza",
        "Соняшник": "sonyashnyk",
    }
    if value in mapping:
        return mapping[value]
    ascii_value = (
        value.encode("ascii", "ignore").decode("ascii") if value.isascii() else value
    )
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in ascii_value).strip("_").lower()
    return cleaned or f"crop_{abs(hash(value)) % 10000}"


def ensure_language(language: str) -> None:
    if language not in LANG_CONFIG:
        raise ValueError(f"Unsupported language: {language}")


def load_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Feature parquet not found: {path}")
    return pd.read_parquet(path)


def get_crop_name(crop: str, language: str) -> str:
    return CROP_NAMES.get(language, {}).get(crop, crop)


def compute_acf_pacf(series: pd.Series) -> tuple[np.ndarray, np.ndarray, int, float]:
    ordered = series.dropna().astype(float).sort_index()
    y = ordered.values
    if len(y) < 3:
        raise ValueError("Time series is too short to compute ACF/PACF.")
    max_lag = min(6, len(y) - 2)
    acf_vals = sm_acf(y, nlags=max_lag, fft=True, adjusted=False)
    pacf_vals = sm_pacf(y, nlags=max_lag, method="yw")
    conf = 1.96 / np.sqrt(len(y))
    return acf_vals[1:], pacf_vals[1:], max_lag, conf


def plot_acf_pacf(series: pd.Series, crop: str, language: str) -> None:
    ensure_language(language)
    texts = LANG_CONFIG[language]
    crop_label = get_crop_name(crop, language)

    try:
        acf_vals, pacf_vals, max_lag, conf = compute_acf_pacf(series)
    except ValueError as exc:
        print(f"[!] Skipping ACF/PACF for {crop}: {exc}")
        return

    lags = np.arange(1, max_lag + 1)
    lang_dir = FIGURES_DIR / language
    lang_dir.mkdir(parents=True, exist_ok=True)

    # ACF
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(lags, acf_vals, color="#4B8BBE")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(conf, color="red", linestyle="--", linewidth=0.8)
    ax.axhline(-conf, color="red", linestyle="--", linewidth=0.8)
    ax.set_title(texts["acf_title"].format(crop=crop_label))
    ax.set_xlabel(texts["lag_label"])
    ax.set_ylabel(texts["pearson_label"])
    ax.set_ylim(-1, 1)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(lang_dir / f"yield_acf_{slugify(crop)}.png", dpi=200)
    plt.close(fig)

    # PACF
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(lags, pacf_vals, color="#FF7F0E")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(conf, color="red", linestyle="--", linewidth=0.8)
    ax.axhline(-conf, color="red", linestyle="--", linewidth=0.8)
    ax.set_title(texts["pacf_title"].format(crop=crop_label))
    ax.set_xlabel(texts["lag_label"])
    ax.set_ylabel(texts["pearson_label"])
    ax.set_ylim(-1, 1)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(lang_dir / f"yield_pacf_{slugify(crop)}.png", dpi=200)
    plt.close(fig)

    print(f"[+] Saved ACF/PACF for {crop} ({language})")


def prepare_factor_table(subset: pd.DataFrame) -> pd.DataFrame:
    columns = ["year", "Yield_t_ha", "Yield_anom"] + LAG_FEATURES
    table = subset[columns].dropna(subset=["Yield_t_ha"])
    for col in columns:
        table[col] = pd.to_numeric(table[col], errors="coerce")
    table = table.dropna()
    std = table.std(axis=0, numeric_only=True)
    keep = std[std > 0].index
    return table[list(keep)]


def corr_with_stats(x: pd.Series, y: pd.Series) -> dict[str, float]:
    pearson_corr, pearson_p = pearsonr(x, y)
    spearman_corr, spearman_p = spearmanr(x, y)
    return {
        "pearson": pearson_corr,
        "pearson_p": pearson_p,
        "spearman": spearman_corr,
        "spearman_p": spearman_p,
    }


def generate_correlation_reports(subset: pd.DataFrame, crop: str, languages: Sequence[str]) -> None:
    table = prepare_factor_table(subset)
    if table.empty:
        print(f"[!] Not enough data for correlations ({crop})")
        return

    factors = [col for col in table.columns if col not in {"year", "Yield_t_ha", "Yield_anom"}]
    if not factors:
        print(f"[!] No factors available for {crop}")
        return

    pearson_matrix = []
    for factor in factors:
        stats = corr_with_stats(table[factor], table["Yield_t_ha"])
        pearson_matrix.append(
            {
                "factor": factor,
                "pearson_yield": stats["pearson"],
                "pearson_yield_p": stats["pearson_p"],
                "spearman_yield": stats["spearman"],
                "spearman_yield_p": stats["spearman_p"],
            }
        )

    anomaly_matrix = []
    for factor in factors:
        stats = corr_with_stats(table[factor], table["Yield_anom"])
        anomaly_matrix.append(
            {
                "factor": factor,
                "pearson_anom": stats["pearson"],
                "pearson_anom_p": stats["pearson_p"],
                "spearman_anom": stats["spearman"],
                "spearman_anom_p": stats["spearman_p"],
            }
        )

    corr_df = pd.DataFrame(pearson_matrix).merge(pd.DataFrame(anomaly_matrix), on="factor", how="outer")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = REPORTS_DIR / f"correlations_{slugify(crop)}.csv"
    corr_df.to_csv(csv_path, index=False)
    print(f"[+] Saved correlation table {csv_path}")

    pivot_data = corr_df.set_index("factor")["pearson_yield"]
    for language in languages:
        ensure_language(language)
        texts = LANG_CONFIG[language]
        crop_label = get_crop_name(crop, language)
        lang_dir = FIGURES_DIR / language
        lang_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(6, max(3, len(pivot_data) * 0.4)))
        values = pivot_data.values[:, None]
        cax = ax.imshow(values, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks([0])
        ax.set_xticklabels([texts["heatmap_xtick"]])
        ax.set_yticks(range(len(pivot_data)))
        ax.set_yticklabels(pivot_data.index)
        for i, value in enumerate(pivot_data.values):
            ax.text(0, i, f"{value:.2f}", ha="center", va="center", color="black")
        ax.set_title(texts["corr_title"].format(crop=crop_label))
        fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04, label=texts["heatmap_colorbar"])
        fig.tight_layout()
        fig.savefig(lang_dir / f"correlation_heatmap_{slugify(crop)}.png", dpi=200)
        plt.close(fig)


def generate_acf_pacf(features_df: pd.DataFrame, crops: Iterable[str], languages: Sequence[str]) -> None:
    for crop in crops:
        subset = features_df[features_df["group_or_crop"] == crop].copy()
        if subset.empty:
            print(f"[!] No data available for {crop}")
            continue
        subset = subset.sort_values("year")
        for language in languages:
            plot_acf_pacf(subset.set_index("year")["Yield_t_ha"], crop, language)


def plot_trends(features_df: pd.DataFrame, crops: Sequence[str], languages: Sequence[str]) -> None:
    subset = features_df[features_df["group_or_crop"].isin(crops)].copy()
    if subset.empty:
        print("[!] No data available for trend plot.")
        return
    subset = subset.sort_values("year")

    for language in languages:
        ensure_language(language)
        texts = LANG_CONFIG[language]
        trend_labels = texts["trend_labels"]
        lang_dir = FIGURES_DIR / language
        lang_dir.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(2, 3, figsize=(18, 8), sharex=True)
        axes = axes.flatten()

        for ax, column in zip(axes, TREND_COLUMNS):
            title, unit = trend_labels[column]
            for crop in crops:
                crop_df = subset[subset["group_or_crop"] == crop]
                if crop_df.empty:
                    continue
                ax.plot(
                    crop_df["year"],
                    crop_df[column],
                    marker="o",
                    label=get_crop_name(crop, language),
                )
            ax.set_title(f"{title} ({unit})")
            ax.set_xlabel(texts["year_label"])
            ax.set_ylabel(unit)
            ax.set_xlim(2010, 2024)
            ax.set_xticks(range(2010, 2025))
            ax.grid(True, alpha=0.3)

        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=len(crops), frameon=False, title=texts["legend_title"])
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(lang_dir / "poltava_trends.png", dpi=200)
        plt.close(fig)
        print(f"[+] Saved trend plot ({language})")


def correlation_pipeline(features_df: pd.DataFrame, crops: Sequence[str], languages: Sequence[str]) -> None:
    for crop in crops:
        subset = features_df[features_df["group_or_crop"] == crop].copy()
        if subset.empty:
            print(f"[!] No data available for {crop}")
            continue
        subset = subset.sort_values("year")
        generate_correlation_reports(subset, crop, languages)


def main(args: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="EDA utilities for agrostats.")
    parser.add_argument(
        "--features-path",
        type=Path,
        default=FEATURES_PATH,
        help="Feature parquet file (default: data/processed/agrostats_poltava_features.parquet)",
    )
    parser.add_argument(
        "--mode",
        choices=["acf", "correlations", "trends"],
        default="acf",
        help="What to generate: acf, correlations, or trends.",
    )
    parser.add_argument(
        "--language",
        default="uk",
        help="Comma-separated list of languages (uk,en).",
    )
    parsed = parser.parse_args(args=args)

    languages = [lang.strip() for lang in parsed.language.split(",") if lang.strip()]
    if not languages:
        languages = ["uk"]
    for language in languages:
        ensure_language(language)

    df = load_features(parsed.features_path)
    if parsed.mode == "acf":
        generate_acf_pacf(df, TARGET_CROPS, languages)
    elif parsed.mode == "correlations":
        correlation_pipeline(df, TARGET_CROPS, languages)
    elif parsed.mode == "trends":
        plot_trends(df, TARGET_CROPS, languages)


if __name__ == "__main__":
    main()
