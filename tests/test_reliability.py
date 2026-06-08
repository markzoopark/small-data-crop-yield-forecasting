import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agrostats.reliability import build_forecast_cards, select_recommended_methods


def _baseline_summary():
    return pd.DataFrame(
        {
            "baseline": ["forecast_linear", "forecast_linear"],
            "crop": ["Wheat", "Maize"],
            "mae": [0.40, 1.00],
            "rmse": [0.45, 1.10],
            "mape": [8.0, 15.0],
        }
    )


def _band_summary():
    return pd.DataFrame(
        {
            "crop": ["Wheat", "Maize"],
            "test_coverage": [0.67, 0.33],
            "mean_band_width": [0.8, 1.5],
        }
    )


def test_selector_keeps_baseline_when_baseline_is_better():
    leaderboard = pd.DataFrame(
        {
            "crop": ["Wheat"],
            "model": ["elasticnet"],
            "mae": [0.50],
        }
    )
    selected = select_recommended_methods(leaderboard, _baseline_summary(), _band_summary(), practical_margin=0.05)

    assert selected.iloc[0]["recommended_type"] == "baseline"
    assert selected.iloc[0]["recommended_method"] == "FORECAST.LINEAR"
    assert selected.iloc[0]["warning_label"] == "baseline safer"


def test_selector_uses_ml_only_when_gain_exceeds_margin():
    leaderboard = pd.DataFrame(
        {
            "crop": ["Maize"],
            "model": ["lightgbm"],
            "mae": [0.80],
        }
    )
    selected = select_recommended_methods(leaderboard, _baseline_summary(), _band_summary(), practical_margin=0.05)

    assert selected.iloc[0]["recommended_type"] == "machine_learning"
    assert selected.iloc[0]["recommended_method"] == "LightGBM"
    assert selected.iloc[0]["ml_gain_vs_baseline"] == pytest.approx(0.20)
    assert selected.iloc[0]["warning_label"] == "outside validation error scale"


def test_forecast_cards_contain_required_interpretation_fields():
    recommended = pd.DataFrame(
        {
            "crop": ["Wheat", "Maize", "Sunflower"],
            "crop_en": ["wheat", "maize", "sunflower"],
            "recommended_type": ["baseline", "machine_learning", "machine_learning"],
            "recommended_method": ["FORECAST.LINEAR", "LightGBM", "LightGBM"],
            "recommended_mae": [0.4, 0.8, 0.04],
            "best_ml_mae": [0.5, 0.8, 0.04],
            "best_baseline_mae": [0.4, 1.0, 0.17],
            "ml_gain_vs_baseline": [-0.1, 0.2, 0.13],
            "warning_label": ["baseline safer", "outside validation error scale", "within expected error"],
        }
    )
    ablation = pd.DataFrame(
        {
            "crop": ["Wheat", "Maize", "Sunflower"],
            "feature_group": ["area", "mineral_share", "fertiliser_p"],
            "delta_mae": [0.1, 0.8, 0.13],
        }
    )
    bands = pd.DataFrame(
        {
            "crop": ["Wheat", "Maize", "Sunflower"],
            "mean_band_width": [1.2, 1.0, 0.9],
        }
    )

    cards = build_forecast_cards(recommended, ablation, bands)

    assert set(cards["crop_en"]) == {"wheat", "maize", "sunflower"}
    for column in ["top_feature_group", "mean_empirical_band_width", "interpretation"]:
        assert column in cards.columns
    assert cards["interpretation"].str.len().min() > 30
