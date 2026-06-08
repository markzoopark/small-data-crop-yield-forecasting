import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agrostats.diagnostics import apply_prediction_bands, build_band_summary, summarise_band_coverage


def test_build_band_summary_uses_validation_abs_error_quantile():
    predictions = pd.DataFrame(
        {
            "crop": ["A", "A", "A"],
            "model": ["m", "m", "m"],
            "scenario": ["lag_only", "lag_only", "lag_only"],
            "lag_config": ["L1", "L1", "L1"],
            "actual": [1.0, 2.0, 3.0],
            "predicted": [0.9, 2.2, 2.7],
        }
    )

    summary = build_band_summary(predictions, quantile=0.8)

    assert len(summary) == 1
    assert summary.iloc[0]["residual_n"] == 3
    assert summary.iloc[0]["validation_mae"] == pytest.approx(0.2)
    assert summary.iloc[0]["empirical_abs_error_q80"] == pytest.approx(0.26)
    assert "not a formal confidence interval" in summary.iloc[0]["band_note"]


def test_apply_prediction_bands_marks_coverage():
    test_predictions = pd.DataFrame(
        {
            "crop": ["A", "A"],
            "model": ["m", "m"],
            "scenario": ["lag_only", "lag_only"],
            "lag_config": ["L1", "L1"],
            "year": [2022, 2023],
            "actual": [10.2, 11.5],
            "predicted": [10.0, 11.0],
        }
    )
    summary = pd.DataFrame(
        {
            "crop": ["A"],
            "model": ["m"],
            "scenario": ["lag_only"],
            "lag_config": ["L1"],
            "empirical_abs_error_q80": [0.3],
            "band_quantile": [0.8],
            "band_note": ["diagnostic"],
        }
    )

    bands = apply_prediction_bands(test_predictions, summary)

    assert bands.loc[0, "band_lower"] == pytest.approx(9.7)
    assert bands.loc[0, "band_upper"] == pytest.approx(10.3)
    assert bands.loc[0, "actual_within_band"]
    assert not bands.loc[1, "actual_within_band"]


def test_summarise_band_coverage_adds_test_coverage():
    bands = pd.DataFrame(
        {
            "crop": ["A", "A"],
            "model": ["m", "m"],
            "scenario": ["lag_only", "lag_only"],
            "lag_config": ["L1", "L1"],
            "year": [2022, 2023],
            "actual_within_band": [True, False],
            "empirical_abs_error_q80": [0.3, 0.3],
        }
    )
    summary = pd.DataFrame(
        {
            "crop": ["A"],
            "model": ["m"],
            "scenario": ["lag_only"],
            "lag_config": ["L1"],
            "residual_n": [3],
        }
    )

    result = summarise_band_coverage(bands, summary)

    assert result.iloc[0]["test_n"] == 2
    assert result.iloc[0]["test_coverage"] == pytest.approx(0.5)
    assert result.iloc[0]["mean_band_width"] == pytest.approx(0.6)
