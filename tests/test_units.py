import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agrostats.normalize import (
    irrigation_mln_m3_to_metrics,
    mass_thousand_tonnes_to_kg_per_ha,
    yield_centner_to_tonnes_per_ha,
)


def test_yield_centner_to_tonnes():
    series = pd.Series([45])
    result = yield_centner_to_tonnes_per_ha(series)
    assert result.iloc[0] == 4.5


def test_mass_thousand_tonnes_to_kg_per_ha_all_crops():
    # 60 тис. т / 300 тис. га = 200 кг/га
    value_series = pd.Series([60])
    area_series = pd.Series([300_000])
    result = mass_thousand_tonnes_to_kg_per_ha(value_series, area_series)
    assert result.iloc[0] == 200


def test_irrigation_mln_m3_to_mm():
    value_series = pd.Series([30])  # млн м3
    area_series = pd.Series([1_500_000])  # га
    metrics = irrigation_mln_m3_to_metrics(value_series, area_series)
    mm_value = metrics["mm"].iloc[0]
    assert pytest.approx(mm_value, rel=1e-6) == 2.0
