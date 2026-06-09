import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agrostats.multiregion import (
    EXPECTED_FILES_PER_REGION,
    build_data_inventory,
    build_threshold_sensitivity,
    canonical_agrostats_name,
)


def test_fertilizer_suffix_mapping_all_types():
    base = "AgroStats_chart-Вінницька область-Добрива-Пшениця_2010-2024"

    assert canonical_agrostats_name(f"{base}.csv", region_slug="vinnytsia").endswith("Мінеральні.csv")
    assert canonical_agrostats_name(f"{base} (1).csv", region_slug="vinnytsia").endswith("Азотні.csv")
    assert canonical_agrostats_name(f"{base} (2).csv", region_slug="vinnytsia").endswith("Фосфорні.csv")
    assert canonical_agrostats_name(f"{base} (3).csv", region_slug="vinnytsia").endswith("Калійні.csv")
    assert canonical_agrostats_name(f"{base} (4).csv", region_slug="vinnytsia").endswith("Органічні.csv")


def test_ukraine_copy_file_maps_to_organic_grain():
    name = "AgroStats_chart-Україна-Добрива-Зернові_2010-2024 copy.csv"

    assert canonical_agrostats_name(name, region_slug="ukraine") == (
        "AgroStats_chart-Україна-Добрива-Зернові_2010-2024 Органічні.csv"
    )


def test_data_inventory_reports_expected_region_files():
    raw_root = Path(__file__).resolve().parents[1] / "data" / "raw" / "agrostats"
    inventory = build_data_inventory(raw_root, regions=("poltava", "vinnytsia", "cherkasy", "ukraine"))

    assert set(inventory["region_slug"]) == {"poltava", "vinnytsia", "cherkasy", "ukraine"}
    assert (inventory["file_count"] == EXPECTED_FILES_PER_REGION).all()
    assert inventory["complete"].all()


def test_threshold_sensitivity_contains_all_configured_margins():
    recommended = pd.DataFrame(
        {
            "region_slug": ["poltava", "poltava", "poltava"],
            "crop": ["Пшениця", "Кукурудза", "Соняшник"],
            "model": ["elasticnet", "lightgbm", "lightgbm"],
            "best_ml_mae": [0.50, 0.80, 0.04],
            "best_baseline_raw": ["forecast_linear", "forecast_linear", "naive_lag1"],
            "best_baseline_mae": [0.40, 1.00, 0.17],
            "test_coverage": [0.67, 0.33, 1.00],
        }
    )

    result = build_threshold_sensitivity(recommended)

    assert set(result["threshold_margin"]) == {0.0, 0.03, 0.05, 0.10}
    assert len(result) == 12
    maize_010 = result[(result["crop"] == "Кукурудза") & (result["threshold_margin"] == 0.10)]
    assert maize_010.iloc[0]["recommended_type"] == "machine_learning"
