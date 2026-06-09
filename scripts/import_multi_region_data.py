"""Import newly downloaded AgroStats region exports into canonical raw folders."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agrostats.multiregion import build_data_inventory, import_region_exports


DEFAULT_SOURCE = (
    ROOT.parent
    / "мини статья2"
    / "data"
)
RAW_ROOT = ROOT / "data" / "raw" / "agrostats"
REPORTS = ROOT / "reports"


def main() -> None:
    imported = import_region_exports(DEFAULT_SOURCE, RAW_ROOT)
    REPORTS.mkdir(parents=True, exist_ok=True)
    imported_path = REPORTS / "imported_multi_region_files.csv"
    imported.to_csv(imported_path, index=False)

    inventory = build_data_inventory(RAW_ROOT)
    inventory_path = REPORTS / "data_inventory.csv"
    inventory.to_csv(inventory_path, index=False)

    incomplete = inventory[~inventory["complete"]]
    print(f"Imported {len(imported)} files from {DEFAULT_SOURCE}")
    print(f"Saved import log to {imported_path}")
    print(f"Saved data inventory to {inventory_path}")
    if not incomplete.empty:
        print(incomplete.to_string(index=False))
        raise SystemExit("Data inventory is incomplete.")


if __name__ == "__main__":
    main()
