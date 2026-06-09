"""Multi-region data import and reliability audit helpers."""

from __future__ import annotations

import re
import shutil
import unicodedata
from pathlib import Path
from typing import Iterable

import pandas as pd

from agrostats.reliability import select_recommended_methods


EXPECTED_FILES_PER_REGION = 91
EXPECTED_REGIONS = ("poltava", "vinnytsia", "cherkasy", "ukraine")
REGION_LABELS = {
    "poltava": "Poltava",
    "vinnytsia": "Vinnytsia",
    "cherkasy": "Cherkasy",
    "ukraine": "Ukraine",
}
SOURCE_DIR_TO_SLUG = {
    "винница": "vinnytsia",
    "черкасы": "cherkasy",
    "украина": "ukraine",
}
SOURCE_REGION_NAMES = {
    "vinnytsia": "Вінницька область",
    "cherkasy": "Черкаська область",
    "ukraine": "Україна",
}
FERTILIZER_SUFFIX_TO_TYPE = {
    "": "Мінеральні",
    "1": "Азотні",
    "2": "Фосфорні",
    "3": "Калійні",
    "4": "Органічні",
    "copy": "Органічні",
}
THRESHOLD_MARGINS = (0.00, 0.03, 0.05, 0.10)

_FERTILIZER_EXPORT_RE = re.compile(
    r"^(?P<prefix>AgroStats_chart-[^-]+-Добрива-[^_]+_\d{4}-\d{4})(?: \((?P<suffix>[1-4])\)| (?P<copy>copy))?\.csv$"
)


def normalise_filename_text(value: str) -> str:
    """Normalise Unicode and a few common browser-download artefacts."""
    value = unicodedata.normalize("NFC", value.strip())
    value = value.replace("Україна", "Україна")
    value = value.replace("Калійні", "Калійні")
    return value


def canonical_agrostats_name(filename: str, *, region_slug: str | None = None) -> str:
    """Return the canonical filename expected by the AgroStats parser."""
    filename = normalise_filename_text(filename)
    if region_slug in SOURCE_REGION_NAMES:
        filename = re.sub(
            r"^AgroStats_chart-[^-]+-",
            f"AgroStats_chart-{SOURCE_REGION_NAMES[region_slug]}-",
            filename,
            count=1,
        )

    match = _FERTILIZER_EXPORT_RE.match(filename)
    if not match:
        return filename

    suffix = match.group("suffix") or ("copy" if match.group("copy") else "")
    fert_type = FERTILIZER_SUFFIX_TO_TYPE[suffix]
    return f"{match.group('prefix')} {fert_type}.csv"


def import_region_exports(source_root: Path, target_root: Path) -> pd.DataFrame:
    """Copy downloaded region exports into canonical raw-data folders."""
    rows = []
    for source_name, slug in SOURCE_DIR_TO_SLUG.items():
        source_dir = source_root / source_name
        if not source_dir.exists():
            raise FileNotFoundError(source_dir)
        target_dir = target_root / slug
        if target_dir.exists():
            shutil.rmtree(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        seen: set[str] = set()
        for source_path in sorted(source_dir.glob("*.csv"), key=lambda path: path.stat().st_mtime):
            target_name = canonical_agrostats_name(source_path.name, region_slug=slug)
            if target_name in seen:
                raise ValueError(f"Duplicate canonical filename for {slug}: {target_name}")
            seen.add(target_name)
            target_path = target_dir / target_name
            shutil.copy2(source_path, target_path)
            rows.append(
                {
                    "region_slug": slug,
                    "source_path": f"{source_name}/{source_path.name}",
                    "target_path": f"data/raw/agrostats/{slug}/{target_name}",
                    "source_name": source_path.name,
                    "target_name": target_name,
                }
            )
    return pd.DataFrame(rows)


def build_data_inventory(raw_root: Path, regions: Iterable[str] = EXPECTED_REGIONS) -> pd.DataFrame:
    """Summarise expected AgroStats export coverage by region folder."""
    rows = []
    for slug in regions:
        raw_dir = raw_root / slug
        files = sorted(raw_dir.glob("AgroStats_chart-*.csv")) if raw_dir.exists() else []
        names = [path.name for path in files]
        rows.append(
            {
                "region_slug": slug,
                "region_label": REGION_LABELS.get(slug, slug),
                "file_count": len(files),
                "expected_file_count": EXPECTED_FILES_PER_REGION,
                "complete": len(files) == EXPECTED_FILES_PER_REGION,
                "yield_files": sum("-Урожайність-" in name for name in names),
                "area_files": sum("-Посівна площа-" in name for name in names),
                "fertilizer_files": sum("-Добрива-" in name for name in names),
                "irrigation_files": sum("-Зрошення_" in name for name in names),
            }
        )
    return pd.DataFrame(rows)


def build_threshold_sensitivity(
    recommended_inputs: pd.DataFrame,
    *,
    margins: Iterable[float] = THRESHOLD_MARGINS,
) -> pd.DataFrame:
    """Re-run the baseline-first decision rule over practical-margin values."""
    rows = []
    for region, group in recommended_inputs.groupby("region_slug"):
        leaderboard = group[
            ["crop", "model", "best_ml_mae"]
        ].rename(columns={"best_ml_mae": "mae"})
        baselines = group[
            ["crop", "best_baseline_raw", "best_baseline_mae"]
        ].rename(columns={"best_baseline_raw": "baseline", "best_baseline_mae": "mae"})
        baselines["rmse"] = baselines["mae"]
        baselines["mape"] = 0.0
        bands = group[["crop", "test_coverage"]].copy()
        for margin in margins:
            selected = select_recommended_methods(
                leaderboard,
                baselines,
                bands,
                practical_margin=float(margin),
            )
            selected.insert(0, "region_slug", region)
            selected.insert(1, "region_label", REGION_LABELS.get(region, region))
            selected.insert(2, "threshold_margin", float(margin))
            rows.append(selected)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_novelty_evidence_table(cards: pd.DataFrame) -> pd.DataFrame:
    """Create a compact publication-facing decision evidence table."""
    rows = []
    for _, row in cards.iterrows():
        if row["recommended_type"] == "baseline":
            reason = "ML did not clear the practical gain rule; transparent baseline remains safer."
        elif row["warning_label"] == "outside validation error scale":
            reason = "ML clears the gain rule, but validation-residual coverage warns against overclaiming."
        else:
            reason = "ML clears the gain rule and test errors mostly stay within the validation-residual band."
        rows.append(
            {
                "region": row["region_label"],
                "crop": row["crop_en"],
                "best_baseline": row["best_baseline_method"],
                "best_ml": row["best_ml_model"],
                "practical_gain_t_ha": row["ml_gain_vs_baseline"],
                "recommended_method": row["recommended_method"],
                "reliability_label": row["warning_label"],
                "reason": reason,
            }
        )
    return pd.DataFrame(rows)


def build_region_comparison_summary(cards: pd.DataFrame) -> pd.DataFrame:
    """Summarise how often each region recommends ML or baseline."""
    return (
        cards.groupby(["region_slug", "region_label"], as_index=False)
        .agg(
            crops=("crop_en", "count"),
            ml_recommendations=("recommended_type", lambda values: int((values == "machine_learning").sum())),
            baseline_recommendations=("recommended_type", lambda values: int((values == "baseline").sum())),
            mean_test_coverage=("test_coverage", "mean"),
            outside_error_scale=("warning_label", lambda values: int((values == "outside validation error scale").sum())),
            baseline_safer=("warning_label", lambda values: int((values == "baseline safer").sum())),
        )
        .sort_values("region_slug")
        .reset_index(drop=True)
    )


def write_multi_region_summary(cards: pd.DataFrame, output_path: Path) -> None:
    """Write a short human-readable multi-region reliability summary."""
    lines = [
        "# Multi-region reliability summary",
        "",
        "This external check applies the same baseline-first decision rule to Poltava, Vinnytsia, Cherkasy, and Ukraine.",
        "Baseline methods are control models; ML is recommended only when it clears the practical MAE margin.",
        "",
    ]
    for region, region_df in cards.groupby("region_label", sort=True):
        lines.extend([f"## {region}", ""])
        for _, row in region_df.sort_values("crop_en").iterrows():
            lines.append(
                f"- {row['crop_en'].title()}: {row['recommended_method']} "
                f"({row['warning_label']}), gain {row['ml_gain_vs_baseline']:.2f} t/ha, "
                f"coverage {row['test_coverage']:.1%}."
            )
        lines.append("")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
