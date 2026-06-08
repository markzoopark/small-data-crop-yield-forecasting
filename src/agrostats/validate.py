"""Data quality validation helpers and CLI."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
import typer
from rich.console import Console

from agrostats import utils


console = Console()
app = typer.Typer(help="Validation CLI for agrostats datasets.")


VALIDATION_REPORT_PATH = Path("reports/validation.md")
FEATURES_PATH = Path("data/processed/agrostats_poltava_features.parquet")
EXPECTED_YEARS = list(range(2010, 2025))
KEY_SERIES = [
    ("Урожайність", "Пшениця"),
    ("Урожайність", "Кукурудза"),
    ("Урожайність", "Соняшник"),
    ("Посівна площа", "Всі культури"),
]
FEATURE_TARGET_CROPS = ("Пшениця", "Кукурудза", "Соняшник")
ALLOWED_UNITS = {"t/ha", "ha", "kg/ha", "share", "m3/ha", "mm"}
UNIT_CANONICAL = {
    None: None,
    "т/га": "t/ha",
    "t/ha": "t/ha",
    "га": "ha",
    "ha": "ha",
    "кг/га": "kg/ha",
    "кг N/га": "kg/ha",
    "кг P2O5/га": "kg/ha",
    "кг K2O/га": "kg/ha",
    "kg/ha": "kg/ha",
    "кг/га ": "kg/ha",
    "доля": "share",
    "share": "share",
    "м³/га": "m3/ha",
    "м3/га": "m3/ha",
    "m3/ha": "m3/ha",
    "мм": "mm",
    "mm": "mm",
}


def require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    """Raise an error if mandatory columns are missing."""
    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def check_year_coverage(df: pd.DataFrame) -> List[dict[str, object]]:
    """Ensure key series cover all expected years without gaps."""
    issues: List[dict[str, object]] = []
    for metric, crop in KEY_SERIES:
        subset = df[(df["metric"] == metric) & (df["group_or_crop"] == crop)]
        if subset.empty:
            issues.append(
                {
                    "metric": metric,
                    "group_or_crop": crop,
                    "reason": "series_missing",
                    "missing_years": EXPECTED_YEARS,
                }
            )
            continue
        years_present = set(int(year) for year in subset["year"].dropna().astype(int))
        missing_years = [year for year in EXPECTED_YEARS if year not in years_present]
        if missing_years:
            issues.append(
                {
                    "metric": metric,
                    "group_or_crop": crop,
                    "reason": "missing_years",
                    "missing_years": missing_years,
                }
            )
    return issues


def canonical_unit(value: Optional[str]) -> Optional[str]:
    """Map raw unit names to canonical representations."""
    if value is None:
        return None
    return UNIT_CANONICAL.get(value.strip(), value.strip())


def check_unit_values(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows where unit_norm does not match the allowed set."""
    units = df["unit_norm"].apply(canonical_unit)
    invalid_mask = ~units.isin(ALLOWED_UNITS)
    invalid_rows = df.loc[invalid_mask, ["metric", "group_or_crop", "fert_type", "year", "unit_norm"]].copy()
    invalid_rows["unit_norm"] = invalid_rows["unit_norm"].fillna("∅")
    invalid_rows["canonical"] = units.loc[invalid_mask].fillna("∅")
    return invalid_rows


def detect_yield_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Detect outliers in yield (value_norm, metric=Урожайність) using the IQR rule."""
    subset = df[(df["metric"] == "Урожайність") & (~df["value_norm"].isna())].copy()
    if subset.empty:
        return pd.DataFrame(columns=["group_or_crop", "year", "value_norm", "lower_bound", "upper_bound", "iqr"])

    records: List[dict[str, object]] = []
    for crop, crop_df in subset.groupby("group_or_crop"):
        values = crop_df["value_norm"].astype(float).dropna()
        if values.count() < 4:
            continue
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (crop_df["value_norm"] < lower) | (crop_df["value_norm"] > upper)
        for _, row in crop_df[mask].iterrows():
            records.append(
                {
                    "group_or_crop": crop,
                    "year": int(row["year"]),
                    "value_norm": float(row["value_norm"]),
                    "lower_bound": lower,
                    "upper_bound": upper,
                    "iqr": iqr,
                }
            )
    if not records:
        return pd.DataFrame(columns=["group_or_crop", "year", "value_norm", "lower_bound", "upper_bound", "iqr"])
    return pd.DataFrame(records).sort_values(["group_or_crop", "year"])


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Convert a dataframe to a markdown-friendly table."""
    if df.empty:
        return "_No records._"
    return df.to_string(index=False)


def _format_problem_years(df: pd.DataFrame, mask: pd.Series) -> str:
    if not mask.any():
        return ""
    subset = df.loc[mask, ["group_or_crop", "year"]].dropna()
    if subset.empty:
        return ""
    subset["year"] = subset["year"].astype(int)
    grouped = subset.groupby("group_or_crop")["year"].apply(lambda years: sorted(set(years)))
    return ", ".join(f"{crop}: {years}" for crop, years in grouped.items())


def check_feature_constraints(df: pd.DataFrame) -> List[str]:
    """Validate feature dataset ranges and structural assumptions."""
    errors: List[str] = []
    required_columns = {
        "Yield_t_ha",
        "Area_ha",
        "N_kg_ha",
        "P2O5_kg_ha",
        "K_kg_ha",
        "Mineral_treated_share",
        "Irrig_mm",
        "group_or_crop",
        "year",
    }
    missing = required_columns - set(df.columns)
    if missing:
        return [f"Required feature columns are missing: {sorted(missing)}"]

    def check_range(column: str, lower: Optional[float] = None, upper: Optional[float] = None, inclusive: bool = True) -> None:
        series = pd.to_numeric(df[column], errors="coerce")
        valid = series.notna()
        mask = pd.Series(False, index=df.index)
        if lower is not None:
            if inclusive:
                mask |= valid & (series < lower)
            else:
                mask |= valid & (series <= lower)
        if upper is not None:
            if inclusive:
                mask |= valid & (series > upper)
            else:
                mask |= valid & (series >= upper)
        if mask.any():
            problems = _format_problem_years(df, mask)
            errors.append(f"{column}: values outside [{lower}, {upper}] — {problems}")

    # Range checks
    check_range("Yield_t_ha", 1, 12, inclusive=True)
    mask_area = pd.to_numeric(df["Area_ha"], errors="coerce") <= 0
    if mask_area.any():
        problems = _format_problem_years(df, mask_area)
        errors.append(f"Area_ha: non-positive values — {problems}")
    check_range("N_kg_ha", 0, 300, inclusive=True)
    check_range("P2O5_kg_ha", 0, 300, inclusive=True)
    check_range("K_kg_ha", 0, 300, inclusive=True)
    check_range("Mineral_treated_share", 0, 1, inclusive=True)
    check_range("Irrig_mm", 0, 50, inclusive=True)

    # Structural checks per crop
    for crop in FEATURE_TARGET_CROPS:
        subset = df[df["group_or_crop"] == crop]
        if subset.empty:
            errors.append(f"No data available for crop {crop}")
            continue
        years = sorted(int(year) for year in subset["year"].dropna().astype(int))
        expected_years = set(EXPECTED_YEARS)
        actual_years = set(years)
        missing_years = sorted(expected_years - actual_years)
        if missing_years:
            errors.append(f"{crop}: missing years {missing_years}")
        duplicates = subset.duplicated(subset=["group_or_crop", "year"], keep=False)
        if duplicates.any():
            dup_years = sorted(set(int(y) for y in subset.loc[duplicates, "year"]))
            errors.append(f"{crop}: duplicate entries for years {dup_years}")

    return errors


def build_report(
    year_issues: Sequence[dict[str, object]],
    invalid_units: pd.DataFrame,
    outliers: pd.DataFrame,
    feature_errors: Sequence[str],
    success: bool,
) -> str:
    """Compose a markdown report from validation results."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    status = "✅ PASSED" if success else "❌ FAILED"

    lines: List[str] = [
        f"# Validation Report ({timestamp})",
        "",
        f"Result: **{status}**",
        "",
        "## Year Coverage",
    ]
    if year_issues:
        for issue in year_issues:
            missing_years = ", ".join(str(year) for year in issue["missing_years"])
            lines.append(
                f"- **{issue['metric']} / {issue['group_or_crop']}** — missing years: {missing_years}."
            )
    else:
        lines.append("- All key series cover 2010–2024 without gaps.")

    lines.extend(
        [
            "",
            "## Unit Consistency",
        ]
    )
    if invalid_units.empty:
        lines.append("- All unit_norm values match the allowed units.")
    else:
        lines.append("- Rows with unsupported units were found:")
        lines.append("")
        lines.append("```")
        lines.append(dataframe_to_markdown(invalid_units))
        lines.append("```")

    lines.extend(
        [
            "",
            "## Yield Outliers (IQR)",
        ]
    )
    if outliers.empty:
        lines.append("- No yield outliers detected.")
    else:
        lines.append("- Potential yield outliers identified:")
        lines.append("")
        lines.append("```")
        lines.append(dataframe_to_markdown(outliers))
        lines.append("```")

    lines.extend(["", "## Feature Constraints"])
    if feature_errors:
        for err in feature_errors:
            lines.append(f"- {err}")
    else:
        lines.append("- Feature checks passed successfully.")

    return "\n".join(lines) + "\n"


def write_report(report: str, path: Path = VALIDATION_REPORT_PATH) -> None:
    """Persist the validation report to disk."""
    utils.ensure_directories([path.parent])
    path.write_text(report, encoding="utf-8")


def validate_agrostats(df: pd.DataFrame) -> Tuple[bool, str]:
    """Run the full agrostats validation suite."""
    required_columns = [
        "year",
        "region",
        "metric",
        "group_or_crop",
        "fert_type",
        "value_norm",
        "unit_norm",
    ]
    require_columns(df, required_columns)

    year_issues = check_year_coverage(df)
    invalid_units = check_unit_values(df)
    outliers = detect_yield_outliers(df)

    feature_errors: List[str] = []
    try:
        feature_df = pd.read_parquet(FEATURES_PATH)
        feature_errors.extend(check_feature_constraints(feature_df))
    except FileNotFoundError:
        feature_errors.append(f"Feature file not found: {FEATURES_PATH}")

    success = not year_issues and invalid_units.empty and not feature_errors
    report = build_report(year_issues, invalid_units, outliers, feature_errors, success)
    write_report(report)
    if feature_errors:
        message = "\n".join(feature_errors)
        return success, report, message
    return success, report, ""


@app.command("agrostats")
def validate_agrostats_command(
    path: Path = typer.Argument(Path("data/interim/agrostats_norm.parquet"), exists=True, file_okay=True),
) -> None:
    """Validate the normalized AgroStats dataset and persist a report."""
    df = pd.read_parquet(path)
    success, report, feature_message = validate_agrostats(df)
    console.print(report)
    status = "green" if success else "red"
    console.print(f"[{status}]Validation {'passed' if success else 'failed'}[/]")
    console.print(f"Report saved to {VALIDATION_REPORT_PATH}")
    if feature_message:
        console.print(f"[red]{feature_message}[/red]")
        raise AssertionError(feature_message)


if __name__ == "__main__":
    app()
