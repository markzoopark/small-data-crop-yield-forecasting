"""Utilities for harmonising measurement units."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd
import typer
from rich.console import Console

from agrostats import utils


console = Console()
app = typer.Typer(help="Unit normalisation commands.")

NORMALISED_SHARE_UNIT = "share"
AREA_GROUP_ALIASES: dict[str, str] = {
    "Зернові": "Всі зернові культури",
    "Кормові": "Всі кормові",
    "Технічні": "Всі технічні культури",
}
AREA_GROUP_ALIASES: dict[str, str] = {
    "Зернові": "Всі зернові культури",
    "Кормові": "Всі кормові",
    "Технічні": "Всі технічні культури",
}


def yield_centner_to_tonnes_per_ha(series: pd.Series) -> pd.Series:
    """Convert yield from c/ha to t/ha."""
    return series / 10.0


def area_thousand_ha_to_ha(series: pd.Series) -> pd.Series:
    """Convert area from thousand hectares to hectares."""
    return series * 1000.0


def treated_share_thousand_ha(value: pd.Series, area_ha: pd.Series) -> pd.Series:
    """Convert treated area recorded in thousand hectares into share of total area."""
    share = (value * 1000.0) / area_ha.replace(0, pd.NA)
    return share


def mass_thousand_tonnes_to_kg_per_ha(value: pd.Series, area_ha_all: pd.Series) -> pd.Series:
    """Convert a mass recorded in thousand tonnes into kilograms per hectare."""
    return (value * 1_000_000.0) / area_ha_all.replace(0, pd.NA)


def irrigation_mln_m3_to_metrics(value: pd.Series, area_ha_all: pd.Series) -> pd.DataFrame:
    """Convert irrigation volume in million cubic metres to m3/ha and mm."""
    m3_per_ha = (value * 1_000_000.0) / area_ha_all.replace(0, pd.NA)
    mm = m3_per_ha / 10.0
    return pd.DataFrame({"m3_per_ha": m3_per_ha, "mm": mm})


def convert_column(
    df: pd.DataFrame,
    *,
    column: str,
    transformer,
    target_column: Optional[str] = None,
) -> pd.DataFrame:
    """Apply a transformer to a column and store the result."""
    target = target_column or column
    df[target] = transformer(df[column])
    return df


def _ensure_required_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Required columns are missing from the table: {missing}")


def _build_area_map(df: pd.DataFrame) -> pd.Series:
    area_mask = df["metric"] == "Посівна площа"
    area_df = df.loc[area_mask, ["region", "group_or_crop", "year", "value_norm"]].copy()
    area_df = area_df.dropna(subset=["value_norm"])
    if area_df.empty:
        raise ValueError("Sown area data could not be found.")
    index_columns = ["region", "group_or_crop", "year"]
    duplicated_mask = area_df.duplicated(subset=index_columns, keep=False)
    if duplicated_mask.any():
        duplicates = area_df.loc[duplicated_mask, index_columns]
        raise ValueError(f"Duplicate area values detected:\n{duplicates}")
    area_map = area_df.set_index(index_columns)["value_norm"]
    return area_map


def normalize_units(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Harmonise measurement units for the raw AgroStats dataset."""
    required_columns = [
        "year",
        "region",
        "metric",
        "group_or_crop",
        "fert_type",
        "value_raw",
        "unit_raw",
    ]
    _ensure_required_columns(df_raw, required_columns)

    df = df_raw.copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["value_raw"] = pd.to_numeric(df["value_raw"], errors="coerce")

    value_norm = pd.Series(index=df.index, dtype="float64")
    unit_norm = pd.Series(index=df.index, dtype="object")

    # Урожайность: ц/га → т/га
    mask_yield = df["metric"] == "Урожайність"
    if mask_yield.any():
        value_norm.loc[mask_yield] = df.loc[mask_yield, "value_raw"] / 10.0
        unit_norm.loc[mask_yield] = "t/ha"

    # Посевная площадь: тис. га → га
    mask_area = df["metric"] == "Посівна площа"
    if mask_area.any():
        value_norm.loc[mask_area] = df.loc[mask_area, "value_raw"] * 1000.0
        unit_norm.loc[mask_area] = "ha"

    area_map = _build_area_map(
        pd.DataFrame(
            {
                "metric": df["metric"],
                "region": df["region"],
                "group_or_crop": df["group_or_crop"],
                "year": df["year"],
                "value_norm": value_norm,
            }
        )
    )

    missing_area_records: List[dict[str, object]] = []
    fallback_area: dict[tuple[str, Optional[str], int], dict[str, object]] = {}

    def fetch_area(region: str, group: Optional[str], year_value) -> Optional[float]:
        if pd.isna(year_value):
            missing_area_records.append(
                {"region": region, "group_or_crop": group, "year": None, "reason": "year_missing"}
            )
            return None
        year = int(year_value)
        candidates: list[Optional[str]] = []
        if group:
            candidates.append(group)
            alias_group = AREA_GROUP_ALIASES.get(group)
            if alias_group:
                candidates.append(alias_group)
        else:
            candidates.append(None)
        if "Всі культури" not in candidates:
            candidates.append("Всі культури")

        seen: set[Optional[str]] = set()
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            key = (region, candidate, year)
            if key in area_map.index:
                area_value = float(area_map.loc[key])
                if area_value <= 0:
                    missing_area_records.append(
                        {
                            "region": region,
                            "group_or_crop": candidate,
                            "year": year,
                            "reason": "non_positive_area",
                        }
                    )
                    return None
                return area_value
            # Try to reuse the most recent available area value.
            try:
                candidate_series = area_map.loc[(region, candidate)]
            except KeyError:
                candidate_series = None
            if isinstance(candidate_series, pd.Series) and not candidate_series.empty:
                prev_years = [candidate_year for candidate_year in candidate_series.index if candidate_year < year]
                if prev_years:
                    fallback_year = max(prev_years)
                    area_value = float(candidate_series.loc[fallback_year])
                    if area_value <= 0:
                        missing_area_records.append(
                            {
                                "region": region,
                                "group_or_crop": candidate,
                                "year": fallback_year,
                                "reason": "non_positive_area",
                            }
                        )
                        return None
                    key_with_year = (region, candidate, year)
                    if key_with_year not in fallback_area:
                        fallback_area[key_with_year] = {
                            "fallback_year": fallback_year,
                            "value_norm": area_value,
                        }
                    missing_area_records.append(
                        {
                            "region": region,
                            "group_or_crop": candidate,
                            "year": year,
                            "reason": "fallback_previous_year",
                            "fallback_year": fallback_year,
                        }
                    )
                    return area_value

        missing_area_records.append({"region": region, "group_or_crop": group, "year": year, "reason": "not_found"})
        return None

    # Mineral fertilisers: thousand ha → share
    mask_mineral = (df["metric"] == "Добрива") & (df["fert_type"] == "Мінеральні")
    mask_mineral_all = mask_mineral & (df["group_or_crop"] == "Всі культури")
    mask_mineral_specific = mask_mineral & ~mask_mineral_all

    if mask_mineral_specific.any():
        def _mineral_share(row):
            if pd.isna(row["value_raw"]):
                return float("nan")
            area_value = fetch_area(row["region"], row["group_or_crop"], row["year"])
            if area_value is None or pd.isna(area_value):
                return float("nan")
            return (row["value_raw"] * 1000.0) / area_value

        value_norm.loc[mask_mineral_specific] = df.loc[mask_mineral_specific].apply(_mineral_share, axis=1)
        unit_norm.loc[mask_mineral_specific] = NORMALISED_SHARE_UNIT

    if mask_mineral_all.any():
        value_norm.loc[mask_mineral_all] = df.loc[mask_mineral_all, "value_raw"]
        unit_norm.loc[mask_mineral_all] = "kg/ha"

    # Nitrogen fertilisers
    mask_nitrogen = (df["metric"] == "Добрива") & (df["fert_type"] == "Азотні")
    mask_nitrogen_all = mask_nitrogen & (df["group_or_crop"] == "Всі культури")
    mask_nitrogen_specific = mask_nitrogen & ~mask_nitrogen_all

    if mask_nitrogen_specific.any():
        value_norm.loc[mask_nitrogen_specific] = df.loc[mask_nitrogen_specific, "value_raw"]
        unit_norm.loc[mask_nitrogen_specific] = "kg/ha"

    if mask_nitrogen_all.any():
        def _nitrogen_all(row):
            if pd.isna(row["value_raw"]):
                return float("nan")
            area_value = fetch_area(row["region"], "Всі культури", row["year"])
            if area_value is None or pd.isna(area_value):
                return float("nan")
            return (row["value_raw"] * 1_000_000.0) / area_value

        value_norm.loc[mask_nitrogen_all] = df.loc[mask_nitrogen_all].apply(_nitrogen_all, axis=1)
        unit_norm.loc[mask_nitrogen_all] = "kg/ha"

    # Phosphorus fertilisers
    mask_phosphorus = (df["metric"] == "Добрива") & (df["fert_type"] == "Фосфорні")
    mask_phosphorus_all = mask_phosphorus & (df["group_or_crop"] == "Всі культури")
    mask_phosphorus_specific = mask_phosphorus & ~mask_phosphorus_all

    if mask_phosphorus_specific.any():
        value_norm.loc[mask_phosphorus_specific] = df.loc[mask_phosphorus_specific, "value_raw"]
        unit_norm.loc[mask_phosphorus_specific] = "kg/ha"

    if mask_phosphorus_all.any():
        def _phosphorus_all(row):
            if pd.isna(row["value_raw"]):
                return float("nan")
            area_value = fetch_area(row["region"], "Всі культури", row["year"])
            if area_value is None or pd.isna(area_value):
                return float("nan")
            return (row["value_raw"] * 1_000_000.0) / area_value

        value_norm.loc[mask_phosphorus_all] = df.loc[mask_phosphorus_all].apply(_phosphorus_all, axis=1)
        unit_norm.loc[mask_phosphorus_all] = "kg/ha"

    # Potassium fertilisers
    mask_potassium = (df["metric"] == "Добрива") & (df["fert_type"] == "Калійні")
    mask_potassium_all = mask_potassium & (df["group_or_crop"] == "Всі культури")
    mask_potassium_specific = mask_potassium & ~mask_potassium_all

    if mask_potassium_specific.any():
        value_norm.loc[mask_potassium_specific] = df.loc[mask_potassium_specific, "value_raw"]
        unit_norm.loc[mask_potassium_specific] = "kg/ha"

    if mask_potassium_all.any():
        def _potassium_all(row):
            if pd.isna(row["value_raw"]):
                return float("nan")
            area_value = fetch_area(row["region"], "Всі культури", row["year"])
            if area_value is None or pd.isna(area_value):
                return float("nan")
            return (row["value_raw"] * 1_000_000.0) / area_value

        value_norm.loc[mask_potassium_all] = df.loc[mask_potassium_all].apply(_potassium_all, axis=1)
        unit_norm.loc[mask_potassium_all] = "kg/ha"

    # Organic fertilisers
    mask_organic = (df["metric"] == "Добрива") & (df["fert_type"] == "Органічні")
    mask_organic_share = mask_organic & (df["unit_raw"] == "тис. га")
    mask_organic_mass = mask_organic & (df["unit_raw"] == "тис. т")

    if mask_organic_share.any():
        def _organic_share(row):
            if pd.isna(row["value_raw"]):
                return float("nan")
            area_value = fetch_area(row["region"], row["group_or_crop"], row["year"])
            if area_value is None or pd.isna(area_value):
                return float("nan")
            return (row["value_raw"] * 1000.0) / area_value

        value_norm.loc[mask_organic_share] = df.loc[mask_organic_share].apply(_organic_share, axis=1)
        unit_norm.loc[mask_organic_share] = NORMALISED_SHARE_UNIT

    if mask_organic_mass.any():
        def _organic_mass(row):
            if pd.isna(row["value_raw"]):
                return float("nan")
            area_value = fetch_area(row["region"], "Всі культури", row["year"])
            if area_value is None or pd.isna(area_value):
                return float("nan")
            return (row["value_raw"] * 1_000_000.0) / area_value

        value_norm.loc[mask_organic_mass] = df.loc[mask_organic_mass].apply(_organic_mass, axis=1)
        unit_norm.loc[mask_organic_mass] = "kg/ha"

    # Irrigation
    mask_irrigation = df["metric"] == "Зрошення"
    irrigation_extras: list[pd.DataFrame] = []
    if mask_irrigation.any():
        def _irrigation_m3(row):
            if pd.isna(row["value_raw"]):
                return float("nan")
            area_value = fetch_area(row["region"], "Всі культури", row["year"])
            if area_value is None or pd.isna(area_value):
                return float("nan")
            return (row["value_raw"] * 1_000_000.0) / area_value

        m3_per_ha = df.loc[mask_irrigation].apply(_irrigation_m3, axis=1)
        value_norm.loc[mask_irrigation] = m3_per_ha
        unit_norm.loc[mask_irrigation] = "m3/ha"

        irr_mm = df.loc[mask_irrigation].copy()
        irr_mm["value_norm"] = m3_per_ha / 10.0
        irr_mm["unit_norm"] = "mm"
        irrigation_extras.append(irr_mm)

    # Remaining categories: keep original values.
    remaining_mask = unit_norm.isna()
    if remaining_mask.any():
        value_norm.loc[remaining_mask] = df.loc[remaining_mask, "value_raw"]
        unit_norm.loc[remaining_mask] = df.loc[remaining_mask, "unit_raw"]

    df["value_norm"] = value_norm
    df["unit_norm"] = unit_norm

    frames = [df]
    if irrigation_extras:
        frames.extend(irrigation_extras)

    # Add synthetic area rows for cases where fallback values were used.
    fallback_rows: List[dict[str, object]] = []
    for (region, group, year), info in fallback_area.items():
        area_value = info.get("value_norm")
        if area_value is None or pd.isna(area_value):
            continue
        mask_existing = (
            (df["metric"] == "Посівна площа")
            & (df["region"] == region)
            & (df["group_or_crop"] == group)
            & (df["year"] == year)
        )
        if mask_existing.any():
            continue
        raw_value = area_value / 1000.0
        fallback_rows.append(
            {
                "region": region,
                "metric": "Посівна площа",
                "group_or_crop": group,
                "fert_type": pd.NA,
                "unit_raw": "тис. га",
                "year": year,
                "value_raw": raw_value,
                "value_norm": area_value,
                "unit_norm": "ha",
                "source_path": f"fallback::{info.get('fallback_year')}",
            }
        )
    if fallback_rows:
        frames.append(pd.DataFrame(fallback_rows))

    result = pd.concat(frames, ignore_index=True)

    if missing_area_records:
        missing_df = pd.DataFrame(missing_area_records).drop_duplicates()
        missing_df = missing_df.sort_values(["region", "group_or_crop", "year"], na_position="last")
        console.log("[yellow]Missing area data prevented full normalisation for the following combinations:[/yellow]")
        console.log(missing_df)

    result = result[
        [
            "year",
            "region",
            "metric",
            "group_or_crop",
            "fert_type",
            "value_raw",
            "unit_raw",
            "value_norm",
            "unit_norm",
        ]
    ].sort_values(["region", "metric", "group_or_crop", "fert_type", "year", "unit_norm"], na_position="last")

    output_path = Path("data/interim/agrostats_norm.parquet")
    utils.ensure_directories([output_path.parent])
    result.to_parquet(output_path, index=False)
    console.log(f"[green]Normalised dataset saved to {output_path}[/green]")
    return result


@app.command("yield")
def yield_command(
    path: Path,
    value_column: str,
    output_column: str = typer.Option("yield_t_ha", help="Destination column for the converted value."),
    output: Optional[Path] = typer.Option(None, help="Optional file to save the updated data."),
) -> None:
    """Convert yield from centners per hectare to tonnes per hectare."""
    df = pd.read_csv(path)
    convert_column(df, column=value_column, transformer=yield_centner_to_tonnes_per_ha, target_column=output_column)
    console.print(df[[value_column, output_column]].head())
    if output:
        utils.ensure_directories([output.parent])
        df.to_csv(output, index=False)
        console.print(f"[green]Saved normalised data to {output}[/green]")


@app.command("area")
def area_command(
    path: Path,
    value_column: str,
    output_column: str = typer.Option("area_ha", help="Destination column name."),
    output: Optional[Path] = typer.Option(None, help="Optional file to save the updated data."),
) -> None:
    """Convert area recorded in thousand hectares to hectares."""
    df = pd.read_csv(path)
    convert_column(df, column=value_column, transformer=area_thousand_ha_to_ha, target_column=output_column)
    console.print(df[[value_column, output_column]].head())
    if output:
        utils.ensure_directories([output.parent])
        df.to_csv(output, index=False)
        console.print(f"[green]Saved normalised data to {output}[/green]")


@app.command("mineral-share")
def mineral_share_command(
    path: Path,
    treated_column: str,
    area_column: str,
    output_column: str = typer.Option("Mineral_treated_share", help="Destination column name."),
    output: Optional[Path] = typer.Option(None, help="Optional file to save the updated data."),
) -> None:
    """Compute treated share for mineral fertilisers."""
    df = pd.read_csv(path)
    df[output_column] = treated_share_thousand_ha(df[treated_column], df[area_column])
    console.print(df[[treated_column, area_column, output_column]].head())
    if output:
        utils.ensure_directories([output.parent])
        df.to_csv(output, index=False)
        console.print(f"[green]Saved normalised data to {output}[/green]")


@app.command("mass")
def mass_command(
    path: Path,
    value_column: str,
    area_all_column: str,
    output_column: str = typer.Option("kg_per_ha", help="Destination column name."),
    output: Optional[Path] = typer.Option(None, help="Optional file to save the updated data."),
) -> None:
    """Convert thousand tonnes indicators to kilograms per hectare."""
    df = pd.read_csv(path)
    df[output_column] = mass_thousand_tonnes_to_kg_per_ha(df[value_column], df[area_all_column])
    console.print(df[[value_column, area_all_column, output_column]].head())
    if output:
        utils.ensure_directories([output.parent])
        df.to_csv(output, index=False)
        console.print(f"[green]Saved normalised data to {output}[/green]")


@app.command("irrigation")
def irrigation_command(
    path: Path,
    value_column: str,
    area_all_column: str,
    m3_column: str = typer.Option("m3_per_ha", help="Column for m3 per hectare."),
    mm_column: str = typer.Option("mm", help="Column for mm equivalent."),
    output: Optional[Path] = typer.Option(None, help="Optional file to save the updated data."),
) -> None:
    """Convert irrigation volume in million m³ to both m³/ha and mm."""
    df = pd.read_csv(path)
    metrics = irrigation_mln_m3_to_metrics(df[value_column], df[area_all_column])
    df[m3_column] = metrics["m3_per_ha"]
    df[mm_column] = metrics["mm"]
    console.print(df[[value_column, area_all_column, m3_column, mm_column]].head())
    if output:
        utils.ensure_directories([output.parent])
        df.to_csv(output, index=False)
        console.print(f"[green]Saved normalised data to {output}[/green]")


@app.command("agrostats")
def normalize_agrostats_command(
    raw_path: Path = typer.Argument(Path("data/interim/agrostats_raw.parquet"), exists=True, file_okay=True),
) -> None:
    """Normalise measurement units for the prepared AgroStats parquet dataset."""
    df_raw = pd.read_parquet(raw_path)
    df_norm = normalize_units(df_raw)
    console.print(df_norm.head())


if __name__ == "__main__":
    app()
