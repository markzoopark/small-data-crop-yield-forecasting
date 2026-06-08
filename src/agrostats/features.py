"""Feature engineering helpers for agronomic datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
import typer
from rich.console import Console

from agrostats import utils


console = Console()
app = typer.Typer(help="Feature engineering commands.")

TARGET_CROPS = {"Пшениця", "Кукурудза", "Соняшник"}
OUTPUT_PARQUET = Path("data/processed/agrostats_poltava_features.parquet")
OUTPUT_CSV = Path("data/processed/agrostats_poltava_features.csv")


def aggregate_features(
    df: pd.DataFrame,
    *,
    group_by: Sequence[str],
    aggregations: Mapping[str, Iterable[str]],
) -> pd.DataFrame:
    """Aggregate numeric columns using provided operations."""
    grouped = df.groupby(list(group_by)).agg(aggregations)
    grouped.columns = ["_".join(filter(None, map(str, col))).strip("_") for col in grouped.columns.to_flat_index()]
    return grouped.reset_index()


def add_ratios(df: pd.DataFrame, numerator: str, denominator: str, target: str) -> pd.DataFrame:
    """Create ratio feature with safe division."""
    df[target] = df[numerator] / df[denominator].replace(0, pd.NA)
    return df


def _select_metric(
    df: pd.DataFrame,
    *,
    metric: str,
    fert_type: Optional[str] = None,
    unit: Optional[str] = None,
    crops: Optional[set[str]] = None,
    prefer_unit_order: Optional[Sequence[str]] = None,
    include_group: bool = True,
) -> pd.DataFrame:
    """Extract metric rows with optional filtering and aggregation."""
    data = df[df["metric"] == metric]
    if fert_type is not None:
        data = data[data["fert_type"] == fert_type]
    if unit is not None:
        data = data[data["unit_norm"] == unit]
    if crops is not None and include_group:
        data = data[data["group_or_crop"].isin(crops)]

    if data.empty:
        return pd.DataFrame(columns=["region", "group_or_crop", "year", "value_norm", "unit_norm"])

    if prefer_unit_order:
        priority = {unit_name: idx for idx, unit_name in enumerate(prefer_unit_order)}
        data = data.assign(
            _priority=data["unit_norm"].map(priority).fillna(len(prefer_unit_order) if prefer_unit_order else 0)
        )
        data = data.sort_values(["region", "group_or_crop", "year", "_priority"])
        data = data.drop_duplicates(subset=["region", "group_or_crop", "year"], keep="first")
        data = data.drop(columns="_priority")

    group_cols = ["region", "group_or_crop", "year"] if include_group else ["region", "year"]
    aggregated = (
        data.groupby(group_cols, as_index=False)["value_norm"]
        .mean()
        .rename(columns={"value_norm": "value"})
    )
    if include_group and "group_or_crop" not in aggregated.columns:
        aggregated["group_or_crop"] = pd.NA
    return aggregated


def _merge_feature(df: pd.DataFrame, feature: pd.DataFrame, *, column_name: str, on: Sequence[str]) -> pd.DataFrame:
    if feature.empty:
        df[column_name] = pd.NA
        return df
    feature = feature.rename(columns={"value": column_name})
    return df.merge(feature, on=list(on), how="left")


def build_features(df_norm: pd.DataFrame) -> pd.DataFrame:
    """Assemble the feature set for key crops and persist processed outputs."""
    df = df_norm.copy()
    df = df[df["group_or_crop"].isin(TARGET_CROPS) | df["metric"].eq("Зрошення")]
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)

    yield_data = df[(df["metric"] == "Урожайність") & (df["group_or_crop"].isin(TARGET_CROPS))].copy()
    yield_base = (
        yield_data.groupby(["region", "group_or_crop", "year"], as_index=False)["value_norm"]
        .mean()
        .rename(columns={"value_norm": "Yield_t_ha"})
    )
    if yield_base.empty:
        raise ValueError("No yield measurements were found for the target crops.")

    features = yield_base.copy()

    area = _select_metric(df, metric="Посівна площа", crops=TARGET_CROPS)
    features = _merge_feature(features, area, column_name="Area_ha", on=["region", "group_or_crop", "year"])

    nitrogen = _select_metric(
        df,
        metric="Добрива",
        fert_type="Азотні",
        unit="kg/ha",
        crops=TARGET_CROPS,
    )
    features = _merge_feature(features, nitrogen, column_name="N_kg_ha", on=["region", "group_or_crop", "year"])

    phosphorus = _select_metric(
        df,
        metric="Добрива",
        fert_type="Фосфорні",
        unit="kg/ha",
        crops=TARGET_CROPS,
    )
    features = _merge_feature(features, phosphorus, column_name="P2O5_kg_ha", on=["region", "group_or_crop", "year"])

    potassium = _select_metric(
        df,
        metric="Добрива",
        fert_type="Калійні",
        unit="kg/ha",
        crops=TARGET_CROPS,
    )
    features = _merge_feature(features, potassium, column_name="K_kg_ha", on=["region", "group_or_crop", "year"])

    mineral_share = _select_metric(
        df,
        metric="Добрива",
        fert_type="Мінеральні",
        unit="share",
        crops=TARGET_CROPS,
    )
    features = _merge_feature(
        features,
        mineral_share,
        column_name="Mineral_treated_share",
        on=["region", "group_or_crop", "year"],
    )

    organics = _select_metric(
        df,
        metric="Добрива",
        fert_type="Органічні",
        crops=TARGET_CROPS,
        prefer_unit_order=("kg/ha", "share"),
    )
    features = _merge_feature(
        features,
        organics,
        column_name="Org_kg_ha_or_share",
        on=["region", "group_or_crop", "year"],
    )

    irrig_m3 = _select_metric(df, metric="Зрошення", unit="m3/ha", include_group=False)
    features = _merge_feature(features, irrig_m3, column_name="Irrig_m3_ha", on=["region", "year"])

    irrig_mm = _select_metric(df, metric="Зрошення", unit="mm", include_group=False)
    features = _merge_feature(features, irrig_mm, column_name="Irrig_mm", on=["region", "year"])

    features = features.sort_values(["group_or_crop", "year"]).reset_index(drop=True)

    factor_columns = [
        "Yield_t_ha",
        "Area_ha",
        "N_kg_ha",
        "P2O5_kg_ha",
        "K_kg_ha",
        "Mineral_treated_share",
        "Org_kg_ha_or_share",
        "Irrig_m3_ha",
        "Irrig_mm",
    ]

    for column in factor_columns:
        for lag in (1, 2, 3):
            lag_column = f"{column}_lag{lag}"
            features[lag_column] = features.groupby("group_or_crop")[column].shift(lag)

    rolling_map = {
        "Yield_t_ha": "ma5_Yield",
        "N_kg_ha": "ma5_N",
        "P2O5_kg_ha": "ma5_P2O5",
        "K_kg_ha": "ma5_K",
        "Mineral_treated_share": "ma5_MineralShare",
        "Org_kg_ha_or_share": "ma5_Org",
        "Irrig_m3_ha": "ma5_Irrig_m3",
        "Irrig_mm": "ma5_Irrig_mm",
    }

    for source, target in rolling_map.items():
        # Use only historical information for moving averages to avoid look-ahead leakage.
        features[target] = (
            features.groupby("group_or_crop")[source]
            .transform(lambda s: s.shift(1).rolling(window=5, min_periods=5).mean())
        )

    features["Yield_anom"] = features["Yield_t_ha"] - features["ma5_Yield"]

    output_columns: List[str] = [
        "region",
        "group_or_crop",
        "year",
        "Yield_t_ha",
        "Area_ha",
        "N_kg_ha",
        "P2O5_kg_ha",
        "K_kg_ha",
        "Mineral_treated_share",
        "Org_kg_ha_or_share",
        "Irrig_m3_ha",
        "Irrig_mm",
    ]
    for lag in (1, 2, 3):
        output_columns.extend([f"{col}_lag{lag}" for col in factor_columns])
    output_columns.extend(rolling_map.values())
    output_columns.append("Yield_anom")

    features = features[output_columns]

    utils.ensure_directories([OUTPUT_PARQUET.parent])
    features.to_parquet(OUTPUT_PARQUET, index=False)
    features.to_csv(OUTPUT_CSV, index=False)

    return features


@app.command("aggregate")
def aggregate_command(
    path: Path,
    output: Path,
    group_by: List[str] = typer.Argument(..., help="Grouping columns."),
    aggregations: List[str] = typer.Argument(..., help="Column specs in the form column:func."),
) -> None:
    """Aggregate a dataset using simple 'column:function' specifications."""
    df = pd.read_csv(path)
    mapping: Dict[str, set[str]] = {}
    for spec in aggregations:
        column, func = spec.split(":", 1)
        mapping.setdefault(column, set()).add(func)
    agg_df = aggregate_features(df, group_by=group_by, aggregations={k: sorted(v) for k, v in mapping.items()})
    utils.ensure_directories([output.parent])
    agg_df.to_csv(output, index=False)
    console.print(f"[green]Saved aggregated features to {output}[/green]")


@app.command("ratio")
def ratio_command(
    path: Path,
    numerator: str,
    denominator: str,
    target: str,
    output: Path,
) -> None:
    """Create a ratio feature and persist the result."""
    df = pd.read_csv(path)
    add_ratios(df, numerator=numerator, denominator=denominator, target=target)
    utils.ensure_directories([output.parent])
    df.to_csv(output, index=False)
    console.print(f"[green]Saved dataset with ratio '{target}' to {output}[/green]")


@app.command("build-poltava")
def build_poltava_command(
    path: Path = typer.Argument(Path("data/interim/agrostats_norm.parquet"), exists=True, file_okay=True),
) -> None:
    """Build the Poltava feature set from the normalised dataset."""
    df_norm = pd.read_parquet(path)
    features = build_features(df_norm)
    console.print(features.head())
    console.print(
        f"[green]Saved {len(features)} rows to {OUTPUT_PARQUET} and {OUTPUT_CSV}[/green]"
    )


if __name__ == "__main__":
    app()
