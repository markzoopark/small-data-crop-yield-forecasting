"""Input/output helpers for the agrostats project."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional

import pandas as pd
import typer
from rich.console import Console
from rich.progress import track

from agrostats import utils


console = Console()
app = typer.Typer(help="Utilities for working with raw AgroStats CSV exports.")


FILE_PATTERN = re.compile(
    r"^AgroStats_chart-(?P<region>[^-]+)-(?P<metric>[^-]+)(?:-(?P<group_or_crop>[^_]+))?_(?P<years>\d{4}-\d{4})(?: (?P<fert_type>[^.]+))?\.csv$"
)

FERT_TYPE_NORMALISATION = {
    "Калійні": "Калійні",
}


def _normalise_text(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = unicodedata.normalize("NFC", value.strip())
    return FERT_TYPE_NORMALISATION.get(value, value)


def _infer_unit(metric: str, group_or_crop: Optional[str], fert_type: Optional[str]) -> str:
    if metric == "Урожайність":
        return "ц/га"
    if metric == "Посівна площа":
        return "тис. га"
    if metric == "Зрошення":
        return "млн м³"
    if metric == "Добрива":
        is_all = (group_or_crop or "").strip() == "Всі культури"
        if fert_type == "Мінеральні":
            return "кг/га" if is_all else "тис. га"
        if fert_type == "Азотні":
            return "тис. т" if is_all else "кг N/га"
        if fert_type == "Фосфорні":
            return "тис. т" if is_all else "кг P2O5/га"
        if fert_type == "Калійні":
            return "тис. т" if is_all else "кг K2O/га"
        if fert_type == "Органічні":
            return "тис. т" if is_all else "тис. га"
    raise ValueError(f"Unable to infer unit_raw for metric={metric}, fert_type={fert_type}.")


def _parse_metadata(path: Path) -> Dict[str, Optional[str]]:
    match = FILE_PATTERN.match(path.name)
    if not match:
        raise ValueError(f"File name does not match expected pattern: {path.name}")
    data = match.groupdict()
    data["region"] = data["region"].strip()
    data["metric"] = data["metric"].strip()
    data["group_or_crop"] = (data.get("group_or_crop") or "").strip() or None
    fert_type = _normalise_text(data.get("fert_type"))
    data["fert_type"] = fert_type
    unit = _infer_unit(data["metric"], data["group_or_crop"], fert_type)
    data["unit_raw"] = unit
    return data


def load_csv(
    path: Path,
    *,
    rename: Optional[Mapping[str, str]] = None,
    dtype: Optional[Mapping[str, Any]] = None,
    parse_dates: Optional[Iterable[str]] = None,
    encoding: str = "utf-8",
) -> pd.DataFrame:
    """Load a CSV file and optionally rename columns."""
    df = pd.read_csv(path, dtype=dtype, parse_dates=parse_dates, encoding=encoding)
    if rename:
        df = df.rename(columns=dict(rename))
    return df


def _detect_columns(df: pd.DataFrame) -> tuple[str, str]:
    columns = [col.strip() for col in df.columns]
    year_candidates = []
    value_candidates = []
    for col in columns:
        series = df[col].astype(str).str.strip()
        if series.str.fullmatch(r"\d{4}").all():
            year_candidates.append(col)
        else:
            numeric_like = series.str.replace(",", ".", regex=False)
            numeric_like = numeric_like.str.replace(r"[^0-9eE+\-\.]", "", regex=True)
            if pd.to_numeric(numeric_like, errors="coerce").notna().sum() >= len(series) * 0.5:
                value_candidates.append(col)
    if not year_candidates:
        year_candidates = [columns[0]]
    if not value_candidates:
        value_candidates = [columns[1] if len(columns) > 1 else columns[0]]
    return year_candidates[0], value_candidates[0]


def read_agrostats_csv(path: Path) -> pd.DataFrame:
    """Read an AgroStats CSV file and enrich it with metadata parsed from the filename."""
    raw_df = pd.read_csv(path, dtype=str)
    if raw_df.empty:
        raise ValueError(f"File {path} does not contain any data.")

    year_column, value_column = _detect_columns(raw_df)

    year_series = raw_df[year_column].astype(str).str.extract(r"(\d{4})")[0]
    year = pd.to_numeric(year_series, errors="coerce").astype("Int64")

    value_str = raw_df[value_column].astype(str).str.strip()
    absent_mask = value_str.str.contains("дані відсутні", case=False, na=False)
    value_str = value_str.mask(absent_mask)
    value_numeric = (
        value_str.str.replace(",", ".", regex=False)
        .str.replace(r"[^0-9eE+\-\.]", "", regex=True)
        .replace("", pd.NA)
    )
    value = pd.to_numeric(value_numeric, errors="coerce")

    df = pd.DataFrame({"year": year, "value_raw": value})
    df = df.dropna(subset=["year", "value_raw"]).reset_index(drop=True)

    metadata = _parse_metadata(path)
    for key, value in metadata.items():
        df[key] = value
    df["source_path"] = str(path)
    return df[["region", "metric", "group_or_crop", "fert_type", "unit_raw", "year", "value_raw", "source_path"]]


def load_rename_mapping(source: Optional[Path]) -> MutableMapping[str, str]:
    """Load a column rename mapping from a JSON file."""
    if source is None:
        return {}
    data = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Rename mapping must be a JSON object.")
    return dict(data)


def rename_columns(df: pd.DataFrame, mapping: Mapping[str, str]) -> pd.DataFrame:
    """Return a dataframe with renamed columns."""
    if not mapping:
        return df
    missing_columns = set(mapping.keys()) - set(df.columns)
    if missing_columns:
        raise KeyError(f"Columns not found for renaming: {sorted(missing_columns)}")
    return df.rename(columns=mapping)


DEFAULT_REGION_SLUG = "poltava"
DEFAULT_RAW_DIR = Path("data/raw/agrostats") / DEFAULT_REGION_SLUG


def load_folder(raw_dir: Path = DEFAULT_RAW_DIR) -> pd.DataFrame:
    """Load all AgroStats exports from a folder into a single dataframe and persist to parquet."""
    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files were found in {raw_dir}.")

    frames = []
    for csv_path in track(csv_files, description="Reading AgroStats files"):
        if not FILE_PATTERN.match(csv_path.name):
            console.log(f"[yellow]Skipping file with unexpected name: {csv_path.name}[/yellow]")
            continue
        try:
            frame = read_agrostats_csv(csv_path)
        except Exception as exc:  # noqa: BLE001
            console.log(f"[red]Failed to read {csv_path.name}: {exc}[/red]")
            raise
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)

    key_columns = ["region", "year", "metric", "group_or_crop", "fert_type"]
    duplicates_mask = combined.duplicated(subset=key_columns, keep=False)
    if duplicates_mask.any():
        dup_records = combined.loc[duplicates_mask, key_columns + ["source_path"]]
        console.log("[red]Duplicate records detected:[/red]")
        console.log(dup_records)
        raise ValueError("Duplicates found for key (region, year, metric, group_or_crop, fert_type).")

    output_path = Path("data/interim/agrostats_raw.parquet")
    utils.ensure_directories([output_path.parent])
    combined.to_parquet(output_path, index=False)
    console.log(f"[green]Saved {len(combined)} rows to {output_path}[/green]")
    return combined


@app.command("preview")
def preview(path: Path, limit: int = 5) -> None:
    """Show the first rows of a CSV file."""
    df = load_csv(path)
    console.print(df.head(limit))


@app.command("rename")
def rename_command(
    path: Path,
    output: Optional[Path] = typer.Option(None, help="Where to write the transformed CSV."),
    mapping_path: Optional[Path] = typer.Option(None, help="JSON mapping file."),
    limit: int = typer.Option(0, help="Optional row limit to validate output."),
) -> None:
    """Rename columns according to a provided mapping."""
    mapping = load_rename_mapping(mapping_path)
    if mapping:
        utils.preview_mapping(mapping)
    df = load_csv(path, rename=mapping)
    if limit:
        console.print(df.head(limit))
    if output:
        utils.ensure_directories([output.parent])
        df.to_csv(output, index=False)
        console.print(f"[green]Saved renamed file to {output}[/green]")


@app.command("split")
def split_command(
    path: Path,
    output_dir: Path,
    chunk_size: int = typer.Option(100000, help="Number of rows per chunk."),
) -> None:
    """Split a large CSV into several files with the same header."""
    utils.ensure_directories([output_dir])
    for idx, chunk in enumerate(track(pd.read_csv(path, chunksize=chunk_size), description="Splitting CSV")):
        chunk_path = output_dir / f"{path.stem}_{idx:03d}.csv"
        chunk.to_csv(chunk_path, index=False)
    console.print(f"[green]Finished splitting {path}[/green]")


@app.command("load-folder")
def load_folder_command(
    raw_dir: Path = typer.Argument(DEFAULT_RAW_DIR, exists=True, dir_okay=True),
) -> None:
    """CLI wrapper for load_folder."""
    df = load_folder(raw_dir)
    console.print(df.head())


if __name__ == "__main__":
    app()
