"""Climate data helpers for supplementary seasonal sensitivity analyses."""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import pandas as pd


POWER_PATH = Path("data/raw/agrostats/poltava/POWER_Point_Daily_20100101_20241231_049d59N_034d55E_LST.csv")


def _extract_power_csv_text(path: Path) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    start_idx = None
    for idx, line in enumerate(lines):
        if line.startswith("YEAR,DOY,"):
            start_idx = idx
            break
    if start_idx is None:
        raise ValueError(f"Could not find POWER CSV header in {path}")
    return "\n".join(lines[start_idx:])


def load_power_daily(path: Path = POWER_PATH) -> pd.DataFrame:
    csv_text = _extract_power_csv_text(path)
    df = pd.read_csv(StringIO(csv_text))
    df["YEAR"] = df["YEAR"].astype(int)
    df["DOY"] = df["DOY"].astype(int)
    df["date"] = pd.to_datetime(df["YEAR"].astype(str) + df["DOY"].astype(str), format="%Y%j")
    df["month"] = df["date"].dt.month
    return df


def aggregate_power_apr_sep(path: Path = POWER_PATH) -> pd.DataFrame:
    df = load_power_daily(path)
    subset = df[df["month"].between(4, 9)].copy()
    summary = (
        subset.groupby("YEAR", as_index=False)
        .agg(
            climate_apr_sep_t2m_mean=("T2M", "mean"),
            climate_apr_sep_t2m_max_mean=("T2M_MAX", "mean"),
            climate_apr_sep_prectotcorr_total=("PRECTOTCORR", "sum"),
            climate_apr_sep_hot_days_gt30=("T2M_MAX", lambda s: int((s > 30).sum())),
        )
        .rename(columns={"YEAR": "year"})
    )
    return summary


def merge_climate_features(features: pd.DataFrame, climate_summary: pd.DataFrame | None = None) -> pd.DataFrame:
    climate_summary = climate_summary if climate_summary is not None else aggregate_power_apr_sep()
    merged = features.merge(climate_summary, on="year", how="left")
    return merged
