"""End-to-end orchestration script for the agrostats pipeline."""

from __future__ import annotations

import sys
from pathlib import Path
# Ensure src/ is on sys.path so `agrostats` package is importable.
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import typer
from rich.console import Console

from agrostats.features import OUTPUT_PARQUET
from agrostats import baselines, features, io, normalize, revision, train, validate


app = typer.Typer(help="Run the full AgroStats pipeline with a single command.")
console = Console()

DEFAULT_REGION = "poltava"
RAW_ROOT = Path("data/raw/agrostats")


def ensure_raw_dir(region_slug: str) -> Path:
    raw_dir = RAW_ROOT / region_slug
    if not raw_dir.exists() or not raw_dir.is_dir():
        raise FileNotFoundError(
            f"Raw directory '{raw_dir}' does not exist. "
            "Copy CSV exports into that folder or provide another slug via --region."
        )
    return raw_dir


def parse_languages(languages: str) -> list[str]:
    langs = [lang.strip() for lang in languages.split(",") if lang.strip()]
    return langs or ["uk"]


def pipeline(
    region: str,
    languages: str,
    run_baselines: bool,
) -> None:
    raw_dir = ensure_raw_dir(region)
    console.rule(f"[bold green]1. Loading CSVs from {raw_dir}")
    raw_df = io.load_folder(raw_dir)
    console.print(f"[green]Loaded {len(raw_df)} rows from {raw_dir}[/green]")

    console.rule("[bold green]2. Normalising units")
    norm_df = normalize.normalize_units(raw_df)
    console.print(f"[green]Normalised {len(norm_df)} rows → data/interim/agrostats_norm.parquet[/green]")

    console.rule("[bold green]3. Building features")
    features_df = features.build_features(norm_df)
    console.print(
        "[green]Feature set saved to data/processed/agrostats_poltava_features.parquet "
        f"({len(features_df)} rows)[/green]"
    )

    console.rule("[bold green]4. Validating integrity")
    success, report, feature_message = validate.validate_agrostats(norm_df)
    console.print(report)
    if not success:
        raise RuntimeError("Validation failed. See reports/validation.md for details.")
    if feature_message:
        raise RuntimeError(feature_message)

    console.rule("[bold green]5. Training models")
    language_list = parse_languages(languages)
    train_languages = ",".join(language_list)
    train.poltava_command(features_path=OUTPUT_PARQUET, languages=train_languages)

    if run_baselines:
        console.rule("[bold green]6. Excel baselines")
        baselines.main()

    console.rule("[bold green]7. Diagnostic reports")
    revision.main()

    console.rule("[bold green]Done")
    console.print("[bold green]All artefacts generated under reports/[/bold green]")


@app.command()
def main(
    region: str = typer.Option(
        DEFAULT_REGION,
        "--region",
        "-r",
        help="Region slug (Latin characters, e.g. 'poltava').",
    ),
    languages: str = typer.Option(
        "uk,en",
        "--languages",
        "-l",
        help="Comma-separated list of languages for plots/reports (e.g. 'uk,en').",
    ),
    skip_baselines: bool = typer.Option(
        False,
        "--skip-baselines",
        help="Skip running the Excel baseline comparison (scripts/baseline_excel.py).",
    ),
) -> None:
    """Run the full pipeline (ingest → normalise → features → validate → models → reports)."""
    pipeline(
        region=region,
        languages=languages,
        run_baselines=not skip_baselines,
    )


if __name__ == "__main__":
    app()
