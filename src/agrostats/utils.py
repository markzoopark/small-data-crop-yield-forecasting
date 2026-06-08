"""General utilities shared across agrostats modules."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, MutableMapping, Sequence

import json
import typer
from rich.console import Console
from rich.table import Table


console = Console()
app = typer.Typer(help="Auxiliary commands for agrostats workflows.")


def ensure_directories(paths: Iterable[Path]) -> None:
    """Create directory tree for each provided path if it does not exist."""
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def resolve_mapping(mapping: Mapping[str, str]) -> MutableMapping[str, str]:
    """Return a mutable copy of a mapping."""
    return dict(mapping)


def preview_mapping(mapping: Mapping[str, str]) -> None:
    """Render a mapping as a table in the console."""
    table = Table(title="Column Mapping", show_header=True, header_style="bold")
    table.add_column("Original")
    table.add_column("Renamed")
    for original, new in mapping.items():
        table.add_row(original, new)
    console.print(table)


@app.command("ensure-dirs")
def ensure_dirs_cli(paths: Sequence[Path]) -> None:
    """Typer wrapper that guarantees the existence of a list of directories."""
    ensure_directories(paths)
    console.print(f"[green]Ensured {len(paths)} directories[/green]")


@app.command("show-mapping")
def show_mapping_cli(mapping: str) -> None:
    """Pretty-print a JSON mapping provided via CLI."""
    data = json.loads(mapping)
    if not isinstance(data, dict):
        raise typer.BadParameter("Mapping JSON must describe an object.")
    preview_mapping(data)


if __name__ == "__main__":
    app()
