"""Command-line interface for the project."""

from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from argentine_deputies_discursive_distance import __version__
from argentine_deputies_discursive_distance.discover import (
    DiscoveryError,
    discover_sessions,
)

DEFAULT_CONFIG_PATH = Path("config/pipeline.toml")
DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_QA_DIR = Path("data/qa")

app = typer.Typer(
    name="deputies-distance",
    help=("Build and analyze a corpus of speeches from Argentina's Chamber of Deputies."),
    no_args_is_help=True,
)

console = Console()


@app.callback()
def callback() -> None:
    """Build and analyze the Argentine Chamber of Deputies corpus."""


@app.command()
def version() -> None:
    """Display the installed project version."""
    console.print(__version__)


def _display_summary(summary: dict[str, Any]) -> None:
    table = Table(title="Session Discovery Summary")
    table.add_column("Metric")
    table.add_column("Value", justify="right")

    displayed_metrics = (
        ("Source SHA-256", "source_sha256"),
        ("Periods", "period_count"),
        ("All records", "record_count"),
        ("PDF available", "pdf_available_count"),
        ("No PDF link", "no_pdf_link_count"),
        ("Earliest date", "minimum_date"),
        ("Latest date", "maximum_date"),
        ("Candidate records", "candidate_record_count"),
        (
            "Candidate PDFs",
            "candidate_pdf_available_count",
        ),
        (
            "Candidate missing PDFs",
            "candidate_no_pdf_link_count",
        ),
        (
            "Candidate earliest date",
            "candidate_minimum_date",
        ),
        (
            "Candidate latest date",
            "candidate_maximum_date",
        ),
    )

    for label, key in displayed_metrics:
        table.add_row(label, str(summary[key]))

    console.print(table)


@app.command("discover-sessions")
def discover_sessions_command(
    config_path: Annotated[
        Path,
        typer.Option(
            "--config",
            help="Path to the pipeline TOML configuration.",
        ),
    ] = DEFAULT_CONFIG_PATH,
    raw_dir: Annotated[
        Path,
        typer.Option(
            "--raw-dir",
            help="Directory for source snapshots and manifests.",
        ),
    ] = DEFAULT_RAW_DIR,
    qa_dir: Annotated[
        Path,
        typer.Option(
            "--qa-dir",
            help="Directory for discovery quality summaries.",
        ),
    ] = DEFAULT_QA_DIR,
    html_path: Annotated[
        Path | None,
        typer.Option(
            "--html-path",
            help="Optional frozen HTML snapshot instead of a live request.",
        ),
    ] = None,
) -> None:
    """Discover and preserve entries from the official session index."""
    try:
        summary = discover_sessions(
            config_path=config_path,
            raw_dir=raw_dir,
            qa_dir=qa_dir,
            html_path=html_path,
        )
    except (DiscoveryError, OSError) as error:
        console.print(f"[bold red]Discovery failed:[/bold red] {error}")
        raise typer.Exit(code=1) from error

    _display_summary(summary)


def main() -> None:
    """Run the command-line application."""
    app()


if __name__ == "__main__":
    main()
