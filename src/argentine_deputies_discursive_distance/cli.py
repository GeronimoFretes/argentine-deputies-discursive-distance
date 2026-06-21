"""Command-line interface for the project."""

import typer
from rich.console import Console

from argentine_deputies_discursive_distance import __version__

app = typer.Typer(
    name="deputies-distance",
    help=(
        "Build and analyze a corpus of speeches from Argentina's "
        "Chamber of Deputies."
    ),
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


def main() -> None:
    """Run the command-line application."""
    app()


if __name__ == "__main__":
    main()