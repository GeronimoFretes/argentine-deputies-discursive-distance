"""Command-line interface for the project."""

from pathlib import Path
from typing import Annotated, Any

import httpx
import typer
from rich.console import Console
from rich.table import Table

from argentine_deputies_discursive_distance import __version__
from argentine_deputies_discursive_distance.corpus_profile import (
    DEFAULT_CONFIG_PATH as DEFAULT_CORPUS_PROFILE_CONFIG_PATH,
)
from argentine_deputies_discursive_distance.corpus_profile import (
    DEFAULT_CORPUS_LOCK_PATH,
    DEFAULT_DOCUMENTS_PATH,
    DEFAULT_EXPORT_MANIFEST_PATH,
    CorpusProfileError,
    profile_modeling_corpus,
)
from argentine_deputies_discursive_distance.corpus_profile import (
    DEFAULT_OUTPUT_DIR as DEFAULT_CORPUS_PROFILE_OUTPUT_DIR,
)
from argentine_deputies_discursive_distance.discover import (
    DiscoveryError,
    discover_sessions,
)
from argentine_deputies_discursive_distance.modeling_corpus import (
    DEFAULT_MAXIMUM_CHUNK_WORDS,
    DEFAULT_MINIMUM_WORDS,
    DEFAULT_MODELING_METADATA_PATH,
    DEFAULT_MODELING_OUTPUT_ROOT,
    DEFAULT_MODELING_OVERRIDES_PATH,
    DEFAULT_SPEAKER_TURN_ROOT,
    ModelingCorpusError,
    export_modeling_corpus,
)
from argentine_deputies_discursive_distance.pdf_batch import (
    PDF_USER_AGENT,
    PdfBatchError,
    run_pdf_batch,
)
from argentine_deputies_discursive_distance.structure_batch import (
    StructureBatchError,
    run_structure_batch,
)

DEFAULT_CONFIG_PATH = Path("config/pipeline.toml")
DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_QA_DIR = Path("data/qa")

DEFAULT_PDF_MANIFEST_PATH = Path("data/raw/session_manifest.csv")
DEFAULT_PDF_SELECTION_PATH = Path("config/pdf_pilot.csv")
DEFAULT_PDF_DIRECTORY = Path("data/raw/pdfs")
DEFAULT_PDF_EXTRACTION_ROOT = Path("data/interim/pdf_extraction")
DEFAULT_PDF_SUMMARY_PATH = Path("data/qa/pdf_extraction_summary.json")

DEFAULT_STRUCTURE_SOURCE_SUMMARY_PATH = DEFAULT_PDF_SUMMARY_PATH
DEFAULT_STRUCTURE_OUTPUT_ROOT = Path("data/interim/structural_segmentation")
DEFAULT_STRUCTURE_SUMMARY_PATH = Path("data/qa/structural_segmentation_summary.json")

app = typer.Typer(
    name="deputies-distance",
    help=("Build and analyze a corpus of speeches from Argentina's Chamber of Deputies."),
    no_args_is_help=True,
)

console = Console()


@app.callback()
def callback() -> None:
    """Build and analyze the Argentine Chamber of Deputies corpus."""


def _display_pdf_batch_summary(
    summary: dict[str, Any],
) -> None:
    """Display one row per processed PDF."""
    table = Table(title="PDF Extraction Batch")

    table.add_column("Label")
    table.add_column("Date")
    table.add_column(
        "Pages",
        justify="right",
    )
    table.add_column(
        "Words",
        justify="right",
    )
    table.add_column("Download")
    table.add_column("Extraction")
    table.add_column("Layouts")
    table.add_column(
        "Empty",
        justify="right",
    )

    for record in summary["records"]:
        layout_counts = record["layout_counts"]
        layouts = ", ".join(f"{layout}:{count}" for layout, count in layout_counts.items())

        table.add_row(
            str(record["label"]),
            str(record["session_date"]),
            str(record["page_count"]),
            f"{int(record['total_words']):,}",
            ("reused" if record["download_reused"] else "downloaded"),
            ("reused" if record["extraction_reused"] else "extracted"),
            layouts,
            str(len(record["empty_page_numbers"])),
        )

    console.print(table)
    console.print(f"Downloads reused: {summary['download_reused_count']}/{summary['record_count']}")
    console.print(
        f"Extractions reused: {summary['extraction_reused_count']}/{summary['record_count']}"
    )


def _display_structure_batch_summary(
    summary: dict[str, Any],
) -> None:
    """Display one row per segmented transcript."""
    table = Table(title="Structural Segmentation Batch")

    table.add_column("Label")
    table.add_column("Date")
    table.add_column("Start")
    table.add_column("End")
    table.add_column("Method")
    table.add_column(
        "Included",
        justify="right",
    )
    table.add_column(
        "Words",
        justify="right",
    )
    table.add_column(
        "Zone mix",
        justify="right",
    )
    table.add_column(
        "Role mix",
        justify="right",
    )
    table.add_column("Segmentation")

    for record in summary["records"]:
        table.add_row(
            str(record["label"]),
            str(record["session_date"]),
            str(record["start_anchor"]),
            str(record["end_anchor"] or "document end"),
            str(record["end_method"]),
            (f"{int(record['included_block_count']):,}/{int(record['block_count']):,}"),
            f"{int(record['included_word_count']):,}",
            str(len(record["mixed_structural_zone_page_numbers"])),
            str(len(record["mixed_content_role_page_numbers"])),
            ("reused" if record["segmentation_reused"] else "segmented"),
        )

    console.print(table)
    console.print(
        f"Segmentations reused: {summary['segmentation_reused_count']}/{summary['record_count']}"
    )


def _display_modeling_corpus_summary(
    summary: dict[str, Any],
) -> None:
    """Display modelling-corpus export summary metrics."""
    table = Table(title="Modelling Corpus Export")
    table.add_column("Metric")
    table.add_column("Value", justify="right")

    for label, key in (
        ("Input sessions", "input_session_count"),
        ("Input turns", "input_turn_count"),
        ("Retained source turns", "retained_source_turn_count"),
        ("Modelling documents", "modeling_document_count"),
        ("Retained words", "retained_source_turn_word_total"),
    ):
        table.add_row(label, f"{int(summary[key]):,}")

    console.print(table)


def _display_corpus_profile_summary(
    manifest: dict[str, Any],
) -> None:
    """Display corpus-profile summary metrics."""
    table = Table(title="Corpus Profile")
    table.add_column("Metric")
    table.add_column("Value", justify="right")

    all_counts = manifest["universes"]["all_sessions"]
    primary_counts = manifest["universes"]["primary"]

    table.add_row("All-session documents", f"{int(all_counts['documents']):,}")
    table.add_row("Primary documents", f"{int(primary_counts['documents']):,}")
    table.add_row("All-session words", f"{int(all_counts['words']):,}")
    table.add_row("Primary words", f"{int(primary_counts['words']):,}")
    table.add_row(
        "All-session sample",
        f"{int(manifest['sample_counts']['all_sessions']['documents']):,}",
    )
    table.add_row("Primary sample", f"{int(manifest['sample_counts']['primary']['documents']):,}")

    console.print(table)


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


@app.command("extract-pdfs")
def extract_pdfs_command(
    manifest_path: Annotated[
        Path,
        typer.Option(
            "--manifest",
            help="Path to the discovered session manifest.",
        ),
    ] = DEFAULT_PDF_MANIFEST_PATH,
    selection_path: Annotated[
        Path,
        typer.Option(
            "--selection",
            help="CSV containing selected source_record_id values.",
        ),
    ] = DEFAULT_PDF_SELECTION_PATH,
    pdf_directory: Annotated[
        Path,
        typer.Option(
            "--pdf-dir",
            help="Directory for cached official PDFs.",
        ),
    ] = DEFAULT_PDF_DIRECTORY,
    extraction_root: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            help="Directory for page and block extraction outputs.",
        ),
    ] = DEFAULT_PDF_EXTRACTION_ROOT,
    summary_path: Annotated[
        Path,
        typer.Option(
            "--summary",
            help="Path for the batch QA summary.",
        ),
    ] = DEFAULT_PDF_SUMMARY_PATH,
    force_download: Annotated[
        bool,
        typer.Option(
            "--force-download",
            help="Download PDFs even when validated cache files exist.",
        ),
    ] = False,
    force_extract: Annotated[
        bool,
        typer.Option(
            "--force-extract",
            help="Rebuild extraction outputs even when valid.",
        ),
    ] = False,
) -> None:
    """Download and extract a selected set of official session PDFs."""
    timeout = httpx.Timeout(
        timeout=180.0,
        connect=30.0,
    )

    try:
        with httpx.Client(
            follow_redirects=True,
            timeout=timeout,
            headers={"User-Agent": PDF_USER_AGENT},
        ) as client:
            summary = run_pdf_batch(
                client=client,
                manifest_path=manifest_path,
                selection_path=selection_path,
                pdf_directory=pdf_directory,
                extraction_root=extraction_root,
                summary_path=summary_path,
                force_download=force_download,
                force_extract=force_extract,
            )

    except PdfBatchError as error:
        console.print(f"[bold red]PDF batch failed:[/bold red] {error}")
        raise typer.Exit(code=1) from error

    _display_pdf_batch_summary(summary)


@app.command("segment-structure")
def segment_structure_command(
    pdf_summary_path: Annotated[
        Path,
        typer.Option(
            "--pdf-summary",
            help=("Path to the PDF extraction batch summary."),
        ),
    ] = DEFAULT_STRUCTURE_SOURCE_SUMMARY_PATH,
    output_root: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            help=("Directory for structural block and page outputs."),
        ),
    ] = DEFAULT_STRUCTURE_OUTPUT_ROOT,
    summary_path: Annotated[
        Path,
        typer.Option(
            "--summary",
            help=("Path for the structural segmentation QA summary."),
        ),
    ] = DEFAULT_STRUCTURE_SUMMARY_PATH,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help=("Rebuild structural outputs even when the cache is valid."),
        ),
    ] = False,
) -> None:
    """Classify transcript structure for extracted PDFs."""
    try:
        summary = run_structure_batch(
            pdf_summary_path=(pdf_summary_path),
            output_root=output_root,
            summary_path=summary_path,
            force=force,
        )
    except StructureBatchError as error:
        console.print(f"[bold red]Structural batch failed:[/bold red] {error}")
        raise typer.Exit(code=1) from error

    _display_structure_batch_summary(summary)


@app.command("export-modeling-corpus")
def export_modeling_corpus_command(
    speaker_turn_root: Annotated[
        Path,
        typer.Option(
            "--speaker-turn-root",
            help="Directory containing per-document speaker-turn outputs.",
        ),
    ] = DEFAULT_SPEAKER_TURN_ROOT,
    overrides_path: Annotated[
        Path,
        typer.Option(
            "--overrides",
            help="Path to versioned modelling-turn overrides.",
        ),
    ] = DEFAULT_MODELING_OVERRIDES_PATH,
    metadata_summary_path: Annotated[
        Path,
        typer.Option(
            "--metadata-summary",
            help="Full-corpus run summary containing session dates and categories.",
        ),
    ] = DEFAULT_MODELING_METADATA_PATH,
    output_root: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            help="Canonical modelling-corpus output directory.",
        ),
    ] = DEFAULT_MODELING_OUTPUT_ROOT,
    minimum_words: Annotated[
        int,
        typer.Option(
            "--minimum-words",
            help="Minimum post-override source-turn word count to retain.",
        ),
    ] = DEFAULT_MINIMUM_WORDS,
    maximum_chunk_words: Annotated[
        int,
        typer.Option(
            "--maximum-chunk-words",
            help="Hard maximum whitespace-delimited words per modelling document.",
        ),
    ] = DEFAULT_MAXIMUM_CHUNK_WORDS,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Overwrite an existing nonempty modelling-corpus output directory.",
        ),
    ] = False,
) -> None:
    """Export the final traceable spoken-discourse modelling corpus."""
    try:
        summary = export_modeling_corpus(
            speaker_turn_root=speaker_turn_root,
            overrides_path=overrides_path,
            metadata_summary_path=metadata_summary_path,
            output_root=output_root,
            minimum_words=minimum_words,
            maximum_chunk_words=maximum_chunk_words,
            force=force,
        )
    except ModelingCorpusError as error:
        console.print(f"[bold red]Modelling corpus export failed:[/bold red] {error}")
        raise typer.Exit(code=1) from error

    _display_modeling_corpus_summary(summary)


@app.command("profile-modeling-corpus")
def profile_modeling_corpus_command(
    documents_path: Annotated[
        Path,
        typer.Option(
            "--documents",
            help="Path to locked modelling-corpus documents JSONL.",
        ),
    ] = DEFAULT_DOCUMENTS_PATH,
    export_manifest_path: Annotated[
        Path,
        typer.Option(
            "--export-manifest",
            help="Path to locked modelling-corpus export manifest.",
        ),
    ] = DEFAULT_EXPORT_MANIFEST_PATH,
    corpus_lock_path: Annotated[
        Path,
        typer.Option(
            "--corpus-lock",
            help="Path to locked modelling-corpus lock file.",
        ),
    ] = DEFAULT_CORPUS_LOCK_PATH,
    config_path: Annotated[
        Path,
        typer.Option(
            "--config",
            help="Path to the versioned corpus-profile configuration.",
        ),
    ] = DEFAULT_CORPUS_PROFILE_CONFIG_PATH,
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            help="Directory for corpus-profile QA outputs.",
        ),
    ] = DEFAULT_CORPUS_PROFILE_OUTPUT_DIR,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Overwrite an existing nonempty corpus-profile output directory.",
        ),
    ] = False,
) -> None:
    """Profile the locked modelling corpus without fitting topic models."""
    try:
        manifest = profile_modeling_corpus(
            documents_path=documents_path,
            export_manifest_path=export_manifest_path,
            corpus_lock_path=corpus_lock_path,
            config_path=config_path,
            output_dir=output_dir,
            force=force,
        )
    except CorpusProfileError as error:
        console.print(f"[bold red]Corpus profile failed:[/bold red] {error}")
        raise typer.Exit(code=1) from error

    _display_corpus_profile_summary(manifest)


def main() -> None:
    """Run the command-line application."""
    app()


if __name__ == "__main__":
    main()
