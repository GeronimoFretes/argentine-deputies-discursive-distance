
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate only the five final-presentation figures whose "
            "annotations need safer spacing. Existing manifests and Figure 9 "
            "are preserved."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root. Defaults to the parent of scripts/.",
    )
    return parser.parse_args()


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 18,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        }
    )


def save_figure(
    fig: plt.Figure,
    path: Path,
    source_note: str,
) -> None:
    fig.text(
        0.01,
        0.018,
        source_note,
        ha="left",
        va="bottom",
        fontsize=8,
    )
    fig.savefig(
        path,
        dpi=220,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)

    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Figure was not created correctly: {path}")


def annotation_box() -> dict[str, object]:
    return {
        "boxstyle": "round,pad=0.3",
        "facecolor": "white",
        "edgecolor": "0.65",
        "alpha": 0.94,
    }


def load_inputs(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    p1_grid_path = (
        root
        / "data"
        / "qa"
        / "topic_modeling"
        / "nmf_grid_v1"
        / "grid_metrics.csv"
    )
    annual_path = (
        root
        / "data"
        / "qa"
        / "topic_modeling"
        / "selected_nmf_k024_v1"
        / "annual_topic_prevalence.csv"
    )
    period_path = (
        root
        / "data"
        / "qa"
        / "topic_modeling"
        / "selected_nmf_k024_v1"
        / "period_topic_prevalence.csv"
    )

    required = [p1_grid_path, annual_path, period_path]
    missing = [path for path in required if not path.is_file()]

    if missing:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(
            "Missing required figure inputs:\n"
            f"{formatted}"
        )

    return (
        pd.read_csv(p1_grid_path),
        pd.read_csv(annual_path),
        pd.read_csv(period_path),
    )


def source_turn_topic(
    annual: pd.DataFrame,
    topic_index: int,
) -> pd.DataFrame:
    frame = annual[
        (annual["aggregation_level"] == "source_turn")
        & (annual["topic_index"].astype(int) == topic_index)
    ].copy()

    frame["year"] = frame["year"].astype(int)
    frame["percent"] = frame["prevalence_share"].astype(float) * 100
    frame = frame.sort_values("year")

    if len(frame) != 18:
        raise AssertionError(
            f"Expected 18 annual rows for topic {topic_index}, "
            f"found {len(frame)}."
        )

    return frame


def build_k_selection(
    p1_grid: pd.DataFrame,
    output_dir: Path,
) -> None:
    frame = p1_grid.copy()
    frame["k"] = frame["k"].astype(int)
    frame["mean_npmi"] = frame[
        "mean_npmi_coherence_top10"
    ].astype(float)

    selected_value = float(
        frame.loc[frame["k"] == 24, "mean_npmi"].iloc[0]
    )
    k12_value = float(
        frame.loc[frame["k"] == 12, "mean_npmi"].iloc[0]
    )

    y_min = float(frame["mean_npmi"].min())
    y_max = float(frame["mean_npmi"].max())
    y_span = y_max - y_min

    fig, ax = plt.subplots(figsize=(12, 6.75))
    fig.subplots_adjust(
        left=0.10,
        right=0.96,
        bottom=0.16,
        top=0.82,
    )

    ax.plot(
        frame["k"],
        frame["mean_npmi"],
        marker="o",
        linewidth=2.2,
    )
    ax.scatter([24], [selected_value], s=125, zorder=3)

    ax.annotate(
        f"K=24\nNPMI={selected_value:.3f}",
        xy=(24, selected_value),
        xytext=(22.4, selected_value - 0.18 * y_span),
        ha="right",
        va="top",
        arrowprops={"arrowstyle": "->"},
        bbox=annotation_box(),
        fontsize=11,
    )
    ax.annotate(
        "K=12 no convergió",
        xy=(12, k12_value),
        xytext=(13.3, k12_value + 0.12 * y_span),
        ha="left",
        va="bottom",
        arrowprops={"arrowstyle": "->"},
        bbox=annotation_box(),
        fontsize=10,
    )

    ax.set_title(
        "K=24 ofrece el mejor equilibrio dentro de la grilla evaluada",
        pad=16,
    )
    ax.set_xlabel("Cantidad de temas (K)")
    ax.set_ylabel("Coherencia media NPMI")
    ax.set_xticks(frame["k"])
    ax.set_ylim(
        y_min - 0.15 * y_span,
        y_max + 0.20 * y_span,
    )
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    save_figure(
        fig,
        output_dir / "01_nmf_k_selection.png",
        (
            "Fuente: grid_metrics.csv, especificación P1. "
            "K=24 también maximiza diversidad y exclusividad."
        ),
    )


def build_single_topic_line(
    annual: pd.DataFrame,
    output_dir: Path,
    *,
    topic_index: int,
    title: str,
    filename: str,
    peak_year: int,
    secondary_year: int | None,
) -> None:
    frame = source_turn_topic(annual, topic_index)

    peak_value = float(
        frame.loc[frame["year"] == peak_year, "percent"].iloc[0]
    )
    maximum = float(frame["percent"].max())
    headroom = max(maximum * 0.22, 1.0)

    fig, ax = plt.subplots(figsize=(12, 6.75))
    fig.subplots_adjust(
        left=0.10,
        right=0.96,
        bottom=0.16,
        top=0.82,
    )

    ax.plot(
        frame["year"],
        frame["percent"],
        marker="o",
        linewidth=2,
    )
    ax.scatter([peak_year], [peak_value], s=125, zorder=3)

    peak_x_offset = -2.6 if peak_year >= 2018 else 1.3
    peak_alignment = "right" if peak_x_offset < 0 else "left"

    ax.annotate(
        f"{peak_year}: {peak_value:.2f}%",
        xy=(peak_year, peak_value),
        xytext=(
            peak_year + peak_x_offset,
            peak_value - 0.18 * maximum,
        ),
        ha=peak_alignment,
        va="top",
        arrowprops={"arrowstyle": "->"},
        bbox=annotation_box(),
        fontsize=11,
    )

    if secondary_year is not None:
        secondary_value = float(
            frame.loc[
                frame["year"] == secondary_year,
                "percent",
            ].iloc[0]
        )
        ax.scatter(
            [secondary_year],
            [secondary_value],
            s=85,
            zorder=3,
        )
        ax.annotate(
            f"{secondary_year}: {secondary_value:.2f}%",
            xy=(secondary_year, secondary_value),
            xytext=(
                secondary_year + 1.1,
                secondary_value - 0.20 * maximum,
            ),
            ha="left",
            va="top",
            arrowprops={"arrowstyle": "->"},
            bbox=annotation_box(),
            fontsize=10,
        )

    ax.set_title(title, pad=16)
    ax.set_xlabel("Año")
    ax.set_ylabel("Prevalencia estimada (%)")
    ax.set_xticks(range(2008, 2026, 2))
    ax.set_xlim(2007.5, 2025.5)
    ax.set_ylim(0, maximum + headroom)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    save_figure(
        fig,
        output_dir / filename,
        (
            "Fuente: prevalencia anual NMF P1 K=24. "
            "Estimador principal: igual peso por intervención fuente."
        ),
    )


def build_fiscal_line(
    annual: pd.DataFrame,
    output_dir: Path,
) -> None:
    budget = source_turn_topic(annual, 17)
    taxes = source_turn_topic(annual, 20)
    maximum = max(
        float(budget["percent"].max()),
        float(taxes["percent"].max()),
    )

    fig, ax = plt.subplots(figsize=(12, 6.75))
    fig.subplots_adjust(
        left=0.10,
        right=0.96,
        bottom=0.16,
        top=0.82,
    )

    ax.plot(
        budget["year"],
        budget["percent"],
        marker="o",
        linewidth=2,
        label="Presupuesto, gasto e inflación",
    )
    ax.plot(
        taxes["year"],
        taxes["percent"],
        marker="o",
        linewidth=2,
        label="Impuestos y política fiscal",
    )

    ax.axvspan(2021, 2023, alpha=0.12)

    for frame in (budget, taxes):
        value = float(
            frame.loc[frame["year"] == 2021, "percent"].iloc[0]
        )
        ax.scatter([2021], [value], s=100, zorder=3)

    ax.text(
        2022,
        maximum * 0.72,
        "Mayor prominencia fiscal\n2021–2023",
        ha="center",
        va="center",
        fontsize=11,
        bbox=annotation_box(),
    )

    ax.set_title(
        "El discurso fiscal se concentra especialmente entre 2021 y 2023",
        pad=16,
    )
    ax.set_xlabel("Año")
    ax.set_ylabel("Prevalencia estimada (%)")
    ax.set_xticks(range(2008, 2026, 2))
    ax.set_xlim(2007.5, 2025.5)
    ax.set_ylim(0, maximum * 1.25)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.98),
    )
    ax.spines[["top", "right"]].set_visible(False)

    save_figure(
        fig,
        output_dir / "06_fiscal_topics_annual.png",
        (
            "Fuente: prevalencia anual NMF P1 K=24. "
            "Estimador principal: igual peso por intervención fuente."
        ),
    )


def build_pensions_periods(
    period: pd.DataFrame,
    output_dir: Path,
) -> None:
    frame = period[
        (period["aggregation_level"] == "source_turn")
        & (period["topic_index"].astype(int) == 18)
    ].copy()

    order = [
        "2008-2011",
        "2012-2015",
        "2016-2019",
        "2020-2023",
        "2024-2025",
    ]
    frame["temporal_period"] = pd.Categorical(
        frame["temporal_period"],
        categories=order,
        ordered=True,
    )
    frame = frame.sort_values("temporal_period")
    frame["percent"] = frame["prevalence_share"].astype(float) * 100

    if list(frame["temporal_period"].astype(str)) != order:
        raise AssertionError("Unexpected pension period ordering.")

    initial = float(frame["percent"].iloc[0])
    final = float(frame["percent"].iloc[-1])
    absolute_change = final - initial
    relative_change = absolute_change / initial * 100

    fig, ax = plt.subplots(figsize=(12, 6.75))
    fig.subplots_adjust(
        left=0.10,
        right=0.96,
        bottom=0.16,
        top=0.82,
    )

    bars = ax.bar(
        frame["temporal_period"].astype(str),
        frame["percent"],
    )

    for bar, value in zip(bars, frame["percent"], strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            float(value) + 0.10,
            f"{float(value):.2f}%",
            ha="center",
            va="bottom",
            fontsize=10.5,
        )

    ax.annotate(
        (
            f"+{absolute_change:.2f} p.p.\n"
            f"({relative_change:.0f}% relativo)"
        ),
        xy=(4, final),
        xytext=(2.45, final * 0.83),
        ha="center",
        va="center",
        arrowprops={"arrowstyle": "->"},
        bbox=annotation_box(),
        fontsize=11,
    )

    ax.set_title(
        "El discurso previsional gana presencia en los períodos recientes",
        pad=16,
    )
    ax.set_xlabel("Período")
    ax.set_ylabel("Prevalencia estimada (%)")
    ax.set_ylim(0, final * 1.30)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    save_figure(
        fig,
        output_dir / "07_pensions_periods.png",
        (
            "Fuente: prevalencia por períodos NMF P1 K=24. "
            "Estimador principal: igual peso por intervención fuente."
        ),
    )


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_dir = (
        root
        / "docs"
        / "figures"
        / "final_presentation"
    )

    manifest_csv = output_dir / "figure_manifest.csv"
    manifest_json = output_dir / "figure_manifest.json"
    figure_9 = output_dir / "09_model_comparison.png"

    required_preserved = [
        manifest_csv,
        manifest_json,
        figure_9,
    ]
    missing_preserved = [
        path for path in required_preserved if not path.is_file()
    ]

    if missing_preserved:
        formatted = "\n".join(
            f"- {path}" for path in missing_preserved
        )
        raise FileNotFoundError(
            "The finalized nine-figure package is incomplete. "
            "Run Phase 7 first:\n"
            f"{formatted}"
        )

    p1_grid, annual, period = load_inputs(root)
    configure_matplotlib()

    original_manifest_csv = manifest_csv.read_bytes()
    original_manifest_json = manifest_json.read_bytes()
    original_figure_9_size = figure_9.stat().st_size

    build_k_selection(p1_grid, output_dir)
    build_single_topic_line(
        annual,
        output_dir,
        topic_index=4,
        title=(
            "La justicia alcanza una concentración "
            "excepcional en 2013"
        ),
        filename="04_justice_annual.png",
        peak_year=2013,
        secondary_year=None,
    )
    build_single_topic_line(
        annual,
        output_dir,
        topic_index=9,
        title=(
            "Derechos, género y salud reproductiva "
            "alcanzan su máximo en 2018"
        ),
        filename="05_rights_gender_annual.png",
        peak_year=2018,
        secondary_year=2020,
    )
    build_fiscal_line(annual, output_dir)
    build_pensions_periods(period, output_dir)

    if manifest_csv.read_bytes() != original_manifest_csv:
        raise AssertionError(
            "figure_manifest.csv changed unexpectedly."
        )
    if manifest_json.read_bytes() != original_manifest_json:
        raise AssertionError(
            "figure_manifest.json changed unexpectedly."
        )
    if figure_9.stat().st_size != original_figure_9_size:
        raise AssertionError(
            "Figure 9 changed unexpectedly."
        )

    print("Final figure layouts refreshed successfully.")
    print("Regenerated: 01, 04, 05, 06, 07")
    print("Preserved: 02, 03, 08, 09")
    print("Preserved: figure_manifest.csv and figure_manifest.json")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
