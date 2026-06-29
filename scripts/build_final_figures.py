
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TOPIC_LABELS_ES = {
    0: "Argumentación general y confrontación",
    1: "Derecho penal y códigos procesales",
    2: "Redacción artículo por artículo",
    3: "Comisiones, dictámenes y actividad parlamentaria",
    4: "Justicia y Poder Judicial",
    5: "Provincias, Buenos Aires y territorio federal",
    6: "Aplausos y reacciones en el recinto",
    7: "Proyectos e iniciativas legislativas",
    8: "Desarrollo productivo y economía",
    9: "Derechos, género y salud reproductiva",
    10: "Trabajo, empleo y condiciones laborales",
    11: "Identidad política y memoria partidaria",
    12: "Posicionamiento oficialismo-oposición",
    13: "Posiciones de bloque y explicación del voto",
    14: "Mociones, reglamento y orden del día",
    15: "Modificaciones al texto de proyectos",
    16: "Poder Ejecutivo, DNU y autoridad constitucional",
    17: "Presupuesto, gasto público e inflación",
    18: "Jubilaciones y sistema previsional",
    19: "Deuda, FMI y política financiera",
    20: "Impuestos y política fiscal",
    21: "Dinámica de la Cámara y de las sesiones",
    22: "Inserción de discursos en el Diario de Sesiones",
    23: "Cuestiones de privilegio",
}

TOPIC_GROUPS = {
    "Temas sustantivos": [1, 4, 5, 8, 9, 10, 16, 17, 18, 19, 20],
    "Procedimiento parlamentario": [2, 3, 7, 13, 14, 15, 21, 22, 23],
    "Retórica y posicionamiento": [0, 11, 12],
    "Componente residual": [6],
}

EXPECTED_FILES = {
    "p1_grid": Path("data/qa/topic_modeling/nmf_grid_v1/grid_metrics.csv"),
    "p0_grid": Path("data/qa/topic_modeling/nmf_p0_k024_v1/grid_metrics.csv"),
    "annual": Path(
        "data/qa/topic_modeling/selected_nmf_k024_v1/"
        "annual_topic_prevalence.csv"
    ),
    "period": Path(
        "data/qa/topic_modeling/selected_nmf_k024_v1/"
        "period_topic_prevalence.csv"
    ),
    "ledger": Path("docs/final_evidence_ledger.csv"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the final presentation figures."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root. Defaults to the parent of scripts/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Defaults to "
            "<root>/docs/figures/final_presentation."
        ),
    )
    return parser.parse_args()


def ensure_files(root: Path) -> dict[str, Path]:
    paths = {name: root / relative for name, relative in EXPECTED_FILES.items()}
    missing = [path for path in paths.values() if not path.is_file()]

    if missing:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(
            "Missing required figure inputs:\n"
            f"{formatted}"
        )

    return paths


def read_inputs(paths: dict[str, Path]) -> dict[str, pd.DataFrame]:
    return {
        name: pd.read_csv(path)
        for name, path in paths.items()
    }


def validate_inputs(data: dict[str, pd.DataFrame]) -> None:
    p1 = data["p1_grid"]
    p0 = data["p0_grid"]
    annual = data["annual"]
    period = data["period"]
    ledger = data["ledger"]

    required_p1 = {
        "k",
        "converged",
        "mean_npmi_coherence_top10",
        "topic_diversity_top10",
        "mean_topic_exclusivity_top10",
        "redundancy_mean_off_diagonal_cosine",
    }
    required_annual = {
        "aggregation_level",
        "year",
        "topic_index",
        "prevalence_share",
    }
    required_period = {
        "aggregation_level",
        "temporal_period",
        "topic_index",
        "prevalence_share",
    }
    required_ledger = {
        "claim_id",
        "status",
        "display_value",
    }

    for name, frame, required in (
        ("P1 grid", p1, required_p1),
        ("P0 grid", p0, required_p1),
        ("annual prevalence", annual, required_annual),
        ("period prevalence", period, required_period),
        ("evidence ledger", ledger, required_ledger),
    ):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(
                f"{name} is missing columns: {sorted(missing)}"
            )

    if set(p1["k"].astype(int)) != {12, 16, 20, 24, 28}:
        raise AssertionError("Unexpected P1 K-grid.")

    if list(p0["k"].astype(int)) != [24]:
        raise AssertionError("P0 sensitivity file must contain only K=24.")

    years = set(
        annual.loc[
            annual["aggregation_level"] == "source_turn",
            "year",
        ].astype(int)
    )
    if years != set(range(2008, 2026)):
        raise AssertionError("Source-turn annual data must cover 2008-2025.")

    locked = int((ledger["status"] == "LOCKED").sum())
    pending = int((ledger["status"] == "PENDING_BERTOPIC").sum())

    if locked != 27 or pending != 1:
        raise AssertionError(
            f"Unexpected ledger status counts: locked={locked}, pending={pending}"
        )


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
            "figure.titlesize": 18,
        }
    )


def save_figure(
    fig: plt.Figure,
    path: Path,
    *,
    title: str,
    source_note: str,
) -> None:
    fig.text(
        0.02,
        0.025,
        source_note,
        ha="left",
        va="bottom",
        fontsize=8,
    )
    fig.tight_layout(rect=(0.02, 0.075, 0.98, 0.96))
    fig.savefig(
        path,
        dpi=220,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)

    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Figure was not created correctly: {path}")

    print(f"Created {path.name}: {title}")


def source_turn_topic(
    annual: pd.DataFrame,
    topic_index: int,
) -> pd.DataFrame:
    result = annual[
        (annual["aggregation_level"] == "source_turn")
        & (annual["topic_index"].astype(int) == topic_index)
    ].copy()

    result["year"] = result["year"].astype(int)
    result["percent"] = result["prevalence_share"].astype(float) * 100
    result = result.sort_values("year")

    if len(result) != 18:
        raise AssertionError(
            f"Expected 18 annual rows for topic {topic_index}, found {len(result)}"
        )

    return result


def build_k_selection(
    p1: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Any]:
    frame = p1.copy()
    frame["k"] = frame["k"].astype(int)
    frame["mean_npmi"] = frame["mean_npmi_coherence_top10"].astype(float)

    selected = frame.loc[frame["k"] == 24].iloc[0]
    selected_value = float(selected["mean_npmi"])

    fig, ax = plt.subplots(figsize=(12, 6.75))
    ax.plot(
        frame["k"],
        frame["mean_npmi"],
        marker="o",
        linewidth=2.2,
    )
    ax.scatter([24], [selected_value], s=130, zorder=3)
    ax.annotate(
        f"K=24\nNPMI={selected_value:.3f}",
        xy=(24, selected_value),
        xytext=(24.7, selected_value + 0.0015),
        arrowprops={"arrowstyle": "->"},
        fontsize=11,
    )
    ax.annotate(
        "K=12 no convergió",
        xy=(
            12,
            float(
                frame.loc[
                    frame["k"] == 12,
                    "mean_npmi",
                ].iloc[0]
            ),
        ),
        xytext=(12.7, 0.252),
        arrowprops={"arrowstyle": "->"},
        fontsize=10,
    )

    ax.set_title("K=24 ofrece el mejor equilibrio dentro de la grilla evaluada", pad=14)
    ax.set_xlabel("Cantidad de temas (K)")
    ax.set_ylabel("Coherencia media NPMI")
    ax.set_xticks(frame["k"])
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    path = output_dir / "01_nmf_k_selection.png"
    save_figure(
        fig,
        path,
        title="Selección de K para NMF",
        source_note=(
            "Fuente: grid_metrics.csv, especificación P1. "
            "K=24 también maximiza diversidad y exclusividad."
        ),
    )

    return {
        "file": path.name,
        "figure": "Selección de K",
        "claims": ["C07", "C08", "C09", "C10"],
    }


def build_p0_p1_table(
    p0: pd.DataFrame,
    p1: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Any]:
    p0_row = p0.loc[p0["k"].astype(int) == 24].iloc[0]
    p1_row = p1.loc[p1["k"].astype(int) == 24].iloc[0]

    rows = [
        (
            "Coherencia media NPMI",
            float(p0_row["mean_npmi_coherence_top10"]),
            float(p1_row["mean_npmi_coherence_top10"]),
            "Mayor es mejor",
        ),
        (
            "Diversidad top-10",
            float(p0_row["topic_diversity_top10"]),
            float(p1_row["topic_diversity_top10"]),
            "Mayor es mejor",
        ),
        (
            "Exclusividad media",
            float(p0_row["mean_topic_exclusivity_top10"]),
            float(p1_row["mean_topic_exclusivity_top10"]),
            "Mayor es mejor",
        ),
        (
            "Redundancia media",
            float(p0_row["redundancy_mean_off_diagonal_cosine"]),
            float(p1_row["redundancy_mean_off_diagonal_cosine"]),
            "Menor es mejor",
        ),
    ]

    cell_text = [
        [label, f"{value_p0:.3f}", f"{value_p1:.3f}", direction]
        for label, value_p0, value_p1, direction in rows
    ]

    fig, ax = plt.subplots(figsize=(12, 6.75))
    ax.axis("off")
    ax.set_title(
        "P1 mejora coherencia, diversidad y exclusividad",
        pad=20,
    )

    table = ax.table(
        cellText=cell_text,
        colLabels=["Métrica", "P0", "P1", "Lectura"],
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.38, 0.14, 0.14, 0.24],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2.1)

    ax.text(
        0.5,
        0.15,
        (
            "P1 elimina solo cinco tratamientos parlamentarios. "
            "La pequeña suba de redundancia no compensa las mejoras "
            "en las tres métricas principales."
        ),
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=11,
        wrap=True,
    )

    path = output_dir / "02_p0_p1_sensitivity.png"
    save_figure(
        fig,
        path,
        title="Sensibilidad P0 versus P1",
        source_note=(
            "Fuente: grid_metrics.csv de P0 y P1, ambos con K=24. "
            "Las matrices fueron ajustadas por separado."
        ),
    )

    return {
        "file": path.name,
        "figure": "Sensibilidad P0-P1",
        "claims": ["C11", "C12", "C13", "C14"],
    }


def build_topic_map(output_dir: Path) -> dict[str, Any]:
    short_labels = {
        0: "Argumentación general y confrontación",
        1: "Derecho penal y códigos procesales",
        2: "Redacción artículo por artículo",
        3: "Comisiones y actividad parlamentaria",
        4: "Justicia y Poder Judicial",
        5: "Provincias y federalismo",
        6: "Aplausos y reacciones del recinto",
        7: "Proyectos e iniciativas legislativas",
        8: "Desarrollo productivo y economía",
        9: "Derechos, género y salud reproductiva",
        10: "Trabajo y condiciones laborales",
        11: "Identidad política y memoria partidaria",
        12: "Posicionamiento oficialismo-oposición",
        13: "Posiciones de bloque y explicación del voto",
        14: "Mociones, reglamento y orden del día",
        15: "Modificaciones a proyectos",
        16: "Poder Ejecutivo, DNU y Constitución",
        17: "Presupuesto, gasto e inflación",
        18: "Jubilaciones y sistema previsional",
        19: "Deuda, FMI y política financiera",
        20: "Impuestos y política fiscal",
        21: "Dinámica de Cámara y sesiones",
        22: "Inserciones en el Diario de Sesiones",
        23: "Cuestiones de privilegio",
    }

    fig, ax = plt.subplots(figsize=(15, 8.44))
    ax.axis("off")
    ax.set_title(
        "Los 24 componentes combinan agendas sustantivas y lenguaje institucional",
        pad=18,
    )

    boxes = [
        ("Temas sustantivos", TOPIC_GROUPS["Temas sustantivos"], 0.03, 0.50, 0.45, 0.41),
        (
            "Procedimiento parlamentario",
            TOPIC_GROUPS["Procedimiento parlamentario"],
            0.52,
            0.50,
            0.45,
            0.41,
        ),
        (
            "Retórica y posicionamiento",
            TOPIC_GROUPS["Retórica y posicionamiento"],
            0.03,
            0.13,
            0.45,
            0.28,
        ),
        (
            "Componente residual",
            TOPIC_GROUPS["Componente residual"],
            0.52,
            0.13,
            0.45,
            0.28,
        ),
    ]

    for group_name, topic_ids, x, y, width, height in boxes:
        ax.add_patch(
            plt.Rectangle(
                (x, y),
                width,
                height,
                transform=ax.transAxes,
                fill=False,
                linewidth=1.2,
                alpha=0.45,
            )
        )
        ax.text(
            x + 0.018,
            y + height - 0.032,
            f"{group_name} ({len(topic_ids)})",
            transform=ax.transAxes,
            fontsize=13,
            fontweight="bold",
            va="top",
        )

        text_y = y + height - 0.085
        step = 0.0285 if len(topic_ids) > 5 else 0.058

        for topic_id in topic_ids:
            ax.text(
                x + 0.018,
                text_y,
                f"T{topic_id:02d}  {short_labels[topic_id]}",
                transform=ax.transAxes,
                fontsize=9.0,
                va="top",
            )
            text_y -= step

    ax.text(
        0.03,
        0.055,
        (
            "Lectura: el modelo no produce 24 políticas públicas puras. "
            "También recupera procedimiento parlamentario, posicionamiento "
            "político y un componente residual de reacciones del recinto."
        ),
        transform=ax.transAxes,
        fontsize=10.8,
        va="bottom",
        wrap=True,
    )

    path = output_dir / "03_topic_map.png"
    save_figure(
        fig,
        path,
        title="Mapa de los 24 temas",
        source_note=(
            "Fuente: términos principales y cinco documentos "
            "representativos por tema; etiquetas asignadas manualmente."
        ),
    )

    return {
        "file": path.name,
        "figure": "Mapa de temas",
        "claims": [],
    }

def build_single_topic_line(
    annual: pd.DataFrame,
    output_dir: Path,
    *,
    topic_index: int,
    title: str,
    filename: str,
    peak_year: int,
    secondary_year: int | None,
    claims: list[str],
) -> dict[str, Any]:
    frame = source_turn_topic(annual, topic_index)

    peak_value = float(
        frame.loc[
            frame["year"] == peak_year,
            "percent",
        ].iloc[0]
    )

    fig, ax = plt.subplots(figsize=(12, 6.75))
    ax.plot(
        frame["year"],
        frame["percent"],
        marker="o",
        linewidth=2,
    )
    ax.scatter([peak_year], [peak_value], s=130, zorder=3)
    ax.annotate(
        f"{peak_year}: {peak_value:.2f}%",
        xy=(peak_year, peak_value),
        xytext=(peak_year + 0.5, peak_value + 0.5),
        arrowprops={"arrowstyle": "->"},
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
            xytext=(secondary_year + 0.4, secondary_value - 1.1),
            arrowprops={"arrowstyle": "->"},
            fontsize=10,
        )

    ax.set_title(title)
    ax.set_xlabel("Año")
    ax.set_ylabel("Prevalencia estimada (%)")
    ax.set_xticks(range(2008, 2026, 2))
    ax.set_xlim(2007.5, 2025.5)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    path = output_dir / filename
    save_figure(
        fig,
        path,
        title=title,
        source_note=(
            "Fuente: prevalencia anual NMF P1 K=24. "
            "Estimador principal: igual peso por intervención fuente."
        ),
    )

    return {
        "file": path.name,
        "figure": title,
        "claims": claims,
    }


def build_fiscal_line(
    annual: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Any]:
    budget = source_turn_topic(annual, 17)
    taxes = source_turn_topic(annual, 20)

    fig, ax = plt.subplots(figsize=(12, 6.75))
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
    ax.text(
        2022,
        max(
            float(budget["percent"].max()),
            float(taxes["percent"].max()),
        )
        - 0.55,
        "Mayor prominencia fiscal\n2021-2023",
        ha="center",
        va="top",
        fontsize=11,
    )

    for frame, year in ((budget, 2021), (taxes, 2021)):
        value = float(
            frame.loc[frame["year"] == year, "percent"].iloc[0]
        )
        ax.scatter([year], [value], s=110, zorder=3)

    ax.set_title(
        "El discurso fiscal se concentra especialmente entre 2021 y 2023",
        pad=14,
    )
    ax.set_xlabel("Año")
    ax.set_ylabel("Prevalencia estimada (%)")
    ax.set_xticks(range(2008, 2026, 2))
    ax.set_xlim(2007.5, 2025.5)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    path = output_dir / "06_fiscal_topics_annual.png"
    save_figure(
        fig,
        path,
        title="Evolución de los temas fiscales",
        source_note=(
            "Fuente: prevalencia anual NMF P1 K=24. "
            "Estimador principal: igual peso por intervención fuente."
        ),
    )

    return {
        "file": path.name,
        "figure": "Temas fiscales",
        "claims": ["C19", "C20", "C21"],
    }


def build_pensions_periods(
    period: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Any]:
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
    bars = ax.bar(
        frame["temporal_period"].astype(str),
        frame["percent"],
    )

    for bar, value in zip(bars, frame["percent"], strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            float(value) + 0.12,
            f"{float(value):.2f}%",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    ax.annotate(
        (
            f"+{absolute_change:.2f} puntos porcentuales\n"
            f"({relative_change:.0f}% relativo)"
        ),
        xy=(4, final),
        xytext=(2.7, final + 0.75),
        arrowprops={"arrowstyle": "->"},
        fontsize=11,
    )

    ax.set_title(
        "El discurso previsional gana presencia en los períodos recientes",
        pad=14,
    )
    ax.set_xlabel("Período")
    ax.set_ylabel("Prevalencia estimada (%)")
    ax.set_ylim(0, final + 1.6)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    path = output_dir / "07_pensions_periods.png"
    save_figure(
        fig,
        path,
        title="Evolución del tema previsional",
        source_note=(
            "Fuente: prevalencia por períodos NMF P1 K=24. "
            "Estimador principal: igual peso por intervención fuente."
        ),
    )

    return {
        "file": path.name,
        "figure": "Jubilaciones por período",
        "claims": ["C22"],
    }


def annual_value(
    annual: pd.DataFrame,
    *,
    topic_index: int,
    year: int,
    aggregation: str,
) -> float:
    matches = annual[
        (annual["topic_index"].astype(int) == topic_index)
        & (annual["year"].astype(int) == year)
        & (annual["aggregation_level"] == aggregation)
    ]

    if len(matches) != 1:
        raise AssertionError(
            "Expected exactly one annual row for "
            f"topic={topic_index}, year={year}, aggregation={aggregation}"
        )

    return float(matches.iloc[0]["prevalence_share"]) * 100


def build_robustness_table(
    annual: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Any]:
    findings = [
        ("Justicia", 4, 2013),
        ("Derechos y género", 9, 2018),
        ("Presupuesto", 17, 2021),
        ("Impuestos", 20, 2021),
    ]
    aggregations = [
        ("document", "Documento"),
        ("source_turn", "Intervención"),
        ("session", "Sesión"),
    ]

    rows = []
    for label, topic_index, year in findings:
        values = [
            annual_value(
                annual,
                topic_index=topic_index,
                year=year,
                aggregation=aggregation,
            )
            for aggregation, _ in aggregations
        ]
        rows.append(
            [
                f"{label} ({year})",
                *(f"{value:.2f}%" for value in values),
                "Sí",
            ]
        )

    fig, ax = plt.subplots(figsize=(12, 6.75))
    ax.axis("off")
    ax.set_title(
        "Las conclusiones principales se mantienen con tres ponderaciones",
        pad=20,
    )

    table = ax.table(
        cellText=rows,
        colLabels=[
            "Hallazgo",
            "Documento",
            "Intervención",
            "Sesión",
            "Misma lectura",
        ],
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.28, 0.16, 0.16, 0.16, 0.16],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11.5)
    table.scale(1, 2.1)

    ax.text(
        0.5,
        0.14,
        (
            "Las magnitudes cambian porque cambia la unidad que recibe igual "
            "peso. Lo importante es que el año de máxima concentración y la "
            "interpretación sustantiva permanecen."
        ),
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=11,
        wrap=True,
    )

    path = output_dir / "08_aggregation_robustness.png"
    save_figure(
        fig,
        path,
        title="Robustez por unidad de ponderación",
        source_note=(
            "Fuente: prevalencia anual NMF P1 K=24 bajo ponderación por "
            "documento, intervención fuente y sesión."
        ),
    )

    return {
        "file": path.name,
        "figure": "Robustez de agregación",
        "claims": ["C24", "C25", "C26", "C27"],
    }


def write_manifest(
    output_dir: Path,
    entries: list[dict[str, Any]],
) -> None:
    csv_path = output_dir / "figure_manifest.csv"
    json_path = output_dir / "figure_manifest.json"

    with csv_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=["file", "figure", "claims"],
            lineterminator="\n",
        )
        writer.writeheader()
        for entry in entries:
            writer.writerow(
                {
                    "file": entry["file"],
                    "figure": entry["figure"],
                    "claims": ",".join(entry["claims"]),
                }
            )

    json_path.write_text(
        json.dumps(
            entries,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else root / "docs" / "figures" / "final_presentation"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = ensure_files(root)
    data = read_inputs(paths)
    validate_inputs(data)
    configure_matplotlib()

    entries = [
        build_k_selection(data["p1_grid"], output_dir),
        build_p0_p1_table(
            data["p0_grid"],
            data["p1_grid"],
            output_dir,
        ),
        build_topic_map(output_dir),
        build_single_topic_line(
            data["annual"],
            output_dir,
            topic_index=4,
            title="La justicia alcanza una concentración excepcional en 2013",
            filename="04_justice_annual.png",
            peak_year=2013,
            secondary_year=None,
            claims=["C16"],
        ),
        build_single_topic_line(
            data["annual"],
            output_dir,
            topic_index=9,
            title="Derechos, género y salud reproductiva alcanzan su máximo en 2018",
            filename="05_rights_gender_annual.png",
            peak_year=2018,
            secondary_year=2020,
            claims=["C17", "C18"],
        ),
        build_fiscal_line(data["annual"], output_dir),
        build_pensions_periods(data["period"], output_dir),
        build_robustness_table(data["annual"], output_dir),
    ]

    write_manifest(output_dir, entries)

    print()
    print("Final figures generated successfully.")
    print(f"Output directory: {output_dir}")
    print(f"Figures: {len(entries)}")
    print("BERTopic comparison figure remains intentionally pending.")


if __name__ == "__main__":
    main()
