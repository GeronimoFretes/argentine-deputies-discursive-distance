"""Genera los gráficos de validación para la presentación final.

Produce dos archivos PNG en presentation/assets/:
  assets/markers_vs_turns.png   — barras agrupadas: marcadores explícitos vs. turnos
  assets/non_speech_spans.png   — dos paneles: acotaciones escénicas / notas editoriales

Ejecutar desde la raíz del repositorio:
    uv run python presentation/generate_charts.py

Usa solo la biblioteca estándar de Python (struct, zlib, pathlib).
Fuente de datos: tabla de pilotos verificada en docs/TEAMMATE_HANDOFF.md.
No modificar los números sin actualizar esa referencia.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "assets"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Datos verificados de los pilotos (docs/TEAMMATE_HANDOFF.md)
# ---------------------------------------------------------------------------

PILOT_LABELS = [
    "Ordinaria\nantigua",
    "Continua-\nción",
    "Remota",
    "Extensa\nreciente",
    "Informa-\ntiva",
    "Ord./esp.\nreciente",
]

EXPLICIT_MARKERS = [260, 545, 540, 587, 146, 846]
PARSED_TURNS = [262, 549, 541, 587, 146, 854]
STAGE_SPANS = [56, 94, 107, 221, 46, 173]
EDITORIAL_SPANS = [2, 18, 5, 8, 0, 33]

# Diferencias turno - marcador para anotación
TURN_MINUS_MARKER = [t - m for t, m in zip(PARSED_TURNS, EXPLICIT_MARKERS, strict=True)]

# ---------------------------------------------------------------------------
# Escritor PNG mínimo (solo biblioteca estándar)
# ---------------------------------------------------------------------------

def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    payload = chunk_type + data
    crc = zlib.crc32(payload) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + payload + struct.pack(">I", crc)


def _save_png(
    pixels: list[list[tuple[int, int, int]]], width: int, height: int, path: Path
) -> None:
    raw = bytearray()
    for row in pixels:
        raw.append(0)
        for r, g, b in row:
            raw += bytes([r, g, b])
    compressed = zlib.compress(bytes(raw), 9)
    data = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + _png_chunk(b"IDAT", compressed)
        + _png_chunk(b"IEND", b"")
    )
    path.write_bytes(data)
    print(f"Guardado {path}")


# ---------------------------------------------------------------------------
# Canvas raster
# ---------------------------------------------------------------------------

WHITE = (255, 255, 255)
BLACK = (20, 20, 20)
LIGHT_GRAY = (220, 220, 220)
MID_GRAY = (150, 150, 150)
DARK_GRAY = (80, 80, 80)
SUBTITLE_GRAY = (100, 100, 100)

# Paleta de colores para las series
C_BLUE = (41, 98, 255)
C_GREEN = (0, 153, 102)
C_RED = (204, 51, 51)
C_AMBER = (200, 130, 0)
C_BLUE_LIGHT = (173, 204, 255)
C_GREEN_LIGHT = (153, 230, 204)


class Canvas:
    def __init__(self, width: int, height: int, bg: tuple[int, int, int] = WHITE) -> None:
        self.w = width
        self.h = height
        self.px: list[list[tuple[int, int, int]]] = [
            [bg] * width for _ in range(height)
        ]

    def set_px(self, x: int, y: int, c: tuple[int, int, int]) -> None:
        if 0 <= x < self.w and 0 <= y < self.h:
            self.px[y][x] = c

    def rect(self, x: int, y: int, w: int, h: int, c: tuple[int, int, int]) -> None:
        for dy in range(max(0, -y), min(h, self.h - y)):
            for dx in range(max(0, -x), min(w, self.w - x)):
                self.px[y + dy][x + dx] = c

    def hline(
        self, x: int, y: int, length: int, c: tuple[int, int, int], thickness: int = 1
    ) -> None:
        for t in range(thickness):
            for dx in range(length):
                self.set_px(x + dx, y + t, c)

    def vline(
        self, x: int, y: int, length: int, c: tuple[int, int, int], thickness: int = 1
    ) -> None:
        for t in range(thickness):
            for dy in range(length):
                self.set_px(x + t, y + dy, c)

    def save(self, path: Path) -> None:
        _save_png(self.px, self.w, self.h, path)

    # ------------------------------------------------------------------
    # Text rendering — scalable 5×9 bitmap glyphs, with scale factor
    # ------------------------------------------------------------------

    def text(
        self,
        s: str,
        x: int,
        y: int,
        c: tuple[int, int, int],
        scale: int = 2,
    ) -> None:
        cx = x
        for ch in s:
            self._glyph(ch, cx, y, c, scale)
            cx += (5 + 1) * scale

    def text_w(self, s: str, scale: int = 2) -> int:
        return len(s) * (5 + 1) * scale

    def text_center(
        self,
        s: str,
        cx: int,
        y: int,
        c: tuple[int, int, int],
        scale: int = 2,
    ) -> None:
        x = cx - self.text_w(s, scale) // 2
        self.text(s, x, y, c, scale)

    def text_right(
        self,
        s: str,
        rx: int,
        y: int,
        c: tuple[int, int, int],
        scale: int = 2,
    ) -> None:
        x = rx - self.text_w(s, scale)
        self.text(s, x, y, c, scale)

    def multiline_center(
        self,
        lines: list[str],
        cx: int,
        y: int,
        c: tuple[int, int, int],
        scale: int = 2,
        line_gap: int = 4,
    ) -> int:
        for line in lines:
            self.text_center(line, cx, y, c, scale)
            y += 9 * scale + line_gap
        return y

    def _glyph(
        self,
        ch: str,
        x: int,
        y: int,
        c: tuple[int, int, int],
        scale: int,
    ) -> None:
        rows = _FONT.get(ch.upper(), _FONT.get("?", [0] * 9))
        for row_i, bits in enumerate(rows):
            for col_i in range(5):
                if bits & (1 << (4 - col_i)):
                    self.rect(
                        x + col_i * scale,
                        y + row_i * scale,
                        scale,
                        scale,
                        c,
                    )


# ---------------------------------------------------------------------------
# 5×9 bitmap font  (uppercase, digits, punctuation needed)
# ---------------------------------------------------------------------------

_FONT: dict[str, list[int]] = {
    " ": [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
    "A": [0x0E, 0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x00, 0x00],
    "B": [0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E, 0x00, 0x00],
    "C": [0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E, 0x00, 0x00],
    "D": [0x1C, 0x12, 0x11, 0x11, 0x11, 0x12, 0x1C, 0x00, 0x00],
    "E": [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F, 0x00, 0x00],
    "F": [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10, 0x00, 0x00],
    "G": [0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0F, 0x00, 0x00],
    "H": [0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11, 0x00, 0x00],
    "I": [0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E, 0x00, 0x00],
    "J": [0x07, 0x02, 0x02, 0x02, 0x02, 0x12, 0x0C, 0x00, 0x00],
    "K": [0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11, 0x00, 0x00],
    "L": [0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F, 0x00, 0x00],
    "M": [0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11, 0x00, 0x00],
    "N": [0x11, 0x19, 0x19, 0x15, 0x13, 0x13, 0x11, 0x00, 0x00],
    "O": [0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E, 0x00, 0x00],
    "P": [0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10, 0x00, 0x00],
    "Q": [0x0E, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0D, 0x00, 0x00],
    "R": [0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11, 0x00, 0x00],
    "S": [0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E, 0x00, 0x00],
    "T": [0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x00, 0x00],
    "U": [0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E, 0x00, 0x00],
    "V": [0x11, 0x11, 0x11, 0x11, 0x0A, 0x0A, 0x04, 0x00, 0x00],
    "W": [0x11, 0x11, 0x11, 0x15, 0x15, 0x1B, 0x11, 0x00, 0x00],
    "X": [0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11, 0x00, 0x00],
    "Y": [0x11, 0x11, 0x0A, 0x04, 0x04, 0x04, 0x04, 0x00, 0x00],
    "Z": [0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F, 0x00, 0x00],
    "0": [0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E, 0x00, 0x00],
    "1": [0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E, 0x00, 0x00],
    "2": [0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F, 0x00, 0x00],
    "3": [0x0E, 0x11, 0x01, 0x06, 0x01, 0x11, 0x0E, 0x00, 0x00],
    "4": [0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02, 0x00, 0x00],
    "5": [0x1F, 0x10, 0x10, 0x1E, 0x01, 0x01, 0x1E, 0x00, 0x00],
    "6": [0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E, 0x00, 0x00],
    "7": [0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08, 0x00, 0x00],
    "8": [0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E, 0x00, 0x00],
    "9": [0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C, 0x00, 0x00],
    ".": [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00],
    ",": [0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x08, 0x00, 0x00],
    "-": [0x00, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x00, 0x00, 0x00],
    "+": [0x00, 0x04, 0x04, 0x1F, 0x04, 0x04, 0x00, 0x00, 0x00],
    "/": [0x01, 0x01, 0x02, 0x04, 0x08, 0x10, 0x10, 0x00, 0x00],
    ":": [0x00, 0x00, 0x04, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00],
    "(": [0x02, 0x04, 0x08, 0x08, 0x08, 0x04, 0x02, 0x00, 0x00],
    ")": [0x08, 0x04, 0x02, 0x02, 0x02, 0x04, 0x08, 0x00, 0x00],
    "?": [0x0E, 0x11, 0x01, 0x06, 0x04, 0x00, 0x04, 0x00, 0x00],
    "!": [0x04, 0x04, 0x04, 0x04, 0x04, 0x00, 0x04, 0x00, 0x00],
    "'": [0x04, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
    "_": [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1F, 0x00, 0x00],
    "%": [0x11, 0x09, 0x02, 0x04, 0x08, 0x12, 0x11, 0x00, 0x00],
    # Letras con tilde/acento — redirigidas a la base
    "Á": [0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11, 0x00, 0x00],  # = A
    "É": [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F, 0x00, 0x00],  # = E
    "Í": [0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E, 0x00, 0x00],  # = I
    "Ó": [0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E, 0x00, 0x00],  # = O
    "Ú": [0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E, 0x00, 0x00],  # = U
    "Ñ": [0x11, 0x19, 0x1D, 0x15, 0x17, 0x13, 0x11, 0x00, 0x00],
    "Ü": [0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E, 0x00, 0x00],  # = U
}


def _fmt(n: int) -> str:
    """Formato argentino con punto como separador de miles."""
    return f"{n:,}".replace(",", ".")


# ---------------------------------------------------------------------------
# Función auxiliar para un gráfico de barras agrupadas completo
# ---------------------------------------------------------------------------

def _draw_bar_chart(
    cv: Canvas,
    *,
    values_a: list[int],
    values_b: list[int],
    labels_x: list[str],
    color_a: tuple[int, int, int],
    color_b: tuple[int, int, int],
    label_a: str,
    label_b: str,
    title: str,
    subtitle: str,
    y_axis_label: str,
    x_axis_label: str,
    chart_left: int,
    chart_top: int,
    chart_right: int,
    chart_bottom: int,
    show_diff: bool = False,
    diff_values: list[int] | None = None,
) -> None:
    n = len(labels_x)
    chart_w = chart_right - chart_left
    chart_h = chart_bottom - chart_top

    all_vals = values_a + values_b
    max_val = max(all_vals) if all_vals else 1
    y_top = int(max_val * 1.20)

    def to_y(v: int) -> int:
        return chart_bottom - int(v / y_top * chart_h)

    # Cuadrícula
    n_grid = 5
    for i in range(n_grid + 1):
        gv = int(y_top * i / n_grid)
        gy = to_y(gv)
        cv.hline(chart_left, gy, chart_w, LIGHT_GRAY, 1)
        lbl = _fmt(gv)
        cv.text_right(lbl, chart_left - 6, gy - 9, DARK_GRAY, scale=2)

    # Ejes
    cv.vline(chart_left, chart_top, chart_h + 1, BLACK, 2)
    cv.hline(chart_left, chart_bottom, chart_w, BLACK, 2)

    # Barras
    group_w = chart_w // n
    bar_w = max(group_w // 3, 20)
    gap = max(bar_w // 4, 6)

    for i, (va, vb) in enumerate(zip(values_a, values_b, strict=True)):
        gx = chart_left + i * group_w + group_w // 2
        ax = gx - bar_w - gap // 2
        bx = gx + gap // 2

        # Barra A
        ah = to_y(0) - to_y(va)
        cv.rect(ax, to_y(va), bar_w, ah, color_a)

        # Valor sobre barra A
        va_str = _fmt(va)
        cv.text_center(va_str, ax + bar_w // 2, to_y(va) - 22, BLACK, scale=2)

        # Barra B
        bh = to_y(0) - to_y(vb)
        cv.rect(bx, to_y(vb), bar_w, bh, color_b)

        # Valor sobre barra B
        vb_str = _fmt(vb)
        if show_diff and diff_values is not None and diff_values[i] > 0:
            diff_str = f"{vb_str} (+{diff_values[i]})"
        else:
            diff_str = vb_str
        cv.text_center(diff_str, bx + bar_w // 2, to_y(vb) - 22, BLACK, scale=2)

    # Etiquetas eje X
    for i, lbl in enumerate(labels_x):
        gx = chart_left + i * group_w + group_w // 2
        lines = lbl.split("\n")
        ly = chart_bottom + 10
        for line in lines:
            cv.text_center(line, gx, ly, DARK_GRAY, scale=2)
            ly += 22

    # Título (centrado en el gráfico)
    mid_x = (chart_left + chart_right) // 2
    cv.text_center(title, mid_x, chart_top - 60, BLACK, scale=3)
    if subtitle:
        cv.text_center(subtitle, mid_x, chart_top - 28, SUBTITLE_GRAY, scale=2)

    # Leyenda
    legend_x = chart_right - 380
    legend_y = chart_top + 8
    cv.rect(legend_x, legend_y, 24, 16, color_a)
    cv.text(label_a, legend_x + 30, legend_y, BLACK, scale=2)
    cv.rect(legend_x, legend_y + 26, 24, 16, color_b)
    cv.text(label_b, legend_x + 30, legend_y + 26, BLACK, scale=2)

    # Etiqueta eje Y
    cv.text(y_axis_label, 12, (chart_top + chart_bottom) // 2 - 20, MID_GRAY, scale=2)

    # Etiqueta eje X
    cv.text_center(x_axis_label, mid_x, chart_bottom + 72, MID_GRAY, scale=2)


# ---------------------------------------------------------------------------
# Gráfico 1: marcadores vs. turnos
# ---------------------------------------------------------------------------

def make_markers_vs_turns_chart() -> None:
    W, H = 1600, 900
    cv = Canvas(W, H)
    cv.rect(0, 0, W, H, WHITE)

    _draw_bar_chart(
        cv,
        values_a=EXPLICIT_MARKERS,
        values_b=PARSED_TURNS,
        labels_x=PILOT_LABELS,
        color_a=C_BLUE,
        color_b=C_GREEN,
        label_a="MARCADORES EXPLICITOS",
        label_b="TURNOS RECONSTRUIDOS",
        title="MARCADORES EXPLICITOS Y TURNOS DE HABLA RECONSTRUIDOS",
        subtitle=(
            "LOS TURNOS PUEDEN SUPERAR A LOS MARCADORES "
            "POR LA PRESERVACION DE CONTENIDO NO ATRIBUIDO"
        ),
        y_axis_label="CANTIDAD",
        x_axis_label="SESION PILOTO",
        chart_left=140,
        chart_top=110,
        chart_right=1540,
        chart_bottom=730,
        show_diff=True,
        diff_values=TURN_MINUS_MARKER,
    )

    cv.save(OUTPUT_DIR / "markers_vs_turns.png")


# ---------------------------------------------------------------------------
# Gráfico 2: dos paneles — acotaciones escénicas / notas editoriales
# ---------------------------------------------------------------------------

def _draw_single_bar_series(
    cv: Canvas,
    *,
    values: list[int],
    labels_x: list[str],
    color: tuple[int, int, int],
    panel_title: str,
    chart_left: int,
    chart_top: int,
    chart_right: int,
    chart_bottom: int,
    show_x_labels: bool = True,
) -> None:
    n = len(values)
    chart_w = chart_right - chart_left
    chart_h = chart_bottom - chart_top

    max_val = max(values) if any(v > 0 for v in values) else 1
    y_top = int(max_val * 1.25)
    if y_top == 0:
        y_top = 1

    def to_y(v: int) -> int:
        return chart_bottom - int(v / y_top * chart_h)

    # Cuadrícula
    n_grid = 4
    for i in range(n_grid + 1):
        gv = int(y_top * i / n_grid)
        gy = to_y(gv)
        cv.hline(chart_left, gy, chart_w, LIGHT_GRAY, 1)
        lbl = _fmt(gv)
        cv.text_right(lbl, chart_left - 6, gy - 9, DARK_GRAY, scale=2)

    # Ejes
    cv.vline(chart_left, chart_top, chart_h + 1, BLACK, 2)
    cv.hline(chart_left, chart_bottom, chart_w, BLACK, 2)

    # Título del panel (alineado a la izquierda del área del gráfico)
    cv.text(panel_title, chart_left, chart_top - 28, DARK_GRAY, scale=2)

    # Barras
    group_w = chart_w // n
    bar_w = max(group_w // 2, 30)

    for i, v in enumerate(values):
        gx = chart_left + i * group_w + group_w // 2
        bx = gx - bar_w // 2
        bh = to_y(0) - to_y(v)
        if bh > 0:
            cv.rect(bx, to_y(v), bar_w, bh, color)

        val_str = _fmt(v)
        cv.text_center(val_str, gx, to_y(v) - 22, BLACK, scale=2)

        # Etiquetas eje X (solo en el panel inferior o si se pide)
        if show_x_labels:
            lines = labels_x[i].split("\n")
            ly = chart_bottom + 10
            for line in lines:
                cv.text_center(line, gx, ly, DARK_GRAY, scale=2)
                ly += 22


def make_non_speech_spans_chart() -> None:
    W, H = 1600, 900
    cv = Canvas(W, H)
    cv.rect(0, 0, W, H, WHITE)

    MID_X = W // 2

    # Título global
    cv.text_center(
        "CONTENIDO NO DISCURSIVO DETECTADO POR SESION PILOTO",
        MID_X, 24, BLACK, scale=3,
    )
    cv.text_center(
        "ESCALAS INDEPENDIENTES PARA FACILITAR LA LECTURA",
        MID_X, 64, SUBTITLE_GRAY, scale=2,
    )

    LEFT = 140
    RIGHT = 1540
    PANEL_SEP = 36       # espacio entre paneles
    TOP_PANEL_TOP = 108
    TOP_PANEL_BOTTOM = 430
    BOT_PANEL_TOP = TOP_PANEL_BOTTOM + PANEL_SEP
    BOT_PANEL_BOTTOM = 800

    # Panel superior — Acotaciones escénicas
    _draw_single_bar_series(
        cv,
        values=STAGE_SPANS,
        labels_x=PILOT_LABELS,
        color=C_RED,
        panel_title="ACOTACIONES ESCENICAS DETECTADAS",
        chart_left=LEFT,
        chart_top=TOP_PANEL_TOP,
        chart_right=RIGHT,
        chart_bottom=TOP_PANEL_BOTTOM,
        show_x_labels=False,
    )

    # Separador
    cv.hline(LEFT, TOP_PANEL_BOTTOM + PANEL_SEP // 2, RIGHT - LEFT, LIGHT_GRAY, 2)

    # Panel inferior — Notas editoriales
    _draw_single_bar_series(
        cv,
        values=EDITORIAL_SPANS,
        labels_x=PILOT_LABELS,
        color=C_AMBER,
        panel_title="NOTAS EDITORIALES DETECTADAS",
        chart_left=LEFT,
        chart_top=BOT_PANEL_TOP,
        chart_right=RIGHT,
        chart_bottom=BOT_PANEL_BOTTOM,
        show_x_labels=True,
    )

    # Etiqueta eje Y global
    cv.text("CANTIDAD", 12, (TOP_PANEL_TOP + BOT_PANEL_BOTTOM) // 2 - 20, MID_GRAY, scale=2)

    # Etiqueta eje X
    cv.text_center("SESION PILOTO", MID_X, BOT_PANEL_BOTTOM + 72, MID_GRAY, scale=2)

    cv.save(OUTPUT_DIR / "non_speech_spans.png")


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    make_markers_vs_turns_chart()
    make_non_speech_spans_chart()
    print("Generacion de graficos completada.")
