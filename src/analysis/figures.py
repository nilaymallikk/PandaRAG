"""Publication-quality SVG figures for the final report.

Self-contained: a small charting layer (grouped bars, bars, scatter, lines) with one
consistent visual system, no external dependency. Every figure is generated from the
same tables the report quotes, so a figure cannot disagree with a number.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
import math
from typing import Any, Iterable, Mapping, Sequence

from src import config
from src.analysis import loaders, tables

FIG_DIR = config.REPO_ROOT / "docs" / "figures"
FONT = "Inter, 'Helvetica Neue', Arial, sans-serif"

PALETTE = {
    "P0": "#6b7280",
    "P1c": "#2563eb",
    "P2c": "#0891b2",
    "P2c10": "#0e7490",
    "P3c": "#059669",
    "P1n": "#93c5fd",
    "P4": "#f59e0b",
    "P5": "#b45309",
    "P6": "#7c3aed",
}
BACKGROUND = "#ffffff"
INK = "#111827"
MUTED = "#6b7280"
GRID = "#e5e7eb"


# ---------------------------------------------------------------------------
# chart primitives (auto-fit layout: no overflow, no overlapping labels)
# ---------------------------------------------------------------------------

WIDTH = 1000
HEIGHT = 580
PAD_LEFT = 78
PAD_RIGHT = 36
PAD_TOP = 84
PAD_BOTTOM = 140

def _text_width(text: str, size: float) -> float:
    """Approximate rendered width of Helvetica-like text (conservative)."""
    return 0.56 * size * len(str(text)) + 1.0


def _escape(text: str) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _nice_max(value: float) -> float:
    """Round a maximum up to a readable axis bound (1/2/2.5/4/5 x 10^n for large ranges)."""
    if value <= 0:
        return 1.0
    for candidate in (0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.75, 1.0):
        if value <= candidate * 1.02:
            return candidate
    magnitude = 10 ** math.floor(math.log10(value))
    for step in (1, 2, 2.5, 4, 5, 10):
        candidate = step * magnitude
        if value <= candidate * 1.005:
            return float(candidate)
    return float(10 * magnitude)


def _tick(value: float, top: float) -> str:
    """Compact axis tick label."""
    if top >= 1000:
        return f"{value / 1000:.1f}k" if value else "0"
    return f"{value:.2f}"


def _plot_area(bottom: float | None = None, left: float | None = None):
    bottom = PAD_BOTTOM if bottom is None else bottom
    left = PAD_LEFT if left is None else left
    return left, PAD_TOP, WIDTH - left - PAD_RIGHT, HEIGHT - PAD_TOP - bottom


def _legend_rows(entries: Sequence[tuple[str, str]], max_width: float, size: float = 11.0,
                 swatch: float = 12.0, gap: float = 18.0) -> list[list[tuple[str, str, float]]]:
    """Lay legend entries out in rows that fit ``max_width``."""
    rows: list[list[tuple[str, str, float]]] = []
    row: list[tuple[str, str, float]] = []
    used = 0.0
    for label, colour in entries:
        width = swatch + 6 + _text_width(label, size) + gap
        if row and used + width > max_width:
            rows.append(row)
            row, used = [], 0.0
        row.append((label, colour, width))
        used += width
    if row:
        rows.append(row)
    return rows


def _legend_height(entries: Sequence[tuple[str, str]], max_width: float) -> float:
    return max(1, len(_legend_rows(entries, max_width))) * 17


def _legend(entries: Sequence[tuple[str, str]], x: float, y_top: float, max_width: float,
            size: float = 11.0) -> list[str]:
    parts: list[str] = []
    swatch = 12.0
    for row_index, row in enumerate(_legend_rows(entries, max_width, size=size)):
        offset = 0.0
        for label, colour, width in row:
            parts.append(
                f'<rect x="{x + offset:.1f}" y="{y_top + row_index * 17:.1f}" width="{swatch}" '
                f'height="{swatch}" rx="3" fill="{colour}"/>'
            )
            parts.append(
                f'<text x="{x + offset + swatch + 5:.1f}" y="{y_top + row_index * 17 + 10:.1f}" '
                f'font-size="{size}" fill="{INK}">{_escape(label)}</text>'
            )
            offset += width
    return parts


def _label_block(categories: Sequence[str], width: float, entries: Sequence[tuple[str, str]] | None = None,
                 size: float = 11.0) -> tuple[float, bool]:
    """Space needed below the axis: tick labels (rotated or not) plus legend rows."""
    slot = width / max(1, len(categories))
    widest = max((_text_width(c, size) for c in categories), default=0.0)
    rotated = widest > slot * 0.92
    label_height = 18.0 if not rotated else min(104.0, widest * 0.66 + 14.0)
    legend_height = _legend_height(entries, width) + 20 if entries else 0.0
    return label_height + legend_height + 18.0, rotated


def _x_labels(categories: Sequence[str], x: float, w: float, y: float, size: float = 11.0,
              rotated: bool | None = None) -> list[str]:
    """Horizontal ticks when they fit, otherwise rotated — never overlapping."""
    parts: list[str] = []
    slot = w / max(1, len(categories))
    widest = max((_text_width(c, size) for c in categories), default=0)
    if rotated is None:
        rotated = widest > slot * 0.92
    if not rotated:
        for index, label in enumerate(categories):
            parts.append(
                f'<text x="{x + index * slot + slot / 2:.1f}" y="{y + 15:.1f}" font-size="{size}" '
                f'fill="{INK}" text-anchor="middle">{_escape(label)}</text>'
            )
    else:
        for index, label in enumerate(categories):
            cx = x + index * slot + slot / 2
            parts.append(
                f'<text x="{cx:.1f}" y="{y + 12:.1f}" font-size="{size - 0.5}" fill="{INK}" '
                f'text-anchor="end" transform="rotate(-38 {cx:.1f} {y + 12:.1f})">{_escape(label)}</text>'
            )
    return parts


def _axis_ticks(top: float) -> list[float]:
    """Tick positions with a readable step (0.25, not 0.1875)."""
    nice = (0.05, 0.1, 0.15, 0.2, 0.25, 0.5, 1.0, 2.0, 2.5, 5.0, 10.0, 20.0, 25.0, 50.0,
            100.0, 200.0, 250.0, 500.0, 1000.0)
    for divisions in (3, 4, 5, 2):
        step = top / divisions
        if any(abs(step - candidate) <= 1e-9 * max(1.0, candidate) for candidate in nice):
            return [step * index for index in range(divisions + 1)]
    return [top * index / 4 for index in range(5)]


def _frame(title: str, ylabel: str, ymax: float, subtitle: str | None = None,
           left: float | None = None, bottom: float | None = None,
           horizontal: bool = False) -> list[str]:
    x, y, w, h = _plot_area(bottom=bottom, left=left)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}" font-family="{FONT}">',
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="{BACKGROUND}"/>',
        f'<text x="{PAD_LEFT}" y="32" font-size="18" font-weight="600" fill="{INK}">{_escape(title)}</text>',
    ]
    if ylabel:
        parts.append(
            f'<text x="{max(14, x - 58):.0f}" y="{PAD_TOP + h / 2:.0f}" font-size="12" fill="{MUTED}" '
            f'transform="rotate(-90 {max(14, x - 58):.0f} {PAD_TOP + h / 2:.0f})">{_escape(ylabel)}</text>'
        )
    if subtitle:
        for index, line in enumerate(_wrap(subtitle, w, 10.5)[:2]):
            parts.append(
                f'<text x="{PAD_LEFT}" y="{49 + index * 13}" font-size="10.5" fill="{MUTED}">{_escape(line)}</text>'
            )
    if not horizontal:
        steps = 5
        for index in range(steps + 1):
            value = ymax * index / steps
            line_y = y + h - (value / ymax) * h if ymax else y + h
            parts.append(
                f'<line x1="{x}" y1="{line_y:.1f}" x2="{x + w}" y2="{line_y:.1f}" stroke="{GRID}" stroke-width="1"/>'
            )
            parts.append(
                f'<text x="{x - 8}" y="{line_y + 4:.1f}" font-size="11" fill="{MUTED}" text-anchor="end">'
                f'{_tick(value, ymax)}</text>'
            )
    return parts


def _wrap(text: str, max_width: float, size: float) -> list[str]:
    """Greedy wrap so no subtitle runs off the canvas."""
    words = str(text).split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and _text_width(candidate, size) > max_width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def grouped_bars(
    title: str,
    categories: Sequence[str],
    series: Sequence[Mapping[str, Any]],
    *,
    ylabel: str = "score",
    ymax: float | None = None,
    value_labels: bool = True,
    subtitle: str | None = None,
) -> str:
    legend_entries = [(item["label"], item["color"]) for item in series]
    probe_w = WIDTH - PAD_LEFT - PAD_RIGHT
    bottom, rotated = _label_block(categories, probe_w, legend_entries)
    x, y, w, h = _plot_area(bottom)
    peak = max([max(values) for values in (item["values"] for item in series) if values] or [1.0])
    top = ymax if ymax is not None else _nice_max(peak * 1.08)
    parts = _frame(title, ylabel, top, subtitle)
    groups, bars_n = len(categories), len(series)
    slot = w / max(1, groups)
    bar_width = slot * 0.78 / max(1, bars_n)
    show_values = value_labels and bar_width >= 15

    for group_index in range(groups):
        base_x = x + group_index * slot + slot * 0.11
        for series_index, item in enumerate(series):
            value = item["values"][group_index] if group_index < len(item["values"]) else None
            if value is None:
                continue
            height = (value / top) * h if top else 0
            bar_x = base_x + series_index * bar_width
            bar_y = y + h - height
            parts.append(
                f'<rect x="{bar_x:.1f}" y="{bar_y:.1f}" width="{bar_width * 0.88:.1f}" '
                f'height="{height:.1f}" rx="1.5" fill="{item["color"]}"/>'
            )
            if show_values:
                parts.append(
                    f'<text x="{bar_x + bar_width * 0.44:.1f}" y="{bar_y - 4:.1f}" font-size="8.5" '
                    f'fill="{MUTED}" text-anchor="middle">{value:.2f}</text>'
                )

    parts.extend(_x_labels(categories, x, w, y + h + 6, rotated=rotated))
    label_height = 18 if not rotated else min(104.0, max(_text_width(c, 11) for c in categories) * 0.66 + 14.0)
    parts.extend(_legend(legend_entries, x, y + h + 6 + label_height + 12, w))
    parts.append("</svg>")
    return "\n".join(parts)


def bars(
    title: str,
    categories: Sequence[str],
    values: Sequence[float],
    *,
    colors: Sequence[str] | None = None,
    ylabel: str = "value",
    ymax: float | None = None,
    subtitle: str | None = None,
) -> str:
    probe_w = WIDTH - PAD_LEFT - PAD_RIGHT
    bottom, rotated = _label_block(categories, probe_w, None)
    x, y, w, h = _plot_area(bottom)
    top = ymax if ymax is not None else _nice_max((max(values) if values else 1.0) * 1.08)
    parts = _frame(title, ylabel, top, subtitle)
    colors = list(colors or [PALETTE["P6"]] * len(categories))
    slot = w / max(1, len(categories))
    for index, (category, value) in enumerate(zip(categories, values)):
        height = (value / top) * h
        bar_x = x + index * slot + slot * 0.2
        parts.append(
            f'<rect x="{bar_x:.1f}" y="{y + h - height:.1f}" width="{slot * 0.6:.1f}" '
            f'height="{height:.1f}" rx="1.5" fill="{colors[index]}"/>'
        )
        if slot >= 46:
            parts.append(
                f'<text x="{bar_x + slot * 0.3:.1f}" y="{y + h - height - 5:.1f}" font-size="9.5" '
                f'fill="{MUTED}" text-anchor="middle">{value:.1f}</text>'
            )
    parts.extend(_x_labels(categories, x, w, y + h + 6, rotated=rotated))
    parts.append("</svg>")
    return "\n".join(parts)


def _gutter(categories: Sequence[str], size: float = 11.0) -> float:
    """Left padding wide enough for the longest category name."""
    widest = max((_text_width(c, size) for c in categories), default=0.0)
    return min(320.0, max(PAD_LEFT, widest + 26))


def horizontal_bars(
    title: str,
    categories: Sequence[str],
    values: Sequence[float],
    *,
    colors: Sequence[str] | None = None,
    annotations: Sequence[str] | None = None,
    xlabel: str = "",
    log_x: bool = False,
    subtitle: str | None = None,
) -> str:
    """Ranked horizontal bars: full names on the y axis, no rotated labels."""
    left = _gutter(categories)
    x, y, w, h = _plot_area(bottom=84, left=left)
    parts = _frame(title, "", 1.0, subtitle, left=left, bottom=84, horizontal=True)
    top = max(values) if values else 1.0
    if log_x:
        import math as _m
        lo = max(min(v for v in values if v > 0) / 1.4, 1e-7) if any(v > 0 for v in values) else 1e-7
        hi = top * 1.4
        span = _m.log10(hi) - _m.log10(lo)

        def pos(value: float) -> float:
            return x + ((_m.log10(max(value, lo)) - _m.log10(lo)) / span) * w
        ticks = [v for v in (1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3) if lo <= v <= hi]
        def _money(v: float) -> str:
            if v >= 1e-3:
                return f"${v:.4f}"
            if v >= 1e-4:
                return f"${v * 1e4:.0f}e-4"
            return f"${v * 1e5:.0f}e-5"
        tick_labels = [_money(v) for v in ticks]
        tick_pos = [pos(v) for v in ticks]
    else:
        hi = _nice_max(top * 1.05) or 1.0

        def pos(value: float) -> float:
            return x + (value / hi) * w
        ticks = _axis_ticks(hi)
        tick_labels = [f"{v:,.0f}" if hi >= 100 else (f"{v:.2f}" if top > 1 else f"{v:.2f}".rstrip("0").rstrip(".")) for v in ticks]
        tick_pos = [pos(v) for v in ticks]

    slot = h / max(1, len(categories))
    bar_h = min(26.0, slot * 0.66)
    colors = list(colors or [PALETTE["P6"]] * len(categories))
    for tick, label in zip(tick_pos, tick_labels):
        parts.append(f'<line x1="{tick:.1f}" y1="{y}" x2="{tick:.1f}" y2="{y + h}" stroke="{GRID}" stroke-width="1"/>')
        parts.append(
            f'<text x="{tick:.1f}" y="{y + h + 16}" font-size="10" fill="{MUTED}" text-anchor="middle">{label}</text>'
        )
    for index, (category, value) in enumerate(zip(categories, values)):
        cy = y + slot * index + slot / 2
        parts.append(
            f'<text x="{x - 10:.1f}" y="{cy + 4:.1f}" font-size="11" fill="{INK}" text-anchor="end">{_escape(category)}</text>'
        )
        end = pos(value)
        parts.append(
            f'<rect x="{x:.1f}" y="{cy - bar_h / 2:.1f}" width="{max(1.5, end - x):.1f}" '
            f'height="{bar_h:.1f}" rx="2" fill="{colors[index]}"/>'
        )
        note = annotations[index] if annotations else None
        if note:
            parts.append(
                f'<text x="{end + 8:.1f}" y="{cy + 4:.1f}" font-size="10" fill="{MUTED}">{_escape(note)}</text>'
            )
    if xlabel:
        parts.append(
            f'<text x="{x + w / 2:.0f}" y="{HEIGHT - 22}" font-size="11.5" fill="{MUTED}" '
            f'text-anchor="middle">{_escape(xlabel)}</text>'
        )
    parts.append("</svg>")
    return "\n".join(parts)


def horizontal_grouped_bars(
    title: str,
    categories: Sequence[str],
    series: Sequence[Mapping[str, Any]],
    *,
    xlabel: str = "",
    subtitle: str | None = None,
) -> str:
    """Horizontal grouped bars: full names on the y axis, legend under the axis."""
    left = _gutter(categories)
    legend_entries = [(item["label"], item["color"]) for item in series]
    bottom = 84 + _legend_height(legend_entries, WIDTH - left - PAD_RIGHT)
    x, y, w, h = _plot_area(bottom=bottom, left=left)
    peak = max([max(v for v in item["values"] if v is not None) for item in series if item["values"]] or [1.0])
    top = _nice_max(peak * 1.10)
    parts = _frame(title, "", top, subtitle, left=left, bottom=bottom, horizontal=True)
    for value in _axis_ticks(top):
        gx = x + (value / top) * w
        parts.append(f'<line x1="{gx:.1f}" y1="{y}" x2="{gx:.1f}" y2="{y + h}" stroke="{GRID}" stroke-width="1"/>')
        parts.append(
            f'<text x="{gx:.1f}" y="{y + h + 16}" font-size="10" fill="{MUTED}" text-anchor="middle">{value:.2f}</text>'
        )
    slot = h / max(1, len(categories))
    bar_h = min(13.0, slot * 0.36)
    for index, category in enumerate(categories):
        cy = y + slot * index + slot / 2
        parts.append(
            f'<text x="{x - 10:.1f}" y="{cy + 4:.1f}" font-size="11" fill="{INK}" text-anchor="end">{_escape(category)}</text>'
        )
        for s_index, item in enumerate(series):
            value = item["values"][index]
            if value is None:
                continue
            width = (value / top) * w
            bar_y = cy - (len(series) * bar_h) / 2 + s_index * bar_h
            parts.append(
                f'<rect x="{x:.1f}" y="{bar_y:.1f}" width="{max(1.5, width):.1f}" height="{bar_h * 0.86:.1f}" '
                f'rx="1.5" fill="{item["color"]}"/>'
            )
            parts.append(
                f'<text x="{x + width + 6:.1f}" y="{bar_y + bar_h * 0.72:.1f}" font-size="9.5" fill="{MUTED}">{value:.2f}</text>'
            )
    if xlabel:
        parts.append(
            f'<text x="{x + w / 2:.0f}" y="{y + h + 30}" font-size="11.5" fill="{MUTED}" '
            f'text-anchor="middle">{_escape(xlabel)}</text>'
        )
    parts.extend(_legend(legend_entries, x, y + h + 44, w))
    parts.append("</svg>")
    return "\n".join(parts)


def scatter(
    title: str,
    points: Sequence[Mapping[str, Any]],
    *,
    xlabel: str,
    ylabel: str,
    xmax: float | None = None,
    ymax: float | None = None,
    subtitle: str | None = None,
    log_x: bool = False,
    x_ticks: Sequence[float] | None = None,
) -> str:
    import math

    def _x_pos(value: float, lo: float, hi: float, x0: float, width: float) -> float:
        if not log_x:
            return x0 + (value / (hi or 1.0)) * width
        lo = max(lo, 1e-6)
        span = math.log10(hi) - math.log10(lo)
        return x0 + ((math.log10(value) - math.log10(lo)) / span) * width

    x, y, w, h = _plot_area()
    ys = [point["y"] for point in points] or [1.0]
    xs = [point["x"] for point in points] or [1.0]
    top = ymax if ymax is not None else _nice_max(max(ys) * 1.15)
    lo = min(xs) / 1.25 if log_x else 0.0
    hi = xmax if xmax is not None else (max(xs) * 1.25 if log_x else max(xs) * 1.15 or 1.0)
    parts = _frame(title, ylabel, top, subtitle)

    # x grid + tick labels
    if x_ticks:
        ticks = list(x_ticks)
    elif log_x:
        candidates = [1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3]
        ticks = [value for value in candidates if lo <= value <= hi] or [lo, (lo * hi) ** 0.5, hi]
    else:
        ticks = [0.0, hi / 2, hi]
    for tick in ticks:
        px = _x_pos(tick, lo, hi, x, w)
        parts.append(f'<line x1="{px:.1f}" y1="{y}" x2="{px:.1f}" y2="{y + h}" stroke="{GRID}" stroke-width="1"/>')
        if tick >= 1e-3:
            label = f"${tick:.3f}"
        elif tick >= 1e-4:
            label = f"${tick * 1e4:.0f}e-4"
        else:
            label = f"${tick * 1e5:.0f}e-5"
        parts.append(
            f'<text x="{px:.1f}" y="{y + h + 16}" font-size="10" fill="{MUTED}" '
            f'text-anchor="middle">{label}</text>'
        )

    placed: list[tuple[float, float, float, float]] = []  # x, y, half_width, half_height

    def _free(cx: float, cy: float, half_w: float) -> bool:
        for ox, oy, ow, oh in placed:
            if abs(cx - ox) < (half_w + ow + 7) and abs(cy - oy) < (oh + 9):
                return False
        return True

    offsets = [(0, -16), (0, 22), (-34, -16), (34, -16), (-34, 22), (34, 22), (-62, 22), (62, 22), (0, -34), (0, 40)]
    for point in sorted(points, key=lambda p: p["x"]):
        px = _x_pos(point["x"], lo, hi, x, w)
        py = y + h - (point["y"] / top) * h
        parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="6" fill="{point["color"]}" opacity="0.92"/>')
        half_w = _text_width(point["label"], 11) / 2
        for dx, dy in offsets:
            cx, cy = px + dx, py + dy
            if cy - 9 < y or cy + 4 > y + h or cx - half_w < x:
                continue
            if _free(cx, cy, half_w):
                parts.append(
                    f'<line x1="{px:.1f}" y1="{py + (-7 if dy < 0 else 7):.1f}" x2="{cx:.1f}" '
                    f'y2="{cy + (-3 if dy < 0 else -9):.1f}" stroke="{MUTED}" stroke-width="0.7" opacity="0.55"/>'
                )
                parts.append(
                    f'<text x="{cx:.1f}" y="{cy:.1f}" font-size="11" font-weight="600" fill="{INK}" '
                    f'text-anchor="middle">{_escape(point["label"])}</text>'
                )
                placed.append((cx, cy, half_w, 7))
                break

    parts.append(
        f'<text x="{x + w / 2:.0f}" y="{y + h + 36}" font-size="11.5" fill="{MUTED}" '
        f'text-anchor="middle">{_escape(xlabel)}</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts)


def lines(
    title: str,
    x_labels: Sequence[str],
    series: Sequence[Mapping[str, Any]],
    *,
    ylabel: str = "score",
    ymax: float | None = None,
    subtitle: str | None = None,
) -> str:
    legend_entries = [(item["label"], item["color"]) for item in series]
    probe_w = WIDTH - PAD_LEFT - PAD_RIGHT
    bottom, rotated = _label_block(x_labels, probe_w, legend_entries)
    x, y, w, h = _plot_area(bottom)
    defined = [v for item in series for v in item["values"] if v is not None]
    top = ymax if ymax is not None else _nice_max((max(defined) if defined else 1.0) * 1.12)
    parts = _frame(title, ylabel, top, subtitle)
    step = w / max(1, len(x_labels) - 1)
    for item in series:
        coordinates = []
        for index, value in enumerate(item["values"]):
            if value is None:
                continue
            coordinates.append((x + index * step, y + h - (value / top) * h))
        if len(coordinates) > 1:
            points = " ".join(f"{px:.1f},{py:.1f}" for px, py in coordinates)
            parts.append(f'<polyline points="{points}" fill="none" stroke="{item["color"]}" stroke-width="2.4"/>')
        for px, py in coordinates:
            parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3.6" fill="{item["color"]}"/>')
    for index, label in enumerate(x_labels):
        parts.append(
            f'<text x="{x + index * step:.1f}" y="{y + h + 20}" font-size="11" fill="{INK}" '
            f'text-anchor="middle">{_escape(label)}</text>'
        )
    parts.extend(_legend(legend_entries, x, y + h + 38, w))
    parts.append("</svg>")
    return "\n".join(parts)


def stacked_bars(
    title: str,
    categories: Sequence[str],
    series: Sequence[Mapping[str, Any]],
    *,
    ylabel: str = "share",
    ymax: float = 1.0,
    subtitle: str | None = None,
) -> str:
    legend_entries = [(item["label"], item["color"]) for item in series]
    probe_w = WIDTH - PAD_LEFT - PAD_RIGHT
    bottom, rotated = _label_block(categories, probe_w, legend_entries)
    x, y, w, h = _plot_area(bottom)
    parts = _frame(title, ylabel, ymax, subtitle)
    slot = w / max(1, len(categories))
    for index, category in enumerate(categories):
        cursor = y + h
        for item in series:
            value = item["values"][index]
            height = (value / ymax) * h
            cursor -= height
            parts.append(
                f'<rect x="{x + index * slot + slot * 0.22:.1f}" y="{cursor:.1f}" '
                f'width="{slot * 0.56:.1f}" height="{height:.1f}" fill="{item["color"]}"/>'
            )
            if value > 0.07 and slot > 90:
                parts.append(
                    f'<text x="{x + index * slot + slot * 0.5:.1f}" y="{cursor + height / 2 + 4:.1f}" '
                    f'font-size="10" fill="#ffffff" text-anchor="middle">{value:.2f}</text>'
                )
    parts.extend(_x_labels(categories, x, w, y + h + 6, rotated=rotated))
    label_height = 18 if not rotated else min(104.0, max(_text_width(c, 11) for c in categories) * 0.66 + 14.0)
    parts.extend(_legend(legend_entries, x, y + h + 6 + label_height + 12, w))
    parts.append("</svg>")
    return "\n".join(parts)


def save(name: str, svg: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{name}.svg"
    path.write_text(svg, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# retrieval curves from stored ids (no recomputation of the index)
# ---------------------------------------------------------------------------


def _curve(records: Sequence[Mapping[str, Any]], k_values: Sequence[int] = (1, 3, 5, 8, 10)) -> tuple[list[float | None], list[float | None]]:
    """Fixed-depth curves. ``None`` where the system never retrieved that many docs.

    Returning the stopping-depth value at a larger K would silently compare R@5 against
    R@10; an undefined point must appear as a gap in the figure.
    """
    recalls: list[float | None] = []
    all_gold: list[float | None] = []
    for k in k_values:
        values, complete = [], []
        for record in records:
            gold = set(record.get("gold_document_ids") or [])
            retrieved = list(record.get("retrieved_doc_ids") or [])
            if not gold or len(retrieved) < k:
                continue
            values.append(len(gold & set(retrieved[:k])) / len(gold))
            complete.append(1.0 if gold.issubset(set(retrieved[:k])) else 0.0)
        recalls.append(round(mean(values), 4) if values else None)
        all_gold.append(round(mean(complete), 4) if complete else None)
    return recalls, all_gold


# ---------------------------------------------------------------------------
# figure set
# ---------------------------------------------------------------------------


def build_all(systems: Sequence[str] | None = None) -> list[Path]:
    systems = list(systems or [s for s in loaders.SYSTEMS if loaders.load(s, "hotpotqa")])
    quality = {row["system"]: row for row in tables.answer_quality() if row["dataset"] == "pooled"}
    present = [s for s in systems if s in quality]
    name = lambda s: loaders.SYSTEMS[s].label  # noqa: E731
    paths: list[Path] = []

    # 1. overall EM / F1 — horizontal so full names sit on the y axis
    paths.append(
        save(
            "fig1_overall_em_f1",
            horizontal_grouped_bars(
                "Answer quality by system (pooled over three datasets)",
                [name(s) for s in present],
                [
                    {"label": "EM", "values": [quality[s]["em"] for s in present], "color": "#111827"},
                    {"label": "F1", "values": [quality[s]["f1"] for s in present], "color": "#9ca3af"},
                ],
                xlabel="score",
            ),
        )
    )

    # 2. per-dataset EM — dataset names on the x axis, systems in a wrapped legend
    datasets = list(loaders.DATASETS)
    per_dataset = {
        (row["system"], row["dataset"]): row
        for row in tables.answer_quality()
        if row["dataset"] != "pooled"
    }
    paths.append(
        save(
            "fig2_per_dataset_em",
            grouped_bars(
                "Exact Match by dataset",
                datasets,
                [
                    {
                        "label": name(s),
                        "values": [per_dataset.get((s, d), {}).get("em") for d in datasets],
                        "color": PALETTE.get(s, "#111827"),
                    }
                    for s in present
                ],
                ylabel="EM",
            ),
        )
    )

    # 3. cost vs quality — cost on a log x axis, names on the y axis, F1 annotated
    efficiency = {(row["system"], row["dataset"]): row for row in tables.efficiency()}
    cost_rows = []
    for system in present:
        costs, f1s = [], []
        for dataset in datasets:
            row = efficiency.get((system, dataset))
            quality_row = per_dataset.get((system, dataset))
            if not row or not quality_row or not row.get("n") or row.get("cost_replay_total") is None:
                continue
            costs.append(float(row["cost_replay_total"]) / float(row["n"]))
            f1s.append(quality_row["f1"])
        if costs:
            cost_rows.append((system, mean(costs), mean(f1s)))
    cost_rows.sort(key=lambda item: item[2])
    paths.append(
        save(
            "fig3_cost_quality_pareto",
            horizontal_bars(
                "Cost and answer quality by system (pooled)",
                [name(item[0]) for item in cost_rows],
                [item[1] for item in cost_rows],
                colors=[PALETTE.get(item[0], "#111827") for item in cost_rows],
                annotations=[f"F1 {item[2]:.2f}" for item in cost_rows],
                xlabel="mean API cost per question (USD, log scale; cache-normalised)",
                log_x=True,
            ),
        )
    )

    # 4. input tokens — horizontal
    token_rows = [
        (s, (efficiency.get((s, "hotpotqa")) or {}).get("input_tokens") or 0.0) for s in present
    ]
    paths.append(
        save(
            "fig4_input_tokens",
            horizontal_bars(
                "Mean input tokens per question (HotpotQA)",
                [name(s) for s, _ in token_rows],
                [value for _, value in token_rows],
                colors=[PALETTE.get(s, "#111827") for s, _ in token_rows],
                xlabel="tokens",
            ),
        )
    )

    # 5. candidate recall@k — legend wraps into rows
    series = []
    for system in present:
        if not loaders.SYSTEMS[system].uses_retrieval:
            continue
        pooled: list[list[float]] = []
        for dataset in datasets:
            records = loaders.load(system, dataset)
            if records:
                pooled.append(_curve(records)[0])
        if not pooled:
            continue
        averaged = []
        for column in zip(*pooled):
            defined = [value for value in column if value is not None]
            averaged.append(round(mean(defined), 4) if defined else None)
        series.append({"label": name(system), "values": averaged, "color": PALETTE.get(system, "#111827")})
    paths.append(
        save(
            "fig5_recall_at_k",
            lines(
                "Candidate document Recall@K at fixed depth (not context recall)",
                ["K=1", "K=3", "K=5", "K=8", "K=10"],
                series,
                ylabel="recall (gap = depth below K)",
            ),
        )
    )

    # 6. all-gold@k
    series = []
    for system in present:
        if not loaders.SYSTEMS[system].uses_retrieval:
            continue
        pooled = []
        for dataset in datasets:
            records = loaders.load(system, dataset)
            if records:
                pooled.append(_curve(records)[1])
        if not pooled:
            continue
        averaged = []
        for column in zip(*pooled):
            defined = [value for value in column if value is not None]
            averaged.append(round(mean(defined), 4) if defined else None)
        series.append({"label": name(system), "values": averaged, "color": PALETTE.get(system, "#111827")})
    paths.append(
        save(
            "fig6_all_gold_at_k",
            lines(
                "All-gold-evidence coverage@K (every gold document retrieved)",
                ["K=1", "K=3", "K=5", "K=8", "K=10"],
                series,
                ylabel="all-gold coverage",
            ),
        )
    )

    # 7. PandaRAG coverage before/after
    behaviour = {row["dataset"]: row for row in tables.pandarag_behaviour()}
    if behaviour:
        order = [d for d in datasets if d in behaviour]
        paths.append(
            save(
                "fig7_pandarag_coverage",
                grouped_bars(
                    "PandaRAG evidence coverage before and after recovery rounds",
                    order,
                    [
                        {"label": "initial coverage", "values": [behaviour[d]["mean_initial_coverage"] for d in order], "color": "#c4b5fd"},
                        {"label": "final coverage", "values": [behaviour[d]["mean_coverage"] for d in order], "color": PALETTE["P6"]},
                        {"label": "all-requirements-covered rate", "values": [behaviour[d]["all_covered_rate"] for d in order], "color": "#5b21b6"},
                    ],
                    ylabel="coverage",
                    ymax=1.0,
                ),
            )
        )

    # 8. retrieval rounds distribution
    round_series = []
    colours = {1: "#ddd6fe", 2: "#8b5cf6", 3: "#4c1d95"}
    for dataset in datasets:
        records = loaders.load("P6", dataset)
        if not records:
            continue
        counts = {}
        for record in records:
            rounds = int(record.get("retrieval_rounds") or 0)
            counts[rounds] = counts.get(rounds, 0) + 1
        total = sum(counts.values())
        round_series.append((dataset, {k: v / total for k, v in counts.items()}))
    if round_series:
        keys = sorted({k for _, counts in round_series for k in counts})
        paths.append(
            save(
                "fig8_retrieval_rounds",
                stacked_bars(
                    "PandaRAG retrieval rounds per question (share of questions)",
                    [dataset for dataset, _ in round_series],
                    [
                        {
                            "label": f"{key} round{'s' if key > 1 else ''}",
                            "values": [counts.get(key, 0.0) for _, counts in round_series],
                            "color": colours.get(key, "#a78bfa"),
                        }
                        for key in keys
                    ],
                    ylabel="share of questions",
                ),
            )
        )

    # 9. faithfulness — horizontal
    faithfulness = _load_faithfulness()
    if faithfulness:
        order = sorted(faithfulness)
        paths.append(
            save(
                "fig9_faithfulness",
                horizontal_grouped_bars(
                    "Grounding: faithfulness and hallucination score",
                    [name(s) if s in loaders.SYSTEMS else s for s in order],
                    [
                        {"label": "faithfulness", "values": [faithfulness[s]["faithfulness"] for s in order], "color": "#059669"},
                        {"label": "hallucination", "values": [faithfulness[s]["hallucination"] for s in order], "color": "#dc2626"},
                    ],
                    xlabel="score",
                ),
            )
        )

    # 10. ablations — component names, not bare codes
    ablation = _load_ablations()
    if ablation:
        caption = {
            "A0": "A0 SAC-RAG (baseline)",
            "A1": "A1 + evidence planner",
            "A2": "A2 + evidence evaluator",
            "A3": "A3 + gap controller",
            "A4": "A4 + evidence-preserving context",
            "A5": "A5 + compression",
        }
        paths.append(
            save(
                "fig10_ablations",
                horizontal_grouped_bars(
                    "Ablation: component contributions (reduced sample, n=300)",
                    [caption.get(row["variant"], row["variant"]) for row in ablation],
                    [
                        {"label": "EM", "values": [row["em"] for row in ablation], "color": "#111827"},
                        {"label": "F1", "values": [row["f1"] for row in ablation], "color": "#9ca3af"},
                    ],
                    xlabel="score",
                ),
            )
        )
    return paths


FAITHFULNESS_GLOB = "results/analysis_scope_b/faithfulness_summary.json"
ABLATION_FILE = "results/ablations/summary.json"


def _load_faithfulness() -> dict[str, dict[str, float]] | None:
    path = config.REPO_ROOT / FAITHFULNESS_GLOB
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    by_directory = {meta.directory: key for key, meta in loaders.SYSTEMS.items()}
    return {
        by_directory.get(system, system): {
            "faithfulness": values["faithfulness"] or 0.0,
            "hallucination": values["hallucination_score"] or 0.0,
        }
        for system, values in payload.items()
    }


def _load_ablations() -> list[dict[str, Any]] | None:
    path = config.REPO_ROOT / ABLATION_FILE
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("rows")


def main() -> int:  # pragma: no cover
    for path in build_all():
        print("wrote", path.relative_to(config.REPO_ROOT))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
