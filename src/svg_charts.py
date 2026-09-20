"""Minimal dependency-free SVG chart helpers for the final report.

The project deliberately avoids extra dependencies; publication figures are
written as plain SVG (a text format) using only the standard library.
"""

from __future__ import annotations

from html import escape
from typing import Sequence

PALETTE = ("#2f6fb5", "#d1701f", "#4f9a4f", "#a0453f", "#6b5b95", "#5a5a5a")
W, H = 760, 440
L, R, T, B = 74, 22, 62, 74


def _plot_box() -> tuple[float, float, float, float]:
    return L, T, W - L - R, H - T - B


def _axes(ymax: float, ylabel: str, n_grid: int = 5) -> list[str]:
    px, py, pw, ph = _plot_box()
    out = [
        f'<line x1="{px}" y1="{py + ph}" x2="{px + pw}" y2="{py + ph}" stroke="#333" stroke-width="1"/>',
        f'<line x1="{px}" y1="{py}" x2="{px}" y2="{py + ph}" stroke="#333" stroke-width="1"/>',
        f'<text x="16" y="{py + ph / 2}" fill="#333" font-size="12" text-anchor="middle" '
        f'transform="rotate(-90 16 {py + ph / 2})">{escape(ylabel)}</text>',
    ]
    for step in range(n_grid + 1):
        value = ymax * step / n_grid
        y = py + ph - ph * step / n_grid
        out.append(f'<line x1="{px}" y1="{y:.1f}" x2="{px + pw}" y2="{y:.1f}" stroke="#e2e2e2" stroke-width="1"/>')
        out.append(
            f'<text x="{px - 8}" y="{y + 4:.1f}" fill="#444" font-size="11" text-anchor="end">{value:.2f}</text>'
        )
    return out


def _legend(series: Sequence[tuple[str, Sequence[float], str]]) -> list[str]:
    out = []
    x = L
    for name, _, color in series:
        out.append(f'<rect x="{x}" y="24" width="12" height="12" fill="{color}"/>')
        out.append(f'<text x="{x + 17}" y="35" fill="#222" font-size="12">{escape(name)}</text>')
        x += 24 + 7.0 * len(name)
    return out


def _title(title: str) -> list[str]:
    return [f'<text x="{W / 2}" y="16" fill="#111" font-size="15" font-weight="bold" text-anchor="middle">{escape(title)}</text>']


def _wrap(body: list[str], title: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">'
        f'<rect width="{W}" height="{H}" fill="white"/>' + "".join(_title(title) + body) + "</svg>"
    )


def _auto_ymax(series) -> float:
    peak = max((max(v) for _, v, _ in series if v), default=1.0)
    return peak * 1.15 if peak > 0 else 1.0


def grouped_bars(
    title: str,
    categories: Sequence[str],
    series: Sequence[tuple[str, Sequence[float], str]],
    ylabel: str = "",
    ymax: float | None = None,
    value_labels: bool = True,
) -> str:
    px, py, pw, ph = _plot_box()
    ymax = ymax if ymax is not None else _auto_ymax(series)
    body = _axes(ymax, ylabel) + _legend(series)
    ncat, nser = len(categories), len(series)
    group = pw / ncat
    bar = group * 0.78 / nser
    for ci, cat in enumerate(categories):
        for si, (_, values, color) in enumerate(series):
            value = values[ci]
            height = ph * value / ymax
            x = px + group * ci + group * 0.11 + bar * si
            y = py + ph - height
            body.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar:.1f}" height="{height:.1f}" fill="{color}"/>')
            if value_labels:
                body.append(
                    f'<text x="{x + bar / 2:.1f}" y="{y - 3:.1f}" fill="#333" font-size="9" text-anchor="middle">{value:.2f}</text>'
                )
        body.append(
            f'<text x="{px + group * ci + group / 2:.1f}" y="{py + ph + 18}" fill="#333" font-size="12" '
            f'text-anchor="middle">{escape(cat)}</text>'
        )
    return _wrap(body, title)


def stacked_bars(
    title: str,
    categories: Sequence[str],
    series: Sequence[tuple[str, Sequence[float], str]],
    ylabel: str = "",
    value_labels: bool = True,
) -> str:
    px, py, pw, ph = _plot_box()
    totals = [sum(s[1][ci] for s in series) for ci in range(len(categories))]
    ymax = max(totals) * 1.15 or 1.0
    body = _axes(ymax, ylabel) + _legend(series)
    group = pw / len(categories)
    bar = group * 0.55
    for ci in range(len(categories)):
        cumulative = 0.0
        for _, values, color in series:
            value = values[ci]
            height = ph * value / ymax
            x = px + group * ci + (group - bar) / 2
            y = py + ph - ph * (cumulative + value) / ymax
            body.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar:.1f}" height="{height:.1f}" fill="{color}"/>')
            if value_labels and value > 0:
                body.append(
                    f'<text x="{x + bar / 2:.1f}" y="{y + height / 2 + 4:.1f}" fill="white" font-size="10" '
                    f'text-anchor="middle">{value:g}</text>'
                )
            cumulative += value
        body.append(
            f'<text x="{px + group * ci + group / 2:.1f}" y="{py + ph + 18}" fill="#333" font-size="12" '
            f'text-anchor="middle">{escape(categories[ci])}</text>'
        )
    return _wrap(body, title)


def line_series(
    title: str,
    xs: Sequence[str],
    series: Sequence[tuple[str, Sequence[float], str]],
    ylabel: str = "",
    ymax: float | None = None,
    markers: Sequence[tuple[float, float, str, str]] = (),
    xlabel: str = "",
) -> str:
    px, py, pw, ph = _plot_box()
    ymax = ymax if ymax is not None else _auto_ymax(series)
    body = _axes(ymax, ylabel) + _legend(series)
    step = pw / max(1, len(xs) - 1)
    for _, values, color in series:
        points = " ".join(f"{px + step * i:.1f},{py + ph - ph * v / ymax:.1f}" for i, v in enumerate(values))
        body.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        for i, v in enumerate(values):
            body.append(f'<circle cx="{px + step * i:.1f}" cy="{py + ph - ph * v / ymax:.1f}" r="3.5" fill="{color}"/>')
    for mx, my, mlabel, color in markers:
        x = px + step * mx
        y = py + ph - ph * my / ymax
        body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="none" stroke="{color}" stroke-width="2.5"/>')
        body.append(f'<text x="{x + 8:.1f}" y="{y + 4:.1f}" fill="{color}" font-size="11">{escape(mlabel)}</text>')
    for i, label in enumerate(xs):
        body.append(
            f'<text x="{px + step * i:.1f}" y="{py + ph + 18}" fill="#333" font-size="12" text-anchor="middle">{escape(label)}</text>'
        )
    if xlabel:
        body.append(f'<text x="{px + pw / 2}" y="{py + ph + 44}" fill="#333" font-size="12" text-anchor="middle">{escape(xlabel)}</text>')
    return _wrap(body, title)


def scatter(
    title: str,
    points: Sequence[tuple[float, float, str, str]],
    xlabel: str,
    ylabel: str,
    x_pad: float = 0.08,
    y_pad: float = 0.10,
) -> str:
    px, py, pw, ph = _plot_box()
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    xlo, xhi = min(xs), max(xs)
    ylo, yhi = min(ys), max(ys)
    xspan = (xhi - xlo) or 1.0
    yspan = (yhi - ylo) or 1.0
    xlo -= xspan * x_pad
    xhi += xspan * x_pad
    ylo = max(0.0, ylo - yspan * y_pad)
    yhi += yspan * y_pad
    body = _axes(1.0, ylabel, n_grid=4)
    # relabel y axis with real values by overlaying text (axes drawn for 0..1)
    body = [b for b in body if "text-anchor=\"end\"" not in b or "font-size=\"11\"" not in b]
    for step in range(5):
        value = ylo + (yhi - ylo) * step / 4
        y = py + ph - ph * step / 4
        body.append(f'<text x="{px - 8}" y="{y + 4:.1f}" fill="#444" font-size="11" text-anchor="end">{value:.3f}</text>')
    for step in range(5):
        value = xlo + (xhi - xlo) * step / 4
        x = px + pw * step / 4
        body.append(f'<text x="{x:.1f}" y="{py + ph + 18}" fill="#333" font-size="11" text-anchor="middle">{value:.4f}</text>')
    for x, y, label, color in points:
        sx = px + pw * (x - xlo) / (xhi - xlo)
        sy = py + ph - ph * (y - ylo) / (yhi - ylo)
        body.append(f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="9" fill="{color}" fill-opacity="0.85"/>')
        body.append(f'<text x="{sx:.1f}" y="{sy - 14:.1f}" fill="#222" font-size="12" text-anchor="middle">{escape(label)}</text>')
    body.append(f'<text x="{px + pw / 2}" y="{py + ph + 44}" fill="#333" font-size="12" text-anchor="middle">{escape(xlabel)}</text>')
    return _wrap(body, title)
