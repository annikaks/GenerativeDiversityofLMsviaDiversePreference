#!/usr/bin/env python3
"""Throwaway plot without external deps: outputs SVG.

Plots thinking-model accuracy vs two deviation metrics:
- Average deviation
- Max deviation
"""

from pathlib import Path


def main() -> None:
    # label, x=accuracy, avg_deviation, max_deviation
    pts = [
        ("GPT", 31.20, 0.0953, 0.1908),
        ("Claude", 57.31, 0.0644, 0.0929),
        ("Grok", 62.29, 0.0520, 0.0736),
    ]
    pts = sorted(pts, key=lambda p: p[1])

    # Canvas + plot region
    W, H = 900, 560
    L, R, T, B = 90, 40, 50, 85
    pw, ph = W - L - R, H - T - B

    xs = [p[1] for p in pts]
    avg_ys = [p[2] for p in pts]
    max_ys = [p[3] for p in pts]
    ys = avg_ys + max_ys

    # Tight y scaling to reveal spacing.
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_pad = max((x_max - x_min) * 0.08, 1.0)
    y_pad = max((y_max - y_min) * 0.15, 0.002)

    x0, x1 = x_min - x_pad, x_max + x_pad
    y0, y1 = y_min - y_pad, y_max + y_pad

    def sx(x: float) -> float:
        return L + (x - x0) / (x1 - x0) * pw

    def sy(y: float) -> float:
        return T + (y1 - y) / (y1 - y0) * ph

    # Build polyline paths in point order.
    poly_avg = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for _, x, y, _ in pts)
    poly_max = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for _, x, _, y in pts)

    # Basic ticks
    xticks = [round(v, 1) for v in [x0, (x0 + x1) / 2, x1]]
    yticks = [round(v, 4) for v in [y0, (y0 + y1) / 2, y1]]

    lines = []
    lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
    lines.append('<rect x="0" y="0" width="100%" height="100%" fill="white"/>')

    # Axes
    lines.append(f'<line x1="{L}" y1="{H-B}" x2="{W-R}" y2="{H-B}" stroke="#222" stroke-width="2"/>')
    lines.append(f'<line x1="{L}" y1="{T}" x2="{L}" y2="{H-B}" stroke="#222" stroke-width="2"/>')

    # Grid + ticks
    for xv in xticks:
        xpix = sx(xv)
        lines.append(f'<line x1="{xpix:.1f}" y1="{T}" x2="{xpix:.1f}" y2="{H-B}" stroke="#ddd" stroke-width="1"/>')
        lines.append(f'<text x="{xpix:.1f}" y="{H-B+24}" text-anchor="middle" font-size="13" fill="#333">{xv}</text>')
    for yv in yticks:
        ypix = sy(yv)
        lines.append(f'<line x1="{L}" y1="{ypix:.1f}" x2="{W-R}" y2="{ypix:.1f}" stroke="#ddd" stroke-width="1"/>')
        lines.append(f'<text x="{L-10}" y="{ypix+4:.1f}" text-anchor="end" font-size="13" fill="#333">{yv}</text>')

    # Title + labels
    lines.append('<text x="450" y="28" text-anchor="middle" font-size="20" font-weight="700" fill="#111">Thinking Models: Deviation vs Accuracy</text>')
    lines.append(f'<text x="{(L+W-R)/2:.1f}" y="{H-30}" text-anchor="middle" font-size="15" fill="#111">Accuracy (LLM-as-judge mean score)</text>')
    lines.append(f'<text x="25" y="{(T+H-B)/2:.1f}" transform="rotate(-90 25 {(T+H-B)/2:.1f})" text-anchor="middle" font-size="15" fill="#111">Deviation</text>')

    # Lines + points
    lines.append(f'<polyline points="{poly_avg}" fill="none" stroke="#1f77b4" stroke-width="3"/>')
    lines.append(f'<polyline points="{poly_max}" fill="none" stroke="#2ca02c" stroke-width="3"/>')

    for label, x, avg_y, max_y in pts:
        xpix = sx(x)
        yavg = sy(avg_y)
        ymax = sy(max_y)
        lines.append(f'<circle cx="{xpix:.1f}" cy="{yavg:.1f}" r="5" fill="#1f77b4"/>')
        lines.append(f'<circle cx="{xpix:.1f}" cy="{ymax:.1f}" r="5" fill="#2ca02c"/>')
        # Label both average and max points.
        lines.append(f'<text x="{xpix+8:.1f}" y="{yavg-8:.1f}" font-size="13" font-weight="700" fill="#111">{label}</text>')
        lines.append(f'<text x="{xpix+8:.1f}" y="{ymax+15:.1f}" font-size="13" font-weight="700" fill="#111">{label}</text>')

    # Legend (key)
    legend_x = W - R - 205
    legend_y = T + 8
    lines.append(f'<rect x="{legend_x}" y="{legend_y}" width="190" height="56" fill="white" stroke="#bbb"/>')
    lines.append(f'<line x1="{legend_x+10}" y1="{legend_y+18}" x2="{legend_x+45}" y2="{legend_y+18}" stroke="#1f77b4" stroke-width="3"/>')
    lines.append(f'<circle cx="{legend_x+27}" cy="{legend_y+18}" r="4" fill="#1f77b4"/>')
    lines.append(f'<text x="{legend_x+55}" y="{legend_y+22}" font-size="12" fill="#111">Average Deviation</text>')
    lines.append(f'<line x1="{legend_x+10}" y1="{legend_y+40}" x2="{legend_x+45}" y2="{legend_y+40}" stroke="#2ca02c" stroke-width="3"/>')
    lines.append(f'<circle cx="{legend_x+27}" cy="{legend_y+40}" r="4" fill="#2ca02c"/>')
    lines.append(f'<text x="{legend_x+55}" y="{legend_y+44}" font-size="12" fill="#111">Max Deviation</text>')

    lines.append('</svg>')

    out_path = Path("outputs/analysis/thinking_accuracy_vs_deviation.svg")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
