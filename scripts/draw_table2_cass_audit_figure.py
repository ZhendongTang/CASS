from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean

from reportlab.lib import colors
from reportlab.lib.pagesizes import inch
from reportlab.pdfgen import canvas


METHODS = ["CASS", "JSON", "MEDS", "XML"]
VARIANT_TO_METHOD = {"summary": "CASS", "json": "JSON", "meds": "MEDS", "xml": "XML"}
METHOD_COLORS = {
    "CASS": colors.HexColor("#1f77b4"),
    "JSON": colors.HexColor("#cc78a8"),
    "MEDS": colors.HexColor("#a9a9a9"),
    "XML": colors.HexColor("#e6863a"),
}
LIGHT_GRID = colors.HexColor("#e8e8e8")
DARK = colors.HexColor("#222222")

TRAIN_SECONDS = {
    "Gemma-2-9B": {"CASS": 3.8, "JSON": 9.3, "MEDS": 10.8, "XML": 10.2},
    "GLM-4-9B": {"CASS": 2.4, "JSON": 6.9, "MEDS": 5.8, "XML": 10.6},
    "Llama-3-8B": {"CASS": 2.1, "JSON": 5.1, "MEDS": 6.4, "XML": 6.4},
}


def collect(root: Path) -> dict[str, dict[str, float]]:
    rows: dict[str, list[dict[str, float]]] = {method: [] for method in METHODS}
    for path in sorted(root.glob("*/*/*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for variant, method in VARIANT_TO_METHOD.items():
            metrics = data["variants"][variant]
            rows[method].append(
                {
                    "n": float(data["n"]),
                    "mean_tokens": float(metrics["token_length"]["mean"]),
                    "p95_tokens": float(metrics["token_length"]["p95"]),
                    "trunc": float(metrics["truncation_rate_at_budget"]),
                    "exact": float(metrics["exact_match_rate_under_legal_reorder"]),
                    "hashes": float(metrics["mean_unique_hashes_per_record"]),
                    "evidence_jaccard": (
                        float(metrics["evidence_jaccard_under_legal_reorder"])
                        if metrics["evidence_jaccard_under_legal_reorder"] is not None
                        else 0.0
                    ),
                    "traceable": float(metrics["records_with_traceable_evidence"]) / float(data["n"]),
                }
            )
    if any(len(items) != 27 for items in rows.values()):
        raise RuntimeError({method: len(items) for method, items in rows.items()})
    return {
        method: {
            "mean_tokens": mean(item["mean_tokens"] for item in items),
            "p95_tokens": mean(item["p95_tokens"] for item in items),
            "trunc": mean(item["trunc"] for item in items),
            "exact": mean(item["exact"] for item in items),
            "hashes": mean(item["hashes"] for item in items),
            "evidence_jaccard": mean(item["evidence_jaccard"] for item in items),
            "traceable": mean(item["traceable"] for item in items),
        }
        for method, items in rows.items()
    }


def lighten(color: colors.Color, amount: float) -> colors.Color:
    amount = max(0.0, min(1.0, amount))
    return colors.Color(
        1 - (1 - color.red) * amount,
        1 - (1 - color.green) * amount,
        1 - (1 - color.blue) * amount,
    )


def set_font(c: canvas.Canvas, name: str = "Times-Roman", size: float = 7.0) -> None:
    c.setFont(name, size)


def panel_title(c: canvas.Canvas, x: float, y: float, letter: str, title: str) -> None:
    c.setFillColor(colors.black)
    c.circle(x, y, 7.0, stroke=0, fill=1)
    c.setFillColor(colors.white)
    set_font(c, "Times-Bold", 7.2)
    c.drawCentredString(x, y - 2.2, letter)
    c.setFillColor(DARK)
    set_font(c, "Times-Bold", 11.0)
    c.drawString(x + 10.5, y - 3.2, title)


def draw_radar(c: canvas.Canvas, x: float, y: float, data: dict[str, dict[str, float]]) -> None:
    panel_title(c, x, y + 101, "a", "Stability profile")
    max_tokens = max(data[m]["mean_tokens"] for m in METHODS)
    max_hashes = max(data[m]["hashes"] for m in METHODS)
    axes = [
        ("tokens", lambda m: 1 - data[m]["mean_tokens"] / max_tokens),
        ("no\ntrunc.", lambda m: 1 - data[m]["trunc"]),
        ("exact\nmatch", lambda m: data[m]["exact"]),
        ("one\nhash", lambda m: 1 - (data[m]["hashes"] - 1) / max(1e-9, max_hashes - 1)),
        ("evid.\nJac.", lambda m: data[m]["evidence_jaccard"]),
        ("trace\nevid.", lambda m: data[m]["traceable"]),
    ]
    cx, cy, radius = x + 68, y + 48, 31
    start = math.pi / 2
    angles = [start - 2 * math.pi * i / len(axes) for i in range(len(axes))]

    c.setStrokeColor(colors.HexColor("#d6d6d6"))
    c.setLineWidth(0.55)
    for frac in [0.25, 0.5, 0.75, 1.0]:
        pts = [(cx + radius * frac * math.cos(a), cy + radius * frac * math.sin(a)) for a in angles]
        path = c.beginPath()
        path.moveTo(*pts[0])
        for pt in pts[1:]:
            path.lineTo(*pt)
        path.close()
        c.drawPath(path, stroke=1, fill=0)
    for a in angles:
        c.line(cx, cy, cx + radius * math.cos(a), cy + radius * math.sin(a))

    for idx, (label, _) in enumerate(axes):
        a = angles[idx]
        lx = cx + (radius + 9) * math.cos(a)
        ly = cy + (radius + 8) * math.sin(a)
        c.setFillColor(DARK)
        set_font(c, "Times-Roman", 5.8)
        lines = label.split("\n")
        if math.cos(a) > 0.25:
            anchor = "left"
        elif math.cos(a) < -0.25:
            anchor = "right"
        else:
            anchor = "center"
        for j, part in enumerate(lines):
            yy = ly - 3.5 * j
            if anchor == "left":
                c.drawString(lx, yy, part)
            elif anchor == "right":
                c.drawRightString(lx, yy, part)
            else:
                c.drawCentredString(lx, yy, part)

    for method in METHODS:
        vals = [max(0.0, min(1.0, fn(method))) for _, fn in axes]
        pts = [(cx + radius * val * math.cos(a), cy + radius * val * math.sin(a)) for val, a in zip(vals, angles)]
        path = c.beginPath()
        path.moveTo(*pts[0])
        for pt in pts[1:]:
            path.lineTo(*pt)
        path.close()
        c.setStrokeColor(METHOD_COLORS[method])
        c.setLineWidth(1.45 if method == "CASS" else 0.75)
        c.setFillColor(lighten(METHOD_COLORS[method], 0.18 if method == "CASS" else 0.06))
        c.drawPath(path, stroke=1, fill=1)

    legend_x, legend_y = x + 2, y + 4
    set_font(c, "Times-Roman", 6.4)
    for i, method in enumerate(METHODS):
        lx = legend_x + i * 32
        c.setStrokeColor(METHOD_COLORS[method])
        c.setLineWidth(1.4)
        c.line(lx, legend_y, lx + 9, legend_y)
        c.setFillColor(DARK)
        c.drawString(lx + 11, legend_y - 2.2, method)


def draw_budget(c: canvas.Canvas, x: float, y: float, data: dict[str, dict[str, float]]) -> None:
    panel_title(c, x, y + 101, "b", "Budget margin")
    plot_x, plot_y, plot_w, plot_h = x + 17, y + 15, 105, 78
    ymax = 4200
    c.setStrokeColor(colors.HexColor("#777777"))
    c.setLineWidth(0.7)
    c.line(plot_x, plot_y, plot_x, plot_y + plot_h)
    c.line(plot_x, plot_y, plot_x + plot_w, plot_y)
    set_font(c, "Times-Roman", 8.0)
    c.setFillColor(DARK)
    for tick in [0, 2000, 4000]:
        ty = plot_y + plot_h * tick / ymax
        c.setStrokeColor(LIGHT_GRID)
        c.line(plot_x, ty, plot_x + plot_w, ty)
        c.setFillColor(DARK)
        c.drawRightString(plot_x - 4, ty - 2.5, str(tick))
    c.setStrokeColor(colors.HexColor("#d06060"))
    c.setDash(3, 2)
    by = plot_y + plot_h * 2048 / ymax
    c.line(plot_x, by, plot_x + plot_w, by)
    c.setDash()
    c.setFillColor(colors.HexColor("#d06060"))
    c.drawString(plot_x + 1, by + 2, "2048")
    bar_w = 14
    for i, method in enumerate(METHODS):
        cx = plot_x + 15 + i * 27
        mean_tok = data[method]["mean_tokens"]
        p95 = data[method]["p95_tokens"]
        h = plot_h * mean_tok / ymax
        c.setFillColor(METHOD_COLORS[method])
        c.rect(cx - bar_w / 2, plot_y, bar_w, h, stroke=0, fill=1)
        err_top = plot_y + plot_h * p95 / ymax
        c.setStrokeColor(DARK)
        c.setLineWidth(0.8)
        c.line(cx, plot_y + h, cx, err_top)
        c.line(cx - 4, err_top, cx + 4, err_top)
        c.setFillColor(METHOD_COLORS[method])
        set_font(c, "Times-Bold", 7.0)
        c.drawCentredString(cx, plot_y + h + 5, f"{mean_tok:.0f}")
        c.saveState()
        c.translate(cx + 5, plot_y - 6)
        c.rotate(35)
        c.setFillColor(DARK)
        set_font(c, "Times-Roman", 8.2)
        c.drawRightString(0, 0, method)
        c.restoreState()
    c.saveState()
    c.translate(plot_x - 28, plot_y + 43)
    c.rotate(90)
    set_font(c, "Times-Roman", 8.6)
    c.setFillColor(DARK)
    c.drawCentredString(0, 0, "tokens")
    c.restoreState()


def draw_truncation(c: canvas.Canvas, x: float, y: float, data: dict[str, dict[str, float]]) -> None:
    panel_title(c, x, y + 101, "c", "Truncation risk")
    plot_x, plot_y, plot_w, plot_h = x + 18, y + 18, 103, 76
    c.setStrokeColor(colors.HexColor("#777777"))
    c.line(plot_x, plot_y, plot_x, plot_y + plot_h)
    c.line(plot_x, plot_y, plot_x + plot_w, plot_y)
    set_font(c, "Times-Roman", 8.4)
    for tick in [0, 50, 100]:
        tx = plot_x + plot_w * tick / 100
        c.setStrokeColor(LIGHT_GRID)
        c.line(tx, plot_y, tx, plot_y + plot_h)
        c.setFillColor(DARK)
        c.drawCentredString(tx, plot_y - 10, str(tick))
    for i, method in enumerate(METHODS):
        yy = plot_y + plot_h - 14 - i * 19
        c.setFillColor(DARK)
        c.drawRightString(plot_x - 9, yy - 3, method)
        c.setFillColor(colors.HexColor("#eeeeee"))
        c.rect(plot_x, yy - 5, plot_w, 9, stroke=0, fill=1)
        val = data[method]["trunc"] * 100
        c.setFillColor(METHOD_COLORS[method])
        c.rect(plot_x, yy - 5, plot_w * min(val, 100) / 100, 9, stroke=0, fill=1)
        label = "<0.01%" if method == "CASS" and val < 0.01 else f"{val:.1f}%"
        set_font(c, "Times-Bold", 7.3)
        c.drawString(plot_x + plot_w + 4, yy - 3, label)
    set_font(c, "Times-Roman", 8.6)
    c.setFillColor(DARK)
    c.drawCentredString(plot_x + plot_w / 2, plot_y - 22, "records >2048 (%)")


def draw_speed(c: canvas.Canvas, x: float, y: float) -> None:
    panel_title(c, x, y + 101, "d", "Fine-tuning sec/step")
    grid_x, grid_y = x + 21, y + 22
    cell_w, cell_h = 25, 21
    llms = list(TRAIN_SECONDS)
    max_ratio = max(TRAIN_SECONDS[llm][m] / TRAIN_SECONDS[llm]["CASS"] for llm in llms for m in METHODS)
    for r, llm in enumerate(llms):
        yy = grid_y + (2 - r) * cell_h
        set_font(c, "Times-Roman", 7.4)
        c.setFillColor(DARK)
        c.drawRightString(grid_x - 5, yy + 7, llm)
        base = TRAIN_SECONDS[llm]["CASS"]
        for col, method in enumerate(METHODS):
            ratio = TRAIN_SECONDS[llm][method] / base
            shade = lighten(colors.HexColor("#e6863a"), 0.12 + 0.88 * ((ratio - 1) / (max_ratio - 1)))
            c.setFillColor(shade)
            c.rect(grid_x + col * cell_w, yy, cell_w, cell_h, stroke=0, fill=1)
            c.setFillColor(colors.white if ratio > 3.0 else DARK)
            set_font(c, "Times-Bold" if method == "CASS" else "Times-Roman", 6.6)
            c.drawCentredString(grid_x + col * cell_w + cell_w / 2, yy + 11.5, f"{TRAIN_SECONDS[llm][method]:.1f}s")
            c.drawCentredString(grid_x + col * cell_w + cell_w / 2, yy + 3.5, f"{ratio:.1f}x")
    set_font(c, "Times-Roman", 8.4)
    for col, method in enumerate(METHODS):
        c.saveState()
        c.translate(grid_x + col * cell_w + 12.5, grid_y - 5)
        c.rotate(35)
        c.setFillColor(DARK)
        c.drawRightString(0, 0, method)
        c.restoreState()
    c.drawCentredString(grid_x + 2 * cell_w, grid_y - 28, "format")


def draw_figure(data: dict[str, dict[str, float]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    width = 4.22 * inch
    height = 3.58 * inch
    c = canvas.Canvas(str(output), pagesize=(width, height))
    c.setTitle("Table 2 CASS audit scorecard")
    draw_radar(c, 32, 136, data)
    draw_budget(c, 168, 136, data)
    draw_truncation(c, 32, 14, data)
    draw_speed(c, 168, 14)
    c.save()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    draw_figure(collect(args.root), args.output)
    print(args.output)


if __name__ == "__main__":
    main()
