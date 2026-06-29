import json
from collections import defaultdict
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas


FIG_DIR = Path(__file__).resolve().parent
DATA = json.loads((FIG_DIR / "summary_figure_data_20260630.json").read_text())

COLORS = {
    "MIMIC-IV": HexColor("#0072B2"),
    "eICU": HexColor("#D55E00"),
    "NWICU": HexColor("#009E73"),
    "5k": HexColor("#8DA0CB"),
    "10k": HexColor("#FC8D62"),
    "20k": HexColor("#66C2A5"),
    "Short-context XML": HexColor("#999999"),
    "ClinTime-Summary": HexColor("#0072B2"),
    "HistGBDT reference": HexColor("#CC79A7"),
}


def mean(values):
    return sum(values) / len(values)


def std(values):
    if len(values) <= 1:
        return 0
    m = mean(values)
    return (sum((v - m) ** 2 for v in values) / len(values)) ** 0.5


def ymap(y, y0, h, y_min, y_max):
    return y0 + (y - y_min) / (y_max - y_min) * h


def draw_text(c, x, y, text, size=7.5, font="Times-Roman", color=colors.black, anchor="start", rotate=0):
    c.saveState()
    c.setFont(font, size)
    c.setFillColor(color)
    c.translate(x, y)
    if rotate:
        c.rotate(rotate)
    width = c.stringWidth(text, font, size)
    dx = 0
    if anchor == "middle":
        dx = -width / 2
    elif anchor == "end":
        dx = -width
    c.drawString(dx, 0, text)
    c.restoreState()


def draw_axes(c, x0, y0, w, h, y_min, y_max, y_label=None, x_label=None, x_ticks=None, x_labels=None):
    c.setStrokeColor(colors.black)
    c.setLineWidth(0.55)
    c.line(x0, y0, x0, y0 + h)
    c.line(x0, y0, x0 + w, y0)

    for tick in [0.5, 0.6, 0.7, 0.8]:
        if y_min <= tick <= y_max:
            yy = ymap(tick, y0, h, y_min, y_max)
            c.setStrokeColor(HexColor("#D7D7D7"))
            c.setLineWidth(0.35)
            c.line(x0, yy, x0 + w, yy)
            c.setStrokeColor(colors.black)
            c.setLineWidth(0.45)
            c.line(x0 - 2, yy, x0, yy)
            draw_text(c, x0 - 5, yy - 2.2, f"{tick:.1f}", size=6.8, anchor="end")

    if x_ticks and x_labels:
        for xx, label in zip(x_ticks, x_labels):
            c.setStrokeColor(colors.black)
            c.setLineWidth(0.45)
            c.line(xx, y0, xx, y0 - 2)
            draw_text(c, xx, y0 - 10, label, size=7.0, anchor="middle")

    if y_label:
        draw_text(c, x0 - 28, y0 + h / 2, y_label, size=8.0, anchor="middle", rotate=90)
    if x_label:
        draw_text(c, x0 + w / 2, y0 - 24, x_label, size=8.0, anchor="middle")


def draw_legend(c, items, x, y, sw=8, gap=62):
    for i, (label, color) in enumerate(items):
        xx = x + i * gap
        c.setFillColor(color)
        c.rect(xx, y - 1, sw, 5, fill=1, stroke=0)
        draw_text(c, xx + sw + 3, y - 1, label, size=7.0)


def fig_mortality_scaling():
    rows = DATA["mortality_scaling"]
    centers = ["MIMIC-IV", "eICU", "NWICU"]
    settings = ["5k", "10k", "20k"]
    W, H = 7.05 * inch, 2.22 * inch
    c = canvas.Canvas(str(FIG_DIR / "fig_summary_mortality_scaling.pdf"), pagesize=(W, H))
    left, bottom, panel_w, panel_h, gap = 0.55 * inch, 0.48 * inch, 1.74 * inch, 1.25 * inch, 0.42 * inch
    y_min, y_max = 0.44, 0.84

    for idx, center in enumerate(centers):
        x0 = left + idx * (panel_w + gap)
        y0 = bottom
        xs = [x0 + panel_w * p for p in (0.13, 0.50, 0.87)]
        draw_axes(
            c, x0, y0, panel_w, panel_h, y_min, y_max,
            y_label="AUROC" if idx == 0 else None,
            x_label="MIMIC train records",
            x_ticks=xs,
            x_labels=settings,
        )
        draw_text(c, x0 + 4, y0 + panel_h - 9, center, size=8.3, font="Times-Bold")

        values_by_setting = defaultdict(list)
        for row in rows:
            if row["center"] == center:
                values_by_setting[row["setting"]].append(row["auroc"])
        means = [mean(values_by_setting[s]) for s in settings]
        sds = [std(values_by_setting[s]) for s in settings]
        pts = [(x, ymap(v, y0, panel_h, y_min, y_max)) for x, v in zip(xs, means)]

        c.setStrokeColor(COLORS[center])
        c.setLineWidth(1.15)
        for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
            c.line(x1, y1, x2, y2)
        for (x, y), sd in zip(pts, sds):
            if sd:
                yy1 = ymap(mean([y_min, y_min]) + (means[pts.index((x, y))] - sd - y_min), y0, panel_h, y_min, y_max)
                yy2 = ymap(means[pts.index((x, y))] + sd, y0, panel_h, y_min, y_max)
                c.line(x, yy1, x, yy2)
                c.line(x - 2, yy1, x + 2, yy1)
                c.line(x - 2, yy2, x + 2, yy2)
        for i, setting in enumerate(settings):
            vals = values_by_setting[setting]
            offsets = [-3.5, 0, 3.5] if len(vals) == 3 else ([0] if len(vals) == 1 else [-2.5, 2.5])
            for off, val in zip(offsets, vals):
                c.setFillColor(colors.white)
                c.setStrokeColor(COLORS[center])
                c.circle(xs[i] + off, ymap(val, y0, panel_h, y_min, y_max), 2.35, fill=1, stroke=1)
        c.setStrokeColor(HexColor("#555555"))
        c.setDash(2, 2)
        c.line(x0, ymap(0.5, y0, panel_h, y_min, y_max), x0 + panel_w, ymap(0.5, y0, panel_h, y_min, y_max))
        c.setDash()
    c.showPage()
    c.save()
    print("Saved", FIG_DIR / "fig_summary_mortality_scaling.pdf")


def fig_multitask():
    rows = DATA["multitask"]
    tasks = ["LOS >48h", "Future intervention"]
    centers = ["MIMIC-IV", "eICU", "NWICU"]
    train_sizes = ["5k", "10k"]
    W, H = 7.05 * inch, 2.28 * inch
    c = canvas.Canvas(str(FIG_DIR / "fig_summary_multitask.pdf"), pagesize=(W, H))
    draw_legend(c, [(s, COLORS[s]) for s in train_sizes], W - 1.65 * inch, H - 0.22 * inch, gap=38)
    left, bottom, panel_w, panel_h, gap = 0.55 * inch, 0.50 * inch, 2.74 * inch, 1.28 * inch, 0.52 * inch
    y_min, y_max = 0.46, 0.86

    for idx, task in enumerate(tasks):
        x0 = left + idx * (panel_w + gap)
        y0 = bottom
        groups = [x0 + panel_w * p for p in (0.18, 0.50, 0.82)]
        draw_axes(
            c, x0, y0, panel_w, panel_h, y_min, y_max,
            y_label="AUROC" if idx == 0 else None,
            x_label="Evaluation center",
            x_ticks=groups,
            x_labels=centers,
        )
        draw_text(c, x0 + 4, y0 + panel_h + 4, task, size=8.3, font="Times-Bold")
        bw = 12
        for gi, center in enumerate(centers):
            for tj, train_size in enumerate(train_sizes):
                val = [r for r in rows if r["task"] == task and r["train_records"] == train_size and r["center"] == center][0]["auroc"]
                x = groups[gi] + (tj - 0.5) * (bw + 2)
                y = ymap(val, y0, panel_h, y_min, y_max)
                c.setFillColor(COLORS[train_size])
                c.setStrokeColor(colors.black)
                c.setLineWidth(0.3)
                c.rect(x - bw / 2, y0, bw, y - y0, fill=1, stroke=1)
                draw_text(c, x, y + 3, f"{val:.2f}", size=6.2, anchor="middle", rotate=90)
        c.setStrokeColor(HexColor("#555555"))
        c.setDash(2, 2)
        c.line(x0, ymap(0.5, y0, panel_h, y_min, y_max), x0 + panel_w, ymap(0.5, y0, panel_h, y_min, y_max))
        c.setDash()
    c.showPage()
    c.save()
    print("Saved", FIG_DIR / "fig_summary_multitask.pdf")


def fig_diagnostic_positioning():
    rows = DATA["diagnostic_positioning"]
    centers = ["MIMIC-IV", "eICU", "NWICU"]
    methods = ["Short-context XML", "ClinTime-Summary", "HistGBDT reference"]
    W, H = 7.05 * inch, 2.35 * inch
    c = canvas.Canvas(str(FIG_DIR / "fig_summary_positioning.pdf"), pagesize=(W, H))
    draw_legend(c, [(m, COLORS[m]) for m in methods], 0.92 * inch, H - 0.22 * inch, gap=108)
    x0, y0, w, h = 0.60 * inch, 0.48 * inch, 5.95 * inch, 1.32 * inch
    y_min, y_max = 0.45, 0.89
    groups = [x0 + w * p for p in (0.17, 0.50, 0.83)]
    draw_axes(c, x0, y0, w, h, y_min, y_max, y_label="AUROC", x_label="Evaluation center", x_ticks=groups, x_labels=centers)
    bw = 15
    for gi, center in enumerate(centers):
        for mi, method in enumerate(methods):
            val = [r for r in rows if r["method"] == method and r["center"] == center][0]["auroc"]
            x = groups[gi] + (mi - 1) * (bw + 3)
            y = ymap(val, y0, h, y_min, y_max)
            c.setFillColor(COLORS[method])
            c.setStrokeColor(colors.black)
            c.setLineWidth(0.3)
            c.rect(x - bw / 2, y0, bw, y - y0, fill=1, stroke=1)
            draw_text(c, x, y + 3, f"{val:.2f}", size=6.4, anchor="middle", rotate=90)
    c.setStrokeColor(HexColor("#555555"))
    c.setDash(2, 2)
    c.line(x0, ymap(0.5, y0, h, y_min, y_max), x0 + w, ymap(0.5, y0, h, y_min, y_max))
    c.setDash()
    c.showPage()
    c.save()
    print("Saved", FIG_DIR / "fig_summary_positioning.pdf")


if __name__ == "__main__":
    fig_mortality_scaling()
    fig_multitask()
    fig_diagnostic_positioning()
