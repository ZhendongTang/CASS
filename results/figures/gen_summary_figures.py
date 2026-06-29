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


def draw_panel_label(c, x, y, label, title):
    c.setFillColor(colors.black)
    c.circle(x + 6, y - 5, 5.2, fill=1, stroke=0)
    draw_text(c, x + 6, y - 8, label, size=7.2, font="Times-Bold", color=colors.white, anchor="middle")
    draw_text(c, x + 16, y - 8, title, size=8.5, font="Times-Bold")


def draw_box(c, x, y, w, h, text, fill, stroke=HexColor("#444444"), size=7.1):
    c.setFillColor(fill)
    c.setStrokeColor(stroke)
    c.setLineWidth(0.45)
    c.roundRect(x, y, w, h, 4, fill=1, stroke=1)
    lines = text.split("\n")
    for i, line in enumerate(lines):
        draw_text(c, x + w / 2, y + h / 2 + (len(lines) - 1) * 4 - i * 8, line, size=size, anchor="middle")


def draw_arrow(c, x1, y1, x2, y2):
    c.setStrokeColor(HexColor("#555555"))
    c.setLineWidth(0.65)
    c.line(x1, y1, x2, y2)
    angle = 0 if x2 >= x1 else 3.14159
    c.setFillColor(HexColor("#555555"))
    if abs(x2 - x1) >= abs(y2 - y1):
        c.line(x2, y2, x2 - 4, y2 + 2.5)
        c.line(x2, y2, x2 - 4, y2 - 2.5)
    else:
        c.line(x2, y2, x2 - 2.5, y2 - 4)
        c.line(x2, y2, x2 + 2.5, y2 - 4)


def blend(c1, c2, t):
    return colors.Color(
        c1.red + (c2.red - c1.red) * t,
        c1.green + (c2.green - c1.green) * t,
        c1.blue + (c2.blue - c1.blue) * t,
    )


def heat_color(value, lo=0.50, hi=0.80):
    t = max(0, min(1, (value - lo) / (hi - lo)))
    return blend(HexColor("#F3F6FB"), HexColor("#0072B2"), t)


def fig_main_composite():
    W, H = 7.05 * inch, 5.55 * inch
    c = canvas.Canvas(str(FIG_DIR / "fig_summary_main_composite.pdf"), pagesize=(W, H))
    margin_x, top = 0.36 * inch, H - 0.28 * inch
    panel_w, panel_h = 3.02 * inch, 2.18 * inch
    gap_x, gap_y = 0.37 * inch, 0.34 * inch
    xL, xR = margin_x, margin_x + panel_w + gap_x
    yT = top - panel_h
    yB = yT - gap_y - panel_h

    # Panel a: compact evidence flow.
    draw_panel_label(c, xL, yT + panel_h, "a", "Compact narrative as the interface")
    box_y = yT + panel_h - 0.66 * inch
    boxes = [
        (xL + 0.04 * inch, box_y, 0.64 * inch, 0.36 * inch, "43.6M\nICU events", HexColor("#E8EDF7")),
        (xL + 0.89 * inch, box_y, 0.76 * inch, 0.36 * inch, "ClinTime\nSummary", HexColor("#DCEFE8")),
        (xL + 1.87 * inch, box_y, 0.68 * inch, 0.36 * inch, "Llama-3\nQLoRA", HexColor("#FCE4D6")),
        (xL + 2.75 * inch, box_y, 0.50 * inch, 0.36 * inch, "Risk\nscores", HexColor("#EFE5F4")),
    ]
    for i, (x, y, w, h, text, fill) in enumerate(boxes):
        draw_box(c, x, y, w, h, text, fill)
        if i:
            prev = boxes[i - 1]
            draw_arrow(c, prev[0] + prev[2] + 4, y + h / 2, x - 4, y + h / 2)
    chips = [
        ("trend", "#DCEFE8"), ("abnormality", "#FCE4D6"), ("intervention", "#EFE5F4"),
        ("source IDs", "#E8EDF7"), ("concept priority", "#FFF2CC"), ("short text", "#E6E6E6"),
    ]
    chip_x, chip_y = xL + 0.10 * inch, yT + 0.80 * inch
    for i, (label, fill) in enumerate(chips):
        xx = chip_x + (i % 3) * 0.94 * inch
        yy = chip_y - (i // 3) * 0.29 * inch
        draw_box(c, xx, yy, 0.80 * inch, 0.18 * inch, label, HexColor(fill), size=6.7)
    draw_text(c, xL + 0.12 * inch, yT + 0.26 * inch, "Claim: compact clinical narration is learnable,", size=7.4)
    draw_text(c, xL + 0.12 * inch, yT + 0.13 * inch, "reusable across ICU tasks, and bounded by domain shift.", size=7.4)

    # Panel b: mortality scaling small multiples.
    draw_panel_label(c, xR, yT + panel_h, "b", "Mortality scaling and seed stability")
    rows = DATA["mortality_scaling"]
    centers = ["MIMIC-IV", "eICU", "NWICU"]
    settings = ["5k", "10k", "20k"]
    y_min, y_max = 0.44, 0.84
    mini_w, mini_h = 0.83 * inch, 1.10 * inch
    for idx, center in enumerate(centers):
        x0 = xR + 0.18 * inch + idx * 0.92 * inch
        y0 = yT + 0.52 * inch
        xs = [x0 + mini_w * p for p in (0.12, 0.50, 0.88)]
        draw_axes(c, x0, y0, mini_w, mini_h, y_min, y_max, y_label="AUROC" if idx == 0 else None,
                  x_ticks=xs, x_labels=settings)
        draw_text(c, x0 + 3, y0 + mini_h - 8, center, size=6.9, font="Times-Bold")
        values_by_setting = defaultdict(list)
        for row in rows:
            if row["center"] == center:
                values_by_setting[row["setting"]].append(row["auroc"])
        means = [mean(values_by_setting[s]) for s in settings]
        pts = [(x, ymap(v, y0, mini_h, y_min, y_max)) for x, v in zip(xs, means)]
        c.setStrokeColor(COLORS[center])
        c.setLineWidth(1.0)
        for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
            c.line(x1, y1, x2, y2)
        for i, setting in enumerate(settings):
            vals = values_by_setting[setting]
            offsets = [-2.5, 0, 2.5] if len(vals) == 3 else ([0] if len(vals) == 1 else [-2, 2])
            for off, val in zip(offsets, vals):
                c.setFillColor(colors.white)
                c.setStrokeColor(COLORS[center])
                c.circle(xs[i] + off, ymap(val, y0, mini_h, y_min, y_max), 1.9, fill=1, stroke=1)
        c.setStrokeColor(HexColor("#555555"))
        c.setDash(2, 2)
        c.line(x0, ymap(0.5, y0, mini_h, y_min, y_max), x0 + mini_w, ymap(0.5, y0, mini_h, y_min, y_max))
        c.setDash()
    draw_text(c, xR + panel_w / 2, yT + 0.22 * inch, "MIMIC train records", size=7.2, anchor="middle")

    # Panel c: multitask heat map with 5k to 10k deltas.
    draw_panel_label(c, xL, yB + panel_h, "c", "Multitask reuse of the same format")
    mt = DATA["multitask"]
    tasks = ["LOS >48h", "Future intervention"]
    centers = ["MIMIC-IV", "eICU", "NWICU"]
    cell_w, cell_h = 0.68 * inch, 0.34 * inch
    start_x, start_y = xL + 0.92 * inch, yB + 1.29 * inch
    for ci, center in enumerate(centers):
        draw_text(c, start_x + ci * cell_w + cell_w / 2, start_y + 0.25 * inch, center, size=7.1, font="Times-Bold", anchor="middle")
    for ti, task in enumerate(tasks):
        y = start_y - ti * 0.58 * inch
        draw_text(c, xL + 0.10 * inch, y + 0.12 * inch, task, size=7.2, font="Times-Bold")
        for ci, center in enumerate(centers):
            vals = {}
            for train in ["5k", "10k"]:
                vals[train] = [r for r in mt if r["task"] == task and r["center"] == center and r["train_records"] == train][0]["auroc"]
            x = start_x + ci * cell_w
            c.setFillColor(heat_color(vals["10k"]))
            c.setStrokeColor(colors.white)
            c.rect(x, y, cell_w - 2, cell_h, fill=1, stroke=1)
            txt = f"{vals['5k']:.2f}->{vals['10k']:.2f}"
            draw_text(c, x + cell_w / 2 - 1, y + 0.12 * inch, txt, size=6.6, anchor="middle",
                      color=colors.white if vals["10k"] > 0.68 else colors.black)
    draw_text(c, xL + 0.10 * inch, yB + 0.33 * inch, "Cells show AUROC change from 5k to 10k training records.", size=7.2)
    for i, val in enumerate([0.50, 0.65, 0.80]):
        xx = xL + 1.65 * inch + i * 0.35 * inch
        c.setFillColor(heat_color(val))
        c.rect(xx, yB + 0.13 * inch, 0.28 * inch, 0.10 * inch, fill=1, stroke=0)
        draw_text(c, xx + 0.14 * inch, yB + 0.03 * inch, f"{val:.2f}", size=6.2, anchor="middle")

    # Panel d: positioning bars.
    draw_panel_label(c, xR, yB + panel_h, "d", "Where Summary sits relative to baselines")
    pos = DATA["diagnostic_positioning"]
    methods = ["Short-context XML", "ClinTime-Summary", "HistGBDT reference"]
    x0, y0, w, h = xR + 0.33 * inch, yB + 0.50 * inch, 2.38 * inch, 1.20 * inch
    y_min, y_max = 0.45, 0.89
    groups = [x0 + w * p for p in (0.18, 0.50, 0.82)]
    draw_axes(c, x0, y0, w, h, y_min, y_max, y_label="AUROC", x_ticks=groups, x_labels=centers)
    bw = 9
    for gi, center in enumerate(centers):
        for mi, method in enumerate(methods):
            val = [r for r in pos if r["method"] == method and r["center"] == center][0]["auroc"]
            x = groups[gi] + (mi - 1) * (bw + 2)
            y = ymap(val, y0, h, y_min, y_max)
            c.setFillColor(COLORS[method])
            c.setStrokeColor(colors.black)
            c.setLineWidth(0.25)
            c.rect(x - bw / 2, y0, bw, y - y0, fill=1, stroke=1)
    draw_legend(c, [(m, COLORS[m]) for m in methods], xR + 0.22 * inch, yB + 0.23 * inch, sw=6, gap=68)

    c.showPage()
    c.save()
    print("Saved", FIG_DIR / "fig_summary_main_composite.pdf")


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
    fig_main_composite()
    fig_mortality_scaling()
    fig_multitask()
    fig_diagnostic_positioning()
