import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = json.loads((ROOT / "summary_assets_20260630.json").read_text(encoding="utf-8"))

COLORS = {
    "blue": "#2C7FB8",
    "orange": "#F28E2B",
    "green": "#59A14F",
    "purple": "#7B61A8",
    "gray": "#6B7280",
    "light": "#F7F8FA",
    "line": "#D1D5DB",
    "text": "#1F2937",
    "muted": "#4B5563",
}


def fmt(x):
    return f"{x:.3f}".lstrip("0")


def svg_open(w=1200, h=520):
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        "<defs>",
        '<style><![CDATA[text{font-family:Arial,Helvetica,sans-serif;fill:#1F2937}.small{font-size:18px}.tiny{font-size:15px;fill:#4B5563}.panel{font-size:20px;font-weight:700}.value{font-size:17px;font-weight:700}.axis{stroke:#D1D5DB;stroke-width:1}.bartext{font-size:15px;fill:#1F2937}]]></style>',
        "</defs>",
        f'<rect x="0" y="0" width="{w}" height="{h}" fill="white"/>',
    ]


def save(name, parts):
    (ROOT / name).write_text("\n".join(parts + ["</svg>\n"]), encoding="utf-8")


def text(x, y, s, cls="small", anchor="start"):
    return f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}">{s}</text>'


def line(x1, y1, x2, y2, color=None, width=1):
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color or COLORS["line"]}" stroke-width="{width}"/>'


def rect(x, y, w, h, fill, stroke="none", rx=6):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}"/>'


def bar(ax, x, y0, width, value, scale, color, label=None):
    h = value * scale
    out = [rect(x, y0 - h, width, h, color, rx=4), text(x + width / 2, y0 - h - 8, fmt(value), "bartext", "middle")]
    if label:
        out.append(text(x + width / 2, y0 + 24, label, "tiny", "middle"))
    return out


def fig1():
    parts = svg_open()
    parts += [text(35, 45, "a", "panel"), text(75, 45, "Event stream to CASS", "panel")]
    x0, y0 = 55, 90
    for i, lab in enumerate(["labs", "vitals", "infusions", "procedures"]):
        parts.append(rect(x0, y0 + i * 48, 165, 30, COLORS["light"], COLORS["line"], 5))
        parts.append(text(x0 + 82, y0 + i * 21 + 20, lab, "tiny", "middle"))
    parts += [line(245, 160, 330, 160, COLORS["gray"], 2), text(285, 142, "aggregate", "tiny", "middle")]
    parts.append(rect(350, 93, 270, 160, "#EEF6FB", COLORS["blue"], 8))
    for i, lab in enumerate(["count/range", "first/last", "trend", "abnormality", "source"]):
        parts.append(text(385, 125 + i * 25, lab, "tiny"))
    parts += [line(640, 160, 720, 160, COLORS["gray"], 2), text(680, 142, "serialize", "tiny", "middle")]
    parts.append(rect(740, 93, 380, 160, "#F4F1FA", COLORS["purple"], 8))
    parts.append(text(765, 128, "MAP: 12 obs, nadir 54, falling, hypotensive", "tiny"))
    parts.append(text(765, 158, "HR: 18 obs, 82-124, rising, last 121", "tiny"))
    parts.append(text(765, 188, "Ventilation: invasive support from hour 2", "tiny"))
    parts.append(text(765, 218, "Vasopressor: norepinephrine started hour 4", "tiny"))

    parts += [text(35, 325, "b", "panel"), text(75, 325, "Source-center AUROC", "panel")]
    base_y, scale = 470, 250
    vals = [("Mortality", 0.776, COLORS["blue"]), ("LOS", 0.619, COLORS["orange"]), ("Future", 0.806, COLORS["green"])]
    parts += [line(80, base_y, 505, base_y), line(80, base_y - 200, 80, base_y)]
    for i, (lab, val, col) in enumerate(vals):
        parts += bar(parts, 135 + i * 125, base_y, 60, val, scale, col, lab)
    parts.append(text(83, base_y - 205, "1.0", "tiny"))
    parts.append(text(83, base_y - 105, "0.5", "tiny"))

    parts += [text(615, 325, "c", "panel"), text(655, 325, "Claim boundary", "panel")]
    rows = [("compact", "681 mean tokens, 0.0% truncation", COLORS["green"]), ("effective", "20k mortality AUROC .761-.787", COLORS["blue"]), ("reusable", "mortality, LOS, future intervention", COLORS["purple"]), ("bounded", "external transfer remains mixed", COLORS["orange"])]
    for i, (k, v, col) in enumerate(rows):
        y = 355 + i * 38
        parts.append(rect(655, y, 110, 25, col, rx=12))
        parts.append(text(710, y + 18, k, "tiny", "middle"))
        parts.append(text(790, y + 18, v, "tiny"))
    save("fig1_main_evidence.svg", parts)


def fig2():
    parts = svg_open()
    parts += [text(35, 45, "a", "panel"), text(75, 45, "Equal-budget MIMIC-IV AUROC", "panel")]
    fmt_vals = {"Flat": 0.5383657199800698, "XML": 0.4831008138183026, "Compact": 0.53284338149809, "CASS": 0.695}
    cols = {"Flat": COLORS["gray"], "XML": COLORS["orange"], "Compact": COLORS["blue"], "CASS": COLORS["green"]}
    base_y, scale = 260, 250
    parts += [line(70, base_y, 535, base_y), line(70, 70, 70, base_y)]
    for i, (lab, val) in enumerate(fmt_vals.items()):
        parts += bar(parts, 120 + i * 100, base_y, 54, val, scale, cols[lab], lab)
    parts += [text(35, 335, "b", "panel"), text(75, 335, "Token footprint and truncation", "panel")]
    tok = {"Flat": (2107, "51.5%"), "XML": (2324, "65.4%"), "Compact": (1394, "13.7%"), "CASS": (681, "0.0%")}
    base_y2, max_tok = 470, 2500
    parts += [line(70, base_y2, 535, base_y2), line(70, 365, 70, base_y2)]
    for i, (lab, (val, tr)) in enumerate(tok.items()):
        h = val / max_tok * 95
        x = 120 + i * 100
        parts.append(rect(x, base_y2 - h, 54, h, cols[lab], rx=4))
        parts.append(text(x + 27, base_y2 - h - 8, str(val), "bartext", "middle"))
        parts.append(text(x + 27, base_y2 + 23, tr, "tiny", "middle"))
    parts.append(text(25, 392, "tokens", "tiny"))
    parts.append(text(70, 508, "trunc.", "tiny"))

    parts += [text(635, 45, "c", "panel"), text(675, 45, "Efficiency frontier", "panel")]
    x0, y0, w, h = 675, 85, 420, 185
    parts += [line(x0, y0 + h, x0 + w, y0 + h), line(x0, y0, x0, y0 + h)]
    for lab, au in fmt_vals.items():
        mt = tok[lab][0]
        x = x0 + (mt - 600) / 1800 * w
        y = y0 + h - (au - 0.45) / 0.28 * h
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="8" fill="{cols[lab]}"/>')
        parts.append(text(x + 12, y + 5, lab, "tiny"))
    parts.append(text(x0 + 180, y0 + h + 35, "mean tokens", "tiny", "middle"))
    parts.append(text(x0 - 10, y0 - 10, "AUROC", "tiny", "end"))

    parts += [text(635, 335, "d", "panel"), text(675, 335, "Retained context before 2048 tokens", "panel")]
    ret = {"Flat": 0.485, "XML": 0.346, "Compact": 0.863, "CASS": 1.0}
    for i, (lab, val) in enumerate(ret.items()):
        y = 370 + i * 30
        parts.append(text(675, y + 16, lab, "tiny"))
        parts.append(rect(780, y, 255, 18, "#EEF2F7", rx=4))
        parts.append(rect(780, y, 255 * val, 18, cols[lab], rx=4))
        parts.append(text(1050, y + 15, f"{val*100:.1f}%", "tiny"))
    save("fig2_format_token_efficiency.svg", parts)


def fig3():
    parts = svg_open()
    parts += [text(35, 45, "a", "panel"), text(75, 45, "Source-to-external AUROC gaps", "panel")]
    rows = [
        ("Mortality 20k", 0.776, 0.580, 0.625),
        ("LOS 10k", 0.619, 0.576, 0.511),
        ("Future 10k", 0.806, 0.468, 0.519),
    ]
    x0, y0 = 150, 95
    for i, (lab, m, e, n) in enumerate(rows):
        y = y0 + i * 70
        parts.append(text(55, y + 5, lab, "tiny"))
        for j, (center, val, col) in enumerate([("MIMIC", m, COLORS["blue"]), ("eICU", e, COLORS["orange"]), ("NWICU", n, COLORS["green"])]):
            x = x0 + j * 150
            parts.append(rect(x, y - val * 80, 55, val * 80, col, rx=4))
            parts.append(text(x + 27, y - val * 80 - 7, fmt(val), "tiny", "middle"))
        parts.append(line(x0 - 15, y, x0 + 370, y))
    parts.append(text(295, 315, "center", "tiny", "middle"))
    for j, lab in enumerate(["MIMIC", "eICU", "NWICU"]):
        parts.append(text(x0 + j * 150 + 27, 320, lab, "tiny", "middle"))

    parts += [text(635, 45, "b", "panel"), text(675, 45, "Future-intervention seed spread", "panel")]
    seeds = [(42, 0.800, 0.522, 0.517), (43, 0.803, 0.524, 0.540), (44, 0.815, 0.357, 0.500)]
    bx, by, bw, bh = 700, 90, 360, 180
    parts += [line(bx, by + bh, bx + bw, by + bh), line(bx, by, bx, by + bh)]
    for center_i, (name, col) in enumerate([("MIMIC", COLORS["blue"]), ("eICU", COLORS["orange"]), ("NWICU", COLORS["green"])]):
        pts = []
        for k, seedrow in enumerate(seeds):
            x = bx + 60 + k * 110
            val = seedrow[center_i + 1]
            y = by + bh - (val - 0.3) / 0.55 * bh
            pts.append((x, y))
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{col}"/>')
        path = " ".join([f"{x:.1f},{y:.1f}" for x, y in pts])
        parts.append(f'<polyline points="{path}" fill="none" stroke="{col}" stroke-width="2"/>')
        parts.append(text(bx + bw + 20, by + 35 + center_i * 28, name, "tiny"))
    for k, seedrow in enumerate(seeds):
        parts.append(text(bx + 60 + k * 110, by + bh + 25, f"s{seedrow[0]}", "tiny", "middle"))

    parts += [text(635, 335, "c", "panel"), text(675, 335, "Supported and unsupported claims", "panel")]
    claims = [("supported", "compact source-center LLM input", COLORS["green"]), ("supported", "reuse across three ICU endpoints", COLORS["blue"]), ("not supported", "serialization alone solves hospital shift", COLORS["orange"])]
    for i, (tag, desc, col) in enumerate(claims):
        y = 365 + i * 42
        parts.append(rect(675, y, 120, 26, col, rx=13))
        parts.append(text(735, y + 18, tag, "tiny", "middle"))
        parts.append(text(815, y + 18, desc, "tiny"))
    save("fig3_generalization_boundary.svg", parts)


fig1()
fig2()
fig3()
print("wrote svg figures")
