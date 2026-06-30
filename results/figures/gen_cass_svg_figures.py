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
        '<marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="#6B7280"/></marker>',
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
    parts = svg_open(1200, 560)

    # Left: heterogeneous, partially ordered ICU events.
    parts.append(text(58, 58, "a", "panel"))
    parts.append(text(92, 58, "Partially ordered ICU event stream", "panel"))
    parts.append(rect(58, 88, 285, 255, "#F7F8FA", COLORS["line"], 8))
    parts.append(line(95, 132, 95, 306, COLORS["gray"], 2))
    for y, hour in [(132, "0h"), (190, "2h"), (248, "4h"), (306, "6h")]:
        parts.append(f'<circle cx="95" cy="{y}" r="5" fill="{COLORS["gray"]}"/>')
        parts.append(text(70, y + 5, hour, "tiny", "end"))
    events = [
        (124, 112, "lab", "lactate 4.0", COLORS["orange"]),
        (174, 147, "vital", "MAP 58", COLORS["blue"]),
        (126, 183, "vital", "HR 121", COLORS["blue"]),
        (188, 219, "device", "ventilation", COLORS["purple"]),
        (130, 255, "drug", "norepinephrine", COLORS["green"]),
        (196, 291, "procedure", "arterial line", COLORS["gray"]),
    ]
    for x, y, typ, val, col in events:
        parts.append(rect(x, y, 130, 28, "white", col, 5))
        parts.append(text(x + 10, y + 19, f"{typ}: {val}", "tiny"))
    parts.append(text(72, 372, "Same-time rows may have no clinical internal order.", "tiny"))
    parts.append(text(72, 396, "Fixed bins blur timing; imputation can add artificial values.", "tiny"))

    # Middle: CASS serialization.
    parts.append(f'<path d="M365 215 C405 215, 410 215, 450 215" fill="none" stroke="{COLORS["gray"]}" stroke-width="2.4" marker-end="url(#arrow)"/>')
    parts.append(text(400, 195, "aggregate", "tiny", "middle"))
    parts.append(text(465, 58, "b", "panel"))
    parts.append(text(498, 58, "CASS serialization", "panel"))
    parts.append(rect(455, 88, 315, 255, "#EEF6FB", COLORS["blue"], 8))
    features = [
        ("bind", "concept + value + unit + source"),
        ("compress", "count, range, first/last, trend"),
        ("prioritize", "intervention, abnormality, recent change"),
    ]
    for i, (k, v) in enumerate(features):
        y = 112 + i * 43
        parts.append(rect(485, y, 86, 25, "#D7ECF8", "none", 13))
        parts.append(text(528, y + 18, k, "tiny", "middle"))
        parts.append(text(585, y + 18, v, "tiny"))
    parts.append(rect(485, 245, 250, 68, "white", COLORS["line"], 6))
    parts.append(text(503, 267, "MAP: 12 obs, 54-86 mmHg, falling", "tiny"))
    parts.append(text(503, 289, "lactate: 2 obs, rising, abnormal high", "tiny"))
    parts.append(text(503, 311, "ventilation + vasopressor started", "tiny"))

    # Right: LLM and multitask predictions.
    parts.append(f'<path d="M790 215 C825 215, 835 215, 870 215" fill="none" stroke="{COLORS["gray"]}" stroke-width="2.4" marker-end="url(#arrow)"/>')
    parts.append(text(832, 195, "fine-tune", "tiny", "middle"))
    parts.append(text(880, 58, "c", "panel"))
    parts.append(text(914, 58, "Multitask ICU LLM", "panel"))
    parts.append(rect(875, 88, 270, 255, "#F4F1FA", COLORS["purple"], 8))
    parts.append(rect(922, 116, 176, 68, "white", COLORS["purple"], 8))
    parts.append(text(1010, 143, "Llama-3-8B", "small", "middle"))
    parts.append(text(1010, 166, "QLoRA risk head", "tiny", "middle"))
    tasks = [
        ("mortality", ".761-.787 AUROC", COLORS["blue"]),
        ("LOS >48h", ".619 AUROC", COLORS["orange"]),
        ("future intervention", ".800-.815 AUROC", COLORS["green"]),
    ]
    for i, (lab, val, col) in enumerate(tasks):
        y = 214 + i * 36
        parts.append(rect(910, y, 88, 25, col, "none", 13))
        parts.append(text(954, y + 18, lab, "tiny", "middle"))
        parts.append(text(1015, y + 18, val, "tiny"))

    # Bottom: evidence strip and boundary.
    parts.append(rect(58, 430, 1087, 78, "#FFFFFF", COLORS["line"], 8))
    badges = [
        ("compact", "681 mean tokens", "0.0% truncation", COLORS["green"]),
        ("effective", "20k mortality", ".761-.787 AUROC", COLORS["blue"]),
        ("reusable", "3 ICU endpoints", "same CASS format", COLORS["purple"]),
        ("bounded", "external transfer", "mixed across centers", COLORS["orange"]),
    ]
    for i, (head, mid, tail, col) in enumerate(badges):
        x = 90 + i * 260
        parts.append(rect(x, 452, 96, 28, col, "none", 14))
        parts.append(text(x + 48, 471, head, "tiny", "middle"))
        parts.append(text(x + 115, 462, mid, "tiny"))
        parts.append(text(x + 115, 487, tail, "tiny"))
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
