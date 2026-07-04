from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any


COLORS = {
    "cass": "#2C7FB8",
    "summary": "#2C7FB8",
    "compact": "#59A14F",
    "json": "#7B61A8",
    "meds": "#8A8F98",
    "xml": "#F28E2B",
    "flat": "#6B7280",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _variant_alias(name: str) -> str:
    name = name.lower()
    return "summary" if name == "cass" else name


def _metric_at(payload: dict[str, Any], key: str) -> float:
    section: Any = payload
    for part in key.split("."):
        section = section[part]
    return float(section)


def collect_token_rows(token_root: Path, formats: list[str], budget: int) -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[str, list[float]]] = {}
    for path in sorted(token_root.glob("**/*.json")):
        payload = _read_json(path)
        variants = payload.get("variants", {})
        for fmt in formats:
            key = _variant_alias(fmt)
            if key not in variants:
                continue
            row = variants[key]
            grouped.setdefault(fmt, {"tokens": [], "truncation": [], "retention": []})
            grouped[fmt]["tokens"].append(float(row["token_length"]["mean"]))
            budget_row = row.get("budgets", {}).get(str(budget), {})
            if "truncation_rate" in budget_row:
                grouped[fmt]["truncation"].append(float(budget_row["truncation_rate"]))
            if budget_row.get("mean_evidence_retention") is not None:
                grouped[fmt]["retention"].append(float(budget_row["mean_evidence_retention"]))
    return {
        fmt: {
            "mean_tokens": mean(values["tokens"]),
            "truncation_rate": mean(values["truncation"]) if values["truncation"] else 0.0,
            "evidence_retention": mean(values["retention"]) if values["retention"] else 1.0,
        }
        for fmt, values in grouped.items()
    }


def collect_performance_rows(metrics_root: Path, formats: list[str], center: str, metric: str) -> dict[str, float]:
    rows: dict[str, list[float]] = {}
    for fmt in formats:
        key = _variant_alias(fmt)
        patterns = [
            f"**/{key}_seed*/evaluation/{center}/metrics.json",
            f"**/{key}_seed*/{center}/metrics.json",
        ]
        for pattern in patterns:
            for path in metrics_root.glob(pattern):
                payload = _read_json(path)
                if metric in payload:
                    rows.setdefault(fmt, []).append(float(payload[metric]))
    return {fmt: mean(values) for fmt, values in rows.items() if values}


def write_csv(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["format", "mean_tokens", "truncation_rate", "evidence_retention", "metric"])
        writer.writeheader()
        writer.writerows(rows)


def _text(x: float, y: float, value: str, size: int = 13, anchor: str = "start") -> str:
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" font-family="Arial, Helvetica, sans-serif">{value}</text>'


def _rect(x: float, y: float, w: float, h: float, fill: str) -> str:
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="4" fill="{fill}"/>'


def _line(x1: float, y1: float, x2: float, y2: float, color: str = "#D1D5DB") -> str:
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="1"/>'


def write_svg(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1100, 520
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        _text(40, 42, "Fig. 2. Format efficiency from computed token and metric audits", 18),
    ]
    max_tokens = max(float(row["mean_tokens"]) for row in rows) if rows else 1.0
    metric_values = [float(row["metric"]) for row in rows if row.get("metric") is not None]
    max_metric = max(metric_values) if metric_values else 1.0

    x0, y0 = 80, 245
    parts += [_text(40, 78, "a", 16), _text(70, 78, "Mean token footprint", 15), _line(x0, y0, 500, y0), _line(x0, 95, x0, y0)]
    for index, row in enumerate(rows):
        x = x0 + 50 + index * 90
        h = float(row["mean_tokens"]) / max_tokens * 130
        color = COLORS.get(str(row["format"]).lower(), "#6B7280")
        parts.append(_rect(x, y0 - h, 46, h, color))
        parts.append(_text(x + 23, y0 + 22, str(row["format"]).upper(), 11, "middle"))
        parts.append(_text(x + 23, y0 - h - 8, f'{float(row["mean_tokens"]):.0f}', 11, "middle"))

    x1, y1 = 615, 245
    parts += [_text(575, 78, "b", 16), _text(605, 78, "Predictive metric at matched budget", 15), _line(x1, y1, 1035, y1), _line(x1, 95, x1, y1)]
    for index, row in enumerate(rows):
        x = x1 + 50 + index * 90
        h = float(row.get("metric") or 0.0) / max_metric * 130
        color = COLORS.get(str(row["format"]).lower(), "#6B7280")
        parts.append(_rect(x, y1 - h, 46, h, color))
        parts.append(_text(x + 23, y1 + 22, str(row["format"]).upper(), 11, "middle"))
        parts.append(_text(x + 23, y1 - h - 8, f'{float(row.get("metric") or 0.0):.3f}'.lstrip("0"), 11, "middle"))

    x2, y2 = 80, 440
    parts += [_text(40, 310, "c", 16), _text(70, 310, "Evidence retained under token budget", 15)]
    for index, row in enumerate(rows):
        y = y2 - 105 + index * 28
        value = float(row["evidence_retention"])
        color = COLORS.get(str(row["format"]).lower(), "#6B7280")
        parts.append(_text(x2, y + 15, str(row["format"]).upper(), 11))
        parts.append(_rect(x2 + 100, y, 260, 18, "#EEF2F7"))
        parts.append(_rect(x2 + 100, y, 260 * value, 18, color))
        parts.append(_text(x2 + 375, y + 15, f"{100 * value:.1f}%", 11))

    x3, y3 = 615, 440
    parts += [_text(575, 310, "d", 16), _text(605, 310, "Metric per 1k tokens", 15)]
    max_eff = max(
        1e-12,
        max((float(row.get("metric") or 0.0) / max(float(row["mean_tokens"]), 1.0) * 1000 for row in rows), default=0.0),
    )
    for index, row in enumerate(rows):
        y = y3 - 105 + index * 28
        eff = float(row.get("metric") or 0.0) / max(float(row["mean_tokens"]), 1.0) * 1000
        color = COLORS.get(str(row["format"]).lower(), "#6B7280")
        parts.append(_text(x3, y + 15, str(row["format"]).upper(), 11))
        parts.append(_rect(x3 + 100, y, 260, 18, "#EEF2F7"))
        parts.append(_rect(x3 + 100, y, 260 * eff / max_eff, 18, color))
        parts.append(_text(x3 + 375, y + 15, f"{eff:.3f}", 11))

    parts.append("</svg>")
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw Fig. 2 from token audit JSON and evaluation metrics.")
    parser.add_argument("--token-root", type=Path, required=True)
    parser.add_argument("--metrics-root", type=Path, required=True)
    parser.add_argument("--formats", nargs="+", default=["cass", "compact", "json", "meds", "xml", "flat"])
    parser.add_argument("--center", default="mimic")
    parser.add_argument("--metric", default="auroc")
    parser.add_argument("--budget", type=int, default=2048)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--svg", type=Path, required=True)
    args = parser.parse_args()

    token_rows = collect_token_rows(args.token_root, args.formats, args.budget)
    metrics = collect_performance_rows(args.metrics_root, args.formats, args.center, args.metric)
    rows = []
    for fmt in args.formats:
        if fmt not in token_rows:
            continue
        rows.append({"format": fmt, **token_rows[fmt], "metric": metrics.get(fmt)})
    write_csv(rows, args.csv)
    write_svg(rows, args.svg)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
