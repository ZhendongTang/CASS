from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any


RUN_RE = re.compile(r"^(?P<variant>[A-Za-z0-9_]+)_seed(?P<seed>\d+)$")
METRICS = ("auroc", "auprc", "brier", "ece")


def _compact(value: float) -> str:
    text = f"{value:.3f}"
    if text.startswith("0"):
        text = text[1:]
    if text.startswith("-0"):
        text = "-" + text[2:]
    return text


def _summary(values: list[float]) -> tuple[float, float]:
    if len(values) == 1:
        return values[0], 0.0
    return mean(values), stdev(values)


def _cell(mean_value: float, sd_value: float) -> str:
    return f"{_compact(mean_value)}$\\pm${_compact(sd_value)}"


def collect(root: Path) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    seeds: dict[tuple[str, str, str], set[int]] = defaultdict(set)
    for path in sorted(root.glob("main/*/*_seed*/evaluation/*/metrics.json")):
        backbone, run_name, center = path.parts[-5], path.parts[-4], path.parts[-2]
        match = RUN_RE.match(run_name)
        if not match:
            continue
        variant = match.group("variant")
        seed = int(match.group("seed"))
        payload = json.loads(path.read_text(encoding="utf-8"))
        key = (backbone, variant, center)
        seeds[key].add(seed)
        for metric in METRICS:
            if metric in payload:
                grouped[key][metric].append(float(payload[metric]))
    rows: list[dict[str, Any]] = []
    for key, by_metric in sorted(grouped.items()):
        backbone, variant, center = key
        row: dict[str, Any] = {
            "backbone": backbone,
            "variant": variant,
            "center": center,
            "seeds": ",".join(str(seed) for seed in sorted(seeds[key])),
            "n_seeds": len(seeds[key]),
        }
        for metric in METRICS:
            values = by_metric.get(metric, [])
            if values:
                mean_value, sd_value = _summary(values)
                row[f"{metric}_mean"] = mean_value
                row[f"{metric}_sd"] = sd_value
        rows.append(row)
    return rows


def write_csv(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| Backbone | Variant | Center | Seeds | AUROC | AUPRC | Brier | ECE |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        values = [
            str(row["backbone"]),
            str(row["variant"]),
            str(row["center"]),
            str(row["seeds"]),
        ]
        for metric in METRICS:
            if f"{metric}_mean" in row:
                values.append(f"{_compact(float(row[f'{metric}_mean']))}+/-{_compact(float(row[f'{metric}_sd']))}")
            else:
                values.append("--")
        lines.append("| " + " | ".join(values) + " |")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_tex(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{table*}[t]",
        r"\caption{Endpoint prediction performance across model backbones, serializations, seeds, and evaluation centers. Values are mean$\pm$standard deviation across completed seeds.}",
        r"\label{tab:endpoint_summary}",
        r"\centering",
        r"\scriptsize",
        r"\begin{tabular}{lllcccc}",
        r"\toprule",
        r"Backbone & Variant & Center & AUROC & AUPRC & Brier & ECE \\",
        r"\midrule",
    ]
    for row in rows:
        cells = [str(row["backbone"]), str(row["variant"]), str(row["center"])]
        for metric in METRICS:
            if f"{metric}_mean" in row:
                cells.append(_cell(float(row[f"{metric}_mean"]), float(row[f"{metric}_sd"])))
            else:
                cells.append("--")
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Table 3 endpoint summary from evaluation metrics.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--tex", type=Path, required=True)
    args = parser.parse_args()
    rows = collect(args.root)
    write_csv(rows, args.csv)
    write_markdown(rows, args.markdown)
    write_tex(rows, args.tex)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
