from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


FORMATS = [
    ("cass", "CASS"),
    ("json", "JSON"),
    ("meds", "MEDS"),
    ("xml", "XML"),
]

METRICS = [
    ("mean_prediction_std", "Prob. std."),
    ("mean_max_probability_gap", "Risk span"),
    ("label_flip_rate", "Flip rate"),
]


def compact(value: float) -> str:
    text = f"{value:.3f}"
    if text.startswith("0"):
        text = text[1:]
    if text.startswith("-0"):
        text = "-" + text[2:]
    return text


def collect(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for fmt, label in FORMATS:
        paths = sorted((root / "permutation" / "llama").glob(f"{fmt}_seed*/mimic/metrics.json"))
        if len(paths) != 3:
            raise FileNotFoundError(f"{fmt}: expected 3 metrics files, found {len(paths)}")
        metric_values = {metric: [] for metric, _ in METRICS}
        seeds: list[int] = []
        for path in paths:
            payload = json.loads(path.read_text(encoding="utf-8"))
            seeds.append(int(payload["seed"]))
            for metric, _ in METRICS:
                metric_values[metric].append(float(payload["permutation"][metric]))
        row: dict[str, object] = {"format": fmt, "label": label, "seeds": seeds}
        for metric, _ in METRICS:
            values = metric_values[metric]
            row[f"{metric}_mean"] = statistics.mean(values)
            row[f"{metric}_std"] = statistics.stdev(values)
        rows.append(row)
    return rows


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["format", "label", "seeds"]
    for metric, _ in METRICS:
        fields.extend([f"{metric}_mean", f"{metric}_std"])
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def write_markdown(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| Representation | Prob. std. | Risk span | Flip rate |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        cells = [str(row["label"])]
        for metric, _ in METRICS:
            mean = compact(float(row[f"{metric}_mean"]))
            std = compact(float(row[f"{metric}_std"]))
            cells.append(f"{mean}+/-{std}")
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def infer_n(root: Path) -> int:
    first = next((root / "permutation" / "llama").glob("*_seed*/mimic/metrics.json"))
    payload = json.loads(first.read_text(encoding="utf-8"))
    return int(payload["n"])


def write_tex(rows: list[dict[str, object]], path: Path, n_records: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{table}[t]",
        rf"\caption{{Surface-order sensitivity diagnostics for short-context clinical serializers. Values are mean$\pm$standard deviation across three seeds and {n_records} MIMIC-IV audit records using Llama-3-8B. Lower values indicate less prediction sensitivity when clinically equivalent serializer-internal units are reordered. CASS shuffles concept-summary rows after the global row; MEDS shuffles event rows sharing the same timestamp; JSON and XML shuffle unordered nodes within time blocks.}}",
        r"\label{tab:permutation_diagnostics}",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Representation & Prob. std. & Risk span & Flip rate \\",
        r"\midrule",
    ]
    for row in rows:
        cells = [str(row["label"])]
        for metric, _ in METRICS:
            mean = compact(float(row[f"{metric}_mean"]))
            std = compact(float(row[f"{metric}_std"]))
            cells.append(f"{mean}$\\pm${std}")
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--tex", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    rows = collect(args.root)
    n_records = infer_n(args.root)
    write_csv(rows, args.csv)
    write_markdown(rows, args.markdown)
    write_tex(rows, args.tex, n_records)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
