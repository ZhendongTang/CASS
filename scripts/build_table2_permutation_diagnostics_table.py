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


def mean_std(values: list[float]) -> tuple[float, float]:
    if len(values) != 3:
        raise ValueError(f"Expected exactly three seeds, found {len(values)}")
    return statistics.mean(values), statistics.stdev(values)


def collect(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for key, label in FORMATS:
        paths = sorted((root / "permutation" / "llama").glob(f"{key}_seed*/mimic/metrics.json"))
        if len(paths) != 3:
            raise FileNotFoundError(f"{key}: expected 3 metrics files, found {len(paths)}")
        values = {metric: [] for metric, _ in METRICS}
        seeds = []
        for path in paths:
            payload = json.loads(path.read_text())
            seeds.append(int(payload["seed"]))
            permutation = payload["permutation"]
            for metric, _ in METRICS:
                values[metric].append(float(permutation[metric]))
        row: dict[str, object] = {"format": key, "label": label, "seeds": seeds}
        for metric, _ in METRICS:
            mean, std = mean_std(values[metric])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
        rows.append(row)
    return rows


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["format", "label", "seeds"]
    for metric, _ in METRICS:
        fields.extend([f"{metric}_mean", f"{metric}_std"])
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def write_tex(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{table}[t]",
        r"\caption{Permutation-sensitivity diagnostics for short-context clinical serializers. Values are mean$\pm$standard deviation across three seeds and 500 MIMIC-IV audit records using Llama-3-8B. Lower values indicate less sensitivity to legal within-time reorderings. CASS and MEDS are deterministic after concept aggregation or canonical event sorting, respectively; JSON and XML are evaluated through string-level legal reorder views.}",
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
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--tex", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    args = parser.parse_args()
    rows = collect(args.root)
    write_tex(rows, args.tex)
    write_csv(rows, args.csv)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
