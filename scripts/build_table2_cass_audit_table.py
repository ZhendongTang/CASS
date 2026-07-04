from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev


VARIANT_LABELS = {
    "xml": "XML",
    "meds": "MEDS-like",
    "json": "JSON",
    "summary": r"\textbf{CASS}",
}


def fmt_int_pm(values: list[float]) -> str:
    return f"{mean(values):.0f}$\\pm${stdev(values):.0f}"


def fmt_pct_pm(values: list[float]) -> str:
    scaled = [100 * value for value in values]
    return f"{mean(scaled):.1f}$\\pm${stdev(scaled):.1f}"


def fmt_float_pm(values: list[float]) -> str:
    return f"{mean(values):.2f}$\\pm${stdev(values):.2f}"


def collect_rows(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(root.glob("*/*/*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        llm = path.relative_to(root).parts[0]
        for variant, metrics in data["variants"].items():
            rows.append(
                {
                    "llm": llm,
                    "task": data["task"],
                    "center": data["dataset"],
                    "variant": variant,
                    "mean_tokens": metrics["token_length"]["mean"],
                    "p95_tokens": metrics["token_length"]["p95"],
                    "truncation_rate": metrics["truncation_rate_at_budget"],
                    "exact_match": metrics["exact_match_rate_under_legal_reorder"],
                    "unique_hashes": metrics["mean_unique_hashes_per_record"],
                }
            )
    return rows


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_latex(rows: list[dict[str, object]], path: Path) -> None:
    groups = {variant: [row for row in rows if row["variant"] == variant] for variant in VARIANT_LABELS}
    lines = [
        r"\begin{table}[t]",
        r"\caption{Serialization stability and token-budget audit. Each cell aggregates 27 tokenizer audits (three LLM backbones, three prediction tasks, and three centers), with 1000 held-out records and five legal same-time reorderings per audit. Lower is better for token count, truncation, and unique hashes; higher is better for exact match.}",
        r"\label{tab:cass_audit}",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.0pt}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Rep. & Mean tok. & p95 tok. & Trunc. & Exact & Hashes \\",
        r"\midrule",
    ]
    ordered_variants = sorted(
        VARIANT_LABELS,
        key=lambda variant: mean(float(item["mean_tokens"]) for item in groups[variant]),
    )
    for variant in ordered_variants:
        label = VARIANT_LABELS[variant]
        group = groups[variant]
        row = [
            label,
            fmt_int_pm([float(item["mean_tokens"]) for item in group]),
            fmt_int_pm([float(item["p95_tokens"]) for item in group]),
            fmt_pct_pm([float(item["truncation_rate"]) for item in group]),
            fmt_pct_pm([float(item["exact_match"]) for item in group]),
            fmt_float_pm([float(item["unique_hashes"]) for item in group]),
        ]
        if variant == "summary":
            row[1] = r"\textbf{" + row[1] + "}"
            row[2] = r"\textbf{" + row[2] + "}"
            row[3] = r"\textbf{" + row[3] + "}"
            row[4] = r"\textbf{" + row[4] + "}"
            row[5] = r"\textbf{" + row[5] + "}"
        lines.append(" & ".join(row) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\end{table}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--tex", required=True, type=Path)
    args = parser.parse_args()

    rows = collect_rows(args.root)
    if len(rows) != 162:
        raise RuntimeError(f"Expected 162 rows from 27 audits x 6 variants, found {len(rows)}")
    write_csv(rows, args.csv)
    write_latex(rows, args.tex)


if __name__ == "__main__":
    main()
