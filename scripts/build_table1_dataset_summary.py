from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pces_icu.io import read_jsonl


def _fmt_int(value: int | float) -> str:
    return f"{int(round(value)):,}"


def _fmt_float(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}".lstrip("0")


def _read_cohort(path: Path | None) -> list[dict[str, str]]:
    if path is None:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _summarize_events(path: Path) -> dict[str, Any]:
    records = list(read_jsonl(path))
    event_counts = [len(record.get("events", [])) for record in records]
    labels = [int(record.get("label", 0)) for record in records]
    splits = Counter(str(record.get("split", "")) or "unknown" for record in records)
    hospitals = {str(record.get("hospital_id", "")) for record in records if record.get("hospital_id")}
    patients = {str(record.get("patient_id", "")) for record in records if record.get("patient_id")}
    stays = {str(record.get("stay_id", "")) for record in records if record.get("stay_id")}
    return {
        "records": len(records),
        "patients": len(patients),
        "stays": len(stays),
        "hospitals": len(hospitals),
        "events": sum(event_counts),
        "median_events": float(median(event_counts)) if event_counts else 0.0,
        "positive_rate": sum(labels) / len(labels) if labels else 0.0,
        "splits": dict(sorted(splits.items())),
    }


def summarize_dataset(dataset: str, events: Path, cohort: Path | None) -> dict[str, Any]:
    summary = {"dataset": dataset, **_summarize_events(events)}
    cohort_rows = _read_cohort(cohort)
    if cohort_rows:
        summary["cohort_rows"] = len(cohort_rows)
    return summary


def write_csv(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset",
        "patients",
        "stays",
        "hospitals",
        "records",
        "events",
        "median_events",
        "positive_rate",
        "splits",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: json.dumps(row[field], sort_keys=True) if field == "splits" else row[field] for field in fields})


def write_markdown(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| Dataset | Patients | Stays | Hospitals | Records | Events | Median events | Positive rate | Splits |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        split_text = ", ".join(f"{key}:{value}" for key, value in row["splits"].items())
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["dataset"]),
                    _fmt_int(row["patients"]),
                    _fmt_int(row["stays"]),
                    _fmt_int(row["hospitals"]),
                    _fmt_int(row["records"]),
                    _fmt_int(row["events"]),
                    _fmt_int(row["median_events"]),
                    _fmt_float(float(row["positive_rate"])),
                    split_text,
                ]
            )
            + " |"
        )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_tex(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{table}[t]",
        r"\caption{Dataset summary for the ICU event-stream prediction tasks. Values are computed from the released cohort and event-building scripts, not stored experiment results.}",
        r"\label{tab:dataset_summary}",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Dataset & Patients & Stays & Hospitals & Events & Med. events & Pos. rate \\",
        r"\midrule",
    ]
    for row in rows:
        values = [
            str(row["dataset"]),
            _fmt_int(row["patients"]),
            _fmt_int(row["stays"]),
            _fmt_int(row["hospitals"]),
            _fmt_int(row["events"]),
            _fmt_int(row["median_events"]),
            _fmt_float(float(row["positive_rate"])),
        ]
        lines.append(" & ".join(values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Table 1 dataset summary from cohort/event files.")
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        metavar="NAME=EVENTS_JSONL_GZ",
        help="Dataset name and event JSONL(.gz), e.g. mimic=work/events/mimic.jsonl.gz.",
    )
    parser.add_argument(
        "--cohort",
        action="append",
        default=[],
        metavar="NAME=COHORT_CSV",
        help="Optional dataset name and cohort CSV.",
    )
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--tex", type=Path, required=True)
    args = parser.parse_args()

    cohorts = dict(item.split("=", 1) for item in args.cohort)
    rows = []
    for item in args.dataset:
        name, path = item.split("=", 1)
        rows.append(summarize_dataset(name, Path(path), Path(cohorts[name]) if name in cohorts else None))
    write_csv(rows, args.csv)
    write_markdown(rows, args.markdown)
    write_tex(rows, args.tex)
    print(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
