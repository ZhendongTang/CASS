from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .cohort import build_eicu_cohort, build_mimic_like_cohort, cohort_summary
from .extract import extract_eicu_events, extract_mimic_like_events
from .io import read_jsonl, write_json
from .metrics import binary_metrics
from .schema import PatientRecord, validate_events
from .splits import assert_disjoint, assign_hospital_splits, assign_patient_splits
from .synthetic import write_synthetic
from .tasks import build_future_intervention_task, build_los_binary_task
from .training import (
    evaluate_baseline_checkpoint,
    evaluate_llm_checkpoint,
    evaluate_transformer_checkpoint,
    train_baseline,
    train_event_transformer,
    train_llm,
)


def _config(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def command_build_cohort(args) -> None:
    if args.dataset == "mimic":
        frame = build_mimic_like_cohort(args.data_root, "mimic", args.window_hours)
        frame = assign_patient_splits(frame, args.seed)
    elif args.dataset == "nwicu":
        frame = build_mimic_like_cohort(args.data_root, "nwicu", args.window_hours, "nw_hosp", "nw_icu")
        frame = assign_patient_splits(frame, args.seed)
    else:
        frame = build_eicu_cohort(args.data_root, args.window_hours)
        frame = assign_hospital_splits(frame, args.seed)
    assert_disjoint(frame)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    summary = cohort_summary(frame)
    summary["split_counts"] = {str(k): int(v) for k, v in frame["split"].value_counts().to_dict().items()}
    write_json(output.with_suffix(".summary.json"), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def command_build_events(args) -> None:
    cohort = pd.read_csv(args.cohort, low_memory=False)
    if args.dataset == "mimic":
        count = extract_mimic_like_events(args.data_root, cohort, args.output, "hosp", "icu", args.window_hours, args.chunksize)
    elif args.dataset == "nwicu":
        count = extract_mimic_like_events(args.data_root, cohort, args.output, "nw_hosp", "nw_icu", args.window_hours, args.chunksize)
    else:
        count = extract_eicu_events(args.data_root, cohort, args.output, args.window_hours, args.chunksize)
    print(json.dumps({"records": count, "output": str(args.output)}, ensure_ascii=False))


def command_audit_events(args) -> None:
    records = [PatientRecord.from_dict(value) for value in read_jsonl(args.data)]
    errors = []
    concept_counts: dict[str, int] = {}
    no_events = 0
    for record in records:
        events = record.events or []
        no_events += int(not events)
        errors.extend(f"{record.stay_id}: {error}" for error in validate_events(events, args.window_hours * 60))
        for event in events:
            concept_counts[event.concept] = concept_counts.get(event.concept, 0) + 1
    result = {
        "records": len(records),
        "records_without_events": no_events,
        "event_count": sum(concept_counts.values()),
        "concept_counts": dict(sorted(concept_counts.items(), key=lambda item: (-item[1], item[0]))),
        "validation_errors": errors[:100],
        "validation_error_count": len(errors),
    }
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(2)


def command_evaluate_predictions(args) -> None:
    rows = list(read_jsonl(args.predictions))
    result = binary_metrics([row["label"] for row in rows], [row["probability"] for row in rows], args.bins)
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def command_build_los_task(args) -> None:
    result = build_los_binary_task(
        args.events,
        args.cohort,
        args.output,
        args.summary,
        args.threshold_hours,
        args.input_window_hours,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def command_build_future_intervention_task(args) -> None:
    result = build_future_intervention_task(
        args.events,
        args.output,
        args.summary,
        args.concept,
        args.landmark_hours,
        args.horizon_hours,
        not args.keep_prevalent,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def command_train(args) -> None:
    config = _config(args.config) if args.config else {}
    if args.trainer == "baseline":
        result = train_baseline(args.data, args.output_dir, int(config.get("seed", 2026)), config.get("model", "hist_gbdt"))
    elif args.trainer == "transformer":
        result = train_event_transformer(args.data, args.output_dir, config)
    else:
        result = train_llm(args.data, args.output_dir, config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def command_evaluate_checkpoint(args) -> None:
    config = _config(args.config) if args.config else {}
    if args.trainer == "baseline":
        result = evaluate_baseline_checkpoint(args.data, args.model_dir, args.output_dir, args.split)
    elif args.trainer == "transformer":
        result = evaluate_transformer_checkpoint(args.data, args.model_dir, args.output_dir, args.split, int(config.get("eval_batch_size", 64)))
    else:
        result = evaluate_llm_checkpoint(args.data, args.model_dir, args.output_dir, config, args.split)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pces-icu", description="PCES-ICU experiment CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("make-synthetic", help="Generate synthetic patient event JSONL")
    p.add_argument("--output", required=True); p.add_argument("--n", type=int, default=240); p.add_argument("--seed", type=int, default=2026)
    p.set_defaults(func=lambda args: print(json.dumps({"records": write_synthetic(args.output, args.n, args.seed), "output": args.output})))

    p = sub.add_parser("build-cohort", help="Build a leakage-safe 6h landmark cohort")
    p.add_argument("--dataset", choices=["mimic", "eicu", "nwicu"], required=True)
    p.add_argument("--data-root", required=True, help="Dataset version directory, not the common parent")
    p.add_argument("--output", required=True); p.add_argument("--window-hours", type=int, default=6); p.add_argument("--seed", type=int, default=2026)
    p.set_defaults(func=command_build_cohort)

    p = sub.add_parser("build-events", help="Extract canonical first-window events")
    p.add_argument("--dataset", choices=["mimic", "eicu", "nwicu"], required=True)
    p.add_argument("--data-root", required=True); p.add_argument("--cohort", required=True); p.add_argument("--output", required=True)
    p.add_argument("--window-hours", type=int, default=6); p.add_argument("--chunksize", type=int, default=500000)
    p.set_defaults(func=command_build_events)

    p = sub.add_parser("audit-events", help="Validate event schema and window leakage")
    p.add_argument("--data", required=True); p.add_argument("--window-hours", type=int, default=6); p.add_argument("--output")
    p.set_defaults(func=command_audit_events)

    p = sub.add_parser("build-los-task", help="Derive a binary LOS task from an event JSONL and cohort CSV")
    p.add_argument("--events", required=True)
    p.add_argument("--cohort", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--summary")
    p.add_argument("--threshold-hours", type=float, default=48.0)
    p.add_argument("--input-window-hours", type=float, default=6.0)
    p.set_defaults(func=command_build_los_task)

    p = sub.add_parser("build-future-intervention-task", help="Derive a future intervention onset task from a longer-window event JSONL")
    p.add_argument("--events", required=True, help="Event JSONL extracted through the full prediction horizon, e.g. 24h")
    p.add_argument("--output", required=True)
    p.add_argument("--summary")
    p.add_argument("--concept", action="append", help="Target intervention concept; repeatable. Defaults to all canonical interventions.")
    p.add_argument("--landmark-hours", type=float, default=6.0)
    p.add_argument("--horizon-hours", type=float, default=24.0)
    p.add_argument("--keep-prevalent", action="store_true", help="Keep patients already receiving a target intervention before the landmark")
    p.set_defaults(func=command_build_future_intervention_task)

    p = sub.add_parser("train", help="Train a baseline, structured Transformer, or LLM")
    p.add_argument("--trainer", choices=["baseline", "transformer", "llm"], required=True)
    p.add_argument("--data", required=True); p.add_argument("--output-dir", required=True); p.add_argument("--config", required=True)
    p.set_defaults(func=command_train)

    p = sub.add_parser("evaluate-predictions", help="Compute ground-truth binary metrics")
    p.add_argument("--predictions", required=True); p.add_argument("--bins", type=int, default=15); p.add_argument("--output")
    p.set_defaults(func=command_evaluate_predictions)

    p = sub.add_parser("evaluate-checkpoint", help="Evaluate a saved checkpoint without refitting on the target dataset")
    p.add_argument("--trainer", choices=["baseline", "transformer", "llm"], required=True)
    p.add_argument("--data", required=True); p.add_argument("--model-dir", required=True); p.add_argument("--output-dir", required=True)
    p.add_argument("--config"); p.add_argument("--split", default="test")
    p.set_defaults(func=command_evaluate_checkpoint)

    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
