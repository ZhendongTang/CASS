from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .io import read_jsonl, write_json, write_jsonl
from .schema import INTERVENTION_PATTERNS, PatientRecord


DEFAULT_INTERVENTION_CONCEPTS = frozenset(INTERVENTION_PATTERNS)


def _cohort_duration_hours(frame: pd.DataFrame) -> pd.Series:
    if "duration_hours" in frame:
        return pd.to_numeric(frame["duration_hours"], errors="coerce")
    if {"intime", "outtime"}.issubset(frame.columns):
        intime = pd.to_datetime(frame["intime"], errors="coerce")
        outtime = pd.to_datetime(frame["outtime"], errors="coerce")
        return (outtime - intime).dt.total_seconds() / 3600
    raise ValueError("cohort must contain duration_hours or both intime/outtime")


def _summary(records: list[PatientRecord], *, task: str, excluded: int = 0, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    labels = [int(record.label) for record in records]
    splits = Counter(record.split or "" for record in records)
    result: dict[str, Any] = {
        "task": task,
        "records": len(records),
        "positives": int(sum(labels)),
        "positive_rate": float(sum(labels) / len(labels)) if labels else 0.0,
        "excluded": int(excluded),
        "split_counts": dict(sorted(splits.items())),
    }
    if extra:
        result.update(extra)
    return result


def _with_events_before(record: PatientRecord, max_minute: int) -> PatientRecord:
    return PatientRecord(
        patient_id=record.patient_id,
        stay_id=record.stay_id,
        label=record.label,
        dataset=record.dataset,
        hospital_id=record.hospital_id,
        split=record.split,
        events=[event for event in (record.events or []) if event.relative_minute <= max_minute],
    )


def build_los_binary_task(
    events_path: str | Path,
    cohort_path: str | Path,
    output_path: str | Path,
    summary_path: str | Path | None = None,
    threshold_hours: float = 48.0,
    input_window_hours: float = 6.0,
) -> dict[str, Any]:
    cohort = pd.read_csv(cohort_path, low_memory=False)
    cohort = cohort.copy()
    cohort["stay_id"] = cohort["stay_id"].astype(str)
    duration_by_stay = dict(zip(cohort["stay_id"], _cohort_duration_hours(cohort)))
    max_minute = int(round(input_window_hours * 60))

    records: list[PatientRecord] = []
    missing_duration = 0
    for value in read_jsonl(events_path):
        record = PatientRecord.from_dict(value)
        stay_duration = duration_by_stay.get(str(record.stay_id))
        if pd.isna(stay_duration):
            missing_duration += 1
            continue
        derived = _with_events_before(record, max_minute)
        derived.label = int(float(stay_duration) > threshold_hours)
        records.append(derived)

    write_jsonl(output_path, (record.to_dict() for record in records))
    result = _summary(
        records,
        task=f"los_gt_{threshold_hours:g}h",
        excluded=missing_duration,
        extra={"threshold_hours": float(threshold_hours), "input_window_hours": float(input_window_hours)},
    )
    if summary_path:
        write_json(summary_path, result)
    return result


def build_future_intervention_task(
    events_path: str | Path,
    output_path: str | Path,
    summary_path: str | Path | None = None,
    concepts: Iterable[str] | None = None,
    landmark_hours: float = 6.0,
    horizon_hours: float = 24.0,
    exclude_prevalent: bool = True,
) -> dict[str, Any]:
    target_concepts = set(concepts or DEFAULT_INTERVENTION_CONCEPTS)
    landmark_minute = int(round(landmark_hours * 60))
    horizon_minute = int(round(horizon_hours * 60))
    if horizon_minute <= landmark_minute:
        raise ValueError("horizon_hours must be greater than landmark_hours")

    records: list[PatientRecord] = []
    excluded_prevalent = 0
    for value in read_jsonl(events_path):
        record = PatientRecord.from_dict(value)
        events = record.events or []
        before = [event for event in events if event.relative_minute <= landmark_minute]
        prevalent = any(event.concept in target_concepts for event in before)
        if exclude_prevalent and prevalent:
            excluded_prevalent += 1
            continue
        future = any(
            landmark_minute < event.relative_minute <= horizon_minute and event.concept in target_concepts
            for event in events
        )
        records.append(
            PatientRecord(
                patient_id=record.patient_id,
                stay_id=record.stay_id,
                label=int(future),
                dataset=record.dataset,
                hospital_id=record.hospital_id,
                split=record.split,
                events=before,
            )
        )

    write_jsonl(output_path, (record.to_dict() for record in records))
    result = _summary(
        records,
        task="future_intervention_onset",
        excluded=excluded_prevalent,
        extra={
            "target_concepts": sorted(target_concepts),
            "landmark_hours": float(landmark_hours),
            "horizon_hours": float(horizon_hours),
            "exclude_prevalent": bool(exclude_prevalent),
        },
    )
    if summary_path:
        write_json(summary_path, result)
    return result
