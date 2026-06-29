from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np

from .schema import PatientRecord, safe_float


STAT_NAMES = ("count", "mean", "min", "max", "last", "abnormal_rate")


def fit_feature_names(records: Iterable[PatientRecord]) -> list[str]:
    concepts = sorted({event.concept for record in records for event in (record.events or [])})
    return [f"{concept}__{stat}" for concept in concepts for stat in STAT_NAMES] + ["total_events", "unique_concepts"]


def aggregate_record(record: PatientRecord, feature_names: list[str]) -> np.ndarray:
    values: dict[str, list[tuple[int, float, bool]]] = defaultdict(list)
    for event in record.events or []:
        numeric = safe_float(event.canonical_value)
        if numeric is not None:
            values[event.concept].append((event.relative_minute, numeric, event.abnormal_flag in {"high", "low"}))
    output: dict[str, float] = {}
    for concept, entries in values.items():
        entries.sort()
        array = np.asarray([entry[1] for entry in entries], dtype=float)
        output[f"{concept}__count"] = float(len(entries))
        output[f"{concept}__mean"] = float(array.mean())
        output[f"{concept}__min"] = float(array.min())
        output[f"{concept}__max"] = float(array.max())
        output[f"{concept}__last"] = float(entries[-1][1])
        output[f"{concept}__abnormal_rate"] = float(np.mean([entry[2] for entry in entries]))
    output["total_events"] = float(len(record.events or []))
    output["unique_concepts"] = float(len({event.concept for event in (record.events or [])}))
    return np.asarray([output.get(name, np.nan) for name in feature_names], dtype=np.float32)

