from __future__ import annotations

import random
from pathlib import Path

from .io import write_jsonl
from .schema import Event, PatientRecord
from .splits import _hash_fraction


CONCEPTS = ("heart_rate", "sbp", "spo2", "respiratory_rate", "lactate", "creatinine", "wbc")


def make_synthetic_records(n: int = 240, seed: int = 2026) -> list[PatientRecord]:
    rng = random.Random(seed)
    records: list[PatientRecord] = []
    for index in range(n):
        patient_id = f"P{index:05d}"
        severity = rng.gauss(0, 1)
        label = int(severity + rng.gauss(0, 0.7) > 1.0)
        events: list[Event] = []
        for minute in sorted(rng.sample(range(0, 361), k=rng.randint(8, 24))):
            for concept in rng.sample(CONCEPTS, k=rng.randint(1, min(3, len(CONCEPTS)))):
                base = {"heart_rate": 85, "sbp": 115, "spo2": 97, "respiratory_rate": 18, "lactate": 1.5, "creatinine": 1.0, "wbc": 8}[concept]
                direction = {"heart_rate": 12, "sbp": -15, "spo2": -2.5, "respiratory_rate": 4, "lactate": 1.2, "creatinine": 0.35, "wbc": 2.0}[concept]
                value = base + direction * severity + rng.gauss(0, abs(direction) * 0.35 + 0.2)
                flag = "high" if (concept != "sbp" and value > base + abs(direction)) else "low" if concept in {"sbp", "spo2"} and value < base - abs(direction) else "normal"
                events.append(Event(f"E{index:05d}_{len(events):04d}", minute, "synthetic", concept, value, value, "", flag))
        fraction = _hash_fraction(patient_id, seed)
        split = "train" if fraction < 0.7 else "val" if fraction < 0.8 else "test"
        records.append(PatientRecord(patient_id, f"S{index:05d}", label, "synthetic", f"H{index % 8}", split, events))
    return records


def write_synthetic(path: str | Path, n: int = 240, seed: int = 2026) -> int:
    return write_jsonl(path, (record.to_dict() for record in make_synthetic_records(n, seed)))

