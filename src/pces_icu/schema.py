from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class Event:
    event_id: str
    relative_minute: int
    source: str
    concept: str
    raw_value: str | float | int | None
    canonical_value: float | str | None
    unit: str = ""
    abnormal_flag: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Event":
        return cls(**value)


@dataclass
class PatientRecord:
    patient_id: str
    stay_id: str
    label: int
    dataset: str
    hospital_id: str = ""
    split: str = ""
    events: list[Event] | None = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["events"] = [event.to_dict() for event in (self.events or [])]
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PatientRecord":
        value = dict(value)
        value["events"] = [Event.from_dict(event) for event in value.get("events", [])]
        return cls(**value)


COMMON_CONCEPT_PATTERNS: dict[str, tuple[str, ...]] = {
    "heart_rate": (r"^heart rate$", r"^hr$", r"pulse"),
    "sbp": (r"systolic.*blood pressure", r"blood pressure.*systolic", r"non.?invasive systolic", r"^sbp$"),
    "dbp": (r"diastolic.*blood pressure", r"blood pressure.*diastolic", r"non.?invasive diastolic", r"^dbp$"),
    "map": (r"mean arterial", r"blood pressure.*mean", r"non.?invasive mean", r"^map$"),
    "respiratory_rate": (r"respiratory rate", r"respiration", r"^rr$"),
    "temperature": (r"temperature", r"^temp"),
    "spo2": (r"oxygen saturation", r"spo2", r"sao2"),
    "glucose": (r"^glucose",),
    "lactate": (r"lactate", r"lactic acid"),
    "creatinine": (r"^creatinine",),
    "bun": (r"urea nitrogen", r"^bun$"),
    "sodium": (r"^sodium",),
    "potassium": (r"^potassium",),
    "chloride": (r"^chloride",),
    "bicarbonate": (r"bicarbonate", r"total co2"),
    "hemoglobin": (r"^hemoglobin",),
    "hematocrit": (r"^hematocrit",),
    "wbc": (r"white blood cell", r"^wbc"),
    "platelet": (r"platelet",),
    "bilirubin": (r"bilirubin,? total",),
    "albumin": (r"^albumin$",),
    "inr": (r"^inr",),
    "ph": (r"^ph$", r"blood gas.*ph"),
    "pco2": (r"pco2", r"carbon dioxide pressure"),
    "po2": (r"^po2", r"oxygen pressure"),
}

INTERVENTION_PATTERNS: dict[str, tuple[str, ...]] = {
    "vasopressor_norepinephrine": (r"norepinephrine", r"noradrenaline", r"levophed"),
    "vasopressor_epinephrine": (r"epinephrine", r"adrenaline"),
    "vasopressor_vasopressin": (r"vasopressin",),
    "vasopressor_phenylephrine": (r"phenylephrine", r"neosynephrine"),
    "vasopressor_dopamine": (r"dopamine",),
    "noninvasive_ventilation": (r"non.?invasive ventilation", r"bipap", r"cpap"),
    "invasive_ventilation": (r"mechanical ventilation", r"(?<!non-)invasive ventilation"),
    "intubation": (r"intubation", r"endotracheal tube"),
    "renal_replacement": (r"hemodialysis", r"continuous renal replacement", r"cvvh", r"crrt"),
}


def normalize_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.strip().lower()
    return re.sub(r"\s+", " ", text)


def canonical_concept(label: Any) -> str | None:
    label = normalize_text(label)
    for concept, patterns in COMMON_CONCEPT_PATTERNS.items():
        if any(re.search(pattern, label, flags=re.I) for pattern in patterns):
            return concept
    return None


def canonical_intervention(label: Any) -> str | None:
    label = normalize_text(label)
    for concept, patterns in INTERVENTION_PATTERNS.items():
        if any(re.search(pattern, label, flags=re.I) for pattern in patterns):
            return concept
    return None


def safe_float(value: Any) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def abnormal_flag(value: float | None, lower: Any = None, upper: Any = None, raw_flag: Any = None) -> str:
    # Pandas may expose warning columns as NumPy scalar values.  Sending some
    # malformed numeric scalars through the regex-based normalize_text path can
    # abort the interpreter inside NumPy's scalar formatter.  Numeric warning
    # fields need no text normalization, so classify them directly.
    numeric_flag = safe_float(raw_flag)
    if numeric_flag == 1.0:
        return "abnormal"
    if isinstance(raw_flag, bytes):
        flag = raw_flag.decode("utf-8", errors="replace").strip().lower()
    elif isinstance(raw_flag, str):
        flag = raw_flag.strip().lower()
    else:
        flag = ""
    if flag in {"1", "true", "yes"}:
        return "abnormal"
    if flag in {"high", "h", "abnormal", "critical high"}:
        return "high"
    if flag in {"low", "l", "critical low"}:
        return "low"
    low, high = safe_float(lower), safe_float(upper)
    if value is not None and low is not None and value < low:
        return "low"
    if value is not None and high is not None and value > high:
        return "high"
    if value is not None and (low is not None or high is not None):
        return "normal"
    return "unknown"


def validate_events(events: Iterable[Event], window_minutes: int) -> list[str]:
    errors: list[str] = []
    seen: set[str] = set()
    for event in events:
        if event.event_id in seen:
            errors.append(f"duplicate event_id={event.event_id}")
        seen.add(event.event_id)
        if not 0 <= event.relative_minute <= window_minutes:
            errors.append(f"event outside window: {event.event_id}@{event.relative_minute}")
        if not event.concept:
            errors.append(f"empty concept: {event.event_id}")
    return errors
