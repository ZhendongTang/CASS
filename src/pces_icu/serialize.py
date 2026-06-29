from __future__ import annotations

import hashlib
import html
import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from .schema import Event, safe_float


@dataclass(frozen=True)
class SerializationConfig:
    include_delta_t: bool = True
    include_set_markers: bool = True
    permute_within_sets: bool = True
    shuffle_across_timestamps: bool = False
    max_events: int = 512
    max_chars: int = 24_000


@dataclass(frozen=True)
class ClinTimeXMLConfig:
    bucket_minutes: int = 60
    permute_within_blocks: bool = True
    max_nodes: int = 128
    max_chars: int = 12_000


@dataclass(frozen=True)
class ClinTimeCompactConfig:
    bucket_minutes: int = 60
    permute_within_blocks: bool = True
    max_nodes: int = 160
    max_chars: int = 6_000
    include_evidence_ids: bool = False
    include_legend: bool = False


@dataclass(frozen=True)
class ClinTimeJSONConfig:
    bucket_minutes: int = 60
    permute_within_blocks: bool = True
    max_nodes: int = 160
    max_chars: int = 6_000
    include_evidence_ids: bool = False


@dataclass(frozen=True)
class ClinTimeFeatureSummaryConfig:
    max_rows: int = 80
    max_chars: int = 3_000


def _event_priority(event: Event) -> tuple[int, int, str]:
    abnormal = 0 if event.abnormal_flag in {"high", "low"} else 1
    return (abnormal, -event.relative_minute, event.event_id)


def select_matched_events(events: Iterable[Event], max_events: int) -> list[Event]:
    """Select one deterministic event set reused by every serialization view."""
    values = list(events)
    if len(values) <= max_events:
        return sorted(values, key=lambda event: (event.relative_minute, event.event_id))
    keep = sorted(values, key=_event_priority)[:max_events]
    return sorted(keep, key=lambda event: (event.relative_minute, event.event_id))


def _clean(value: object) -> str:
    return str(value).replace("\n", " ").replace("\r", " ").replace("<", "[").replace(">", "]")


def serialize_events(events: Iterable[Event], config: SerializationConfig, seed: int) -> str:
    rng = random.Random(seed)
    selected = select_matched_events(events, config.max_events)
    groups: dict[int, list[Event]] = defaultdict(list)
    for event in selected:
        groups[event.relative_minute].append(event)
    times = sorted(groups)
    if config.shuffle_across_timestamps:
        rng.shuffle(times)
    blocks: list[str] = []
    previous = 0
    for minute in times:
        current = list(groups[minute])
        if config.permute_within_sets:
            rng.shuffle(current)
        else:
            current.sort(key=lambda event: (event.concept, event.event_id))
        delta = minute - previous
        header = f"<T=+{minute}m>"
        if config.include_delta_t:
            header += f"<DT={delta}m>"
        lines = [header + ("<SET>" if config.include_set_markers else "")]
        for event in current:
            value = event.canonical_value if event.canonical_value is not None else event.raw_value
            lines.append(
                f"[{_clean(event.event_id)}] {_clean(event.concept)}={_clean(value)}"
                f" {_clean(event.unit)}; {_clean(event.abnormal_flag)}; source={_clean(event.source)}"
            )
        if config.include_set_markers:
            lines.append("</SET>")
        block = "\n".join(lines)
        if sum(len(value) + 1 for value in blocks) + len(block) > config.max_chars:
            break
        blocks.append(block)
        previous = minute
    return "\n".join(blocks)


def paired_views(events: Iterable[Event], config: SerializationConfig, seed: int) -> tuple[str, str]:
    return serialize_events(events, config, seed), serialize_events(events, config, seed + 1_000_003)


def stable_view_seed(patient_id: str, seed: int) -> int:
    digest = hashlib.blake2b(f"{patient_id}:{seed}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little")


def _xml_value(value: object) -> str:
    return html.escape(str(value), quote=True)


def _compact_number(value: float | None) -> str:
    return "" if value is None else f"{value:.5g}"


CONCEPT_CODES = {
    "heart_rate": "HR",
    "sbp": "SBP",
    "dbp": "DBP",
    "map": "MAP",
    "respiratory_rate": "RR",
    "temperature": "TEMP",
    "spo2": "SPO2",
    "glucose": "GLU",
    "lactate": "LAC",
    "creatinine": "CR",
    "bun": "BUN",
    "sodium": "NA",
    "potassium": "K",
    "chloride": "CL",
    "bicarbonate": "HCO3",
    "hemoglobin": "HGB",
    "hematocrit": "HCT",
    "wbc": "WBC",
    "platelet": "PLT",
    "bilirubin": "BILI",
    "albumin": "ALB",
    "inr": "INR",
    "ph": "PH",
    "pco2": "PCO2",
    "po2": "PO2",
    "vasopressor_norepinephrine": "NE",
    "vasopressor_epinephrine": "EPI",
    "vasopressor_vasopressin": "VP",
    "vasopressor_phenylephrine": "PE",
    "vasopressor_dopamine": "DA",
    "noninvasive_ventilation": "NIV",
    "invasive_ventilation": "VENT",
    "intubation": "INTUB",
    "renal_replacement": "RRT",
}

FLAG_CODES = {
    "high": "H",
    "low": "L",
    "mixed": "M",
    "abnormal": "A",
    "normal": "N",
    "unknown": "U",
}

SOURCE_CODES = {
    "vital": "V",
    "lab": "L",
    "medication_admin": "MA",
    "medication_order": "MO",
    "procedure": "P",
    "treatment": "T",
}


def _code_token(value: str) -> str:
    value = value.strip()
    if value in CONCEPT_CODES:
        return CONCEPT_CODES[value]
    token = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").upper()
    return token[:18] or "UNK"


def _summary_flag(events: list[Event]) -> str:
    flags = {event.abnormal_flag for event in events}
    if "high" in flags and "low" in flags:
        return "mixed"
    for value in ("high", "low", "abnormal", "normal"):
        if value in flags:
            return value
    return "unknown"


def _summary_node(events: list[Event]) -> tuple[str, list[str], tuple[int, int, int, str]]:
    ordered = sorted(events, key=lambda event: (event.relative_minute, event.event_id))
    numeric = [(event, safe_float(event.canonical_value)) for event in ordered]
    numeric = [(event, value) for event, value in numeric if value is not None]
    evidence: list[str] = []
    attributes: dict[str, object] = {
        "concept": ordered[0].concept,
        "n": len(ordered),
        "flag": _summary_flag(ordered),
        "source": ",".join(sorted({event.source for event in ordered})),
    }
    units = [event.unit for event in ordered if event.unit]
    if units:
        attributes["unit"] = sorted(set(units), key=lambda unit: (-units.count(unit), unit))[0]
    if numeric:
        first_event, first = numeric[0]
        last_event, last = numeric[-1]
        min_event, minimum = min(numeric, key=lambda item: item[1])
        max_event, maximum = max(numeric, key=lambda item: item[1])
        scale = max(abs(first), abs(last), 1.0)
        relative_change = (last - first) / scale
        trend = "up" if relative_change > 0.05 else "down" if relative_change < -0.05 else "stable"
        attributes.update(
            first=_compact_number(first), last=_compact_number(last), min=_compact_number(minimum),
            max=_compact_number(maximum), trend=trend,
        )
        evidence = [first_event.event_id, min_event.event_id, max_event.event_id, last_event.event_id]
    else:
        latest = ordered[-1]
        value = latest.canonical_value if latest.canonical_value is not None else latest.raw_value
        attributes["last"] = "" if value is None else value
        attributes["trend"] = "categorical"
        evidence = [ordered[0].event_id, latest.event_id]
    evidence = list(dict.fromkeys(evidence))
    attributes["evidence_ids"] = ",".join(evidence)
    rendered = " ".join(f'{key}="{_xml_value(value)}"' for key, value in attributes.items())
    intervention = int(any(event.source in {"medication_admin", "medication_order", "procedure", "treatment"} for event in ordered))
    abnormal = int(attributes["flag"] in {"high", "low", "mixed", "abnormal"})
    priority = (-intervention, -abnormal, -ordered[-1].relative_minute, ordered[0].concept)
    return f"<observation {rendered} />", evidence, priority


def serialize_clintime_xml(events: Iterable[Event], config: ClinTimeXMLConfig, seed: int) -> str:
    """Serialize an ICU trajectory as ordered time blocks containing unordered clinical-state nodes."""
    rng = random.Random(seed)
    buckets: dict[tuple[int, str], list[Event]] = defaultdict(list)
    for event in events:
        bucket = max(0, event.relative_minute) // config.bucket_minutes
        buckets[(bucket, event.concept)].append(event)
    candidates: list[tuple[tuple[int, int, int, str], int, str]] = []
    for (bucket, _), values in buckets.items():
        node, _, priority = _summary_node(values)
        candidates.append((priority, bucket, node))
    candidates.sort(key=lambda item: item[0])
    selected: list[tuple[int, str]] = []
    # Reserve space for the root/schema note plus up to seven timeblock wrappers.
    used_chars = 1_600
    for _, bucket, node in candidates:
        if len(selected) >= config.max_nodes or used_chars + len(node) + 1 > config.max_chars:
            continue
        selected.append((bucket, node))
        used_chars += len(node) + 1
    grouped: dict[int, list[str]] = defaultdict(list)
    for bucket, node in selected:
        grouped[bucket].append(node)
    lines = [
        '<icu_stream schema="ClinTime-XML/1.0" window_min="0" window_max="360">',
        '<schema_note>timeblocks are ordered; observations within each timeblock are an unordered set; each observation atomically binds concept, values, unit, abnormality, and source event IDs</schema_note>',
    ]
    previous_start = 0
    for bucket in sorted(grouped):
        start = bucket * config.bucket_minutes
        end = min(360, start + config.bucket_minutes - 1)
        nodes = list(grouped[bucket])
        if config.permute_within_blocks:
            rng.shuffle(nodes)
        else:
            nodes.sort()
        lines.append(
            f'<timeblock start_min="{start}" end_min="{end}" delta_from_previous="{start - previous_start}" semantics="unordered">'
        )
        lines.extend(nodes)
        lines.append("</timeblock>")
        previous_start = start
    lines.append("</icu_stream>")
    return "\n".join(lines)


def _compact_summary_row(events: list[Event], include_evidence_ids: bool = False) -> tuple[str, list[str], tuple[int, int, int, str]]:
    ordered = sorted(events, key=lambda event: (event.relative_minute, event.event_id))
    numeric = [(event, safe_float(event.canonical_value)) for event in ordered]
    numeric = [(event, value) for event, value in numeric if value is not None]
    concept = _code_token(ordered[0].concept)
    flag = FLAG_CODES.get(_summary_flag(ordered), "U")
    sources = sorted({SOURCE_CODES.get(event.source, _code_token(event.source)) for event in ordered})
    source = ",".join(sources[:3]) if sources else "-"
    units = [event.unit for event in ordered if event.unit]
    unit = sorted(set(units), key=lambda item: (-units.count(item), item))[0] if units else "-"
    unit = re.sub(r"\s+", "", _clean(unit))[:12] or "-"
    evidence: list[str]
    if numeric:
        first_event, first = numeric[0]
        last_event, last = numeric[-1]
        min_event, minimum = min(numeric, key=lambda item: item[1])
        max_event, maximum = max(numeric, key=lambda item: item[1])
        scale = max(abs(first), abs(last), 1.0)
        relative_change = (last - first) / scale
        trend = "U" if relative_change > 0.05 else "D" if relative_change < -0.05 else "S"
        payload = (
            f"{concept}|n={len(ordered)}|f={flag}|src={source}|u={unit}|"
            f"v={_compact_number(first)}>{_compact_number(last)}|"
            f"r={_compact_number(minimum)}..{_compact_number(maximum)}|tr={trend}"
        )
        evidence = [first_event.event_id, min_event.event_id, max_event.event_id, last_event.event_id]
    else:
        latest = ordered[-1]
        value = latest.canonical_value if latest.canonical_value is not None else latest.raw_value
        payload = (
            f"{concept}|n={len(ordered)}|f={flag}|src={source}|u={unit}|"
            f"last={_clean(value)[:24] if value is not None else '-'}|tr=C"
        )
        evidence = [ordered[0].event_id, latest.event_id]
    evidence = list(dict.fromkeys(evidence))
    if include_evidence_ids:
        payload = f"{payload}|ids={','.join(evidence)}"
    intervention = int(any(event.source in {"medication_admin", "medication_order", "procedure", "treatment"} for event in ordered))
    abnormal = int(flag in {"H", "L", "M", "A"})
    priority = (-intervention, -abnormal, -ordered[-1].relative_minute, ordered[0].concept)
    return payload, evidence, priority


def serialize_clintime_compact(events: Iterable[Event], config: ClinTimeCompactConfig, seed: int) -> str:
    """Token-efficient ICU event summary with ordered time buckets and unordered rows."""
    rng = random.Random(seed)
    buckets: dict[tuple[int, str], list[Event]] = defaultdict(list)
    for event in events:
        bucket = max(0, event.relative_minute) // config.bucket_minutes
        buckets[(bucket, event.concept)].append(event)
    candidates: list[tuple[tuple[int, int, int, str], int, str]] = []
    for (bucket, _), values in buckets.items():
        row, _, priority = _compact_summary_row(values, config.include_evidence_ids)
        candidates.append((priority, bucket, row))
    candidates.sort(key=lambda item: item[0])
    selected: list[tuple[int, str]] = []
    used_chars = 260
    for _, bucket, row in candidates:
        if len(selected) >= config.max_nodes or used_chars + len(row) + 4 > config.max_chars:
            continue
        selected.append((bucket, row))
        used_chars += len(row) + 4
    grouped: dict[int, list[str]] = defaultdict(list)
    for bucket, row in selected:
        grouped[bucket].append(row)
    lines = []
    if config.include_legend:
        lines.extend([
            "ClinTime-Compact/1 time buckets ordered; rows inside a bucket are unordered.",
            "row: B=start-end|concept|n=count|f=flag(H/L/M/A/N/U)|src=source|u=unit|v=first>last|r=min..max|tr=U/D/S/C|id-list",
        ])
    for bucket in sorted(grouped):
        start = bucket * config.bucket_minutes
        end = min(360, start + config.bucket_minutes - 1)
        rows = list(grouped[bucket])
        if config.permute_within_blocks:
            rng.shuffle(rows)
        else:
            rows.sort()
        lines.extend(f"B={start}-{end}|{row}" for row in rows)
    return "\n".join(lines)


def _json_summary_node(events: list[Event], include_evidence_ids: bool = False) -> tuple[dict[str, object], tuple[int, int, int, str]]:
    ordered = sorted(events, key=lambda event: (event.relative_minute, event.event_id))
    numeric = [(event, safe_float(event.canonical_value)) for event in ordered]
    numeric = [(event, value) for event, value in numeric if value is not None]
    flag = FLAG_CODES.get(_summary_flag(ordered), "U")
    sources = sorted({SOURCE_CODES.get(event.source, _code_token(event.source)) for event in ordered})
    units = [event.unit for event in ordered if event.unit]
    node: dict[str, object] = {
        "c": _code_token(ordered[0].concept),
        "n": len(ordered),
        "f": flag,
        "src": sources[:3],
    }
    if units:
        node["u"] = re.sub(r"\s+", "", _clean(sorted(set(units), key=lambda item: (-units.count(item), item))[0]))[:12]
    evidence: list[str]
    if numeric:
        first_event, first = numeric[0]
        last_event, last = numeric[-1]
        min_event, minimum = min(numeric, key=lambda item: item[1])
        max_event, maximum = max(numeric, key=lambda item: item[1])
        scale = max(abs(first), abs(last), 1.0)
        relative_change = (last - first) / scale
        node.update({
            "v": [_compact_number(first), _compact_number(last)],
            "r": [_compact_number(minimum), _compact_number(maximum)],
            "tr": "U" if relative_change > 0.05 else "D" if relative_change < -0.05 else "S",
        })
        evidence = [first_event.event_id, min_event.event_id, max_event.event_id, last_event.event_id]
    else:
        latest = ordered[-1]
        value = latest.canonical_value if latest.canonical_value is not None else latest.raw_value
        node.update({"last": _clean(value)[:24] if value is not None else "", "tr": "C"})
        evidence = [ordered[0].event_id, latest.event_id]
    if include_evidence_ids:
        node["ids"] = list(dict.fromkeys(evidence))
    intervention = int(any(event.source in {"medication_admin", "medication_order", "procedure", "treatment"} for event in ordered))
    abnormal = int(flag in {"H", "L", "M", "A"})
    priority = (-intervention, -abnormal, -ordered[-1].relative_minute, ordered[0].concept)
    return node, priority


def serialize_clintime_json(events: Iterable[Event], config: ClinTimeJSONConfig, seed: int) -> str:
    """Minified JSON ICU summary. Time buckets are ordered; rows inside buckets are unordered."""
    rng = random.Random(seed)
    buckets: dict[tuple[int, str], list[Event]] = defaultdict(list)
    for event in events:
        bucket = max(0, event.relative_minute) // config.bucket_minutes
        buckets[(bucket, event.concept)].append(event)
    candidates: list[tuple[tuple[int, int, int, str], int, dict[str, object]]] = []
    for (bucket, _), values in buckets.items():
        node, priority = _json_summary_node(values, config.include_evidence_ids)
        candidates.append((priority, bucket, node))
    candidates.sort(key=lambda item: item[0])
    selected: list[tuple[int, dict[str, object]]] = []
    for _, bucket, node in candidates:
        if len(selected) >= config.max_nodes:
            continue
        selected.append((bucket, node))
        grouped_preview: dict[int, list[dict[str, object]]] = defaultdict(list)
        for b, n in selected:
            grouped_preview[b].append(n)
        preview = {
            "fmt": "CTJ1",
            "ord": "time",
            "set": "bucket_rows_unordered",
            "b": [
                {"t": [b * config.bucket_minutes, min(360, b * config.bucket_minutes + config.bucket_minutes - 1)], "x": x}
                for b, x in sorted(grouped_preview.items())
            ],
        }
        if len(json.dumps(preview, ensure_ascii=False, separators=(",", ":"))) > config.max_chars:
            selected.pop()
            break
    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for bucket, node in selected:
        grouped[bucket].append(node)
    blocks = []
    for bucket in sorted(grouped):
        nodes = list(grouped[bucket])
        if config.permute_within_blocks:
            rng.shuffle(nodes)
        else:
            nodes.sort(key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")))
        start = bucket * config.bucket_minutes
        blocks.append({"t": [start, min(360, start + config.bucket_minutes - 1)], "x": nodes})
    return json.dumps(
        {"fmt": "CTJ1", "ord": "time", "set": "bucket_rows_unordered", "b": blocks},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _feature_summary_row(events: list[Event]) -> tuple[str, tuple[int, int, int, str]]:
    ordered = sorted(events, key=lambda event: (event.relative_minute, event.event_id))
    numeric = [(event, safe_float(event.canonical_value)) for event in ordered]
    numeric = [(event, value) for event, value in numeric if value is not None]
    concept = _code_token(ordered[0].concept)
    flag = FLAG_CODES.get(_summary_flag(ordered), "U")
    sources = sorted({SOURCE_CODES.get(event.source, _code_token(event.source)) for event in ordered})
    abnormal_high = sum(1 for event in ordered if event.abnormal_flag == "high")
    abnormal_low = sum(1 for event in ordered if event.abnormal_flag == "low")
    start = ordered[0].relative_minute
    end = ordered[-1].relative_minute
    if numeric:
        first = numeric[0][1]
        last = numeric[-1][1]
        values = [value for _, value in numeric]
        scale = max(abs(first), abs(last), 1.0)
        relative_change = (last - first) / scale
        trend = "U" if relative_change > 0.05 else "D" if relative_change < -0.05 else "S"
        row = (
            f"{concept}|n={len(ordered)}|t={start}-{end}|f={flag}|hl={abnormal_high}/{abnormal_low}|"
            f"first={_compact_number(first)}|last={_compact_number(last)}|"
            f"min={_compact_number(min(values))}|max={_compact_number(max(values))}|"
            f"mean={_compact_number(sum(values) / len(values))}|tr={trend}|src={','.join(sources[:3]) or '-'}"
        )
    else:
        latest = ordered[-1]
        value = latest.canonical_value if latest.canonical_value is not None else latest.raw_value
        row = (
            f"{concept}|n={len(ordered)}|t={start}-{end}|f={flag}|hl={abnormal_high}/{abnormal_low}|"
            f"last={_clean(value)[:24] if value is not None else '-'}|tr=C|src={','.join(sources[:3]) or '-'}"
        )
    intervention = int(any(event.source in {"medication_admin", "medication_order", "procedure", "treatment"} for event in ordered))
    abnormal = int(flag in {"H", "L", "M", "A"})
    priority = (-intervention, -abnormal, -len(ordered), -ordered[-1].relative_minute, ordered[0].concept)
    return row, priority


def serialize_clintime_feature_summary(events: Iterable[Event], config: ClinTimeFeatureSummaryConfig) -> str:
    """Very compact per-concept aggregate features for short-context LLM classifiers."""
    values = sorted(list(events), key=lambda event: (event.relative_minute, event.event_id))
    if not values:
        return "ClinTime-FeatureSummary/1\nGLOBAL|events=0"
    by_concept: dict[str, list[Event]] = defaultdict(list)
    for event in values:
        by_concept[event.concept].append(event)
    rows: list[tuple[tuple[int, int, int, str], str]] = []
    for concept_events in by_concept.values():
        row, priority = _feature_summary_row(concept_events)
        rows.append((priority, row))
    rows.sort(key=lambda item: item[0])
    global_row = (
        f"GLOBAL|events={len(values)}|concepts={len(by_concept)}|span={values[0].relative_minute}-{values[-1].relative_minute}|"
        f"abn={sum(1 for event in values if event.abnormal_flag in {'high', 'low', 'abnormal'})}|"
        f"treat={sum(1 for event in values if event.source in {'medication_admin', 'medication_order', 'procedure', 'treatment'})}"
    )
    lines = ["ClinTime-FeatureSummary/1", global_row]
    used = sum(len(line) + 1 for line in lines)
    for _, row in rows:
        if len(lines) - 2 >= config.max_rows or used + len(row) + 1 > config.max_chars:
            break
        lines.append(row)
        used += len(row) + 1
    return "\n".join(lines)


def extract_event_ids(text: str) -> list[str]:
    result: list[str] = []
    for line in text.splitlines():
        if line.startswith("[") and "]" in line:
            result.append(line[1 : line.index("]")])
    for match in re.finditer(r'evidence_ids="([^"]*)"', text):
        result.extend(value for value in match.group(1).split(",") if value)
    for match in re.finditer(r"\bids=([A-Za-z0-9_,.\-:]+)", text):
        result.extend(value for value in match.group(1).split(",") if value)
    for match in re.finditer(r'"ids":\["([^]]*)"\]', text):
        result.extend(value.strip('"') for value in match.group(1).split('","') if value)
    return result
