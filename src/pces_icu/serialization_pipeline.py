from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from .io import read_jsonl, write_json, write_jsonl
from .schema import PatientRecord
from .serialize import (
    ClinTimeFeatureSummaryConfig,
    ClinTimeJSONConfig,
    ClinTimeXMLConfig,
    MEDSConfig,
    serialize_clintime_feature_summary,
    serialize_clintime_json,
    serialize_clintime_xml,
    serialize_meds_events,
    stable_view_seed,
)


SERIALIZATION_FORMATS = ("cass", "json", "xml", "meds")


def serialize_record(
    record: PatientRecord,
    fmt: str,
    seed: int = 2026,
    include_prompt: bool = False,
    task_prompt: str = "",
) -> str:
    fmt = fmt.lower()
    view_seed = stable_view_seed(record.patient_id, seed)
    events = record.events or []
    if fmt == "cass":
        text = serialize_clintime_feature_summary(events, ClinTimeFeatureSummaryConfig(max_rows=80, max_chars=3_000))
    elif fmt == "json":
        text = serialize_clintime_json(events, ClinTimeJSONConfig(permute_within_blocks=False, max_nodes=160, max_chars=6_000), view_seed)
    elif fmt == "xml":
        text = serialize_clintime_xml(events, ClinTimeXMLConfig(permute_within_blocks=False, max_nodes=128, max_chars=12_000), view_seed)
    elif fmt == "meds":
        text = serialize_meds_events(events, MEDSConfig(max_events=160, max_chars=6_000))
    else:
        raise ValueError(f"Unknown serialization format: {fmt!r}")
    if include_prompt and task_prompt.strip():
        return task_prompt.rstrip() + "\n" + text
    return text


def serialized_rows(
    records: Iterable[PatientRecord],
    fmt: str,
    seed: int = 2026,
    include_prompt: bool = False,
    task_prompt: str = "",
) -> Iterable[dict[str, Any]]:
    for record in records:
        yield {
            "patient_id": record.patient_id,
            "stay_id": record.stay_id,
            "dataset": record.dataset,
            "hospital_id": record.hospital_id,
            "split": record.split,
            "label": int(record.label),
            "format": fmt.lower(),
            "text": serialize_record(record, fmt, seed, include_prompt, task_prompt),
        }


def build_serialization_pipeline(
    events_path: str | Path,
    output_root: str | Path,
    formats: Iterable[str] = SERIALIZATION_FORMATS,
    seed: int = 2026,
    include_prompt: bool = False,
    task_prompt: str = "",
    manifest_name: str = "manifest.json",
) -> dict[str, Any]:
    output_root = Path(output_root)
    selected_formats = tuple(fmt.lower() for fmt in formats)
    unknown = sorted(set(selected_formats) - set(SERIALIZATION_FORMATS))
    if unknown:
        raise ValueError(f"Unknown serialization format(s): {', '.join(unknown)}")

    records = [PatientRecord.from_dict(value) for value in read_jsonl(events_path)]
    outputs: dict[str, dict[str, Any]] = {}
    for fmt in selected_formats:
        output = output_root / f"{fmt}.jsonl.gz"
        count = write_jsonl(output, serialized_rows(records, fmt, seed, include_prompt, task_prompt))
        outputs[fmt] = {"path": str(output), "records": count}

    manifest = {
        "input": str(events_path),
        "output_root": str(output_root),
        "formats": list(selected_formats),
        "seed": seed,
        "include_prompt": include_prompt,
        "task_prompt": task_prompt if include_prompt else "",
        "records": len(records),
        "outputs": outputs,
        "configs": {
            "cass": asdict(ClinTimeFeatureSummaryConfig(max_rows=80, max_chars=3_000)),
            "json": asdict(ClinTimeJSONConfig(permute_within_blocks=False, max_nodes=160, max_chars=6_000)),
            "xml": asdict(ClinTimeXMLConfig(permute_within_blocks=False, max_nodes=128, max_chars=12_000)),
            "meds": asdict(MEDSConfig(max_events=160, max_chars=6_000)),
        },
    }
    write_json(output_root / manifest_name, manifest)
    return manifest
