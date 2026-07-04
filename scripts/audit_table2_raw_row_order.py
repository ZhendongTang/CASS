from __future__ import annotations

import argparse
import hashlib
import html
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from pces_icu.io import write_json, write_jsonl
from pces_icu.metrics import binary_metrics, permutation_metrics
from pces_icu.multicenter_llm import (
    DEFAULT_TASK_PROMPT,
    _batch,
    _inference_variant,
    _load_base,
    _predict,
    _records,
    _text,
)
from pces_icu.schema import Event, PatientRecord, safe_float
from pces_icu.serialize import (
    ClinTimeFeatureSummaryConfig,
    ClinTimeJSONConfig,
    ClinTimeXMLConfig,
    MEDSConfig,
    _clean,
    _compact_number,
    _json_summary_node,
    _summary_node,
    select_matched_events,
    serialize_clintime_feature_summary,
    stable_view_seed,
)


def permute_raw_same_time(events: list[Event], seed: int) -> list[Event]:
    rng = random.Random(seed)
    grouped: dict[int, list[Event]] = defaultdict(list)
    for event in events:
        grouped[event.relative_minute].append(event)
    result: list[Event] = []
    for minute in sorted(grouped):
        current = list(grouped[minute])
        rng.shuffle(current)
        result.extend(current)
    return result


def with_events(record: PatientRecord, events: list[Event]) -> PatientRecord:
    return PatientRecord(
        patient_id=record.patient_id,
        stay_id=record.stay_id,
        label=record.label,
        dataset=record.dataset,
        hospital_id=record.hospital_id,
        split=record.split,
        events=events,
    )


def serialize_meds_raw_order(events: list[Event], config: MEDSConfig) -> str:
    selected_ids = {event.event_id for event in select_matched_events(events, config.max_events)}
    grouped: dict[int, list[Event]] = defaultdict(list)
    for event in events:
        if event.event_id in selected_ids:
            grouped[event.relative_minute].append(event)
    lines = [
        "MEDS-like event table; rows are time-ordered observations.",
        "time_min|code|numeric_value|text_value|unit|source|flag",
    ]
    used = sum(len(line) + 1 for line in lines)
    for minute in sorted(grouped):
        for event in grouped[minute]:
            value = event.canonical_value if event.canonical_value is not None else event.raw_value
            numeric = safe_float(value)
            numeric_value = _compact_number(numeric) if numeric is not None else ""
            text_value = "" if numeric is not None or value is None else _clean(value)[:32]
            row = (
                f"{event.relative_minute}|{_clean(event.concept)}|{numeric_value}|{text_value}|"
                f"{_clean(event.unit)[:16]}|{_clean(event.source)}|{_clean(event.abnormal_flag)}"
            )
            if used + len(row) + 1 > config.max_chars:
                return "\n".join(lines)
            lines.append(row)
            used += len(row) + 1
    return "\n".join(lines)


def _first_position(events: list[Event], bucket_minutes: int) -> dict[tuple[int, str], int]:
    result: dict[tuple[int, str], int] = {}
    for index, event in enumerate(events):
        bucket = max(0, event.relative_minute) // bucket_minutes
        result.setdefault((bucket, event.concept), index)
    return result


def serialize_xml_raw_order(events: list[Event], config: ClinTimeXMLConfig) -> str:
    positions = _first_position(events, config.bucket_minutes)
    buckets: dict[tuple[int, str], list[Event]] = defaultdict(list)
    for event in events:
        bucket = max(0, event.relative_minute) // config.bucket_minutes
        buckets[(bucket, event.concept)].append(event)
    candidates: list[tuple[tuple[int, int, int, str], int, str, int]] = []
    for (bucket, concept), values in buckets.items():
        node, _, priority = _summary_node(values)
        candidates.append((priority, bucket, node, positions[(bucket, concept)]))
    candidates.sort(key=lambda item: item[0])
    selected: list[tuple[int, str, int]] = []
    used_chars = 1_600
    for _, bucket, node, position in candidates:
        if len(selected) >= config.max_nodes or used_chars + len(node) + 1 > config.max_chars:
            continue
        selected.append((bucket, node, position))
        used_chars += len(node) + 1
    grouped: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for bucket, node, position in selected:
        grouped[bucket].append((position, node))
    lines = [
        '<icu_stream schema="ClinTime-XML/1.0" window_min="0" window_max="360">',
        '<schema_note>timeblocks are ordered; observations within each timeblock are an unordered set; each observation atomically binds concept, values, unit, abnormality, and source event IDs</schema_note>',
    ]
    previous_start = 0
    for bucket in sorted(grouped):
        start = bucket * config.bucket_minutes
        end = min(360, start + config.bucket_minutes - 1)
        lines.append(
            f'<timeblock start_min="{start}" end_min="{end}" delta_from_previous="{start - previous_start}" semantics="unordered">'
        )
        lines.extend(node for _, node in sorted(grouped[bucket], key=lambda item: item[0]))
        lines.append("</timeblock>")
        previous_start = start
    lines.append("</icu_stream>")
    return "\n".join(lines)


def serialize_json_raw_order(events: list[Event], config: ClinTimeJSONConfig) -> str:
    positions = _first_position(events, config.bucket_minutes)
    buckets: dict[tuple[int, str], list[Event]] = defaultdict(list)
    for event in events:
        bucket = max(0, event.relative_minute) // config.bucket_minutes
        buckets[(bucket, event.concept)].append(event)
    candidates: list[tuple[tuple[int, int, int, str], int, dict[str, object], int]] = []
    for (bucket, concept), values in buckets.items():
        node, priority = _json_summary_node(values, config.include_evidence_ids)
        candidates.append((priority, bucket, node, positions[(bucket, concept)]))
    candidates.sort(key=lambda item: item[0])
    selected: list[tuple[int, dict[str, object], int]] = []
    for _, bucket, node, position in candidates:
        if len(selected) >= config.max_nodes:
            continue
        selected.append((bucket, node, position))
        grouped_preview: dict[int, list[dict[str, object]]] = defaultdict(list)
        for b, n, _ in selected:
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
    grouped: dict[int, list[tuple[int, dict[str, object]]]] = defaultdict(list)
    for bucket, node, position in selected:
        grouped[bucket].append((position, node))
    blocks = []
    for bucket in sorted(grouped):
        start = bucket * config.bucket_minutes
        nodes = [node for _, node in sorted(grouped[bucket], key=lambda item: item[0])]
        blocks.append({"t": [start, min(360, start + config.bucket_minutes - 1)], "x": nodes})
    return json.dumps(
        {"fmt": "CTJ1", "ord": "time", "set": "bucket_rows_unordered", "b": blocks},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def raw_row_text(record: PatientRecord, variant: str, seed: int, task_prompt: str) -> str:
    events = permute_raw_same_time(record.events or [], stable_view_seed(record.patient_id, seed))
    inference_variant = _inference_variant(variant)
    if inference_variant == "summary":
        body = serialize_clintime_feature_summary(events, ClinTimeFeatureSummaryConfig(max_rows=80, max_chars=3_000))
    elif inference_variant == "meds":
        body = serialize_meds_raw_order(events, MEDSConfig(max_events=160, max_chars=6_000))
    elif inference_variant == "json":
        body = serialize_json_raw_order(events, ClinTimeJSONConfig(permute_within_blocks=False, max_nodes=160, max_chars=6_000))
    elif inference_variant == "xml":
        body = serialize_xml_raw_order(events, ClinTimeXMLConfig(permute_within_blocks=False, max_nodes=128, max_chars=12_000))
    else:
        body = _text(with_events(record, events), inference_variant, seed, force_permutation=False, task_prompt="").strip()
    return task_prompt.rstrip() + "\n" + body


def predict_raw_permutations(model, tokenizer, records, variant: str, seed: int, max_length: int, permutations: int, view_batch_size: int, task_prompt: str) -> tuple[np.ndarray, list[list[str]]]:
    import torch

    model.eval()
    texts: list[str] = []
    hashes: list[list[str]] = []
    for record in records:
        row_hashes: list[str] = []
        for index in range(permutations):
            text = raw_row_text(record, variant, seed + 10_007 * index, task_prompt)
            texts.append(text)
            row_hashes.append(hashlib.sha256(text.encode("utf-8")).hexdigest())
        hashes.append(row_hashes)
    probabilities: list[float] = []
    with torch.inference_mode():
        for start in range(0, len(texts), max(1, view_batch_size)):
            batch = texts[start:start + max(1, view_batch_size)]
            logits = model(**_batch(tokenizer, batch, max_length)).logits.view(-1)
            probabilities.extend(torch.sigmoid(logits.float()).detach().cpu().tolist())
    return np.asarray(probabilities, dtype=float).reshape(len(records), permutations), hashes


def text_metrics(hashes: list[list[str]]) -> dict[str, float]:
    unique_counts = [len(set(row)) for row in hashes]
    exact = [int(len(set(row)) == 1) for row in hashes]
    return {
        "mean_unique_hashes": float(np.mean(unique_counts)),
        "exact_match_rate": float(np.mean(exact)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", default="mimic")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-records", type=int, default=500)
    parser.add_argument("--permutations", type=int, default=5)
    parser.add_argument("--audit-seed", type=int, default=2026)
    parser.add_argument("--view-batch-size", type=int, default=16)
    args = parser.parse_args()

    from peft import PeftModel

    checkpoint = Path(args.checkpoint)
    run_config = json.loads((checkpoint / "run_config.json").read_text(encoding="utf-8"))
    variant, seed = run_config["variant"], int(run_config["seed"])
    sample_seed = int(run_config.get("sample_seed", seed))
    task_prompt = run_config.get("task_prompt", DEFAULT_TASK_PROMPT)
    records = _records(args.data, args.split, args.max_records, sample_seed + 2)
    tokenizer, model, _ = _load_base(run_config["model"], training=False)
    model = PeftModel.from_pretrained(model, checkpoint / "adapter")
    max_length = int(run_config["max_length"])
    canonical = _predict(model, tokenizer, records, variant, seed, max_length, task_prompt)
    matrix, hashes = predict_raw_permutations(
        model, tokenizer, records, variant, args.audit_seed, max_length, args.permutations, args.view_batch_size, task_prompt
    )
    labels = [record.label for record in records]
    result = {
        "dataset": args.dataset,
        "split": args.split,
        "variant": variant,
        "seed": seed,
        "audit_seed": args.audit_seed,
        "n": len(records),
        "audit_type": "raw_row_leakage",
        "canonical": binary_metrics(labels, canonical),
        "permutation_ensemble": binary_metrics(labels, matrix.mean(axis=1)),
        "permutation": permutation_metrics(matrix),
        "text": text_metrics(hashes),
        "mean_abs_canonical_ensemble_gap": float(np.abs(canonical - matrix.mean(axis=1)).mean()),
    }
    output = Path(args.output)
    write_json(output / "metrics.json", result)
    np.save(output / "permutation_probabilities.npy", matrix)
    write_jsonl(
        output / "predictions.jsonl",
        (
            {
                "patient_id": record.patient_id,
                "stay_id": record.stay_id,
                "label": record.label,
                "canonical_probability": float(canonical[index]),
                "permutation_probabilities": [float(value) for value in matrix[index]],
                "text_hashes": hashes[index],
            }
            for index, record in enumerate(records)
        ),
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
