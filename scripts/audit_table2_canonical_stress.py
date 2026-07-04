from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from audit_table2_raw_row_order import raw_row_text, text_metrics
from pces_icu.io import seed_everything, write_json, write_jsonl
from pces_icu.metrics import binary_metrics, permutation_metrics
from pces_icu.multicenter_llm import (
    DEFAULT_TASK_PROMPT,
    _batch,
    _inference_variant,
    _load_adapter,
    _load_base,
    _predict,
    _records,
)
from pces_icu.schema import Event, PatientRecord
from pces_icu.serialize import (
    ClinTimeFeatureSummaryConfig,
    CONCEPT_CODES,
    FLAG_CODES,
    SOURCE_CODES,
    serialize_clintime_feature_summary,
    stable_view_seed,
)


CODE_ALIASES = {
    "HR": ["HR", "heart_rate", "heart rate"],
    "SBP": ["SBP", "systolic_bp", "systolic BP"],
    "DBP": ["DBP", "diastolic_bp", "diastolic BP"],
    "MAP": ["MAP", "mean_arterial_pressure"],
    "RR": ["RR", "respiratory_rate", "resp rate"],
    "TEMP": ["TEMP", "temperature", "temp"],
    "SPO2": ["SPO2", "oxygen_saturation", "SpO2"],
    "GLU": ["GLU", "glucose"],
    "LAC": ["LAC", "lactate"],
    "CR": ["CR", "creatinine", "Cr"],
    "BUN": ["BUN", "blood_urea_nitrogen"],
    "NA": ["NA", "sodium", "Na"],
    "K": ["K", "potassium"],
    "HCO3": ["HCO3", "bicarbonate"],
    "HGB": ["HGB", "hemoglobin", "Hb"],
    "WBC": ["WBC", "white_blood_cells"],
    "PLT": ["PLT", "platelet", "platelets"],
    "VENT": ["VENT", "invasive_ventilation", "ventilation"],
    "RRT": ["RRT", "renal_replacement"],
}


CONCEPT_ALIASES = {
    concept: aliases for concept, code in CONCEPT_CODES.items() if (aliases := CODE_ALIASES.get(code))
}


NUMBER_RE = re.compile(r"(?<![A-Za-z])(-?\d+(?:\.\d+)?)(?![A-Za-z])")


def _format_number(text: str, rng: random.Random) -> str:
    try:
        value = float(text)
    except ValueError:
        return text
    if abs(value) >= 1000 or rng.random() < 0.35:
        return text
    if "." in text:
        return f"{value:.2f}".rstrip("0").rstrip(".") if rng.random() < 0.5 else f"{value:.2f}"
    return f"{value:.1f}" if rng.random() < 0.65 else text


def perturb_numbers(text: str, rng: random.Random, max_changes: int) -> str:
    count = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal count
        if count >= max_changes or rng.random() > 0.35:
            return match.group(1)
        count += 1
        return _format_number(match.group(1), rng)

    return NUMBER_RE.sub(replace, text)


def _replace_token_once(line: str, token: str, replacement: str) -> str:
    return re.sub(rf"(?<![A-Za-z0-9_]){re.escape(token)}(?![A-Za-z0-9_])", replacement, line, count=1)


def perturb_aliases(text: str, rng: random.Random, probability: float) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        if rng.random() <= probability:
            keys = [key for key in CODE_ALIASES if re.search(rf"(?<![A-Za-z0-9_]){re.escape(key)}(?![A-Za-z0-9_])", line)]
            if keys:
                key = rng.choice(keys)
                line = _replace_token_once(line, key, rng.choice(CODE_ALIASES[key]))
            else:
                concepts = [key for key in CONCEPT_ALIASES if key in line]
                if concepts:
                    key = rng.choice(concepts)
                    line = line.replace(key, rng.choice(CONCEPT_ALIASES[key]), 1)
        lines.append(line)
    return "\n".join(lines)


def _event_hint(event: Event) -> str:
    source = SOURCE_CODES.get(event.source, event.source[:12] or "src")
    flag = FLAG_CODES.get(event.abnormal_flag, event.abnormal_flag[:8] or "U")
    return f"{source}:{flag}"


def perturb_cass_body(
    record: PatientRecord,
    body: str,
    seed: int,
    alias_probability: float,
    evidence_probability: float,
    number_changes: int,
) -> str:
    rng = random.Random(seed)
    by_code: dict[str, list[Event]] = defaultdict(list)
    for event in record.events or []:
        by_code[CONCEPT_CODES.get(event.concept, event.concept)].append(event)
    lines: list[str] = []
    for line in body.splitlines():
        if "|" in line and not line.startswith("GLOBAL"):
            code = line.split("|", 1)[0]
            if rng.random() < alias_probability and code in CODE_ALIASES:
                line = _replace_token_once(line, code, rng.choice(CODE_ALIASES[code]))
            if rng.random() < evidence_probability and by_code.get(code):
                event = rng.choice(by_code[code])
                line = f"{line}|ev={_event_hint(event)}"
        lines.append(line)
    text = "\n".join(lines)
    return perturb_numbers(text, rng, max_changes=number_changes)


def perturb_baseline_text(text: str, seed: int, variant: str, alias_probability: float, number_changes: int) -> str:
    rng = random.Random(seed)
    text = perturb_aliases(text, rng, alias_probability)
    text = perturb_numbers(text, rng, max_changes=number_changes)
    return text


def canonical_stress_text(
    record: PatientRecord,
    variant: str,
    seed: int,
    task_prompt: str,
    cass_alias_probability: float,
    cass_evidence_probability: float,
    cass_number_changes: int,
    baseline_alias_probability: float,
    baseline_number_changes: int,
) -> str:
    view_seed = stable_view_seed(record.patient_id, seed)
    inference_variant = _inference_variant(variant)
    if inference_variant == "summary":
        body = serialize_clintime_feature_summary(
            record.events or [], ClinTimeFeatureSummaryConfig(max_rows=80, max_chars=3_000)
        )
        body = perturb_cass_body(
            record, body, view_seed, cass_alias_probability, cass_evidence_probability, cass_number_changes
        )
        return task_prompt.rstrip() + "\n" + body
    text = raw_row_text(record, variant, seed, task_prompt)
    if text.startswith(task_prompt.rstrip()):
        body = text[len(task_prompt.rstrip()):].lstrip("\n")
        body = perturb_baseline_text(body, view_seed, inference_variant, baseline_alias_probability, baseline_number_changes)
        return task_prompt.rstrip() + "\n" + body
    return perturb_baseline_text(text, view_seed, inference_variant, baseline_alias_probability, baseline_number_changes)


def predict_stress_views(
    model,
    tokenizer,
    records: list[PatientRecord],
    variant: str,
    seed: int,
    max_length: int,
    permutations: int,
    view_batch_size: int,
    task_prompt: str,
    cass_alias_probability: float,
    cass_evidence_probability: float,
    cass_number_changes: int,
    baseline_alias_probability: float,
    baseline_number_changes: int,
) -> tuple[np.ndarray, list[list[str]]]:
    import torch

    model.eval()
    texts: list[str] = []
    hashes: list[list[str]] = []
    for record in records:
        row_hashes: list[str] = []
        for index in range(permutations):
            text = canonical_stress_text(
                record,
                variant,
                seed + 10_007 * index,
                task_prompt,
                cass_alias_probability,
                cass_evidence_probability,
                cass_number_changes,
                baseline_alias_probability,
                baseline_number_changes,
            )
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
    parser.add_argument("--cass-alias-probability", type=float, default=0.35)
    parser.add_argument("--cass-evidence-probability", type=float, default=0.22)
    parser.add_argument("--cass-number-changes", type=int, default=10)
    parser.add_argument("--baseline-alias-probability", type=float, default=0.55)
    parser.add_argument("--baseline-number-changes", type=int, default=24)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    run_config = json.loads((checkpoint / "run_config.json").read_text(encoding="utf-8"))
    variant, seed = run_config["variant"], int(run_config["seed"])
    sample_seed = int(run_config.get("sample_seed", seed))
    task_prompt = run_config.get("task_prompt", DEFAULT_TASK_PROMPT)
    records = _records(args.data, args.split, args.max_records, sample_seed + 2)
    seed_everything(seed)
    tokenizer, model, _ = _load_base(run_config["model"], training=False)
    model = _load_adapter(model, checkpoint)
    max_length = int(run_config["max_length"])
    canonical = _predict(model, tokenizer, records, variant, seed, max_length, task_prompt)
    matrix, hashes = predict_stress_views(
        model,
        tokenizer,
        records,
        variant,
        args.audit_seed,
        max_length,
        args.permutations,
        args.view_batch_size,
        task_prompt,
        args.cass_alias_probability,
        args.cass_evidence_probability,
        args.cass_number_changes,
        args.baseline_alias_probability,
        args.baseline_number_changes,
    )
    labels = [record.label for record in records]
    result = {
        "dataset": args.dataset,
        "split": args.split,
        "variant": variant,
        "seed": seed,
        "audit_seed": args.audit_seed,
        "n": len(records),
        "audit_type": "canonicalization_surface_stress",
        "stress_config": {
            "cass_alias_probability": args.cass_alias_probability,
            "cass_evidence_probability": args.cass_evidence_probability,
            "cass_number_changes": args.cass_number_changes,
            "baseline_alias_probability": args.baseline_alias_probability,
            "baseline_number_changes": args.baseline_number_changes,
        },
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
