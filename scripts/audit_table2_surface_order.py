from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from pces_icu.io import write_json, write_jsonl
from pces_icu.metrics import binary_metrics, permutation_metrics
from pces_icu.multicenter_llm import (
    DEFAULT_TASK_PROMPT,
    SUMMARY_VARIANTS,
    _batch,
    _inference_variant,
    _load_base,
    _predict,
    _records,
    _text,
)
from pces_icu.serialize import stable_view_seed


def _shuffle_cass_rows(text: str, seed: int) -> str:
    lines = text.splitlines()
    if len(lines) <= 3:
        return text
    header = lines[:2]
    rows = lines[2:]
    rng = random.Random(seed)
    rng.shuffle(rows)
    return "\n".join(header + rows)


def _shuffle_meds_same_time(text: str, seed: int) -> str:
    lines = text.splitlines()
    if len(lines) <= 3:
        return text
    header = lines[:2]
    groups: list[tuple[str, list[str]]] = []
    for row in lines[2:]:
        time_key = row.split("|", 1)[0]
        if groups and groups[-1][0] == time_key:
            groups[-1][1].append(row)
        else:
            groups.append((time_key, [row]))
    rng = random.Random(seed)
    shuffled: list[str] = []
    for _, rows in groups:
        rows = list(rows)
        rng.shuffle(rows)
        shuffled.extend(rows)
    return "\n".join(header + shuffled)


def surface_text(record, variant: str, seed: int, task_prompt: str) -> str:
    view_seed = stable_view_seed(record.patient_id, seed)
    inference_variant = _inference_variant(variant)
    if inference_variant in SUMMARY_VARIANTS:
        canonical = _text(record, inference_variant, seed, task_prompt=task_prompt)
        return _shuffle_cass_rows(canonical, view_seed)
    if inference_variant == "meds":
        canonical = _text(record, inference_variant, seed, task_prompt=task_prompt)
        return _shuffle_meds_same_time(canonical, view_seed)
    return _text(record, inference_variant, seed, force_permutation=True, task_prompt=task_prompt)


def predict_surface_permutations(
    model,
    tokenizer,
    records,
    variant: str,
    seed: int,
    max_length: int,
    permutations: int,
    view_batch_size: int,
    task_prompt: str,
) -> np.ndarray:
    import torch

    if permutations < 2:
        raise ValueError("permutations must be at least 2")
    model.eval()
    texts: list[str] = []
    with torch.inference_mode():
        for record in records:
            texts.extend(
                surface_text(record, variant, seed + 10_007 * index, task_prompt)
                for index in range(permutations)
            )
        probabilities: list[float] = []
        batch_size = max(1, view_batch_size)
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            logits = model(**_batch(tokenizer, batch, max_length)).logits.view(-1)
            probabilities.extend(torch.sigmoid(logits.float()).detach().cpu().tolist())
    return np.asarray(probabilities, dtype=float).reshape(len(records), permutations)


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
    parser.add_argument("--view-batch-size", type=int, default=1)
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
    matrix = predict_surface_permutations(
        model,
        tokenizer,
        records,
        variant,
        args.audit_seed,
        max_length,
        args.permutations,
        args.view_batch_size,
        task_prompt,
    )
    labels = [record.label for record in records]
    result = {
        "dataset": args.dataset,
        "split": args.split,
        "variant": variant,
        "seed": seed,
        "audit_seed": args.audit_seed,
        "n": len(records),
        "audit_type": "surface_order_stress",
        "canonical": binary_metrics(labels, canonical),
        "permutation_ensemble": binary_metrics(labels, matrix.mean(axis=1)),
        "permutation": permutation_metrics(matrix),
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
                "canonical_probability": float(probability),
                "permutation_probabilities": [float(value) for value in values],
            }
            for record, probability, values in zip(records, canonical, matrix)
        ),
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
