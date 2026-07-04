from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TORCH", "1")

from pces_icu.multicenter_llm import _load_tokenizer, _records, _text
from pces_icu.serialize import extract_event_ids


TASK_PROMPTS = {
    "mortality": (
        "Task: predict in-hospital mortality from observations recorded during the first 6 ICU hours. "
        "Use only the supplied events. Return a mortality risk score.\n"
    ),
    "los": (
        "Task: predict whether ICU length of stay exceeds 48 hours from observations recorded during "
        "the first 6 ICU hours. Use only the supplied events. Return a risk score.\n"
    ),
    "future": (
        "Task: predict whether a new major intervention starts between 6 and 24 ICU hours from "
        "observations recorded during the first 6 ICU hours. Exclude interventions already present "
        "in the input. Use only the supplied events. Return a risk score.\n"
    ),
}


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    if not len(array):
        return {"mean": math.nan, "p50": math.nan, "p90": math.nan, "p95": math.nan, "p99": math.nan}
    return {
        "mean": float(array.mean()),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
    }


def _digest(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()


def _jaccard(left: set[str], right: set[str]) -> float | None:
    if not left and not right:
        return None
    union = left | right
    return len(left & right) / len(union) if union else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--task", choices=sorted(TASK_PROMPTS), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-records", type=int, default=500)
    parser.add_argument("--sample-seed", type=int, default=2026)
    parser.add_argument("--views", type=int, default=5)
    parser.add_argument("--budget", type=int, default=2048)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["flat", "xml", "meds", "json", "compact", "summary"],
        help="Serializer variants to audit. The paper's CASS row is the summary variant.",
    )
    args = parser.parse_args()

    tokenizer = _load_tokenizer(args.model)
    records = _records(args.data, args.split, args.max_records, args.sample_seed + 17)
    prompt = TASK_PROMPTS[args.task]

    result: dict[str, object] = {
        "dataset": args.dataset,
        "task": args.task,
        "model": args.model,
        "split": args.split,
        "sample_seed": args.sample_seed,
        "views": args.views,
        "budget": args.budget,
        "n": len(records),
        "prevalence": float(np.mean([record.label for record in records])) if records else math.nan,
        "variants": {},
    }

    for variant in args.variants:
        token_lengths: list[int] = []
        exact_rates: list[float] = []
        unique_hashes: list[float] = []
        jaccards: list[float] = []
        evidence_available = 0

        for record in records:
            canonical = _text(record, variant, args.sample_seed, force_permutation=False, task_prompt=prompt)
            canonical_hash = _digest(canonical)
            canonical_ids = set(extract_event_ids(canonical))
            token_lengths.append(len(tokenizer.encode(canonical, add_special_tokens=True)))

            view_hashes = []
            matches = []
            for index in range(args.views):
                view = _text(
                    record,
                    variant,
                    args.sample_seed + 10_007 * (index + 1),
                    force_permutation=True,
                    task_prompt=prompt,
                )
                view_hash = _digest(view)
                view_hashes.append(view_hash)
                matches.append(float(view_hash == canonical_hash))
                view_ids = set(extract_event_ids(view))
                jac = _jaccard(canonical_ids, view_ids)
                if jac is not None:
                    jaccards.append(float(jac))
            if canonical_ids:
                evidence_available += 1
            exact_rates.append(float(np.mean(matches)))
            unique_hashes.append(float(len(set([canonical_hash] + view_hashes))))

        variant_result = {
            "token_length": _summary(token_lengths),
            "truncation_rate_at_budget": float(np.mean(np.asarray(token_lengths) > args.budget)) if token_lengths else math.nan,
            "exact_match_rate_under_legal_reorder": float(np.mean(exact_rates)) if exact_rates else math.nan,
            "mean_unique_hashes_per_record": float(np.mean(unique_hashes)) if unique_hashes else math.nan,
            "evidence_jaccard_under_legal_reorder": float(np.mean(jaccards)) if jaccards else None,
            "records_with_traceable_evidence": int(evidence_available),
        }
        result["variants"][variant] = variant_result

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
