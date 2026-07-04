from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from pces_icu.multicenter_llm import SYSTEM_PREFIX, _load_tokenizer, _records, _text
from pces_icu.serialize import ClinTimeCompactConfig, ClinTimeJSONConfig, serialize_clintime_compact, serialize_clintime_json
from pces_icu.serialize import extract_event_ids


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-records", type=int, default=1000)
    parser.add_argument("--sample-seed", type=int, default=2026)
    parser.add_argument("--budgets", type=int, nargs="+", default=[256, 512, 1024])
    args = parser.parse_args()

    tokenizer = _load_tokenizer(args.model)
    records = _records(args.data, args.split, args.max_records, args.sample_seed + 2)
    result: dict[str, object] = {
        "dataset": args.dataset,
        "model": args.model,
        "split": args.split,
        "sample_seed": args.sample_seed,
        "n": len(records),
        "prevalence": float(np.mean([record.label for record in records])),
        "event_count": _summary([len(record.events or []) for record in records]),
        "variants": {},
    }
    for variant in ("flat", "meds", "xml", "compact", "compact_trace", "json", "json_trace", "summary"):
        lengths: list[int] = []
        full_evidence: list[set[str]] = []
        encoded_rows: list[list[int]] = []
        for record in records:
            if variant == "compact_trace":
                text = SYSTEM_PREFIX + serialize_clintime_compact(
                    record.events or [],
                    ClinTimeCompactConfig(include_evidence_ids=True, include_legend=True),
                    args.sample_seed,
                )
            elif variant == "json_trace":
                text = SYSTEM_PREFIX + serialize_clintime_json(
                    record.events or [],
                    ClinTimeJSONConfig(include_evidence_ids=True),
                    args.sample_seed,
                )
            else:
                text = _text(record, variant, args.sample_seed)
            encoded = tokenizer.encode(text, add_special_tokens=True)
            encoded_rows.append(encoded)
            lengths.append(len(encoded))
            full_evidence.append(set(extract_event_ids(text)))
        variant_result: dict[str, object] = {"token_length": _summary(lengths), "budgets": {}}
        for budget in args.budgets:
            retention: list[float] = []
            for ids, evidence in zip(encoded_rows, full_evidence):
                decoded = tokenizer.decode(ids[:budget], skip_special_tokens=False)
                kept = set(extract_event_ids(decoded))
                if evidence:
                    retention.append(len(kept & evidence) / len(evidence))
            variant_result["budgets"][str(budget)] = {
                "truncation_rate": float(np.mean(np.asarray(lengths) > budget)),
                "mean_evidence_retention": float(np.mean(retention)) if retention else None,
                "p10_evidence_retention": float(np.quantile(retention, 0.10)) if retention else None,
            }
        result["variants"][variant] = variant_result

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
