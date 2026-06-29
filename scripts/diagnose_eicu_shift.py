from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from pces_icu.metrics import binary_metrics
from pces_icu.multicenter_llm import _load_tokenizer, _records, _text


def safe_metrics(labels: list[int], probabilities: list[float]) -> dict:
    if not labels:
        return {"n": 0}
    result = {
        "n": len(labels),
        "prevalence": float(np.mean(labels)),
        "mean_probability": float(np.mean(probabilities)),
    }
    if len(set(labels)) == 2:
        result.update(binary_metrics(labels, probabilities))
    else:
        result["brier"] = float(np.mean((np.asarray(probabilities) - np.asarray(labels)) ** 2))
    return result


def quartile_groups(values: np.ndarray) -> tuple[np.ndarray, list[float]]:
    cuts = [float(x) for x in np.quantile(values, [0.25, 0.5, 0.75])]
    return np.digitize(values, cuts, right=True), cuts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--predictions-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-records", type=int, default=500)
    parser.add_argument("--sample-seed", type=int, default=2026)
    parser.add_argument("--token-budget", type=int, default=1024)
    parser.add_argument("--min-hospital-n", type=int, default=20)
    args = parser.parse_args()

    records = _records(args.data, "test", args.max_records, args.sample_seed + 2)
    by_id = {str(record.patient_id): record for record in records}
    event_count = np.asarray([len(record.events or []) for record in records], dtype=float)
    concept_count = np.asarray([len({event.concept for event in record.events or []}) for record in records], dtype=float)
    event_group, event_cuts = quartile_groups(event_count)
    concept_group, concept_cuts = quartile_groups(concept_count)

    tokenizer = _load_tokenizer(args.model)
    token_length = np.asarray([
        len(tokenizer.encode(_text(record, "xml", args.sample_seed), add_special_tokens=True))
        for record in records
    ], dtype=float)
    token_group, token_cuts = quartile_groups(token_length)

    features = {
        str(record.patient_id): {
            "event_count": int(event_count[index]),
            "concept_count": int(concept_count[index]),
            "xml_token_length": int(token_length[index]),
            "event_density_quartile": int(event_group[index] + 1),
            "concept_coverage_quartile": int(concept_group[index] + 1),
            "token_length_quartile": int(token_group[index] + 1),
            "truncated": bool(token_length[index] > args.token_budget),
        }
        for index, record in enumerate(records)
    }

    analyses = {}
    for pred_path in sorted(Path(args.predictions_root).glob("*/evaluation/eicu/predictions.jsonl")):
        run = pred_path.parts[-4]
        rows = [json.loads(line) for line in pred_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        rows = [row for row in rows if str(row["patient_id"]) in by_id]
        run_result = {"overall": safe_metrics([int(r["label"]) for r in rows], [float(r["probability"]) for r in rows])}
        for field in ("event_density_quartile", "concept_coverage_quartile", "token_length_quartile"):
            strata = {}
            for group in range(1, 5):
                selected = [r for r in rows if features[str(r["patient_id"])][field] == group]
                strata[str(group)] = safe_metrics(
                    [int(r["label"]) for r in selected], [float(r["probability"]) for r in selected]
                )
            run_result[field] = strata
        hospitals = defaultdict(list)
        for row in rows:
            hospital = by_id[str(row["patient_id"])].hospital_id
            if hospital:
                hospitals[str(hospital)].append(row)
        per_hospital = {}
        for hospital, selected in hospitals.items():
            labels = [int(r["label"]) for r in selected]
            if len(selected) >= args.min_hospital_n:
                per_hospital[hospital] = safe_metrics(labels, [float(r["probability"]) for r in selected])
        valid_aurocs = [x["auroc"] for x in per_hospital.values() if "auroc" in x and np.isfinite(x["auroc"])]
        run_result["hospital"] = {
            "eligible_hospitals": len(per_hospital),
            "macro_auroc": float(np.mean(valid_aurocs)) if valid_aurocs else None,
            "per_hospital": per_hospital,
        }
        analyses[run] = run_result

    result = {
        "n": len(records),
        "sample_seed": args.sample_seed,
        "token_budget": args.token_budget,
        "feature_summary": {
            "event_count_quartile_cuts": event_cuts,
            "concept_count_quartile_cuts": concept_cuts,
            "xml_token_length_quartile_cuts": token_cuts,
            "truncation_rate": float(np.mean(token_length > args.token_budget)),
        },
        "runs": analyses,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(output), "runs": len(analyses), "n": len(records)}))


if __name__ == "__main__":
    main()
