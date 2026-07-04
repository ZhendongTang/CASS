from __future__ import annotations

import argparse
import json
from pathlib import Path


DATA = "work/events/mimic.jsonl.gz"
PYTHON = "/root/anaconda3/bin/python3"
SEEDS = (42, 43, 44)
FORMATS = ("cass", "json", "meds", "xml")


def audit_job(root: str, source_root: str, fmt: str, seed: int, max_records: int) -> dict:
    checkpoint = f"{source_root}/train/llama/{fmt}_seed{seed}"
    output = f"{root}/permutation/llama/{fmt}_seed{seed}/mimic"
    cmd = (
        "export PYTHONPATH=src TRANSFORMERS_NO_TF=1 USE_TF=0 PYTHONUNBUFFERED=1 "
        "TOKENIZERS_PARALLELISM=false && "
        f"{PYTHON} scripts/audit_table2_raw_row_order.py --data {DATA} "
        f"--checkpoint {checkpoint} --output {output} --dataset mimic --split test "
        f"--max-records {max_records} --permutations 5 --audit-seed 2026 --view-batch-size 16"
    )
    return {
        "id": f"raw_row_llama_{fmt}_s{seed}_mimic",
        "phase": "raw_row_order_audit",
        "cmd": cmd,
        "expected_output": f"{output}/metrics.json",
        "meta": {"llm": "llama", "format": fmt, "seed": seed, "audit": "raw_row_leakage", "max_records": max_records},
    }


def build(root: str, source_root: str, max_records: int) -> dict:
    jobs = [audit_job(root, source_root, fmt, seed, max_records) for fmt in FORMATS for seed in SEEDS]
    return {
        "project": "table2_raw_row_order_500_v1",
        "cwd": "/home/tang/pces_icu",
        "conda": "",
        "gpus": [0],
        "max_parallel": 1,
        "gpu_free_threshold_mib": 500,
        "oom_retry": {"delay": 180, "max_attempts": 2},
        "notes": [
            "Table II raw-row leakage audit: perturb same-timestamp source row order.",
            "CASS uses concept aggregation and canonical priority output.",
            "MEDS/JSON/XML use raw-order-leaky row/node order within same-time or timeblock groups.",
            f"Audit records per checkpoint: {max_records}.",
        ],
        "jobs": jobs,
        "phases": [{"name": "raw_row_order_audit", "jobs": jobs}],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="work/final_v2/table2_raw_row_order_500_v1")
    parser.add_argument("--source-root", default="work/final_v2/table2_format_permutation_v2")
    parser.add_argument("--max-records", type=int, default=500)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build(args.root, args.source_root, args.max_records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2))
    print(args.output)
    print(f"jobs: {len(manifest['jobs'])}")


if __name__ == "__main__":
    main()
