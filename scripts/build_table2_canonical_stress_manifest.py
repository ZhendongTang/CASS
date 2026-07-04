from __future__ import annotations

import argparse
import json
from pathlib import Path


DATA = "work/events/mimic.jsonl.gz"
PYTHON = "/root/anaconda3/bin/python3"
SEEDS = (42, 43, 44)
FORMATS = ("cass", "json", "meds", "xml")


def audit_job(root: str, source_root: str, fmt: str, seed: int, max_records: int, stress_args: str) -> dict:
    checkpoint = f"{source_root}/train/llama/{fmt}_seed{seed}"
    output = f"{root}/permutation/llama/{fmt}_seed{seed}/mimic"
    cmd = (
        "export PYTHONPATH=src TRANSFORMERS_NO_TF=1 USE_TF=0 PYTHONUNBUFFERED=1 "
        "TOKENIZERS_PARALLELISM=false && "
        f"{PYTHON} scripts/audit_table2_canonical_stress.py --data {DATA} "
        f"--checkpoint {checkpoint} --output {output} --dataset mimic --split test "
        f"--max-records {max_records} --permutations 5 --audit-seed 2026 --view-batch-size 16 "
        f"{stress_args}"
    )
    return {
        "id": f"canonical_stress_llama_{fmt}_s{seed}_mimic",
        "phase": "canonical_surface_stress_audit",
        "cmd": cmd,
        "expected_output": f"{output}/metrics.json",
        "meta": {
            "llm": "llama",
            "format": fmt,
            "seed": seed,
            "audit": "canonicalization_surface_stress",
            "max_records": max_records,
        },
    }


def build(root: str, source_root: str, max_records: int, project: str, stress_args: str) -> dict:
    jobs = [audit_job(root, source_root, fmt, seed, max_records, stress_args) for fmt in FORMATS for seed in SEEDS]
    return {
        "project": project,
        "cwd": "/home/tang/pces_icu",
        "conda": "",
        "gpus": [0],
        "max_parallel": 1,
        "gpu_free_threshold_mib": 500,
        "oom_retry": {"delay": 180, "max_attempts": 2},
        "notes": [
            "Table II canonicalization stress audit: legal surface variants with 5 views per record.",
            "CASS retains concept aggregation but receives local alias, numeric-format, and evidence-hint changes.",
            "MEDS/JSON/XML additionally expose raw row or node ordering plus the same surface-variant family.",
            f"Audit records per checkpoint: {max_records}.",
        ],
        "jobs": jobs,
        "phases": [{"name": "canonical_surface_stress_audit", "jobs": jobs}],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="work/final_v2/table2_canonical_stress_500_v1")
    parser.add_argument("--project", default="table2_canonical_stress_500_v1")
    parser.add_argument("--source-root", default="work/final_v2/table2_format_permutation_v2")
    parser.add_argument("--max-records", type=int, default=500)
    parser.add_argument(
        "--stress-args",
        default=(
            "--cass-alias-probability 0.35 --cass-evidence-probability 0.22 "
            "--cass-number-changes 10 --baseline-alias-probability 0.55 --baseline-number-changes 24"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build(args.root, args.source_root, args.max_records, args.project, args.stress_args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(args.output)
    print(f"jobs: {len(manifest['jobs'])}")


if __name__ == "__main__":
    main()
