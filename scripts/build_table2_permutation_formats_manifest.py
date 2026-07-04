from __future__ import annotations

import argparse
import json
from pathlib import Path


DATA = "work/events/mimic.jsonl.gz"
MODEL = "/data/models/Meta-Llama-3-8B"
PYTHON = "/root/anaconda3/bin/python3"
VARIANTS = {
    "cass": "summary",
    "json": "json",
    "meds": "meds",
    "xml": "xml",
}
SEEDS = (42, 43, 44)
PREFIX = (
    "export PYTHONPATH=src TRANSFORMERS_NO_TF=1 USE_TF=0 PYTHONUNBUFFERED=1 "
    "TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && "
)


def train_job(root: str, display: str, variant: str, seed: int) -> dict[str, object]:
    output = f"{root}/train/llama/{display}_seed{seed}"
    cmd = (
        f"{PYTHON} -m pces_icu.multicenter_llm train --data {DATA} --model {MODEL} "
        f"--output {output} --variant {variant} --seed {seed} --sample-seed 2026 "
        "--epochs 1 --max-length 256 --grad-accum 8 --lr 2e-5 --auto-pos-weight "
        "--kl-weight 0.2 --max-train-records 1000 --max-val-records 256 "
        f"--eval-data mimic={DATA} --max-eval-records 500"
    )
    return {
        "id": f"train_llama_{display}_s{seed}",
        "cmd": PREFIX + cmd,
        "expected_output": f"{output}/train_metrics.json",
        "meta": {"llm": "llama", "format": display, "variant": variant, "seed": seed},
    }


def audit_job(root: str, display: str, seed: int) -> dict[str, object]:
    checkpoint = f"{root}/train/llama/{display}_seed{seed}"
    output = f"{root}/permutation/llama/{display}_seed{seed}/mimic"
    cmd = (
        f"{PYTHON} -m pces_icu.multicenter_llm audit-permutation --data {DATA} "
        f"--checkpoint {checkpoint} --output {output} --dataset mimic --split test "
        "--max-records 500 --permutations 5 --audit-seed 2026 --view-batch-size 1"
    )
    return {
        "id": f"audit_llama_{display}_s{seed}_mimic",
        "cmd": PREFIX + cmd,
        "expected_output": f"{output}/metrics.json",
        "meta": {"llm": "llama", "format": display, "seed": seed, "dataset": "mimic"},
    }


def build(root: str) -> dict[str, object]:
    train_jobs = [
        train_job(root, display, variant, seed)
        for display, variant in VARIANTS.items()
        for seed in SEEDS
    ]
    audit_jobs = [
        audit_job(root, display, seed)
        for display in VARIANTS
        for seed in SEEDS
    ]
    return {
        "project": "table2_llama_format_permutation_v2",
        "cwd": "/home/tang/pces_icu",
        "conda": "",
        "conda_hook": "",
        "gpus": [0],
        "max_parallel": 1,
        "gpu_free_threshold_mib": 500,
        "oom_retry": {"delay": 180, "max_attempts": 2},
        "other_retry": {"delay": 120, "max_attempts": 2},
        "notes": [
            "Rebuild Table II as row-order prediction-sensitivity diagnostics.",
            "Formats: CASS(summary), JSON, MEDS-like, XML.",
            "Backbone: Llama-3-8B; data: MIMIC-IV mortality; seeds: 42,43,44.",
            "Training: 1000 records, max_length=256. Audit: 500 held-out MIMIC records, 5 legal reorder views.",
        ],
        "phases": [
            {"name": "train_short_context_formats", "depends_on": [], "jobs": train_jobs},
            {"name": "audit_permutation_formats", "depends_on": ["train_short_context_formats"], "jobs": audit_jobs},
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--root", default="work/final_v2/table2_format_permutation_v2")
    args = parser.parse_args()

    manifest = build(args.root)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    jobs = sum(len(phase["jobs"]) for phase in manifest["phases"])
    print(json.dumps({"output": str(path), "jobs": jobs}, ensure_ascii=False))


if __name__ == "__main__":
    main()
