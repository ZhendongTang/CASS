from __future__ import annotations

import argparse
import json
from pathlib import Path


MODELS = {
    "glm": "/data/models/glm-4-9b-chat",
    "llama": "/data/models/Meta-Llama-3-8B",
    "gemma": "/data/models/gemma-2-9b",
}
DATA = {
    "mimic": "work/events/mimic.jsonl.gz",
    "eicu": "work/events/eicu.jsonl.gz",
    "nwicu": "work/events/nwicu.jsonl.gz",
}
VARIANTS = ("flat", "xml", "xml_aug", "xml_kl")
SEEDS = (42, 43, 44)
PREFIX = (
    "source .venv/bin/activate && export PYTHONPATH=src TRANSFORMERS_NO_TF=1 USE_TF=0 "
    "PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false "
    "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && "
)


def job(job_id: str, command: str, expected: str) -> dict[str, str]:
    return {"id": job_id, "cmd": PREFIX + command, "expected_output": expected}


def token_jobs(root: str) -> list[dict[str, str]]:
    jobs = []
    for backbone, model in MODELS.items():
        for dataset, data in DATA.items():
            output = f"{root}/token_audit/{backbone}/{dataset}.json"
            command = (
                f"python scripts/audit_token_lengths.py --data {data} --dataset {dataset} "
                f"--model {model} --output {output} --max-records 1000 "
                "--sample-seed 2026 --budgets 256 512 1024"
            )
            jobs.append(job(f"token_{backbone}_{dataset}", command, output))
    return jobs


def training_jobs(root: str) -> list[dict[str, str]]:
    jobs = []
    evaluations = " ".join(f"--eval-data {name}={path}" for name, path in DATA.items())
    for backbone, model in MODELS.items():
        for seed in SEEDS:
            for variant in VARIANTS:
                output = f"{root}/main/{backbone}/{variant}_seed{seed}"
                command = (
                    f"python -m pces_icu.multicenter_llm train --data {DATA['mimic']} --model {model} "
                    f"--output {output} --variant {variant} --seed {seed} --sample-seed 2026 "
                    "--epochs 1 --max-length 256 --grad-accum 8 --lr 2e-5 --kl-weight 0.2 "
                    f"--max-train-records 1000 --max-val-records 256 {evaluations} --max-eval-records 500"
                )
                jobs.append(job(f"train_{backbone}_{variant}_s{seed}", command, f"{output}/train_metrics.json"))
    return jobs


def audit_jobs(root: str) -> list[dict[str, str]]:
    jobs = []
    for backbone in MODELS:
        for seed in SEEDS:
            for variant in VARIANTS:
                checkpoint = f"{root}/main/{backbone}/{variant}_seed{seed}"
                for dataset, data in DATA.items():
                    output = f"{root}/permutation/{backbone}/{variant}_seed{seed}/{dataset}"
                    command = (
                        f"python -m pces_icu.multicenter_llm audit-permutation --data {data} "
                        f"--checkpoint {checkpoint} --output {output} --dataset {dataset} "
                        "--max-records 500 --permutations 5 --audit-seed 2026 --view-batch-size 1"
                    )
                    jobs.append(job(
                        f"audit_{backbone}_{variant}_s{seed}_{dataset}", command, f"{output}/metrics.json"
                    ))
    return jobs


def transformer_jobs(root: str) -> list[dict[str, str]]:
    jobs = []
    for seed in SEEDS:
        output = f"{root}/baselines/event_transformer_seed{seed}"
        config = f"configs/event_transformer_seed{seed}.json"
        command = (
            f"pces-icu train --trainer transformer --data {DATA['mimic']} --config {config} --output-dir {output} && "
            f"pces-icu evaluate-checkpoint --trainer transformer --data {DATA['eicu']} "
            f"--model-dir {output} --config {config} --split test --output-dir {output}/evaluation/eicu && "
            f"pces-icu evaluate-checkpoint --trainer transformer --data {DATA['nwicu']} "
            f"--model-dir {output} --config {config} --split test --output-dir {output}/evaluation/nwicu"
        )
        jobs.append(job(f"event_transformer_s{seed}", command, f"{output}/evaluation/nwicu/metrics.json"))
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--root", default="work/final_v1")
    args = parser.parse_args()
    manifest = {
        "project": "clintime_complete_evidence_v1",
        "cwd": "/home/tang/pces_icu",
        "conda": "base",
        "conda_hook": "/root/anaconda3/bin/conda",
        "gpus": [0],
        "max_parallel": 1,
        "gpu_free_threshold_mib": 500,
        "oom_retry": {"delay": 120, "max_attempts": 3},
        "other_retry": {"delay": 60, "max_attempts": 3},
        "phases": [
            {"name": "token_audit", "depends_on": [], "jobs": token_jobs(args.root)},
            {"name": "matched_training", "depends_on": ["token_audit"], "jobs": training_jobs(args.root)},
            {"name": "post_training_audit", "depends_on": ["matched_training"], "jobs": audit_jobs(args.root)},
            {"name": "sequential_baseline", "depends_on": ["post_training_audit"], "jobs": transformer_jobs(args.root)},
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "jobs": sum(len(p["jobs"]) for p in manifest["phases"])}, indent=2))


if __name__ == "__main__":
    main()
