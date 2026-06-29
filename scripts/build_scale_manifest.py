from __future__ import annotations

import argparse
import json
from pathlib import Path


DATA = {
    "mimic": "work/events/mimic.jsonl.gz",
    "eicu": "work/events/eicu.jsonl.gz",
    "nwicu": "work/events/nwicu.jsonl.gz",
}
VARIANTS = ("flat", "xml", "xml_aug", "xml_kl")
PREFIX = (
    "source .venv/bin/activate && export PYTHONPATH=src TRANSFORMERS_NO_TF=1 USE_TF=0 "
    "PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false "
    "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && "
)


def make_job(job_id: str, command: str, expected: str) -> dict[str, str]:
    return {"id": job_id, "cmd": PREFIX + command, "expected_output": expected}


def train_job(root: str, model: str, tag: str, variant: str, seed: int, sample_seed: int,
              train_records: int, max_length: int) -> dict[str, str]:
    output = f"{root}/{tag}/{variant}_seed{seed}"
    eval_args = " ".join(f"--eval-data {name}={path}" for name, path in DATA.items())
    command = (
        f"python -m pces_icu.multicenter_llm train --data {DATA['mimic']} --model {model} "
        f"--output {output} --variant {variant} --seed {seed} --sample-seed {sample_seed} "
        f"--epochs 1 --max-length {max_length} --grad-accum 8 --lr 2e-5 --kl-weight 0.2 "
        f"--max-train-records {train_records} --max-val-records 512 {eval_args} --max-eval-records 1000"
    )
    return make_job(f"train_{tag}_{variant}_s{seed}", command, f"{output}/train_metrics.json")


def audit_jobs(root: str, tag: str, seed: int) -> list[dict[str, str]]:
    jobs = []
    for variant in VARIANTS:
        checkpoint = f"{root}/{tag}/{variant}_seed{seed}"
        for center, data in DATA.items():
            output = f"{root}/{tag}_permutation/{variant}_seed{seed}/{center}"
            command = (
                f"python -m pces_icu.multicenter_llm audit-permutation --data {data} "
                f"--checkpoint {checkpoint} --output {output} --dataset {center} "
                "--max-records 1000 --permutations 5 --audit-seed 2026 --view-batch-size 1"
            )
            jobs.append(make_job(
                f"audit_{tag}_{variant}_s{seed}_{center}", command, f"{output}/metrics.json"
            ))
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--mode", choices=("context", "scale"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-seed", type=int, default=2026)
    parser.add_argument("--contexts", type=int, nargs="+", default=[1024, 4096])
    parser.add_argument("--sizes", type=int, nargs="+", default=[5000, 20000])
    parser.add_argument("--context", type=int, default=1024)
    args = parser.parse_args()

    phases = []
    previous = None
    if args.mode == "context":
        for context in args.contexts:
            tag = f"{args.backbone}_n1000_l{context}"
            jobs = [
                train_job(args.root, args.model, tag, variant, args.seed, args.sample_seed, 1000, context)
                for variant in VARIANTS
            ]
            name = f"context_{context}"
            phases.append({"name": name, "depends_on": [previous] if previous else [], "jobs": jobs})
            previous = name
            audit_name = f"context_{context}_audit"
            phases.append({"name": audit_name, "depends_on": [name], "jobs": audit_jobs(args.root, tag, args.seed)})
            previous = audit_name
    else:
        for size in args.sizes:
            tag = f"{args.backbone}_n{size}_l{args.context}"
            jobs = [
                train_job(args.root, args.model, tag, variant, args.seed, args.sample_seed, size, args.context)
                for variant in VARIANTS
            ]
            name = f"scale_{size}"
            phases.append({"name": name, "depends_on": [previous] if previous else [], "jobs": jobs})
            previous = name
            audit_name = f"scale_{size}_audit"
            phases.append({"name": audit_name, "depends_on": [name], "jobs": audit_jobs(args.root, tag, args.seed)})
            previous = audit_name

    manifest = {
        "project": f"clintime_{args.mode}_{args.backbone}",
        "cwd": "/home/tang/pces_icu",
        "conda": "base",
        "conda_hook": "/root/anaconda3/bin/conda",
        "gpus": [0],
        "max_parallel": 1,
        "gpu_free_threshold_mib": 500,
        "oom_retry": {"delay": 180, "max_attempts": 2},
        "other_retry": {"delay": 120, "max_attempts": 2},
        "phases": phases,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "jobs": sum(len(x["jobs"]) for x in phases)}))


if __name__ == "__main__":
    main()
