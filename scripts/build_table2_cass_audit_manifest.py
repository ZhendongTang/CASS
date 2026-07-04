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
    "mortality": {
        "mimic": "work/events/mimic.jsonl.gz",
        "eicu": "work/events/eicu.jsonl.gz",
        "nwicu": "work/events/nwicu.jsonl.gz",
    },
    "los": {
        "mimic": "work/tasks/los_gt48/mimic.jsonl.gz",
        "eicu": "work/tasks/los_gt48/eicu.jsonl.gz",
        "nwicu": "work/tasks/los_gt48/nwicu.jsonl.gz",
    },
    "future": {
        "mimic": "work/tasks/future_intervention/mimic.jsonl.gz",
        "eicu": "work/tasks/future_intervention/eicu.jsonl.gz",
        "nwicu": "work/tasks/future_intervention/nwicu.jsonl.gz",
    },
}

DEFAULT_PYTHON = "/usr/local/bin/python3"

PREFIX = (
    "export PYTHONPATH=src TRANSFORMERS_NO_TF=1 USE_TF=0 PYTHONUNBUFFERED=1 "
    "TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && "
)


def make_job(
    llm: str,
    task: str,
    center: str,
    root: str,
    max_records: int,
    views: int,
    budget: int,
    python_bin: str,
) -> dict[str, object]:
    output = f"{root}/{llm}/{task}/{center}.json"
    cmd = (
        f"{python_bin} scripts/audit_table2_cass.py --data {DATA[task][center]} --dataset {center} "
        f"--task {task} --model {MODELS[llm]} --output {output} --split test "
        f"--max-records {max_records} --sample-seed 2026 --views {views} --budget {budget} "
        "--variants flat xml meds json compact summary"
    )
    return {
        "id": f"table2_{llm}_{task}_{center}",
        "cmd": PREFIX + cmd,
        "expected_output": output,
        "meta": {
            "llm": llm,
            "model": MODELS[llm],
            "python": python_bin,
            "task": task,
            "center": center,
            "purpose": "Table 2 CASS determinism/token-budget audit",
        },
    }


def make_jobs(llm: str, root: str, max_records: int, views: int, budget: int, python_bin: str) -> list[dict[str, object]]:
    return [
        make_job(llm, task, center, root, max_records, views, budget, python_bin)
        for task in ("mortality", "los", "future")
        for center in ("mimic", "eicu", "nwicu")
    ]


def build_manifest(llm: str, root: str, max_records: int, views: int, budget: int, python_bin: str) -> dict[str, object]:
    jobs = [
        *make_jobs(llm, root, max_records, views, budget, python_bin),
    ]
    return {
        "project": f"cass_table2_audit_{llm}",
        "cwd": "/home/tang/pces_icu",
        "conda": "",
        "conda_hook": "",
        "gpus": [0],
        "max_parallel": 1,
        "gpu_free_threshold_mib": 100000,
        "oom_retry": {"delay": 120, "max_attempts": 2},
        "other_retry": {"delay": 60, "max_attempts": 3},
        "notes": [
            "Table 2 redesign: CASS/summary determinism and token-budget audit.",
            "No model training; jobs use each LLM tokenizer to report token footprint.",
            "GPU threshold is intentionally high because these tokenizer-only jobs are CPU-light.",
            "Outputs support a table with exact-match-under-reorder, unique hashes, p95 tokens, and truncation rate.",
        ],
        "phases": [
            {
                "name": f"{llm}_table2_cass_audit",
                "depends_on": [],
                "jobs": jobs,
            }
        ],
    }


def build_combined_manifest(root: str, max_records: int, views: int, budget: int, python_bin: str) -> dict[str, object]:
    phases = []
    for llm in ("glm", "llama", "gemma"):
        phases.append(
            {
                "name": f"{llm}_table2_cass_audit",
                "depends_on": [],
                "jobs": make_jobs(llm, root, max_records, views, budget, python_bin),
            }
        )
    return {
        "project": "cass_table2_audit_server1_all_llms",
        "cwd": "/home/tang/pces_icu",
        "conda": "",
        "conda_hook": "",
        "gpus": [0],
        "max_parallel": 1,
        "gpu_free_threshold_mib": 100000,
        "oom_retry": {"delay": 120, "max_attempts": 2},
        "other_retry": {"delay": 60, "max_attempts": 3},
        "notes": [
            "Table 2 redesign: CASS/summary determinism and token-budget audit.",
            "All 3 x 9 tokenizer-audit jobs are scheduled on server1.",
            "Uses /usr/local/bin/python3 on server1 because it can load GLM, Llama, and Gemma tokenizers.",
            "No model training; jobs use each LLM tokenizer to report token footprint.",
            "Outputs support a table with exact-match-under-reorder, unique hashes, p95 tokens, and truncation rate.",
        ],
        "phases": phases,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--root", default="work/final_v2/table2_cass_audit")
    parser.add_argument("--max-records", type=int, default=1000)
    parser.add_argument("--views", type=int, default=5)
    parser.add_argument("--budget", type=int, default=2048)
    parser.add_argument("--python-bin", default=DEFAULT_PYTHON)
    parser.add_argument("--combined", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.combined:
        manifest = build_combined_manifest(args.root, args.max_records, args.views, args.budget, args.python_bin)
        path = output_dir / "manifest_table2_server1_all_llms.json"
        path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        jobs = sum(len(phase["jobs"]) for phase in manifest["phases"])
        print(json.dumps({"llm": "all", "path": str(path), "jobs": jobs}, ensure_ascii=False))
        return

    for llm in ("glm", "llama", "gemma"):
        manifest = build_manifest(llm, args.root, args.max_records, args.views, args.budget, args.python_bin)
        path = output_dir / f"manifest_table2_{llm}.json"
        path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({"llm": llm, "path": str(path), "jobs": len(manifest["phases"][0]["jobs"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
