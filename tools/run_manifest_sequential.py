#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def now() -> str:
    return datetime.utcnow().isoformat() + "Z"


def resolve_conda_hook(manifest_hook: str | None = None) -> str:
    if manifest_hook in {"", "none", "None", None}:
        return ""
    if manifest_hook:
        return manifest_hook if manifest_hook.startswith("eval") else f'eval "$({manifest_hook} shell.bash hook)"'
    for path in (
        Path.home() / "anaconda3/bin/conda",
        Path.home() / "miniconda3/bin/conda",
        Path("/opt/anaconda3/bin/conda"),
        Path("/opt/miniconda3/bin/conda"),
    ):
        if path.exists():
            return f'eval "$({path} shell.bash hook)"'
    return 'eval "$(conda shell.bash hook)"'


def output_exists(path_pattern: str | None, cwd: str) -> bool:
    if not path_pattern:
        return False
    path = Path(path_pattern)
    if not path.is_absolute():
        path = Path(cwd) / path
    return bool(list(path.parent.glob(path.name)))


def save_state(state: dict, state_path: Path) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = state_path.with_suffix(state_path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(state_path)


def build_initial_state(manifest: dict) -> dict:
    jobs = []
    for phase in manifest.get("phases", []):
        for job in phase.get("jobs", []):
            jobs.append(
                {
                    "id": job["id"],
                    "phase": phase.get("name"),
                    "cmd": job["cmd"],
                    "expected_output": job.get("expected_output"),
                    "status": "pending",
                    "gpu": None,
                    "screen_name": None,
                    "pid": None,
                    "attempts": 0,
                    "started": None,
                    "completed": None,
                    "error": None,
                    "meta": job.get("meta", {}),
                }
            )
    return {
        "meta": {"project": manifest.get("project", "unknown"), "started": now(), "runner": "sequential"},
        "phases": [
            {"name": p.get("name", f"phase_{i}"), "depends_on": p.get("depends_on", []), "status": "pending"}
            for i, p in enumerate(manifest.get("phases", []))
        ],
        "jobs": jobs,
    }


def phase_ready(phase: dict, state: dict) -> bool:
    phases = {item["name"]: item for item in state["phases"]}
    return all(phases.get(dep, {}).get("status") == "completed" for dep in phase.get("depends_on", []))


def run_job(job: dict, manifest: dict, log_dir: Path) -> int:
    cwd = manifest.get("cwd", ".")
    conda = manifest.get("conda", "base")
    conda_hook = resolve_conda_hook(manifest.get("conda_hook"))
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{job['id']}.log"
    setup = f"cd {shlex.quote(cwd)}"
    if conda:
        setup += f" && {conda_hook} && conda activate {shlex.quote(conda)}"
    full_cmd = f"{setup} && {job['cmd']}"
    with log_path.open("ab") as log:
        log.write(f"\\n[{now()}] START {job['id']}\\n".encode())
        proc = subprocess.run(["bash", "-lc", full_cmd], stdout=log, stderr=subprocess.STDOUT)
        log.write(f"\\n[{now()}] EXIT {proc.returncode} {job['id']}\\n".encode())
        return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    state_path = Path(args.state)
    if args.resume and state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    else:
        state = build_initial_state(manifest)
    save_state(state, state_path)

    phase_defs = manifest.get("phases", [])
    for phase_def in phase_defs:
        phase_name = phase_def.get("name")
        phase_state = next(item for item in state["phases"] if item["name"] == phase_name)
        if not phase_ready(phase_def, state):
            phase_state["status"] = "stuck"
            save_state(state, state_path)
            continue
        phase_state["status"] = "running"
        save_state(state, state_path)
        for job in [item for item in state["jobs"] if item.get("phase") == phase_name]:
            if job["status"] == "completed":
                continue
            if output_exists(job.get("expected_output"), manifest.get("cwd", ".")):
                job["status"] = "completed"
                job["completed"] = now()
                save_state(state, state_path)
                continue
            job["status"] = "running"
            job["attempts"] += 1
            job["started"] = now()
            save_state(state, state_path)
            rc = run_job(job, manifest, Path(args.log_dir))
            if rc == 0 and output_exists(job.get("expected_output"), manifest.get("cwd", ".")):
                job["status"] = "completed"
                job["completed"] = now()
                job["error"] = None
            else:
                job["status"] = "stuck"
                job["completed"] = now()
                job["error"] = f"exit={rc}; expected output missing or command failed"
            save_state(state, state_path)
        phase_jobs = [item for item in state["jobs"] if item.get("phase") == phase_name]
        phase_state["status"] = "completed" if all(item["status"] == "completed" for item in phase_jobs) else "failed"
        save_state(state, state_path)
    return 0 if all(item["status"] == "completed" for item in state["jobs"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
