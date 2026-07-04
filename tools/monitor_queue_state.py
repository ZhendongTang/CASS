#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from datetime import datetime
from pathlib import Path


def now() -> str:
    return datetime.utcnow().isoformat() + "Z"


def load_state(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--interval", type=int, default=120)
    args = parser.parse_args()

    state_path = Path(args.state)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    while True:
        state = load_state(state_path)
        if state is None:
            line = f"{now()} waiting_for_state {state_path}\n"
            done = False
        else:
            jobs = state.get("jobs", [])
            counts = Counter(job.get("status", "unknown") for job in jobs)
            running = [job.get("id") for job in jobs if job.get("status") == "running"]
            stuck = [job.get("id") for job in jobs if job.get("status") in {"stuck", "failed", "failed_other"}]
            line = (
                f"{now()} total={len(jobs)} completed={counts.get('completed', 0)} "
                f"running={counts.get('running', 0)} pending={counts.get('pending', 0)} "
                f"stuck={len(stuck)} running_ids={','.join(running[:3])}"
            )
            if stuck:
                line += f" stuck_ids={','.join(stuck[:10])}"
            line += "\n"
            done = len(jobs) > 0 and (counts.get("completed", 0) == len(jobs) or bool(stuck))
        with out_path.open("a", encoding="utf-8") as handle:
            handle.write(line)
        if done:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
