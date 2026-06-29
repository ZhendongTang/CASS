from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pces_icu.io import read_jsonl
from pces_icu.metrics import binary_metrics, permutation_metrics
from pces_icu.schema import PatientRecord, validate_events
from pces_icu.serialize import SerializationConfig, extract_event_ids, paired_views
from pces_icu.synthetic import write_synthetic


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "synthetic.jsonl.gz"
        n = write_synthetic(path, n=80, seed=2026)
        records = [PatientRecord.from_dict(value) for value in read_jsonl(path)]
        assert len(records) == n == 80
        assert {record.split for record in records} == {"train", "val", "test"}
        assert not [error for record in records for error in validate_events(record.events or [], 360)]
        record = next(record for record in records if len(record.events or []) >= 4)
        a, b = paired_views(record.events or [], SerializationConfig(max_events=128), seed=7)
        assert sorted(extract_event_ids(a)) == sorted(extract_event_ids(b))
        probabilities = [0.8 if record.label else 0.2 for record in records]
        metrics = binary_metrics([record.label for record in records], probabilities)
        permutation = permutation_metrics([[p, p + 0.01, p - 0.01] for p in probabilities])
        print(json.dumps({"records": n, "metrics": metrics, "permutation": permutation}, indent=2))


if __name__ == "__main__":
    main()

