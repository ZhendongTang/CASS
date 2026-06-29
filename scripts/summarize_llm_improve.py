from __future__ import annotations

import json
from pathlib import Path


root = Path("work/tasks/llm_improve_s42")
for path in sorted(root.glob("*/train_metrics.json")):
    data = json.loads(path.read_text())
    print(json.dumps({
        "run": path.parent.name,
        "variant": data.get("variant"),
        "train_records": data.get("train_records"),
        "val_records": data.get("val_records"),
        "train_positive_rate": data.get("train_positive_rate"),
        "pos_weight": data.get("pos_weight"),
        "mean_train_loss": data.get("mean_train_loss"),
        "train_eval": data.get("train_eval"),
        "validation": data.get("validation"),
    }, ensure_ascii=False))
    for metrics_path in sorted((path.parent / "evaluation").glob("*/metrics.json")):
        metrics = json.loads(metrics_path.read_text())
        print(json.dumps({
            "run": path.parent.name,
            "evaluation": metrics_path.parent.name,
            "auroc": metrics.get("auroc"),
            "auprc": metrics.get("auprc"),
            "n": metrics.get("n"),
        }, ensure_ascii=False))
