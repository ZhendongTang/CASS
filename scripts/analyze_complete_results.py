from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from pces_icu.metrics import binary_metrics


METRICS = ("auroc", "auprc", "brier", "ece", "sensitivity_at_90_specificity")
COMPARISONS = (("xml_kl", "xml"), ("xml_kl", "xml_aug"), ("xml", "flat"))
RUN_RE = re.compile(r"^(flat|xml|xml_aug|xml_kl)_seed(\d+)$")


def load_rows(path: Path) -> dict[str, dict[str, float | int | str]]:
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows[str(row["patient_id"])] = row
    return rows


def mean_std(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    return {
        "n_seeds": int(len(array)),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
    }


def endpoint_artifacts(root: Path):
    metrics = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    predictions = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for path in root.glob("main/*/*_seed*/evaluation/*/metrics.json"):
        backbone, run_name, center = path.parts[-5], path.parts[-4], path.parts[-2]
        match = RUN_RE.match(run_name)
        if not match:
            continue
        variant, seed = match.group(1), int(match.group(2))
        metrics[backbone][center][variant][seed] = json.loads(path.read_text(encoding="utf-8"))
        pred_path = path.with_name("predictions.jsonl")
        if pred_path.exists():
            predictions[backbone][center][variant][seed] = load_rows(pred_path)
    return metrics, predictions


def summarize_endpoints(metrics) -> dict:
    result = {}
    for backbone, by_center in metrics.items():
        result[backbone] = {}
        for center, by_variant in by_center.items():
            result[backbone][center] = {}
            for variant, by_seed in by_variant.items():
                result[backbone][center][variant] = {
                    metric: mean_std([float(value[metric]) for value in by_seed.values()])
                    for metric in METRICS
                }
    return result


def paired_bootstrap(predictions, reps: int, bootstrap_seed: int) -> dict:
    rng = np.random.default_rng(bootstrap_seed)
    result = {}
    for backbone, by_center in predictions.items():
        result[backbone] = {}
        for center, by_variant in by_center.items():
            comparisons = {}
            for left, right in COMPARISONS:
                common_seeds = sorted(set(by_variant.get(left, {})) & set(by_variant.get(right, {})))
                if not common_seeds:
                    continue
                common_ids = None
                for seed in common_seeds:
                    ids = set(by_variant[left][seed]) & set(by_variant[right][seed])
                    common_ids = ids if common_ids is None else common_ids & ids
                ids = sorted(common_ids or [])
                if not ids:
                    continue
                labels = np.asarray([int(by_variant[left][common_seeds[0]][pid]["label"]) for pid in ids])
                positive = np.flatnonzero(labels == 1)
                negative = np.flatnonzero(labels == 0)
                if not len(positive) or not len(negative):
                    continue
                arrays = {}
                for variant in (left, right):
                    arrays[variant] = {
                        seed: np.asarray([float(by_variant[variant][seed][pid]["probability"]) for pid in ids])
                        for seed in common_seeds
                    }
                observed = {metric: [] for metric in ("auroc", "auprc", "brier")}
                for seed in common_seeds:
                    lmet = binary_metrics(labels, arrays[left][seed])
                    rmet = binary_metrics(labels, arrays[right][seed])
                    for metric in observed:
                        observed[metric].append(float(lmet[metric] - rmet[metric]))
                draws = {metric: [] for metric in observed}
                for _ in range(reps):
                    sampled = np.concatenate([
                        rng.choice(positive, size=len(positive), replace=True),
                        rng.choice(negative, size=len(negative), replace=True),
                    ])
                    y = labels[sampled]
                    for metric in draws:
                        deltas = []
                        for seed in common_seeds:
                            lmet = binary_metrics(y, arrays[left][seed][sampled])
                            rmet = binary_metrics(y, arrays[right][seed][sampled])
                            deltas.append(float(lmet[metric] - rmet[metric]))
                        draws[metric].append(float(np.mean(deltas)))
                comparisons[f"{left}_minus_{right}"] = {
                    "n_patients": len(ids),
                    "seeds": common_seeds,
                    "metrics": {
                        metric: {
                            "mean_delta": float(np.mean(observed[metric])),
                            "ci95": [float(np.quantile(draws[metric], 0.025)), float(np.quantile(draws[metric], 0.975))],
                        }
                        for metric in draws
                    },
                }
            result[backbone][center] = comparisons
    return result


def summarize_permutations(root: Path) -> dict:
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    for path in root.glob("permutation/*/*_seed*/*/metrics.json"):
        backbone, run_name, center = path.parts[-4], path.parts[-3], path.parts[-2]
        match = RUN_RE.match(run_name)
        if not match:
            continue
        variant, _seed = match.group(1), int(match.group(2))
        data = json.loads(path.read_text(encoding="utf-8"))
        for metric, value in data["permutation"].items():
            if metric not in ("patients", "permutations"):
                grouped[backbone][center][variant][metric].append(float(value))
        grouped[backbone][center][variant]["canonical_ensemble_gap"].append(
            float(data["mean_abs_canonical_ensemble_gap"])
        )
    return {
        backbone: {
            center: {
                variant: {metric: mean_std(values) for metric, values in by_metric.items()}
                for variant, by_metric in by_variant.items()
            }
            for center, by_variant in by_center.items()
        }
        for backbone, by_center in grouped.items()
    }


def permutation_patient_artifacts(root: Path):
    result = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    eps = 1e-7
    for path in root.glob("permutation/*/*_seed*/*/predictions.jsonl"):
        backbone, run_name, center = path.parts[-4], path.parts[-3], path.parts[-2]
        match = RUN_RE.match(run_name)
        if not match:
            continue
        variant, seed = match.group(1), int(match.group(2))
        patients = {}
        for row in load_rows(path).values():
            values = np.asarray(row["permutation_probabilities"], dtype=float)
            clipped = np.clip(values, eps, 1 - eps)
            mean = clipped.mean()
            js = clipped * np.log(clipped / mean) + (1 - clipped) * np.log((1 - clipped) / (1 - mean))
            patients[str(row["patient_id"])] = {
                "label": int(row["label"]),
                "prediction_std": float(values.std()),
                "max_probability_gap": float(values.max() - values.min()),
                "js_like_divergence": float(js.mean()),
                "label_flip": float(np.any((values >= 0.5) != (values[0] >= 0.5))),
                "canonical_ensemble_gap": float(abs(float(row["canonical_probability"]) - values.mean())),
            }
        result[backbone][center][variant][seed] = patients
    return result


def paired_permutation_bootstrap(root: Path, reps: int, bootstrap_seed: int) -> dict:
    artifacts = permutation_patient_artifacts(root)
    rng = np.random.default_rng(bootstrap_seed + 1)
    features = (
        "prediction_std", "max_probability_gap", "js_like_divergence",
        "label_flip", "canonical_ensemble_gap",
    )
    result = {}
    for backbone, by_center in artifacts.items():
        result[backbone] = {}
        for center, by_variant in by_center.items():
            comparisons = {}
            for left, right in (("xml_kl", "xml"), ("xml_kl", "xml_aug")):
                common_seeds = sorted(set(by_variant.get(left, {})) & set(by_variant.get(right, {})))
                if not common_seeds:
                    continue
                common_ids = None
                for seed in common_seeds:
                    ids = set(by_variant[left][seed]) & set(by_variant[right][seed])
                    common_ids = ids if common_ids is None else common_ids & ids
                ids = sorted(common_ids or [])
                if not ids:
                    continue
                labels = np.asarray([by_variant[left][common_seeds[0]][pid]["label"] for pid in ids])
                positive = np.flatnonzero(labels == 1)
                negative = np.flatnonzero(labels == 0)
                arrays = {
                    variant: {
                        seed: {
                            feature: np.asarray([by_variant[variant][seed][pid][feature] for pid in ids])
                            for feature in features
                        }
                        for seed in common_seeds
                    }
                    for variant in (left, right)
                }
                observed = {
                    feature: float(np.mean([
                        np.mean(arrays[left][seed][feature] - arrays[right][seed][feature])
                        for seed in common_seeds
                    ]))
                    for feature in features
                }
                draws = {feature: [] for feature in features}
                for _ in range(reps):
                    if len(positive) and len(negative):
                        sampled = np.concatenate([
                            rng.choice(positive, size=len(positive), replace=True),
                            rng.choice(negative, size=len(negative), replace=True),
                        ])
                    else:
                        sampled = rng.choice(np.arange(len(ids)), size=len(ids), replace=True)
                    for feature in features:
                        seed_deltas = [
                            np.mean(arrays[left][seed][feature][sampled] - arrays[right][seed][feature][sampled])
                            for seed in common_seeds
                        ]
                        draws[feature].append(float(np.mean(seed_deltas)))
                comparisons[f"{left}_minus_{right}"] = {
                    "n_patients": len(ids),
                    "seeds": common_seeds,
                    "metrics": {
                        feature: {
                            "mean_delta": observed[feature],
                            "ci95": [
                                float(np.quantile(draws[feature], 0.025)),
                                float(np.quantile(draws[feature], 0.975)),
                            ],
                        }
                        for feature in features
                    },
                }
            result[backbone][center] = comparisons
    return result


def render_markdown(data: dict) -> str:
    lines = ["# Complete Experiment Analysis", "", "## Endpoint metrics (mean +/- SD across seeds)", ""]
    lines.append("| Backbone | Center | Variant | AUROC | AUPRC | Brier | ECE |")
    lines.append("|---|---|---|---:|---:|---:|---:|")
    for backbone, by_center in data["endpoint_summary"].items():
        for center, by_variant in by_center.items():
            for variant, values in by_variant.items():
                def cell(metric):
                    item = values[metric]
                    return f"{item['mean']:.4f} +/- {item['std']:.4f}"
                lines.append(f"| {backbone} | {center} | {variant} | {cell('auroc')} | {cell('auprc')} | {cell('brier')} | {cell('ece')} |")
    lines.extend(["", "## Paired patient-level bootstrap", ""])
    for backbone, by_center in data["paired_bootstrap"].items():
        for center, comparisons in by_center.items():
            for name, result in comparisons.items():
                lines.append(f"- **{backbone}/{center}/{name}** (n={result['n_patients']}, seeds={result['seeds']}):")
                for metric, value in result["metrics"].items():
                    lines.append(f"  - {metric}: {value['mean_delta']:+.4f}, 95% CI [{value['ci95'][0]:+.4f}, {value['ci95'][1]:+.4f}]")
    lines.extend(["", "## Permutation stability", ""])
    lines.append("See `analysis.json` for the full seed-aggregated stability hierarchy.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    args = parser.parse_args()
    root = Path(args.root)
    metrics, predictions = endpoint_artifacts(root)
    result = {
        "root": str(root),
        "endpoint_summary": summarize_endpoints(metrics),
        "paired_bootstrap": paired_bootstrap(predictions, args.bootstrap_reps, args.bootstrap_seed),
        "permutation_summary": summarize_permutations(root),
        "permutation_paired_bootstrap": paired_permutation_bootstrap(
            root, args.bootstrap_reps, args.bootstrap_seed
        ),
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "analysis.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "RESULTS.md").write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps({"output": str(output), "backbones": sorted(result["endpoint_summary"])}))


if __name__ == "__main__":
    main()
