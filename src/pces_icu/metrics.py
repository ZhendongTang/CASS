from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def _arrays(y_true: Iterable[int], y_prob: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(list(y_true), dtype=np.int64)
    p = np.asarray(list(y_prob), dtype=np.float64)
    if y.shape != p.shape or y.ndim != 1:
        raise ValueError("y_true and y_prob must be aligned one-dimensional arrays")
    if len(y) == 0 or not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("binary labels with both classes are required")
    if np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("probabilities must be finite and in [0, 1]")
    return y, p


def auroc(y_true: Iterable[int], y_prob: Iterable[float]) -> float:
    y, p = _arrays(y_true, y_prob)
    positives, negatives = int(y.sum()), int((1 - y).sum())
    if not positives or not negatives:
        return math.nan
    order = np.argsort(p, kind="mergesort")
    sorted_p = p[order]
    ranks = np.arange(1, len(p) + 1, dtype=float)
    start = 0
    while start < len(p):
        end = start + 1
        while end < len(p) and sorted_p[end] == sorted_p[start]:
            end += 1
        ranks[start:end] = ranks[start:end].mean()
        start = end
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    rank_sum = ranks[inverse][y == 1].sum()
    return float((rank_sum - positives * (positives + 1) / 2) / (positives * negatives))


def average_precision(y_true: Iterable[int], y_prob: Iterable[float]) -> float:
    y, p = _arrays(y_true, y_prob)
    positives = int(y.sum())
    if not positives:
        return math.nan
    order = np.argsort(-p, kind="mergesort")
    sorted_y = y[order]
    precision = np.cumsum(sorted_y) / np.arange(1, len(y) + 1)
    return float((precision * sorted_y).sum() / positives)


def brier_score(y_true: Iterable[int], y_prob: Iterable[float]) -> float:
    y, p = _arrays(y_true, y_prob)
    return float(np.mean((p - y) ** 2))


def expected_calibration_error(y_true: Iterable[int], y_prob: Iterable[float], bins: int = 15) -> float:
    y, p = _arrays(y_true, y_prob)
    edges = np.linspace(0, 1, bins + 1)
    indices = np.minimum(np.digitize(p, edges[1:-1]), bins - 1)
    ece = 0.0
    for index in range(bins):
        mask = indices == index
        if np.any(mask):
            ece += mask.mean() * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(ece)


def sensitivity_at_specificity(y_true: Iterable[int], y_prob: Iterable[float], target_specificity: float = 0.9) -> float:
    y, p = _arrays(y_true, y_prob)
    negatives = p[y == 0]
    positives = p[y == 1]
    if not len(negatives) or not len(positives):
        return math.nan
    threshold = float(np.quantile(negatives, target_specificity, method="higher"))
    return float(np.mean(positives >= threshold))


def binary_metrics(y_true: Iterable[int], y_prob: Iterable[float], bins: int = 15) -> dict[str, float]:
    y, p = _arrays(y_true, y_prob)
    return {
        "n": int(len(y)),
        "prevalence": float(y.mean()),
        "auprc": average_precision(y, p),
        "auroc": auroc(y, p),
        "brier": brier_score(y, p),
        "ece": expected_calibration_error(y, p, bins),
        "sensitivity_at_90_specificity": sensitivity_at_specificity(y, p, 0.9),
    }


def permutation_metrics(probability_matrix: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    values = np.asarray(probability_matrix, dtype=float)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("Expected [patients, permutations] with at least two permutations")
    eps = 1e-7
    clipped = np.clip(values, eps, 1 - eps)
    means = clipped.mean(axis=1, keepdims=True)
    kl = clipped * np.log(clipped / means) + (1 - clipped) * np.log((1 - clipped) / (1 - means))
    flips = np.any((values >= threshold) != (values[:, :1] >= threshold), axis=1)
    return {
        "patients": int(values.shape[0]),
        "permutations": int(values.shape[1]),
        "mean_prediction_std": float(values.std(axis=1).mean()),
        "mean_max_probability_gap": float((values.max(axis=1) - values.min(axis=1)).mean()),
        "mean_js_like_divergence": float(kl.mean()),
        "label_flip_rate": float(flips.mean()),
    }

