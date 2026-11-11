"""The metrics a rung has to report, and the definitions they are computed from.

Every rung reports the same seven things, because accuracy alone hides the
defect that matters for a voice assistant: a classifier that is 93% accurate can
still act on four of every ten rows that are not commands at all. The reject
curve is what exposes that — how much of the traffic the cheap tier may answer
while staying at 90% precision, and how wrong it is on what it hands on.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

CONFIDENCE_BINS = 10
PRECISION_TARGETS = (0.90, 0.95, 0.975)


def accuracy(predicted, truth) -> float:
    hits = sum(1 for p, t in zip(predicted, truth) if p == t)
    return hits / len(truth) if len(truth) else 0.0


def expected_calibration_error(
    confidences, predicted, truth, *, bins: int = CONFIDENCE_BINS
) -> float:
    """Top-label ECE: |accuracy − confidence| averaged over confidence bins."""
    if not len(truth):
        return 0.0
    conf = np.asarray(confidences, dtype=float)
    correct = np.asarray([p == t for p, t in zip(predicted, truth)], dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        last = hi >= 1.0
        mask = (conf >= lo) & (conf <= hi if last else conf < hi)
        if not mask.any():
            continue
        total += mask.mean() * abs(float(correct[mask].mean()) - float(conf[mask].mean()))
    return float(total)


def reject_curve(
    confidences,
    predicted,
    truth,
    *,
    targets: Sequence[float] = PRECISION_TARGETS,
) -> dict[str, object]:
    """Coverage at each precision target, and the residue the tier hands on.

    Coverage is the share of rows the tier answers instead of escalating;
    ``residue_accuracy`` is the argmax accuracy on the rows it hands on, which
    is what the expensive tier inherits. The threshold at each point is chosen
    on the rows being scored, so these are oracle operating points — the ceiling
    a perfectly chosen threshold would reach, not the shipped number.

    0.90 on its own is a weak instrument here: any rung above 90% accuracy
    reaches it by accepting everything, so the higher targets are reported
    beside it.
    """
    conf = np.asarray(confidences, dtype=float)
    correct = np.asarray([p == t for p, t in zip(predicted, truth)], dtype=bool)
    n = len(correct)
    if not n:
        return {"n_rows": 0, "points": []}

    # Sweep the confidence floor from the top down: every step accepts one more
    # tie-batch of rows. n is small (382), so the quadratic sweep is free.
    order = np.argsort(-conf, kind="stable")
    best: dict[float, dict[str, float]] = {float(t): {"coverage": -1.0} for t in targets}
    floor = np.inf
    for value in conf[order]:
        if value == floor:
            continue
        floor = value
        mask = conf >= value
        covered = int(mask.sum())
        precision = float(correct[mask].sum()) / covered if covered else 1.0
        residue = ~mask
        residue_accuracy = round(float(correct[residue].mean()) if residue.any() else 0.0, 4)
        for target in best:
            if precision < target:
                continue
            coverage = covered / n
            if coverage > best[target]["coverage"]:
                best[target] = {
                    "threshold": round(float(value), 4),
                    "coverage": round(coverage, 4),
                    "precision": round(precision, 4),
                    "residue_rows": int(residue.sum()),
                    "residue_accuracy": residue_accuracy,
                }

    points = []
    for target in sorted(best):
        point = {"target_precision": target, **best[target]}
        if point["coverage"] < 0:
            point["coverage"] = 0.0
        points.append(point)
    return {"n_rows": n, "points": points}


def none_route_recall(predicted, truth, none_label: str) -> float | None:
    idx = [i for i, t in enumerate(truth) if t == none_label]
    if not idx:
        return None
    return round(sum(1 for i in idx if predicted[i] == none_label) / len(idx), 4)


def false_none_rate(predicted, truth, none_label: str) -> float | None:
    idx = [i for i, t in enumerate(truth) if t != none_label]
    if not idx:
        return None
    return round(sum(1 for i in idx if predicted[i] == none_label) / len(idx), 4)


def per_intent_ledger(predicted, truth, labels) -> dict[str, dict[str, float | int]]:
    """Per-label n, correct and recall, worst first."""
    ledger = {}
    for label in labels:
        idx = [i for i, t in enumerate(truth) if t == label]
        if not idx:
            continue
        correct = sum(1 for i in idx if predicted[i] == label)
        ledger[label] = {
            "n": len(idx),
            "correct": correct,
            "recall": round(correct / len(idx), 3),
        }
    return dict(sorted(ledger.items(), key=lambda kv: (kv[1]["recall"], kv[0])))


def latency(latencies_ms) -> dict[str, float]:
    values = np.asarray(list(latencies_ms), dtype=float)
    if not values.size:
        return {"p50": 0.0, "p95": 0.0, "mean": 0.0}
    return {
        "p50": round(float(np.percentile(values, 50)), 3),
        "p95": round(float(np.percentile(values, 95)), 3),
        "mean": round(float(values.mean()), 3),
    }


def spread(values) -> dict[str, float]:
    """Fold-to-fold spread. A win smaller than this has not won."""
    arr = np.asarray(list(values), dtype=float)
    return {
        "mean": round(float(arr.mean()), 4),
        "min": round(float(arr.min()), 4),
        "max": round(float(arr.max()), 4),
        "sd": round(float(arr.std(ddof=1)), 4) if arr.size > 1 else 0.0,
    }


def paired_delta(fold_scores, baseline_scores) -> dict[str, float | int]:
    """Per-fold difference against another rung on the same folds.

    Same folds on both sides, so the informative quantity is the spread of the
    *difference*, not the spread of either accuracy: a rung that gains the same
    amount in every fold has won even if the folds disagree wildly about the
    absolute level.
    """
    a = np.asarray(list(fold_scores), dtype=float)
    b = np.asarray(list(baseline_scores), dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"fold count mismatch: {a.shape} vs {b.shape}")
    delta = a - b
    return {
        "mean_points": round(100.0 * float(delta.mean()), 2),
        "sd_points": round(100.0 * float(delta.std(ddof=1)), 2) if delta.size > 1 else 0.0,
        "min_points": round(100.0 * float(delta.min()), 2),
        "max_points": round(100.0 * float(delta.max()), 2),
        "folds_better": int((delta > 0).sum()),
        "folds_worse": int((delta < 0).sum()),
        "folds_tied": int((delta == 0).sum()),
    }
