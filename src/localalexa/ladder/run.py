"""Scoring loop shared by every rung.

Two protocols, both measured, because they answer different questions:

``rowwise``
    Train on all 200 references, score all 382 rows. This is the protocol the
    93.19% / 0.61 ms baseline was measured under, so it is the only one on which
    "did we beat it" is a fair question.

``grouped``
    Five folds over phrasing families; a fold trains on the references that do
    not render into that fold's families. This is the number that survives
    rephrasing, and the one to ship a decision on.

Latency is measured per single utterance — transform plus classify, one row at a
time, warm — which is how the historical 0.61 ms was taken.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence

from ..data import (
    NONE_LABEL,
    Registry,
    Row,
    load_registry,
    load_routing_set,
    phrasing_families,
    template_index,
)
from ..evaluate import cluster_bootstrap_ci
from . import metrics
from .folds import DEFAULT_FOLDS, FoldAssignment, build_folds
from .pools import fold_pool, reference_pool


@dataclass(frozen=True)
class Bench:
    registry: Registry
    rows: tuple[Row, ...]
    families: tuple[str, ...]
    folds: FoldAssignment
    labels: tuple[str, ...]

    @property
    def texts(self) -> list[str]:
        return [row.text for row in self.rows]

    @property
    def truth(self) -> list[str]:
        return [row.tool for row in self.rows]


class Rung:
    """A rung is a name, a way to build an unfitted predictor, and maybe more rows.

    ``augment`` is called per fold with the surviving reference pool and the set
    of held-out phrasing families, so an augmentation that draws on the
    registry's templates can leave the held-out families alone.
    """

    def __init__(self, name: str, build, *, augment=None, note: str = "") -> None:
        self.name = name
        self.build = build
        self.augment = augment
        self.note = note

    def pool_for(
        self, base: Sequence[tuple[str, str]], held_out: frozenset[str] = frozenset()
    ) -> list[tuple[str, str]]:
        if self.augment is None:
            return list(base)
        return list(base) + list(self.augment(list(base), held_out))


def load_bench(n_folds: int = DEFAULT_FOLDS) -> Bench:
    registry = load_registry()
    rows = tuple(load_routing_set())
    families = tuple(phrasing_families(list(rows), template_index(registry)))
    return Bench(
        registry=registry,
        rows=rows,
        families=families,
        folds=build_folds(families, n_folds),
        labels=tuple(registry.label_ids),
    )


def predict_with_confidence(predictor, texts: Sequence[str]):
    """Argmax label and its probability-like score, per text."""
    matrix = predictor.scores(list(texts))
    classes = predictor.classes
    best = matrix.argmax(axis=1)
    predicted = [classes[int(i)] for i in best]
    confidences = [float(matrix[i, int(b)]) for i, b in enumerate(best)]
    return predicted, confidences


def block_metrics(
    predicted: Sequence[str],
    confidences: Sequence[float],
    truth: Sequence[str],
    clusters: Sequence[str],
) -> dict[str, object]:
    curve = metrics.reject_curve(confidences, predicted, truth)
    return {
        "accuracy": round(metrics.accuracy(predicted, truth), 4),
        "ece": round(metrics.expected_calibration_error(confidences, predicted, truth), 4),
        "mean_confidence": round(sum(confidences) / len(confidences), 4) if confidences else 0.0,
        "reject_curve": curve,
        "none_recall": metrics.none_route_recall(predicted, truth, NONE_LABEL),
        "false_none_rate": metrics.false_none_rate(predicted, truth, NONE_LABEL),
        "ledger": metrics.per_intent_ledger(predicted, truth, sorted(set(truth))),
        "cluster_bootstrap_ci95": list(cluster_bootstrap_ci(truth, predicted, clusters)),
    }


def timed_latencies(predictor, texts: Sequence[str]) -> list[float]:
    latencies = []
    for text in texts:
        started = time.perf_counter()
        predictor.scores([text])
        latencies.append((time.perf_counter() - started) * 1000.0)
    return latencies


def evaluate_rowwise(rung: Rung, bench: Bench) -> dict[str, object]:
    pool = rung.pool_for(reference_pool(bench.registry))
    predictor = rung.build().fit([t for t, _ in pool], [l for _, l in pool])
    texts = bench.texts
    predictor.scores(texts[:1])  # warm the caches
    latencies = timed_latencies(predictor, texts)
    predicted, confidences = predict_with_confidence(predictor, texts)
    return {
        "protocol": "rowwise",
        "train_rows": len(pool),
        "n_rows": len(texts),
        "latency_ms": metrics.latency(latencies),
        "per_fold_accuracy": [],
        "fold_spread": None,
        **block_metrics(predicted, confidences, bench.truth, bench.families),
        "predictions": predicted,
    }


def evaluate_grouped(rung: Rung, bench: Bench) -> dict[str, object]:
    predicted = [""] * len(bench.rows)
    confidences = [0.0] * len(bench.rows)
    latencies: list[float] = []
    fold_scores: list[float] = []
    fold_rows: list[int] = []
    train_rows: list[int] = []

    for indices in bench.folds.indices(list(bench.families)):
        held_out = {bench.families[i] for i in indices}
        pool = rung.pool_for(fold_pool(bench.registry, held_out), frozenset(held_out))
        texts = [bench.rows[i].text for i in indices]
        truth = [bench.rows[i].tool for i in indices]
        predictor = rung.build().fit([t for t, _ in pool], [l for _, l in pool])
        predictor.scores(texts[:1])  # warm the caches
        latencies.extend(timed_latencies(predictor, texts))
        out, conf = predict_with_confidence(predictor, texts)
        for i, label, score in zip(indices, out, conf):
            predicted[i] = label
            confidences[i] = score
        fold_scores.append(metrics.accuracy(out, truth))
        fold_rows.append(len(indices))
        train_rows.append(len(pool))

    return {
        "protocol": "grouped",
        "n_folds": bench.folds.n_folds,
        "train_rows": train_rows,
        "fold_rows": fold_rows,
        "n_rows": len(bench.rows),
        "latency_ms": metrics.latency(latencies),
        "per_fold_accuracy": [round(s, 4) for s in fold_scores],
        "fold_spread": metrics.spread(fold_scores),
        **block_metrics(predicted, confidences, bench.truth, bench.families),
        "predictions": predicted,
    }
