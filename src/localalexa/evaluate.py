"""Scoring: row-wise accuracy next to the split that holds whole phrasings out.

Two protocols, and the gap between them is the point:

``row_wise``
    Train on the 200 registry references, score all 382 rows. This is the
    number that gets quoted, and it is inflated: some training references are
    literal slot fills of the templates that generated the scored rows.

``family_held_out``
    For each of the 110 phrasing families, retrain without the rows derived
    from that family, then score only that family's rows. Nothing the model
    sees shares wording with what it is scored on.

``family_held_out_strict``
    Same held-out families, but the training pool is rebuilt from template
    fills of the *other* families only — no hand-written references at all.
    This is the phrasing-generalisation estimate with the reference pool's
    paraphrase overlap removed too.

Confidence on the held-out numbers comes from a cluster bootstrap: whole
phrasing families are resampled, not individual rows, because rows inside a
family share wording and are not independent evidence.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .classifiers import DEFAULT_MODEL, TextClassifier
from .data import (
    NONE_LABEL,
    Registry,
    Row,
    expand_template,
    load_registry,
    load_routing_set,
    phrasing_families,
    template_index,
    training_pairs,
)


@dataclass(frozen=True)
class SplitResult:
    """Scored predictions for one protocol, with the cluster each row belongs to."""

    name: str
    truth: tuple[str, ...]
    predicted: tuple[str, ...]
    clusters: tuple[str, ...]

    @property
    def total(self) -> int:
        return len(self.truth)

    @property
    def correct(self) -> int:
        return sum(1 for p, t in zip(self.predicted, self.truth) if p == t)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "split": self.name,
            "n_rows": self.total,
            "n_correct": self.correct,
            "accuracy": round(self.accuracy, 4),
            "n_clusters": len(set(self.clusters)),
        }


def _predict_all(
    model: str, train: Sequence[tuple[str, str]], texts: Sequence[str]
) -> list[str]:
    classifier = TextClassifier(model).fit([t for t, _ in train], [l for _, l in train])
    return classifier.predict(texts)


def row_wise(
    rows: Sequence[Row],
    registry: Registry,
    families: Sequence[str],
    model: str = DEFAULT_MODEL,
) -> SplitResult:
    """Train on the references, score every row."""
    predicted = _predict_all(model, training_pairs(registry), [r.text for r in rows])
    return SplitResult(
        name="row_wise",
        truth=tuple(r.tool for r in rows),
        predicted=tuple(predicted),
        clusters=tuple(families),
    )


def _strict_training_pool(registry: Registry, held_out: str) -> list[tuple[str, str]]:
    """Template fills of every family except ``held_out``, plus none-route refs."""
    pool: list[tuple[str, str]] = []
    for tool in registry.tools:
        for template in tool.templates:
            if template == held_out:
                continue
            pool.extend((text, tool.id) for text in expand_template(tool, template))
    pool.extend((ref, NONE_LABEL) for ref in registry.none_references)
    return pool


def family_held_out(
    rows: Sequence[Row],
    registry: Registry,
    families: Sequence[str],
    *,
    strict: bool = False,
    model: str = DEFAULT_MODEL,
) -> SplitResult:
    """Score each phrasing family with a model that never saw that family.

    ``strict=False`` drops only the training references that are slot fills of
    the held-out template; the rest of the reference pool — hand-written
    paraphrases of the same tool — stays in. ``strict=True`` rebuilds the pool
    from template fills of the other families instead.
    """
    references = training_pairs(registry)
    predicted: list[str] = [""] * len(rows)
    for family in sorted(set(families)):
        idx = [i for i, f in enumerate(families) if f == family]
        if not idx:
            continue
        if strict:
            train = _strict_training_pool(registry, family)
        else:
            contaminated = _family_texts(registry, family)
            train = [pair for pair in references if pair[0] not in contaminated]
        out = _predict_all(model, train, [rows[i].text for i in idx])
        for i, label in zip(idx, out):
            predicted[i] = label
    return SplitResult(
        name="family_held_out_strict" if strict else "family_held_out",
        truth=tuple(r.tool for r in rows),
        predicted=tuple(predicted),
        clusters=tuple(families),
    )


def _family_texts(registry: Registry, family: str) -> set[str]:
    """Every utterance the generator can produce from one phrasing family."""
    texts: set[str] = set()
    for tool in registry.tools:
        for template in tool.templates:
            if template == family:
                texts.update(expand_template(tool, template))
    return texts


def cluster_bootstrap_ci(
    truth: Sequence[str],
    predicted: Sequence[str],
    clusters: Sequence[str],
    *,
    n_resamples: int = 2000,
    seed: int = 42,
    level: float = 0.95,
) -> tuple[float, float]:
    """Percentile interval from resampling whole clusters with replacement."""
    order = sorted(set(clusters))
    position = {c: i for i, c in enumerate(order)}
    correct = np.zeros(len(order))
    total = np.zeros(len(order))
    for t, p, c in zip(truth, predicted, clusters):
        correct[position[c]] += 1.0 if p == t else 0.0
        total[position[c]] += 1.0
    rng = random.Random(seed)
    draws = []
    for _ in range(n_resamples):
        pick = [rng.randrange(len(order)) for _ in range(len(order))]
        denominator = total[pick].sum()
        if denominator:
            draws.append(correct[pick].sum() / denominator)
    if not draws:
        return (0.0, 0.0)
    alpha = (1.0 - level) / 2.0
    lo = float(np.percentile(draws, 100 * alpha))
    hi = float(np.percentile(draws, 100 * (1 - alpha)))
    return (round(lo, 4), round(hi, 4))


def measure(
    *,
    model: str = DEFAULT_MODEL,
    splits: Sequence[str] = ("full",),
    bootstrap_resamples: int = 2000,
) -> dict[str, object]:
    """Run the requested splits on the shipped evaluation set and report the gaps.

    ``splits`` is any of ``rowwise``, ``family`` and ``strict``; ``full`` means
    all three.
    """
    wanted = set(splits)
    full = "full" in wanted
    registry = load_registry()
    rows = load_routing_set()
    index = template_index(registry)
    families = phrasing_families(rows, index)

    results = []
    if full or "rowwise" in wanted:
        results.append(row_wise(rows, registry, families, model=model))
    if full or "family" in wanted:
        results.append(family_held_out(rows, registry, families, strict=False, model=model))
    if full or "strict" in wanted:
        results.append(family_held_out(rows, registry, families, strict=True, model=model))
    if not results:
        raise ValueError(f"no known split in {sorted(wanted)}")

    payload = {"model": model, "splits": []}
    baseline = results[0].accuracy
    for result in results:
        ci = cluster_bootstrap_ci(
            result.truth,
            result.predicted,
            result.clusters,
            n_resamples=bootstrap_resamples,
        )
        payload["splits"].append(
            {
                **result.as_dict(),
                "cluster_bootstrap_ci95": list(ci),
                "gap_vs_row_wise_points": round(100.0 * (result.accuracy - baseline), 1)
                if baseline
                else 0.0,
            }
        )
    return payload


def load(rows_path=None, registry_path=None) -> tuple[Registry, list[Row], list[str]]:
    """Registry, evaluation rows and the phrasing family of each row."""
    registry = load_registry(registry_path)
    rows = load_routing_set(rows_path)
    index = template_index(registry)
    return registry, rows, phrasing_families(rows, index)
