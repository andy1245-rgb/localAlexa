"""Job 2 — the two-stage bucket, built from the data's own confusions.

The operator's diagnosis is that the confusions are within-family, so the task should be
hierarchical: a coarse bucket first, then the specific action. This builds that hierarchy from
the *current model's out-of-fold confusions* rather than from intuition, then compares it with
the flat incumbent on the same grouped folds — including the accuracy of stage A on its own,
because a two-stage model's ceiling is stage A: an error there is unrepairable.

Three arms, same folds:

``flat``
    the 18-way incumbent over every label.
``two_stage``
    stage A picks one bucket, stage B runs only over that bucket's labels.
``top2``
    stage A's two best buckets, stage B runs over their union and the best label wins.

And the failure mode is reported as a number: how many rows stage A put in the wrong bucket, and
whether stage B would have answered them correctly had it been given the right one — the size of
the mistake the hierarchy itself makes.

    python job2_two_stage_bucket.py --out results/dell321/job2.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
from sklearn.cluster import AgglomerativeClustering  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402

from localalexa.data import (  # noqa: E402
    NONE_LABEL,
    load_registry,
    load_routing_set,
    phrasing_families,
    template_index,
)
from localalexa.ladder.folds import DEFAULT_FOLDS, build_folds  # noqa: E402
from localalexa.ladder.pools import fold_pool  # noqa: E402
from localalexa.variants import fit_plain, logreg, word_vectorizer  # noqa: E402

BUCKET_SIZES = (2, 3, 4, 5, 6)


def fold_plan(registry, rows):
    families = phrasing_families(rows, template_index(registry))
    assignment = build_folds(families, DEFAULT_FOLDS)
    plan = []
    for fold in range(DEFAULT_FOLDS):
        indices = [i for i in range(len(rows)) if assignment.fold_of(families[i]) == fold]
        held = {families[i] for i in indices}
        plan.append((indices, held, fold_pool(registry, held)))
    return plan


def oof_predictions(registry, rows, plan) -> tuple[list[str], list[str]]:
    """Out-of-fold predictions from the flat incumbent, and the truth beside them."""
    predicted = [""] * len(rows)
    for indices, _held, pool in plan:
        model = fit_plain(pool)
        for i in indices:
            predicted[i] = model.predict(rows[i].text)
    return predicted, [row.tool for row in rows]


def confusion_distance(labels, predicted, truth) -> np.ndarray:
    """Distance between two labels: 1 minus how much of their traffic crosses between them.

    ``d(i, j) = 1 - (confused(i->j) + confused(j->i)) / (n_i + n_j)`` so two labels the model
    mixes up sit close together and two it never mixes sit at distance 1. That is the whole
    premise of the hierarchy, measured instead of assumed.
    """
    confusion = Counter(zip(truth, predicted))
    counts = Counter(truth)
    size = len(labels)
    distance = np.ones((size, size))
    for a in range(size):
        for b in range(size):
            if a == b:
                distance[a, b] = 0.0
                continue
            crossing = confusion[(labels[a], labels[b])] + confusion[(labels[b], labels[a])]
            total = counts[labels[a]] + counts[labels[b]]
            distance[a, b] = 1.0 - (crossing / total if total else 0.0)
    return distance


def cluster_labels(labels, distance, k: int) -> dict[str, int]:
    model = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average")
    assignment = model.fit_predict(distance)
    return {label: int(bucket) for label, bucket in zip(labels, assignment)}


def bucket_evidence(buckets: dict[str, int], confusion) -> list[dict]:
    """The crossing pairs that justify each bucket, strongest first."""
    out = []
    for true_label, predicted_label in confusion:
        if true_label == predicted_label:
            continue
        if buckets.get(true_label) == buckets.get(predicted_label):
            out.append({"true": true_label, "pred": predicted_label,
                        "n": confusion[(true_label, predicted_label)],
                        "bucket": buckets.get(true_label)})
    out.sort(key=lambda row: -row["n"])
    return out[:12]


class TwoStage:
    """Bucket first, then the labels inside the bucket(s)."""

    def __init__(self, buckets: dict[str, int], labels, top_k: int = 1) -> None:
        self.buckets = buckets
        self.labels = labels
        self.top_k = top_k
        self.bucket_model = None
        self.member_models: dict[int, object] = {}
        self.bucket_names: list[int] = []
        self.member_labels: dict[int, list[str]] = defaultdict(list)

    def fit(self, pool) -> "TwoStage":
        bucket_of = self.buckets
        self.bucket_names = sorted(set(bucket_of.values()))
        self.bucket_model = make_pipeline(word_vectorizer(), logreg())
        self.bucket_model.fit([text for text, _ in pool],
                              [f"b{bucket_of[label]}" for _, label in pool])
        by_bucket: dict[int, list[tuple[str, str]]] = defaultdict(list)
        for text, label in pool:
            by_bucket[bucket_of[label]].append((text, label))
            self.member_labels[bucket_of[label]].append(label)
        for bucket, rows in by_bucket.items():
            labels_here = sorted({label for _, label in rows})
            if len(labels_here) < 2:
                self.member_models[bucket] = ("constant", labels_here[0])
                continue
            model = make_pipeline(word_vectorizer(), logreg())
            model.fit([text for text, _ in rows], [label for _, label in rows])
            self.member_models[bucket] = ("model", model)
        return self

    def _bucket_distribution(self, text: str) -> dict[int, float]:
        row = self.bucket_model.predict_proba([text])[0]
        names = list(self.bucket_model.classes_)
        return {int(name[1:]): float(p) for name, p in zip(names, row)}

    def _member_distribution(self, bucket: int, text: str) -> dict[str, float]:
        kind, model = self.member_models[bucket]
        if kind == "constant":
            return {model: 1.0}
        row = model.predict_proba([text])[0]
        return {str(label): float(p) for label, p in zip(model.classes_, row)}

    def predict(self, text: str) -> str:
        buckets = self._bucket_distribution(text)
        chosen = sorted(buckets, key=lambda b: -buckets[b])[: self.top_k]
        best_label, best_score = None, -1.0
        for bucket in chosen:
            for label, score in self._member_distribution(bucket, text).items():
                weighted = score * buckets[bucket]
                if weighted > best_score:
                    best_label, best_score = label, weighted
        return best_label or NONE_LABEL

    def stage_a(self, text: str) -> int:
        buckets = self._bucket_distribution(text)
        return max(buckets, key=lambda b: buckets[b])


def timed(predict, texts) -> dict:
    samples = []
    for text in texts:
        started = time.perf_counter()
        predict(text)
        samples.append((time.perf_counter() - started) * 1000.0)
    samples.sort()
    return {
        "p50_ms": round(statistics.median(samples), 3),
        "p95_ms": round(samples[int(0.95 * len(samples))], 3),
        "mean_ms": round(statistics.mean(samples), 2),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="results/dell321/job2.json")
    parser.add_argument("--sizes", default=",".join(str(s) for s in BUCKET_SIZES))
    args = parser.parse_args(argv)

    registry = load_registry()
    rows = load_routing_set()
    labels = registry.label_ids
    plan = fold_plan(registry, rows)
    predicted, truth = oof_predictions(registry, rows, plan)
    flat_accuracy = sum(1 for p, t in zip(predicted, truth) if p == t) / len(rows)
    print(f"rows {len(rows)}  labels {len(labels)}  flat out-of-fold {flat_accuracy:.4f}")

    confusion = Counter((t, p) for p, t in zip(predicted, truth) if p != t)
    distance = confusion_distance(labels, predicted, truth)
    payload = {
        "config": {"folds": DEFAULT_FOLDS, "labels": len(labels)},
        "flat": {"accuracy": round(flat_accuracy, 4),
                 "per_fold": [
                     round(sum(1 for i in indices
                               if predicted[i] == rows[i].tool) / len(indices), 4)
                     for indices, _held, _pool in plan]},
        "confusion": {"pairs": [{"true": t, "pred": p, "n": n}
                                for (t, p), n in confusion.most_common(15)]},
        "buckets": {},
    }

    baseline_folds = payload["flat"]["per_fold"]
    proposals = {"agglomerative_k": {k: cluster_labels(labels, distance, k)
                                     for k in [int(v) for v in args.sizes.split(",")]}}

    # The operator's hypothesis, as a bucketing: the registry's own families. Worth scoring even
    # though it is "intuitive", because the brief's claim is precisely that it is the right one.
    family_of = {tool.id: tool.family for tool in registry.tools}
    family_of[NONE_LABEL] = "none"
    family_names = {family: i for i, family in enumerate(sorted(set(family_of.values())))}
    proposals["registry_family"] = {"family": {label: family_names[family_of[label]]
                                               for label in labels}}
    # And the strongest single confusion, as the smallest hierarchy the data supports.
    strongest = confusion.most_common(1)[0][0] if confusion else (labels[0], labels[1])
    pair = set(strongest)
    proposals["strongest_pair"] = {"pair": {label: (0 if label in pair else 1)
                                            for label in labels}}

    for proposal, variants in proposals.items():
        for name, buckets in variants.items():
            sizes = Counter(buckets.values())
            errors = [(t, p) for p, t in zip(predicted, truth) if p != t]
            within = sum(1 for t, p in errors if buckets.get(t) == buckets.get(p))
            entry = {
                "assignment": dict(sorted(buckets.items(), key=lambda kv: kv[1])),
                "bucket_sizes": dict(sorted(sizes.items())),
                "evidence": bucket_evidence(buckets, confusion),
                "errors_within_a_bucket": within,
                "errors_between_buckets": len(errors) - within,
                "within_bucket_share": round(within / len(errors), 4) if errors else None,
            }
            for top_k in (1, 2):
                per_fold, stage_a_correct, rescueable = [], 0, 0
                last_model = None
                for indices, _held, pool in plan:
                    model = TwoStage(buckets, labels, top_k=top_k).fit(pool)
                    correct = 0
                    for i in indices:
                        correct += 1 if model.predict(rows[i].text) == rows[i].tool else 0
                        bucket_truth = buckets[rows[i].tool]
                        if model.stage_a(rows[i].text) == bucket_truth:
                            stage_a_correct += 1
                        elif top_k == 1:
                            member = model.member_models.get(bucket_truth)
                            if member is not None:
                                kind, sub = member
                                if kind == "constant":
                                    rescueable += 1 if sub == rows[i].tool else 0
                                else:
                                    rescueable += 1 if sub.predict([rows[i].text])[0] == rows[i].tool else 0
                    per_fold.append(round(correct / len(indices), 4))
                    last_model = model
                indices, _held, _pool = plan[-1]
                texts = [rows[i].text for i in indices][:60]
                entry[f"top{top_k}"] = {
                    "per_fold": per_fold,
                    "fold_mean": round(statistics.mean(per_fold), 4),
                    "fold_sd": round(statistics.stdev(per_fold), 4),
                    "stage_a_accuracy": round(stage_a_correct / len(rows), 4),
                    "unrepairable_rows": len(rows) - stage_a_correct,
                    "stage_b_would_have_right": rescueable,
                    "latency_ms": timed(last_model.predict, texts),
                    "per_fold_vs_flat": [round(100 * (a - b), 2)
                                         for a, b in zip(per_fold, baseline_folds)],
                }
                print(f"{proposal}/{name} top{top_k}: mean {statistics.mean(per_fold):.4f} "
                      f"(sd {statistics.stdev(per_fold):.4f}) · stageA "
                      f"{stage_a_correct / len(rows):.4f} · within-bucket errors "
                      f"{entry['within_bucket_share']} · p50 "
                      f"{entry[f'top{top_k}']['latency_ms']['p50_ms']:.2f} ms", flush=True)
            payload["buckets"].setdefault(proposal, {})[name] = entry

    # ---- the flat path's latency, same measurement style, on the same fold's models
    indices, _held, pool = plan[-1]
    payload["flat"]["latency_ms"] = timed(fit_plain(pool).predict,
                                          [rows[i].text for i in indices][:60])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
