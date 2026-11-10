"""Rung 2: probabilities, and the reject curve they make possible.

`LinearSVC` decides; it does not estimate. Softmax over its decision values
gives a confidence that is monotone-ish but not a probability, and the reject
curve built on it is meaningless — the baseline's softmax scores are so sharp
that "coverage at 90% precision" comes back as 99.5%, which is exactly the sort
of number that ships a broken product.

This sweep scores the candidates on what a voice assistant actually needs:

* coverage at 90% precision (how much of the traffic the cheap tier may answer)
* accuracy on the rejected residue (what the expensive tier inherits)
* ``__none__`` recall (40% of "this is not a command" being acted on is the defect)
* ECE, and plain accuracy so a calibration win cannot hide an accuracy loss

The threshold in ``coverage_at_precision_0.9`` is picked on the rows being
scored, so it is an *oracle* operating point: the ceiling, not the shipped
number. ``--nested`` re-picks it inside the training pool (grouped inner folds)
for the variants worth the extra fits.

    .venv/bin/python experiments/ladder/sweep_calibration.py [--nested]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.svm import LinearSVC

from localalexa.ladder import metrics
from localalexa.ladder.predictors import SklearnPredictor
from localalexa.ladder.pools import fold_pool
from localalexa.ladder.run import block_metrics, load_bench, predict_with_confidence
from localalexa.ladder.folds import build_folds


def word(ngram=(1, 2)):
    return TfidfVectorizer(ngram_range=ngram, min_df=1, sublinear_tf=True)


def union():
    return FeatureUnion(
        [
            ("word", word()),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True
                ),
            ),
        ]
    )


VARIANTS = {
    "r0 word12 svc C=1 (softmax)": lambda: make_pipeline(word(), LinearSVC(C=1.0, random_state=0)),
    "r2-word12 logreg C=1": lambda: make_pipeline(word(), LogisticRegression(max_iter=2000, C=1.0, random_state=0)),
    "r2-word12 logreg C=10": lambda: make_pipeline(word(), LogisticRegression(max_iter=2000, C=10.0, random_state=0)),
    "r2-word12 logreg C=100": lambda: make_pipeline(word(), LogisticRegression(max_iter=2000, C=100.0, random_state=0)),
    "r2-word12 svc sigmoid cv3": lambda: make_pipeline(
        word(), CalibratedClassifierCV(LinearSVC(C=1.0, random_state=0), method="sigmoid", cv=3)
    ),
    "r2-word12 svc sigmoid cv5": lambda: make_pipeline(
        word(), CalibratedClassifierCV(LinearSVC(C=1.0, random_state=0), method="sigmoid", cv=5)
    ),
    "r2-word12 svc isotonic cv3": lambda: make_pipeline(
        word(), CalibratedClassifierCV(LinearSVC(C=1.0, random_state=0), method="isotonic", cv=3)
    ),
    "r2-union svc sigmoid cv3": lambda: make_pipeline(
        union(), CalibratedClassifierCV(LinearSVC(C=1.0, random_state=0), method="sigmoid", cv=3)
    ),
    "r2-union logreg C=10": lambda: make_pipeline(union(), LogisticRegression(max_iter=2000, C=10.0, random_state=0)),
    "r2-word12 logreg C=100 bal": lambda: make_pipeline(
        word(), LogisticRegression(max_iter=2000, C=100.0, class_weight="balanced", random_state=0)
    ),
    "r2-word12 svc C=1 bal": lambda: make_pipeline(
        word(), LinearSVC(C=1.0, class_weight="balanced", random_state=0)
    ),
}


def build(factory) -> SklearnPredictor:
    return SklearnPredictor(factory)


def main(argv: list[str]) -> int:
    nested = "--nested" in argv
    bench = load_bench()
    base_dir = Path(__file__).resolve().parents[2] / "results" / "ladder"
    base_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    baseline_folds = None
    for name, factory in VARIANTS.items():
        predicted = [""] * len(bench.rows)
        confidences = [0.0] * len(bench.rows)
        fold_scores = []
        for indices in bench.folds.indices(list(bench.families)):
            held = {bench.families[i] for i in indices}
            pool = fold_pool(bench.registry, held)
            model = build(factory).fit([t for t, _ in pool], [l for _, l in pool])
            texts = [bench.rows[i].text for i in indices]
            out, conf = predict_with_confidence(model, texts)
            for i, label, score in zip(indices, out, conf):
                predicted[i] = label
                confidences[i] = score
            fold_scores.append(metrics.accuracy(out, [bench.rows[i].tool for i in indices]))
        if baseline_folds is None:
            baseline_folds = fold_scores
        block = block_metrics(predicted, confidences, bench.truth, bench.families)
        entry = {
            "variant": name,
            "grouped_accuracy": block["accuracy"],
            "paired_vs_r0": metrics.paired_delta(fold_scores, baseline_folds),
            "ece": block["ece"],
            "none_recall": block["none_recall"],
            "false_none_rate": block["false_none_rate"],
            "reject_curve_oracle": block["reject_curve"],
        }
        if nested:
            entry["reject_curve_nested"] = _nested_coverage(factory, bench)
        rows.append(entry)
        curve = {p["target_precision"]: p for p in block["reject_curve"]["points"]}
        print(
            f"{name:30s} acc={block['accuracy']:.4f} ece={block['ece']:.3f} "
            f"cov@.95={curve[0.95]['coverage']:.3f}/res={curve[0.95]['residue_accuracy']:.3f} "
            f"cov@.975={curve[0.975]['coverage']:.3f}/res={curve[0.975]['residue_accuracy']:.3f} "
            f"none={block['none_recall']:.2f} fnone={block['false_none_rate']:.3f}"
            + (f" nested_cov={entry['reject_curve_nested']['coverage']:.3f}" if nested else ""),
            flush=True,
        )

    out = base_dir / ("sweep_calibration_nested.json" if nested else "sweep_calibration.json")
    out.write_text(json.dumps({"folds": bench.folds.as_dict(), "variants": rows}, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


def _nested_coverage(factory, bench) -> dict[str, float]:
    """Coverage at 90% precision when the floor is picked inside each fold's pool."""
    covered = 0
    hits = 0
    total = 0
    for indices in bench.folds.indices(list(bench.families)):
        held = {bench.families[i] for i in indices}
        pool = fold_pool(bench.registry, held)
        floor = _inner_floor(factory, pool)
        model = build(factory).fit([t for t, _ in pool], [l for _, l in pool])
        texts = [bench.rows[i].text for i in indices]
        truth = [bench.rows[i].tool for i in indices]
        out, conf = predict_with_confidence(model, texts)
        for label, score, want in zip(out, conf, truth):
            if score >= floor:
                covered += 1
                hits += 1 if label == want else 0
            total += 1
    return {
        "coverage": round(covered / total, 4),
        "precision": round(hits / covered, 4) if covered else 0.0,
    }


def _inner_floor(factory, pool) -> float:
    """Grouped 5-fold out-of-fold confidence floor that reaches 90% precision."""
    texts = [t for t, _ in pool]
    truth = [l for _, l in pool]
    inner = build_folds(texts, 5)
    predicted = [""] * len(texts)
    confidences = [0.0] * len(texts)
    for indices in inner.indices(texts):
        held = set(indices)
        subset = [(t, l) for i, (t, l) in enumerate(pool) if i not in held]
        model = build(factory).fit([t for t, _ in subset], [l for _, l in subset])
        rows = [texts[i] for i in indices]
        out, conf = predict_with_confidence(model, rows)
        for i, label, score in zip(indices, out, conf):
            predicted[i] = label
            confidences[i] = score
    return float(metrics.reject_curve(confidences, predicted, truth)["threshold"])


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
