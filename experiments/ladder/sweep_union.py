"""Rung 1, taken seriously: does *any* word+char union beat word-only?

The rung as written (union, `LinearSVC(C=1.0)`) comes out behind the baseline.
Before that is written down as a refutation, the union gets the sweep the
baseline never needed: the concatenated feature space changes the scale of the
decision values, so C is no longer a free parameter, and the classifier on top
is part of the same choice.

Reported per variant: mean per-fold accuracy on the shared grouped folds, the
paired per-fold delta against rung 0, and the p50 latency.

    .venv/bin/python experiments/ladder/sweep_union.py

Selection happens on these folds, so the winner's number is optimistic — it is
re-checked on a second fold assignment by ``ladder_report.py``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.svm import LinearSVC

from localalexa.ladder import metrics
from localalexa.ladder.predictors import SklearnPredictor
from localalexa.ladder.run import Rung, evaluate_grouped, load_bench


def word(ngram=(1, 2), min_df=1):
    return TfidfVectorizer(ngram_range=ngram, min_df=min_df, sublinear_tf=True)


def char(ngram=(3, 5), min_df=1):
    return TfidfVectorizer(analyzer="char_wb", ngram_range=ngram, min_df=min_df, sublinear_tf=True)


def union(word_ngram=(1, 2), char_ngram=(3, 5), min_df=1):
    return FeatureUnion(
        [("word", word(word_ngram, min_df)), ("char", char(char_ngram, min_df))]
    )


VARIANTS = {
    "r0 word12 svc C=1": lambda: make_pipeline(word(), LinearSVC(C=1.0, random_state=0)),
    "r1 union svc C=1": lambda: make_pipeline(union(), LinearSVC(C=1.0, random_state=0)),
    "r1 union svc C=0.25": lambda: make_pipeline(union(), LinearSVC(C=0.25, random_state=0)),
    "r1 union svc C=4": lambda: make_pipeline(union(), LinearSVC(C=4.0, random_state=0)),
    "r1 union svc C=1 mdf2": lambda: make_pipeline(union(min_df=2), LinearSVC(C=1.0, random_state=0)),
    "r1 union svc C=4 mdf2": lambda: make_pipeline(union(min_df=2), LinearSVC(C=4.0, random_state=0)),
    "r1 union logreg C=10": lambda: make_pipeline(
        union(), LogisticRegression(max_iter=2000, C=10.0, random_state=0)
    ),
    "r1 union logreg C=100": lambda: make_pipeline(
        union(), LogisticRegression(max_iter=2000, C=100.0, random_state=0)
    ),
    "r1 char35 svc C=1": lambda: make_pipeline(char(), LinearSVC(C=1.0, random_state=0)),
    "r1 char25 svc C=1": lambda: make_pipeline(char((2, 5)), LinearSVC(C=1.0, random_state=0)),
    "r1 word13 svc C=1": lambda: make_pipeline(word((1, 3)), LinearSVC(C=1.0, random_state=0)),
    "r1 union23 svc C=1": lambda: make_pipeline(
        union((1, 2), (2, 5)), LinearSVC(C=1.0, random_state=0)
    ),
}


def main() -> int:
    bench = load_bench()
    rows = []
    baseline_folds = None
    for name, build in VARIANTS.items():
        rung = Rung(name, lambda b=build: SklearnPredictor(b))
        block = evaluate_grouped(rung, bench)
        folds = block["per_fold_accuracy"]
        if baseline_folds is None:
            baseline_folds = folds
        rows.append(
            {
                "variant": name,
                "grouped_accuracy": block["accuracy"],
                "fold_mean": block["fold_spread"]["mean"],
                "fold_sd": block["fold_spread"]["sd"],
                "fold_min": block["fold_spread"]["min"],
                "fold_max": block["fold_spread"]["max"],
                "paired_vs_r0": metrics.paired_delta(folds, baseline_folds),
                "latency_p50_ms": block["latency_ms"]["p50"],
                "none_recall": block["none_recall"],
                "reject_curve": block["reject_curve"],
            }
        )
        print(
            f"{name:26s} acc={block['accuracy']:.4f} foldmean={block['fold_spread']['mean']:.4f} "
            f"delta={rows[-1]['paired_vs_r0']['mean_points']:+.2f}±{rows[-1]['paired_vs_r0']['sd_points']:.2f} "
            f"p50={block['latency_ms']['p50']:.2f}ms none={block['none_recall']}",
            flush=True,
        )

    out = Path(__file__).resolve().parents[2] / "results" / "ladder" / "sweep_union.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {"folds": bench.folds.as_dict(), "variants": rows},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
