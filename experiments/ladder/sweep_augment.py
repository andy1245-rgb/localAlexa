"""Rung 6a: how much of the registry's own slot fills the references will take.

Adding every fill of every non-held-out template swamps 194 hand-written
references with 1038 generated ones — ``reminder_create`` alone goes from 12 rows
to 396 — and the grouped number falls. This sweep measures the dose instead of
guessing it: cap the fills per label, and try a class-balanced loss on top.

    .venv/bin/python experiments/ladder/sweep_augment.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from localalexa.classifiers import SEED
from localalexa.data import load_registry
from localalexa.ladder import metrics
from localalexa.ladder.augment_rung import template_augmenter
from localalexa.ladder.predictors import SklearnPredictor
from localalexa.ladder.rungs import rung_0_baseline, rung_2_logreg_c100, word_vectorizer
from localalexa.ladder.run import Rung, evaluate_grouped, load_bench


def logreg(**kwargs):
    return lambda: SklearnPredictor(
        lambda: make_pipeline(
            word_vectorizer(), LogisticRegression(max_iter=2000, random_state=SEED, **kwargs)
        )
    )


def main() -> int:
    bench = load_bench(5)
    registry = load_registry()

    candidates = [("r0 baseline", rung_0_baseline()), ("r2 logreg C=100", rung_2_logreg_c100())]
    for cap in (8, 16, 32, 64, None):
        candidates.append(
            (
                f"r6 caps={cap}",
                Rung(
                    f"r6-tmpl-cap{cap}",
                    logreg(C=100.0),
                    augment=template_augmenter(registry, cap_per_label=cap),
                ),
            )
        )
    candidates.append(
        (
            "r6 caps=16 balanced",
            Rung(
                "r6-tmpl-cap16-bal",
                logreg(C=100.0, class_weight="balanced"),
                augment=template_augmenter(registry, cap_per_label=16),
            ),
        )
    )

    baseline_folds = None
    rows = []
    for label, rung in candidates:
        block = evaluate_grouped(rung, bench)
        folds = block["per_fold_accuracy"]
        if baseline_folds is None:
            baseline_folds = folds
        rows.append(
            {
                "variant": label,
                "train_rows_per_fold": block["train_rows"],
                "grouped_accuracy": block["accuracy"],
                "paired_vs_r0": metrics.paired_delta(folds, baseline_folds),
                "ece": block["ece"],
                "none_recall": block["none_recall"],
                "reject_curve": block["reject_curve"],
                "latency_p50_ms": block["latency_ms"]["p50"],
            }
        )
        print(
            f"{label:22s} acc={block['accuracy']:.4f} "
            f"delta={rows[-1]['paired_vs_r0']['mean_points']:+.2f}+-{rows[-1]['paired_vs_r0']['sd_points']:.2f} "
            f"ece={block['ece']:.3f} none={block['none_recall']:.2f} "
            f"train={block['train_rows']}",
            flush=True,
        )

    out = Path(__file__).resolve().parents[2] / "results" / "ladder" / "sweep_augment.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"folds": bench.folds.as_dict(), "variants": rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
