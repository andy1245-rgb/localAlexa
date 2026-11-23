"""Run the six structural variants on the ladder's grouped folds, and the error analysis.

Same folds, same pools, same scoring as the ladder, so every number sits beside the 90.84%
(logreg C=100, grouped) without a footnote. Each arm reports per-fold accuracy, the mean, the
spread across folds, the paired delta against the baseline and how many folds moved which way —
a win smaller than the fold spread is not called a win here either.

    .venv/bin/python experiments/final/six_variants.py --arms fast
    .venv-embed/bin/python experiments/final/six_variants.py --arms all     # adds torch arms

Writes ``results/final/six-variants.json`` and ``results/final/per-intent.md``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from localalexa.cascade import e2e_accuracy
from localalexa.data import NONE_LABEL, load_registry, load_routing_set
from localalexa.ladder.folds import DEFAULT_FOLDS
from localalexa.ladder.pools import fold_pool
from localalexa.ladder.run import load_bench
from localalexa.variants import (
    Fuser,
    Hierarchical,
    NoneFirst,
    choose_class_thresholds,
    fit_plain,
    fit_regex_features,
    logreg,
    threshold_decision,
    word_vectorizer,
)

RESULTS = Path(__file__).resolve().parents[2] / "results" / "final"
TIER2_GROUPED = 0.9503  # the measured ensemble on these folds, from tier2-ensemble-by-split.json


def folds(registry, rows):
    bench = load_bench(DEFAULT_FOLDS)
    families = list(bench.families)
    out = []
    for indices in bench.folds.indices(families):
        held = {families[i] for i in indices}
        out.append((fold_pool(registry, held), indices))
    return out


def evaluate(registry, rows, plan, fit_predict) -> dict:
    """Per-fold accuracy plus the pooled predictions, for one arm."""
    per_fold = []
    predicted: list[str] = [""] * len(rows)
    for pool, indices in plan:
        predict = fit_predict(registry, pool)
        hits = 0
        for i in indices:
            label = predict(rows[i].text)
            predicted[i] = label
            hits += 1 if label == rows[i].tool else 0
        per_fold.append({"n": len(indices), "correct": hits, "accuracy": round(hits / len(indices), 4)})
    total = sum(f["correct"] for f in per_fold)
    return {
        "per_fold": [f["accuracy"] for f in per_fold],
        "correct": total,
        "n": len(rows),
        "accuracy": round(total / len(rows), 4),
        "spread": {
            "mean": round(statistics.mean(f["accuracy"] for f in per_fold), 4),
            "sd": round(statistics.stdev(f["accuracy"] for f in per_fold), 4),
            "min": round(min(f["accuracy"] for f in per_fold), 4),
            "max": round(max(f["accuracy"] for f in per_fold), 4),
        },
        "predictions": predicted,
    }


def paired(fold_scores, baseline_scores) -> dict:
    deltas = [a - b for a, b in zip(fold_scores, baseline_scores)]
    return {
        "mean_points": round(100.0 * statistics.mean(deltas), 2),
        "sd_points": round(100.0 * statistics.stdev(deltas), 2) if len(deltas) > 1 else 0.0,
        "folds_better": sum(1 for d in deltas if d > 0),
        "folds_worse": sum(1 for d in deltas if d < 0),
        "folds_tied": sum(1 for d in deltas if d == 0),
    }


# ------------------------------------------------------------------ the arms


def arm_baseline(registry, pool):
    model = fit_plain(pool)
    return model.predict


def arm_regex_features(registry, pool):
    model = fit_regex_features(registry, pool)
    return model.predict


def arm_fuse_linear_regex(registry, pool):
    model = Fuser(("linear", "regex")).fit(registry, pool)
    return model.predict


def arm_fuse_router(registry, pool):
    from localalexa.router import SemanticRouter

    router = SemanticRouter(registry)
    router.predict("warm up")
    model = Fuser(("linear", "regex", "router")).fit(registry, pool, router=router)
    return model.predict


def arm_none_first(registry, pool):
    model = NoneFirst().fit(pool)
    return model.predict


def arm_hierarchical(registry, pool):
    model = Hierarchical().fit(registry, pool)
    return model.predict


def arm_distilled(registry, pool):
    """Teacher = the measured ensemble; student = one fast logreg on its answers."""
    from localalexa.ladder.ensemble_rung import rung_5_ensemble_setfit_standardised
    from localalexa.ladder.predictors import predict_rows
    from localalexa.data import expand_template

    teacher = rung_5_ensemble_setfit_standardised().build().fit(
        [text for text, _ in pool], [label for _, label in pool]
    )
    # unlabelled rows the student may use: template fills the *teacher's* pool cannot see
    held_texts = {text for text, _ in pool}
    unlabelled: list[str] = []
    for tool in registry.tools:
        for template in tool.templates:
            for text in expand_template(tool, template):
                if text not in held_texts:
                    unlabelled.append(text)
    pseudo = predict_rows(teacher, unlabelled) if unlabelled else []
    student_pool = list(pool) + list(zip(unlabelled, pseudo))
    student = fit_plain(student_pool)
    return student.predict


def per_class_thresholds(plan, registry, rows) -> dict:
    """Per-intent confidence floors chosen on out-of-fold pool predictions, then applied."""
    from sklearn.pipeline import make_pipeline

    from localalexa.ladder.folds import build_folds

    targets = {"0.70": 0.70, "0.80": 0.80, "0.90": 0.90}
    out: dict = {"targets": {}, "unanswerable_classes": {}}
    for key, target in targets.items():
        answered = correct = 0
        per_fold = []
        unanswerable = 0
        answered_by_class: dict[str, int] = {}
        correct_by_class: dict[str, int] = {}
        for pool, indices in plan:
            texts = [text for text, _ in pool]
            truth = [label for _, label in pool]
            # group the inner folds by *text*, not by label: grouping by label would hold whole
            # classes out of each calibration fit, so the out-of-fold distribution a threshold
            # is chosen on would be missing the classes it has to score
            inner = build_folds(texts, 3)
            oof: list[dict[str, float]] = [{} for _ in texts]
            for inner_indices in inner.indices(texts):
                held = set(inner_indices)
                subset = [(t, l) for i, (t, l) in enumerate(pool) if i not in held]
                pipeline = make_pipeline(word_vectorizer(), logreg())
                pipeline.fit([t for t, _ in subset], [l for _, l in subset])
                classes = [str(c) for c in pipeline.classes_]
                for i in inner_indices:
                    row = pipeline.predict_proba([texts[i]])[0]
                    oof[i] = {label: float(p) for label, p in zip(classes, row)}
            thresholds = choose_class_thresholds(oof, truth, target_precision=target)
            unanswerable += sum(1 for value in thresholds.thresholds.values() if value > 1.0)
            pipeline = make_pipeline(word_vectorizer(), logreg())
            pipeline.fit(texts, truth)
            classes = [str(c) for c in pipeline.classes_]
            for i in indices:
                row = pipeline.predict_proba([rows[i].text])[0]
                distribution = {label: float(p) for label, p in zip(classes, row)}
                label, _score = threshold_decision(distribution, thresholds.thresholds)
                if label is None:
                    per_fold.append(0)
                    continue
                answered += 1
                per_fold.append(1)
                answered_by_class[label] = answered_by_class.get(label, 0) + 1
                right = label == rows[i].tool
                correct += 1 if right else 0
                if right:
                    correct_by_class[label] = correct_by_class.get(label, 0) + 1
        escalated = len(rows) - answered
        out["targets"][key] = {
            "target_precision": target,
            "answered": answered,
            "coverage": round(answered / len(rows), 4),
            "correct": correct,
            "precision": round(correct / answered, 4) if answered else None,
            "escalated": escalated,
            "e2e_at_tier2_measured": round(
                e2e_accuracy(correct, escalated, len(rows), TIER2_GROUPED) / 100.0, 4),
            "answered_by_class": dict(sorted(answered_by_class.items(), key=lambda kv: -kv[1])),
            "classes_that_answer": len(answered_by_class),
            "classes_that_cannot": unanswerable,
        }
        out["unanswerable_classes"][key] = unanswerable
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arms", default="fast", choices=("fast", "all"))
    parser.add_argument("--out", default=str(RESULTS / "six-variants.json"))
    args = parser.parse_args(argv)

    registry = load_registry()
    rows = load_routing_set()
    plan = folds(registry, rows)

    arms = {
        "baseline_logreg": arm_baseline,
        "regex_features": arm_regex_features,
        "fuse_linear_regex": arm_fuse_linear_regex,
        "none_first": arm_none_first,
        "hierarchical": arm_hierarchical,
    }
    if args.arms == "all":
        arms["fuse_router"] = arm_fuse_router
        arms["distilled"] = arm_distilled

    results = {}
    for name, fit_predict in arms.items():
        block = evaluate(registry, rows, plan, fit_predict)
        results[name] = block
        print(f"{name:20s} {block['correct']:3d}/382 = {block['accuracy']:.4f} "
              f"(fold sd {block['spread']['sd']:.4f})", flush=True)

    baseline = results["baseline_logreg"]
    for name, block in results.items():
        if name == "baseline_logreg":
            continue
        block["paired_vs_baseline"] = paired(block["per_fold"], baseline["per_fold"])

    # (6) per-class thresholds, measured the same way: escalate instead of guessing. Three
    # targets, because a single one at 0.90 turns out to answer nothing and that is a result,
    # not a number to hide.
    results["per_class_thresholds"] = per_class_thresholds(plan, registry, rows)
    for target, block in results["per_class_thresholds"]["targets"].items():
        print(f"per-class @ {target:>4s}     coverage {100 * block['coverage']:5.1f}% at precision "
              f"{100 * (block['precision'] or 0):5.2f}% (answered {block['answered']}, "
              f"e2e {100 * block['e2e_at_tier2_measured']:.2f}%)", flush=True)

    RESULTS.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps({"folds": DEFAULT_FOLDS, "tier2_grouped": TIER2_GROUPED, "arms": results},
                   indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
