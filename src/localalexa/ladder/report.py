"""Assemble the ladder artifact: every rung, both protocols, one fold assignment.

The artifact is the result. It carries the fold assignment, the per-intent
ledger, the reject curve, the paired per-fold delta against rung 0, and the
error correlation between rungs — so a later reader can check the claim instead
of trusting it.

A rung is reported next to rung 0 on the *same folds*, and the number that
decides anything is the paired delta: the spread of the difference, not the
spread of the absolute accuracy, is what a win has to beat.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

from . import metrics
from .run import Bench, Rung, evaluate_grouped, evaluate_rowwise, load_bench


def error_correlation(predictions: dict[str, Sequence[str]], truth: Sequence[str]) -> list[dict]:
    """Pairwise disagreement and error agreement between rungs.

    Two models that are wrong on the same rows make an ensemble that is slower
    and no better; this table is what says whether stacking is worth trying
    before any stacking is written.
    """
    names = list(predictions)
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            pa, pb = predictions[a], predictions[b]
            disagree = sum(1 for x, y in zip(pa, pb) if x != y) / len(truth)
            wrong_a = {k for k, (p, t) in enumerate(zip(pa, truth)) if p != t}
            wrong_b = {k for k, (p, t) in enumerate(zip(pb, truth)) if p != t}
            both = len(wrong_a & wrong_b)
            union = len(wrong_a | wrong_b) or 1
            rows.append(
                {
                    "a": a,
                    "b": b,
                    "disagreement_rate": round(disagree, 4),
                    "errors_a": len(wrong_a),
                    "errors_b": len(wrong_b),
                    "errors_both": both,
                    "error_jaccard": round(both / union, 4),
                    "only_a": len(wrong_a - wrong_b),
                    "only_b": len(wrong_b - wrong_a),
                }
            )
    return rows


def run_ladder(
    rungs: Iterable[Rung],
    *,
    fold_counts: Sequence[int] = (5,),
    protocols: Sequence[str] = ("grouped", "rowwise"),
    keep_predictions: bool = False,
) -> dict[str, object]:
    rungs = list(rungs)
    if not rungs:
        raise ValueError("no rungs to run")

    runs = []
    comparisons = []
    correlations: dict[str, list[dict]] = {}
    fold_assignments = {}
    all_predictions: dict[str, list[str]] = {}

    for n_folds in fold_counts:
        bench = load_bench(n_folds)
        if n_folds == fold_counts[0]:
            fold_assignments[str(n_folds)] = bench.folds.as_dict()
        for protocol in protocols:
            evaluate = evaluate_grouped if protocol == "grouped" else evaluate_rowwise
            predictions: dict[str, list[str]] = {}
            baseline_folds: list[float] | None = None
            for rung in rungs:
                block = evaluate(rung, bench)
                predictions[rung.name] = block.pop("predictions")
                runs.append(
                    {
                        "rung": rung.name,
                        "note": rung.note,
                        "fold_count": n_folds if protocol == "grouped" else None,
                        **block,
                    }
                )
                if baseline_folds is None:
                    baseline_folds = block["per_fold_accuracy"] or None
                elif block["per_fold_accuracy"]:
                    comparisons.append(
                        {
                            "rung": rung.name,
                            "baseline": rungs[0].name,
                            "fold_count": n_folds,
                            "protocol": protocol,
                            "paired": metrics.paired_delta(
                                block["per_fold_accuracy"], baseline_folds
                            ),
                        }
                    )
            if protocol == "grouped" and n_folds == fold_counts[0]:
                correlations[str(n_folds)] = error_correlation(predictions, bench.truth)
                all_predictions = predictions

    payload: dict[str, object] = {
        "bench": bench_summary(load_bench(fold_counts[0])),
        "fold_assignments": fold_assignments,
        "runs": runs,
        "comparisons": comparisons,
        "error_correlation": correlations,
        "protocol_notes": {
            "rowwise": "train on all 200 references, score all 382 rows — the protocol "
            "the 93.19% / 0.61 ms baseline was measured under",
            "grouped": "folds over phrasing families; a fold trains on references that do "
            "not render into that fold's families",
            "reject_curve": "thresholds chosen on the scored rows: oracle operating "
            "points, the ceiling, not the shipped number",
        },
    }
    if keep_predictions:
        payload["predictions"] = all_predictions
    return payload


def bench_summary(bench: Bench) -> dict[str, object]:
    references = sum(len(tool.references) for tool in bench.registry.tools)
    references += len(bench.registry.none_references)
    return {
        "n_rows": len(bench.rows),
        "n_families": len(set(bench.families)),
        "n_labels": len(bench.labels),
        "n_reference_rows": references,
        "examples_per_intent": round(len(bench.rows) / len(bench.labels), 1),
    }


def write_artifact(payload: dict[str, object], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return target


def format_table(payload: dict[str, object]) -> str:
    lines = []
    header = f"{'rung':28s} {'proto':8s} {'folds':>5s} {'accuracy':>9s} {'p50 ms':>8s} {'ece':>6s} {'cov@.95':>8s} {'res@.95':>8s} {'none':>5s} {'delta':>12s}"
    lines.append(header)
    lines.append("-" * len(header))
    comparisons = {
        (c["rung"], c["fold_count"]): c["paired"] for c in payload["comparisons"]
    }
    for run in payload["runs"]:
        curve = {p["target_precision"]: p for p in run["reject_curve"]["points"]}
        point = curve[0.95]
        delta = comparisons.get((run["rung"], run["fold_count"]))
        delta_text = (
            f"{delta['mean_points']:+.2f}±{delta['sd_points']:.2f}"
            if delta
            else ("baseline" if run["rung"].startswith("r0") else "—")
        )
        lines.append(
            f"{run['rung']:28s} {run['protocol']:8s} "
            f"{str(run['fold_count'] or '—'):>5s} "
            f"{run['accuracy']:9.4f} {run['latency_ms']['p50']:8.2f} "
            f"{run['ece']:6.3f} {point['coverage']:8.3f} {point['residue_accuracy']:8.3f} "
            f"{run['none_recall'] if run['none_recall'] is not None else float('nan'):5.2f} "
            f"{delta_text:>12s}"
        )
    return "\n".join(lines)
