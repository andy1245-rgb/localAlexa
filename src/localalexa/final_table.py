"""The final table: the cascade and the single classifiers, side by side, on three splits.

Every number here is measured under the split it is quoted beside, and the tier-2 accuracy the
cascade's end-to-end figure multiplies is the measured ensemble's accuracy *for that split* — passed
in from ``experiments/final/tier2_ensemble.py`` rather than assumed, because mixing a grouped
95.03% into a rowwise cascade is precisely the error this repo spent a week correcting.

Three splits, the three the README's own table uses:

``rowwise``
    fit on the 200 references, score all 382 rows. Inflated: some references are literal slot
    fills of the templates that generated the scored rows.
``family``
    five folds over the 110 phrasing families; a fold fits on the reference pool minus the slot
    fills of its held-out families.
``strict``
    the same folds, with the pool rebuilt from the other families' template fills and no
    hand-written references at all.

Three arms per split:

``tier 1 alone``
    word TF-IDF + logistic regression C=100 answering every row.
``tier 2 alone``
    the measured ensemble (logreg + SetFit, row-standardised) answering every row. This is the
    best single classifier the repo has, and it is the thing the cascade has to beat to be worth
    its structure.
``cascade``
    tier 0 regex, else tier 1 when the gate allows, else escalate to tier 2 at the measured
    accuracy. Reported as coverage, accuracy on what it answers, and end-to-end.

The end-to-end figure is ``(auto_correct + x · escalated) / n`` with ``x`` measured per split, and
the break-even column says what ``x`` would have to be for the cascade to match the best single
classifier. When the break-even is above 1.0 the cascade cannot win at any tier-2 accuracy, which
is the case the table is designed to expose.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

from .cascade import CompiledPatterns, e2e_accuracy, tier2_breakeven
from .compare import _linear_tier1
from .data import Registry, load_registry, load_routing_set
from .evaluate import _strict_training_pool
from .ladder.folds import DEFAULT_FOLDS, build_folds
from .ladder.pools import family_texts, fold_pool, reference_pool

SPLITS = ("rowwise", "family", "strict")
SPLIT_NOTES = {
    "rowwise": "fit on the 200 references, score all 382 rows (inflated: 29 references are "
    "literal slot fills of the scoring templates)",
    "family": "5 folds over the 110 phrasing families; reference pool minus the held-out "
    "families' slot fills",
    "strict": "same folds; pool rebuilt from the other families' template fills, no references, "
    "with any text a held-out family can render removed by text (template-identity filtering "
    "leaks 132-340 rows per fold here)",
}
# Latency measured elsewhere, quoted so the table can be argued about on both axes:
# ladder/latency_pass.py (idle box, median of five passes) and ladder_results.
LATENCY_MS = {"tier1": 0.38, "tier2": 7.85}


def _pool_and_rows(
    split: str, registry: Registry, rows, families, folds: int
) -> list[tuple[list[tuple[str, str]], list[int]]]:
    """(training pool, scored rows) for each fit this split needs."""
    assignment = build_folds(families, folds)
    if split == "rowwise":
        return [(reference_pool(registry), list(range(len(rows))))]
    if split not in ("family", "strict"):
        raise ValueError(f"unknown split {split!r}; known: {', '.join(SPLITS)}")
    plan = []
    for indices in assignment.indices(list(families)):
        held = {families[i] for i in indices}
        if split == "family":
            pool = fold_pool(registry, held)
        else:
            # ``_strict_training_pool`` drops a template by its *string*, which is not enough
            # once folds hold out many families at once: another template can render the same
            # utterance, and on this corpus 132-340 of the pool rows do just that (measured:
            # 77 of 77 scored rows present verbatim before this filter, 0 after). Filtering by
            # the rendered text is the same rule ``fold_pool`` uses for the non-strict split.
            leaked = family_texts(registry, held)
            pool = [(text, label) for text, label in _strict_training_pool(registry, held)
                    if text not in leaked]
        plan.append((pool, indices))
    return plan


def measure_split(
    split: str,
    registry: Registry,
    rows,
    families,
    *,
    folds: int = DEFAULT_FOLDS,
    gate: float = 0.35,
    tier2_accuracy: float | None = None,
) -> dict[str, object]:
    """Both single classifiers and the cascade, for one split."""
    compiled = CompiledPatterns.from_registry(registry)
    n = len(rows)
    tier1_correct = tier2_correct = 0
    auto_correct = escalated = 0
    tier0_rows = tier0_correct = 0
    tier1_rows = tier1_correct_answered = 0

    for pool, indices in _pool_and_rows(split, registry, rows, families, folds):
        tier1 = _linear_tier1(pool)
        for i in indices:
            row = rows[i]
            label, confidence, _ = tier1(row.text)
            tier1_correct += 1 if label == row.tool else 0
            hits = compiled.probe(row.text)
            if len(hits) == 1:
                predicted, tier = hits[0], 0
            elif confidence < gate:
                escalated += 1
                continue
            else:
                predicted, tier = label, 1
            right = predicted == row.tool
            auto_correct += 1 if right else 0
            if tier == 0:
                tier0_rows += 1
                tier0_correct += 1 if right else 0
            else:
                tier1_rows += 1
                tier1_correct_answered += 1 if right else 0

    tier2_correct = round(tier2_accuracy * n) if tier2_accuracy is not None else None
    return {
        "split": split,
        "note": SPLIT_NOTES[split],
        "n": n,
        "tier1_alone": {
            "correct": tier1_correct,
            "accuracy": round(tier1_correct / n, 4),
            "latency_ms": LATENCY_MS["tier1"],
        },
        "tier2_alone": {
            "correct": tier2_correct,
            "accuracy": None if tier2_accuracy is None else round(tier2_accuracy, 4),
            "latency_ms": LATENCY_MS["tier2"],
            "measured": tier2_accuracy is not None,
        },
        "cascade": {
            "gate": gate,
            "escalated": escalated,
            "coverage": round((n - escalated) / n, 4),
            "auto_correct": auto_correct,
            "auto_accuracy": round(auto_correct / (n - escalated), 4) if n > escalated else None,
            "tier0_rows": tier0_rows,
            "tier0_accuracy": round(tier0_correct / tier0_rows, 4) if tier0_rows else None,
            "tier1_rows": tier1_rows,
            "tier1_accuracy": round(tier1_correct_answered / tier1_rows, 4) if tier1_rows else None,
            "e2e_at_measured_tier2": (
                None
                if tier2_accuracy is None
                else round(e2e_accuracy(auto_correct, escalated, n, tier2_accuracy) / 100.0, 4)
            ),
            "breakeven_vs_tier2_alone": (
                None
                if tier2_correct is None
                else tier2_breakeven(auto_correct, escalated, n, tier2_correct)
            ),
            "breakeven_vs_tier1_alone": tier2_breakeven(
                auto_correct, escalated, n, tier1_correct
            ),
        },
    }


def load_tier2(path: str | Path) -> dict[str, float]:
    """Per-split tier-2 accuracies from the ensemble measurement."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        split: float(block["accuracy"])
        for split, block in payload.get("splits", {}).items()
        if block.get("accuracy") is not None
    }


def build(
    *,
    tier2_path: str | Path | None = None,
    tier2: Mapping[str, float] | None = None,
    folds: int = DEFAULT_FOLDS,
    gate: float = 0.35,
    registry: Registry | None = None,
    splits: Sequence[str] = SPLITS,
) -> dict[str, object]:
    registry = registry or load_registry()
    rows = load_routing_set()
    bench_families = _families(registry, rows)
    measured = dict(tier2 or {})
    if tier2_path is not None:
        measured.update(load_tier2(tier2_path))
    blocks = [
        measure_split(
            split,
            registry,
            rows,
            bench_families,
            folds=folds,
            gate=gate,
            tier2_accuracy=measured.get(split),
        )
        for split in splits
    ]
    return {
        "meta": {
            "folds": folds,
            "gate": gate,
            "tier2": "logreg + SetFit, row-standardised average (ladder rung 5 winner)",
            "tier2_accuracies_used": {k: round(v, 4) for k, v in measured.items()},
            "protocol": "each split measured under its own protocol; the cascade's end-to-end "
            "figure multiplies the tier-2 accuracy measured for that same split",
        },
        "splits": blocks,
    }


def _families(registry: Registry, rows) -> list[str]:
    from .data import phrasing_families, template_index

    return phrasing_families(rows, template_index(registry))


def render(payload: Mapping[str, object]) -> str:
    """The table, with the column that decides it on the left."""
    meta = payload["meta"]
    lines = [
        f"folds {meta['folds']} · cascade gate {meta['gate']} · tier 2 = {meta['tier2']}",
        "",
        f"{'split':10s} {'arm':14s} {'correct':>9s} {'accuracy':>9s} {'ms':>6s}  note",
        "-" * 92,
    ]
    for block in payload["splits"]:
        tier1 = block["tier1_alone"]
        tier2 = block["tier2_alone"]
        cascade = block["cascade"]
        lines.append(
            f"{block['split']:10s} {'tier 1 (logreg)':14s} {tier1['correct']:5d}/{block['n']:<3d} "
            f"{100 * tier1['accuracy']:8.2f}% {tier1['latency_ms']:6.2f}  answers every row"
        )
        if tier2["measured"]:
            lines.append(
                f"{'':10s} {'tier 2 (ens)':14s} {tier2['correct']:5d}/{block['n']:<3d} "
                f"{100 * tier2['accuracy']:8.2f}% {tier2['latency_ms']:6.2f}  answers every row"
            )
        else:
            lines.append(
                f"{'':10s} {'tier 2 (ens)':14s} {'—':>9s} {'not measured':>9s} "
                f"{tier2['latency_ms']:6.2f}  no measurement for this split"
            )
        coverage = 100 * cascade["coverage"]
        auto = 100 * (cascade["auto_accuracy"] or 0)
        lines.append(
            f"{'':10s} {'cascade':14s} {'':9s} {'':9s} {'':6s}  answers {coverage:.1f}% itself "
            f"at {auto:.2f}%, escalates {cascade['escalated']}"
        )
        if cascade["e2e_at_measured_tier2"] is not None:
            lines.append(
                f"{'':10s} {'  end to end':14s} {'':9s} "
                f"{100 * cascade['e2e_at_measured_tier2']:8.2f}% {'':6s}  "
                f"at the tier-2 accuracy measured for this split"
            )
        breakeven = cascade["breakeven_vs_tier2_alone"]
        if breakeven is not None:
            needed = breakeven["tier2_accuracy_needed"]
            lines.append(
                f"{'':10s} {'  break-even':14s} {'':9s} "
                f"{(100 * needed if needed is not None else float('nan')):8.2f}% {'':6s}  "
                f"{breakeven['note']}"
            )
        lines.append("")
    return "\n".join(lines)
