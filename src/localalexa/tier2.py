"""Tier 2, measured instead of assumed: join the cascade's escalations to a real model's answers.

``compare`` reports the end-to-end number as ``e2e(x) = (auto + x·escalated)/n`` because no
tier-2 model had been scored on this corpus. One has now, so this module replaces the parameter
with a join:

* the cascade side comes from ``compare.collect()`` — for each of the 382 rows, which tier
  would answer it and whether that answer is right, at each gate threshold;
* the tier-2 side comes from a measurement file (``experiments/optional/exp_cactus_hybrid.py``
  writes one): per row, the model's label, its own confidence, and whether it was right;
* the rows are joined by utterance text, and a row the tier-2 model never scored is **reported
  as missing**, never silently treated as wrong or right.

Two numbers come out, and they are the two the brief asked for:

``coverage``
    share of all traffic the cascade answers without handing on — tiers 0 and 1 plus the
    escalated rows the tier-2 model is confident enough to answer;
``precision``
    accuracy on exactly those rows, which is what the operator experiences.

The tier-2 confidence is used as a *second* gate — a third tier above this one — so the curve
over both thresholds shows what the arrangement buys and what it hands on. That is the same
instrument as the ladder's reject curve, applied one level down.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .compare import DEFAULT_GATES, RowScore, collect

TIER2_THRESHOLDS: tuple[float, ...] = (0.0, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)
PRECISION_TARGETS: tuple[float, ...] = (0.90, 0.95)


@dataclass(frozen=True)
class Tier2Row:
    """One row as the tier-2 model saw it."""

    text: str
    predicted: str
    confidence: float | None
    correct: bool


def load_measurement(path: str | Path) -> dict[str, Tier2Row]:
    """Tier-2 rows keyed by utterance text."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows: dict[str, Tier2Row] = {}
    for row in payload.get("per_row", ()):
        text = str(row["text"])
        rows[text] = Tier2Row(
            text=text,
            predicted=str(row.get("pred", "")),
            confidence=None if row.get("confidence") is None else float(row["confidence"]),
            correct=bool(row.get("correct")),
        )
    if not rows:
        raise ValueError(f"{path}: no per_row entries — is this a measurement file?")
    return rows


def _answers(
    score: RowScore, cascade_gate: float, tier2_gate: float, tier2: Mapping[str, Tier2Row]
) -> tuple[bool | None, str]:
    """(correct?, who answered) for one row. ``None`` means nobody below the next tier did."""
    if score.regex_label is not None:
        return score.regex_label == score.truth, "tier0"
    if score.model_confidence >= cascade_gate:
        return score.model_label == score.truth, "tier1"
    seen = tier2.get(score.text)
    if seen is None:
        return None, "missing"
    if seen.confidence is None or seen.confidence < tier2_gate:
        return None, "tier3"
    return seen.correct, "tier2"


def _point(
    scores: Sequence[RowScore],
    tier2: Mapping[str, Tier2Row],
    cascade_gate: float,
    tier2_gate: float,
) -> dict[str, object]:
    n = len(scores)
    answered = 0
    correct = 0
    tier3 = 0
    missing = 0
    by_tier = {"tier0": 0, "tier1": 0, "tier2": 0}
    for score in scores:
        right, who = _answers(score, cascade_gate, tier2_gate, tier2)
        if who in ("tier3", "missing"):
            tier3 += who == "tier3"
            missing += who == "missing"
            continue
        answered += 1
        right = bool(right)
        correct += 1 if right else 0
        by_tier[who] = by_tier.get(who, 0) + 1
    return {
        "cascade_gate": cascade_gate,
        "tier2_gate": tier2_gate,
        "answered": answered,
        "coverage": round(answered / n, 4),
        "correct": correct,
        "precision": round(correct / answered, 4) if answered else None,
        "handed_to_tier3": tier3,
        "missing_tier2_rows": missing,
        "answered_by_tier": by_tier,
        "accuracy_over_all_rows_if_tier3_perfect": round((correct + n - answered) / n, 4),
        "accuracy_over_all_rows_if_tier3_useless": round(correct / n, 4),
    }


def _best_for_precision(
    points: Sequence[Mapping[str, object]], target: float
) -> Mapping[str, object] | None:
    eligible = [p for p in points if (p["precision"] or 0.0) >= target and p["answered"]]
    if not eligible:
        return None
    return max(eligible, key=lambda p: p["coverage"])


def measure(
    *,
    tier2_path: str | Path,
    folds: int = 5,
    tier1_kind: str = "logreg",
    cascade_gates: Sequence[float] = DEFAULT_GATES,
    tier2_gates: Sequence[float] = TIER2_THRESHOLDS,
    precision_targets: Sequence[float] = PRECISION_TARGETS,
    production_gate: float = 0.35,
) -> dict[str, object]:
    """Join the cascade to a measured tier 2 and report coverage against precision."""
    tier2 = load_measurement(tier2_path)
    scores, _registry, best_single = collect(folds=folds, tier1_kind=tier1_kind)
    n = len(scores)
    present = sum(1 for score in scores if score.text in tier2)

    points = [
        _point(scores, tier2, cascade_gate, tier2_gate)
        for cascade_gate in cascade_gates
        for tier2_gate in tier2_gates
    ]
    best = {
        f"{target:.2f}": _best_for_precision(points, target) for target in precision_targets
    }

    # what the tier-2 model is actually asked about at the gate the product runs
    escalated_rows = [
        score
        for score in scores
        if score.regex_label is None and score.model_confidence < production_gate
    ]
    escalated_measured = [tier2[score.text] for score in escalated_rows if score.text in tier2]
    return {
        "meta": {
            "tier2_measurement": str(tier2_path),
            "folds": folds,
            "tier1": tier1_kind,
            "n_rows": n,
            "rows_joined": present,
            "production_gate": production_gate,
            "cascade_gates": list(cascade_gates),
            "tier2_gates": list(tier2_gates),
            "protocol": "coverage and precision measured by joining the cascade's per-row "
            "decisions to the tier-2 model's own answers on the same utterances",
        },
        "bars": {
            "best_single_classifier_correct": best_single,
            "best_single_classifier_accuracy": round(best_single / n, 4),
        },
        "tier2_model": {
            "rows_measured": len(tier2),
            "accuracy": round(sum(1 for row in tier2.values() if row.correct) / len(tier2), 4),
            "rows_it_would_see": len(escalated_rows),
            "rows_of_those_measured": len(escalated_measured),
            "accuracy_on_those": round(
                sum(1 for row in escalated_measured if row.correct) / len(escalated_measured), 4
            )
            if escalated_measured
            else None,
        },
        "points": points,
        "best_at_precision": best,
    }


def render(payload: Mapping[str, object]) -> str:
    meta = payload["meta"]
    model = payload["tier2_model"]
    bars = payload["bars"]
    lines = [
        f"tier 2 = {Path(meta['tier2_measurement']).name} · {model['rows_measured']} rows "
        f"measured, {meta['rows_joined']}/{meta['n_rows']} of the cascade's rows joined",
        f"tier-2 accuracy {100 * model['accuracy']:.2f}% overall · "
        f"{100 * (model['accuracy_on_those'] or 0):.2f}% on the {model['rows_it_would_see']} rows "
        f"the cascade escalates",
        f"bar: best single classifier {bars['best_single_classifier_correct']}/{meta['n_rows']} "
        f"= {100 * bars['best_single_classifier_accuracy']:.2f}%",
        "",
        f"{'cascade gate':>12s} {'tier2 gate':>10s} {'coverage':>9s} {'precision':>9s} "
        f"{'to tier 3':>9s}",
    ]
    for point in payload["points"]:
        if point["tier2_gate"] not in (0.0, 0.5, 0.6, 0.7):
            continue
        lines.append(
            f"{point['cascade_gate']:12.2f} {point['tier2_gate']:10.2f} "
            f"{100 * point['coverage']:8.1f}% "
            f"{100 * (point['precision'] or 0):8.2f}% {point['handed_to_tier3']:9d}"
        )
    for target, best in payload["best_at_precision"].items():
        if best is None:
            lines.append(f"no point reaches {100 * float(target):.1f}% precision")
        else:
            lines.append(
                f"best coverage at {100 * float(target):.1f}% precision: "
                f"{100 * best['coverage']:.1f}% (cascade gate {best['cascade_gate']:.2f}, "
                f"tier-2 gate {best['tier2_gate']:.2f})"
            )
    return "\n".join(lines)
