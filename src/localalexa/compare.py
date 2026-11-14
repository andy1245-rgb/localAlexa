"""The cascade against one classifier, measured grouped and without the leak.

The comparison this module exists to replace used the row-wise number for the
single classifier (93.19%) and a tier-1 model trained on every reference,
including the slot fills of the families being scored. Both flatter the cheap
tiers: measured properly, tier 1 falls from 93.46% to 86.41% and the cascade's
verdict is not the one that was recorded. So the comparison is a measurement now,
not an argument:

* **one classifier** — the ladder's best linear model (word TF-IDF + logistic
  regression C=100), refitted per fold on the reference pool minus the slot fills
  of the held-out families, scoring every row;
* **the cascade** — the same model behind the regex tier and the confidence gate,
  so the only thing that changes between the arms is the structure, not the model;
* **the oracle gate** — the same cheap tiers with escalation decided perfectly,
  which is the ceiling any policy could reach.

Two things are deliberately not assumed. The tier-2 accuracy is a parameter, not a
constant, so the end-to-end number comes out as ``e2e(x) = (auto + x·escalated)/n``
with the break-even point solved for ``x``. And the gate threshold is swept rather
than fixed, because at the production threshold the gate is the binding
constraint, not the expensive tier — a fact that only shows up on the sweep.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from .cascade import (
    TIER2_GRID,
    CompiledPatterns,
    e2e_accuracy,
    e2e_curve,
    tier2_breakeven,
)
from .data import Registry, load_registry
from .ladder.folds import DEFAULT_FOLDS
from .ladder.pools import fold_pool, reference_pool
from .ladder.rungs import rung_2_logreg_c100
from .ladder.run import load_bench

Tier1 = Callable[[str], tuple[str, float, float]]
DEFAULT_GATES = (0.35, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99)


@dataclass(frozen=True)
class RowScore:
    """One row, scored once: everything every threshold needs to know about it."""

    fold: int
    truth: str
    regex_label: str | None
    model_label: str
    model_confidence: float
    model_similarity: float

    @property
    def one_classifier_correct(self) -> bool:
        return self.model_label == self.truth

    @property
    def regex_covers(self) -> bool:
        return self.regex_label is not None


def _linear_tier1(pool: Sequence[tuple[str, str]]) -> Tier1:
    """Word TF-IDF + logistic regression, the ladder's rung 2, as ``text -> (label, p, p)``."""
    predictor = rung_2_logreg_c100().build()
    predictor.fit([text for text, _ in pool], [label for _, label in pool])
    classes = list(predictor.classes)

    def decide(text: str) -> tuple[str, float, float]:
        scores = predictor.scores([text])[0]
        best = max(range(len(classes)), key=lambda index: scores[index])
        # A probability has no cosine similarity. The top probability stands in for both
        # gate inputs and the similarity floor is left off, rather than applied to a number
        # that is not a similarity.
        return classes[best], float(scores[best]), float(scores[best])

    return decide


def _router_tier1(registry: Registry) -> Tier1:
    """The production tier 1: MiniLM reference similarity. Needs the embeddings extra."""
    from .router import SemanticRouter

    router = SemanticRouter(registry)
    router.predict("warm up")

    def decide(text: str) -> tuple[str, float, float]:
        choice = router.predict(text)
        return choice.top_tool, float(choice.confidence), float(choice.top_sim)

    return decide


def _tier1_for(kind: str, registry: Registry, pool: Sequence[tuple[str, str]]) -> Tier1:
    if kind == "logreg":
        return _linear_tier1(pool)
    if kind == "router":
        del pool  # the encoder router is built from the registry's references, not a pool
        return _router_tier1(registry)
    raise ValueError(f"unknown tier1 {kind!r}; known: logreg, router")


def collect(
    *,
    folds: int = DEFAULT_FOLDS,
    tier1_kind: str = "logreg",
    registry: Registry | None = None,
) -> tuple[list[RowScore], object, int]:
    """Score every row once, per fold, so any threshold can be read off afterwards.

    Also returns the correct count of the *best single classifier* on the same folds —
    word TF-IDF + logistic regression C=100, the ladder's rung 2. That is the bar the
    cascade has to beat, and it is measured here rather than quoted, so the artifact
    answers "does the cascade win" without borrowing a number from another run.
    """
    registry = registry or load_registry()
    bench = load_bench(folds)
    rows = list(bench.rows)
    compiled = CompiledPatterns.from_registry(registry)
    scores: list[RowScore] = []
    best_single_correct = 0

    for fold, indices in enumerate(bench.folds.indices(list(bench.families))):
        held = {bench.families[i] for i in indices}
        pool = fold_pool(registry, held)
        tier1 = _tier1_for(tier1_kind, registry, pool)
        best_single = _linear_tier1(pool) if tier1_kind != "logreg" else tier1
        for i in indices:
            row = rows[i]
            hits = compiled.probe(row.text)
            label, confidence, similarity = tier1(row.text)
            if best_single(row.text)[0] == row.tool:
                best_single_correct += 1
            scores.append(
                RowScore(
                    fold=fold,
                    truth=row.tool,
                    regex_label=hits[0] if len(hits) == 1 else None,
                    model_label=label,
                    model_confidence=confidence,
                    model_similarity=similarity,
                )
            )
    return scores, registry, best_single_correct


def _operating_point(
    scores: Sequence[RowScore],
    gate: float,
    grid: Sequence[float],
    one_correct: int,
    use_sim_floor: bool = False,
    sim_floor: float = 0.45,
) -> dict[str, object]:
    """Aggregate the cascade at one gate threshold."""
    n = len(scores)
    auto_correct = 0
    escalated = 0
    tier0_rows = tier0_correct = 0
    tier1_rows = tier1_correct = 0
    auto_wrong = 0

    for score in scores:
        if score.regex_covers:
            predicted, tier = score.regex_label, 0
        elif score.model_confidence < gate or (
            use_sim_floor and score.model_similarity < sim_floor
        ):
            escalated += 1
            continue
        else:
            predicted, tier = score.model_label, 1
        right = predicted == score.truth
        auto_correct += 1 if right else 0
        auto_wrong += 0 if right else 1
        if tier == 0:
            tier0_rows += 1
            tier0_correct += 1 if right else 0
        else:
            tier1_rows += 1
            tier1_correct += 1 if right else 0

    return {
        "gate": gate,
        "coverage": round((n - escalated) / n, 4),
        "escalated": escalated,
        "auto_correct": auto_correct,
        "auto_accuracy": round(auto_correct / (n - escalated), 4) if n > escalated else None,
        "tier0": {
            "rows": tier0_rows,
            "correct": tier0_correct,
            "accuracy": round(tier0_correct / tier0_rows, 4) if tier0_rows else None,
        },
        "tier1": {
            "rows": tier1_rows,
            "correct": tier1_correct,
            "accuracy": round(tier1_correct / tier1_rows, 4) if tier1_rows else None,
        },
        "e2e_curve_pct": e2e_curve(auto_correct, escalated, n, grid=grid),
        "breakeven_vs_one_classifier": tier2_breakeven(auto_correct, escalated, n, one_correct),
        "oracle_gate_e2e_pct": {
            f"{value:.2f}": round(e2e_accuracy(n - auto_wrong, auto_wrong, n, value), 1)
            for value in grid
        },
    }


def _leaky_reference(registry: Registry, rows) -> dict[str, object]:
    """Tier 1 trained on every reference — the number the old verdict was drawn from."""
    tier1 = _linear_tier1(reference_pool(registry))
    correct = sum(1 for row in rows if tier1(row.text)[0] == row.tool)
    return {
        "correct": correct,
        "n": len(rows),
        "accuracy": round(correct / len(rows), 4),
        "note": "trained on all 200 references, including slot fills of the scored families",
    }


def compare(
    *,
    folds: int = DEFAULT_FOLDS,
    tier1_kind: str = "logreg",
    gates: Sequence[float] = DEFAULT_GATES,
    production_gate: float = 0.35,
    tier2_grid: Sequence[float] = TIER2_GRID,
    registry: Registry | None = None,
) -> dict[str, object]:
    """Both arms on the same grouped folds, with the gate swept and tier 2 parameterised."""
    scores, registry, best_single_correct = collect(
        folds=folds, tier1_kind=tier1_kind, registry=registry
    )
    use_sim_floor = tier1_kind == "router"
    n = len(scores)
    one_correct = sum(1 for score in scores if score.one_classifier_correct)
    bar = best_single_correct  # the best single classifier on these folds
    rows = load_bench(folds).rows

    points = [
        _operating_point(scores, gate, tier2_grid, bar, use_sim_floor) for gate in gates
    ]
    production = _operating_point(scores, production_gate, tier2_grid, bar, use_sim_floor)

    # For each tier-2 accuracy, the best end-to-end number any gate reaches, and the
    # escalation rate that got there. This is the question the old verdict answered
    # from a single operating point.
    best_by_tier2: dict[str, dict[str, object]] = {}
    for value in tier2_grid:
        key = f"{value:.2f}"
        winner = max(points, key=lambda point: point["e2e_curve_pct"][key])
        best_by_tier2[key] = {
            "e2e_pct": winner["e2e_curve_pct"][key],
            "gate": winner["gate"],
            "coverage": winner["coverage"],
            "beats_one_classifier": 100.0 * bar / n <= winner["e2e_curve_pct"][key],
        }

    return {
        "meta": {
            "folds": folds,
            "tier1": tier1_kind,
            "n_rows": n,
            "production_gate": production_gate,
            "gates_swept": list(gates),
            "tier2_grid": list(tier2_grid),
            "protocol": "grouped CV over phrasing families; tier 1 refitted per fold on the "
            "reference pool minus the held-out families' slot fills",
        },
        "one_classifier": {
            "correct": one_correct,
            "n": n,
            "accuracy": round(one_correct / n, 4),
            "model": f"the cascade's own tier 1 ({tier1_kind}) answering every row",
            "per_fold": [
                round(
                    sum(1 for s in scores if s.fold == fold and s.one_classifier_correct)
                    / max(sum(1 for s in scores if s.fold == fold), 1),
                    4,
                )
                for fold in sorted({s.fold for s in scores})
            ],
        },
        "best_single_classifier": {
            "correct": bar,
            "n": n,
            "accuracy": round(bar / n, 4),
            "model": "word TF-IDF + logistic regression C=100, refitted per fold",
            "note": "the bar the cascade has to beat; measured here, not quoted",
        },
        "production_point": production,
        "gate_sweep": points,
        "best_gate_per_tier2_accuracy": best_by_tier2,
        "leaky_reference": _leaky_reference(registry, rows),
        "tier0_alone": {
            "rows": sum(1 for s in scores if s.regex_covers),
            "correct": sum(
                1 for s in scores if s.regex_covers and s.regex_label == s.truth
            ),
            "note": "the regex tier is deterministic, so this is not affected by the leak",
        },
    }


def render(payload: Mapping[str, object]) -> str:
    """The comparison as a few lines someone can argue with."""
    meta = payload["meta"]
    one = payload["one_classifier"]
    best = payload["best_single_classifier"]
    point = payload["production_point"]
    leaky = payload["leaky_reference"]
    lines = [
        f"grouped {meta['folds']}-fold · tier 1 = {meta['tier1']} · {meta['n_rows']} rows",
        f"best single classifier {best['correct']:3d}/{best['n']} = "
        f"{100 * best['accuracy']:.2f}%  (word logreg C=100)",
        f"tier 1 alone          {one['correct']:3d}/{one['n']} = {100 * one['accuracy']:.2f}%  "
        f"({meta['tier1']} answering every row)",
        f"tier 1 under the leak {leaky['correct']:3d}/{leaky['n']} = "
        f"{100 * leaky['accuracy']:.2f}%  (the number the old verdict used)",
        "",
        f"gate ≥ {point['gate']:.2f} (production): coverage {100 * point['coverage']:.1f}% · "
        f"auto-answered at {100 * (point['auto_accuracy'] or 0):.2f}%",
        f"  tier 0 regex        {point['tier0']['correct']:3d}/{point['tier0']['rows']} = "
        f"{100 * (point['tier0']['accuracy'] or 0):.2f}%",
        f"  tier 1 model        {point['tier1']['correct']:3d}/{point['tier1']['rows']} = "
        f"{100 * (point['tier1']['accuracy'] or 0):.2f}%",
        "  end-to-end by tier-2 accuracy: "
        + ", ".join(f"{k}→{v}%" for k, v in point["e2e_curve_pct"].items()),
        f"  break-even vs one classifier: {point['breakeven_vs_one_classifier']['tier2_accuracy_needed']}"
        f" ({point['breakeven_vs_one_classifier']['note']})",
        "",
        "best gate per tier-2 accuracy:",
    ]
    for key, value in payload["best_gate_per_tier2_accuracy"].items():
        verdict = "beats" if value["beats_one_classifier"] else "loses to"
        lines.append(
            f"  tier 2 at {key}: {value['e2e_pct']}% at gate {value['gate']:.2f} "
            f"(coverage {100 * value['coverage']:.0f}%) — {verdict} one classifier"
        )
    return "\n".join(lines)
