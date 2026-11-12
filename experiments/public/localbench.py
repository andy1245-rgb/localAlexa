"""bench — one fold assignment, one scoring seam, for every measurement in this experiment.

The 89.79% baseline is a *protocol*, not just a model: five folds over the 110 phrasing
families, and a fold trains on the registry's 200 references minus any reference that renders
into that fold's families. Reproducing that number means reusing `localalexa.ladder.folds`
and `localalexa.ladder.pools.fold_pool` rather than re-deriving a split, because a fold map
that differs by one family moves this corpus by a point.

Scoring goes through `bench.harness.metrics`, so a number here and a number from
`bench.harness.run` come from the same functions. The only thing this module owns is the
loop: which rows a model may see, and what gets recorded.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

LOCALALEXA = Path(__file__).resolve().parents[2]
# The harness is a sibling checkout; the env var is there because the two repositories do not have
# to sit next to each other on another box.
TOOLINTERFACER = Path(os.environ.get("TOOLINTERFACER_ROOT", LOCALALEXA.parent / "toolInterfacer"))
if not (TOOLINTERFACER / "bench" / "harness").is_dir():
    raise SystemExit(
        f"harness not found at {TOOLINTERFACER}; set TOOLINTERFACER_ROOT to the checkout that "
        f"contains bench/harness/")
for root in (LOCALALEXA / "src", TOOLINTERFACER):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from bench.harness import metrics  # noqa: E402
from bench.harness.rows import Row as HRow  # noqa: E402
from localalexa.data import NONE_LABEL, load_registry, load_routing_set, phrasing_families  # noqa: E402
from localalexa.data import template_index  # noqa: E402
from localalexa.ladder.pools import fold_pool, reference_pool  # noqa: E402
from localalexa.ladder.run import load_bench  # noqa: E402

FOLDS = 5


@dataclass(frozen=True)
class Bench:
    """The 382 rows as harness rows, plus the ladder's fold per row."""

    rows: tuple[HRow, ...]
    fold_of_row: tuple[int, ...]
    families: tuple[str, ...]
    options: tuple[str, ...]
    registry: object

    def held_out(self, fold: int) -> frozenset[str]:
        return frozenset(f for f, k in zip(self.families, self.fold_of_row) if k == fold)

    def in_fold(self, fold: int) -> list[HRow]:
        return [r for r, k in zip(self.rows, self.fold_of_row) if k == fold]


def load(folds: int = FOLDS) -> Bench:
    """The 382 rows as harness rows on the ladder's own five-family folds."""
    bench = load_bench(folds)
    registry = bench.registry
    options = tuple(registry.label_ids)
    rows = tuple(
        HRow(id=f"a382-{i:03d}", text=row.text, label=row.tool, group=row.family,
             options=options, meta={"family": row.family})
        for i, row in enumerate(bench.rows)
    )
    fold_of_row = tuple(bench.folds.fold_of(f) for f in bench.families)
    return Bench(rows=rows, fold_of_row=fold_of_row, families=tuple(bench.families),
                 options=options, registry=registry)


def pool_rows(pairs: Iterable[tuple[str, str]], options: Sequence[str]) -> list[HRow]:
    """Reference/extra (text, label) pairs as harness rows for `fit`.

    The id and group are only bookkeeping for the fit side; nothing in the fit reads them.
    """
    return [HRow(id=f"pool-{i:04d}", text=text, label=label, group="pool",
                 options=tuple(options))
            for i, (text, label) in enumerate(pairs)]


def fit_and_score(arm: str, builder, train: Sequence[HRow], test: Sequence[HRow]):
    """Fit a fresh predictor on `train`, score `test` once, return the records."""
    predictor = builder()
    predictor.fit(list(train))
    return metrics.score(arm, list(test), predictor)


def grouped_refs(builder, bench: Bench, extra: Sequence[tuple[str, str]] = ()):
    """Protocol L: per fold, train on references minus that fold's families (+ extra rows).

    `extra` is the public corpus mapped to localAlexa's labels. It is appended to every fold's
    pool: public rows are not renders of localAlexa templates, so they cannot contaminate a
    held-out family the way a registry reference can.
    """
    recs = []
    per_fold = []
    for fold in range(FOLDS):
        held = bench.held_out(fold)
        pairs = fold_pool(bench.registry, held) + list(extra)
        train = pool_rows(pairs, bench.options)
        test = bench.in_fold(fold)
        fold_recs = fit_and_score(f"fold{fold}", builder, train, test)
        acc = metrics.accuracy(fold_recs)
        recs.extend(fold_recs)
        per_fold.append({"fold": fold, "n_train": len(train), "n_test": acc["n"],
                         "accuracy": round(acc["accuracy"], 4), "correct": acc["correct"]})
    return recs, per_fold


def fit_all_score(builder, train: Sequence[HRow], test: Sequence[HRow]):
    """One fit, one scoring pass — for the arms whose training side is not localAlexa's rows."""
    return fit_and_score("all", builder, train, test), None


def summarise(recs: Sequence) -> dict:
    acc = metrics.accuracy(recs)
    none_rows = [r for r in recs if r.label == NONE_LABEL]
    none_correct = sum(1 for r in none_rows if r.correct)
    return {
        "n": acc["n"],
        "correct": acc["correct"],
        "accuracy": round(acc["accuracy"], 4),
        "wilson95": [round(x, 4) for x in acc["wilson95"]],
        "none_recall": round(none_correct / len(none_rows), 4) if none_rows else None,
        "n_none": len(none_rows),
    }
