"""Tier 2's measured accuracy, per split — the number the end-to-end figure is a function of.

Tier 2 is the ladder's winner: word TF-IDF + logistic regression and SetFit, averaged with
row standardisation (95.03% grouped, 7.85 ms p50). Using that number inside a *different*
split's cascade would be the same mistake this repo spent a week unpicking, so the ensemble is
re-measured under each protocol it will be quoted beside:

``rowwise``
    fit on all 200 references, score all 382 rows — the protocol the 93.19% was measured under.
``family``
    5 folds over the 110 phrasing families; a fold fits on the reference pool minus slot fills
    of the held-out families, and scores that fold's rows.
``strict``
    the same folds, but the pool is rebuilt from the *other* families' template fills with no
    hand-written references at all — the pool the README's 74.9% came from.

The two grouped splits use five folds rather than leave-one-family-out because SetFit costs a
fine-tune per fold: 110 folds is 110 fine-tunes, which is hours, and the five-fold version is
the protocol every ladder number was measured under. That is a deliberate, stated substitution,
not a quiet one.

    .venv-embed/bin/python experiments/final/tier2_ensemble.py

Writes ``results/final/tier2-ensemble-by-split.json``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from localalexa.data import load_registry, load_routing_set
from localalexa.final_table import _families, _pool_and_rows
from localalexa.ladder.ensemble_rung import rung_5_ensemble_setfit_standardised
from localalexa.ladder.folds import DEFAULT_FOLDS
from localalexa.ladder.predictors import predict_rows

RESULTS = Path(__file__).resolve().parents[2] / "results" / "final"


def score_pool(rung, pool, rows, indices) -> dict:
    """Fit on ``pool``, score the rows at ``indices``."""
    started = time.perf_counter()
    predictor = rung.build().fit([text for text, _ in pool], [label for _, label in pool])
    fit_seconds = time.perf_counter() - started
    texts = [rows[i].text for i in indices]
    truth = [rows[i].tool for i in indices]
    predicted = predict_rows(predictor, texts)
    latencies = []
    for text in texts:
        began = time.perf_counter()
        predictor.scores([text])
        latencies.append((time.perf_counter() - began) * 1000.0)
    correct = sum(1 for p, t in zip(predicted, truth) if p == t)
    return {
        "n": len(indices),
        "correct": correct,
        "fit_seconds": round(fit_seconds, 2),
        "latency_p50_ms": round(statistics.median(latencies), 2) if latencies else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--splits", default="rowwise,family,strict",
                        help="comma list of splits to measure (the others are kept as they are)")
    args = parser.parse_args(argv)
    wanted = [s.strip() for s in args.splits.split(",") if s.strip()]

    registry = load_registry()
    rows = load_routing_set()
    families = _families(registry, rows)
    rung = rung_5_ensemble_setfit_standardised()

    target = RESULTS / "tier2-ensemble-by-split.json"
    payload: dict[str, object] = (
        json.loads(target.read_text(encoding="utf-8"))
        if target.is_file()
        else {"tier2": rung.name, "note": rung.note, "splits": {}}
    )
    payload["tier2"] = rung.name
    payload["note"] = rung.note
    print(f"rows {len(rows)}  folds {DEFAULT_FOLDS}  tier 2 = {rung.name}")

    for split in wanted:
        total_correct = total_n = 0
        fits, latencies = [], []
        plan = _pool_and_rows(split, registry, rows, families, DEFAULT_FOLDS)
        for step, (pool, indices) in enumerate(plan):
            result = score_pool(rung, pool, rows, indices)
            total_correct += result["correct"]
            total_n += result["n"]
            fits.append(result["fit_seconds"])
            latencies.append(result["latency_p50_ms"])
            print(f"  {split} step {step}: pool {len(pool):4d} rows · "
                  f"{result['correct']}/{result['n']} · fit {result['fit_seconds']}s", flush=True)
        payload["splits"][split] = {
            "n": total_n,
            "correct": total_correct,
            "accuracy": round(total_correct / total_n, 4),
            "fit_seconds_total": round(sum(fits), 1),
            "latency_p50_ms": round(statistics.median(l for l in latencies if l), 2),
            "steps": len(plan),
            "pool": SPOOLS[split],
        }
        print(f"  {split}: {total_correct}/{total_n} = "
              f"{payload['splits'][split]['accuracy']:.4f} (fits {sum(fits):.0f}s total)")
        # save as each split completes, so a kill keeps what is already measured
        RESULTS.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    RESULTS.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {target}")
    return 0


SPOOLS = {
    "rowwise": "the 200 references, scoring all 382 rows",
    "family": "reference pool minus the held-out families' slot fills, 5 folds",
    "strict": "template fills of the other families only, filtered by rendered text, 5 folds",
}


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
