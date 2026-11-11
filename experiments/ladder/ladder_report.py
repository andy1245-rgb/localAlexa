"""The three runs the results doc quotes, so every number has a command.

    .venv-embed/bin/python experiments/ladder/ladder_report.py rungs
    .venv-embed/bin/python experiments/ladder/ladder_report.py linear
    .venv-embed/bin/python experiments/ladder/ladder_report.py forward
    .venv-embed/bin/python experiments/ladder/ladder_report.py rung5
    .venv-embed/bin/python experiments/ladder/ladder_report.py confirm
    .venv-embed/bin/python experiments/ladder/ladder_report.py latency

``linear`` is the fast pass (no torch): the baseline and every linear rung at
three fold counts plus row-wise. ``forward`` adds the rungs that need the heavy
venv — SetFit, the cross-encoder, the mixed ensemble. ``confirm`` re-measures the
finalists on a different fold count, because two of them were chosen by a sweep
on the folds they are scored on.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from localalexa.ladder.catalog import available_rungs
from localalexa.ladder.report import format_table, run_ladder, write_artifact

RESULTS = Path(__file__).resolve().parents[2] / "results" / "ladder"

STAGES = {
    "linear": (
        ["r0-word-svc", "r1-union-svc", "r1-union-svc-C4", "r2-word-logreg-C100", "r2-word-svc-sigmoid"],
        (4, 5, 10),
        ("grouped", "rowwise"),
        "linear-rungs.json",
    ),
    "forward": (
        [
            "r0-word-svc",
            "r2-word-logreg-C100",
            "r3-setfit-minilm",
            "r4-crossencoder-top3",
            "r5-ensemble-setfit-z",
            "r5-ensemble-linear",
            "r6-template-cap16",
        ],
        (5,),
        ("grouped", "rowwise"),
        "forward-rungs.json",
    ),
    "rung5": (
        ["r0-word-svc", "r2-word-logreg-C100", "r3-setfit-minilm", "r5-ensemble-setfit-z"],
        (5,),
        ("grouped", "rowwise"),
        "rung5-ensemble.json",
    ),
    "confirm": (
        ["r0-word-svc", "r2-word-logreg-C100", "r3-setfit-minilm"],
        (10,),
        ("grouped",),
        "folds10-confirmation.json",
    ),
}


def main(argv: list[str]) -> int:
    if not argv or argv[0] == "rungs":
        print("stages:", ", ".join(STAGES))
        for rung in available_rungs("all"):
            print(f"  {rung.name:26s} {rung.note}")
        return 0
    stage = argv[0]
    if stage not in STAGES:
        raise SystemExit(f"unknown stage {stage!r}; known: {sorted(STAGES)}")
    names, fold_counts, protocols, filename = STAGES[stage]
    payload = run_ladder(
        available_rungs(",".join(names)),
        fold_counts=fold_counts,
        protocols=protocols,
    )
    print(format_table(payload))
    print(f"\nwrote {write_artifact(payload, RESULTS / filename)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
