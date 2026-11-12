"""report — the artifact as a table, so the numbers a reader checks are the numbers stored.

Rendering reads `results/public/transfer.json` and writes nothing: `python
experiments/public/report.py` prints the artifact in `docs/public-data-2026-11-10.md`'s shape.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Sequence

RESULTS = Path(__file__).resolve().parents[2] / "results" / "public" / "transfer.json"
MODELS = ("svc", "logreg")


def format_table(payload: dict) -> str:
    lines = []
    for model in MODELS:
        block = payload.get(f"ceiling_{model}")
        if not block:
            continue
        lines.append(f"ceiling / {model} — grouped, 5 folds, 382 rows")
        lines.append(f"{'arm':16s} {'acc':>7s} {'none':>6s} {'delta':>8s} {'fixed':>6s} "
                     f"{'broke':>6s} {'p':>7s} {'folds':>10s}")
        base = block["arms"]["refs"]["summary"]["accuracy"]
        for name, arm in block["arms"].items():
            s = arm["summary"]
            cmp_ = block["vs_refs"].get(name)
            delta = f"{s['accuracy'] - base:+.4f}" if cmp_ is not None else "—"
            sig = (f"{cmp_['fixed']}" if cmp_ else "—")
            broke = (f"{cmp_['broke']}" if cmp_ else "—")
            p = (f"{cmp_['mcnemar_p']:.4f}" if cmp_ else "—")
            folds = (f"{cmp_['folds_better']}/{cmp_['folds_worse']}/{cmp_['folds_tied']}"
                     if cmp_ else "—")
            lines.append(f"{name:16s} {s['accuracy']:7.4f} {str(s['none_recall']):>6s} "
                         f"{delta:>8s} {sig:>6s} {broke:>6s} {p:>7s} {folds:>10s}")
        lines.append("")
    if "public_only" in payload:
        lines.append("public_only — train on public rows, score the 382 rows")
        for name, arm in payload["public_only"]["arms"].items():
            s = arm["summary"]
            if s["accuracy"] is None:
                lines.append(f"{name:16s} {'refused':>7s} train={arm['n_train']:6d} "
                             f"({arm.get('refused') or 'not trainable'})")
                continue
            cmp_ = payload["public_only"]["vs_refs"].get(name)
            delta = f"vs refs {cmp_['delta_accuracy']:+.4f}" if cmp_ else "—"
            lines.append(f"{name:16s} {s['accuracy']:7.4f} train={arm['n_train']:6d} {delta}")
        lines.append("")
        lines.append("reverse — train on the 382 rows, score public test rows")
        for name, arm in payload["reverse"].items():
            if name == "union":
                continue
            lines.append(f"{name:16s} {arm['summary']['accuracy']:7.4f} n={arm['summary']['n']}")
        s = payload["reverse"]["union"]
        lines.append(f"{'union':16s} {s['accuracy']:7.4f} n={s['n']}")
    return "\n".join(lines)


def main(argv: Sequence[str]) -> int:
    path = Path(argv[0]) if argv else RESULTS
    print(format_table(json.loads(path.read_text(encoding="utf-8"))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
