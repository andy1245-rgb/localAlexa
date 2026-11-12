"""End-to-end: a handful of utterances through the tap, then derive and summarise.

Not a unit test — a runnable demonstration that the five pieces fit together, that
the derived files say what the doc claims, and that a row with no signal produces
no label. Writes to a temporary root, so it never touches the real log.

    .venv/bin/python experiments/collect/smoke_session.py
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from localalexa.collection import (
    AutoAcceptRule,
    CollectionTap,
    CorrectionRule,
    EffectRecord,
    RequestLog,
    RequestRecord,
    SecondOpinion,
    derive,
    render,
    summarise,
    write_jsonl,
)
from localalexa.data import load_registry
from localalexa.router import decide

# text, (top label, top sim), (runner-up, sim), seconds since the previous row
UTTERANCES = [
    ("set a timer for five minutes", ("timer_create", 0.70), ("timer_cancel", 0.61), 0.0),
    ("no i said ten minutes", ("timer_create", 0.71), ("timer_cancel", 0.60), 6.0),
    ("add milk to the shopping list", ("shoppinglist_add", 0.46), ("shoppinglist_view", 0.44), 90.0),
    ("is the porch light on", ("light_query", 0.45), ("light_set", 0.43), 30.0),
    ("turn on the kitchen lights", ("light_set", 0.66), ("light_query", 0.40), 20.0),
    ("text 555-867-5309 the grocery list", ("shoppinglist_view", 0.47), ("shoppinglist_add", 0.45), 15.0),
]


def similarities(labels, top: str, top_sim: float, second: str, second_sim: float) -> list[float]:
    vector = [0.30] * len(labels)
    vector[labels.index(top)] = top_sim
    vector[labels.index(second)] = second_sim
    return vector


def main() -> int:
    registry = load_registry()
    labels = registry.label_ids
    clock = {"now": time.time()}

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "requests.jsonl"
        log = RequestLog(path, session="demo")
        tap = CollectionTap(log, registry=registry, clock=lambda: clock["now"])

        for text, top, second, gap in UTTERANCES:
            clock["now"] += gap
            tap.observe(text, decide(similarities(labels, *top, *second), labels))

        rows = [row for row in log.rows() if row["kind"] == "request"]
        # The world answers two of them: the timer exists, the shopping list did not
        # change. Both checks are the operator's tooling reporting, not attention.
        log.append_effect(EffectRecord(rows[0]["id"], time.time(), True, "timer_registry", "timer 4c1f"))
        log.append_effect(EffectRecord(rows[2]["id"], time.time(), False, "list_diff", "no change"))
        log.append_effect(EffectRecord(rows[4]["id"], time.time(), False, "light_state", "switch off"))
        log.close()

        reader = RequestLog(path)
        records = [RequestRecord.from_dict(row) for row in reader.rows() if row["kind"] == "request"]
        effects = [EffectRecord.from_dict(row) for row in reader.effects()]
        reader.close()

        derived = derive(
            records,
            effects,
            correction_rule=CorrectionRule(),
            auto_rule=AutoAcceptRule(),
            opinion=SecondOpinion(registry),
        )
        write_jsonl(Path(tmp) / "labels.jsonl", [row.as_dict() for row in derived.labels])

        print("== rows")
        for row in rows:
            candidates = "/".join(c["label"] for c in row["candidates"][:2]) or "—"
            print(
                f"  tier{row['tier']} gate={str(row['gate']):5s} label={str(row['label']):18s} "
                f"margin={row['margin']:.3f} top2={candidates:38s} {row['text']!r}"
            )

        print("\n== corrections")
        for correction in derived.corrections:
            print(
                f"  {correction.earlier_id} corrected by {correction.later_id} "
                f"dt={correction.delta_seconds:.0f}s branch={correction.evidence['branch']}"
            )

        print("\n== labels")
        names = {row["id"]: row["text"] for row in rows}
        for label in derived.labels:
            suppressed = label.detail.get("suppressed", [])
            extra = f" (suppressed {suppressed})" if suppressed else ""
            print(
                f"  {label.polarity:+d} {label.strength:6s} {label.source:22s} "
                f"{label.label:18s} {names[label.request_id]!r}{extra}"
            )

        print("\n== silent rows")
        labelled = {label.request_id for label in derived.labels}
        for row in rows:
            if row["id"] not in labelled:
                print(f"  no signal: {row['text']!r}")

        print("\n== stats", dict(derived.stats))
        print("\n== weekly summary")
        print(render(summarise(records, [row.as_dict() for row in derived.labels], days=7.0)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
