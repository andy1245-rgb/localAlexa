"""Per-intent error analysis: which intents fail, in which pairs, and for which structure.

The repo has quoted a confusion pair here and there (``light_query``/``light_set``,
``thermostat_set``/``thermostat_offset``) but never published the ledger. This does, from the
same grouped folds every other number uses, for each structural variant: per-class recall, the
confusion pairs ranked by count, and the rows that every variant gets wrong — the errors that
are about the corpus rather than about the model.

    .venv/bin/python experiments/final/error_analysis.py

Reads ``results/final/six-variants.json`` (written by ``six_variants.py``) and writes
``results/final/per-intent.md`` plus ``results/final/per-intent.json``.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from localalexa.data import NONE_LABEL, load_registry, load_routing_set

RESULTS = Path(__file__).resolve().parents[2] / "results" / "final"


def ledger(predictions, rows, labels) -> dict:
    per_label = {}
    for label in labels:
        idx = [i for i, row in enumerate(rows) if row.tool == label]
        if not idx:
            continue
        correct = sum(1 for i in idx if predictions[i] == label)
        per_label[label] = {
            "n": len(idx),
            "correct": correct,
            "recall": round(correct / len(idx), 3),
            "confused_with": dict(
                Counter(predictions[i] for i in idx if predictions[i] != label).most_common(3)
            ),
        }
    return dict(sorted(per_label.items(), key=lambda kv: (kv[1]["recall"], kv[0])))


def pairs(predictions, rows, limit: int = 12) -> list[dict]:
    counts = Counter(
        (row.tool, predictions[i])
        for i, row in enumerate(rows)
        if predictions[i] != row.tool
    )
    return [{"true": true, "pred": predicted, "n": n}
            for (true, predicted), n in counts.most_common(limit)]


def render(payload: dict) -> str:
    lines = ["# Per-intent error analysis", "",
             f"Grouped 5-fold over the 110 phrasing families, {payload['n']} rows. "
             "Recall is per class; `confused with` lists where that class goes when it is wrong.",
             ""]
    base = payload["arms"]["baseline_logreg"]
    lines += ["## The worst intents (word logreg C=100, the baseline)", "",
              "| intent | n | recall | confused with |", "|---|---:|---:|---|"]
    for label, block in base["ledger"].items():
        if block["recall"] >= 0.9:
            continue
        confused = ", ".join(f"{k} ×{v}" for k, v in block["confused_with"].items()) or "—"
        lines.append(f"| `{label}` | {block['n']} | {block['recall']:.3f} | {confused} |")

    lines += ["", "## Per-variant recall on the intents that are hard", ""]
    variants = [name for name in payload["arms"] if "ledger" in payload["arms"][name]]
    hard = [label for label, block in base["ledger"].items() if block["recall"] < 0.9]
    lines.append("| intent | n | " + " | ".join(v.replace("_", " ") for v in variants) + " |")
    lines.append("|---|---:|" + "---:|" * len(variants))
    for label in hard:
        row = [f"| `{label}` | {base['ledger'][label]['n']} |"]
        for name in variants:
            block = payload["arms"][name]["ledger"].get(label)
            row.append(f" {block['recall']:.2f} |" if block else " — |")
        lines.append("".join(row))

    lines += ["", "## Confusion pairs, ranked", ""]
    for name in variants:
        block = payload["arms"][name]
        lines.append(f"**{name.replace('_', ' ')}** — "
                     + " · ".join(f"{p['true']} → {p['pred']} ({p['n']})"
                                  for p in block["pairs"][:5]))
        lines.append("")

    lines += ["## Rows every variant gets wrong", "",
              "These are corpus problems, not model problems: the phrasing genuinely does not "
              "determine the intent, or two intents share a template.", "",
              "| utterance | truth | what the variants said |", "|---|---|---|"]
    for row in payload["all_wrong"][:20]:
        said = ", ".join(f"{name.split('_')[0]}={label}" for name, label in row["said"].items())
        lines.append(f"| {row['text']} | `{row['truth']}` | {said} |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    artifact = RESULTS / "six-variants.json"
    if not artifact.is_file():
        raise SystemExit(f"{artifact} not found; run six_variants.py first")
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    registry = load_registry()
    rows = load_routing_set()
    labels = registry.label_ids

    arms = {}
    for name, block in payload["arms"].items():
        predictions = block.get("predictions")
        if not predictions:
            continue
        arms[name] = {
            "accuracy": block["accuracy"],
            "ledger": ledger(predictions, rows, labels),
            "pairs": pairs(predictions, rows),
            "predictions": predictions,
        }

    all_wrong = []
    for i, row in enumerate(rows):
        said = {name: block["predictions"][i] for name, block in arms.items()}
        if all(label != row.tool for label in said.values()):
            all_wrong.append({"text": row.text, "truth": row.tool, "said": said})

    out = {
        "n": len(rows),
        "arms": {name: {k: v for k, v in block.items() if k != "predictions"}
                 for name, block in arms.items()},
        "all_wrong": all_wrong,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "per-intent.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    (RESULTS / "per-intent.md").write_text(render(out), encoding="utf-8")
    print(render(out)[:4000])
    print(f"\nwrote {RESULTS / 'per-intent.md'} and per-intent.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
