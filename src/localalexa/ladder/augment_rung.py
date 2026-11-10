"""Rung 6: augmentation, which raises the ceiling for every rung above it.

Two sources of extra rows, both $0 and both fold-aware:

``template``
    The registry's own slot vocabularies, expanded over the templates of the
    phrasing families *not* held out. This is data the repo already has and
    never trains on — 1318 rows of well-formed commands across all families,
    of which a fold keeps all but its own. It needs no model and no download,
    so it is the first thing to measure.

``paraphrase``
    Rows written by a local generative model from the training references, kept
    only where the current model is unsure (see ``experiments/ladder/paraphrase.py``).
    Each row carries the reference it came from, so a fold drops the paraphrases
    of any reference it had to drop.

Both are scored against rung 2 on the same folds. The point of the rung is not
that a template-expanded model looks good on template-generated test rows — it
is whether more phrasing per intent is what the 22-example ceiling was hiding.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from ..data import expand_template, load_registry
from .run import Rung
from .rungs import rung_2_logreg_c100

PARAPHRASE_PATH = Path(__file__).resolve().parents[2] / "results" / "ladder" / "paraphrases.json"


def template_fills(registry, held_out: frozenset[str]) -> list[tuple[str, str]]:
    """Slot fills of every template whose phrasing family is not held out."""
    rows: dict[tuple[str, str], None] = {}
    for tool in registry.tools:
        for template in tool.templates:
            if template in held_out:
                continue
            for text in expand_template(tool, template):
                rows[(text, tool.id)] = None
    return sorted(rows)


def template_augmenter(registry, cap_per_label: int | None = None):
    """The rung-6 augment callable: rows from the registry's own vocabularies."""

    def augment(base: Sequence[tuple[str, str]], held_out: frozenset[str]) -> list[tuple[str, str]]:
        seen = {text for text, _ in base}
        extras = [
            (text, label)
            for text, label in template_fills(registry, held_out)
            if text not in seen
        ]
        if cap_per_label is None:
            return extras
        kept: list[tuple[str, str]] = []
        counts: dict[str, int] = {}
        for text, label in extras:
            if counts.get(label, 0) >= cap_per_label:
                continue
            counts[label] = counts.get(label, 0) + 1
            kept.append((text, label))
        return kept

    return augment


def load_paraphrases(path: str | Path = PARAPHRASE_PATH) -> list[tuple[str, str, str]]:
    """Rows as (text, label, source reference), or nothing if not generated yet."""
    target = Path(path)
    if not target.is_file():
        return []
    raw = json.loads(target.read_text(encoding="utf-8"))
    return [(row["text"], row["label"], row["source"]) for row in raw["rows"]]


def paraphrase_augmenter(rows: Sequence[tuple[str, str, str]]):
    """Keep a paraphrase only when the reference it came from survived the fold."""

    def augment(base: Sequence[tuple[str, str]], held_out: frozenset[str]) -> list[tuple[str, str]]:
        del held_out
        surviving = {text for text, _ in base}
        seen = set(surviving)
        extras = []
        for text, label, source in rows:
            if source not in surviving or text in seen:
                continue
            extras.append((text, label))
            seen.add(text)
        return extras

    return augment


def rung_6_template_augmented() -> Rung:
    """Rung 2's model, retrained on the references plus the registry's slot fills."""
    registry = load_registry()
    base = rung_2_logreg_c100()
    return Rung(
        "r6-template-augmented",
        base.build,
        augment=template_augmenter(registry),
        note="rung 6: + slot fills of the registry's non-held-out templates",
    )


def rung_6_paraphrase_augmented() -> Rung:
    """Rung 2's model, retrained on the references plus local-model paraphrases."""
    rows = load_paraphrases()
    if not rows:
        raise ImportError(f"no paraphrase rows at {PARAPHRASE_PATH}")
    base = rung_2_logreg_c100()
    return Rung(
        "r6-paraphrase-augmented",
        base.build,
        augment=paraphrase_augmenter(rows),
        note=f"rung 6: + {len(rows)} local-model paraphrases, provenance-filtered per fold",
    )


__all__ = [
    "paraphrase_augmenter",
    "rung_6_paraphrase_augmented",
    "rung_6_template_augmented",
    "template_augmenter",
    "template_fills",
]
