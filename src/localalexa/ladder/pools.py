"""Which utterances a fold is allowed to train on.

The supervision budget is the registry's 200 references, the same pool the
93.19% baseline was trained on. For a held-out fold the pool loses any
reference that is a literal slot fill of one of that fold's phrasing families —
otherwise the model has been shown the exact wording it is about to be scored
on, which is the contamination the README already documents.

Augmented rows carry the reference they were paraphrased from, so they are
dropped with it.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from ..data import NONE_LABEL, Registry, expand_template, training_pairs


def family_texts(registry: Registry, families: Iterable[str]) -> set[str]:
    """Every utterance the registry's templates can render for these families."""
    wanted = set(families)
    out: set[str] = set()
    for tool in registry.tools:
        for template in tool.templates:
            if template in wanted:
                out.update(expand_template(tool, template))
    return out


def reference_pool(registry: Registry) -> list[tuple[str, str]]:
    return training_pairs(registry)


def fold_pool(registry: Registry, held_out: Iterable[str]) -> list[tuple[str, str]]:
    """The references a model may see when scored on ``held_out`` families."""
    contaminated = family_texts(registry, held_out)
    return [(text, label) for text, label in training_pairs(registry) if text not in contaminated]


def augmented_pool(
    base: Sequence[tuple[str, str]],
    augmented: Sequence[tuple[str, str, str]],
) -> list[tuple[str, str]]:
    """``base`` plus paraphrases whose source reference survived the fold filter."""
    surviving = {text for text, _ in base}
    extra = [(text, label) for text, label, source in augmented if source in surviving]
    return list(base) + extra


def label_counts(pairs: Iterable[tuple[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for _, label in pairs:
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items()))


def missing_labels(pairs: Iterable[tuple[str, str]], labels: Sequence[str]) -> list[str]:
    present = {label for _, label in pairs}
    return [label for label in labels if label not in present and label != NONE_LABEL]
