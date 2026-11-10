"""One fold assignment per ladder run, grouped by phrasing family.

Every rung is scored on exactly these folds, and the assignment is written into
the artifact. The grouping is the whole point: rows generated from one phrasing
family share wording, so a split that puts the same template on both sides of
the fence measures memorisation and reports a number that does not survive
rephrasing.

Folds are balanced on row count by ``GroupKFold``, which sorts groups by size
and deals them out — deterministic, no shuffling, no seed to keep.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from sklearn.model_selection import GroupKFold

DEFAULT_FOLDS = 5


@dataclass(frozen=True)
class FoldAssignment:
    """Phrasing family -> fold, plus the row weight each fold carries."""

    n_folds: int
    family_fold: dict[str, int]
    rows_per_fold: tuple[int, ...]

    def __post_init__(self) -> None:
        seen = set(self.family_fold.values())
        if not seen:
            raise ValueError("fold assignment is empty")
        if seen != set(range(self.n_folds)):
            raise ValueError(f"folds {sorted(seen)} are not 0..{self.n_folds - 1}")

    def fold_of(self, family: str) -> int:
        return self.family_fold[family]

    def indices(self, families: Sequence[str]) -> list[list[int]]:
        """Row indices per fold, in row order inside each fold."""
        out: list[list[int]] = [[] for _ in range(self.n_folds)]
        for i, family in enumerate(families):
            out[self.fold_of(family)].append(i)
        return out

    def as_dict(self) -> dict[str, object]:
        return {
            "n_folds": self.n_folds,
            "n_families": len(self.family_fold),
            "rows_per_fold": list(self.rows_per_fold),
            "family_fold": dict(sorted(self.family_fold.items())),
        }

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.as_dict(), indent=2) + "\n", encoding="utf-8")
        return target


def build_folds(
    families: Sequence[str], n_folds: int = DEFAULT_FOLDS
) -> FoldAssignment:
    """Balanced, deterministic fold per phrasing family."""
    n_families = len(set(families))
    if n_folds < 2:
        raise ValueError("need at least two folds")
    if n_folds > n_families:
        raise ValueError(f"{n_folds} folds cannot cover {n_families} families")

    splitter = GroupKFold(n_splits=n_folds)
    family_fold: dict[str, int] = {}
    rows_per_fold = [0] * n_folds
    placeholder = [[0] for _ in families]
    for fold, (_, test_idx) in enumerate(splitter.split(placeholder, groups=list(families))):
        for i in test_idx:
            family_fold[families[i]] = fold
        rows_per_fold[fold] = len(test_idx)
    return FoldAssignment(
        n_folds=n_folds,
        family_fold=family_fold,
        rows_per_fold=tuple(rows_per_fold),
    )
