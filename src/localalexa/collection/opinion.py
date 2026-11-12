"""The second opinion, and the reason its agreement is weak evidence.

The brief lists "a second independent model agrees with the routing" as a weak
positive, and weak is the right word twice over:

* it is trained on the same 200 reference utterances as the router, so its errors
  are correlated with the router's — agreement is partly just shared training
  data, and correlated confirmation is not confirmation;
* it is cheap enough to run offline over the whole log, which is the only reason
  it is usable at all: a per-request second model would cost the request path
  more than the entire logging budget.

The model is rung 2 of the classifier ladder (word TF-IDF + logistic regression,
C=100) — a different architecture family from the MiniLM reference router, which
is the narrow sense in which it is independent.

Use: bulk weak positives, counted and weighted as weak, never as a substitute for
a correction or an effect check. ``docs/data-collection.md`` §5 says how the
signals are weighted against each other.
"""

from __future__ import annotations

from typing import Sequence

from ..data import NONE_LABEL, Registry, training_pairs
from ..ladder.predictors import SklearnPredictor
from ..ladder.rungs import rung_2_logreg_c100


class SecondOpinion:
    """A fitted, deterministic, offline second classifier over the same references."""

    def __init__(self, registry: Registry | None = None) -> None:
        self._registry = registry
        self._predictor: SklearnPredictor | None = None
        self._labels: list[str] = []

    def _fit(self) -> SklearnPredictor:
        if self._predictor is not None:
            return self._predictor
        registry = self._registry
        if registry is None:
            from ..data import load_registry

            registry = load_registry()
        pairs = training_pairs(registry)
        predictor = rung_2_logreg_c100().build()
        predictor.fit([text for text, _ in pairs], [label for _, label in pairs])
        self._predictor = predictor
        self._labels = list(predictor.classes)
        return predictor

    def labels(self, texts: Sequence[str]) -> list[str]:
        """One predicted label per text, fitted on first use."""
        if not texts:
            return []
        predictor = self._fit()
        matrix = predictor.scores(list(texts))
        return [self._labels[int(row.argmax())] for row in matrix]

    def label(self, text: str) -> str:
        return self.labels([text])[0]

    def agrees(self, text: str, label: str | None) -> bool:
        """True only when the opinion and the router said the same non-none label."""
        if label is None or label == NONE_LABEL:
            return False
        return self.label(text) == label
