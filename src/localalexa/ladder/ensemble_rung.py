"""Rung 5: an ensemble, built only after the correlation table says it is worth it.

Two models that are wrong on the same rows average into a slower model with the
same accuracy. So the members here are chosen for *different* failure modes —
different feature space (word vs word+char) and different loss (hinge vs
cross-entropy) — and the ensemble is reported next to the pairwise error
Jaccard from ``report.error_correlation`` so the claim can be checked.

Probabilities are averaged with equal weights: the members are all linear on the
same 200 rows, so a learned weight per member would be fitting noise at this
sample size.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from .predictors import Predictor, SklearnPredictor, softmax
from .run import Rung
from .rungs import word_vectorizer, union_vectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

from ..classifiers import SEED


class EnsemblePredictor:
    """Equal-weight probability average over members with different biases."""

    def __init__(self, members: Sequence[Callable[[], Predictor]]) -> None:
        self._factories = list(members)
        self._members: list[Predictor] = []
        self._classes: list[str] = []

    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> "EnsemblePredictor":
        self._members = [factory().fit(texts, labels) for factory in self._factories]
        self._classes = sorted({label for member in self._members for label in member.classes})
        return self

    def scores(self, texts: Sequence[str]) -> np.ndarray:
        if not self._members:
            raise RuntimeError("predictor used before fit")
        total = np.zeros((len(texts), len(self._classes)))
        for member in self._members:
            block = member.scores(texts)
            for column, label in enumerate(member.classes):
                total[:, self._classes.index(label)] += block[:, column]
        return total / len(self._members)

    @property
    def classes(self) -> list[str]:
        return list(self._classes)


class StandardisedEnsemble(EnsemblePredictor):
    """Average each member's row-standardised scores instead of its probabilities.

    A probability average hands the whole vote to whichever member is sharpest:
    the log-linear model's probabilities are near 0/1 while SetFit's sit around
    0.4, so the sharp model would drown the one whose errors are different. Row
    standardisation puts every member on the same scale per utterance.
    """

    def scores(self, texts: Sequence[str]) -> np.ndarray:
        if not self._members:
            raise RuntimeError("predictor used before fit")
        total = np.zeros((len(texts), len(self._classes)))
        for member in self._members:
            block = member.scores(texts)
            spread = block.std(axis=1, keepdims=True)
            spread[spread == 0] = 1.0
            standardised = (block - block.mean(axis=1, keepdims=True)) / spread
            for column, label in enumerate(member.classes):
                total[:, self._classes.index(label)] += standardised[:, column]
        return softmax(total / len(self._members))


def _members() -> list[Callable[[], Predictor]]:
    return [
        lambda: SklearnPredictor(
            lambda: make_pipeline(word_vectorizer(), LinearSVC(C=1.0, random_state=SEED))
        ),
        lambda: SklearnPredictor(
            lambda: make_pipeline(word_vectorizer(), LogisticRegression(max_iter=2000, C=100.0, random_state=SEED))
        ),
        lambda: SklearnPredictor(
            lambda: make_pipeline(union_vectorizer(), LinearSVC(C=4.0, random_state=SEED))
        ),
    ]


def _setfit_member() -> Predictor:
    from .setfit_rung import SetFitPredictor

    return SetFitPredictor()


def _logreg_member() -> Predictor:
    return SklearnPredictor(
        lambda: make_pipeline(
            word_vectorizer(), LogisticRegression(max_iter=2000, C=100.0, random_state=SEED)
        )
    )


def rung_5_ensemble() -> Rung:
    return Rung(
        "r5-ensemble-linear",
        lambda: EnsemblePredictor(_members()),
        note="rung 5: equal-weight average of word-svc, word-logreg and union-svc",
    )


def rung_5_ensemble_setfit() -> Rung:
    """The pairing the correlation table actually argues for: linear + SetFit."""
    return Rung(
        "r5-ensemble-setfit",
        lambda: EnsemblePredictor([_logreg_member, _setfit_member]),
        note="rung 5: probability average of word-logreg and SetFit (error Jaccard 0.12)",
    )


def rung_5_ensemble_setfit_standardised() -> Rung:
    return Rung(
        "r5-ensemble-setfit-z",
        lambda: StandardisedEnsemble([_logreg_member, _setfit_member]),
        note="rung 5: row-standardised average of word-logreg and SetFit",
    )
