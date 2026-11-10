"""One predictor interface, so every rung is scored by the same code.

A rung is a factory: calling it returns an unfitted predictor. The runner fits
it on a fold's training pool and asks for a score matrix — probability-like
values, one column per label — from which accuracy, the reject curve and the
calibration error all come. Rungs that genuinely have no probabilities (an
uncalibrated ``LinearSVC`` decides, it does not estimate) still return a matrix,
built by softmax over the decision values, and the ECE reports exactly how
badly that pretends.
"""

from __future__ import annotations

from typing import Callable, Protocol, Sequence

import numpy as np


class Predictor(Protocol):
    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> "Predictor": ...

    def scores(self, texts: Sequence[str]) -> np.ndarray: ...

    @property
    def classes(self) -> list[str]: ...


def softmax(matrix: np.ndarray) -> np.ndarray:
    shifted = matrix - matrix.max(axis=1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


class SklearnPredictor:
    """Wraps an unfitted sklearn estimator behind the score-matrix interface."""

    def __init__(self, build: Callable[[], object]) -> None:
        self._build = build
        self._estimator = None
        self._classes: list[str] = []

    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> "SklearnPredictor":
        self._estimator = self._build()
        self._estimator.fit(list(texts), list(labels))
        self._classes = [str(c) for c in self._estimator.classes_]
        return self

    def scores(self, texts: Sequence[str]) -> np.ndarray:
        if self._estimator is None:
            raise RuntimeError("predictor used before fit")
        if hasattr(self._estimator, "predict_proba"):
            return np.asarray(self._estimator.predict_proba(list(texts)), dtype=float)
        return softmax(np.asarray(self._estimator.decision_function(list(texts)), dtype=float))

    @property
    def classes(self) -> list[str]:
        return list(self._classes)


def predict_rows(predictor: Predictor, texts: Sequence[str]) -> list[str]:
    """Argmax label per text, using the predictor's own class order."""
    matrix = predictor.scores(texts)
    classes = predictor.classes
    return [classes[int(i)] for i in matrix.argmax(axis=1)]


def top_k(predictor: Predictor, texts: Sequence[str], k: int = 3) -> list[list[str]]:
    """The k highest-scoring labels per text, best first."""
    matrix = predictor.scores(texts)
    classes = predictor.classes
    order = np.argsort(-matrix, axis=1, kind="stable")[:, :k]
    return [[classes[int(i)] for i in row] for row in order]
