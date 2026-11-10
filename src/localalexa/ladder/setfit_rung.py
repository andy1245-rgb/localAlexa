"""Rung 3: the SetFit recipe, which is what "fine-tuning" honestly means at 22 rows.

A full fine-tune of a sentence encoder on 382 rows would memorise them. SetFit
instead builds contrastive pairs from the labelled set, fine-tunes the encoder
on those pairs, and puts a light classifier head on the frozen embeddings — a
recipe designed for 8–16 examples per class, which is this corpus.

The encoder is 22M parameters and the whole fit is about a minute per fold on
CPU. That is 5–15x the baseline's per-utterance latency, which is rung 3's
bargain stated up front: it has to win on accuracy, because it cannot win on
speed.
"""

from __future__ import annotations

import tempfile
from typing import Sequence

import numpy as np

from ..classifiers import SEED
from .run import Rung

ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
HEAD_ITERATIONS = 8


class SetFitPredictor:
    """Contrastive encoder fine-tune plus a logistic head, behind the score interface."""

    def __init__(
        self,
        encoder: str = ENCODER,
        *,
        num_iterations: int = HEAD_ITERATIONS,
        batch_size: int = 16,
    ) -> None:
        self.encoder = encoder
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self._model = None
        self._classes: list[str] = []

    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> "SetFitPredictor":
        from datasets import Dataset
        from setfit import SetFitModel, Trainer, TrainingArguments

        self._model = SetFitModel.from_pretrained(self.encoder)
        arguments = TrainingArguments(
            batch_size=self.batch_size,
            num_epochs=1,
            num_iterations=self.num_iterations,
            seed=SEED,
            report_to=[],
            output_dir=tempfile.mkdtemp(prefix="setfit-"),
        )
        trainer = Trainer(
            model=self._model,
            train_dataset=Dataset.from_dict(
                {"text": list(texts), "label": list(labels)}
            ),
            args=arguments,
        )
        trainer.train()
        self._classes = [str(label) for label in self._model.labels]
        return self

    def scores(self, texts: Sequence[str]) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("predictor used before fit")
        return np.asarray(self._model.predict_proba(list(texts)), dtype=float)

    @property
    def classes(self) -> list[str]:
        return list(self._classes)


def rung_3_setfit() -> Rung:
    return Rung(
        "r3-setfit-minilm",
        lambda: SetFitPredictor(),
        note="rung 3: SetFit contrastive fine-tune + logistic head, MiniLM-L6 encoder",
    )
