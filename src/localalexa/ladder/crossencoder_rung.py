"""Rung 4: re-rank the top 3 with a cross-encoder.

The gate was checked first (``experiments/ladder/top3_diagnostic.py``): on the
grouped folds the linear model's truth is inside its top 3 for 98.4% of rows, and
29 of its 35 mistakes are in-top-3 mistakes. So a re-ranker has something to
recover — up to 7.6 points if it were perfect — and the rung is worth building.

The re-ranker is a MiniLM encoder with a two-way head over a ``(utterance,
candidate intent)`` pair. It trains on hard negatives: the base model's own top
3 for each training utterance, so it learns exactly the discrimination it will be
asked for. Latency goes from under a millisecond to a few milliseconds per
candidate, which is the trade this rung exists to price.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..classifiers import SEED
from .predictors import Predictor, SklearnPredictor, softmax
from .run import Rung

ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
MAX_LENGTH = 48
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 3e-5
HEAD_LEARNING_RATE = 1e-3
CANDIDATES = 3


def intent_phrase(label: str) -> str:
    """How a label is spelled out for the pair: ``light_set`` -> "light set"."""
    return label.replace("__none__", "no command").replace("_", " ")


class CrossEncoderReranker:
    """Base linear top-3, then a pair-scoring cross-encoder picks among them."""

    def __init__(
        self,
        base: Predictor | None = None,
        *,
        encoder: str = ENCODER,
        epochs: int = EPOCHS,
        candidates: int = CANDIDATES,
    ) -> None:
        self._base = base if base is not None else SklearnPredictor(_base_pipeline)
        self.encoder = encoder
        self.epochs = epochs
        self.candidates = candidates
        self._model = None
        self._tokenizer = None
        self._classes: list[str] = []

    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> "CrossEncoderReranker":
        from transformers import AutoTokenizer

        texts = list(texts)
        labels = list(labels)
        self._base.fit(texts, labels)
        self._classes = list(self._base.classes)
        self._tokenizer = AutoTokenizer.from_pretrained(self.encoder)
        self._model = self._build_model()

        pairs, targets = self._training_pairs(texts, labels)
        self._train(pairs, targets)
        return self

    def _build_model(self):
        import torch
        from transformers import AutoModelForSequenceClassification

        torch.manual_seed(SEED)
        return AutoModelForSequenceClassification.from_pretrained(
            self.encoder, num_labels=2
        ).train()

    def _training_pairs(
        self, texts: Sequence[str], labels: Sequence[str]
    ) -> tuple[list[tuple[str, str]], list[int]]:
        """Hard negatives: the base model's top-k for each training utterance."""
        matrix = self._base.scores(texts)
        order = np.argsort(-matrix, axis=1, kind="stable")[:, : self.candidates]
        pairs, targets = [], []
        for row, text, truth in zip(order, texts, labels):
            offered = {self._classes[int(i)] for i in row} | {truth}
            for candidate in offered:
                pairs.append((text, intent_phrase(candidate)))
                targets.append(1 if candidate == truth else 0)
        return pairs, targets

    def _train(self, pairs: list[tuple[str, str]], targets: list[int]) -> None:
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        encoded = self._tokenizer(
            [p[0] for p in pairs],
            [p[1] for p in pairs],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        dataset = TensorDataset(
            encoded["input_ids"],
            encoded["attention_mask"],
            torch.tensor(targets, dtype=torch.long),
        )
        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            generator=torch.Generator().manual_seed(SEED),
        )
        # The encoder arrives pretrained; the head is random. One learning rate for
        # both leaves the head crawling, and 582 pairs cannot afford that.
        encoder_parameters, head_parameters = [], []
        for name, parameter in self._model.named_parameters():
            (head_parameters if "classifier" in name else encoder_parameters).append(parameter)
        optimizer = torch.optim.AdamW(
            [
                {"params": encoder_parameters, "lr": LEARNING_RATE},
                {"params": head_parameters, "lr": HEAD_LEARNING_RATE},
            ]
        )
        # One positive per utterance against ~3 hard negatives: without this the
        # model minimises loss by answering "not this one" to everything.
        weight = torch.tensor([1.0, float(len(targets) - sum(targets)) / max(sum(targets), 1)])
        self._model.train()
        for _ in range(self.epochs):
            for input_ids, attention_mask, target in loader:
                optimizer.zero_grad()
                logits = self._model(
                    input_ids=input_ids, attention_mask=attention_mask
                ).logits
                loss = torch.nn.functional.cross_entropy(logits, target, weight=weight)
                loss.backward()
                optimizer.step()
        self._model.eval()

    def _pair_scores(self, texts: Sequence[str], candidates: Sequence[str]) -> np.ndarray:
        """P(match) for one (text, candidate) pair per position."""
        import torch

        encoded = self._tokenizer(
            list(texts),
            [intent_phrase(c) for c in candidates],
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = self._model(**encoded).logits
        return softmax(logits.numpy())[:, 1]

    def scores(self, texts: Sequence[str]) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("predictor used before fit")
        texts = list(texts)
        base = self._base.scores(texts)
        order = np.argsort(-base, axis=1, kind="stable")[:, : self.candidates]

        # One forward pass over every (text, candidate) pair, not one per row.
        flat_texts, flat_candidates = [], []
        for i, text in enumerate(texts):
            for column in order[i]:
                flat_texts.append(text)
                flat_candidates.append(self._classes[int(column)])
        flat = self._pair_scores(flat_texts, flat_candidates)

        reranked = np.zeros_like(base)
        width = self.candidates
        for i in range(len(texts)):
            window = flat[i * width : (i + 1) * width]
            total = float(window.sum())
            weights = window / total if total else np.full(width, 1.0 / width)
            for position, column in enumerate(order[i]):
                reranked[i, int(column)] = float(weights[position])
        return reranked

    @property
    def classes(self) -> list[str]:
        return list(self._classes)


def _base_pipeline():
    from sklearn.pipeline import make_pipeline

    from .rungs import word_vectorizer
    from sklearn.linear_model import LogisticRegression

    return make_pipeline(
        word_vectorizer(), LogisticRegression(max_iter=2000, C=100.0, random_state=SEED)
    )


def rung_4_crossencoder() -> Rung:
    return Rung(
        "r4-crossencoder-top3",
        lambda: CrossEncoderReranker(SklearnPredictor(_base_pipeline)),
        note="rung 4: MiniLM cross-encoder re-ranks the linear model's top 3",
    )
