"""Classic-ML tier: TF-IDF over the registry references, linear classifier on top.

Three configurations, the same ones the experiments measured. Trained on the
200 reference utterances only, so the supervision budget matches the semantic
router (which embeds those same references).

Sub-millisecond per query, no embedding model to download, retrained in under a
second as the registry grows.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC

# Fixed so a fit is reproducible: liblinear's coordinate descent is shuffled.
SEED = 0


@dataclass(frozen=True)
class ModelSpec:
    name: str
    vectorizer: str  # "word" | "char"
    classifier: str  # "logreg" | "linearsvc"


MODEL_SPECS: dict[str, ModelSpec] = {
    "tfidf-word+logreg": ModelSpec("tfidf-word+logreg", "word", "logreg"),
    "tfidf-char+logreg": ModelSpec("tfidf-char+logreg", "char", "logreg"),
    "tfidf-word+linearsvc": ModelSpec("tfidf-word+linearsvc", "word", "linearsvc"),
}
DEFAULT_MODEL = "tfidf-word+linearsvc"


def build_vectorizer(kind: str) -> TfidfVectorizer:
    if kind == "word":
        return TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    if kind == "char":
        return TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True
        )
    raise ValueError(f"unknown vectorizer kind: {kind!r}")


def build_classifier(kind: str):
    if kind == "logreg":
        return LogisticRegression(max_iter=2000, C=10.0, random_state=SEED)
    if kind == "linearsvc":
        return LinearSVC(C=1.0, random_state=SEED)
    raise ValueError(f"unknown classifier kind: {kind!r}")


class TextClassifier:
    """TF-IDF + linear classifier over utterance text."""

    def __init__(self, spec: ModelSpec | str = DEFAULT_MODEL) -> None:
        if isinstance(spec, str):
            spec = MODEL_SPECS[spec]
        self.spec = spec
        self.vectorizer = build_vectorizer(spec.vectorizer)
        self.classifier = build_classifier(spec.classifier)
        self.labels_: list[str] | None = None

    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> "TextClassifier":
        matrix = self.vectorizer.fit_transform(list(texts))
        self.classifier.fit(matrix, list(labels))
        self.labels_ = sorted(set(labels))
        return self

    def predict(self, texts: Sequence[str]) -> list[str]:
        return [str(p) for p in self.classifier.predict(self.vectorizer.transform(list(texts)))]

    def predict_one(self, text: str) -> str:
        return self.predict([text])[0]


def accuracy(predicted: Sequence[str], truth: Sequence[str]) -> float:
    return float(np.mean([p == t for p, t in zip(predicted, truth)]))


def per_class_accuracy(
    predicted: Sequence[str], truth: Sequence[str], labels: Sequence[str]
) -> dict[str, float]:
    scores = {}
    for label in labels:
        idx = [i for i, t in enumerate(truth) if t == label]
        if idx:
            scores[label] = round(
                float(np.mean([predicted[i] == truth[i] for i in idx])), 3
            )
    return scores


def top_confusions(
    predicted: Sequence[str], truth: Sequence[str], labels: Sequence[str], limit: int = 8
) -> list[dict[str, object]]:
    matrix = confusion_matrix(truth, predicted, labels=list(labels))
    pairs = []
    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            if i != j and matrix[i, j] >= 2:
                pairs.append(
                    {"true": true_label, "pred": pred_label, "n": int(matrix[i, j])}
                )
    pairs.sort(key=lambda c: -int(c["n"]))
    return pairs[:limit]


def train_and_score(
    model: str,
    train_pairs: Sequence[tuple[str, str]],
    test_texts: Sequence[str],
    test_labels: Sequence[str],
    *,
    none_label: str = "__none__",
) -> dict[str, object]:
    """Fit one configuration and return the metrics block the experiments record."""
    classifier = TextClassifier(model)
    started = time.perf_counter()
    classifier.fit([t for t, _ in train_pairs], [l for _, l in train_pairs])
    train_seconds = time.perf_counter() - started

    latencies, predictions = [], []
    for text in test_texts:
        t0 = time.perf_counter()
        predictions.append(classifier.predict_one(text))
        latencies.append((time.perf_counter() - t0) * 1000)
    latencies_arr = np.asarray(latencies)

    labels = sorted({label for _, label in train_pairs})
    none_idx = [i for i, t in enumerate(test_labels) if t == none_label]
    tool_idx = [i for i, t in enumerate(test_labels) if t != none_label]
    scores = per_class_accuracy(predictions, test_labels, labels)
    return {
        "model": classifier.spec.name,
        "train_seconds": round(train_seconds, 2),
        "accuracy": round(accuracy(predictions, test_labels), 4),
        "none_detection": round(
            float(np.mean([predictions[i] == none_label for i in none_idx])), 3
        )
        if none_idx
        else None,
        "false_none_rate": round(
            float(np.mean([predictions[i] == none_label for i in tool_idx])), 3
        )
        if tool_idx
        else None,
        "latency_p50_ms": round(float(np.percentile(latencies_arr, 50)), 2),
        "latency_p95_ms": round(float(np.percentile(latencies_arr, 95)), 2),
        "worst_classes": dict(sorted(scores.items(), key=lambda kv: kv[1])[:5]),
        "top_confusions": top_confusions(predictions, test_labels, labels),
    }
