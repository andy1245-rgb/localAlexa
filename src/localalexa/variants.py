"""Six things the ladder never tried, expressed as fitted predictors over the same folds.

Each one is a different *structure*, not a different hyper-parameter, and each is given the
same grouped folds, the same training pools and the same scoring as the ladder so the numbers
can sit beside 90.84% (word logreg C=100, grouped) without a footnote:

``regex_features``
    the hand-written patterns are inputs instead of a gate: a 17-column block saying which
    tools' patterns fire, unioned onto the TF-IDF features. The patterns encode knowledge no
    classifier can learn from 200 rows, and until now they were thrown away after being used
    to short-circuit.
``fuse``
    the tiers averaged, not cascaded: router similarity, TF-IDF probabilities and regex scores
    combined as distributions, each row standardised first. The other project gained 26 rows
    this way while every cascade measured here lost.
``none_first``
    "is this a command at all" asked separately from "which command", because they are
    different questions and the abstain class is the one the linear tiers are worst at.
``hierarchical``
    family, then action inside the family — the confusions this repo has documented
    (``light_set``/``light_query``, ``thermostat_set``/``thermostat_offset``) are within-family.
``distilled``
    the measured ensemble's answers used as supervision for a fast model: one 0.38 ms
    student instead of a 7.85 ms ensemble.
``per_class_thresholds``
    one global gate replaced by per-intent thresholds chosen to hold a precision target,
    because a confidence of 0.4 means different things for ``timer_cancel`` and ``docs_query``.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.preprocessing import StandardScaler

from .classifiers import SEED
from .cascade import CompiledPatterns
from .data import NONE_LABEL, Registry, Row, expand_template
from .router import Choice, entropy_confidence, decide

# ------------------------------------------------------------------ shared pieces

FAMILIES_ORDER = ("climate", "knowledge", "lights", "none", "reminders", "shopping", "timers")


def word_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)


class RuleFeatures(BaseEstimator, TransformerMixin):
    """Which tools' hand-written patterns fire, as binary columns.

    One column per tool plus one for "any pattern fired". Sparse output so it unions cleanly
    with the TF-IDF block without turning the pipeline dense. The patterns come in as a
    ``CompiledPatterns`` because that is the object the cascade already builds.
    """

    def __init__(self, tool_ids: Sequence[str], compiled: CompiledPatterns) -> None:
        self.tool_ids = tuple(tool_ids)
        self.compiled = compiled

    def fit(self, texts, y=None) -> "RuleFeatures":
        return self

    def transform(self, texts) -> "object":
        import scipy.sparse as sp

        position = {tool: i for i, tool in enumerate(self.tool_ids)}
        rows = []
        for text in texts:
            hits = self.compiled.probe(text)
            vector = np.zeros(len(self.tool_ids) + 1, dtype=np.float32)
            for tool in hits:
                if tool in position:
                    vector[position[tool]] = 1.0
            vector[-1] = 1.0 if hits else 0.0
            rows.append(vector)
        return sp.csr_matrix(np.vstack(rows)) if rows else sp.csr_matrix((0, len(self.tool_ids) + 1))


def logreg() -> LogisticRegression:
    return LogisticRegression(max_iter=2000, C=100.0, random_state=SEED)


class LinearPredictor:
    """A fitted logreg pipeline behind ``predict``/``probabilities``."""

    def __init__(self, pipeline) -> None:
        self.pipeline = pipeline
        self.classes = [str(c) for c in pipeline.classes_]

    def predict(self, text: str) -> str:
        return self.classes[int(self.pipeline.predict_proba([text])[0].argmax())]

    def probabilities(self, text: str) -> dict[str, float]:
        row = self.pipeline.predict_proba([text])[0]
        return {label: float(p) for label, p in zip(self.classes, row)}


def fit_plain(pool: Sequence[tuple[str, str]]) -> LinearPredictor:
    """The ladder's rung 2: word TF-IDF + logistic regression C=100."""
    pipeline = make_pipeline(word_vectorizer(), logreg())
    pipeline.fit([text for text, _ in pool], [label for _, label in pool])
    return LinearPredictor(pipeline)


# ------------------------------------------------------------------ (1) regex as features


def fit_regex_features(registry: Registry, pool: Sequence[tuple[str, str]]) -> LinearPredictor:
    compiled = CompiledPatterns.from_registry(registry)
    pipeline = make_pipeline(
        FeatureUnion([
            ("text", word_vectorizer()),
            ("rules", RuleFeatures(tuple(registry.tool_ids), compiled=compiled)),
        ]),
        logreg(),
    )
    pipeline.fit([text for text, _ in pool], [label for _, label in pool])
    return LinearPredictor(pipeline)


# ------------------------------------------------------------------ (2) fusion


class Fuser:
    """Members averaged as distributions, each row-standardised first.

    The members are the three things the repo already computes and never combines: the
    reference-similarity router's per-tool distribution, the TF-IDF model's probabilities, and
    the regex patterns as a distribution. Standardising per row is what stops the sharpest
    member — the TF-IDF model — from deciding every row alone.
    """

    def __init__(self, members: Sequence[str]) -> None:
        self.members = tuple(members)
        self.linear: LinearPredictor | None = None
        self.compiled: CompiledPatterns | None = None
        self.router = None
        self.labels: list[str] = []

    def fit(
        self,
        registry: Registry,
        pool: Sequence[tuple[str, str]],
        router=None,
    ) -> "Fuser":
        self.labels = list(registry.label_ids)
        self.compiled = CompiledPatterns.from_registry(registry)
        if "linear" in self.members:
            self.linear = fit_plain(pool)
        if "router" in self.members:
            self.router = router
        return self

    def _regex_distribution(self, text: str) -> dict[str, float]:
        hits = set(self.compiled.probe(text)) if self.compiled else set()
        scores = {label: (1.0 if label in hits else 0.0) for label in self.labels}
        total = sum(scores.values())
        return ({label: 1.0 / len(self.labels) for label in self.labels} if total == 0
                else {label: value / total for label, value in scores.items()})

    def _member_distributions(self, text: str) -> list[dict[str, float]]:
        out = []
        if self.linear is not None:
            out.append(self.linear.probabilities(text))
        if self.router is not None:
            choice: Choice = self.router.predict(text)
            out.append({label: float(choice.probabilities.get(label, 0.0)) for label in self.labels})
        if self.compiled is not None:
            out.append(self._regex_distribution(text))
        return out

    def predict(self, text: str) -> str:
        total = np.zeros(len(self.labels))
        distributions = self._member_distributions(text)
        for distribution in distributions:
            values = np.array([distribution.get(label, 0.0) for label in self.labels])
            spread = values.std() or 1.0
            total += (values - values.mean()) / spread
        return self.labels[int(total.argmax())]


# ------------------------------------------------------------------ (3) none first


class NoneFirst:
    """A binary "is this a command" stage in front of a 17-way stage."""

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self.gate = None
        self.action: LinearPredictor | None = None
        self.none_label = NONE_LABEL

    def fit(self, pool: Sequence[tuple[str, str]]) -> "NoneFirst":
        commands = [(text, label) for text, label in pool if label != NONE_LABEL]
        gate = make_pipeline(
            word_vectorizer(),
            LogisticRegression(max_iter=2000, C=100.0, class_weight="balanced", random_state=SEED),
        )
        gate.fit(
            [text for text, _ in pool],
            ["command" if label != NONE_LABEL else "none" for text, label in pool],
        )
        self.gate = gate
        pipeline = make_pipeline(word_vectorizer(), logreg())
        pipeline.fit([text for text, _ in commands], [label for _, label in commands])
        self.action = LinearPredictor(pipeline)
        return self

    def p_none(self, text: str) -> float:
        row = self.gate.predict_proba([text])[0]
        classes = list(self.gate.classes_)
        return float(row[classes.index("none")]) if "none" in classes else 0.0

    def predict(self, text: str) -> str:
        if self.p_none(text) >= self.threshold:
            return self.none_label
        return self.action.predict(text)


# ------------------------------------------------------------------ (4) hierarchical


class Hierarchical:
    """Family first, then the action inside that family."""

    def __init__(self) -> None:
        self.family = None
        self.per_family: dict[str, LinearPredictor] = {}
        self.family_of_label: dict[str, str] = {}

    def fit(self, registry: Registry, pool: Sequence[tuple[str, str]]) -> "Hierarchical":
        family_of_tool = {tool.id: tool.family for tool in registry.tools}
        family_of_tool[NONE_LABEL] = "none"
        self.family_of_label = family_of_tool
        pipeline = make_pipeline(word_vectorizer(), logreg())
        pipeline.fit(
            [text for text, _ in pool],
            [family_of_tool.get(label, "none") for _, label in pool],
        )
        self.family = LinearPredictor(pipeline)
        for family in sorted(set(family_of_tool.values())):
            subset = [(text, label) for text, label in pool if family_of_tool.get(label) == family]
            if len({label for _, label in subset}) < 2:
                continue
            pipeline = make_pipeline(word_vectorizer(), logreg())
            pipeline.fit([text for text, _ in subset], [label for _, label in subset])
            self.per_family[family] = LinearPredictor(pipeline)
        return self

    def predict(self, text: str) -> str:
        family = self.family.predict(text)
        member = self.per_family.get(family)
        if member is None:
            # a family with one label (or none) has nothing to disambiguate
            return next((label for label, fam in self.family_of_label.items() if fam == family),
                        NONE_LABEL)
        distribution = member.probabilities(text)
        inside = {label: score for label, score in distribution.items()
                  if self.family_of_label.get(label) == family}
        # a member may only answer inside its own family; outside that the family's own
        # most-probable label wins
        return mapping_top(inside) or next(
            (label for label, fam in self.family_of_label.items() if fam == family), NONE_LABEL)


# ------------------------------------------------------------------ (6) per-class thresholds


@dataclass(frozen=True)
class ClassThresholds:
    thresholds: Mapping[str, float]
    target_precision: float

    def as_dict(self) -> dict[str, object]:
        return {"target_precision": self.target_precision,
                "thresholds": {k: round(v, 4) for k, v in self.thresholds.items()}}


def choose_class_thresholds(
    probabilities: Sequence[Mapping[str, float]],
    truth: Sequence[str],
    *,
    target_precision: float = 0.90,
) -> ClassThresholds:
    """A per-class confidence floor that holds ``target_precision`` on the classes that can.

    Chosen on out-of-fold predictions inside the training pool, never on the rows being scored.
    The sweep walks each class's claims from highest score down and keeps the *lowest* score at
    which the precision of everything above it still meets the target — that is the widest
    coverage the class can honestly serve. A class that never gets there is given a floor of 1.1,
    so it escalates everything it would otherwise claim: refusing to answer is the honest
    outcome at the wrong precision, and pretending 0.0 is a threshold would answer everything.
    """
    labels = sorted({label for mapping in probabilities for label in mapping})
    thresholds: dict[str, float] = {}
    for label in labels:
        scored = sorted(
            ((mapping.get(label, 0.0), want == label)
             for mapping, want in zip(probabilities, truth)),
            key=lambda pair: -pair[0],
        )
        qualifying = 1.1
        claimed = correct = 0
        for score, right in scored:
            claimed += 1
            correct += 1 if right else 0
            if correct / claimed >= target_precision:
                qualifying = score
        thresholds[label] = round(float(qualifying), 6)
    return ClassThresholds(thresholds=thresholds, target_precision=target_precision)


def mapping_top(mapping: Mapping[str, float], default: str = "") -> str:
    """The argmax of a distribution, with a stated default for an empty one."""
    return max(mapping, key=lambda key: mapping[key]) if mapping else default


def threshold_decision(
    distribution: Mapping[str, float], thresholds: Mapping[str, float]
) -> tuple[str | None, float]:
    """The label to answer with, or ``None`` to escalate, and the confidence that decided."""
    label = mapping_top(distribution, "")
    score = distribution.get(label, 0.0)
    required = thresholds.get(label, 1.1)
    return (None if score < required else label), score
