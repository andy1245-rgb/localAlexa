"""The linear rungs: the baseline, the vectoriser union, and calibration.

Everything here is in-process sklearn, same latency order as the 0.61 ms
baseline, and retrainable in about a second. Rungs 3–6 live in their own
modules because they need a model or a loop that the others do not.
"""

from __future__ import annotations

from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.svm import LinearSVC

from ..classifiers import SEED, build_classifier, build_vectorizer
from .predictors import SklearnPredictor
from .run import Rung


def word_vectorizer() -> TfidfVectorizer:
    return build_vectorizer("word")


def char_vectorizer() -> TfidfVectorizer:
    return build_vectorizer("char")


def union_vectorizer() -> FeatureUnion:
    """Word 1–2 grams beside char_wb 3–5 grams, each TF-IDF weighted on its own."""
    return FeatureUnion(
        [
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True
                ),
            ),
        ]
    )


def rung_0_baseline() -> Rung:
    """The quoted model: word TF-IDF + LinearSVC, 93.19% at 0.61 ms."""
    return Rung(
        "r0-word-svc",
        lambda: SklearnPredictor(lambda: make_pipeline(word_vectorizer(), build_classifier("linearsvc"))),
        note="the 93.19% / 0.61 ms baseline, re-measured on these folds",
    )


def rung_1_union() -> Rung:
    """Word + char TF-IDF union, same classifier."""
    return Rung(
        "r1-union-svc",
        lambda: SklearnPredictor(lambda: make_pipeline(union_vectorizer(), LinearSVC(C=1.0, random_state=SEED))),
        note="rung 1: vectoriser union",
    )


def rung_2_logreg() -> Rung:
    """The union with a model that has probabilities natively."""
    return Rung(
        "r2-union-logreg",
        lambda: SklearnPredictor(
            lambda: make_pipeline(
                union_vectorizer(), LogisticRegression(max_iter=2000, C=10.0, random_state=SEED)
            )
        ),
        note="rung 2: probability outputs from logistic regression",
    )


def rung_2_calibrated() -> Rung:
    """The union with LinearSVC wrapped in a sigmoid calibrator."""
    return Rung(
        "r2-union-svc-sigmoid",
        lambda: SklearnPredictor(
            lambda: make_pipeline(
                union_vectorizer(),
                CalibratedClassifierCV(LinearSVC(C=1.0, random_state=SEED), method="sigmoid", cv=3),
            )
        ),
        note="rung 2: LinearSVC + sigmoid calibration",
    )


def rung_2_calibrated_word() -> Rung:
    """Calibration alone, on the baseline's own word features."""
    return Rung(
        "r2-word-svc-sigmoid",
        lambda: SklearnPredictor(
            lambda: make_pipeline(
                word_vectorizer(),
                CalibratedClassifierCV(LinearSVC(C=1.0, random_state=SEED), method="sigmoid", cv=3),
            )
        ),
        note="rung 2: isolates calibration from the union",
    )


def rung_1_union_c4() -> Rung:
    """The best of the twelve union variants swept in ``sweep_union.py``.

    C=4 is not a free parameter: concatenating a second TF-IDF block rescales the
    decision values, so the baseline's C=1 is no longer tuned for the space. This
    one was picked on the folds it is scored on, so its grouped number is
    optimistic by whatever the sweep overfitted.
    """
    return Rung(
        "r1-union-svc-C4",
        lambda: SklearnPredictor(lambda: make_pipeline(union_vectorizer(), LinearSVC(C=4.0, random_state=SEED))),
        note="rung 1: union with C swept (selection on these folds)",
    )


def rung_2_logreg_c100() -> Rung:
    """Logistic regression on the baseline's word features, C=100.

    Picked by ``sweep_calibration.py``: the union buys nothing, and calibrating a
    LinearSVC with 200 rows means each calibrator sees 133 of them.
    """
    return Rung(
        "r2-word-logreg-C100",
        lambda: SklearnPredictor(
            lambda: make_pipeline(
                word_vectorizer(), LogisticRegression(max_iter=2000, C=100.0, random_state=SEED)
            )
        ),
        note="rung 2: natively calibrated, word features",
    )


LINEAR_RUNGS = (
    rung_0_baseline(),
    rung_1_union(),
    rung_1_union_c4(),
    rung_2_logreg_c100(),
    rung_2_calibrated(),
    rung_2_calibrated_word(),
)
