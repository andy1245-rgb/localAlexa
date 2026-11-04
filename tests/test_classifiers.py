"""Tests for the TF-IDF tier: real fits, above-chance accuracy, determinism."""

from __future__ import annotations

import pytest

from localalexa.classifiers import (
    DEFAULT_MODEL,
    MODEL_SPECS,
    TextClassifier,
    accuracy,
    build_classifier,
    build_vectorizer,
    per_class_accuracy,
    top_confusions,
    train_and_score,
)

TRAIN = [
    ("set a timer for three minutes", "timer_create"),
    ("start a ten minute countdown for pasta", "timer_create"),
    ("kick off a timer for the oven", "timer_create"),
    ("cancel the kitchen timer", "timer_cancel"),
    ("stop the pasta countdown", "timer_cancel"),
    ("get rid of that timer", "timer_cancel"),
    ("dim the living room lights", "light_set"),
    ("turn the kitchen lamp down", "light_set"),
    ("make the bedroom lights brighter", "light_set"),
    ("are the porch lights on", "light_query"),
    ("check whether the garage light is on", "light_query"),
    ("is the bedroom lamp still lit", "light_query"),
]

# Different wording, same intents: the classifier has to key off vocabulary that
# survives rephrasing rather than memorising whole utterances.
TEST = [
    ("begin a five minute countdown", "timer_create"),
    ("stop counting down", "timer_cancel"),
    ("dim the basement lights", "light_set"),
    ("is the kitchen light on", "light_query"),
    ("start a ninety second timer", "timer_create"),
    ("cancel my countdown", "timer_cancel"),
    ("turn the hall lamp down", "light_set"),
    ("are the basement lights still lit", "light_query"),
]


def test_fit_scores_above_chance_on_rephrased_input():
    model = TextClassifier("tfidf-word+linearsvc").fit(
        [t for t, _ in TRAIN], [l for _, l in TRAIN]
    )
    predictions = model.predict([t for t, _ in TEST])
    truth = [l for _, l in TEST]
    score = accuracy(predictions, truth)
    assert score >= 0.75, predictions
    assert score > 0.25 * 2  # chance is 1 in 4


def test_every_configuration_learns_the_synthetic_set():
    for name in MODEL_SPECS:
        model = TextClassifier(name).fit([t for t, _ in TRAIN], [l for _, l in TRAIN])
        score = accuracy(model.predict([t for t, _ in TEST]), [l for _, l in TEST])
        assert score >= 0.5, f"{name} scored {score}"


def test_predictions_are_deterministic_across_fits():
    texts = [t for t, _ in TEST]
    labels = [l for _, l in TRAIN]
    first = TextClassifier(DEFAULT_MODEL).fit([t for t, _ in TRAIN], labels).predict(texts)
    second = TextClassifier(DEFAULT_MODEL).fit([t for t, _ in TRAIN], labels).predict(texts)
    assert first == second


def test_predict_one_matches_predict_of_one_element():
    model = TextClassifier(DEFAULT_MODEL).fit([t for t, _ in TRAIN], [l for _, l in TRAIN])
    text = "begin a five minute countdown"
    assert model.predict_one(text) == model.predict([text])[0]


def test_fit_returns_self_and_records_labels():
    model = TextClassifier(DEFAULT_MODEL)
    assert model.fit([t for t, _ in TRAIN], [l for _, l in TRAIN]) is model
    assert model.labels_ == ["light_query", "light_set", "timer_cancel", "timer_create"]


def test_accuracy_and_per_class_accuracy_are_exact():
    truth = ["a", "a", "b", "b"]
    predicted = ["a", "b", "b", "b"]
    assert accuracy(predicted, truth) == 0.75
    scores = per_class_accuracy(predicted, truth, ["a", "b"])
    assert scores == {"a": 0.5, "b": 1.0}


def test_top_confusions_counts_repeated_pairs_only():
    truth = ["timer_cancel"] * 3 + ["light_set"] * 2 + ["timer_create"]
    predicted = ["timer_create"] * 3 + ["light_set", "light_set", "light_create"]
    confusions = top_confusions(predicted, truth, ["timer_cancel", "timer_create", "light_set"])
    assert confusions == [{"true": "timer_cancel", "pred": "timer_create", "n": 3}]


def test_unknown_model_and_vectorizer_kinds_are_rejected():
    with pytest.raises(ValueError):
        build_vectorizer("trigram")
    with pytest.raises(ValueError):
        build_classifier("xgboost")
    with pytest.raises(KeyError):
        TextClassifier("no-such-model")


def test_train_and_score_reports_the_documented_fields():
    train = TRAIN + [("what is the tallest mountain", "__none__")]
    test_texts = [t for t, _ in TEST] + ["what is the tallest mountain"]
    test_labels = [l for _, l in TEST] + ["__none__"]
    result = train_and_score(DEFAULT_MODEL, train, test_texts, test_labels)

    assert result["model"] == DEFAULT_MODEL
    assert 0.0 <= float(result["accuracy"]) <= 1.0
    assert float(result["accuracy"]) == pytest.approx(
        accuracy(
            TextClassifier(DEFAULT_MODEL)
            .fit([t for t, _ in train], [l for _, l in train])
            .predict(test_texts),
            test_labels,
        ),
        abs=1e-4,
    )
    assert float(result["none_detection"]) == 1.0
    assert float(result["false_none_rate"]) == 0.0
    assert float(result["latency_p50_ms"]) >= 0.0
    assert float(result["latency_p95_ms"]) >= float(result["latency_p50_ms"])
    assert set(result["worst_classes"]) <= set(test_labels)
    assert isinstance(result["top_confusions"], list)


def test_train_and_score_handles_a_test_set_without_none_rows():
    result = train_and_score(
        DEFAULT_MODEL, TRAIN, [t for t, _ in TEST], [l for _, l in TEST]
    )
    assert result["none_detection"] is None
    assert result["false_none_rate"] == 0.0


def test_recorded_accuracies_reproduce_on_the_shipped_data():
    """The three configs must land on the accuracies in results/historical/."""
    import json
    from pathlib import Path

    from localalexa.data import NONE_LABEL, load_registry, load_routing_set

    recorded = {
        row["model"]: row
        for row in json.loads(
            (
                Path(__file__).resolve().parents[1]
                / "results"
                / "historical"
                / "classic_results.json"
            ).read_text(encoding="utf-8")
        )
    }
    registry = load_registry()
    rows = load_routing_set()
    train = [
        (ref, tool.id) for tool in registry.tools for ref in tool.references
    ] + [(ref, NONE_LABEL) for ref in registry.none_references]
    for name in MODEL_SPECS:
        result = train_and_score(name, train, [r.text for r in rows], [r.tool for r in rows])
        assert result["accuracy"] == recorded[name]["accuracy"], name
        assert result["none_detection"] == recorded[name]["none_detection"], name
        assert result["worst_classes"] == recorded[name]["worst_classes"], name
        assert result["top_confusions"] == recorded[name]["top_confusions"], name
