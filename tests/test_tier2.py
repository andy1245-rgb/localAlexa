"""Tests for the measured tier-2 join: who answered, whether they were right, and what is missing.

The join is where a parameter becomes a measurement, so the cases that matter are the ones a
wrong implementation would get quietly: a row nothing scored must be reported as missing rather
than counted as wrong, the tier-2 confidence must be a gate rather than decoration, and coverage
and precision must be over the same set of rows.
"""

from __future__ import annotations

import json

import pytest

from localalexa.compare import RowScore
from localalexa.tier2 import (
    TIER2_THRESHOLDS,
    Tier2Row,
    _answers,
    _best_for_precision,
    _point,
    load_measurement,
    measure,
    render,
)


def score(truth: str, model: str, confidence: float, regex: str | None = None,
          text: str | None = None) -> RowScore:
    return RowScore(
        fold=0,
        truth=truth,
        regex_label=regex,
        model_label=model,
        model_confidence=confidence,
        model_similarity=confidence,
        text=text or f"utterance for {truth}",
    )


def write_measurement(path, rows) -> str:
    payload = {
        "per_row": [
            {"text": text, "pred": pred, "confidence": confidence, "correct": correct,
             "true": truth}
            for text, truth, pred, confidence, correct in rows
        ]
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


# ------------------------------------------------------------------ who answers


def test_a_regex_row_is_answered_by_tier_zero_and_never_reaches_tier_two():
    row = score("lights", "lights", 0.99, regex="lights")
    assert _answers(row, 0.5, 0.0, {}) == (True, "tier0")


def test_a_confident_model_row_is_answered_by_tier_one():
    row = score("timer", "timer", 0.90)
    assert _answers(row, 0.5, 0.0, {}) == (True, "tier1")


def test_an_escalated_row_is_answered_by_tier_two_when_its_confidence_clears():
    row = score("timer", "wrong", 0.10, text="a timer please")
    tier2 = {"a timer please": Tier2Row("a timer please", "timer", 0.80, True)}
    assert _answers(row, 0.5, 0.5, tier2) == (True, "tier2")


def test_tier_two_confidence_is_a_gate_and_hands_the_rest_upward():
    row = score("timer", "wrong", 0.10, text="a timer please")
    unsure = {"a timer please": Tier2Row("a timer please", "timer", 0.20, True)}
    assert _answers(row, 0.5, 0.5, unsure) == (None, "tier3")
    no_confidence = {"a timer please": Tier2Row("a timer please", "timer", None, True)}
    assert _answers(row, 0.5, 0.5, no_confidence) == (None, "tier3")


def test_a_row_the_tier_two_model_never_scored_is_missing_not_wrong():
    row = score("timer", "wrong", 0.10, text="never measured")
    assert _answers(row, 0.5, 0.0, {}) == (None, "missing")


# ------------------------------------------------------------------ coverage and precision


def test_coverage_and_precision_are_over_the_same_rows():
    scores = [
        score("a", "a", 0.9),               # tier 1, right
        score("b", "x", 0.10, text="b"),    # escalated, tier 2 right
        score("c", "x", 0.10, text="c"),    # escalated, tier 2 wrong
        score("d", "x", 0.10, text="d"),    # escalated, tier 2 unsure -> tier 3
    ]
    tier2 = {
        "b": Tier2Row("b", "b", 0.9, True),
        "c": Tier2Row("c", "y", 0.9, False),
        "d": Tier2Row("d", "d", 0.1, True),
    }
    point = _point(scores, tier2, cascade_gate=0.5, tier2_gate=0.5)
    assert point["answered"] == 3
    assert point["coverage"] == pytest.approx(0.75)
    assert point["correct"] == 2
    assert point["precision"] == pytest.approx(2 / 3, abs=1e-3)
    assert point["handed_to_tier3"] == 1
    assert point["missing_tier2_rows"] == 0
    assert point["answered_by_tier"] == {"tier0": 0, "tier1": 1, "tier2": 2}


def test_a_perfect_tier_three_is_a_ceiling_and_a_useless_one_is_a_floor():
    scores = [
        score("a", "a", 0.9),
        score("b", "x", 0.10, text="b"),
    ]
    tier2 = {"b": Tier2Row("b", "b", 0.1, True)}  # unsure, so it goes to tier 3
    point = _point(scores, tier2, cascade_gate=0.5, tier2_gate=0.5)
    assert point["coverage"] == pytest.approx(0.5)
    assert point["accuracy_over_all_rows_if_tier3_perfect"] == pytest.approx(1.0)
    assert point["accuracy_over_all_rows_if_tier3_useless"] == pytest.approx(0.5)


def test_best_coverage_at_a_precision_target_is_the_high_coverage_point():
    points = [
        {"coverage": 0.90, "precision": 0.80, "answered": 90},
        {"coverage": 0.70, "precision": 0.93, "answered": 70},
        {"coverage": 0.50, "precision": 0.97, "answered": 50},
    ]
    best = _best_for_precision(points, 0.90)
    assert best is not None and best["coverage"] == 0.70


def test_no_point_reaching_the_target_returns_none_rather_than_the_closest():
    points = [{"coverage": 0.90, "precision": 0.80, "answered": 90}]
    assert _best_for_precision(points, 0.95) is None


# ------------------------------------------------------------------ the measurement file


def test_loading_a_measurement_keys_rows_by_text(tmp_path):
    path = write_measurement(tmp_path / "m.json", [
        ("turn on the lights", "light_set", "light_set", 0.9, True),
        ("a timer", "timer_create", "timer_cancel", 0.4, False),
    ])
    loaded = load_measurement(path)
    assert set(loaded) == {"turn on the lights", "a timer"}
    assert loaded["a timer"].correct is False
    assert loaded["a timer"].confidence == pytest.approx(0.4)


def test_an_empty_measurement_file_is_an_error_not_an_empty_join(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text(json.dumps({"per_row": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="no per_row"):
        load_measurement(path)


def test_the_join_runs_on_the_real_corpus_and_says_how_many_rows_it_joined(tmp_path):
    # every row scored as correct at high confidence: coverage must reach 100% and precision
    # must be perfect, which is only true if the join reaches tier 0, tier 1 and tier 2 rows
    rows = json.loads((__import__("pathlib").Path(__file__).resolve().parents[1]
                       / "data" / "routing_382.json").read_text(encoding="utf-8"))
    path = write_measurement(
        tmp_path / "perfect.json",
        [(row["text"], row["tool"], row["tool"], 0.99, True) for row in rows],
    )
    payload = measure(tier2_path=path, folds=2, cascade_gates=(0.35,), tier2_gates=(0.0,))
    point = payload["points"][0]
    assert payload["meta"]["rows_joined"] == 382
    # every row is answered, and the escalated ones are now right — but the cheap tiers' own
    # errors are still errors, so precision is above the cascade's auto accuracy, not 1.0
    assert point["coverage"] == pytest.approx(1.0)
    assert point["handed_to_tier3"] == 0
    assert point["answered_by_tier"]["tier2"] > 0
    assert point["precision"] > payload["bars"]["best_single_classifier_accuracy"]
    assert "tier-2 accuracy 100.00%" in render(payload)


def test_the_join_reports_its_own_incompleteness(tmp_path):
    # a measurement that covers two rows cannot be reported as if it covered the corpus
    path = write_measurement(tmp_path / "tiny.json", [
        ("set a timer for 90 seconds", "timer_create", "timer_create", 0.9, True),
        ("run a timer for 2 hours", "timer_create", "timer_create", 0.9, True),
    ])
    payload = measure(tier2_path=path, folds=2, cascade_gates=(0.35,), tier2_gates=(0.0,))
    assert payload["meta"]["rows_joined"] == 2
    point = payload["points"][0]
    model = payload["tier2_model"]
    # only escalated rows consult the measurement, so what is missing is exactly the escalated
    # rows the file does not contain
    assert point["missing_tier2_rows"] == model["rows_it_would_see"] - model["rows_of_those_measured"]
    assert point["missing_tier2_rows"] > 0
    assert "2/382" in render(payload)
