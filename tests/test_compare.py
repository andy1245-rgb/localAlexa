"""Tests for the grouped cascade-vs-classifier comparison.

The comparison is mostly arithmetic over a set of per-row scores, so the cases here are
hand-built rows with known answers: the regex tier is right, the model is confidently
wrong, the gate lets something through, and the break-even has to come out solved rather
than assumed.
"""

from __future__ import annotations

import pytest

from localalexa.compare import RowScore, _operating_point, compare, render
from localalexa.ladder.folds import DEFAULT_FOLDS

GRID = (0.90, 0.95, 1.00)


def score(truth: str, model: str, confidence: float, regex: str | None = None,
          similarity: float | None = None) -> RowScore:
    return RowScore(
        fold=0,
        truth=truth,
        regex_label=regex,
        model_label=model,
        model_confidence=confidence,
        model_similarity=confidence if similarity is None else similarity,
    )


def test_the_regex_row_is_answered_by_the_regex_and_not_by_the_model():
    rows = [score("light_set", "light_query", 0.99, regex="light_set")]
    point = _operating_point(rows, gate=0.5, grid=GRID, one_correct=0)
    assert point["tier0"]["rows"] == 1
    assert point["tier0"]["correct"] == 1
    assert point["tier1"]["rows"] == 0
    assert point["coverage"] == 1.0


def test_a_confident_row_is_answered_by_the_model():
    rows = [score("timer_create", "timer_create", 0.90)]
    point = _operating_point(rows, gate=0.5, grid=GRID, one_correct=1)
    assert point["tier1"]["rows"] == 1
    assert point["tier1"]["correct"] == 1
    assert point["escalated"] == 0


def test_a_row_under_the_gate_escalates_and_is_excluded_from_auto_accuracy():
    rows = [
        score("timer_create", "timer_create", 0.90),
        score("light_set", "light_query", 0.10),
    ]
    point = _operating_point(rows, gate=0.5, grid=GRID, one_correct=1)
    assert point["escalated"] == 1
    assert point["coverage"] == 0.5
    assert point["auto_correct"] == 1
    assert point["auto_accuracy"] == 1.0


def test_the_end_to_end_curve_is_linear_in_the_measured_tier2_accuracy():
    rows = [
        score("a", "a", 0.9),
        score("b", "b", 0.9),
        score("c", "x", 0.1),  # escalated; the model would have been wrong anyway
        score("d", "x", 0.1),  # escalated
    ]
    point = _operating_point(rows, gate=0.5, grid=GRID, one_correct=2)
    # two auto-answered and right, two escalated: e2e(x) = (2 + 2x) / 4
    assert point["e2e_curve_pct"]["0.90"] == 95.0
    assert point["e2e_curve_pct"]["1.00"] == 100.0
    # matching the single classifier's 2/4 needs nothing from tier 2 at all
    assert point["breakeven_vs_one_classifier"]["tier2_accuracy_needed"] == 0.0


def test_a_confidently_wrong_row_shows_up_in_the_oracle_ceiling():
    rows = [
        score("a", "a", 0.9),
        score("b", "x", 0.9),  # auto-answered and wrong
        score("c", "x", 0.1),  # escalated
    ]
    point = _operating_point(rows, gate=0.5, grid=GRID, one_correct=1)
    # one of the two auto-answered is wrong; an oracle escalating it gives (2 + x)/3
    assert point["oracle_gate_e2e_pct"]["1.00"] == 100.0
    assert point["oracle_gate_e2e_pct"]["0.90"] == pytest.approx(96.7, abs=0.1)


def test_the_similarity_floor_only_applies_to_the_encoder_router():
    rows = [score("a", "a", confidence=0.99, similarity=0.10)]
    without = _operating_point(rows, gate=0.5, grid=GRID, one_correct=1)
    with_floor = _operating_point(rows, gate=0.5, grid=GRID, one_correct=1,
                                  use_sim_floor=True, sim_floor=0.45)
    assert without["escalated"] == 0
    assert with_floor["escalated"] == 1


def test_the_comparison_runs_grouped_and_reports_both_arms():
    payload = compare(folds=2, tier1_kind="logreg", gates=(0.35, 0.70), tier2_grid=GRID)
    assert payload["one_classifier"]["n"] == 382
    assert payload["meta"]["folds"] == 2
    assert payload["production_point"]["gate"] == 0.35
    assert len(payload["gate_sweep"]) == 2
    assert set(payload["best_gate_per_tier2_accuracy"]) == {"0.90", "0.95", "1.00"}
    # the leak is reported next to the honest number, not instead of it
    assert payload["leaky_reference"]["accuracy"] > payload["production_point"]["tier1"]["accuracy"]
    assert "grouped" in render(payload)


def test_the_comparison_reports_the_tier_accuracies_it_measured():
    payload = compare(folds=2, tier1_kind="logreg", gates=(0.35,), tier2_grid=(0.95,))
    tier0 = payload["production_point"]["tier0"]
    tier1 = payload["production_point"]["tier1"]
    assert tier0["rows"] + tier1["rows"] + payload["production_point"]["escalated"] == 382
    assert 0.0 <= tier0["accuracy"] <= 1.0
    assert 0.0 <= tier1["accuracy"] <= 1.0
