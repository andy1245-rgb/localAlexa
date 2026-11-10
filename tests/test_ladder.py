"""Tests for the classifier ladder: folds, metrics, pools and the scoring loop.

These cover the parts that would silently produce a wrong ladder — a fold
assignment that leaks, a reject curve that reports coverage it does not have, an
augmentation that keeps rows it should have dropped, a correlation table that
flatters an ensemble.
"""

from __future__ import annotations

import pytest

from localalexa.data import NONE_LABEL, load_registry, load_routing_set, phrasing_families, template_index
from localalexa.ladder import metrics
from localalexa.ladder.augment_rung import paraphrase_augmenter, template_augmenter, template_fills
from localalexa.ladder.catalog import available_rungs
from localalexa.ladder.folds import build_folds
from localalexa.ladder.pools import fold_pool, reference_pool
from localalexa.ladder.predictors import SklearnPredictor
from localalexa.ladder.report import error_correlation
from localalexa.ladder.rungs import rung_0_baseline, rung_2_logreg_c100
from localalexa.ladder.run import Rung, evaluate_grouped, evaluate_rowwise, load_bench


@pytest.fixture(scope="module")
def bench():
    return load_bench(5)


def test_folds_partition_rows_and_cover_every_family(bench):
    indices = bench.folds.indices(list(bench.families))
    flat = sorted(i for fold in indices for i in fold)
    assert flat == list(range(len(bench.rows)))
    assert set(bench.folds.family_fold) == set(bench.families)
    assert sum(bench.folds.rows_per_fold) == len(bench.rows)


def test_folds_are_balanced_and_deterministic():
    families = ["a", "b", "c", "d", "e", "f", "g", "h"] * 3
    first = build_folds(families, 4)
    second = build_folds(families, 4)
    assert first.as_dict() == second.as_dict()
    sizes = first.indices(families)
    assert max(len(s) for s in sizes) - min(len(s) for s in sizes) <= 2
    for family, fold in first.family_fold.items():
        rows = [i for i, f in enumerate(families) if f == family]
        assert all(i in sizes[fold] for i in rows)


def test_folds_reject_impossible_requests():
    with pytest.raises(ValueError):
        build_folds(["a", "b"], 1)
    with pytest.raises(ValueError):
        build_folds(["a", "b"], 5)


def test_fold_pool_drops_slot_fills_of_the_held_out_family():
    registry = load_registry()
    index = template_index(registry)
    rows = load_routing_set()
    families = phrasing_families(rows, index)
    held = {families[0]}
    full = {text for text, _ in reference_pool(registry)}
    pool = fold_pool(registry, held)
    dropped = full - {text for text, _ in pool}
    # The held-out family renders some references verbatim; those have to go.
    held_texts = {
        row.text for row, family in zip(rows, families) if family in held
    }
    assert dropped <= (full & held_texts)
    assert all(text not in held_texts for text, _ in pool)
    assert pool  # the rest of the pool survives


def test_reference_pool_is_the_200_utterances():
    registry = load_registry()
    pool = reference_pool(registry)
    assert len(pool) == 200
    assert [label for _, label in pool].count(NONE_LABEL) == 16


def test_template_fills_respect_held_out_families():
    registry = load_registry()
    template = registry.tools[0].templates[0]
    everything = template_fills(registry, frozenset())
    without = template_fills(registry, frozenset({template}))
    assert len(without) < len(everything)
    excluded = {(text, label) for text, label in everything} - set(without)
    assert excluded


def test_template_augmenter_caps_per_label_and_skips_seen_rows():
    registry = load_registry()
    augment = template_augmenter(registry, cap_per_label=3)
    base = [("do the thing", "light_set")]
    extras = augment(base, frozenset())
    counts: dict[str, int] = {}
    for _, label in extras:
        counts[label] = counts.get(label, 0) + 1
    assert counts and max(counts.values()) <= 3
    assert all(text != "do the thing" for text, _ in extras)


def test_paraphrase_augmenter_drops_rows_whose_source_left_the_pool():
    rows = [
        ("turn the lamp down", "light_set", "dim the lights"),
        ("stop the timer", "timer_cancel", "cancel the timer"),
    ]
    augment = paraphrase_augmenter(rows)
    kept = augment([("dim the lights", "light_set")], frozenset())
    assert kept == [("turn the lamp down", "light_set")]


def test_reject_curve_holds_each_precision_target_separately():
    # 19 right with descending confidence, then one wrong at 0.80. Accepting 19
    # rows is 95.0% precise: enough for the 0.90 target, not for 0.975.
    confidences = [round(0.99 - 0.01 * i, 2) for i in range(19)] + [0.80]
    predicted = ["a"] * 19 + ["b"]
    truth = ["a"] * 19 + ["a"]
    curve = {p["target_precision"]: p for p in metrics.reject_curve(confidences, predicted, truth)["points"]}
    assert curve[0.975]["coverage"] == pytest.approx(0.95)
    assert curve[0.975]["precision"] == pytest.approx(1.0)
    assert curve[0.90]["coverage"] == pytest.approx(1.0)
    # The 0.975 point does hand on one row, and it is the wrong one.
    assert curve[0.975]["residue_rows"] == 1
    assert curve[0.975]["residue_accuracy"] == pytest.approx(0.0)


def test_reject_curve_never_claims_coverage_it_cannot_hold():
    # Nothing reaches 90% precision, so no point may be reported at all.
    curve = metrics.reject_curve([0.9, 0.5], ["b", "b"], ["a", "a"])
    assert all(point["coverage"] == 0.0 for point in curve["points"])


def test_reject_curve_is_perfect_when_everything_is_right():
    curve = {p["target_precision"]: p for p in metrics.reject_curve([0.9, 0.8], ["a", "b"], ["a", "b"])["points"]}
    assert curve[0.975]["coverage"] == 1.0
    assert curve[0.975]["residue_rows"] == 0


def test_expected_calibration_error_is_zero_when_confidence_is_honest():
    perfect = metrics.expected_calibration_error([0.0, 0.0, 1.0, 1.0], ["b", "b", "a", "a"], ["a", "a", "a", "a"])
    assert perfect == pytest.approx(0.0, abs=1e-9)
    lying = metrics.expected_calibration_error([1.0, 1.0, 1.0, 1.0], ["a", "b", "b", "b"], ["a", "a", "a", "a"])
    assert lying == pytest.approx(0.75)


def test_paired_delta_reports_the_spread_of_the_difference():
    delta = metrics.paired_delta([0.9, 0.8, 0.7], [0.88, 0.78, 0.72])
    assert delta["mean_points"] == pytest.approx(0.67, abs=0.02)
    assert delta["folds_better"] == 2
    assert delta["folds_worse"] == 1
    with pytest.raises(ValueError):
        metrics.paired_delta([0.9], [0.9, 0.8])


def test_error_correlation_sees_an_ensemble_that_would_be_a_waste():
    truth = ["a", "b", "c", "d"]
    same = {"m1": ["a", "b", "x", "y"], "m2": ["a", "b", "x", "y"]}
    table = error_correlation(same, truth)
    assert table[0]["disagreement_rate"] == 0.0
    assert table[0]["error_jaccard"] == pytest.approx(1.0)
    assert table[0]["only_a"] == 0 and table[0]["only_b"] == 0

    different = {"m1": ["a", "b", "x", "y"], "m2": ["x", "y", "c", "d"]}
    table = error_correlation(different, truth)
    assert table[0]["error_jaccard"] == pytest.approx(0.0)
    assert table[0]["only_a"] == 2 and table[0]["only_b"] == 2


def test_grouped_evaluation_scores_every_row_once(bench):
    rung = Rung("probe", lambda: SklearnPredictor(_pipeline))
    block = evaluate_grouped(rung, bench)
    assert len(block["predictions"]) == len(bench.rows)
    assert all(label for label in block["predictions"])
    assert set(block["predictions"]) <= set(bench.labels)
    assert len(block["per_fold_accuracy"]) == bench.folds.n_folds
    assert 0.5 < block["accuracy"] <= 1.0
    assert block["fold_spread"]["sd"] >= 0.0
    assert block["reject_curve"]["points"]


def test_rowwise_evaluation_uses_the_full_reference_pool(bench):
    rung = Rung("probe", lambda: SklearnPredictor(_pipeline))
    block = evaluate_rowwise(rung, bench)
    assert block["train_rows"] == 200
    assert block["n_rows"] == len(bench.rows)
    assert block["accuracy"] >= 0.85


def test_ladder_names_resolve_and_unknown_names_are_rejected():
    names = [rung.name for rung in available_rungs("all")]
    assert "r0-word-svc" in names
    chosen = available_rungs("r0-word-svc,r2-word-logreg-C100")
    assert [rung.name for rung in chosen] == ["r0-word-svc", "r2-word-logreg-C100"]
    with pytest.raises(SystemExit):
        available_rungs("r99-nope")


def test_linear_rungs_are_constructible_without_the_heavy_venv():
    for rung in (rung_0_baseline(), rung_2_logreg_c100()):
        assert rung.build().classes == []


def _pipeline():
    from sklearn.pipeline import make_pipeline

    from localalexa.ladder.rungs import word_vectorizer
    from sklearn.svm import LinearSVC

    return make_pipeline(word_vectorizer(), LinearSVC(C=1.0, random_state=0))
