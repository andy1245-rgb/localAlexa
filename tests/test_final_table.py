"""Tests for the final table: the arithmetic, the joins, and the split that had a leak.

The table is a report, so the cases that matter are the invariants a wrong report would break
quietly: every row accounted for exactly once, the end-to-end figure being a function of the
supplied tier-2 accuracy, and the strict split not training on the utterances it scores.
"""

from __future__ import annotations

import json

import pytest

from localalexa.data import load_registry, load_routing_set
from localalexa.final_table import (
    LATENCY_MS,
    SPLITS,
    _pool_and_rows,
    _families,
    build,
    load_tier2,
    measure_split,
    render,
)
from localalexa.ladder.pools import family_texts


@pytest.fixture(scope="module")
def registry():
    return load_registry()


@pytest.fixture(scope="module")
def rows():
    return load_routing_set()


def test_every_row_is_accounted_for_exactly_once(registry, rows):
    families = _families(registry, rows)
    for split in SPLITS:
        block = measure_split(split, registry, rows, families, folds=2)
        cascade = block["cascade"]
        assert block["n"] == 382
        assert cascade["tier0_rows"] + cascade["tier1_rows"] + cascade["escalated"] == 382
        # every answered row is either right or wrong; every unanswered row is escalated
        answered = 382 - cascade["escalated"]
        assert 0 <= cascade["auto_correct"] <= answered
        assert cascade["auto_accuracy"] == pytest.approx(
            cascade["auto_correct"] / answered, abs=1e-4)
        # coverage is the share answered, not the share correct
        assert cascade["coverage"] == pytest.approx((382 - cascade["escalated"]) / 382, abs=1e-4)


def test_the_end_to_end_figure_is_a_function_of_the_supplied_tier2_accuracy(registry, rows):
    families = _families(registry, rows)
    low = measure_split("family", registry, rows, families, folds=2, tier2_accuracy=0.80)
    high = measure_split("family", registry, rows, families, folds=2, tier2_accuracy=0.95)
    auto = low["cascade"]["auto_correct"]
    escalated = low["cascade"]["escalated"]
    assert low["cascade"]["e2e_at_measured_tier2"] == pytest.approx(
        (auto + 0.80 * escalated) / 382, abs=1e-4
    )
    assert high["cascade"]["e2e_at_measured_tier2"] == pytest.approx(
        (auto + 0.95 * escalated) / 382, abs=1e-4
    )
    assert high["cascade"]["e2e_at_measured_tier2"] > low["cascade"]["e2e_at_measured_tier2"]


def test_without_a_measurement_the_table_says_so_rather_than_assuming_one(registry, rows):
    families = _families(registry, rows)
    block = measure_split("rowwise", registry, rows, families, folds=2)
    assert block["tier2_alone"]["measured"] is False
    assert block["tier2_alone"]["accuracy"] is None
    assert block["cascade"]["e2e_at_measured_tier2"] is None
    assert block["cascade"]["breakeven_vs_tier2_alone"] is None


def test_the_strict_split_never_trains_on_an_utterance_it_scores(registry, rows):
    families = _families(registry, rows)
    for pool, indices in _pool_and_rows("strict", registry, rows, families, 2):
        pool_texts = {text for text, _ in pool}
        leaked = [rows[i].text for i in indices if rows[i].text in pool_texts]
        # template-identity filtering alone leaks hundreds of rows here; the split filters by
        # rendered text, so nothing scored is also trained on
        assert leaked == []


def test_the_family_split_also_scores_nothing_it_trained_on(registry, rows):
    families = _families(registry, rows)
    for pool, indices in _pool_and_rows("family", registry, rows, families, 2):
        pool_texts = {text for text, _ in pool}
        assert [rows[i].text for i in indices if rows[i].text in pool_texts] == []


def test_the_strict_pool_is_smaller_than_the_unfiltered_one(registry, rows):
    from localalexa.evaluate import _strict_training_pool

    families = _families(registry, rows)
    plan = _pool_and_rows("strict", registry, rows, families, 2)
    assert len(plan) == 2  # one pool per fold
    for pool, indices in plan:
        held = {families[i] for i in indices}
        unfiltered = _strict_training_pool(registry, held)
        assert len(pool) < len(unfiltered)
        assert pool and all(text not in family_texts(registry, held) for text, _ in pool)


def test_rowwise_trains_on_the_two_hundred_references_and_scores_everything(registry, rows):
    families = _families(registry, rows)
    plan = _pool_and_rows("rowwise", registry, rows, families, 5)
    assert len(plan) == 1  # one fit for the whole corpus
    pool, indices = plan[0]
    assert len(pool) == 200
    assert indices == list(range(382))


def test_the_break_even_is_reported_against_the_tier2_alone_bar(registry, rows):
    families = _families(registry, rows)
    block = measure_split("family", registry, rows, families, folds=2, tier2_accuracy=0.95)
    breakeven = block["cascade"]["breakeven_vs_tier2_alone"]
    tier2_correct = round(0.95 * 382)
    auto = block["cascade"]["auto_correct"]
    escalated = block["cascade"]["escalated"]
    assert breakeven["tier2_accuracy_needed"] == pytest.approx(
        round((tier2_correct - auto) / escalated, 4)
    )


def test_loading_a_measurement_file_keys_by_split(tmp_path):
    path = tmp_path / "tier2.json"
    path.write_text(json.dumps({"splits": {
        "rowwise": {"accuracy": 0.9686},
        "family": {"accuracy": 0.9503},
        "strict": {"accuracy": None},
    }}), encoding="utf-8")
    loaded = load_tier2(path)
    assert loaded == {"rowwise": 0.9686, "family": 0.9503}


def test_the_built_table_renders_every_split_and_quotes_latency(registry):
    payload = build(folds=2, splits=("rowwise",), registry=registry,
                    tier2={"rowwise": 0.9686})
    text = render(payload)
    assert "rowwise" in text
    assert f"{LATENCY_MS['tier1']:.2f}" in text
    assert "end to end" in text
    assert "96.86%" in text or "96.9" in text
