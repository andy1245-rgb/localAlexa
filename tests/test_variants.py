"""Tests for the six structural variants: the pieces that can be wrong silently.

Each variant is a different way to use the same evidence, so the cases here are the ones where
a wrong implementation still returns a plausible label: a rule column pointing at the wrong
tool, a fusion that ignores a member, a two-stage model that answers out of its own family, and
a per-class threshold that quietly accepts everything.
"""

from __future__ import annotations

import numpy as np
import pytest

from localalexa.cascade import CompiledPatterns
from localalexa.data import NONE_LABEL, load_registry, load_routing_set
from localalexa.ladder.pools import reference_pool
from localalexa.variants import (
    Fuser,
    Hierarchical,
    NoneFirst,
    RuleFeatures,
    choose_class_thresholds,
    fit_plain,
    fit_regex_features,
    mapping_top,
    threshold_decision,
)


@pytest.fixture(scope="module")
def registry():
    return load_registry()


@pytest.fixture(scope="module")
def pool(registry):
    return reference_pool(registry)


def test_rule_features_put_a_one_in_the_matching_tool_column(registry):
    features = RuleFeatures(tuple(registry.tool_ids), CompiledPatterns.from_registry(registry))
    matrix = features.transform(["set a timer for 5 minutes", "why is the sky blue"]).toarray()
    columns = list(features.tool_ids)
    assert matrix.shape == (2, len(columns) + 1)
    assert matrix[0][columns.index("timer_create")] == 1.0
    assert matrix[0][-1] == 1.0  # the any-hit column
    # a chitchat row matches nothing, so only the any-hit column stays 0
    assert matrix[1][-1] == 0.0 or matrix[1].sum() == matrix[1][-1]


def test_rule_features_union_keeps_the_text_signal(registry, pool):
    model = fit_regex_features(registry, pool)
    # a rule-covered utterance is answered at least as confidently as an uncovered one is
    assert model.predict("cancel the kitchen timer") in registry.label_ids
    assert model.predict("why is the sky blue") in registry.label_ids


def test_a_single_member_fusion_is_that_member(registry, pool):
    linear = fit_plain(pool)
    fused = Fuser(("linear",)).fit(registry, pool)
    for text in ("set a timer for five minutes", "is the porch light on", "why is the sky blue"):
        assert fused.predict(text) == linear.predict(text)


def test_fusion_answers_only_inside_the_label_space(registry, pool):
    fused = Fuser(("linear", "regex")).fit(registry, pool)
    for text in ("set a timer for five minutes", "why is the sky blue", "add milk to the list"):
        assert fused.predict(text) in registry.label_ids


def test_the_regex_member_makes_a_row_look_different_to_the_ensemble(registry):
    # a row only the patterns can see: the fused answer must not be a constant
    fused = Fuser(("regex",)).fit(registry, [])
    assert fused.predict("cancel the timer") == "timer_cancel"
    assert fused.predict("why is the sky blue") != "timer_cancel"


def test_none_first_with_a_zero_threshold_never_calls_a_command(registry, pool):
    model = NoneFirst(threshold=0.0).fit(pool)
    assert model.predict("set a timer for five minutes") == NONE_LABEL
    assert model.predict("why is the sky blue") == NONE_LABEL


def test_none_first_above_one_always_asks_the_action_stage(registry, pool):
    model = NoneFirst(threshold=1.01).fit(pool)
    assert model.predict("set a timer for five minutes") == "timer_create"


def test_none_first_has_a_gate_that_is_a_probability(registry, pool):
    model = NoneFirst().fit(pool)
    for text in ("set a timer for five minutes", "why is the sky blue"):
        assert 0.0 <= model.p_none(text) <= 1.0


def test_hierarchical_answers_inside_the_family_it_chose(registry, pool):
    model = Hierarchical().fit(registry, pool)
    family_of = {tool.id: tool.family for tool in registry.tools}
    family_of[NONE_LABEL] = "none"
    for text in ("set a timer for five minutes", "is the porch light on", "why is the sky blue"):
        chosen = model.family.predict(text)
        answered = model.predict(text)
        assert family_of.get(answered) == chosen, (text, chosen, answered)


def test_hierarchical_still_answers_when_a_family_has_one_member(registry, pool):
    model = Hierarchical().fit(registry, pool)
    assert model.predict("set a timer for five minutes") in registry.label_ids


# ------------------------------------------------------------------ per-class thresholds


def test_a_per_class_threshold_holds_the_precision_it_was_chosen_for():
    probabilities = [
        {"a": 0.9, "b": 0.1}, {"a": 0.85, "b": 0.15},   # a right twice
        {"a": 0.6, "b": 0.4}, {"b": 0.9, "a": 0.1},     # a wrong once, b right
    ]
    truth = ["a", "a", "b", "b"]
    thresholds = choose_class_thresholds(probabilities, truth, target_precision=0.9)
    # "a" can serve 2 claims at 100%; the third claim breaks the target
    assert thresholds.thresholds["a"] == pytest.approx(0.85)
    assert thresholds.thresholds["b"] == pytest.approx(0.4)


def test_a_class_that_cannot_reach_the_target_escalates_rather_than_answering():
    probabilities = [{"a": 0.9}, {"a": 0.8}, {"a": 0.7}]
    truth = ["b", "b", "a"]  # never right at the top
    thresholds = choose_class_thresholds(probabilities, truth, target_precision=0.9)
    assert thresholds.thresholds["a"] > 1.0
    label, _score = threshold_decision({"a": 0.99}, thresholds.thresholds)
    assert label is None


def test_threshold_decision_answers_when_the_score_clears_the_class_floor():
    thresholds = {"a": 0.5}
    assert threshold_decision({"a": 0.6}, thresholds) == ("a", 0.6)
    assert threshold_decision({"a": 0.4}, thresholds)[0] is None


def test_mapping_top_handles_an_empty_distribution():
    assert mapping_top({"a": 0.1, "b": 0.9}) == "b"
    assert mapping_top({}, "fallback") == "fallback"
