"""Tests for the router: entropy confidence, softmax, decision shape, reduction."""

from __future__ import annotations

import math

import numpy as np
import pytest

from localalexa.router import (
    Choice,
    SemanticRouter,
    decide,
    entropy_confidence,
    softmax,
)


def test_entropy_confidence_uniform_is_zero():
    for k in (2, 3, 4, 18):
        assert entropy_confidence([1.0 / k] * k) == pytest.approx(0.0, abs=1e-12)


def test_entropy_confidence_one_hot_is_one():
    assert entropy_confidence([0.0, 1.0, 0.0, 0.0]) == pytest.approx(1.0)
    assert entropy_confidence([1.0, 0.0]) == pytest.approx(1.0)


def test_entropy_confidence_single_label_is_certain():
    # H(p) = 0 and ln(K) = 0 for K = 1, so the ratio has no spread to normalise.
    assert entropy_confidence([1.0]) == 1.0


def test_entropy_confidence_hand_computed():
    # H([0.5, 0.25, 0.25]) = 1.0397207..., ln 3 = 1.0986122...
    expected = 1.0 - 1.0397207708399179 / 1.0986122886681098
    assert entropy_confidence([0.5, 0.25, 0.25]) == pytest.approx(expected, abs=1e-12)
    assert expected == pytest.approx(0.053605, abs=1e-6)


def test_entropy_confidence_is_order_invariant():
    a = entropy_confidence([0.7, 0.2, 0.1])
    b = entropy_confidence([0.1, 0.7, 0.2])
    assert a == pytest.approx(b)


def test_entropy_confidence_increases_as_distribution_sharpens():
    flat = entropy_confidence([0.34, 0.33, 0.33])
    mid = entropy_confidence([0.6, 0.25, 0.15])
    sharp = entropy_confidence([0.94, 0.04, 0.02])
    assert flat < mid < sharp


def test_entropy_confidence_normalises_over_the_full_label_count():
    # Entropy ln 2, spread over K = 3 labels, not over the 2-label support.
    expected = 1.0 - math.log(2) / math.log(3)
    assert entropy_confidence([0.5, 0.5, 0.0]) == pytest.approx(expected, abs=1e-12)
    assert entropy_confidence([1.0, 0.0, 0.0]) == pytest.approx(1.0)


def test_softmax_normalises_and_preserves_order():
    probs = softmax([0.2, 0.5, 0.1], tau=0.08)
    assert probs.sum() == pytest.approx(1.0)
    assert np.argmax(probs) == 1
    assert probs[1] > probs[0] > probs[2]


def test_softmax_of_equal_similarities_is_uniform():
    probs = softmax([0.4, 0.4, 0.4, 0.4], tau=0.08)
    assert probs == pytest.approx([0.25] * 4)


def test_softmax_is_shift_invariant():
    base = softmax([0.1, 0.2, 0.3], tau=0.08)
    shifted = softmax([10.1, 10.2, 10.3], tau=0.08)
    assert shifted == pytest.approx(base)


def test_softmax_temperature_controls_sharpness():
    sims = [0.5, 0.45, 0.44]
    sharp = softmax(sims, tau=0.02)
    loose = softmax(sims, tau=0.5)
    assert sharp.max() > loose.max()
    assert loose.min() > sharp.min()


def test_softmax_rejects_non_positive_temperature():
    with pytest.raises(ValueError):
        softmax([0.1, 0.2], tau=0.0)
    with pytest.raises(ValueError):
        softmax([0.1, 0.2], tau=-1.0)


def test_decide_reports_top_runner_up_and_margin():
    ids = ["timer_create", "reminder_create", "light_set"]
    choice = decide([0.62, 0.58, 0.10], ids, tau=0.08, low_sim_floor=0.35)
    assert isinstance(choice, Choice)
    assert choice.top_tool == "timer_create"
    assert choice.runner_up == "reminder_create"
    assert choice.top_sim == pytest.approx(0.62)
    assert choice.runner_up_sim == pytest.approx(0.58)
    assert choice.margin == pytest.approx(0.04)
    assert set(choice.probabilities) == set(ids)
    assert sum(choice.probabilities.values()) == pytest.approx(1.0)


def test_decide_low_similarity_flag_tracks_the_floor():
    ids = ["a", "b"]
    assert decide([0.34, 0.1], ids, low_sim_floor=0.35).low_sim is True
    assert decide([0.35, 0.1], ids, low_sim_floor=0.35).low_sim is False


def test_decide_confidence_matches_the_formula():
    ids = ["a", "b", "c"]
    choice = decide([0.5, 0.3, 0.1], ids, tau=0.08)
    expected = entropy_confidence(softmax([0.5, 0.3, 0.1], 0.08))
    assert choice.confidence == pytest.approx(expected)


def test_decide_temperature_moves_confidence_not_the_argmax():
    ids = ["a", "b", "c"]
    sims = [0.5, 0.3, 0.1]
    choices = [decide(sims, ids, tau=t) for t in (0.05, 0.08, 0.12)]
    assert {c.top_tool for c in choices} == {"a"}
    assert choices[0].confidence > choices[1].confidence > choices[2].confidence


def test_decide_rejects_length_mismatch():
    with pytest.raises(ValueError):
        decide([0.1, 0.2, 0.3], ["a", "b"])


def test_choice_as_dict_round_trips():
    choice = decide([0.6, 0.2], ["a", "b"])
    payload = choice.as_dict()
    assert payload["top_tool"] == "a"
    assert payload["probabilities"] == choice.probabilities
    assert payload["confidence"] == choice.confidence


class WordCountEncoder:
    """Deterministic stand-in for MiniLM: L2-normalised bag of words."""

    def __init__(self, vocab: list[str]) -> None:
        self.vocab = vocab

    def encode(self, texts):
        matrix = np.zeros((len(texts), len(self.vocab)))
        for row, text in enumerate(texts):
            for token in text.lower().split():
                if token in self.vocab:
                    matrix[row, self.vocab.index(token)] += 1.0
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms


def _tiny_registry():
    from localalexa.data import Registry, Tool

    def tool(tool_id, references):
        return Tool(
            id=tool_id,
            family="tiny",
            cls="1tool",
            entity_domain="test",
            commit="immediate",
            patterns=(),
            references=tuple(references),
            templates=(),
        )

    return Registry(
        embedding_model="fake",
        seed=0,
        tools=(
            tool("timer_create", ["set a timer for three minutes", "start a timer"]),
            tool("light_set", ["dim the lights", "turn the lights down"]),
            tool("shoppinglist_add", ["add milk to the shopping list"]),
        ),
        none_references=("what is the tallest mountain",),
    )


def test_semantic_router_reduces_max_over_references_per_tool():
    registry = _tiny_registry()
    vocab = sorted(
        {
            token
            for ref in [r for t in registry.tools for r in t.references]
            + list(registry.none_references)
            for token in ref.split()
        }
    )
    router = SemanticRouter(registry, encoder=WordCountEncoder(vocab))

    choice = router.predict("set a timer for three minutes")
    assert choice.top_tool == "timer_create"
    assert choice.top_sim == pytest.approx(1.0, abs=1e-9)
    assert choice.low_sim is False
    assert router.predict("dim the lights").top_tool == "light_set"
    assert router.predict("what is the tallest mountain").top_tool == "__none__"


def test_semantic_router_batch_matches_single_predictions():
    registry = _tiny_registry()
    vocab = sorted(
        {
            token
            for ref in [r for t in registry.tools for r in t.references]
            + list(registry.none_references)
            for token in ref.split()
        }
    )
    router = SemanticRouter(registry, encoder=WordCountEncoder(vocab))
    texts = ["set a timer", "dim the lights", "add milk to the shopping list"]
    batch = router.predict_batch(texts)
    assert [c.top_tool for c in batch] == [router.predict(t).top_tool for t in texts]
    assert [c.confidence for c in batch] == pytest.approx(
        [router.predict(t).confidence for t in texts]
    )


def test_semantic_router_label_set_is_every_tool_plus_none():
    registry = _tiny_registry()
    router = SemanticRouter(registry, encoder=WordCountEncoder(["timer"]))
    assert router.tool_ids == [
        "timer_create",
        "light_set",
        "shoppinglist_add",
        "__none__",
    ]
    choice = router.predict("timer")
    assert len(choice.probabilities) == 4


def test_decision_confidence_sits_strictly_between_the_extremes():
    choice = decide([0.6, 0.5, 0.2], ["a", "b", "c"], tau=0.08)
    uniform = entropy_confidence([1 / 3, 1 / 3, 1 / 3])
    certain = entropy_confidence([1.0, 0.0, 0.0])
    assert uniform < choice.confidence < certain
