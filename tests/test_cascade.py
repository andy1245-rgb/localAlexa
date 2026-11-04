"""Tests for the cascade: regex tier, tier assignment, metrics, escalation."""

from __future__ import annotations

import pytest

from localalexa.cascade import (
    POINTS,
    CascadeRow,
    CompiledPatterns,
    assign_tiers,
    cascade_metrics,
    none_class_flow,
    regex_tier_metrics,
    run_cascade,
)
from localalexa.data import NONE_LABEL, Registry, Row, Tool
from localalexa.router import Choice


def choice(top_tool: str, *, top_sim: float = 0.6, confidence: float = 0.7) -> Choice:
    return Choice(
        top_tool=top_tool,
        top_sim=top_sim,
        runner_up="other",
        runner_up_sim=top_sim - 0.05,
        probabilities={top_tool: confidence},
        confidence=confidence,
        margin=0.05,
        low_sim=top_sim < 0.35,
    )


def _registry() -> Registry:
    def tool(tool_id, patterns, references):
        return Tool(
            id=tool_id,
            family="tiny",
            cls="1tool",
            entity_domain="test",
            commit="immediate",
            patterns=tuple(patterns),
            references=tuple(references),
            templates=(),
        )

    return Registry(
        embedding_model="fake",
        seed=0,
        tools=(
            tool("timer_create", [r"\btimer\b", r"\bcountdown\b"], ["set a timer"]),
            tool("light_set", [r"\blights?\b", r"\bdim\b"], ["dim the lights"]),
            tool("shoppinglist_add", [r"\bshopping list\b"], ["add milk to the list"]),
        ),
        none_references=("what is the tallest mountain",),
    )


def test_probe_returns_tools_in_registry_order():
    compiled = CompiledPatterns.from_registry(_registry())
    assert compiled.probe("set a timer for 3 minutes") == ["timer_create"]
    assert compiled.probe("dim the lights") == ["light_set"]
    both = compiled.probe("set a timer and dim the lights")
    assert both == ["timer_create", "light_set"]


def test_probe_is_case_insensitive_and_returns_nothing_on_chitchat():
    compiled = CompiledPatterns.from_registry(_registry())
    assert compiled.probe("Start a TIMER") == ["timer_create"]
    assert compiled.probe("what is the tallest mountain") == []


def test_regex_metrics_count_coverage_ambiguity_and_misses():
    registry = _registry()
    rows = [
        Row("set a timer", "timer_create", "timers"),
        Row("start a timer and dim the lights", "timer_create", "timers"),
        Row("add milk to the list", "shoppinglist_add", "shopping"),
        Row("what is the tallest mountain", NONE_LABEL, "none"),
    ]
    metrics = regex_tier_metrics(rows, CompiledPatterns.from_registry(registry))
    assert metrics["covered_rows"] == 2  # the ambiguous row is the second hit
    assert metrics["coverage_pct"] == 50.0
    assert metrics["ambiguous_rows"] == 1
    assert metrics["accuracy_on_covered_pct"] == 100.0
    assert metrics["miss_rows_tool_true"] == 1
    assert metrics["none_false_positive_rows"] == 0
    metrics.pop("_hits")


def test_single_hit_resolves_at_tier_zero_without_asking_the_router():
    rows = [Row("set a timer", "timer_create", "timers")]
    hits = {0: ["timer_create"]}
    assigned = assign_tiers(rows, hits, {}, sim_threshold=0.45, conf_threshold=0.35)
    assert assigned[0].tier == 0
    assert assigned[0].predicted == "timer_create"
    assert assigned[0].correct is True


def test_confident_router_answer_resolves_at_tier_one():
    rows = [Row("start a timer and dim the lights", "timer_create", "timers")]
    hits = {0: ["timer_create", "light_set"]}
    assigned = assign_tiers(
        rows, hits, {0: choice("timer_create")}, sim_threshold=0.45, conf_threshold=0.35
    )
    assert assigned[0].tier == 1
    assert assigned[0].predicted == "timer_create"


@pytest.mark.parametrize(
    "bad_choice",
    [choice("light_set", top_sim=0.44), choice("light_set", confidence=0.34)],
)
def test_low_similarity_or_low_confidence_escalates(bad_choice):
    rows = [Row("start a timer and dim the lights", "timer_create", "timers")]
    hits = {0: ["timer_create", "light_set"]}
    assigned = assign_tiers(
        rows, hits, {0: bad_choice}, sim_threshold=0.45, conf_threshold=0.35
    )
    assert assigned[0].tier == 2
    assert assigned[0].predicted is None
    assert assigned[0].correct is False


def test_metrics_count_escalations_as_errors():
    rows = [
        CascadeRow("a", "timer_create", "timers", ("timer_create",), 0, "timer_create"),
        CascadeRow("b", "light_set", "lights", ("light_set", "dim"), 1, "light_set"),
        CascadeRow("c", "timer_create", "timers", (), 2, None),
        CascadeRow("d", "shoppinglist_add", "shopping", ("shopping",), 0, "timer_create"),
    ]
    metrics = cascade_metrics(rows, sim_threshold=0.45, conf_threshold=0.35)
    assert metrics["tier_counts"] == {
        "tier0_regex": 2,
        "tier1_router": 1,
        "tier2_escalated": 1,
    }
    assert metrics["tier_accuracy_pct"]["tier0_regex"] == 50.0
    assert metrics["tier_accuracy_pct"]["tier1_router"] == 100.0
    assert metrics["overall_cascade_accuracy_pct"] == 50.0
    assert metrics["auto_resolved_share_pct"] == 75.0
    assert metrics["escalation_rate_pct"] == 25.0
    # two of the three auto-resolved were right, one escalation at 95%
    assert metrics["e2e_smart95_pct"] == 73.8


def test_none_class_flow_separates_leaks_from_escalations():
    rows = [
        CascadeRow("a", NONE_LABEL, "none", ("lights",), 0, "lights"),
        CascadeRow("b", NONE_LABEL, "none", (), 1, NONE_LABEL),
        CascadeRow("c", NONE_LABEL, "none", (), 1, "timer_create"),
        CascadeRow("d", NONE_LABEL, "none", (), 2, None),
    ]
    flow = none_class_flow(rows)
    assert flow["tier0_regex_false_positive"] == 1
    assert flow["tier1_routed_to_none_correct"] == 1
    assert flow["tier1_leaked_to_real_tool"] == 1
    assert flow["tier2_escalated"] == 1
    assert flow["leaked_tools"] == {"timer_create": 1}


def test_run_cascade_only_asks_the_router_about_the_residue():
    registry = _registry()
    rows = [
        Row("set a timer", "timer_create", "timers"),
        Row("start a timer and dim the lights", "timer_create", "timers"),
        Row("what is the tallest mountain", NONE_LABEL, "none"),
    ]
    asked: list[list[str]] = []

    def predict_batch(texts: list[str]) -> list[Choice]:
        asked.append(list(texts))
        return [choice("timer_create") for _ in texts]

    payload = run_cascade(rows, registry, predict_batch)
    assert asked == [["start a timer and dim the lights", "what is the tallest mountain"]]
    assert payload["meta"]["rows_routed_to_router"] == 2
    assert payload["meta"]["n_rows"] == 3
    assert payload["regex_tier"]["covered_rows"] == 2
    point = payload["cascade_operating_point"]
    assert point["tier_counts"]["tier0_regex"] == 1
    assert point["tier_counts"]["tier1_router"] == 2
    assert payload["router_residue"]["router_alone_accuracy_pct"] == 50.0
    assert set(payload["cascade_sweep"]) == {name for name, _, _ in POINTS}
    assert "_hits" not in payload["regex_tier"]


def test_run_cascade_rejects_a_short_prediction_batch():
    registry = _registry()
    rows = [Row("nothing matches here", "timer_create", "timers")]
    with pytest.raises(ValueError):
        run_cascade(rows, registry, lambda texts: [])


def test_regex_tier_reproduces_the_recorded_experiment():
    """The adapted tier 0 must still match results/historical/cascade_results.json."""
    import json
    from pathlib import Path

    from localalexa.data import load_registry, load_routing_set

    recorded = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "results"
            / "historical"
            / "cascade_results.json"
        ).read_text(encoding="utf-8")
    )["regex_tier"]
    measured = regex_tier_metrics(
        load_routing_set(), CompiledPatterns.from_registry(load_registry())
    )
    measured.pop("_hits")
    assert measured == recorded
