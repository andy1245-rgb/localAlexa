"""Tier-0 regex -> tier-1 router -> tier-2 abstain cascade.

Tier 0 is hand-written per-tool regex: free, brittle, catches the obvious
traffic. Anything the regex does not resolve to exactly one tool falls to the
router, which may still abstain and let the caller escalate to a real LLM. That
abstain is the point of the whole design: the expensive tier is only reached
when the cheap tiers admit they do not know.

Tier-1 predictions are injected, so the cascade can be measured against any
predictor that returns :class:`~localalexa.router.Choice` objects.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

from .data import NONE_LABEL, Registry, Row
from .router import Choice

SMART_ASSUMED_ACC = 0.95  # accuracy assumed for the tier-2 model when costing e2e
POINTS: tuple[tuple[str, float, float], ...] = (
    ("strict", 0.55, 0.50),
    ("medium", 0.45, 0.35),
    ("loose", 0.35, 0.25),
)
DEFAULT_POINT = "medium"


def pct(numerator: int, denominator: int) -> float:
    return round(100.0 * numerator / denominator, 1) if denominator else 0.0


@dataclass(frozen=True)
class CompiledPatterns:
    """Per-tool compiled regex, in registry order."""

    tool_ids: tuple[str, ...]
    patterns: tuple[tuple[str, tuple[re.Pattern[str], ...]], ...]

    @classmethod
    def from_registry(cls, registry: Registry) -> "CompiledPatterns":
        return cls(
            tool_ids=tuple(registry.tool_ids),
            patterns=tuple(
                (tool.id, tuple(re.compile(p, re.IGNORECASE) for p in tool.patterns))
                for tool in registry.tools
            ),
        )

    def probe(self, text: str) -> list[str]:
        """Tools whose patterns match, in registry (first-hit) order."""
        return [
            tool_id
            for tool_id, pats in self.patterns
            if any(p.search(text) for p in pats)
        ]


@dataclass(frozen=True)
class CascadeRow:
    """One evaluated utterance, with the tier that resolved it."""

    text: str
    true_tool: str
    family: str
    pattern_hits: tuple[str, ...]
    tier: int  # 0 = regex, 1 = router, 2 = escalated
    predicted: str | None

    @property
    def correct(self) -> bool:
        return self.predicted is not None and self.predicted == self.true_tool


def regex_tier_metrics(rows: Sequence[Row], compiled: CompiledPatterns) -> dict:
    """What the regex tier does on its own, before any router involvement."""
    hits = {i: compiled.probe(row.text) for i, row in enumerate(rows)}
    none_rows = [i for i, r in enumerate(rows) if r.tool == NONE_LABEL]
    tool_rows = [i for i, r in enumerate(rows) if r.tool != NONE_LABEL]
    covered = [i for i in hits if hits[i]]
    covered_correct = [i for i in covered if hits[i][0] == rows[i].tool]
    ambiguous = [i for i in covered if len(hits[i]) >= 2]
    none_false_positive = [i for i in none_rows if hits[i]]
    misses = [i for i in tool_rows if not hits[i]]
    miss_by_tool = Counter(rows[i].tool for i in misses)
    worst_misses = sorted(misses, key=lambda i: (-miss_by_tool[rows[i].tool], i))[:10]
    per_tool = {}
    for label in [*dict.fromkeys(r.tool for r in rows)]:
        idx = [i for i, r in enumerate(rows) if r.tool == label]
        hit = [i for i in idx if hits[i]]
        if label == NONE_LABEL:
            # "covered" means a pattern fired wrongly; "correct" is a clean rejection.
            correct = [i for i in idx if not hits[i]]
        else:
            correct = [i for i in hit if hits[i][0] == label]
        per_tool[label] = {
            "n": len(idx),
            "covered": len(hit),
            "coverage_pct": pct(len(hit), len(idx)),
            "correct": len(correct),
            "accuracy_pct": pct(len(correct), len(idx)),
        }
    return {
        "coverage_pct": pct(len(covered), len(rows)),
        "covered_rows": len(covered),
        "accuracy_on_covered_pct": pct(len(covered_correct), len(covered)),
        "ambiguous_rows": len(ambiguous),
        "ambiguous_share_of_covered_pct": pct(len(ambiguous), len(covered)),
        "none_false_positive_rows": len(none_false_positive),
        "none_false_positive_rate_pct": pct(len(none_false_positive), len(none_rows)),
        "miss_rows_tool_true": len(misses),
        "miss_counts_by_tool": dict(miss_by_tool.most_common()),
        "worst_misses": [
            {"tool": rows[i].tool, "text": rows[i].text} for i in worst_misses
        ],
        "per_tool": per_tool,
        "_hits": hits,
    }


def assign_tiers(
    rows: Sequence[Row],
    hits: dict[int, list[str]],
    choices: dict[int, Choice],
    sim_threshold: float,
    conf_threshold: float,
) -> list[CascadeRow]:
    """Resolve each row at the earliest tier that can answer it.

    Exactly one regex hit -> tier 0. Otherwise the router decides, unless its
    top similarity or entropy confidence is under threshold, in which case the
    row escalates (tier 2, no prediction, counted as an error when scoring).
    """
    out = []
    for i, row in enumerate(rows):
        row_hits = tuple(hits[i])
        if len(row_hits) == 1:
            out.append(CascadeRow(row.text, row.tool, row.family, row_hits, 0, row_hits[0]))
            continue
        choice = choices[i]
        if (
            choice.top_sim < sim_threshold
            or choice.confidence < conf_threshold
        ):
            out.append(CascadeRow(row.text, row.tool, row.family, row_hits, 2, None))
        else:
            out.append(CascadeRow(row.text, row.tool, row.family, row_hits, 1, choice.top_tool))
    return out


def cascade_metrics(
    cascade_rows: Sequence[CascadeRow], sim_threshold: float, conf_threshold: float
) -> dict:
    n = len(cascade_rows)
    by_tier = {tier: [r for r in cascade_rows if r.tier == tier] for tier in (0, 1, 2)}
    t0_correct = sum(1 for r in by_tier[0] if r.correct)
    t1_correct = sum(1 for r in by_tier[1] if r.correct)
    auto_n = len(by_tier[0]) + len(by_tier[1])
    auto_correct = t0_correct + t1_correct
    escalated = len(by_tier[2])
    return {
        "thresholds": {"top_sim_min": sim_threshold, "confidence_min": conf_threshold},
        "tier_counts": {
            "tier0_regex": len(by_tier[0]),
            "tier1_router": len(by_tier[1]),
            "tier2_escalated": escalated,
        },
        "tier_share_pct": {
            "tier0_regex": pct(len(by_tier[0]), n),
            "tier1_router": pct(len(by_tier[1]), n),
            "tier2_escalated": pct(escalated, n),
        },
        "tier_accuracy_pct": {
            "tier0_regex": pct(t0_correct, len(by_tier[0])),
            "tier1_router": pct(t1_correct, len(by_tier[1])),
        },
        "tier_correct": {"tier0_regex": t0_correct, "tier1_router": t1_correct},
        "overall_cascade_accuracy_pct": pct(auto_correct, n),
        "auto_resolved_share_pct": pct(auto_n, n),
        "auto_resolved_accuracy_pct": pct(auto_correct, auto_n),
        "escalation_rate_pct": pct(escalated, n),
        "e2e_smart95_pct": round(
            100.0 * (auto_correct + SMART_ASSUMED_ACC * escalated) / n, 1
        ),
    }


def none_class_flow(
    cascade_rows: Sequence[CascadeRow], none_label: str = NONE_LABEL
) -> dict:
    none_rows = [r for r in cascade_rows if r.true_tool == none_label]
    leaked = [r for r in none_rows if r.tier == 1 and r.predicted != none_label]
    return {
        "tier0_regex_false_positive": sum(1 for r in none_rows if r.tier == 0),
        "tier1_routed_to_none_correct": sum(
            1 for r in none_rows if r.tier == 1 and r.predicted == none_label
        ),
        "tier1_leaked_to_real_tool": len(leaked),
        "tier2_escalated": sum(1 for r in none_rows if r.tier == 2),
        "leaked_tools": dict(Counter(r.predicted for r in leaked)),
    }


def run_cascade(
    rows: Sequence[Row],
    registry: Registry,
    predict_batch: Callable[[list[str]], Sequence[Choice]],
    *,
    sim_threshold: float = 0.45,
    conf_threshold: float = 0.35,
    sweep_points: Iterable[tuple[str, float, float]] = POINTS,
) -> dict:
    """Measure the cascade at one operating point and across a threshold sweep.

    The tier-1 predictor is only called on the residue the regex did not
    uniquely resolve — the same traffic shape as production.
    """
    compiled = CompiledPatterns.from_registry(registry)
    regex = regex_tier_metrics(rows, compiled)
    hits = regex.pop("_hits")

    residue = [i for i, row in enumerate(rows) if len(hits[i]) != 1]
    predictions = list(predict_batch([rows[i].text for i in residue]))
    if len(predictions) != len(residue):
        raise ValueError(
            f"predictor returned {len(predictions)} choices for {len(residue)} rows"
        )
    choices = dict(zip(residue, predictions))

    chosen = assign_tiers(rows, hits, choices, sim_threshold, conf_threshold)
    sweep = {}
    for name, sim_t, conf_t in sweep_points:
        sweep[name] = cascade_metrics(
            assign_tiers(rows, hits, choices, sim_t, conf_t), sim_t, conf_t
        )

    router_alone_correct = sum(
        1 for i in residue if choices[i].top_tool == rows[i].tool
    )
    tier0_errors = [r for r in chosen if r.tier == 0 and not r.correct]
    tier1_errors = [r for r in chosen if r.tier == 1 and not r.correct]
    confusions = Counter((r.true_tool, r.predicted) for r in tier0_errors + tier1_errors)

    return {
        "meta": {
            "n_rows": len(rows),
            "registry_tools": len(registry.tool_ids),
            "embedding_model": registry.embedding_model,
            "rows_routed_to_router": len(residue),
            "smart_model_assumed_accuracy": SMART_ASSUMED_ACC,
        },
        "regex_tier": regex,
        "router_residue": {
            "rows": len(residue),
            "router_alone_accuracy_pct": pct(router_alone_correct, len(residue)),
        },
        "cascade_sweep": sweep,
        "cascade_operating_point": {
            **cascade_metrics(chosen, sim_threshold, conf_threshold),
            "none_class_flow": none_class_flow(chosen),
            "error_counts": {
                "tier0_wrong_hits": len(tier0_errors),
                "tier1_wrong": len(tier1_errors),
            },
            "top_confusion_pairs": [
                {"true": t, "pred": p, "count": c}
                for (t, p), c in confusions.most_common(5)
            ],
            "tier0_error_examples": [
                {"text": r.text, "true": r.true_tool, "predicted": r.predicted}
                for r in tier0_errors[:10]
            ],
            "tier1_error_examples": [
                {"text": r.text, "true": r.true_tool, "predicted": r.predicted}
                for r in tier1_errors[:10]
            ],
        },
    }
