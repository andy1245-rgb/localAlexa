"""Signals in, labels out — and the one signal that is deliberately missing.

The table this module implements, strongest first:

===================  =========  ==================================================
signal               label      how it is recognised
===================  =========  ==================================================
correction           strong −   the heuristic in ``corrections.py``: an action
effect check fails   weak −     the operator's tooling looked and found nothing
effect check passes  weak +     the timer exists, the list changed
auto-accept          auto +     top two agree, or margin over the threshold
second opinion       weak +     a different architecture says the same label
*no correction*      **none**   not a signal. Inattention is not satisfaction
===================  =========  ==================================================

Two shapes of the same thing:

``events``
    every signal that fired for a row, appendable and idempotent — what the live
    collector journals, one line each, as they happen.

``reduced``
    one label per request, strongest wins, which is what a training set wants.

Keeping both is not redundancy. "Auto-accepted by margin, then corrected by the
operator" is the audit trail for the auto-accept rule, and a reduction that
silently dropped the superseded row would destroy the only evidence of whether
that rule is safe. The reducer keeps the winner and records the losers in
``detail.suppressed``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from ..data import NONE_LABEL
from .autoaccept import AutoAcceptRule, decide_all as decide_auto
from .corrections import Correction, CorrectionRule, detect
from .opinion import SecondOpinion
from .records import EffectRecord, LabelRow, RequestRecord

STRONG = "strong"
WEAK = "weak"
AUTO = "auto"

SOURCE_PRIORITY: dict[str, int] = {
    "correction": 50,
    "effect_failed": 40,
    "effect": 30,
    "auto_accept_same_class": 20,
    "auto_accept_margin": 20,
    "opinion_agreement": 10,
}


def priority(source: str) -> int:
    """How strongly a signal outranks the others for the same row."""
    return SOURCE_PRIORITY.get(source, 0)


def label_for(
    record: RequestRecord,
    label: str | None,
    polarity: int,
    strength: str,
    source: str,
    detail: Mapping[str, Any],
) -> LabelRow:
    """One label event. The single place a signal becomes a row."""
    return LabelRow(
        request_id=record.id,
        ts=record.ts,
        text=record.text,
        label=label or "",
        polarity=polarity,
        strength=strength,
        source=source,
        detail=detail,
    )


def reduce_labels(events: Iterable[LabelRow]) -> list[LabelRow]:
    """One label per request: the strongest, with the losers recorded."""
    grouped: dict[str, list[LabelRow]] = {}
    order: list[str] = []
    for event in events:
        if event.request_id not in grouped:
            grouped[event.request_id] = []
            order.append(event.request_id)
        key = (event.request_id, event.source)
        if all((existing.request_id, existing.source) != key for existing in grouped[event.request_id]):
            grouped[event.request_id].append(event)

    reduced: list[LabelRow] = []
    for request_id in order:
        ranked = sorted(grouped[request_id], key=lambda row: -priority(row.source))
        winner = ranked[0]
        if len(ranked) == 1:
            reduced.append(winner)
            continue
        reduced.append(
            LabelRow(
                request_id=winner.request_id,
                ts=winner.ts,
                text=winner.text,
                label=winner.label,
                polarity=winner.polarity,
                strength=winner.strength,
                source=winner.source,
                detail={**dict(winner.detail), "suppressed": [row.source for row in ranked[1:]]},
            )
        )
    return reduced


@dataclass(frozen=True)
class Derived:
    """Everything the raw log implies. Rebuildable, never authoritative."""

    reject_pile: tuple[dict[str, Any], ...] = ()
    corrections: tuple[Correction, ...] = ()
    events: tuple[LabelRow, ...] = ()
    labels: tuple[LabelRow, ...] = ()
    stats: Mapping[str, Any] = field(default_factory=dict)

    def by_id(self) -> dict[str, LabelRow]:
        return {row.request_id: row for row in self.labels}


def derive(
    records: Sequence[RequestRecord],
    effects: Sequence[EffectRecord] = (),
    *,
    correction_rule: CorrectionRule = CorrectionRule(),
    auto_rule: AutoAcceptRule = AutoAcceptRule(),
    aliases: Mapping[str, str] | None = None,
    opinion: SecondOpinion | None = None,
) -> Derived:
    """Replay the log into the reject pile, the corrections and the labels."""
    corrections = detect(records, correction_rule)
    corrected = {c.earlier_id: c for c in corrections}
    latest_effect = {e.request_id: e for e in effects}
    acceptances = {a.request_id: a for a in decide_auto(records, auto_rule, aliases)}

    opinions: dict[str, str] = {}
    if opinion is not None and records:
        predicted = opinion.labels([record.text for record in records])
        opinions = {record.id: label for record, label in zip(records, predicted)}

    events: list[LabelRow] = []
    for record in records:
        correction = corrected.get(record.id)
        if correction is not None and record.label is not None:
            events.append(
                label_for(record, record.label, -1, STRONG, "correction", {
                    "corrected_by": correction.later_id,
                    "delta_seconds": round(correction.delta_seconds, 3),
                    "rule": dict(correction.rule),
                    "evidence": dict(correction.evidence),
                })
            )

        effect = latest_effect.get(record.id)
        if effect is not None and record.label is not None:
            source = "effect" if effect.verified else "effect_failed"
            events.append(
                label_for(record, record.label, 1 if effect.verified else -1, WEAK, source, {
                    "check": effect.check,
                    "detail": effect.detail,
                })
            )

        acceptance = acceptances.get(record.id)
        if acceptance is not None:
            events.append(
                label_for(record, acceptance.label, 1, AUTO, f"auto_accept_{acceptance.rule}", {
                    "rule_version": acceptance.rule_version,
                    "detail": dict(acceptance.detail),
                })
            )

        opinion_label = opinions.get(record.id)
        if (
            record.label is not None
            and opinion_label is not None
            and opinion_label == record.label
            and record.label != NONE_LABEL
        ):
            events.append(
                label_for(record, record.label, 1, WEAK, "opinion_agreement", {
                    "opinion": opinion_label,
                })
            )

    labels = reduce_labels(events)
    reject_pile = tuple(
        {
            "id": record.id,
            "ts": round(record.ts, 6),
            "at": record.at,
            "session": record.session,
            "text": record.text,
            "confidence": round(record.confidence, 6),
            "top_sim": round(record.top_sim, 6),
            "margin": round(record.margin, 6),
            "candidates": [c.as_dict() for c in record.candidates],
        }
        for record in records
        if record.gate
    )

    return Derived(
        reject_pile=reject_pile,
        corrections=tuple(corrections),
        events=tuple(events),
        labels=tuple(labels),
        stats=_stats(records, labels, corrections, effects, reject_pile),
    )


def _stats(records, labels, corrections, effects, reject_pile) -> dict[str, Any]:
    by_source: dict[str, int] = {}
    for row in labels:
        by_source[row.source] = by_source.get(row.source, 0) + 1
    positives = sum(1 for row in labels if row.polarity > 0)
    negatives = sum(1 for row in labels if row.polarity < 0)
    return {
        "rows": len(records),
        "rows_below_gate": len(reject_pile),
        "corrections": len(corrections),
        "effects_checked": len(effects),
        "labels": len(labels),
        "labels_positive": positives,
        "labels_negative": negatives,
        "rows_without_signal": len(records) - len(labels),
        "labels_by_source": dict(sorted(by_source.items())),
    }
