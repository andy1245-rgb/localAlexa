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

One label per row, the strongest one that applies; anything outranked is recorded
in the row's ``detail.suppressed`` rather than thrown away, because "the model
agreed and the operator corrected it" is information about the *agreement rule*,
not about the operator.

Rows with no signal produce no label at all. That absence is the design: a log of
382 utterances with 40 corrections yields 40 negatives and nothing else until an
effect check or the top-2 rule earns a positive. The count of silent rows is
reported so the yield is never a surprise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ..data import NONE_LABEL
from .autoaccept import AutoAcceptRule, decide_all as decide_auto
from .corrections import Correction, CorrectionRule, detect
from .opinion import SecondOpinion
from .records import EffectRecord, LabelRow, RequestRecord

STRONG = "strong"
WEAK = "weak"
AUTO = "auto"

SUPPRESSED = "suppressed"


@dataclass(frozen=True)
class Derived:
    """Everything the raw log implies. Rebuildable, never authoritative."""

    reject_pile: tuple[dict[str, Any], ...] = ()
    corrections: tuple[Correction, ...] = ()
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
    """Replay the log into the reject pile, the correction list and the labels."""
    corrections = detect(records, correction_rule)
    corrected = {c.earlier_id: c for c in corrections}
    latest_effect = {e.request_id: e for e in effects}
    acceptances = {a.request_id: a for a in decide_auto(records, auto_rule, aliases)}

    opinions: dict[str, str] = {}
    if opinion is not None and records:
        predicted = opinion.labels([record.text for record in records])
        opinions = {record.id: label for record, label in zip(records, predicted)}

    labels: list[LabelRow] = []
    for record in records:
        candidates: list[tuple[int, LabelRow]] = []

        correction = corrected.get(record.id)
        if correction is not None and record.label is not None:
            candidates.append(
                (
                    50,
                    _row(record, record.label, -1, STRONG, "correction", {
                        "corrected_by": correction.later_id,
                        "delta_seconds": round(correction.delta_seconds, 3),
                        "rule": dict(correction.rule),
                        "evidence": dict(correction.evidence),
                    }),
                )
            )

        effect = latest_effect.get(record.id)
        if effect is not None and record.label is not None:
            if effect.verified:
                candidates.append(
                    (
                        30,
                        _row(record, record.label, 1, WEAK, "effect", {
                            "check": effect.check,
                            "detail": effect.detail,
                        }),
                    )
                )
            else:
                candidates.append(
                    (
                        40,
                        _row(record, record.label, -1, WEAK, "effect_failed", {
                            "check": effect.check,
                            "detail": effect.detail,
                        }),
                    )
                )

        acceptance = acceptances.get(record.id)
        if acceptance is not None:
            candidates.append(
                (
                    20,
                    _row(record, acceptance.label, 1, AUTO, f"auto_accept_{acceptance.rule}", {
                        "rule_version": acceptance.rule_version,
                        "detail": dict(acceptance.detail),
                    }),
                )
            )

        opinion_label = opinions.get(record.id)
        if (
            record.label is not None
            and opinion_label is not None
            and opinion_label == record.label
            and record.label != NONE_LABEL
        ):
            candidates.append(
                (
                    10,
                    _row(record, record.label, 1, WEAK, "opinion_agreement", {
                        "opinion": opinion_label,
                    }),
                )
            )

        if not candidates:
            continue
        candidates.sort(key=lambda pair: -pair[0])
        chosen = candidates[0][1]
        if len(candidates) > 1:
            chosen = _row(
                record,
                chosen.label,
                chosen.polarity,
                chosen.strength,
                chosen.source,
                {**dict(chosen.detail), SUPPRESSED: [c[1].source for c in candidates[1:]]},
            )
        labels.append(chosen)

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
        labels=tuple(labels),
        stats=_stats(records, labels, corrections, effects, reject_pile),
    )


def _row(
    record: RequestRecord,
    label: str,
    polarity: int,
    strength: str,
    source: str,
    detail: Mapping[str, Any],
) -> LabelRow:
    return LabelRow(
        request_id=record.id,
        ts=record.ts,
        text=record.text,
        label=label,
        polarity=polarity,
        strength=strength,
        source=source,
        detail=detail,
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
