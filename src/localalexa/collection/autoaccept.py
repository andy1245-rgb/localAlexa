"""The auto-accept rule: the only clean label that asks nothing of the operator.

Two conditions, either of which accepts a row (rule ``auto-1``):

* **same class** — the top two candidates normalise to the same label. With one
  model over one label space this never fires; it exists because the thing being
  logged may be an ensemble or a router whose candidates come from different
  label spellings, and in that case two spellings of one intent is agreement.
* **margin** — the top similarity beats the runner-up by at least the stated
  threshold. This is the rule that fires in practice, and it is worth being
  precise about what it means: it says the router was unambiguous, not that the
  router was right. That is why these labels are marked ``strength="auto"`` and
  never mixed with the verified ones.

Rows that escalated are not eligible — there is no label to accept. Rows answered
by the regex tier are not eligible either: they are deterministic, so accepting
them would be recording the pattern table back to itself.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .records import RequestRecord

RULE_VERSION = "auto-1"
DEFAULT_MARGIN = 0.05
NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalise_label(label: str, aliases: Mapping[str, str] | None = None) -> str:
    """Canonical form: lowercase, alphanumerics only, then any alias mapping."""
    key = NON_ALNUM.sub("", label.strip().lower())
    if aliases:
        return aliases.get(key, key)
    return key


@dataclass(frozen=True)
class AutoAcceptRule:
    version: str = RULE_VERSION
    margin_threshold: float = DEFAULT_MARGIN
    same_class: bool = True
    margin: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "margin_threshold": self.margin_threshold,
            "same_class": self.same_class,
            "margin": self.margin,
        }


@dataclass(frozen=True)
class Acceptance:
    request_id: str
    label: str
    rule: str
    rule_version: str
    detail: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "label": self.label,
            "rule": self.rule,
            "rule_version": self.rule_version,
            "detail": dict(self.detail),
        }


def decide(
    record: RequestRecord,
    rule: AutoAcceptRule = AutoAcceptRule(),
    aliases: Mapping[str, str] | None = None,
) -> Acceptance | None:
    """Accept this row as a label, or leave it alone."""
    if record.tier != 1 or record.label is None:
        return None
    candidates = record.candidates
    if len(candidates) < 2:
        return None

    top, second = candidates[0], candidates[1]
    if rule.same_class:
        left = normalise_label(top.label, aliases)
        right = normalise_label(second.label, aliases)
        if left == right:
            return Acceptance(
                request_id=record.id,
                label=left,
                rule="same_class",
                rule_version=rule.version,
                detail={"top": top.label, "runner_up": second.label, "form": left},
            )

    if rule.margin and record.margin >= rule.margin_threshold:
        return Acceptance(
            request_id=record.id,
            label=record.label,
            rule="margin",
            rule_version=rule.version,
            detail={
                "margin": round(float(record.margin), 6),
                "threshold": rule.margin_threshold,
                "top_sim": round(float(record.top_sim), 6),
            },
        )
    return None


def decide_all(
    records: Sequence[RequestRecord],
    rule: AutoAcceptRule = AutoAcceptRule(),
    aliases: Mapping[str, str] | None = None,
) -> list[Acceptance]:
    out = []
    for record in records:
        acceptance = decide(record, rule, aliases)
        if acceptance is not None:
            out.append(acceptance)
    return out
