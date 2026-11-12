"""The correction heuristic, and the fact that it is one.

A correction is the only negative in this design that needs nothing from the
model, because it is an *action*: the operator heard the answer, noticed it was
wrong, and said something else. That is why it is strong. It is also why it is
inferential — "something else, shortly after, about the same thing" is my reading
of an event I cannot see, so every correction this module produces carries the
rule version, the window, the branch that fired and the tokens that justified it.

The heuristic will be revised. Detection therefore reads the raw log and writes a
derived file; nothing is overwritten, and old rows stay readable under the rule
that produced them.

Four conditions, all required (rule ``corr-1``):

1. same session as the earlier row
2. the earlier row is the most recent *acted-on* row of that session (tier 0 or
   1 — an escalated row produced no answer to correct)
3. ``0 < Δt ≤ window`` (default 45 s) and the two texts are not identical; an
   identical repeat is a repeat, not a correction
4. one of three branches fires:
   a. *cue* — the later text contains a correction cue ("no", "actually", "I
      meant", …)
   b. *same operation* — the two rows carry the same label and share at least
      one content token, i.e. the operator amended parameters of the same intent
   c. *overlap* — token Jaccard at or above the floor, whatever the labels say

Branch (b) is what makes the heuristic useful with a working router: "a timer for
five minutes" followed by "a timer for ten minutes" is a correction without a cue.
Branch (a) is what catches the operator who does say "no". Branch (c) catches the
rephrasing that changes both the label and most of the words.

Limits, stated because they will bite: a repeat said twice with a change too small
to move a token ("seven minutes" → "7 minutes") is missed; two unrelated requests
inside 45 s that share an entity ("lights on" / "lights off") are flagged; and a
correction that arrives as a fresh session is invisible.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .records import RequestRecord

RULE_VERSION = "corr-1"
DEFAULT_WINDOW_SECONDS = 45.0

CORRECTION_CUES: tuple[str, ...] = (
    "no",
    "nope",
    "not that",
    "not what i",
    "i said",
    "i meant",
    "i mean",
    "instead",
    "actually",
    "make it",
    "change it",
    "rather",
    "wrong",
    "cancel that",
    "scratch that",
    "undo",
    "that's not",
    "thats not",
    "you misunderstood",
    "again but",
)

STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "for",
        "to",
        "of",
        "in",
        "on",
        "at",
        "and",
        "or",
        "my",
        "me",
        "i",
        "it",
        "is",
        "are",
        "be",
        "please",
        "can",
        "could",
        "would",
        "you",
        "that",
        "this",
        "with",
        "do",
        "does",
        "from",
        "what's",
        "whats",
    }
)

WORD = re.compile(r"[a-z0-9']+")


def content_tokens(text: str) -> set[str]:
    """Lowercased content words: what the two utterances must have in common."""
    return {
        token
        for token in WORD.findall(text.lower())
        if token not in STOPWORDS and len(token) > 1
    }


def has_cue(text: str, cues: Sequence[str] = CORRECTION_CUES) -> str | None:
    """The cue that fired, matched on word boundaries so "no" misses "north"."""
    lowered = text.lower()
    for cue in cues:
        if re.search(rf"(?<![a-z]){re.escape(cue)}(?![a-z])", lowered):
            return cue
    return None


@dataclass(frozen=True)
class CorrectionRule:
    """Every knob the heuristic has, plus the version written onto its output."""

    version: str = RULE_VERSION
    window_seconds: float = DEFAULT_WINDOW_SECONDS
    cues: tuple[str, ...] = CORRECTION_CUES
    min_token_overlap: int = 1
    jaccard_floor: float = 0.5
    branch_cue: bool = True
    branch_same_operation: bool = True
    branch_overlap: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "window_seconds": self.window_seconds,
            "min_token_overlap": self.min_token_overlap,
            "jaccard_floor": self.jaccard_floor,
            "branches": {
                "cue": self.branch_cue,
                "same_operation": self.branch_same_operation,
                "overlap": self.branch_overlap,
            },
            "cues": list(self.cues),
        }


@dataclass(frozen=True)
class Correction:
    """One earlier row marked as corrected, with the evidence that marked it."""

    earlier_id: str
    later_id: str
    delta_seconds: float
    rule: Mapping[str, Any] = field(default_factory=dict)
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": "correction",
            "earlier_id": self.earlier_id,
            "later_id": self.later_id,
            "delta_seconds": round(self.delta_seconds, 3),
            "rule": dict(self.rule),
            "evidence": dict(self.evidence),
        }


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def detect(
    records: Sequence[RequestRecord], rule: CorrectionRule = CorrectionRule()
) -> list[Correction]:
    """Corrections among a session-ordered sequence of request records."""
    ordered = sorted(records, key=lambda r: r.ts)
    by_session: dict[str, list[RequestRecord]] = {}
    for record in ordered:
        by_session.setdefault(record.session, []).append(record)

    corrections: list[Correction] = []
    claimed: set[str] = set()
    for session_rows in by_session.values():
        for index in range(1, len(session_rows)):
            later = session_rows[index]
            earlier = _previous_acted_on(session_rows, index, rule.window_seconds)
            if earlier is None or earlier.id == later.id:
                continue
            if earlier.id in claimed:
                continue
            delta = later.ts - earlier.ts
            if earlier.text == later.text:
                continue

            hit = _classify(earlier, later, rule)
            if hit is None:
                continue
            claimed.add(earlier.id)
            corrections.append(
                Correction(
                    earlier_id=earlier.id,
                    later_id=later.id,
                    delta_seconds=delta,
                    rule=rule.as_dict(),
                    evidence=hit,
                )
            )
    return corrections


def _previous_acted_on(
    session_rows: Sequence[RequestRecord], index: int, window: float
) -> RequestRecord | None:
    later = session_rows[index]
    for earlier in reversed(session_rows[:index]):
        if later.ts - earlier.ts > window:
            return None
        if earlier.label is not None:
            return earlier
    return None


def _classify(
    earlier: RequestRecord, later: RequestRecord, rule: CorrectionRule
) -> dict[str, Any] | None:
    """Which branch fires, with the evidence for it. ``None`` when none does."""
    earlier_tokens = content_tokens(earlier.text)
    later_tokens = content_tokens(later.text)
    shared = sorted(earlier_tokens & later_tokens)

    if rule.branch_cue:
        cue = has_cue(later.text, rule.cues)
        if cue is not None:
            return {
                "branch": "cue",
                "cue": cue,
                "shared_tokens": shared,
                "earlier_label": earlier.label,
                "later_label": later.label,
            }

    if rule.branch_same_operation and earlier.label is not None and earlier.label == later.label:
        if len(shared) >= rule.min_token_overlap:
            return {
                "branch": "same_operation",
                "shared_tokens": shared,
                "earlier_label": earlier.label,
                "later_label": later.label,
            }

    if rule.branch_overlap:
        overlap = _jaccard(earlier_tokens, later_tokens)
        if overlap >= rule.jaccard_floor:
            return {
                "branch": "overlap",
                "jaccard": round(overlap, 3),
                "shared_tokens": shared,
                "earlier_label": earlier.label,
                "later_label": later.label,
            }
    return None
