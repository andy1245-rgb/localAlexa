"""What is dropped before an utterance is written, and why.

The raw utterance has to survive: it is the training example. Everything else
that a spoken sentence can carry incidentally does not, and the exclusion list is
here rather than inline in the writer so it can be read, argued with and changed
in one place.

The list is a *redaction* list, not a filter: the utterance is kept, the matched
span is replaced with a typed marker, and the markers are recorded on the row so
a later reader can see what was removed without seeing it. What is not on the
list goes to disk, including anything nobody thought of. That is the honest limit
of this approach and it is written down in ``docs/data-collection.md``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

MAX_TEXT = 400


@dataclass(frozen=True)
class Exclusion:
    """One redaction rule: what it is called, what it matches, what replaces it.

    ``requires`` is a cheap pre-gate: if none of those substrings occurs in the
    utterance the pattern cannot match, so it is not run. Eight regex scans on
    every utterance is 7 us of the request path; most utterances contain none of
    these characters and the scans are skipped.
    """

    name: str
    pattern: re.Pattern[str]
    marker: str
    requires: tuple[str, ...] = ()


DIGITS: tuple[str, ...] = tuple("0123456789")

EXCLUSIONS: tuple[Exclusion, ...] = (
    Exclusion(
        "url",
        re.compile(r"\b(?:https?://|www\.)\S+", re.IGNORECASE),
        "[url]",
        ("://", "www.", "WWW."),
    ),
    Exclusion(
        "email",
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        "[email]",
        ("@",),
    ),
    # A phone number, not a date or a time: the patterns require a leading +, a
    # parenthesised area code, or the 3-3-4 grouping. Dates and durations survive.
    Exclusion(
        "phone",
        re.compile(
            r"(?:\+\d[\d\s().-]{7,}\d|\(\d{3}\)\s*\d{3}[-.\s]\d{4}|\b\d{3}[-.]\d{3}[-.]\d{4}\b)"
        ),
        "[phone]",
        DIGITS,
    ),
    Exclusion(
        "path_posix",
        re.compile(r"(?<![\w/])(?:~|\.{1,2})?/(?:[\w.-]+/)+[\w.-]*"),
        "[path]",
        ("/",),
    ),
    Exclusion(
        "path_windows",
        re.compile(r"\b[A-Za-z]:\\(?:[^\s\\]+\\)*[^\s\\]*"),
        "[path]",
        ("\\",),
    ),
    Exclusion("ipv4", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "[ip]", DIGITS),
    Exclusion(
        "secret",
        re.compile(
            r"\b(?:sk-[A-Za-z0-9]{8,}|ghp_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{12,}|"
            r"xox[baprs]-[A-Za-z0-9-]{8,})\b"
        ),
        "[secret]",
        ("sk-", "ghp_", "AKIA", "xox"),
    ),
    Exclusion("long_number", re.compile(r"\b\d{12,}\b"), "[long-number]", DIGITS),
)


def redact(text: str, exclusions: Sequence[Exclusion] = EXCLUSIONS) -> tuple[str, tuple[str, ...]]:
    """The utterance with excluded spans replaced, plus the names that fired."""
    cleaned = " ".join(text.split())
    fired: list[str] = []
    for exclusion in exclusions:
        if exclusion.requires and not any(r in cleaned for r in exclusion.requires):
            continue
        cleaned, hits = exclusion.pattern.subn(exclusion.marker, cleaned)
        if hits:
            fired.append(exclusion.name)
    if len(cleaned) > MAX_TEXT:
        cleaned = cleaned[:MAX_TEXT] + "…"
        fired.append("truncated")
    return cleaned, tuple(fired)
