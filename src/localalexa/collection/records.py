"""The two records that are written at request time, and nothing else.

``RequestRecord``
    one per utterance the router saw. Written before the response is returned.

``EffectRecord``
    one per action whose outcome the operator's tooling could check (a timer that
    exists, a list that changed). Written when the check happens, not in the hot
    path, because the effect is only knowable after the action ran.

Everything else in the collection pipeline — corrections, the reject pile,
auto-accepted labels, the weekly summary — is derived from these two files by
replay. A derived artefact can be thrown away and rebuilt when a heuristic is
revised; the raw log cannot, which is why the split is where it is.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Mapping

SCHEMA = 1
TOP_CANDIDATES = 3


@dataclass(frozen=True)
class Candidate:
    label: str
    score: float

    def as_dict(self) -> dict[str, Any]:
        return {"label": self.label, "score": round(float(self.score), 6)}


@dataclass(frozen=True)
class RequestRecord:
    """One routing decision, as it happened."""

    id: str
    ts: float
    session: str
    text: str
    tier: int
    label: str | None
    confidence: float
    top_sim: float
    margin: float
    candidates: tuple[Candidate, ...]
    gate: bool
    sim_threshold: float
    conf_threshold: float
    redactions: tuple[str, ...] = ()
    route_ms: float | None = None
    schema: int = SCHEMA

    @property
    def at(self) -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self.ts))

    @property
    def escalated(self) -> bool:
        return self.tier == 2

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": "request",
            "schema": self.schema,
            "id": self.id,
            "ts": round(float(self.ts), 6),
            "at": self.at,
            "session": self.session,
            "text": self.text,
            "redactions": list(self.redactions),
            "tier": self.tier,
            "label": self.label,
            "confidence": round(float(self.confidence), 6),
            "top_sim": round(float(self.top_sim), 6),
            "margin": round(float(self.margin), 6),
            "gate": self.gate,
            "sim_threshold": self.sim_threshold,
            "conf_threshold": self.conf_threshold,
            "route_ms": None if self.route_ms is None else round(float(self.route_ms), 3),
            "candidates": [c.as_dict() for c in self.candidates],
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "RequestRecord":
        return cls(
            id=str(raw["id"]),
            ts=float(raw["ts"]),
            session=str(raw["session"]),
            text=str(raw["text"]),
            tier=int(raw["tier"]),
            label=None if raw.get("label") is None else str(raw["label"]),
            confidence=float(raw.get("confidence", 0.0)),
            top_sim=float(raw.get("top_sim", 0.0)),
            margin=float(raw.get("margin", 0.0)),
            candidates=tuple(
                Candidate(str(c["label"]), float(c["score"]))
                for c in raw.get("candidates", ())
            ),
            gate=bool(raw.get("gate", False)),
            sim_threshold=float(raw.get("sim_threshold", 0.45)),
            conf_threshold=float(raw.get("conf_threshold", 0.35)),
            redactions=tuple(str(r) for r in raw.get("redactions", ())),
            route_ms=None if raw.get("route_ms") is None else float(raw["route_ms"]),
            schema=int(raw.get("schema", SCHEMA)),
        )


@dataclass(frozen=True)
class EffectRecord:
    """An action's effect, as checked by the operator's own tooling.

    ``verified=True`` is evidence from the world that the routing was right.
    ``verified=False`` means the check ran and found nothing — which is evidence
    against the label, but weaker than a correction, because a tool can also fail
    for reasons that have nothing to do with the routing.
    """

    request_id: str
    ts: float
    verified: bool
    check: str
    detail: str = ""
    schema: int = SCHEMA

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": "effect",
            "schema": self.schema,
            "request_id": self.request_id,
            "ts": round(float(self.ts), 6),
            "verified": bool(self.verified),
            "check": self.check,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "EffectRecord":
        return cls(
            request_id=str(raw["request_id"]),
            ts=float(raw["ts"]),
            verified=bool(raw["verified"]),
            check=str(raw["check"]),
            detail=str(raw.get("detail", "")),
            schema=int(raw.get("schema", SCHEMA)),
        )


@dataclass(frozen=True)
class LabelRow:
    """One derived label. Absent rows are the point: silence produces no label."""

    request_id: str
    ts: float
    text: str
    label: str
    polarity: int
    strength: str
    source: str
    detail: Mapping[str, Any] = field(default_factory=dict)
    schema: int = SCHEMA

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": "label",
            "schema": self.schema,
            "request_id": self.request_id,
            "ts": round(float(self.ts), 6),
            "text": self.text,
            "label": self.label,
            "polarity": self.polarity,
            "strength": self.strength,
            "source": self.source,
            "detail": dict(self.detail),
        }


def to_line(payload: Mapping[str, Any]) -> str:
    """One JSON object, one line, no newlines inside. The log's unit of durability."""
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
