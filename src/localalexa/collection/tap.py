"""The request-path tap: route, record, write, return — in that order.

The tap sits *around* a router, not inside it. ``SemanticRouter`` is unchanged,
``cascade.py`` is unchanged; the tap takes the ``Choice`` the router produced,
works out which tier would have answered, appends one row, and only then returns
the choice to the caller. A crash between the decision and the response therefore
loses the response but not the record.

The tier rule is ``cascade.assign_tiers`` evaluated for a single utterance — the
same two thresholds, the same "exactly one regex hit wins" test — because the log
has to say what the cascade said. ``tests/test_collection.py`` holds that claim
to account against ``assign_tiers`` on all 382 evaluation rows, so if the cascade
ever changes, the tap fails a test instead of drifting.
"""

from __future__ import annotations

import time
import uuid
from typing import Callable, Sequence

from ..cascade import CompiledPatterns
from ..data import Registry
from ..router import Choice
from .log import RequestLog
from .records import TOP_CANDIDATES, Candidate, RequestRecord
from .redact import redact

TIER_REGEX = 0
TIER_ROUTER = 1
TIER_ESCALATED = 2


def tier_for(
    text: str,
    choice: Choice | None,
    compiled: CompiledPatterns,
    sim_threshold: float,
    conf_threshold: float,
) -> tuple[int, str | None]:
    """The tier that answers this utterance, and the label it answers with.

    Same branches, same order, same thresholds as ``cascade.assign_tiers``.
    """
    hits = compiled.probe(text)
    if len(hits) == 1:
        return TIER_REGEX, hits[0]
    if choice is None:
        return TIER_ESCALATED, None
    if choice.top_sim < sim_threshold or choice.confidence < conf_threshold:
        return TIER_ESCALATED, None
    return TIER_ROUTER, choice.top_tool


def top_candidates(choice: Choice | None, limit: int = TOP_CANDIDATES) -> tuple[Candidate, ...]:
    if choice is None:
        return ()
    ranked = sorted(choice.probabilities.items(), key=lambda kv: -kv[1])[:limit]
    return tuple(Candidate(label, float(score)) for label, score in ranked)


class CollectionTap:
    """Wraps a router so every utterance it sees is logged before the answer."""

    def __init__(
        self,
        log: RequestLog,
        *,
        registry: Registry | None = None,
        compiled: CompiledPatterns | None = None,
        sim_threshold: float = 0.45,
        conf_threshold: float = 0.35,
        clock=time.time,
        id_factory: Callable[[], str] | None = None,
    ) -> None:
        self.log = log
        self.compiled = compiled or (CompiledPatterns.from_registry(registry) if registry else None)
        self.sim_threshold = sim_threshold
        self.conf_threshold = conf_threshold
        self._clock = clock
        self._id_factory = id_factory or (lambda: uuid.uuid4().hex[:16])

    def observe(
        self, text: str, choice: Choice | None = None, *, route_ms: float | None = None
    ) -> RequestRecord:
        """Build the record for one utterance and write it."""
        cleaned, redactions = redact(text)
        if self.compiled is not None:
            tier, label = tier_for(
                text, choice, self.compiled, self.sim_threshold, self.conf_threshold
            )
        else:
            # No regex tier configured: every row is the router's, gate included.
            escalated = choice is None or (
                choice.top_sim < self.sim_threshold
                or choice.confidence < self.conf_threshold
            )
            tier, label = (TIER_ESCALATED, None) if escalated else (TIER_ROUTER, choice.top_tool)

        candidates = top_candidates(choice)
        if tier == TIER_REGEX and label is not None:
            # The regex answered; the router's opinion never reached the operator.
            candidates = (Candidate(label, 1.0),)

        record = RequestRecord(
            id=self._id_factory(),
            ts=self._clock(),
            session=self.log.session,
            text=cleaned,
            tier=tier,
            label=label,
            confidence=float(choice.confidence) if choice is not None else 1.0,
            top_sim=float(choice.top_sim) if choice is not None else 1.0,
            margin=float(choice.margin) if choice is not None else 0.0,
            candidates=candidates,
            gate=tier == TIER_ESCALATED,
            sim_threshold=self.sim_threshold,
            conf_threshold=self.conf_threshold,
            redactions=redactions,
            route_ms=route_ms,
        )
        self.log.append(record)
        return record

    def wrap(self, predict: Callable[[str], Choice]) -> Callable[[str], Choice]:
        """Decorate a single-utterance predictor."""

        def logged(text: str) -> Choice:
            started = time.perf_counter()
            choice = predict(text)
            route_ms = (time.perf_counter() - started) * 1000.0
            self.observe(text, choice, route_ms=route_ms)
            return choice

        logged.__name__ = f"logged_{getattr(predict, '__name__', 'predict')}"
        return logged

    def wrap_batch(
        self, predict_batch: Callable[[Sequence[str]], Sequence[Choice]]
    ) -> Callable[[Sequence[str]], list[Choice]]:
        """Decorate a batch predictor; one row is written per utterance."""

        def logged(texts: Sequence[str]) -> list[Choice]:
            started = time.perf_counter()
            choices = list(predict_batch(texts))
            total_ms = (time.perf_counter() - started) * 1000.0
            shared = total_ms / max(len(texts), 1)
            for text, choice in zip(texts, choices):
                self.observe(text, choice, route_ms=shared)
            return choices

        logged.__name__ = f"logged_{getattr(predict_batch, '__name__', 'predict_batch')}"
        return logged
