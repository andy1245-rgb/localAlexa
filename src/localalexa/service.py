"""The request path: one utterance in, one response out, logged before it returns.

This is the composition root the firmware or an HTTP layer would call. It exists
because "wire the collector into the real request path" needs a real request path
to wire it into, and until now the package only had a batch evaluator.

The order inside ``handle`` is the contract:

1. tier 0 — if exactly one tool's regex fires, that is the answer and the router
   is never called, which is what ``run_cascade`` does with its residue;
2. otherwise the router decides, and its choice may still escalate;
3. the row is appended (this is the part that must not be lost);
4. the collector settles: it reads the log, runs the correction detector and the
   auto-accept rule, and rewrites the derived views if anything changed;
5. only then is the response returned.

Steps 3 and 4 are both after the decision and before the answer, so a crash
anywhere in between loses the response but never the record. ``settle=False``
moves step 4 to an explicit call for a caller that would rather batch it; the row
guarantee is unaffected.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Sequence

from .cascade import CompiledPatterns
from .collection.autoaccept import AutoAcceptRule
from .collection.collector import Collector, SettleReport
from .collection.corrections import CorrectionRule
from .collection.records import Candidate, RequestRecord
from .data import Registry, load_registry
from .router import Choice, entropy_confidence

Predictor = Callable[[str], Choice]


@dataclass(frozen=True)
class Response:
    """What the assistant did with one utterance."""

    text: str
    label: str | None
    tier: int
    escalated: bool
    confidence: float
    candidates: tuple[Candidate, ...]
    record_id: str
    route_ms: float | None
    settled: SettleReport | None = None

    def describe(self) -> str:
        if self.escalated:
            head = "escalated to the model tier (no label chosen)"
        else:
            head = f"{self.label} (tier {self.tier})"
        ranked = ", ".join(f"{c.label} {c.score:.2f}" for c in self.candidates)
        return (
            f"{head} · confidence {self.confidence:.3f} · "
            f"{(f'{self.route_ms:.1f} ms' if self.route_ms is not None else 'no router call')}"
            + (f"\n  candidates: {ranked}" if ranked else "")
        )


class Assistant:
    """Routes one utterance at a time and records every one of them."""

    def __init__(
        self,
        registry: Registry | None = None,
        *,
        predictor: Predictor | None = None,
        collector: Collector | None = None,
        log=None,
        sim_threshold: float = 0.45,
        conf_threshold: float = 0.35,
        settle: bool = True,
        record_requests: bool = True,
        clock=time.time,
    ) -> None:
        self.registry = registry or load_registry()
        self.compiled = CompiledPatterns.from_registry(self.registry)
        self.sim_threshold = sim_threshold
        self.conf_threshold = conf_threshold
        self.settle_on_handle = settle
        self.record_requests = record_requests
        self._predictor = predictor
        if collector is not None:
            self.collector = collector
        elif record_requests:
            self.collector = Collector(
                log,
                registry=self.registry,
                compiled=self.compiled,
                sim_threshold=sim_threshold,
                conf_threshold=conf_threshold,
                correction_rule=CorrectionRule(),
                auto_rule=AutoAcceptRule(),
            )
        else:
            self.collector = None

    @property
    def predictor(self) -> Predictor:
        if self._predictor is None:
            self._predictor = SemanticPredictor(self.registry).predict
        return self._predictor

    def handle(self, text: str) -> Response:
        """Route one utterance, record it, settle, and answer."""
        hits = self.compiled.probe(text)
        choice: Choice | None = None
        route_ms: float | None = None
        if len(hits) != 1:
            started = time.perf_counter()
            choice = self.predictor(text)
            route_ms = (time.perf_counter() - started) * 1000.0

        if self.collector is None:
            return self._response(text, choice, hits, route_ms, None, record=None)

        record = self.collector.record(text, choice, route_ms=route_ms)
        settled = self.collector.settle() if self.settle_on_handle else None
        return self._response(text, choice, hits, route_ms, settled, record=record)

    def _response(
        self,
        text: str,
        choice: Choice | None,
        hits: Sequence[str],
        route_ms: float | None,
        settled: SettleReport | None,
        *,
        record: RequestRecord | None,
    ) -> Response:
        if len(hits) == 1:
            label, tier, candidates = hits[0], 0, (Candidate(hits[0], 1.0),)
            confidence = 1.0
        elif choice is None:
            label, tier, candidates, confidence = None, 2, (), 0.0
        elif choice.top_sim < self.sim_threshold or choice.confidence < self.conf_threshold:
            label, tier, candidates, confidence = None, 2, tuple(), choice.confidence
        else:
            label, tier = choice.top_tool, 1
            candidates = tuple(
                Candidate(name, float(score))
                for name, score in sorted(choice.probabilities.items(), key=lambda kv: -kv[1])[:3]
            )
            confidence = choice.confidence
        return Response(
            text=text,
            label=label,
            tier=tier,
            escalated=tier == 2,
            confidence=float(confidence),
            candidates=candidates,
            record_id=record.id if record is not None else "",
            route_ms=None if route_ms is None else round(route_ms, 3),
            settled=settled,
        )

    def settle(self) -> SettleReport | None:
        return None if self.collector is None else self.collector.settle()

    def append_effect(self, request_id: str, verified: bool, check: str, detail: str = "") -> None:
        """Report an action's effect, which is the only world-verified signal."""
        from .collection.records import EffectRecord

        if self.collector is None:
            raise RuntimeError("collection is disabled for this assistant")
        self.collector.append_effect(
            EffectRecord(request_id, time.time(), verified, check, detail)
        )

    def close(self) -> None:
        if self.collector is not None:
            self.collector.close()

    def __enter__(self) -> "Assistant":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


class SemanticPredictor:
    """The measured tier-1 model: MiniLM reference similarity behind ``predict``."""

    def __init__(self, registry: Registry) -> None:
        from .router import SemanticRouter

        self.router = SemanticRouter(registry)
        self.router.predict("warm up")  # load the encoder now, not on the first request

    def predict(self, text: str) -> Choice:
        return self.router.predict(text)


def linear_predictor(registry: Registry) -> Predictor:
    """A stand-in for the encoder tier when ``sentence-transformers`` is absent.

    This is rung 2 of the classifier ladder — word TF-IDF + logistic regression —
    wrapped so it emits the same :class:`Choice` shape. It is *not* the model the
    cascade measurement used, and its numbers belong to the ladder, not to the
    README's cascade table. It exists so the request path runs in the base venv
    for tests and for ``localalexa ask``.
    """
    from .ladder.rungs import rung_2_logreg_c100

    predictor = rung_2_logreg_c100().build()
    pairs = [(text, tool.id) for tool in registry.tools for text in tool.references]
    pairs += [(text, "__none__") for text in registry.none_references]
    predictor.fit([text for text, _ in pairs], [label for _, label in pairs])
    classes = list(predictor.classes)

    def predict(text: str) -> Choice:
        scores = predictor.scores([text])[0]
        probabilities = {label: float(score) for label, score in zip(classes, scores)}
        ranked = sorted(probabilities.items(), key=lambda kv: -kv[1])
        top_label, top_score = ranked[0]
        second_label, second_score = ranked[1]
        return Choice(
            top_tool=top_label,
            top_sim=top_score,
            runner_up=second_label,
            runner_up_sim=second_score,
            probabilities=probabilities,
            confidence=entropy_confidence(list(probabilities.values())),
            margin=top_score - second_score,
            low_sim=top_score < 0.35,
        )

    return predict


def default_predictor(registry: Registry) -> tuple[Predictor, str]:
    """The encoder tier if it is installed, else the linear stand-in, with the reason."""
    try:
        return SemanticPredictor(registry).predict, "sentence-transformers router"
    except ImportError as exc:
        return linear_predictor(registry), f"linear stand-in ({exc})"
