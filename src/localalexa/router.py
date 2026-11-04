"""Reference-similarity router: per-tool max cosine similarity + entropy confidence.

Confidence is Jev-style normalised entropy, ``1 - H(p) / ln(K)``, taken over a
softmax of the per-tool maximum cosine similarities. 1.0 means certain, 0.0
means the similarities are uniform across the label set.

The embedding backend is injected. ``SentenceTransformerEncoder`` is the
production default and imports ``sentence-transformers`` lazily, so the router
math and its tests run with numpy alone.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

from .data import Registry


class Encoder(Protocol):
    """Anything that turns texts into L2-normalised row vectors."""

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class SentenceTransformerEncoder:
    """Default encoder: all-MiniLM-L6-v2 from the registry's ``embedding_model``."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ImportError(
                    "the semantic router needs sentence-transformers: "
                    "pip install 'localalexa[embeddings]'"
                ) from exc
            self._model = SentenceTransformer(self.model_name)
        return np.asarray(self._model.encode(list(texts), normalize_embeddings=True))


def softmax(sims: Sequence[float], tau: float) -> np.ndarray:
    """Numerically stable softmax of similarities scaled by 1/tau."""
    if tau <= 0:
        raise ValueError("tau must be positive")
    x = np.asarray(sims, dtype=np.float64) / tau
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def entropy_confidence(probs: Sequence[float]) -> float:
    """Jev-style confidence: ``1 - H(p) / ln(K)``.

    Uniform over K labels -> 0.0. All mass on one label -> 1.0. A single-label
    distribution has zero entropy and no spread to normalise against, so it is
    reported as certain.
    """
    p = np.asarray(probs, dtype=np.float64)
    k = p.size
    if k < 2:
        return 1.0
    h = -float(np.sum(p[p > 0] * np.log(p[p > 0])))
    return 1.0 - h / math.log(k)


@dataclass(frozen=True)
class Choice:
    """One router decision, shaped like the experiment scripts' dicts."""

    top_tool: str
    top_sim: float
    runner_up: str
    runner_up_sim: float
    probabilities: dict[str, float]
    confidence: float
    margin: float
    low_sim: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "top_tool": self.top_tool,
            "top_sim": self.top_sim,
            "runner_up": self.runner_up,
            "runner_up_sim": self.runner_up_sim,
            "probabilities": dict(self.probabilities),
            "confidence": self.confidence,
            "margin": self.margin,
            "low_sim": self.low_sim,
        }


def decide(
    per_tool_sims: Sequence[float],
    tool_ids: Sequence[str],
    *,
    tau: float = 0.08,
    low_sim_floor: float = 0.35,
) -> Choice:
    """Turn per-tool similarities into a decision. Pure — no encoder involved."""
    sims = np.asarray(per_tool_sims, dtype=np.float64)
    if sims.size != len(tool_ids):
        raise ValueError(
            f"{sims.size} similarities for {len(tool_ids)} tools"
        )
    order = np.argsort(-sims)
    top, second = int(order[0]), int(order[1])
    probs = softmax(sims, tau)
    return Choice(
        top_tool=tool_ids[top],
        top_sim=float(sims[top]),
        runner_up=tool_ids[second],
        runner_up_sim=float(sims[second]),
        probabilities={t: float(p) for t, p in zip(tool_ids, probs)},
        confidence=entropy_confidence(probs),
        margin=float(sims[top] - sims[second]),
        low_sim=bool(sims[top] < low_sim_floor),
    )


class SemanticRouter:
    """Routers a utterance against the registry's reference utterances.

    One decision per text; the label set is the 17 tools plus the none route.
    """

    def __init__(
        self,
        registry: Registry,
        encoder: Encoder | None = None,
        tau: float = 0.08,
        low_sim_floor: float = 0.35,
    ) -> None:
        self.registry = registry
        self.encoder = encoder or SentenceTransformerEncoder(registry.embedding_model)
        self.tau = tau
        self.low_sim_floor = low_sim_floor
        self.tool_ids = registry.label_ids
        self._ref_matrix: np.ndarray | None = None
        self._ref_tool_idx: np.ndarray | None = None

    def _reference_matrix(self) -> tuple[np.ndarray, np.ndarray]:
        if self._ref_matrix is None:
            texts, idx = [], []
            for position, tool in enumerate(self.registry.tools):
                for ref in tool.references:
                    texts.append(ref)
                    idx.append(position)
            none_position = len(self.registry.tools)
            for ref in self.registry.none_references:
                texts.append(ref)
                idx.append(none_position)
            self._ref_matrix = np.asarray(self.encoder.encode(texts), dtype=np.float64)
            self._ref_tool_idx = np.asarray(idx)
        return self._ref_matrix, self._ref_tool_idx

    def _per_tool_sims(self, queries: np.ndarray) -> np.ndarray:
        ref_matrix, ref_idx = self._reference_matrix()
        sims = queries @ ref_matrix.T  # (B, N_refs)
        per_tool = np.full((queries.shape[0], len(self.tool_ids)), -1.0)
        for t_i, s in zip(ref_idx, sims.T):
            np.maximum(per_tool[:, t_i], s, out=per_tool[:, t_i])
        return per_tool

    def predict(self, text: str) -> Choice:
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: Sequence[str]) -> list[Choice]:
        queries = np.asarray(self.encoder.encode(texts), dtype=np.float64)
        return [
            decide(row, self.tool_ids, tau=self.tau, low_sim_floor=self.low_sim_floor)
            for row in self._per_tool_sims(queries)
        ]

