"""Rung 3's SetFit recipe with the encoder as a parameter.

The ladder's rung 3 (`localalexa.ladder.setfit_rung.SetFitPredictor`) hard-codes
one thing that a base-model sweep has to vary: the encoder. Everything else is
copied from it verbatim — the SetFit contrastive pair generator, one epoch, eight
iterations, batch size 16, ``classifiers.SEED``, a fresh ``SetFitModel`` per fit —
and the constants are imported from that module rather than retyped, so they
cannot drift. ``RECIPE`` below is what lands in every sweep artifact.

Two things are written out here instead of inherited:

* ``encoder``. ``SetFitPredictor`` already takes it, but it also hides the flag
  below, and the flag cannot be passed through.
* ``normalize_embeddings``. In setfit 1.2.0 the sklearn head sees whatever
  geometry the body emits, and the default is *unnormalised*. The survey's
  recipe puts the logistic head on unit-norm embeddings, because that is the
  geometry the port reproduced and the one that makes an L2-normalised head
  scoring a model it did not train on impossible
  (``docs/ladder-results-2026-11-10.md``, "The harness port"). Fit and predict
  both read this one attribute inside ``SetFitModel``, so the pair can only move
  together — which is the property that matters. Measured both ways on MiniLM in
  the survey README, so the choice is a number, not a preference.

Two things the ladder never had to handle, because it only ever ran MiniLM, are
handled on the way in:

* **The body's dtype.** A bf16 checkpoint
  (``ibm-granite/granite-embedding-30m-english``) loads as bf16 on CPU and the
  float32 head cannot train on it — ``Found dtype Float but expected
  BFloat16``. :func:`survey_dtype.coerce_float32` casts the body once, right
  after the load, and asserts that a real forward pass returns float32
  embeddings before anything is fitted. A cast that does not take raises
  :class:`survey_dtype.DtypeCoercionError` with the dtype string in the message,
  so the stage that records it can say ``bfloat16`` instead of "harness".
* **The load path.** A dead Hub client is process-wide, so a network blink used
  to fail every fit after it. :func:`survey_load.load_with_fallback` tries
  online once and then retries from the local cache, recording which path
  produced the model in ``load_mode``. ``offline=True`` (``--offline``) skips
  the online attempt.
"""

from __future__ import annotations

import atexit
import random
import shutil
import tempfile
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from localalexa.classifiers import SEED
from localalexa.ladder.setfit_rung import ENCODER, HEAD_ITERATIONS

from survey_dtype import coerce_float32
from survey_load import load_with_fallback

HEAD = "setfit.LogisticRegression on body embeddings"
OUTPUT_ROOT_PREFIX = "surveysweep-"
RECIPE = {
    "name": "rung-3-setfit",
    "source": "localalexa.ladder.setfit_rung (pairs, epochs, iterations, batch, seed)",
    "pairs": "SetFit contrastive pairs generated from the fold's training pool",
    "num_iterations": HEAD_ITERATIONS,
    "num_epochs": 1,
    "batch_size": 16,
    "head": HEAD,
    "normalize_embeddings": True,
    "seed": SEED,
}


_output_root: Path | None = None
_fit_count = 0


def _remove_output_root() -> None:
    global _output_root
    root, _output_root = _output_root, None
    if root is not None:
        shutil.rmtree(root, ignore_errors=True)


def output_root() -> Path:
    """The one directory every fit in this process writes its trainer output into.

    ``SentenceTransformerTrainer`` saves the whole body on ``on_train_end``
    whether or not checkpointing is on, plus a checkpoint with the optimizer and
    scheduler state: about 260 MB per fit, measured, and nothing here ever reads
    it back. That write used to land in a directory of its own, created per fit
    and deleted in a ``finally``: a five-fold sweep with the rowwise and
    determinism fits is seven create/write/list/delete cycles of a 260 MB tree in
    ``%TEMP%``, against one in the smoke run. On Windows that is seven chances to
    remove a directory whose files a trainer, an mmap or a cache lock is still
    holding, and the removal was ``ignore_errors=True``, so a failure there was
    invisible.

    One root per process, removed once at exit, replaces them. Each fit still
    gets a fresh subdirectory of it: a fit that rewrites a path an earlier fit
    wrote is how a half-written ``model.safetensors`` gets memory-mapped, and
    that is the shape of failure this harness is least able to see.
    """
    global _output_root
    if _output_root is None:
        _output_root = Path(tempfile.mkdtemp(prefix=OUTPUT_ROOT_PREFIX))
        atexit.register(_remove_output_root)
    return _output_root


def fit_output_dir() -> Path:
    """A subdirectory of :func:`output_root` that is this fit's alone.

    The previous fit's directory is cleared here rather than at the end of that
    fit, which is the point: a fit never deletes the tree it just wrote, and
    never writes into a tree it is deleting. Two directories on disk at once is
    the peak, instead of one per fit for the whole sweep. The clear is
    best-effort — it frees space nothing reads, and a failure there is not worth
    ending a fold over. The exit-time removal in :func:`output_root` is the one
    that must not leak; a process killed outright leaves at most two fit
    directories for the OS temp cleaner, where the old code left one per fit
    whose removal it had swallowed.
    """
    global _fit_count
    _fit_count += 1
    root = output_root()
    if _fit_count > 1:
        shutil.rmtree(root / f"fit-{_fit_count - 1:02d}", ignore_errors=True)
    path = root / f"fit-{_fit_count:02d}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def current_output_root() -> Path | None:
    """The trainer output root if a fit has created one, else ``None``."""
    return _output_root


class SurveySetFitPredictor:
    """Contrastive encoder fine-tune plus a logistic head, for any sentence encoder."""

    def __init__(
        self,
        encoder: str = ENCODER,
        *,
        num_iterations: int = HEAD_ITERATIONS,
        batch_size: int = 16,
        normalize_embeddings: bool = True,
        offline: bool = False,
    ) -> None:
        self.encoder = encoder
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.offline = offline
        self.fit_seconds = 0.0
        self.n_pairs = 0
        self.load: dict[str, object] | None = None
        self.load_mode: str | None = None
        self.dtype: dict[str, object] | None = None
        self._model = None
        self._classes: list[str] = []

    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> "SurveySetFitPredictor":
        from datasets import Dataset
        from setfit import SetFitModel, Trainer, TrainingArguments

        random.seed(SEED)
        np.random.seed(SEED)
        started = time.perf_counter()
        self._model, self.load = load_with_fallback(
            lambda offline: SetFitModel.from_pretrained(
                self.encoder,
                normalize_embeddings=self.normalize_embeddings,
                local_files_only=offline,
            ),
            offline=self.offline,
            model_id=self.encoder,
        )
        self.load_mode = self.load["mode"]  # type: ignore[index]
        self.dtype = coerce_float32(self._model, [str(text) for text in list(texts)[:1]])
        arguments = TrainingArguments(
            batch_size=self.batch_size,
            num_epochs=1,
            num_iterations=self.num_iterations,
            seed=SEED,
            report_to=[],
            show_progress_bar=False,
            output_dir=str(fit_output_dir()),
        )
        trainer = Trainer(
            model=self._model,
            train_dataset=Dataset.from_dict(
                {"text": list(texts), "label": list(labels)}
            ),
            args=arguments,
        )
        trainer.train()
        self._classes = [str(label) for label in self._model.labels]
        self.n_pairs = len(texts) * 2 * self.num_iterations
        self.fit_seconds = time.perf_counter() - started
        return self

    def scores(self, texts: Sequence[str]) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("predictor used before fit")
        return np.asarray(self._model.predict_proba(list(texts)), dtype=float)

    @property
    def classes(self) -> list[str]:
        return list(self._classes)


def recipe_for(encoder: str, normalize_embeddings: bool = True) -> dict[str, object]:
    """What this sweep actually ran, for the artifact."""
    return {
        **RECIPE,
        "encoder": encoder,
        "normalize_embeddings": normalize_embeddings,
        "head": HEAD,
    }
