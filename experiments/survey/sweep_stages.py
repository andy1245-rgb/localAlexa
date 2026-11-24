"""Who failed, at which stage, with what traceback — and what may still be written.

The sweep used to have one ``except Exception`` around the whole run, which
turned every failure into ``failed_stage: "harness"`` and threw the stack away.
That is how a Windows ``OSError: [Errno 22]`` cost the whole sweep: the five
fold accuracies had already been measured and were in memory when the exception
replaced the row with a two-line tombstone.

So each stage is attempted through this book. A stage records one of three
outcomes — ``ok``, ``failed``, ``skipped`` — with its wall time, its error text,
its exception type and the full ``traceback`` text when it raised. The book then
decides the row's ``status``:

``ok``
    every stage that was asked for ran, and every fold was measured.
``partial``
    something raised or could not be measured, but the folds that did run still
    carry their numbers; the row names what is missing.
``error``
    no fold was measured. There is nothing to be partial about.

A stage marked ``skipped`` because the operator asked for it (``--no-rowwise``)
is not a failure and does not make the row partial; that belongs in the row's
``skipped`` list, where the flag that caused it is named. Only a stage that
raised, or a fold that could not be fitted, makes a row partial.

The failure `category` is the coarse taxonomy a reader triages on, and it is
narrower than "something broke":

===============  ================================================================
``load``         the encoder could not be loaded online or from the cache
``dtype``        it loaded in a dtype the float32 head cannot train with, or could
                 not be cast to float32
``fit``          a fold's contrastive fine-tune / head fit raised
``rowwise``      the 200-reference rowwise fit raised
``determinism``  the second fit of the first fold raised
``latency``      the dedicated latency pass raised
===============  ================================================================

``load``, ``dtype``, ``harness`` and ``fold_assignment`` are fatal; the rest are
not. A fold that fails is reported as ``fold <n>`` because the fold is what the
reader needs, with its category beside it: ``fold 0`` + ``dtype`` says "the
granite cast, on the first fold" where a bare ``fold 0`` said nothing.
"""

from __future__ import annotations

import sys
import time
import traceback
from typing import Callable

OK = "ok"
FAILED = "failed"
SKIPPED = "skipped"

_TRACEBACK_LIMIT = 20000
_FATAL_FIRST = ("harness", "load", "dtype", "fold_assignment")

#: Stage name -> failure category. A stage absent from this map is its own
#: category; a fold failure carries the category of the exception that killed it.
CATEGORY_OF_STAGE = {
    "load": "load",
    "dtype": "dtype",
    "rowwise": "rowwise",
    "determinism": "determinism",
    "latency_pass": "latency",
}
CATEGORY_FIT = "fit"


def category_of_stage(stage: str) -> str:
    return CATEGORY_OF_STAGE.get(stage, stage)


class StageAbort(RuntimeError):
    """A stage marked fatal failed; the book already holds the record."""

    def __init__(self, stage: str) -> None:
        super().__init__(f"fatal stage failed: {stage}")
        self.stage = stage


def traceback_text(exc: BaseException) -> str:
    text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return text[-_TRACEBACK_LIMIT:]


def describe(exc: BaseException, limit: int = 600) -> str:
    text = f"{type(exc).__name__}: {exc}".replace("\n", " ").strip()
    return text[:limit]


class StageBook:
    """Ordered stage records plus the fold-level failures of the grouped run."""

    def __init__(self) -> None:
        self.records: dict[str, dict[str, object]] = {}
        self.fold_failures: dict[int, dict[str, object]] = {}
        self.order: list[str] = []

    def _put(self, stage: str, record: dict[str, object]) -> None:
        if stage not in self.records:
            self.order.append(stage)
        self.records[stage] = record

    def ok(self, stage: str, seconds: float) -> None:
        self._put(stage, {"status": OK, "seconds": round(seconds, 3)})

    def skipped(self, stage: str, why: str) -> None:
        self._put(stage, {"status": SKIPPED, "why": why})

    def failed(self, stage: str, exc: BaseException, seconds: float | None = None) -> None:
        record: dict[str, object] = {
            "status": FAILED,
            "category": category_of_stage(stage),
            "error": describe(exc),
            "exception_type": type(exc).__name__,
            "traceback": traceback_text(exc),
        }
        if seconds is not None:
            record["seconds"] = round(seconds, 3)
        self._put(stage, record)
        print(f"  ! {stage}: {record['error']}", file=sys.stderr, flush=True)

    def fold_failed(
        self,
        fold: int,
        exc: BaseException,
        seconds: float | None = None,
        category: str = CATEGORY_FIT,
    ) -> None:
        self.fold_failures[fold] = {
            "stage": f"fold {fold}",
            "fold": fold,
            "status": FAILED,
            "category": category,
            "error": describe(exc),
            "exception_type": type(exc).__name__,
            "traceback": traceback_text(exc),
        }
        if seconds is not None:
            self.fold_failures[fold]["seconds"] = round(seconds, 3)
        print(
            f"  ! fold {fold}: {self.fold_failures[fold]['error']}", file=sys.stderr, flush=True
        )

    def attempt(
        self,
        stage: str,
        fn: Callable[[], object],
        *,
        fatal: bool = False,
    ):
        """Run one stage, record its outcome, return its value or ``None``.

        ``fatal`` means the run cannot continue without this stage: the failure
        is recorded and then re-raised as :class:`StageAbort`, which the caller
        is expected to catch and hand to :meth:`apply`.
        """
        started = time.perf_counter()
        try:
            value = fn()
        except Exception as exc:
            self.failed(stage, exc, seconds=time.perf_counter() - started)
            if fatal:
                raise StageAbort(stage) from exc
            return None
        self.ok(stage, seconds=time.perf_counter() - started)
        return value

    def failures(self) -> dict[str, dict[str, object]]:
        return {k: v for k, v in self.records.items() if v["status"] == FAILED}

    def missing(self) -> list[str]:
        missing = [stage for stage in self.order if self.records[stage]["status"] == FAILED]
        missing.extend(f"fold {fold}" for fold in sorted(self.fold_failures))
        return missing

    def first_failure(self) -> dict[str, object] | None:
        """The failure to report as ``failed_stage``, worst news first.

        A fatal stage (``load``, or a bug in the assembly) outranks everything;
        a fold that could not be fitted outranks an optional stage such as the
        latency pass or the model-card lookup, because it is the one that
        changes which numbers exist. Neither is hidden: every one of them is
        in ``stage_errors``.
        """
        for stage in _FATAL_FIRST:
            if self.records.get(stage, {}).get("status") == FAILED:
                return {"stage": stage, **self.records[stage]}
        for fold in sorted(self.fold_failures):
            return dict(self.fold_failures[fold])
        for stage in self.order:
            if self.records[stage]["status"] == FAILED:
                return {"stage": stage, **self.records[stage]}
        return None

    def status(self, *, measured: bool) -> str:
        if not measured:
            return "error"
        if any(record["status"] == FAILED for record in self.records.values()):
            return "partial"
        if self.fold_failures:
            return "partial"
        return "ok"

    def apply(self, row: dict[str, object], *, measured: bool) -> dict[str, object]:
        """Write the book's verdict onto the row: status, error, traceback, stage_errors."""
        first = self.first_failure()
        row["status"] = self.status(measured=measured)
        row["stage_errors"] = {
            stage: {k: v for k, v in self.records[stage].items() if k != "status"}
            for stage in self.order
            if self.records[stage]["status"] == FAILED
        }
        if self.fold_failures:
            row["stage_errors"]["folds"] = {
                str(fold): record for fold, record in sorted(self.fold_failures.items())
            }
        row["missing"] = self.missing()
        row["stage_order"] = list(self.order)
        row["failed_category"] = None if first is None else first.get("category")
        if first is None:
            row["failed_stage"] = None
            row["error"] = None
            row["traceback"] = None
        else:
            row["failed_stage"] = first.get("stage")
            row["error"] = first.get("error")
            row["traceback"] = first.get("traceback")
        return row
