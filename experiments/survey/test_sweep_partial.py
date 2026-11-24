"""What the sweep writes when one stage raises: a partial artefact, not a tombstone.

    .venv-embed/bin/python experiments/survey/test_sweep_partial.py

The failure this covers is the one the first Windows run produced: the full
sweep died with ``OSError: [Errno 22] Invalid argument``, ``failed_stage:
"harness"``, no traceback, and four folds that had already been measured were
dropped because the row was replaced by a two-line tombstone.

Every case below injects ``OSError(EINVAL)`` into one stage and asserts the
artefact still names the stage, carries the traceback text, and keeps the fold
numbers that were measured. The first case injects it into the trainer's
per-fit output directory, because that is the one filesystem cycle the full
sweep performs and the smoke run does not (``.venv-embed/bin/python
/tmp/fit_os_probe.py`` measured it: an ``atexit``-less ``%TEMP%/surveysweep-*``
tree written and deleted once per fit).

No model is loaded and nothing is fine-tuned: ``sweep()`` takes a stub predictor
and a stub loader, and the host facts and model-card lookups are stubbed too, so
this runs in seconds and needs no network. The corpus, the fold assignment and
the metrics are the real ones.

Exit code is 0 only when every check passes.
"""

from __future__ import annotations

import argparse
import contextlib
import errno
import json
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))

import numpy as np

import setfit_sweep
import survey_env
import survey_setfit

FAILURES: list[str] = []
CHECKS = 0
EINVAL = OSError(errno.EINVAL, os.strerror(errno.EINVAL))


def check(case: str, condition: bool, message: str) -> None:
    global CHECKS
    CHECKS += 1
    if not condition:
        FAILURES.append(f"{case}: {message}")
        print(f"  FAIL {case}: {message}", file=sys.stderr)


class StubParameter:
    numel_value = 22713216
    dtype = np.dtype("float32")

    def numel(self) -> int:
        return self.numel_value


class StubProbe:
    """Stands in for a loaded ``SentenceTransformer``; no torch is imported.

    It carries the surface the ``dtype`` stage touches as well as the one
    ``model_facts`` reads: parameters have a dtype, and ``encode`` returns a
    float32 array, so the pre-fit cast has nothing to do here. The bf16 case is
    ``test_dtype_offline.py``'s, with real torch dtypes behind it.
    """

    max_seq_length = 256
    device = "cpu"

    def parameters(self):
        return [StubParameter()]

    def get_embedding_dimension(self) -> int:
        return 384

    def encode(self, texts, convert_to_tensor=False, show_progress_bar=False):
        return np.zeros((len(list(texts)), 384), dtype=np.float32)

    def float(self):
        return self


class StubPredictor:
    """Memorises what it was fitted on and votes for it: enough shape for metrics.

    ``fit`` calls the same trainer-output-directory helper the real predictor
    calls, so an injected failure there lands where the Windows one did.
    """

    def __init__(self) -> None:
        self.fit_seconds = 0.0
        self._memory: dict[str, str] = {}
        self._classes: list[str] = []

    def fit(self, texts, labels) -> "StubPredictor":
        survey_setfit.fit_output_dir()
        self._classes = sorted(set(labels))
        self._memory = dict(zip(texts, labels))
        return self

    def scores(self, texts) -> np.ndarray:
        out = np.full((len(texts), max(len(self._classes), 1)), 0.01)
        for row, text in enumerate(texts):
            label = self._memory.get(text)
            if label is not None:
                out[row, self._classes.index(label)] = 0.9
        return out

    @property
    def classes(self) -> list[str]:
        return list(self._classes)


def stub_factory() -> StubPredictor:
    return StubPredictor()


def args_for(out: Path, **overrides) -> argparse.Namespace:
    values = {
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "out": str(out),
        "folds": "all",
        "artifact": str(setfit_sweep.DEFAULT_ARTIFACT),
        "passes": 2,
        "no_normalize_embeddings": False,
        "no_rowwise": False,
        "determinism_check_requested": True,
        "keep_predictions": False,
        "offline": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


@contextlib.contextmanager
def patched(**replacements):
    saved = {name: getattr(survey_setfit if name in survey_setfit.__dict__ else setfit_sweep, name)
             for name in replacements}
    modules = {}
    for name, value in replacements.items():
        module = survey_setfit if name in survey_setfit.__dict__ else setfit_sweep
        modules[name] = module
        setattr(module, name, value)
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(modules[name], name, value)


@contextlib.contextmanager
def raise_einval_once(module, attribute: str):
    """Make ``module.attribute`` raise ``OSError(EINVAL)`` on its first call, then behave."""
    original = getattr(module, attribute)
    calls = {"n": 0}

    def failing(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise EINVAL
        return original(*args, **kwargs)

    setattr(module, attribute, failing)
    try:
        yield calls
    finally:
        setattr(module, attribute, original)


def base_stubs():
    return {
        "environment": lambda: {
            "hostname": "fixture",
            "platform": "Windows-10",
            "python": "3.11.9",
            "torch_threads": 8,
            "loadavg_1m": None,
        },
        "model_card": lambda model_id: {"license": "apache-2.0", "error": None},
        "snapshot": lambda model_id: {"bytes": 91578415, "files": 11, "error": None},
    }


def run_sweep(out: Path, **overrides) -> dict:
    stubs = base_stubs()
    with patched(**stubs):
        return setfit_sweep.sweep(
            args_for(out, **overrides),
            factory=stub_factory,
            loader=lambda model_id, offline=False: StubProbe(),
        )


def fold_numbers(row: dict) -> list[int]:
    grouped = next((r for r in row["runs"] if r.get("protocol") == "grouped"), {})
    return sorted(int(f) for f in (grouped.get("per_fold_accuracy_by_fold") or {}))


def case_clean_sweep(tmp: Path) -> None:
    case = "clean-sweep"
    row = run_sweep(tmp / "clean.json")
    check(case, row["status"] == "ok", f"status is {row['status']}")
    check(case, row["failed_stage"] is None, f"failed_stage is {row['failed_stage']}")
    check(case, row["traceback"] is None, "traceback must be null on a clean run")
    check(case, row["missing"] == [], f"missing is {row['missing']}")
    check(case, fold_numbers(row) == [0, 1, 2, 3, 4], f"folds {fold_numbers(row)}")
    protocols = sorted(r.get("protocol") for r in row["runs"])
    check(case, protocols == ["grouped", "rowwise"], f"protocols {protocols}")
    check(case, (row.get("latency_passes") or {}).get("p50_ms") is not None, "latency pass ran")
    check(case, row["determinism"].get("checked") is True, "determinism receipt missing")
    check(case, bool(row["halves"]), "halves missing")
    check(case, row["stage_errors"] == {}, f"stage_errors {row['stage_errors']}")
    check(case, row["trainer_output_dir"] is not None, "fit wrote no trainer output directory")


def case_trainer_output_einval(tmp: Path) -> None:
    case = "trainer-output-einval"
    with raise_einval_once(survey_setfit, "fit_output_dir") as calls:
        row = run_sweep(tmp / "einval.json")
    check(case, calls["n"] >= 1, "the injected operation was never called")
    check(case, row["status"] == "partial", f"status is {row['status']}")
    check(case, row["failed_stage"] == "fold 0", f"failed_stage is {row['failed_stage']}")
    check(case, "OSError" in (row["error"] or ""), f"error is {row['error']}")
    check(case, "Invalid argument" in (row["error"] or ""), f"error is {row['error']}")
    check(case, "fit_output_dir" in (row["traceback"] or ""), "traceback does not name the operation")
    check(case, "Errno 22" in (row["traceback"] or ""), "traceback lost the errno")
    check(case, fold_numbers(row) == [1, 2, 3, 4], f"surviving folds {fold_numbers(row)}")
    check(case, "fold 0" in row["missing"], f"missing is {row['missing']}")
    check(case, "folds" in row["stage_errors"], "fold failure not recorded in stage_errors")
    grouped = next(r for r in row["runs"] if r.get("protocol") == "grouped")
    check(case, grouped["folds_scored"] == [1, 2, 3, 4], f"folds_scored {grouped['folds_scored']}")
    check(case, grouped["accuracy"] is not None, "the surviving folds carry no pooled accuracy")
    selection = row["halves"]["selection"]
    check(case, selection["missing_folds"] == [0], f"selection missing {selection['missing_folds']}")
    check(case, selection["complete"] is False, "a half with a missing fold claimed completeness")


def case_rowwise_einval(tmp: Path) -> None:
    case = "rowwise-einval"
    with raise_einval_once(setfit_sweep, "evaluate_rowwise"):
        row = run_sweep(tmp / "rowwise.json")
    check(case, row["status"] == "partial", f"status is {row['status']}")
    check(case, row["failed_stage"] == "rowwise", f"failed_stage is {row['failed_stage']}")
    check(case, "Errno 22" in (row["traceback"] or ""), "traceback lost the errno")
    check(case, fold_numbers(row) == [0, 1, 2, 3, 4], f"folds {fold_numbers(row)}")
    check(case, "rowwise" in row["missing"], f"missing is {row['missing']}")
    protocols = sorted(r.get("protocol") for r in row["runs"])
    check(case, protocols == ["grouped"], f"protocols {protocols}")


def case_latency_einval(tmp: Path) -> None:
    case = "latency-einval"
    with raise_einval_once(setfit_sweep, "latency_passes"):
        row = run_sweep(tmp / "latency.json")
    check(case, row["status"] == "partial", f"status is {row['status']}")
    check(case, row["failed_stage"] == "latency_pass", f"failed_stage is {row['failed_stage']}")
    check(case, fold_numbers(row) == [0, 1, 2, 3, 4], f"folds {fold_numbers(row)}")
    check(case, "latency_pass" in row["missing"], f"missing is {row['missing']}")
    check(case, row["latency_passes"] if "latency_passes" in row else True, "no stray block")


def case_determinism_einval(tmp: Path) -> None:
    case = "determinism-einval"
    with raise_einval_once(setfit_sweep, "determinism_check"):
        row = run_sweep(tmp / "determinism.json")
    check(case, row["status"] == "partial", f"status is {row['status']}")
    check(case, row["failed_stage"] == "determinism", f"failed_stage is {row['failed_stage']}")
    check(case, row["determinism"] == {"checked": False}, f"determinism {row['determinism']}")
    check(case, fold_numbers(row) == [0, 1, 2, 3, 4], f"folds {fold_numbers(row)}")


def case_every_fold_fails(tmp: Path) -> None:
    case = "every-fold-fails"

    class Broken:
        def fit(self, texts, labels):
            raise EINVAL

    stubs = base_stubs()
    with patched(**stubs):
        row = setfit_sweep.sweep(
            args_for(tmp / "broken.json"),
            factory=Broken,
            loader=lambda model_id, offline=False: StubProbe(),
        )
    check(case, row["status"] == "error", f"status is {row['status']}")
    check(case, row["failed_stage"] == "fold 0", f"failed_stage is {row['failed_stage']}")
    check(case, "Errno 22" in (row["traceback"] or ""), "traceback lost the errno")
    check(case, row["runs"] == [], f"runs {row['runs']}")
    check(case, sorted(row["stage_errors"]["folds"]) == ["0", "1", "2", "3", "4"], "not every fold recorded")


def case_artefact_written(tmp: Path) -> None:
    case = "artefact-on-disk"
    with raise_einval_once(survey_setfit, "fit_output_dir"):
        row = run_sweep(tmp / "written.json")
    path = survey_env.write_json(row, tmp / "nested" / "dir" / "written.json")
    raw = path.read_bytes()
    check(case, b"\r\n" not in raw, "artefact carries CRLF line endings")
    payload = json.loads(raw.decode("utf-8"))
    for field in ("status", "traceback", "missing", "stage_errors", "failed_stage"):
        check(case, field in payload, f"artefact has no {field}")
    check(case, payload["status"] == "partial", f"status is {payload['status']}")
    check(case, payload["traceback"].startswith("Traceback"), "traceback is not a traceback")
    leftovers = sorted(p.name for p in path.parent.iterdir() if p.name != path.name)
    check(case, leftovers == [], f"the atomic write left stragglers: {leftovers}")


def case_output_root_lifecycle(tmp: Path) -> None:
    case = "output-root-lifecycle"
    root = survey_setfit.current_output_root()
    check(case, root is not None, "no fit ever created a trainer output root")
    entries = sorted(p.name for p in root.iterdir()) if root is not None else []
    check(case, len(entries) == 1, f"more than one fit directory survived: {entries}")
    check(case, bool(entries) and entries[0].startswith("fit-"), f"entries {entries}")


def case_optional_stage_skips_are_not_failures(tmp: Path) -> None:
    case = "requested-skips"
    row = run_sweep(
        tmp / "skips.json",
        no_rowwise=True,
        determinism_check_requested=False,
        passes=0,
    )
    check(case, row["status"] == "ok", f"a requested skip made the row {row['status']}")
    check(case, row["missing"] == [], f"missing is {row['missing']}")
    check(case, len(row["skipped"]) == 3, f"skipped {row['skipped']}")


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="survey-partial-test-"))
    print(f"fixtures in {root}\n", flush=True)
    cases = (
        case_clean_sweep,
        case_trainer_output_einval,
        case_rowwise_einval,
        case_latency_einval,
        case_determinism_einval,
        case_every_fold_fails,
        case_artefact_written,
        case_output_root_lifecycle,
        case_optional_stage_skips_are_not_failures,
    )
    for case in cases:
        print(f"case {case.__name__}", flush=True)
        case(root)
    print(f"\n{CHECKS} checks, {len(FAILURES)} failures")
    for failure in FAILURES:
        print(f"  {failure}")
    return 0 if not FAILURES else 1


if __name__ == "__main__":
    raise SystemExit(main())
