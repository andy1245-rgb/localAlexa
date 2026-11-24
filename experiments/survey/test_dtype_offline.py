"""The bf16 cast and the offline retry, driven by stubs. No model, no network.

    .venv-embed/bin/python experiments/survey/test_dtype_offline.py

Two live failures on the target box, and the two contracts that came out of them:

1. ``ibm-granite/granite-embedding-30m-english`` arrives in bfloat16, setfit's
   logistic head trains in float32, and the first pass through the body raised
   ``Found dtype Float but expected BFloat16`` — recorded as ``failed_stage:
   "harness"``, which named neither the model nor the dtype. The fix is a cast on
   the encoder right after it is loaded, plus an assertion that a real forward
   pass hands the head float32. Both halves are tested here: the cast that takes,
   the cast that does not take, a cast that is silently ignored, and a body whose
   parameters are float32 while its embeddings are not. The last one is why the
   assertion reads the tensor dtype rather than numpy's: on this box
   ``np.asarray`` on a bf16 tensor raises ``Got unsupported ScalarType
   BFloat16``, so a numpy reading cannot tell you what the body emits.

2. ``RuntimeError: Cannot send a request, as the client has been closed.`` hit all
   four models at ``failed_stage: load`` while the Hub was reachable and every
   model was already cached. The contract: try online, retry once from the cache
   with ``HF_HUB_OFFLINE=1``/``TRANSFORMERS_OFFLINE=1`` set in-process, record
   ``load.mode``. Tested: the online path, the fallback path, the sticky
   behaviour after a fallback, ``--offline`` forcing it, both attempts failing,
   and a non-Hub error not being retried at all.

``test_sweep_partial.py`` covers what a failed stage writes. This file covers the
two failures themselves, and the sweep's `dtype` stage and `failed_category`
taxonomy through the same stub harness.

Exit code is 0 only when every check passes.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))

import numpy as np
import torch

import setfit_sweep
import survey_dtype
import survey_load
import sweep_stages
import test_sweep_partial as partial

CLIENT_CLOSED = "Cannot send a request, as the client has been closed."

RESULTS: list[tuple[str, int, list[str]]] = []


class Case:
    """One named case, with its own PASS/FAIL line."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.checks = 0
        self.failures: list[str] = []
        print(f"case {name}", flush=True)

    def check(self, condition: bool, message: str) -> None:
        self.checks += 1
        if not condition:
            self.failures.append(message)
            print(f"  FAIL {message}", file=sys.stderr)

    def done(self) -> None:
        RESULTS.append((self.name, self.checks, list(self.failures)))


class TorchBody(torch.nn.Module):
    """A minimal encoder body: parameters carry a real torch dtype, encode a tensor."""

    max_seq_length = 256

    def __init__(self, dtype=torch.float32, *, emit=None, cast_error=None) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 2).to(dtype)
        self.emit = emit
        self.cast_error = cast_error
        self.float_calls = 0

    def float(self, *args, **kwargs):
        self.float_calls += 1
        if self.cast_error is not None:
            raise self.cast_error
        return super().float()

    def get_embedding_dimension(self) -> int:
        return 2

    def encode(self, texts, convert_to_tensor=True, show_progress_bar=False):
        """Emits its parameters' dtype unless ``emit`` pins one — a cast moves it."""
        dtype = self.emit or self.linear.weight.dtype
        with torch.no_grad():
            out = self.linear(
                torch.zeros(len(list(texts)), 4, dtype=self.linear.weight.dtype)
            ).to(dtype)
        return out if convert_to_tensor else out.to(torch.float32).numpy()


class StubbornBody:
    """Parameters claim bfloat16 and ``float()`` refuses; no torch module involved."""

    max_seq_length = 256

    class Parameter:
        dtype = torch.bfloat16

        def numel(self) -> int:
            return 4

    def parameters(self):
        return [self.Parameter()]

    def float(self):
        raise RuntimeError("cannot cast this body")

    def get_embedding_dimension(self) -> int:
        return 2


def expect_dtype_error(name: str, body, expect_text: str) -> Case:
    case = Case(name)
    try:
        survey_dtype.coerce_float32(body, ["probe"])
    except survey_dtype.DtypeCoercionError as exc:
        case.check(expect_text in str(exc), f"message does not name {expect_text!r}: {exc}")
        case.check(bool(exc.dtypes), "the error carries no dtype list")
    else:
        case.check(False, "a body that is not float32 was accepted")
    case.done()
    return case


def case_bf16_coerced() -> None:
    case = Case("bf16-coerced")
    body = TorchBody(torch.bfloat16)
    record = survey_dtype.coerce_float32(body, ["probe"])
    case.check(record["before"] == ["bfloat16"], f"before is {record['before']}")
    case.check(record["after"] == ["float32"], f"after is {record['after']}")
    case.check(record["coerced"] is True, f"coerced is {record['coerced']}")
    case.check(record["device"] == "cpu", f"device is {record['device']}")
    case.check(body.float_calls == 1, f"float() called {body.float_calls} times")
    case.check(
        {p.dtype for p in body.parameters()} == {torch.float32},
        "the parameters are not float32 after the cast",
    )
    case.check(
        (record["embeddings"] or {}).get("torch") == "float32",
        f"embeddings are {record['embeddings']}",
    )
    case.check(
        (record["embeddings"] or {}).get("numpy") == "float32",
        f"numpy dtype is {record['embeddings']}",
    )
    case.done()


def case_fp32_untouched() -> None:
    body = TorchBody(torch.float32)
    case = Case("fp32-untouched")
    record = survey_dtype.coerce_float32(body, ["probe"])
    case.check(record["before"] == ["float32"], f"before is {record['before']}")
    case.check(record["coerced"] is False, f"coerced is {record['coerced']}")
    case.check(body.float_calls == 0, f"a float32 body was cast {body.float_calls} times")
    case.check(record["after"] == ["float32"], f"after is {record['after']}")
    case.done()


def case_cast_refused() -> None:
    expect_dtype_error(
        "cast-refused",
        StubbornBody(),
        "could not cast the encoder from bfloat16 to float32",
    )


def case_cast_silently_ignored() -> None:
    """``float()`` returns without changing anything — the post-cast assertion must catch it."""
    class Liar(TorchBody):
        def float(self, *args, **kwargs):
            self.float_calls += 1
            return self

    body = Liar(torch.bfloat16)
    case = expect_dtype_error("cast-silently-ignored", body, "bfloat16")
    case.check(body.float_calls == 1, f"float() called {body.float_calls} times")


def case_embeddings_not_float32() -> None:
    """Parameters float32, embeddings bf16: the assertion reads the tensor, not numpy."""
    body = TorchBody(torch.float32, emit=torch.bfloat16)
    case = Case("embeddings-not-float32")
    try:
        survey_dtype.coerce_float32(body, ["probe"])
    except survey_dtype.DtypeCoercionError as exc:
        case.check("bfloat16" in str(exc), f"message does not name bfloat16: {exc}")
        case.check(
            (exc.embedding_dtypes or {}).get("torch") == "bfloat16",
            f"error carries {exc.embedding_dtypes}",
        )
    else:
        case.check(False, "a bf16 embedding was accepted as float32")
    try:
        numpy_view = str(np.asarray(torch.zeros(2, 2, dtype=torch.bfloat16)).dtype)
    except TypeError as exc:
        numpy_view = f"refused: {exc}"
    case.check(
        numpy_view.startswith("refused"),
        f"numpy described a bf16 tensor as {numpy_view} — the numpy reading is not the check",
    )
    case.done()


def case_sweep_dtype_stage_fatal(tmp: Path) -> None:
    case = Case("sweep-dtype-stage-fatal")
    with partial.patched(**partial.base_stubs()):
        row = setfit_sweep.sweep(
            partial.args_for(tmp / "stubborn.json"),
            factory=partial.stub_factory,
            loader=lambda model_id, offline=False: StubbornBody(),
        )
    case.check(row["status"] == "error", f"status is {row['status']}")
    case.check(row["failed_stage"] == "dtype", f"failed_stage is {row['failed_stage']}")
    case.check(row["failed_category"] == "dtype", f"failed_category is {row['failed_category']}")
    case.check("bfloat16" in (row["error"] or ""), f"error does not name the dtype: {row['error']}")
    case.check("dtype" in (row["traceback"] or ""), "traceback does not reach the cast")
    case.check(row["runs"] == [], f"runs {row['runs']}")
    case.check("dtype" in row["missing"], f"missing is {row['missing']}")
    case.check("dtype" in row["stage_errors"], "stage_errors has no dtype record")
    case.check(
        (row.get("load") or {}).get("mode") == "online",
        f"load block lost its mode: {row.get('load')}",
    )
    case.done()


def sweep_with_failing_fit(tmp: Path, exc: BaseException, name: str) -> dict:
    class Broken:
        def fit(self, texts, labels):
            raise exc

    with partial.patched(**partial.base_stubs()):
        return setfit_sweep.sweep(
            partial.args_for(tmp / f"{name}.json"),
            factory=Broken,
            loader=lambda model_id, offline=False: partial.StubProbe(),
        )


def case_fold_categories(tmp: Path) -> None:
    case = Case("fold-categories")
    dtype_row = sweep_with_failing_fit(
        tmp,
        survey_dtype.DtypeCoercionError("encoder is still bfloat16 after the cast"),
        "fold-dtype",
    )
    case.check(dtype_row["failed_stage"] == "fold 0", f"failed_stage is {dtype_row['failed_stage']}")
    case.check(
        dtype_row["failed_category"] == "dtype", f"failed_category is {dtype_row['failed_category']}"
    )
    case.check(
        dtype_row["stage_errors"]["folds"]["0"]["category"] == "dtype",
        "the fold record does not carry its category",
    )
    case.check("bfloat16" in (dtype_row["error"] or ""), f"error is {dtype_row['error']}")

    fit_row = sweep_with_failing_fit(tmp, ValueError("trainer exploded"), "fold-fit")
    case.check(fit_row["failed_stage"] == "fold 0", f"failed_stage is {fit_row['failed_stage']}")
    case.check(
        fit_row["failed_category"] == "fit", f"failed_category is {fit_row['failed_category']}"
    )
    case.check(
        fit_row["status"] == "error", f"five failed folds should be error, not {fit_row['status']}"
    )

    load_row = sweep_with_failing_fit(
        tmp,
        survey_load.LoadFailure("stub/model", [
            {"mode": "online", "ok": False, "error": "RuntimeError: closed", "hub_error": True},
            {"mode": "offline-fallback", "ok": False, "error": "OSError: missing", "hub_error": True},
        ]),
        "fold-load",
    )
    case.check(
        load_row["failed_category"] == "load",
        f"a fit's load failure is categorised {load_row['failed_category']}",
    )

    taxonomy = {
        "load": "load",
        "dtype": "dtype",
        "rowwise": "rowwise",
        "determinism": "determinism",
        "latency_pass": "latency",
        "halves": "halves",
    }
    for stage, expected in taxonomy.items():
        case.check(
            sweep_stages.category_of_stage(stage) == expected,
            f"{stage} maps to {sweep_stages.category_of_stage(stage)}, not {expected}",
        )
    case.done()


def hub_error() -> RuntimeError:
    return RuntimeError(CLIENT_CLOSED)


def load_case(name: str, load_fn, *, offline: bool = False):
    case = Case(name)
    survey_load.reset()
    try:
        value, record = survey_load.load_with_fallback(load_fn, offline=offline, model_id="stub/model")
    except survey_load.LoadFailure as exc:
        case.check(False, f"unexpected LoadFailure: {exc}")
        return case, None
    return case, record


def case_load_online() -> None:
    seen: list[bool] = []

    def load_fn(offline: bool):
        seen.append(offline)
        return {"model": "stub"}

    case, record = load_case("load-online", load_fn)
    case.check(record is not None, "no record")
    if record is None:
        case.done()
        return
    case.check(record["mode"] == "online", f"mode is {record['mode']}")
    case.check(record["forced"] is False, f"forced is {record['forced']}")
    case.check(seen == [False], f"load_fn saw {seen}")
    case.check(len(record["attempts"]) == 1, f"attempts {record['attempts']}")
    case.check(record["status"] == "ok", f"status is {record['status']}")
    case.check(not os.environ.get("HF_HUB_OFFLINE"), "an online load set HF_HUB_OFFLINE")
    case.done()


def case_load_fallback() -> None:
    seen: list[bool] = []

    def load_fn(offline: bool):
        seen.append(offline)
        if not offline:
            raise hub_error()
        return {"model": "cached"}

    case, record = load_case("load-offline-fallback", load_fn)
    case.check(record is not None, "no record")
    if record is None:
        case.done()
        return
    case.check(seen == [False, True], f"load_fn saw {seen}")
    case.check(record["mode"] == "offline-fallback", f"mode is {record['mode']}")
    case.check(record["forced"] is False, f"forced is {record['forced']}")
    case.check(
        record["offline_reason"] and CLIENT_CLOSED in str(record["offline_reason"]),
        f"offline_reason is {record['offline_reason']}",
    )
    case.check(len(record["attempts"]) == 2, f"attempts {record['attempts']}")
    case.check(record["attempts"][0]["hub_error"] is True, "the first failure is not marked a hub error")
    case.check(os.environ.get("HF_HUB_OFFLINE") == "1", "HF_HUB_OFFLINE was not set in-process")
    case.check(os.environ.get("TRANSFORMERS_OFFLINE") == "1", "TRANSFORMERS_OFFLINE was not set")
    try:
        import huggingface_hub.constants as constants

        case.check(bool(constants.HF_HUB_OFFLINE) is True, "the hub constant was not patched")
    except ImportError:
        pass

    seen.clear()

    def later(offline: bool):
        seen.append(offline)
        return {"model": "cached"}

    _, second = survey_load.load_with_fallback(later, model_id="stub/model")
    case.check(seen == [True], f"the switch is not sticky: load_fn saw {seen}")
    case.check(second["mode"] == "offline-fallback", f"second mode is {second['mode']}")
    case.check(second["forced"] is False, "a sticky offline load claims --offline")
    case.check(len(second["attempts"]) == 1, f"second attempts {second['attempts']}")
    survey_load.reset()
    case.check(not os.environ.get("HF_HUB_OFFLINE"), "reset() left HF_HUB_OFFLINE set")
    case.done()


def case_load_forced() -> None:
    seen: list[bool] = []

    def load_fn(offline: bool):
        seen.append(offline)
        return {"model": "cached"}

    case, record = load_case("load-forced-offline", load_fn, offline=True)
    case.check(record is not None, "no record")
    if record is None:
        case.done()
        return
    case.check(seen == [True], f"the online path was attempted: {seen}")
    case.check(record["mode"] == "offline-fallback", f"mode is {record['mode']}")
    case.check(record["forced"] is True, f"forced is {record['forced']}")
    case.check(
        record["offline_reason"] == "--offline", f"reason is {record['offline_reason']}"
    )
    case.check(len(record["attempts"]) == 1, f"attempts {record['attempts']}")
    survey_load.reset()
    case.done()


def case_load_both_fail() -> None:
    case = Case("load-both-fail")
    survey_load.reset()

    def load_fn(offline: bool):
        raise hub_error() if not offline else OSError("no such file in the cache")
    try:
        survey_load.load_with_fallback(load_fn, model_id="stub/model")
    except survey_load.LoadFailure as exc:
        case.check(exc.mode == "offline-fallback", f"mode is {exc.mode}")
        case.check(len(exc.attempts) == 2, f"attempts {exc.attempts}")
        case.check(CLIENT_CLOSED in str(exc), f"the client error was dropped: {exc}")
        case.check("no such file in the cache" in str(exc), f"the retry error was dropped: {exc}")
    else:
        case.check(False, "both attempts failed but LoadFailure was not raised")
    survey_load.reset()

    def dead_hub(model_id: str, offline: bool = False):
        if not offline:
            raise hub_error()
        raise OSError("no such file in the cache")

    with partial.patched(**partial.base_stubs()):
        row = setfit_sweep.sweep(
            partial.args_for(Path(tempfile.mkdtemp()) / "both-fail.json"),
            factory=partial.stub_factory,
            loader=dead_hub,
        )
    survey_load.reset()
    case.check(row["status"] == "error", f"status is {row['status']}")
    case.check(row["failed_stage"] == "load", f"failed_stage is {row['failed_stage']}")
    case.check(row["failed_category"] == "load", f"failed_category is {row['failed_category']}")
    load = row["load"] or {}
    case.check(load.get("mode") == "offline-fallback", f"load.mode is {load.get('mode')}")
    case.check(len(load.get("attempts") or []) == 2, f"load.attempts is {load.get('attempts')}")
    case.check("stub" in (row["error"] or "") or "could not load" in (row["error"] or ""),
               f"error is {row['error']}")
    case.done()


def case_load_not_retried() -> None:
    seen: list[bool] = []

    def load_fn(offline: bool):
        seen.append(offline)
        raise ValueError("this model id is not a model id")

    case = Case("load-not-retried")
    survey_load.reset()
    try:
        survey_load.load_with_fallback(load_fn, model_id="stub/model")
    except survey_load.LoadFailure as exc:
        case.check(len(exc.attempts) == 1, f"a non-hub error was retried: {exc.attempts}")
        case.check(exc.mode == "online", f"mode is {exc.mode}")
        case.check("not a model id" in str(exc), f"the original error was lost: {exc}")
    else:
        case.check(False, "no error was raised")
    case.check(seen == [False], f"load_fn saw {seen}")
    case.check(not survey_load.offline_active(), "a non-hub error switched the process offline")
    case.done()


def case_sweep_offline_flag(tmp: Path) -> None:
    case = Case("sweep-offline-flag")
    seen: list[bool] = []
    survey_load.reset()

    def loader(model_id: str, offline: bool = False):
        seen.append(offline)
        return partial.StubProbe()

    with partial.patched(**partial.base_stubs()):
        row = setfit_sweep.sweep(
            partial.args_for(tmp / "offline.json", offline=True),
            factory=partial.stub_factory,
            loader=loader,
        )
    survey_load.reset()
    case.check(seen and all(seen), f"--offline still tried the network: {seen}")
    case.check(row["status"] == "ok", f"status is {row['status']}")
    load = row["load"] or {}
    case.check(load.get("mode") == "offline-fallback", f"load.mode is {load.get('mode')}")
    case.check(load.get("forced") is True, f"load.forced is {load.get('forced')}")
    case.check(
        (load.get("attempts") or [{}])[0].get("mode") == "offline-fallback",
        f"first attempt is {load.get('attempts')}",
    )
    case.check(
        (load.get("dtype") or {}).get("coerced") is False,
        f"dtype record is {load.get('dtype')}",
    )
    case.check(
        load.get("fit_modes") is None or load["fit_modes"] == [],
        f"a stub fit should report no load mode, got {load.get('fit_modes')}",
    )
    case.done()


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="dtype-offline-test-"))
    print(f"fixtures in {root}\n", flush=True)
    case_bf16_coerced()
    case_fp32_untouched()
    case_cast_refused()
    case_cast_silently_ignored()
    case_embeddings_not_float32()
    case_sweep_dtype_stage_fatal(root)
    case_fold_categories(root)
    case_load_online()
    case_load_fallback()
    case_load_forced()
    case_load_both_fail()
    case_load_not_retried()
    case_sweep_offline_flag(root)

    failures = 0
    checks = 0
    print()
    for name, count, problems in RESULTS:
        checks += count
        failures += len(problems)
        print(f"  {'PASS' if not problems else 'FAIL'} {name} ({count} checks)")
        for problem in problems:
            print(f"       {problem}")
    print(f"\n{len(RESULTS)} cases, {checks} checks, {failures} failures")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
