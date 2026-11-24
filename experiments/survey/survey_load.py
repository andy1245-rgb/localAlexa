"""Online first, cache second: a Hub blink must not cost a sweep.

The failure this exists for, reproduced on the target box: four models in one
run each died at ``failed_stage: load`` with

    RuntimeError: Cannot send a request, as the client has been closed.

seconds apart, while ``huggingface.co:443`` was reachable, and every one of those
models was already in the local HF cache. The client that died is process-wide
(``httpx`` raises that ``RuntimeError`` for a connection pool that was closed
under it), so every load after the first failure fails the same way — a network
blink is amplified into a dead sweep. A cached model must never fail a sweep
because the network blinked.

So: try online; if the load raises a Hub-shaped error, set ``HF_HUB_OFFLINE=1``
and ``TRANSFORMERS_OFFLINE=1`` and retry once, exactly equal to the cached
artefacts that are already on disk. The path taken is recorded as
``mode = "online" | "offline-fallback"`` so a row never has to be guessed at.

Two details that are easy to get wrong:

* **The environment variable alone does nothing after the import.**
  ``huggingface_hub.constants.HF_HUB_OFFLINE`` is computed once, at module
  import, from the environment; writing ``os.environ`` afterwards changes
  nothing for a process that has already imported it — and by the time a model
  is being loaded, it has. Both are set, and the constant is patched.
* **The switch is sticky for the rest of the process.** Once the pool has been
  closed, an online attempt per fit is four more failures, not four more
  chances. A fallback therefore also covers every later load in the process, and
  each caller passes ``local_files_only`` down to the loader so the cached path
  holds even if a future Hub version stops reading the constant.

``reset()`` exists for the tests and is never called by the harness: within one
process the mode only ever moves forward.
"""

from __future__ import annotations

import os
import time

from survey_env import describe_error

OFFLINE_ENV = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
MODE_ONLINE = "online"
MODE_OFFLINE_FALLBACK = "offline-fallback"

#: Exception class names that mean "the Hub call failed", from the libraries that
#: stand between this harness and the cache. httpx's ``RuntimeError`` is a
#: different class from the builtin one (it subclasses ``Exception`` directly),
#: which is why this is a name set rather than an ``isinstance`` check against
#: three types: the recorded error for the live failure is ``RuntimeError:
#: Cannot send a request, as the client has been closed.``
HUB_ERROR_TYPES = frozenset(
    {
        "RuntimeError",
        "ConnectionError",
        "OSError",  # local_file/module miss, and what a broken socket surfaces as
        "TimeoutError",
        "HTTPError",
        "RemoteProtocolError",
        "ConnectError",
        "ConnectTimeout",
        "ReadTimeout",
        "WriteTimeout",
        "ProxyError",
        "TooManyRedirects",
        "HfHubHTTPError",
        "OfflineModeIsEnabled",
        "LocalEntryNotFoundError",
        "LocalTokenNotFoundError",
        "EntryNotFoundError",
        "RepositoryNotFoundError",
    }
)
HUB_MODULES = ("httpx", "httpcore", "huggingface_hub", "requests", "urllib3", "aiohttp")

_sticky: dict[str, object] = {"offline": False, "reason": None, "saved_env": {}, "saved": None}


class LoadFailure(RuntimeError):
    """Every attempt failed. Carries both attempts so the row can name them."""

    def __init__(self, model_id: str, attempts: list[dict[str, object]]) -> None:
        self.model_id = model_id
        self.attempts = list(attempts)
        self.mode = str(attempts[-1]["mode"]) if attempts else MODE_ONLINE
        detail = "; ".join(f"{a['mode']}: {a['error']}" for a in attempts)
        super().__init__(f"could not load {model_id!r} ({detail})")


def is_hub_error(exc: BaseException) -> bool:
    """Is this the kind of failure a cached copy can survive?"""
    for cls in type(exc).__mro__:
        if cls.__name__ in HUB_ERROR_TYPES:
            return True
    module = type(exc).__module__ or ""
    return any(module == m or module.startswith(m + ".") for m in HUB_MODULES)


def offline_active() -> bool:
    """Has this process already switched to the cache?"""
    return bool(_sticky["offline"])


def offline_reason() -> str | None:
    return _sticky["reason"]  # type: ignore[return-value]


def _patch_constant(value: bool) -> object:
    """Patch ``huggingface_hub.constants.HF_HUB_OFFLINE``; returns what to restore.

    ``transformers.utils.hub`` imports ``is_offline_mode`` from the Hub, and that
    function reads this module global at call time, so patching the constant is
    what reaches both libraries in one place. Returns ``None`` when the Hub is
    not importable, which is not an error: ``load_with_fallback`` still sets the
    environment, and the loader still passes ``local_files_only``.
    """
    try:
        import huggingface_hub.constants as constants
    except Exception:
        return None
    saved = getattr(constants, "HF_HUB_OFFLINE", None)
    try:
        constants.HF_HUB_OFFLINE = value
    except Exception:
        return None
    return ("huggingface_hub.constants.HF_HUB_OFFLINE", saved)


def enable_offline(reason: str) -> dict[str, object]:
    """Switch this process to the cache, in the environment and in the libraries."""
    if not offline_active():
        _sticky["saved_env"] = {name: os.environ.get(name) for name in OFFLINE_ENV}
        _sticky["saved"] = _patch_constant(True)
        _sticky["offline"] = True
        _sticky["reason"] = reason
    for name in OFFLINE_ENV:
        os.environ[name] = "1"
    return {
        "offline": True,
        "reason": _sticky["reason"],
        "environment": list(OFFLINE_ENV),
        "patched": None if _sticky["saved"] is None else _sticky["saved"][0],
    }


def reset() -> None:
    """Undo :func:`enable_offline`. Tests only — the harness never walks this back."""
    saved_env = _sticky.get("saved_env") or {}
    for name in OFFLINE_ENV:
        previous = saved_env.get(name)  # type: ignore[union-attr]
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous
    saved = _sticky.get("saved")
    if saved is not None:
        try:
            import huggingface_hub.constants as constants

            constants.HF_HUB_OFFLINE = saved[1]
        except Exception:
            pass
    _sticky.update({"offline": False, "reason": None, "saved_env": {}, "saved": None})


def _attempt(
    record: dict[str, object], mode: str, load_fn, go_offline: bool
) -> object:
    started = time.perf_counter()
    try:
        value = load_fn(go_offline)
    except BaseException as exc:  # recorded, then re-raised by the caller
        record["attempts"].append(  # type: ignore[union-attr]
            {
                "mode": mode,
                "ok": False,
                "exception_type": type(exc).__name__,
                "error": describe_error(exc),
                "hub_error": is_hub_error(exc),
                "seconds": round(time.perf_counter() - started, 3),
            }
        )
        raise
    record["attempts"].append(  # type: ignore[union-attr]
        {
            "mode": mode,
            "ok": True,
            "error": None,
            "seconds": round(time.perf_counter() - started, 3),
        }
    )
    return value


def load_with_fallback(
    load_fn, *, offline: bool = False, model_id: str = ""
) -> tuple[object, dict[str, object]]:
    """``load_fn(offline: bool)`` online, then once from the cache if that failed.

    Returns the loader's value and the record that belongs in the artefact:
    ``mode`` (which path produced the model), ``forced`` (``--offline``),
    ``offline_reason`` and the per-attempt log. Every failure comes out as
    :class:`LoadFailure`, carrying the attempts that were made and the original
    error text in each, so the artefact can name the path as well as the error.
    A non-Hub error is not retried — retrying a bad config or a bad id from the
    cache costs a second and hides the real message — but it is still a
    ``LoadFailure``, with one attempt in it.
    """
    record: dict[str, object] = {
        "status": "pending",
        "model_id": model_id,
        "mode": None,
        "forced": bool(offline),
        "offline_reason": None,
        "attempts": [],
    }
    if offline:
        enable_offline("--offline")
    record["offline_reason"] = offline_reason()

    first_offline = bool(offline or offline_active())
    first_mode = MODE_OFFLINE_FALLBACK if first_offline else MODE_ONLINE
    try:
        value = _attempt(record, first_mode, load_fn, first_offline)
    except BaseException as exc:
        if first_offline or not is_hub_error(exc):
            record["mode"] = first_mode
            record["status"] = "error"
            raise LoadFailure(model_id, record["attempts"]) from exc  # type: ignore[arg-type]
        reason = describe_error(exc)
        enable_offline(reason)
        record["offline_reason"] = reason
        try:
            value = _attempt(record, MODE_OFFLINE_FALLBACK, load_fn, True)
        except BaseException as retry_exc:
            record["mode"] = MODE_OFFLINE_FALLBACK
            record["status"] = "error"
            raise LoadFailure(model_id, record["attempts"]) from retry_exc  # type: ignore[arg-type]
        record["mode"] = MODE_OFFLINE_FALLBACK
        record["status"] = "ok"
        return value, record
    record["mode"] = first_mode
    record["status"] = "ok"
    return value, record
