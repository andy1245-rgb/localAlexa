"""Measurement conditions, host facts and Hub metadata, shared by both survey scripts.

Every survey row has to say what it was measured on. A latency from a box with
another fit running on it, or a memory delta taken with a different allocator,
is not comparable with the row beside it — the ladder learned that the hard way
(`docs/ladder-results-2026-11-10.md`, "Latency was measured twice"). So the
conditions block is built in one place and every script writes the same one.

Nothing here imports torch at module import time: the scripts measure a memory
baseline *before* the heavy imports and would not want this module to spoil it.
"""

from __future__ import annotations

import importlib
import json
import os
import platform
import socket
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def describe_error(exc: BaseException, limit: int = 600) -> str:
    """One line a JSON row can carry instead of a stack trace."""
    text = f"{type(exc).__name__}: {exc}".replace("\n", " ").strip()
    return text[:limit]


def _version(module: str) -> str | None:
    try:
        return str(getattr(importlib.import_module(module), "__version__", "unknown"))
    except Exception:
        return None


def environment() -> dict[str, object]:
    """What this row was measured with. Imports torch, so call after the baseline."""
    import torch

    block: dict[str, object] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "cuda_available": bool(torch.cuda.is_available()),
        "numpy": _version("numpy"),
        "sklearn": _version("sklearn"),
        "transformers": _version("transformers"),
        "sentence_transformers": _version("sentence_transformers"),
        "setfit": _version("setfit"),
        "datasets": _version("datasets"),
        "huggingface_hub": _version("huggingface_hub"),
        "psutil": _version("psutil"),
    }
    # Windows has no load average; saying so is more useful than omitting the key.
    try:
        block["loadavg_1m"] = round(float(os.getloadavg()[0]), 3)
    except (AttributeError, OSError):
        block["loadavg_1m"] = None
    return block


def _windows_working_set() -> int | None:
    """Current working set via psapi, for a Windows box without psutil.

    Untested on Windows from this machine's linux checkout; the linux and
    psutil paths are the ones covered by the survey's own verification runs.
    """
    try:
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.windll.kernel32
        psapi = ctypes.windll.psapi
        ok = psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        )
        return int(counters.WorkingSetSize) if ok else None
    except Exception:
        return None


def rss_bytes() -> tuple[int | None, str]:
    """Current resident bytes of this process, plus a label saying how it was read.

    psutil when it is installed (the same number the ladder's box would show),
    then ``/proc/self/status`` on linux, then psapi on Windows, then
    ``ru_maxrss`` as a last resort — which is a *peak*, not a current reading,
    and the label says so because the difference matters when the delta is read.
    """
    if not os.environ.get("SURVEY_DISABLE_PSUTIL"):
        try:
            import psutil

            return int(psutil.Process().memory_info().rss), "psutil.Process().memory_info().rss (current)"
        except Exception:
            pass
    if sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024, "/proc/self/status VmRSS (current)"
        except OSError:
            pass
    if sys.platform == "win32":
        value = _windows_working_set()
        if value is not None:
            return value, "psapi GetProcessMemoryInfo WorkingSetSize (current)"
    try:
        import resource

        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if sys.platform == "darwin":
            return peak, "ru_maxrss (bytes, peak not current)"
        return peak * 1024, "ru_maxrss (kilobytes, peak not current)"
    except Exception:
        return None, "unavailable: no psutil, no /proc, no psapi, no resource"


def memory_block(readings: list[tuple[str, int | None, str]]) -> dict[str, object]:
    """``[(label, bytes, method)]`` in measurement order, plus the deltas that matter."""
    block: dict[str, object] = {"readings": {}}
    methods = []
    for label, value, method in readings:
        block["readings"][label] = value
        if method not in methods:
            methods.append(method)
    block["method"] = "; ".join(methods) if methods else "unknown"
    ordered = [(label, value) for label, value, _ in readings]
    deltas: dict[str, int | None] = {}
    for (prev_label, prev), (next_label, nxt) in zip(ordered, ordered[1:]):
        deltas[f"{next_label}_minus_{prev_label}"] = (
            None if prev is None or nxt is None else int(nxt - prev)
        )
    block["deltas_bytes"] = deltas
    return block


def _cache_root() -> Path:
    home = os.environ.get("HF_HOME")
    if home:
        return Path(home).expanduser() / "hub"
    hf_hub_cache = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if hf_hub_cache:
        return Path(hf_hub_cache).expanduser()
    return Path.home() / ".cache" / "huggingface" / "hub"


def _tree_size(root: Path) -> tuple[int, int]:
    """Bytes and file count of a snapshot, following its symlinks into the blobs.

    Hard-linked or symlinked duplicates of one blob are counted once, by inode.
    """
    total = 0
    files = 0
    seen: set[tuple[int, int]] = set()
    for dirpath, _dirnames, filenames in os.walk(root, followlinks=False):
        for name in filenames:
            path = Path(dirpath) / name
            try:
                stat = path.stat()
            except OSError:
                continue
            key = (stat.st_dev, stat.st_ino)
            if key in seen:
                continue
            seen.add(key)
            total += stat.st_size
            files += 1
    return total, files


def snapshot(model_id: str) -> dict[str, object]:
    """Where the downloaded snapshot is on disk, and how big it is."""
    block: dict[str, object] = {
        "path": None,
        "bytes": None,
        "files": None,
        "source": None,
        "error": None,
    }
    path: Path | None = None
    try:
        from huggingface_hub import snapshot_download

        path = Path(snapshot_download(model_id, local_files_only=True))
        block["source"] = "huggingface_hub.snapshot_download(local_files_only=True)"
    except Exception as exc:
        block["error"] = describe_error(exc)
        cache = _cache_root() / ("models--" + model_id.replace("/", "--")) / "snapshots"
        try:
            candidates = sorted(p for p in cache.iterdir() if p.is_dir())
        except OSError:
            candidates = []
        if candidates:
            path = candidates[-1]
            block["source"] = "hf cache scan of " + str(cache)
    if path is None:
        return block
    block["path"] = str(path)
    try:
        total, files = _tree_size(path)
        block["bytes"] = total
        block["files"] = files
    except OSError as exc:
        block["error"] = describe_error(exc)
    return block


def _license_from_readme(model_id: str) -> str | None:
    """Last-resort licence read: the snapshot's own model card front matter."""
    block = snapshot(model_id)
    if not block.get("path"):
        return None
    readme = Path(str(block["path"])) / "README.md"
    try:
        lines = readme.read_text(encoding="utf-8", errors="replace").splitlines()[:60]
    except OSError:
        return None
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("license:"):
            value = stripped.split(":", 1)[1].strip().strip("'\"")
            return value or None
    return None


def model_card(model_id: str) -> dict[str, object]:
    """Licence and card facts, hub first, local snapshot second, ``None`` third.

    The hub call needs network; a sweep run offline still gets the licence from
    the snapshot it already has, and ``license_source`` says which one answered
    so nobody has to guess whether the row was measured online.
    """
    card: dict[str, object] = {
        "license": None,
        "license_source": None,
        "pipeline_tag": None,
        "gated": None,
        "downloads": None,
        "tags": [],
        "error": None,
    }
    try:
        from huggingface_hub import model_info

        info = model_info(model_id)
        data = getattr(info, "cardData", None) or {}
        tags = [str(t) for t in (getattr(info, "tags", None) or [])]
        card["pipeline_tag"] = getattr(info, "pipeline_tag", None)
        card["gated"] = getattr(info, "gated", None)
        card["downloads"] = getattr(info, "downloads", None)
        card["tags"] = tags[:25]
        license_value = data.get("license") or data.get("license_name") or data.get("licence")
        if isinstance(license_value, (list, tuple)):
            license_value = ", ".join(str(v) for v in license_value)
        if license_value:
            card["license"] = str(license_value)
            card["license_source"] = "huggingface_hub.model_info cardData"
        else:
            for tag in tags:
                if tag.startswith("license:"):
                    card["license"] = tag.split(":", 1)[1]
                    card["license_source"] = "huggingface_hub.model_info tags"
                    break
    except Exception as exc:
        card["error"] = describe_error(exc)
    if card["license"] is None:
        local = _license_from_readme(model_id)
        if local:
            card["license"] = local
            card["license_source"] = "snapshot README front matter (offline fallback)"
    return card


def _published_mode(target: Path) -> int:
    """The mode the published file should carry: 0666 minus the umask, or the one it had."""
    try:
        return target.stat().st_mode & 0o777
    except OSError:
        umask = os.umask(0)
        os.umask(umask)
        return 0o666 & ~umask


def write_json(payload: dict[str, object], path: str | Path) -> Path:
    """Write the artefact atomically, as UTF-8 with LF endings, on every platform.

    The temporary file is created in the target's own directory so the rename
    that publishes it never crosses a volume — a rename across volumes is a
    different errno on Windows and fails outright on some network paths — and
    so the published file is either the previous artefact or the new one, never
    half of the new one. ``mkstemp`` makes the staged file 0600, so the mode is
    set explicitly before it is published; ``newline="\\n"`` keeps a Windows
    artefact byte-identical to the Linux one it is compared against.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2) + "\n"
    mode = _published_mode(target)
    handle, temporary = tempfile.mkstemp(
        dir=str(target.parent), prefix=target.name + ".", suffix=".tmp"
    )
    staged = Path(temporary)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
        os.chmod(staged, mode)
        os.replace(staged, target)
    except BaseException:
        staged.unlink(missing_ok=True)
        raise
    return target


def model_slug(model_id: str) -> str:
    """``BAAI/bge-small-en-v1.5`` -> ``BAAI__bge-small-en-v1.5``, for filenames."""
    return model_id.replace("/", "__").replace("\\", "__")


def latency_passes(encoder, texts: list[str], passes: int, warmup: int = 8) -> dict[str, object]:
    """Per-sentence latency the way ``experiments/ladder/latency_pass.py`` measured it.

    One utterance at a time, warm, one process, ``passes`` repeats, and the
    reported number is the median of the per-pass medians — not the mean over
    every call, which a single outlier can move and a single cold call can ruin.
    """
    for text in texts[:warmup]:
        encoder([text])
    pass_p50: list[float] = []
    pass_p95: list[float] = []
    for _ in range(passes):
        values = []
        for text in texts:
            started = time.perf_counter()
            encoder([text])
            values.append((time.perf_counter() - started) * 1000.0)
        values.sort()
        pass_p50.append(values[len(values) // 2])
        pass_p95.append(values[int(len(values) * 0.95)])
    ordered = sorted(pass_p50)
    ordered95 = sorted(pass_p95)
    return {
        "passes": passes,
        "n_sentences_per_pass": len(texts),
        "warmup_calls": min(warmup, len(texts)),
        "pass_p50_ms": [round(v, 3) for v in pass_p50],
        "pass_p95_ms": [round(v, 3) for v in pass_p95],
        "p50_ms": round(ordered[len(ordered) // 2], 3),
        "p95_ms": round(ordered95[len(ordered95) // 2], 3),
        "p50_spread_ms": round(max(pass_p50) - min(pass_p50), 3),
        "method": "single utterance per call, warm, one process, median of the per-pass medians",
    }
