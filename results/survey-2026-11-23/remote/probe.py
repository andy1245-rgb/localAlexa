"""Smoke test: can this box reach the Hub and run a sentence-transformer on CPU?

Contract: stdout is exactly one JSON line. Human-readable detail goes to stderr.
Not a benchmark - it answers "does it load and embed at all", nothing more.
"""
from __future__ import annotations

import ctypes
import importlib.metadata as md
import json
import os
import platform
import statistics
import struct
import sys
import time
from pathlib import Path

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
N_SENTENCES = 32
N_PASSES = 5

SENTENCES = [
    "turn on the kitchen lights",
    "set a timer for ten minutes",
    "what is the weather like tomorrow",
    "play some jazz in the living room",
    "remind me to call my sister",
    "how many cups are in a quart",
    "lock the front door",
    "what time is it in tokyo",
    "add milk to my shopping list",
    "dim the bedroom lamp to thirty percent",
    "who won the game last night",
    "start the coffee maker",
    "what is the capital of portugal",
    "set an alarm for six thirty",
    "turn the volume up a bit",
    "is it going to rain this afternoon",
    "find a recipe for banana bread",
    "pause the music please",
    "how far away is the moon",
    "open the garage door",
    "remind me about the meeting at noon",
    "what is two hundred divided by eight",
    "switch off the porch light",
    "tell me a joke",
    "how do you spell necessary",
    "turn down the thermostat two degrees",
    "what is on my calendar today",
    "play the next track",
    "how tall is mount everest",
    "set the living room to movie mode",
    "what is the meaning of serendipity",
    "mute the television",
]


def ram_gb() -> float | None:
    try:
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return round(stat.ullTotalPhys / 1024**3, 2)
    except Exception:
        pass
    return None


def rss_mb() -> float | None:
    try:
        import psutil  # noqa: PLC0415

        return round(psutil.Process().memory_info().rss / 1024**2, 1)
    except Exception:
        return None


def version(dist: str) -> str | None:
    try:
        return md.version(dist)
    except Exception:
        return None


def model_cache_dir() -> Path | None:
    cache = os.environ.get("HF_HUB_CACHE")
    if not cache:
        hf_home = os.environ.get("HF_HOME")
        base = Path(hf_home) if hf_home else Path.home() / ".cache" / "huggingface"
        cache = str(base / "hub")
    d = Path(cache) / ("models--" + MODEL_ID.replace("/", "--"))
    return d if d.is_dir() else None


def resolved_snapshot(repo_dir: Path | None) -> Path | None:
    """Follow refs/main to the snapshot the loader actually used."""
    if repo_dir is None:
        return None
    ref = repo_dir / "refs" / "main"
    if ref.is_file():
        snap = repo_dir / "snapshots" / ref.read_text(encoding="utf-8").strip()
        if snap.is_dir():
            return snap
    snaps = sorted(p for p in (repo_dir / "snapshots").glob("*") if p.is_dir())
    return snaps[-1] if snaps else None


def dir_bytes(root: Path, skip: tuple[str, ...] = ()) -> int:
    total = 0
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(root).parts
        if rel and rel[0] in skip:
            continue
        total += p.stat().st_size
    return total


def emit(payload: dict) -> None:
    print(json.dumps(payload, separators=(",", ":"), sort_keys=True))


def log(line: str) -> None:
    print(line, file=sys.stderr, flush=True)


def main() -> int:
    facts = {
        "python": sys.version.split()[0],
        "python_impl": platform.python_implementation(),
        "arch": platform.machine(),
        "bits": 8 * struct.calcsize("P"),
        "executable": sys.executable,
        "core_count": os.cpu_count(),
        "ram_gb": ram_gb(),
        "torch": version("torch"),
        "transformers": version("transformers"),
        "sentence_transformers": version("sentence-transformers"),
        "setfit": version("setfit"),
        "scikit_learn": version("scikit-learn"),
        "numpy": version("numpy"),
        "huggingface_hub": version("huggingface-hub"),
    }

    log(f"python                {facts['python']} ({facts['python_impl']}, {facts['arch']})")
    log(f"executable            {facts['executable']}")
    log(f"cores / ram           {facts['core_count']} / {facts['ram_gb']} GB")
    for k in ("torch", "transformers", "sentence_transformers", "setfit",
              "scikit_learn", "numpy", "huggingface_hub"):
        log(f"{k:<21} {facts[k]}")

    import torch  # noqa: PLC0415
    from sentence_transformers import SentenceTransformer  # noqa: PLC0415

    facts["torch_threads"] = torch.get_num_threads()
    facts["torch_cuda"] = bool(torch.cuda.is_available())
    log(f"torch threads / cuda  {facts['torch_threads']} / {facts['torch_cuda']}")

    t0 = time.perf_counter()
    model = SentenceTransformer(MODEL_ID, device="cpu")
    load_s = time.perf_counter() - t0
    dim_fn = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
    dim = int(dim_fn())
    log(f"load {MODEL_ID}  {load_s * 1000:.0f} ms, dim={dim}")

    t1 = time.perf_counter()
    model.encode(["warm up the encoder"], convert_to_numpy=True)
    first_ms = (time.perf_counter() - t1) * 1000

    per_sentence_ms: list[float] = []
    for _ in range(N_PASSES):
        for s in SENTENCES:
            t = time.perf_counter()
            model.encode([s], convert_to_numpy=True)
            per_sentence_ms.append((time.perf_counter() - t) * 1000)

    median_ms = statistics.median(per_sentence_ms)
    ordered = sorted(per_sentence_ms)
    p95_ms = ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]

    log(f"encodes               {len(per_sentence_ms)} single-sentence calls "
        f"({N_SENTENCES} sentences x {N_PASSES} passes)")
    log(f"latency per sentence  median {median_ms:.3f} ms, p95 {p95_ms:.3f} ms, "
        f"first warm call {first_ms:.0f} ms")

    repo_dir = model_cache_dir()
    snap = resolved_snapshot(repo_dir)
    snapshot_bytes = reduced_bytes = None
    extra_dirs: list[str] = []
    if snap is not None:
        extra_dirs = sorted(p.name for p in snap.iterdir() if p.is_dir())
        snapshot_bytes = dir_bytes(snap)
        reduced_bytes = dir_bytes(snap, skip=("onnx", "openvino"))
        log(f"snapshot {snap.name[:12]}  {snapshot_bytes / 1024**2:.1f} MB total, "
            f"{reduced_bytes / 1024**2:.1f} MB excluding {extra_dirs or 'nothing'}")
    if repo_dir is not None:
        log(f"hf repo cache dir     {dir_bytes(repo_dir) / 1024**2:.1f} MB at {repo_dir}")

    emit({
        **facts,
        "model": MODEL_ID,
        "encoding": "utf-8",
        "embed_dim": dim,
        "sentences": N_SENTENCES,
        "passes": N_PASSES,
        "encodes": len(per_sentence_ms),
        "load_ms": round(load_s * 1000, 1),
        "first_call_ms": round(first_ms, 3),
        "median_ms_per_sentence": round(median_ms, 4),
        "p95_ms_per_sentence": round(p95_ms, 4),
        "model_size_bytes": reduced_bytes,
        "snapshot_bytes": snapshot_bytes,
        "snapshot_extra_dirs": extra_dirs,
        "rss_mb": rss_mb(),
        "ok": True,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
