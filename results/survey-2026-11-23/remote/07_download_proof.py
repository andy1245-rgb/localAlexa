"""Prove the download leg of the smoke test, which the MiniLM run did not exercise.

MiniLM-L6-v2 was already resident in this box's HF cache (snapshot created 2026-09-21),
so loading it only proved Hub metadata reachability. This pulls a sentence-transformer
that is provably absent from the cache, from the Hub, and embeds with it.

stdout: one JSON line. stderr: human-readable detail.
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path

MODEL_ID = "sentence-transformers/paraphrase-MiniLM-L3-v2"
N_SENTENCES = 32
N_PASSES = 3

SENTENCES = [
    "turn on the kitchen lights", "set a timer for ten minutes",
    "what is the weather like tomorrow", "play some jazz in the living room",
    "remind me to call my sister", "how many cups are in a quart",
    "lock the front door", "what time is it in tokyo",
    "add milk to my shopping list", "dim the bedroom lamp to thirty percent",
    "who won the game last night", "start the coffee maker",
    "what is the capital of portugal", "set an alarm for six thirty",
    "turn the volume up a bit", "is it going to rain this afternoon",
    "find a recipe for banana bread", "pause the music please",
    "how far away is the moon", "open the garage door",
    "remind me about the meeting at noon", "what is two hundred divided by eight",
    "switch off the porch light", "tell me a joke",
    "how do you spell necessary", "turn down the thermostat two degrees",
    "what is on my calendar today", "play the next track",
    "how tall is mount everest", "set the living room to movie mode",
    "what is the meaning of serendipity", "mute the television",
]


def hub_cache() -> Path:
    cache = os.environ.get("HF_HUB_CACHE")
    if cache:
        return Path(cache)
    hf_home = os.environ.get("HF_HOME")
    base = Path(hf_home) if hf_home else Path.home() / ".cache" / "huggingface"
    return base / "hub"


def log(line: str) -> None:
    print(line, file=sys.stderr, flush=True)


def repo_dir_bytes(repo: Path) -> int:
    return sum(p.stat().st_size for p in repo.rglob("*") if p.is_file())


def main() -> int:
    cache = hub_cache()
    target = cache / ("models--" + MODEL_ID.replace("/", "--"))

    cached_before = sorted(
        p.name.removeprefix("models--").replace("--", "/")
        for p in cache.glob("models--*")
        if p.is_dir()
    )
    log(f"hub cache            {cache}")
    log(f"repos present before {len(cached_before)}: {', '.join(cached_before)}")

    if target.is_dir():
        log(f"ABORT: {MODEL_ID} is already cached - this would not prove a download")
        print(json.dumps({"model": MODEL_ID, "ok": False,
                          "reason": "already cached, download not exercised"}))
        return 2

    import torch  # noqa: PLC0415
    from sentence_transformers import SentenceTransformer  # noqa: PLC0415

    t0 = time.perf_counter()
    model = SentenceTransformer(MODEL_ID, device="cpu")
    download_and_load_s = time.perf_counter() - t0

    dim_fn = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
    dim = int(dim_fn())
    on_disk = repo_dir_bytes(target) if target.is_dir() else None
    log(f"downloaded+loaded    {download_and_load_s:.1f} s, dim={dim}, "
        f"on disk {on_disk / 1024**2:.1f} MB")

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
    log(f"latency              median {median_ms:.3f} ms per single-sentence encode "
        f"({len(per_sentence_ms)} calls)")

    print(json.dumps({
        "model": MODEL_ID,
        "hub_cache": str(cache),
        "repos_present_before": len(cached_before),
        "download_and_load_s": round(download_and_load_s, 1),
        "on_disk_bytes": on_disk,
        "embed_dim": dim,
        "torch": torch.__version__,
        "encodes": len(per_sentence_ms),
        "first_call_ms": round(first_ms, 3),
        "median_ms_per_sentence": round(median_ms, 4),
        "downloaded_from_hub": True,
        "ok": True,
    }, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
