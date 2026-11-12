"""What logging costs the request path, measured rather than asserted.

The brief asks for a file append and says that if it costs more than a few
microseconds the number should be reported. This measures the append alone —
``os.write`` of one already-formatted line — and the whole tap call, so the answer
distinguishes "the write is cheap" from "building the record is not".

``examples/collect_bench.py`` prints it; ``tests/test_collection.py`` fails if the
append leaves its budget.
"""

from __future__ import annotations

import json
import statistics
import tempfile
import time
import uuid
from pathlib import Path

from .log import RequestLog
from .records import Candidate, RequestRecord

APPEND_BUDGET_US = 50.0


def _record(i: int) -> RequestRecord:
    return RequestRecord(
        id=f"{i:016x}",
        ts=time.time(),
        session="bench",
        text="turn on the kitchen lights",
        tier=1,
        label="light_set",
        confidence=0.72,
        top_sim=0.61,
        margin=0.09,
        candidates=(
            Candidate("light_set", 0.51),
            Candidate("light_query", 0.31),
            Candidate("thermostat_set", 0.18),
        ),
        gate=False,
        sim_threshold=0.45,
        conf_threshold=0.35,
    )


def measure_append(iterations: int = 2000) -> dict[str, float]:
    """Microseconds per append: the hot path, isolated."""
    with tempfile.TemporaryDirectory() as tmp:
        log = RequestLog(Path(tmp) / "requests.jsonl", session="bench")
        try:
            payloads = [json.dumps(_record(i).as_dict()) for i in range(iterations)]
            samples = []
            for payload in payloads:
                started = time.perf_counter_ns()
                log.write_line(log._fd, payload + "\n")
                samples.append((time.perf_counter_ns() - started) / 1000.0)
        finally:
            log.close()
    return _stats(samples)


def measure_tap(iterations: int = 500) -> dict[str, float]:
    """Microseconds per record build + append, without a router in the way."""
    from .tap import CollectionTap

    with tempfile.TemporaryDirectory() as tmp:
        log = RequestLog(Path(tmp) / "requests.jsonl", session="bench")
        tap = CollectionTap(log, id_factory=lambda: uuid.uuid4().hex[:16])
        try:
            samples = []
            for i in range(iterations):
                started = time.perf_counter_ns()
                tap.observe(f"turn on the kitchen lights {i}", None)
                samples.append((time.perf_counter_ns() - started) / 1000.0)
        finally:
            log.close()
    return _stats(samples)


def _stats(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    return {
        "n": float(len(ordered)),
        "p50_us": round(statistics.median(ordered), 3),
        "p95_us": round(ordered[int(len(ordered) * 0.95)], 3),
        "p99_us": round(ordered[min(int(len(ordered) * 0.99), len(ordered) - 1)], 3),
        "max_us": round(ordered[-1], 3),
    }
