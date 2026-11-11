"""Latency, measured on an idle box, five passes, median of the per-pass p50.

The ladder's latency column is the one number a busy machine can wreck: every
rung's per-utterance time was taken while another rung was training on all eight
cores, and the baseline drifted between 0.34 ms and 1.68 ms depending on what
else was running. This pass runs alone, repeats, and reports the median so the
table in the doc is a measurement rather than a sample of the load.

    .venv-embed/bin/python experiments/ladder/latency_pass.py

Writes ``results/ladder/latency.json``.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from localalexa.data import load_registry, load_routing_set
from localalexa.ladder.pools import reference_pool

PASSES = 5


def measure(rung, texts, pool) -> dict[str, object]:
    predictor = rung.build().fit([t for t, _ in pool], [l for _, l in pool])
    predictor.scores(texts[:1])  # warm

    pass_p50, pass_p95 = [], []
    for _ in range(PASSES):
        latencies = []
        for text in texts:
            started = time.perf_counter()
            predictor.scores([text])
            latencies.append((time.perf_counter() - started) * 1000.0)
        latencies.sort()
        pass_p50.append(latencies[len(latencies) // 2])
        pass_p95.append(latencies[int(len(latencies) * 0.95)])
    return {
        "rung": rung.name,
        "pass_p50_ms": [round(v, 3) for v in pass_p50],
        "pass_p95_ms": [round(v, 3) for v in pass_p95],
        "p50_ms": round(statistics.median(pass_p50), 3),
        "p95_ms": round(statistics.median(pass_p95), 3),
        "p50_spread_ms": round(max(pass_p50) - min(pass_p50), 3),
    }


def main() -> int:
    from localalexa.ladder.catalog import available_rungs

    registry = load_registry()
    texts = [row.text for row in load_routing_set()]
    pool = reference_pool(registry)

    wanted = sys.argv[1:] or ["r0-word-svc", "r2-word-logreg-C100", "r3-setfit-minilm", "r4-crossencoder-top3"]
    rungs = available_rungs(",".join(wanted))

    rows = []
    for rung in rungs:
        block = measure(rung, texts, pool)
        rows.append(block)
        print(
            f"{block['rung']:24s} p50={block['p50_ms']:7.3f} ms (spread {block['p50_spread_ms']:.3f}) "
            f"p95={block['p95_ms']:7.3f} ms",
            flush=True,
        )

    out = Path(__file__).resolve().parents[2] / "results" / "ladder" / "latency.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {"passes": PASSES, "n_rows": len(texts), "protocol": "single utterance, warm, idle box", "rungs": rows},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
