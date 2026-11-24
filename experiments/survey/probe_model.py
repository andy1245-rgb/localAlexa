"""Phase 3: can this encoder load, embed and answer inside the budget on this box?

No fine-tuning. Load the model through ``sentence-transformers``, embed all 382
rows of the routing set, and report the four things that decide whether it is
worth a phase-4 fit at all: what the model is (params, dim, max length,
licence, bytes on disk), what it costs resident, what it costs per utterance,
and whether it loaded in the first place.

    .venv-embed/bin/python experiments/survey/probe_model.py \
        --model sentence-transformers/all-MiniLM-L6-v2 \
        --out results/survey-2026-11-23/probe-minilm-l6.json

Latency follows ``experiments/ladder/latency_pass.py``: one utterance per call,
warm, one process, repeat passes, median of the per-pass medians. The number is
only meaningful if nothing else is training on the box, so the row records
``environment.loadavg_1m`` and ``torch_threads`` beside it.

A model that cannot be downloaded or loaded still writes a row. The JSON carries
``status: "error"`` and the error text; the process exits 1 and never prints a
stack trace over the artifact. That is the difference between a sweep that
records "this one is out" and a sweep that has to be re-run by hand.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from survey_env import (
    describe_error,
    environment,
    iso_now,
    latency_passes,
    memory_block,
    model_card,
    rss_bytes,
    snapshot,
    write_json,
)
from survey_dtype import coerce_float32
from survey_load import MODE_OFFLINE_FALLBACK, MODE_ONLINE, load_with_fallback, offline_active

SCHEMA = "survey-probe/1"
DEFAULT_PASSES = 5
BATCH_SIZE = 32


def probe(
    model_id: str,
    *,
    passes: int = DEFAULT_PASSES,
    limit: int | None = None,
    batch_size: int = BATCH_SIZE,
    warmup: int = 8,
    offline: bool = False,
) -> dict[str, object]:
    readings: list[tuple[str, int | None, str]] = []

    def sample(label: str) -> None:
        value, method = rss_bytes()
        readings.append((label, value, method))

    sample("start")
    row: dict[str, object] = {
        "schema": SCHEMA,
        "phase": 3,
        "model_id": model_id,
        "status": "ok",
        "error": None,
        "failed_stage": None,
        "measured_at": iso_now(),
        "environment": None,
        "corpus": None,
        "load": {"status": "not_attempted", "load_ms": None, "import_ms": None, "error": None},
        "model": None,
        "embeddings": None,
        "memory": None,
        "latency": None,
        "conditions": {
            "protocol": "single utterance per call, warm, one process, nothing else started by this harness",
            "passes_requested": passes,
            "warmup_calls": warmup,
            "batch_size_single_call": 1,
            "embed_batch_size": batch_size,
            "latency_source": "same shape as experiments/ladder/latency_pass.py",
        },
        "stage_errors": {},
    }

    # Imported before anything else, because `environment()` reads every package's
    # __version__ — which would import sentence-transformers and make this 0 ms.
    import_started = time.perf_counter()
    try:
        from sentence_transformers import SentenceTransformer

        import_ms = (time.perf_counter() - import_started) * 1000.0
    except Exception as exc:
        row["status"] = "error"
        row["failed_stage"] = "import"
        row["error"] = describe_error(exc)
        row["load"] = {
            "status": "error",
            "load_ms": None,
            "import_ms": None,
            "error": describe_error(exc),
        }
        row["memory"] = memory_block(readings)
        return row
    sample("after_import")

    try:
        row["environment"] = environment()
    except Exception as exc:
        row["stage_errors"]["environment"] = describe_error(exc)

    from localalexa.data import load_routing_set

    rows = load_routing_set()
    texts = [row_.text for row_ in rows]
    if limit is not None:
        texts = texts[:limit]
    row["corpus"] = {
        "source": "localalexa.data.load_routing_set (<repo>/data/routing_382.json)",
        "n_rows": len(texts),
        "truncated": limit is not None,
    }

    sample("before_load")
    load_started = time.perf_counter()
    try:
        model, record = load_with_fallback(
            lambda offline: SentenceTransformer(model_id, local_files_only=offline),
            offline=offline,
            model_id=model_id,
        )
    except Exception as exc:
        row["status"] = "error"
        row["failed_stage"] = "load"
        row["error"] = describe_error(exc)
        row["load"] = {
            "status": "error",
            "load_ms": round((time.perf_counter() - load_started) * 1000.0, 1),
            "import_ms": round(import_ms, 1),
            "error": describe_error(exc),
            "mode": getattr(exc, "mode", None)
            or (MODE_OFFLINE_FALLBACK if offline_active() else MODE_ONLINE),
            "attempts": getattr(exc, "attempts", []),
        }
        sample("after_failed_load")
        row["memory"] = memory_block(readings)
        return row
    row["load"] = {
        **record,
        "load_ms": round((time.perf_counter() - load_started) * 1000.0, 1),
        "import_ms": round(import_ms, 1),
        "error": None,
    }
    sample("after_load")

    try:
        row["load"]["dtype"] = coerce_float32(model, texts[:2])
    except Exception as exc:
        row["status"] = "error"
        row["failed_stage"] = "dtype"
        row["error"] = describe_error(exc)
        row["stage_errors"]["dtype"] = describe_error(exc)
        sample("after_failed_dtype")
        row["memory"] = memory_block(readings)
        return row

    try:
        card = model_card(model_id)
        snap = snapshot(model_id)
        dimension_getter = getattr(model, "get_embedding_dimension", None) or getattr(
            model, "get_sentence_embedding_dimension"
        )
        row["model"] = {
            "params": int(sum(p.numel() for p in model.parameters())),
            "embedding_dim": int(dimension_getter()),
            "max_seq_length": getattr(model, "max_seq_length", None),
            "license": card["license"],
            "license_source": card["license_source"],
            "license_error": card["error"],
            "gated": card["gated"],
            "pipeline_tag": card["pipeline_tag"],
            "snapshot_path": snap["path"],
            "snapshot_bytes": snap["bytes"],
            "snapshot_files": snap["files"],
            "snapshot_source": snap["source"],
            "snapshot_error": snap["error"],
        }
    except Exception as exc:
        row["stage_errors"]["model_facts"] = describe_error(exc)

    try:
        started = time.perf_counter()
        matrix = np.asarray(
            model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        )
        wall_ms = (time.perf_counter() - started) * 1000.0
        norms = np.linalg.norm(matrix, axis=1)
        row["embeddings"] = {
            "n_rows": int(matrix.shape[0]),
            "dim": int(matrix.shape[1]),
            "dtype": str(matrix.dtype),
            "wall_ms": round(wall_ms, 1),
            "ms_per_row": round(wall_ms / max(len(texts), 1), 3),
            "mean_l2_norm": round(float(norms.mean()), 4),
            "min_l2_norm": round(float(norms.min()), 4),
            "max_l2_norm": round(float(norms.max()), 4),
            "finite": bool(np.isfinite(matrix).all()),
        }
    except Exception as exc:
        row["stage_errors"]["embed"] = describe_error(exc)
    sample("after_embed")

    try:
        latency = latency_passes(
            lambda batch: model.encode(
                batch, batch_size=1, convert_to_numpy=True, show_progress_bar=False
            ),
            texts,
            passes,
            warmup,
        )
        row["latency"] = latency
    except Exception as exc:
        row["stage_errors"]["latency"] = describe_error(exc)
    sample("after_latency")

    row["memory"] = memory_block(readings)
    row["conditions"]["passes_recorded"] = (row.get("latency") or {}).get("passes")
    row["conditions"]["loadavg_1m_at_end"] = _loadavg()
    if row["stage_errors"] and row["status"] == "ok":
        row["status"] = "partial"
    return row


def _loadavg() -> float | None:
    try:
        import os

        return round(float(os.getloadavg()[0]), 3)
    except (AttributeError, OSError):
        return None


def summarise(row: dict[str, object]) -> str:
    if row["status"] == "error":
        return f"{row['model_id']}: ERROR at {row['failed_stage']}: {row['error']}"
    model = row.get("model") or {}
    latency = row.get("latency") or {}
    memory = row.get("memory") or {}
    delta = (memory.get("deltas_bytes") or {}).get("after_load_minus_before_load")
    embeddings = row.get("embeddings") or {}
    parts = [
        f"{row['model_id']}: {row['status']}",
        f"load {row['load']['load_ms']} ms",
        f"{model.get('params')} params",
        f"dim {model.get('embedding_dim')}",
        f"max_seq {model.get('max_seq_length')}",
        f"licence {model.get('license')}",
        f"disk {model.get('snapshot_bytes')} B",
        f"rss +{delta} B",
        f"embed {embeddings.get('wall_ms')} ms / {embeddings.get('n_rows')} rows",
        f"p50 {latency.get('p50_ms')} ms",
        f"p95 {latency.get('p95_ms')} ms",
    ]
    return " | ".join(parts)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Phase 3 capability probe for one Hugging Face encoder."
    )
    parser.add_argument("--model", required=True, help="Hugging Face model id")
    parser.add_argument("--out", required=True, help="path to write the JSON row")
    parser.add_argument(
        "--passes", type=int, default=DEFAULT_PASSES, help="latency passes (default 5)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="only embed the first N corpus rows (smoke tests; default: all 382)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE, help="embed batch size (default 32)"
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="load from the local cache only (HF_HUB_OFFLINE), no online attempt",
    )
    args = parser.parse_args(argv)

    try:
        row = probe(
            args.model,
            passes=max(1, args.passes),
            limit=args.limit,
            batch_size=max(1, args.batch_size),
            offline=args.offline,
        )
    except Exception as exc:  # a row must exist even when the harness itself trips
        row = {
            "schema": SCHEMA,
            "phase": 3,
            "model_id": args.model,
            "status": "error",
            "error": describe_error(exc),
            "failed_stage": "harness",
            "measured_at": iso_now(),
        }

    written = write_json(row, args.out)
    print(summarise(row))
    print(f"wrote {written}")
    return 0 if row["status"] in {"ok", "partial"} else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
