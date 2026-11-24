"""Run the phase-4 sweep for a list of model ids, one process per model.

    .venv-embed/bin/python experiments/survey/run_survey.py \
        --models sentence-transformers/all-MiniLM-L6-v2,BAAI/bge-small-en-v1.5 \
        --out-dir results/survey-2026-11-23/sweeps

Phase 4 trains, so this is serial by default and each model gets its own
interpreter: one encoder's allocator, thread pool and temp checkpoints cannot
leak into the next row, and a model that segfaults takes down its own process
instead of the sweep. ``--parallel 2`` exists because the brief allows two at a
time on eight cores; it is a flag, not the default, because two CPU-bound
fine-tunes on the same box distort each other's latency numbers.

Every model writes its own JSON (the sweep artifact) plus a ``.log`` with the
sweep's stdout, and the run writes ``summary.json`` with one row per model: the
grouped and rowwise accuracies, the paired delta and its spread, the two half
numbers, latency, and the status. A failed model leaves a row with the error
text, not a hole.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from survey_env import REPO_ROOT, describe_error, iso_now, model_slug, write_json

SWEEP_SCRIPT = Path(__file__).resolve().parent / "setfit_sweep.py"
DEFAULT_OUT_DIR = REPO_ROOT / "results" / "survey-2026-11-23" / "sweeps"


def parse_models(args) -> list[str]:
    models: list[str] = []
    for chunk in args.models or []:
        models.extend(part.strip() for part in chunk.split(",") if part.strip())
    if args.models_file:
        text = Path(args.models_file).read_text(encoding="utf-8")
        for line in text.splitlines():
            stripped = line.split("#", 1)[0].strip()
            if stripped:
                models.extend(part.strip() for part in stripped.split(",") if part.strip())
    seen: set[str] = set()
    unique = []
    for model in models:
        if model not in seen:
            seen.add(model)
            unique.append(model)
    if not unique:
        raise SystemExit("no models: pass --models id1,id2 or --models-file path")
    return unique


def run_one(model_id: str, out_dir: Path, args) -> dict[str, object]:
    slug = model_slug(model_id)
    out_json = out_dir / f"{slug}.json"
    log_path = out_dir / "logs" / f"{slug}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        args.python,
        str(SWEEP_SCRIPT),
        "--model",
        model_id,
        "--out",
        str(out_json),
        "--folds",
        args.folds,
        "--passes",
        str(args.passes),
        "--artifact",
        args.artifact,
    ]
    if args.no_rowwise:
        command.append("--no-rowwise")
    if args.skip_determinism_check:
        command.append("--no-determinism-check")
    if args.no_normalize_embeddings:
        command.append("--no-normalize-embeddings")
    if args.offline:
        command.append("--offline")

    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n")
        log.flush()
        completed = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            stdout=log,
            stderr=subprocess.STDOUT,
            env={
                **os.environ,
                "PYTHONUNBUFFERED": "1",
                "PYTHONIOENCODING": "utf-8:replace",
            },
        )
    elapsed = time.perf_counter() - started

    row: dict[str, object] = {
        "model_id": model_id,
        "status": "error",
        "exit_code": completed.returncode,
        "json": str(out_json),
        "log": str(log_path),
        "wall_seconds": round(elapsed, 1),
        "error": None,
        "failed_stage": None,
        "failed_category": None,
        "load_mode": None,
        "missing": None,
    }
    try:
        artifact = json.loads(out_json.read_text(encoding="utf-8"))
    except Exception as exc:
        row["error"] = f"no artifact: {describe_error(exc)}"
        return row

    row["status"] = artifact.get("status")
    row["error"] = artifact.get("error")
    row["failed_stage"] = artifact.get("failed_stage")
    row["failed_category"] = artifact.get("failed_category")
    row["load_mode"] = (artifact.get("load") or {}).get("mode")
    row["missing"] = artifact.get("missing")
    row["folds_run"] = artifact.get("folds_run")
    grouped = next(
        (r for r in artifact.get("runs", []) if r.get("protocol") == "grouped"), {}
    )
    rowwise = next(
        (r for r in artifact.get("runs", []) if r.get("protocol") == "rowwise"), {}
    )
    delta = next(
        (c.get("paired") for c in artifact.get("comparisons", []) if c.get("protocol") == "grouped"),
        None,
    )
    halves = artifact.get("halves") or {}
    row.update(
        {
            "grouped_accuracy": grouped.get("accuracy"),
            "rowwise_accuracy": rowwise.get("accuracy"),
            "per_fold_accuracy": grouped.get("per_fold_accuracy"),
            "fold_spread": grouped.get("fold_spread"),
            "delta_mean_points": (delta or {}).get("mean_points"),
            "delta_sd_points": (delta or {}).get("sd_points"),
            "folds_better": (delta or {}).get("folds_better"),
            "selection_grouped_accuracy": (halves.get("selection") or {}).get("grouped_accuracy"),
            "validation_grouped_accuracy": (halves.get("validation") or {}).get("grouped_accuracy"),
            "none_recall": grouped.get("none_recall"),
            "p50_ms": (grouped.get("latency_ms") or {}).get("p50"),
            "p50_ms_idle_pass": (artifact.get("latency_passes") or {}).get("p50_ms"),
            "determinism_identical": (artifact.get("determinism") or {}).get(
                "identical_predictions"
            ),
            "fit_seconds_total": round(
                sum(
                    block.get("fit_seconds") or 0.0
                    for block in grouped.get("per_fold", [])
                ),
                1,
            ),
        }
    )
    return row


def format_summary(rows: list[dict[str, object]]) -> str:
    header = (
        f"{'model':52s} {'status':7s} {'grouped':>8s} {'rowwise':>8s} {'delta':>14s} "
        f"{'select':>7s} {'valid':>7s} {'p50 ms':>7s}"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        delta = (
            f"{row['delta_mean_points']:+.2f}±{row['delta_sd_points']:.2f}"
            if row.get("delta_mean_points") is not None
            else "—"
        )
        lines.append(
            f"{str(row['model_id'])[:52]:52s} {str(row['status']):7s} "
            f"{_fmt(row.get('grouped_accuracy')):>8s} {_fmt(row.get('rowwise_accuracy')):>8s} "
            f"{delta:>14s} {_fmt(row.get('selection_grouped_accuracy')):>7s} "
            f"{_fmt(row.get('validation_grouped_accuracy')):>7s} "
            f"{_fmt(row.get('p50_ms')):>7s}"
        )
    return "\n".join(lines)


def _fmt(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, (int, float)):
        return f"{value:.4f}" if abs(value) <= 1 else f"{value:.1f}"
    return str(value)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Run the phase-4 SetFit sweep for several encoders."
    )
    parser.add_argument("--models", action="append", help="comma-separated model ids (repeatable)")
    parser.add_argument("--models-file", help="file with model ids, one per line, # comments ok")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--folds", default="all", help="passed through to setfit_sweep.py")
    parser.add_argument("--passes", type=int, default=5, help="latency passes per model")
    parser.add_argument("--artifact", default=str(REPO_ROOT / "results" / "ladder" / "rung5-ensemble.json"))
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="models at a time (default 1: phase 4 trains; 2 only with cores to spare)",
    )
    parser.add_argument("--python", default=sys.executable, help="interpreter for the sweep")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="keep an existing ok artifact instead of re-running that model",
    )
    parser.add_argument("--no-rowwise", action="store_true")
    parser.add_argument("--skip-determinism-check", action="store_true")
    parser.add_argument("--no-normalize-embeddings", action="store_true")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="passed through: load every encoder from the local cache only",
    )
    args = parser.parse_args(argv)

    models = parse_models(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    workers = max(1, min(args.parallel, len(models)))
    started = iso_now()

    pending: list[str] = []
    rows: list[dict[str, object]] = []
    for model_id in models:
        existing = out_dir / f"{model_slug(model_id)}.json"
        if args.skip_existing and existing.is_file():
            try:
                artifact = json.loads(existing.read_text(encoding="utf-8"))
            except Exception:
                artifact = {}
            if artifact.get("status") == "ok":
                print(f"skip {model_id}: {existing} already ok", flush=True)
                rows.append(_row_from_existing(model_id, existing, artifact))
                continue
        pending.append(model_id)

    print(
        f"{len(pending)} model(s) to run, {workers} at a time, folds={args.folds}",
        flush=True,
    )
    results: list[dict[str, object]] = []
    if workers == 1:
        for model_id in pending:
            print(f"→ {model_id}", flush=True)
            row = run_one(model_id, out_dir, args)
            results.append(row)
            print(f"  {row['status']}: {row.get('error') or _fmt(row.get('grouped_accuracy'))}", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = []
            for model_id in pending:
                print(f"→ {model_id}", flush=True)
                futures.append((model_id, pool.submit(run_one, model_id, out_dir, args)))
            for model_id, future in futures:
                row = future.result()
                results.append(row)
                print(
                    f"  {model_id}: {row['status']}: "
                    f"{row.get('error') or _fmt(row.get('grouped_accuracy'))}",
                    flush=True,
                )

    rows.extend(results)
    summary = {
        "schema": "survey-runner/1",
        "started_at": started,
        "finished_at": iso_now(),
        "folds": args.folds,
        "parallel": workers,
        "artifact": args.artifact,
        "python": args.python,
        "sweep_script": str(SWEEP_SCRIPT),
        "models": rows,
        "notes": {
            "serial_by_default": (
                "phase 4 trains; two CPU-bound fits on one box distort each other's "
                "latency, so --parallel is opt-in"
            ),
            "selection": "selection_grouped_accuracy is folds 0-2, validation is folds 3-4",
        },
    }
    written = write_json(summary, out_dir / "summary.json")
    print()
    print(format_summary(rows))
    print(f"\nwrote {written}")
    failures = [row for row in rows if row.get("status") != "ok"]
    partial = [row for row in rows if row.get("status") == "partial"]
    if partial:
        print(
            "partial rows carry the folds that measured: "
            + ", ".join(
                f"{row['model_id']} (missing {', '.join(row.get('missing') or ['unrecorded'])})"
                for row in partial
            ),
            file=sys.stderr,
        )
    return 1 if failures else 0


def _row_from_existing(model_id: str, path: Path, artifact: dict) -> dict[str, object]:
    grouped = next((r for r in artifact.get("runs", []) if r.get("protocol") == "grouped"), {})
    rowwise = next((r for r in artifact.get("runs", []) if r.get("protocol") == "rowwise"), {})
    delta = next(
        (c.get("paired") for c in artifact.get("comparisons", []) if c.get("protocol") == "grouped"),
        None,
    )
    halves = artifact.get("halves") or {}
    return {
        "model_id": model_id,
        "status": artifact.get("status"),
        "exit_code": 0,
        "json": str(path),
        "log": None,
        "wall_seconds": None,
        "error": artifact.get("error"),
        "failed_stage": artifact.get("failed_stage"),
        "missing": artifact.get("missing"),
        "folds_run": artifact.get("folds_run"),
        "grouped_accuracy": grouped.get("accuracy"),
        "rowwise_accuracy": rowwise.get("accuracy"),
        "per_fold_accuracy": grouped.get("per_fold_accuracy"),
        "fold_spread": grouped.get("fold_spread"),
        "delta_mean_points": (delta or {}).get("mean_points"),
        "delta_sd_points": (delta or {}).get("sd_points"),
        "folds_better": (delta or {}).get("folds_better"),
        "selection_grouped_accuracy": (halves.get("selection") or {}).get("grouped_accuracy"),
        "validation_grouped_accuracy": (halves.get("validation") or {}).get("grouped_accuracy"),
        "none_recall": grouped.get("none_recall"),
        "p50_ms": (grouped.get("latency_ms") or {}).get("p50"),
        "p50_ms_idle_pass": (artifact.get("latency_passes") or {}).get("p50_ms"),
        "determinism_identical": (artifact.get("determinism") or {}).get("identical_predictions"),
        "reused": True,
    }


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
