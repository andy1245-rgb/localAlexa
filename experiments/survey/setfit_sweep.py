"""Phase 4: rung 3's SetFit recipe on one encoder, on the ladder's own folds.

Same recipe, same folds, same metrics as ``docs/ladder-results-2026-11-10.md``
rung 3 — that is the whole point, because the question is not "which base model
scores highest" but "does any of them beat SetFit-on-MiniLM by more than the
fold spread, on rows it was not selected on".

    .venv-embed/bin/python experiments/survey/setfit_sweep.py \
        --model sentence-transformers/all-MiniLM-L6-v2 \
        --out results/survey-2026-11-23/sweeps/sentence-transformers__all-MiniLM-L6-v2.json

What makes the numbers comparable rather than merely similar:

* **Folds are read, never regenerated.** The 110-family GroupKFold assignment
  comes out of ``results/ladder/rung5-ensemble.json`` ->
  ``fold_assignments.5.family_fold``. Row counts are verified against the corpus
  (382 rows, 76-77 per fold) and the run refuses to start if they disagree, so a
  drifted corpus cannot produce a quietly different experiment.
* **The pool is the ladder's.** ``pools.fold_pool`` drops each fold's contaminated
  references (a reference that is a literal slot fill of a held-out family), so
  the model never trains on wording it is about to be scored on.
* **The metrics are the ladder's code.** ``run.block_metrics``,
  ``run.predict_with_confidence`` and ``run.timed_latencies`` are the functions
  rung 3 was scored with; the artifact keeps the ladder's field names.
* **Selection and validation are separated in the output.** Folds 0-2 are the
  selection half, folds 3-4 the validation half. Both halves carry their own
  grouped accuracy, so a later script can choose a winner on one half and read
  the other half's number without having selected on it.
* **Determinism is a receipt, not a claim.** The first fold is fitted twice and
  the two prediction vectors are compared; the result is in the artifact under
  ``determinism``.
* **A stage that fails does not cost the sweep.** Every stage is attempted
  through ``sweep_stages.StageBook``, which records the stage name, its failure
  category, the error text and the full traceback, and lets the run continue
  where continuing is honest. Fold accuracies that were already measured are
  written out with ``status: "partial"`` and the missing parts named under
  ``missing``. Only a run with no fold measured at all is ``status: "error"``. A
  five-fold sweep is fourteen minutes; a crash that discards four measured folds
  because the fifth hit an OSError on a Windows box is the failure mode this
  replaces.
* **The encoder is float32 before anything is trained on it.** A bf16 checkpoint
  (granite-embedding-30m) loads as bf16 on CPU and the float32 SetFit head cannot
  train on it; ``load.dtype`` records the dtypes before and after the cast and the
  dtype a real forward pass hands the head, and a cast that does not take fails the
  ``dtype`` stage with the actual dtype in the message. See ``survey_dtype``.
* **A Hub blink does not fail the sweep.** The load is attempted online and then
  once from the local cache with ``HF_HUB_OFFLINE=1``/``TRANSFORMERS_OFFLINE=1``
  set in-process; ``load.mode`` says which path produced the model, and
  ``--offline`` starts on the cached path. See ``survey_load``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from localalexa.data import EXPECTED_ROWS
from localalexa.ladder import metrics
from localalexa.ladder.folds import FoldAssignment, build_folds
from localalexa.ladder.pools import fold_pool
from localalexa.ladder.run import (
    Rung,
    block_metrics,
    evaluate_rowwise,
    load_bench,
    predict_with_confidence,
    timed_latencies,
)

from survey_env import (
    REPO_ROOT,
    describe_error,
    environment,
    iso_now,
    latency_passes,
    model_card,
    model_slug,
    snapshot,
    write_json,
)
from survey_dtype import DtypeCoercionError, coerce_float32
from survey_load import (
    MODE_OFFLINE_FALLBACK,
    MODE_ONLINE,
    LoadFailure,
    load_with_fallback,
    offline_active,
)
from survey_setfit import RECIPE, SurveySetFitPredictor, current_output_root, recipe_for
from sweep_stages import StageAbort, StageBook, traceback_text

SCHEMA = "survey-sweep/1"
DEFAULT_ARTIFACT = REPO_ROOT / "results" / "ladder" / "rung5-ensemble.json"
INCUMBENT_RUNG = "r3-setfit-minilm"
INCUMBENT_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
FOLD_KEY = "5"
SELECTION_FOLDS = (0, 1, 2)
VALIDATION_FOLDS = (3, 4)
DEFAULT_PASSES = 5


class FoldAssignmentError(RuntimeError):
    """The ladder's fold assignment does not match this corpus; refuse to run."""


def fold_category(exc: BaseException) -> str:
    """The taxonomy bucket a failed fold falls in: load, dtype, or the fit itself."""
    if isinstance(exc, DtypeCoercionError):
        return "dtype"
    if isinstance(exc, LoadFailure):
        return "load"
    return "fit"


def verified_folds(artifact_path: Path, bench) -> tuple[FoldAssignment, dict[str, object]]:
    """The ladder's fold assignment, verified against the corpus before it is used."""
    raw = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
    try:
        block = raw["fold_assignments"][FOLD_KEY]
        family_fold = {str(k): int(v) for k, v in block["family_fold"].items()}
        rows_per_fold = tuple(int(v) for v in block["rows_per_fold"])
    except (KeyError, TypeError, ValueError) as exc:
        raise FoldAssignmentError(
            f"{artifact_path} carries no usable fold_assignments[{FOLD_KEY}]: {describe_error(exc)}"
        ) from exc

    problems: list[str] = []
    if len(bench.rows) != EXPECTED_ROWS:
        problems.append(f"corpus has {len(bench.rows)} rows, expected {EXPECTED_ROWS}")
    if block.get("n_folds") is not None and int(block["n_folds"]) != len(rows_per_fold):
        problems.append(f"n_folds={block['n_folds']} but {len(rows_per_fold)} row counts")
    bench_families = set(bench.families)
    if bench_families != set(family_fold):
        missing = sorted(bench_families - set(family_fold))[:3]
        extra = sorted(set(family_fold) - bench_families)[:3]
        problems.append(
            f"family sets differ (corpus only: {missing} …, artifact only: {extra} …)"
        )
    else:
        computed = [0] * len(rows_per_fold)
        for family in bench.families:
            computed[int(family_fold[family])] += 1
        if tuple(computed) != rows_per_fold:
            problems.append(f"rows per fold {computed} != artifact {list(rows_per_fold)}")
        out_of_range = [n for n in computed if not 76 <= n <= 77]
        if out_of_range:
            problems.append(f"folds outside 76-77 rows: {out_of_range}")
    if problems:
        raise FoldAssignmentError(
            "refusing to run: " + "; ".join(problems) + f" (assignment read from {artifact_path})"
        )

    assignment = FoldAssignment(
        n_folds=len(rows_per_fold),
        family_fold=family_fold,
        rows_per_fold=rows_per_fold,
    )
    regenerated = build_folds(bench.families, assignment.n_folds)
    info: dict[str, object] = {
        "source": str(artifact_path),
        "key": f"fold_assignments.{FOLD_KEY}",
        "n_folds": assignment.n_folds,
        "n_families": len(assignment.family_fold),
        "rows_per_fold": list(assignment.rows_per_fold),
        "verified": {
            "corpus_rows": len(bench.rows),
            "rows_per_fold_recomputed": [
                sum(1 for family in bench.families if assignment.fold_of(family) == f)
                for f in range(assignment.n_folds)
            ],
            "row_counts_match_artifact": True,
            "folds_run_in_range_76_77": True,
        },
        "regenerated_matches_artifact": regenerated.family_fold
        == assignment.family_fold,
        "note": "read from the artifact, not regenerated; regeneration is only cross-checked",
    }
    return assignment, info


def folds_to_run(spec: str, n_folds: int) -> list[int]:
    if spec.strip().lower() in {"all", ""}:
        return list(range(n_folds))
    wanted = sorted({int(part) for part in spec.split(",") if part.strip()})
    out_of_range = [f for f in wanted if not 0 <= f < n_folds]
    if out_of_range:
        raise SystemExit(f"folds {out_of_range} outside 0..{n_folds - 1}")
    return wanted


@dataclass(frozen=True)
class Prepared:
    """Everything the run needs before it touches a model, resolved in one place."""

    bench: object
    assignment: FoldAssignment
    folds: list[int]
    fold_info: dict[str, object]
    rung_name: str
    note: str
    normalize: bool


def prepare(args) -> Prepared:
    bench = load_bench()
    assignment, fold_info = verified_folds(Path(args.artifact), bench)
    bench = replace(bench, folds=assignment)
    folds = folds_to_run(args.folds, assignment.n_folds)
    rung_name = (
        INCUMBENT_RUNG
        if args.model == INCUMBENT_ENCODER
        else f"r3-setfit-{model_slug(args.model)}"
    )
    note = (
        f"survey phase 4: rung-3 recipe on {args.model}, "
        f"{len(folds)} fold(s) of the ladder's 5-fold assignment"
    )
    return Prepared(
        bench=bench,
        assignment=assignment,
        folds=folds,
        fold_info=fold_info,
        rung_name=rung_name,
        note=note,
        normalize=not args.no_normalize_embeddings,
    )


def run_grouped(
    factory,
    bench,
    folds: list[int],
    *,
    rung_name: str,
    note: str,
    book: StageBook | None = None,
) -> tuple[dict[str, object] | None, dict[int, dict[str, object]], dict[str, object]]:
    """The ladder's grouped protocol over a subset of folds, scored by ladder code.

    A fold that raises is recorded and skipped, not re-raised: four measured
    folds are worth more than a tombstone, and the fold that failed is named in
    the artefact. Returns the run block (``None`` when no fold survived), the
    per-fold predictions, and a holder carrying the last fitted predictor so the
    dedicated latency pass can reuse it instead of paying for another fine-tune.
    """
    holder: dict[str, object] = {"predictor": None}
    indices_by_fold = bench.folds.indices(list(bench.families))
    predicted = [""] * len(bench.rows)
    confidences = [0.0] * len(bench.rows)
    latencies: list[float] = []
    fold_scores: list[float] = []
    train_rows: list[int] = []
    fold_rows: list[int] = []
    per_fold: list[dict[str, object]] = []
    detail: dict[int, dict[str, object]] = {}
    scored: list[int] = []

    for fold in folds:
        started = time.perf_counter()
        try:
            indices = indices_by_fold[fold]
            held_out = {bench.families[i] for i in indices}
            pool = fold_pool(bench.registry, held_out)
            predictor = factory().fit([t for t, _ in pool], [label for _, label in pool])
            holder["predictor"] = predictor
            texts = [bench.rows[i].text for i in indices]
            truth = [bench.rows[i].tool for i in indices]
            predictor.scores(texts[:1])  # warm the caches
            fold_latencies = timed_latencies(predictor, texts)
            out, conf = predict_with_confidence(predictor, texts)
        except Exception as exc:
            if book is None:
                raise
            book.fold_failed(
                fold,
                exc,
                seconds=time.perf_counter() - started,
                category=fold_category(exc),
            )
            continue

        mode = getattr(predictor, "load_mode", None)
        modes = holder.setdefault("load_modes", [])
        if mode is not None and mode not in modes:  # type: ignore[operator]
            modes.append(mode)  # type: ignore[union-attr]

        for i, label, score in zip(indices, out, conf):
            predicted[i] = label
            confidences[i] = score
        score = metrics.accuracy(out, truth)
        fold_scores.append(score)
        scored.append(fold)
        train_rows.append(len(pool))
        fold_rows.append(len(indices))
        latencies.extend(fold_latencies)
        per_fold.append(
            {
                "fold": fold,
                "n_rows": len(indices),
                "train_rows": len(pool),
                "correct": sum(1 for p, t in zip(out, truth) if p == t),
                "accuracy": round(score, 4),
                "fit_seconds": round(float(getattr(predictor, "fit_seconds", 0.0)), 2),
                "latency_ms": metrics.latency(fold_latencies),
            }
        )
        detail[fold] = {"indices": indices, "predicted": out, "truth": truth, "confidences": conf}
        print(
            f"  fold {fold}: acc {score:.4f} ({per_fold[-1]['correct']}/{len(indices)}) "
            f"train {len(pool)} fit {per_fold[-1]['fit_seconds']:.1f}s "
            f"p50 {per_fold[-1]['latency_ms']['p50']:.3f} ms",
            flush=True,
        )

    if not scored:
        return None, detail, holder

    run_rows = sorted(i for fold in scored for i in indices_by_fold[fold])
    block: dict[str, object] = {
        "rung": rung_name,
        "note": note,
        "protocol": "grouped",
        "fold_count": len(scored),
        "n_folds": len(scored),
        "complete": len(scored) == bench.folds.n_folds,
        "folds": folds,
        "folds_scored": scored,
        "train_rows": train_rows,
        "fold_rows": fold_rows,
        "n_rows": len(run_rows),
        "latency_ms": metrics.latency(latencies),
        "per_fold_accuracy": [round(s, 4) for s in fold_scores],
        "per_fold_accuracy_by_fold": {
            str(fold): round(score, 4) for fold, score in zip(scored, fold_scores)
        },
        "fold_spread": metrics.spread(fold_scores),
        "per_fold": per_fold,
        **block_metrics(
            [predicted[i] for i in run_rows],
            [confidences[i] for i in run_rows],
            [bench.rows[i].tool for i in run_rows],
            [bench.families[i] for i in run_rows],
        ),
    }
    return block, detail, holder


def half_block(
    name: str,
    wanted: tuple[int, ...],
    per_fold_accuracy: dict[int, float],
    bench,
    per_fold_detail: dict[int, dict[str, object]],
    baseline_folds: dict[int, float] | None,
) -> dict[str, object]:
    """One half of the split, with the pooled number a comparison would quote."""
    present = [f for f in wanted if f in per_fold_accuracy]
    missing = [f for f in wanted if f not in per_fold_accuracy]
    indices_by_fold = bench.folds.indices(list(bench.families))
    rows = [i for fold in present for i in indices_by_fold[fold]]
    predicted = []
    truth = []
    for fold in present:
        detail = per_fold_detail[fold]
        predicted.extend(detail["predicted"])
        truth.extend(detail["truth"])
    grouped = metrics.accuracy(predicted, truth) if rows else None
    block: dict[str, object] = {
        "folds": list(wanted),
        "folds_run": present,
        "complete": not missing,
        "missing_folds": missing,
        "n_folds": len(present),
        "n_rows": len(rows),
        "grouped_accuracy": None if grouped is None else round(grouped, 4),
        "mean_fold_accuracy": (
            round(mean([per_fold_accuracy[f] for f in present]), 4) if present else None
        ),
        "per_fold_accuracy": {str(f): round(per_fold_accuracy[f], 4) for f in present},
        "definition": (
            "grouped_accuracy is pooled over the half's rows; mean_fold_accuracy is the "
            "mean of the half's per-fold accuracies (equal weight per fold)"
        ),
    }
    if baseline_folds:
        block["baseline_per_fold_accuracy"] = {
            str(f): baseline_folds[f] for f in present if f in baseline_folds
        }
        shared = [f for f in present if f in baseline_folds]
        block["baseline_mean_fold_accuracy"] = (
            round(mean([baseline_folds[f] for f in shared]), 4) if shared else None
        )
    return block


def determinism_check(factory, bench, fold: int, first: dict[str, object]) -> dict[str, object]:
    """Fit one fold a second time from scratch and diff its predictions."""
    indices = bench.folds.indices(list(bench.families))[fold]
    held_out = {bench.families[i] for i in indices}
    pool = fold_pool(bench.registry, held_out)
    predictor = factory().fit([t for t, _ in pool], [label for _, label in pool])
    texts = [bench.rows[i].text for i in indices]
    out, conf = predict_with_confidence(predictor, texts)
    differing = [i for i, (a, b) in enumerate(zip(out, first["predicted"])) if a != b]
    confidence_gap = max(
        (abs(a - b) for a, b in zip(conf, first["confidences"])), default=0.0
    )
    return {
        "checked": True,
        "fold": fold,
        "identical_predictions": not differing,
        "rows_differing": len(differing),
        "differing_row_positions": differing[:10],
        "max_confidence_abs_diff": round(float(confidence_gap), 12),
        "accuracy_first": round(metrics.accuracy(first["predicted"], first["truth"]), 4),
        "accuracy_second": round(metrics.accuracy(out, first["truth"]), 4),
        "fit_seconds_second": round(float(getattr(predictor, "fit_seconds", 0.0)), 2),
        "note": (
            "same fold rebuilt from scratch in the same process; seeds are in this "
            "artifact under `seeds`. Two runs are identical when rows_differing is 0."
        ),
    }


def incumbent_block(artifact_path: Path) -> dict[str, object]:
    """Rung 3's numbers as the ladder wrote them, for the delta to be against."""
    block: dict[str, object] = {
        "rung": INCUMBENT_RUNG,
        "encoder": INCUMBENT_ENCODER,
        "source": str(artifact_path),
        "grouped_accuracy": None,
        "rowwise_accuracy": None,
        "per_fold_accuracy": None,
        "latency_p50_ms": None,
        "none_recall": None,
        "note": "read from the ladder artifact; its per-fold values are rounded to 4 dp there",
    }
    try:
        raw = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
    except Exception as exc:
        block["error"] = describe_error(exc)
        return block
    for run in raw.get("runs", []):
        if run.get("rung") != INCUMBENT_RUNG:
            continue
        if run.get("protocol") == "grouped":
            block["grouped_accuracy"] = run.get("accuracy")
            block["per_fold_accuracy"] = run.get("per_fold_accuracy")
            block["latency_p50_ms"] = (run.get("latency_ms") or {}).get("p50")
            block["none_recall"] = run.get("none_recall")
        elif run.get("protocol") == "rowwise":
            block["rowwise_accuracy"] = run.get("accuracy")
    if block["per_fold_accuracy"] is None:
        block["error"] = "artifact carries no grouped run for " + INCUMBENT_RUNG
    return block


def load_encoder(model_id: str, offline: bool = False):
    """One encoder, from the Hub or — ``offline`` — from the cache only.

    ``local_files_only`` is passed as well as the environment variable, because
    the Hub reads that variable once, at import, and ``survey_load`` must not
    depend on which Hub version is installed to make the retry mean something.
    """
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_id, local_files_only=offline)


def skeleton(args) -> dict[str, object]:
    """The row as it exists before any measurement, so a failure has somewhere to land."""
    return {
        "schema": SCHEMA,
        "phase": 4,
        "model_id": args.model,
        "status": "running",
        "error": None,
        "failed_stage": None,
        "failed_category": None,
        "traceback": None,
        "missing": [],
        "stage_errors": {},
        "stage_order": [],
        "measured_at": iso_now(),
        "environment": None,
        "load": None,
        "model": None,
        "bench": None,
        "fold_assignments": {},
        "recipe": recipe_for(args.model, not args.no_normalize_embeddings),
        "seeds": {
            "setfit": RECIPE["seed"],
            "torch": RECIPE["seed"],
            "numpy": RECIPE["seed"],
            "sklearn": RECIPE["seed"],
            "note": (
                "set in survey_setfit.fit and by SetFit's TrainingArguments(seed=...); "
                "GroupKFold has no seed, the assignment is read from the artifact"
            ),
        },
        "folds_run": [],
        "halves": {},
        "runs": [],
        "comparisons": [],
        "incumbent": None,
        "determinism": {"checked": False},
        "trainer_output_dir": None,
        "skipped": [],
        "protocol_notes": {
            "grouped": "folds over phrasing families; a fold trains on the references that "
            "do not render into that fold's families (ladder pools.fold_pool)",
            "rowwise": "train on all 200 references, score all 382 rows — the protocol the "
            "93.19% / 0.61 ms baseline was measured under",
            "reject_curve": "thresholds chosen on the scored rows: oracle operating points, "
            "the ceiling, not the shipped number",
            "selection": "folds 0-2 select, folds 3-4 validate; never select on validation",
        },
    }


def _run(
    row: dict[str, object],
    book: StageBook,
    args,
    prepared: Prepared,
    *,
    factory,
    loader,
) -> None:
    bench = prepared.bench
    folds = prepared.folds

    def environment_stage() -> None:
        row["environment"] = environment()

    book.attempt("environment", environment_stage)

    def load_stage() -> object:
        started = time.perf_counter()
        try:
            probe, record = load_with_fallback(
                lambda offline: loader(args.model, offline=offline),
                offline=bool(getattr(args, "offline", False)),
                model_id=args.model,
            )
        except Exception as exc:
            row["load"] = {
                "status": "error",
                "load_ms": round((time.perf_counter() - started) * 1000.0, 1),
                "error": describe_error(exc),
                "mode": getattr(exc, "mode", None)
                or (MODE_OFFLINE_FALLBACK if offline_active() else MODE_ONLINE),
                "attempts": getattr(exc, "attempts", []),
            }
            raise
        row["load"] = {
            **record,
            "load_ms": round((time.perf_counter() - started) * 1000.0, 1),
            "error": None,
        }
        return probe

    probe = book.attempt("load", load_stage, fatal=True)

    def dtype_stage() -> dict[str, object]:
        """Cast the encoder to float32 and prove the cast, before any fit."""
        record = coerce_float32(probe, list(bench.texts[:2]))
        row["load"]["dtype"] = record
        print(
            f"  dtype: {'/'.join(record['before'])} -> {'/'.join(record['after'])}"
            f" (coerced {record['coerced']}), embeddings "
            f"{(record['embeddings'] or {}).get('torch')}",
            flush=True,
        )
        return record

    book.attempt("dtype", dtype_stage, fatal=True)

    def model_facts() -> None:
        dimension = getattr(probe, "get_embedding_dimension", None) or (
            probe.get_sentence_embedding_dimension
        )
        row["model"] = {
            "params": int(sum(p.numel() for p in probe.parameters())),
            "embedding_dim": int(dimension()),
            "max_seq_length": getattr(probe, "max_seq_length", None),
            "license": model_card(args.model)["license"],
            "snapshot_bytes": snapshot(args.model)["bytes"],
        }

    book.attempt("model_facts", model_facts)

    grouped, detail, holder = run_grouped(
        factory,
        bench,
        folds,
        rung_name=prepared.rung_name,
        note=prepared.note,
        book=book,
    )
    if grouped is not None:
        row["runs"].append(grouped)
        if args.keep_predictions:
            grouped["predictions"] = {
                str(fold): block["predicted"] for fold, block in sorted(detail.items())
            }
    row["trainer_output_dir"] = (
        str(current_output_root()) if current_output_root() is not None else None
    )
    fit_modes = holder.get("load_modes") or []
    if fit_modes and isinstance(row.get("load"), dict):
        row["load"]["fit_modes"] = sorted(fit_modes)

    per_fold_accuracy = {
        int(k): float(v) for k, v in (grouped or {}).get("per_fold_accuracy_by_fold", {}).items()
    }
    scored_folds = [f for f in folds if f in per_fold_accuracy]

    if args.passes <= 0:
        row["skipped"].append("latency_pass (--passes 0)")
        book.skipped("latency_pass", "--passes 0")
    elif holder["predictor"] is None:
        book.skipped("latency_pass", "no fold was fitted, so there is no model to time")
    else:
        print(f"  latency: {args.passes} passes over {len(bench.rows)} rows...", flush=True)

        def latency_stage() -> dict[str, object]:
            block = latency_passes(holder["predictor"].scores, bench.texts, args.passes)
            block["conditions"] = {
                "measured_on": f"model trained on fold {scored_folds[-1]}'s pool, "
                f"scored on all {len(bench.rows)} rows",
                "torch_threads": (row.get("environment") or {}).get("torch_threads"),
                "loadavg_1m": (row.get("environment") or {}).get("loadavg_1m"),
                "protocol": "same as experiments/ladder/latency_pass.py: run alone, repeat, median",
            }
            return block

        block = book.attempt("latency_pass", latency_stage)
        if block is not None:
            row["latency_passes"] = block
    holder["predictor"] = None

    rowwise: dict[str, object] | None = None
    if args.no_rowwise:
        row["skipped"].append("rowwise (--no-rowwise)")
        book.skipped("rowwise", "--no-rowwise")
    else:
        print("  rowwise fit (200 references, all rows)...", flush=True)

        def rowwise_stage() -> dict[str, object]:
            block = evaluate_rowwise(Rung(prepared.rung_name, factory, note=prepared.note), bench)
            block.pop("predictions", None)
            block["rung"] = prepared.rung_name
            block["note"] = prepared.note
            block["fold_count"] = None
            return block

        rowwise = book.attempt("rowwise", rowwise_stage)
        if rowwise is not None:
            row["runs"].append(rowwise)

    incumbent = row["incumbent"] or {}
    baseline_folds = incumbent.get("per_fold_accuracy") or []
    baseline_by_fold: dict[int, float] = {}
    if baseline_folds and len(baseline_folds) == prepared.assignment.n_folds:
        baseline_by_fold = {f: float(v) for f, v in enumerate(baseline_folds)}

    def halves_stage() -> dict[str, object]:
        return {
            "selection": half_block(
                "selection", SELECTION_FOLDS, per_fold_accuracy, bench, detail, baseline_by_fold
            ),
            "validation": half_block(
                "validation", VALIDATION_FOLDS, per_fold_accuracy, bench, detail, baseline_by_fold
            ),
            "folds_scored": scored_folds,
            "rule": (
                "select on the selection half (folds 0-2) and quote the validation half "
                "(folds 3-4); the halves are disjoint family sets by construction of the "
                "family-grouped assignment, so the validation number was never selected on"
            ),
        }

    if scored_folds:
        halves = book.attempt("halves", halves_stage)
        if halves is not None:
            row["halves"] = halves

    def comparisons_stage() -> list[dict[str, object]]:
        comparisons: list[dict[str, object]] = []
        if not baseline_by_fold:
            return comparisons
        comparisons.append(
            {
                "rung": prepared.rung_name,
                "baseline": INCUMBENT_RUNG,
                "fold_count": len(scored_folds),
                "protocol": "grouped",
                "paired": metrics.paired_delta(
                    [per_fold_accuracy[f] for f in scored_folds],
                    [baseline_by_fold[f] for f in scored_folds],
                ),
                "baseline_source": str(args.artifact),
            }
        )
        if rowwise is not None and incumbent.get("rowwise_accuracy") is not None:
            comparisons.append(
                {
                    "rung": prepared.rung_name,
                    "baseline": INCUMBENT_RUNG,
                    "fold_count": None,
                    "protocol": "rowwise",
                    "paired": None,
                    "delta_points": round(
                        100.0 * (rowwise["accuracy"] - incumbent["rowwise_accuracy"]), 2
                    ),
                    "baseline_source": str(args.artifact),
                }
            )
        return comparisons

    if scored_folds:
        comparisons = book.attempt("comparisons", comparisons_stage)
        if comparisons is not None:
            row["comparisons"] = comparisons

    if not args.determinism_check_requested:
        row["skipped"].append("determinism (--no-determinism-check)")
        book.skipped("determinism", "--no-determinism-check")
    elif not scored_folds:
        book.skipped("determinism", "no fold was fitted to refit")
    else:
        fold = scored_folds[0]
        print(f"  determinism: refitting fold {fold}...", flush=True)

        def determinism_stage() -> dict[str, object]:
            return determinism_check(factory, bench, fold, detail[fold])

        receipt = book.attempt("determinism", determinism_stage)
        if receipt is not None:
            row["determinism"] = receipt

    if args.model == INCUMBENT_ENCODER and baseline_by_fold and scored_folds:

        def provenance_stage() -> dict[str, object]:
            mine = [per_fold_accuracy[f] for f in scored_folds]
            theirs = [baseline_by_fold[f] for f in scored_folds]
            return {
                "reproduces_artifact": all(abs(a - b) < 0.015 for a, b in zip(mine, theirs)),
                "per_fold_delta_from_artifact_points": [
                    round(100.0 * (a - b), 2) for a, b in zip(mine, theirs)
                ],
                "note": (
                    "this model IS the incumbent encoder, so these deltas measure whether the "
                    "survey's recipe reproduces rung 3 rather than beating it"
                ),
            }

        provenance = book.attempt("provenance", provenance_stage)
        if provenance is not None:
            row["provenance"] = provenance


def sweep(args, *, factory=None, loader=None) -> dict[str, object]:
    """Measure one encoder and return the row, whatever happened inside it.

    ``factory`` and ``loader`` exist so the stage machinery can be exercised
    without a fine-tune: the tests drive this function with a stub predictor and
    a stub loader, which is what makes a forced ``OSError(EINVAL)`` in the
    trainer's output directory testable in a second instead of fourteen minutes.
    """
    if factory is None:
        normalize = not args.no_normalize_embeddings
        offline = bool(getattr(args, "offline", False))

        def factory():
            return SurveySetFitPredictor(
                args.model, normalize_embeddings=normalize, offline=offline
            )

    if loader is None:
        loader = load_encoder

    row = skeleton(args)
    book = StageBook()
    measured = False
    try:
        prepared = prepare(args)
        row["bench"] = {
            "n_rows": len(prepared.bench.rows),
            "n_families": len(set(prepared.bench.families)),
            "n_labels": len(prepared.bench.labels),
            "n_reference_rows": sum(len(t.references) for t in prepared.bench.registry.tools)
            + len(prepared.bench.registry.none_references),
        }
        row["fold_assignments"] = {FOLD_KEY: prepared.fold_info}
        row["folds_run"] = prepared.folds
        row["incumbent"] = incumbent_block(Path(args.artifact))
        book.ok("setup", 0.0)
        _run(row, book, args, prepared, factory=factory, loader=loader)
        measured = any(run.get("protocol") == "grouped" for run in row["runs"])
    except StageAbort:
        measured = False
    except FoldAssignmentError:
        raise
    except Exception as exc:
        book.failed("harness", exc)
        print(traceback_text(exc), file=sys.stderr, flush=True)
    return book.apply(row, measured=measured)


def summarise(row: dict[str, object]) -> str:
    if row["status"] == "error":
        return f"{row['model_id']}: ERROR at {row['failed_stage']}: {row['error']}"
    runs = row.get("runs") or []
    grouped = next((r for r in runs if r.get("protocol") == "grouped"), {})
    rowwise = next((r for r in runs if r.get("protocol") == "rowwise"), {})
    delta = next(
        (c["paired"] for c in row.get("comparisons") or [] if c.get("protocol") == "grouped"),
        None,
    )
    halves = row.get("halves") or {}
    selection = (halves.get("selection") or {}).get("grouped_accuracy")
    validation = (halves.get("validation") or {}).get("grouped_accuracy")
    parts = [
        f"{row['model_id']}: {row['status']}",
        f"folds {row.get('folds_run')}",
        f"grouped {grouped.get('accuracy')}",
        f"rowwise {rowwise.get('accuracy')}",
        f"delta {delta['mean_points']:+.2f}±{delta['sd_points']:.2f}" if delta else "delta -",
        f"selection {selection}",
        f"validation {validation}",
        f"p50 {(grouped.get('latency_ms') or {}).get('p50')} ms",
        f"none_recall {grouped.get('none_recall')}",
    ]
    if row.get("determinism", {}).get("checked"):
        parts.append(
            f"determinism {'identical' if row['determinism']['identical_predictions'] else 'DIFFERS'}"
        )
    if row["status"] == "partial":
        parts.append(f"missing {', '.join(row.get('missing') or []) or 'unrecorded'}")
        parts.append(
            f"first failure at {row.get('failed_stage')} ({row.get('failed_category')})"
        )
    load = row.get("load") or {}
    if load.get("mode"):
        parts.append(f"load {load['mode']}")
    return " | ".join(parts)


def configure_console() -> None:
    """Never let an unencodable character end a run that has already measured folds.

    A sweep launched by ``run_survey.py`` inherits a file handle rather than a
    console, so CPython wraps it with the locale codec -- ``cp1252`` or ``cp437``
    on Windows. One character outside that codec in a progress line used to
    raise ``UnicodeEncodeError`` inside a stage that only the full sweep reaches.
    The progress lines are ASCII now, and this is the belt to that braces: an
    unencodable character becomes ``?`` instead of an exception.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(errors="replace")
            except (ValueError, OSError):
                pass


def main(argv: list[str]) -> int:
    configure_console()
    parser = argparse.ArgumentParser(
        description="Phase 4 SetFit sweep for one Hugging Face encoder on the ladder's folds."
    )
    parser.add_argument("--model", required=True, help="Hugging Face model id")
    parser.add_argument("--out", required=True, help="path to write the JSON artifact")
    parser.add_argument(
        "--folds", default="all", help="comma-separated fold ids, or 'all' (default)"
    )
    parser.add_argument(
        "--artifact",
        default=str(DEFAULT_ARTIFACT),
        help="ladder artifact carrying the 5-fold assignment (default: rung5-ensemble.json)",
    )
    parser.add_argument(
        "--passes",
        type=int,
        default=DEFAULT_PASSES,
        help="repeats for the dedicated latency pass (default 5, 0 disables)",
    )
    parser.add_argument(
        "--no-normalize-embeddings",
        action="store_true",
        help="feed the head raw body embeddings (setfit 1.2.0 default); for provenance only",
    )
    parser.add_argument(
        "--no-rowwise", action="store_true", help="skip the extra 200-reference fit"
    )
    parser.add_argument(
        "--no-determinism-check",
        dest="determinism_check_requested",
        action="store_false",
        help="skip the second fit of the first fold",
    )
    parser.add_argument(
        "--keep-predictions", action="store_true", help="include per-fold predictions"
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="load every encoder from the local cache only (HF_HUB_OFFLINE), no online attempt",
    )
    args = parser.parse_args(argv)

    try:
        row = sweep(args)
    except FoldAssignmentError as exc:
        row = {
            "schema": SCHEMA,
            "phase": 4,
            "model_id": args.model,
            "status": "error",
            "failed_stage": "fold_assignment",
            "failed_category": "fold_assignment",
            "error": str(exc),
            "traceback": None,
            "missing": ["fold_assignment"],
            "refused": True,
            "measured_at": iso_now(),
        }
        write_json(row, args.out)
        print(f"refused: {exc}", file=sys.stderr)
        print(f"wrote {Path(args.out)}", file=sys.stderr)
        return 2
    except Exception as exc:
        row = {
            "schema": SCHEMA,
            "phase": 4,
            "model_id": args.model,
            "status": "error",
            "failed_stage": "harness",
            "failed_category": "harness",
            "error": describe_error(exc),
            "traceback": traceback_text(exc),
            "missing": ["harness"],
            "measured_at": iso_now(),
        }

    written = write_json(row, args.out)
    print(summarise(row))
    print(f"wrote {written}")
    if row["status"] == "partial":
        print(
            "partial: the measured folds are in the artefact, "
            f"the rest did not run ({', '.join(row.get('missing') or [])})",
            file=sys.stderr,
        )
    return 0 if row["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
