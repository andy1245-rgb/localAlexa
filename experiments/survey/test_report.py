"""Self-test for report_phase3.py and report_phase4.py. Stdlib only, fixtures in a temp dir.

    .venv-embed/bin/python experiments/survey/test_report.py

Builds synthetic probe and sweep artefacts in the shape ``probe_model.py`` and
``setfit_sweep.py`` actually write, runs each reporter as a separate process, and
asserts on the artefacts and on the process's exit code. Every fixture lives in
``tempfile.mkdtemp()`` (``/tmp`` here, ``%TEMP%`` on Windows) — nothing is written
under ``results/``, so a failing fixture can never be mistaken for a measurement.

The cases are the failure modes the brief names, not happy paths:

* a model that beats the incumbent on both halves — must emit the win form;
* a model that wins the selection half and loses the validation half — must emit
  "the ranking does not transfer" and show both orderings;
* one fold disagreeing, so the delta mean sits inside its own sd — must NOT win;
* the same, with the disagreement in magnitude rather than sign — must NOT win;
* a tie at zero — must NOT win, and the tie-break must leave the incumbent first;
* pooled-across-all-folds accuracy moved to an absurd value — the verdict must
  not change, which is the mechanical form of "not a selection criterion";
* a missing fold — must refuse and write nothing;
* a single model — must refuse rather than emit a one-model "sweep";
* a model that was not picked clearing the spread on validation — must emit
  "the ranking does not transfer", not a win;
* a failed-load probe — must appear in the table with its error text;
* a probe over the latency ceiling — must be dropped, with the ceiling printed;
* two probes of one model under different conditions — both rows, no decision;
* two probes of one model under identical conditions — one row, a named note.

Exit code is 0 only when every check passes.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PYTHON = sys.executable

FOLDS = (0, 1, 2, 3, 4)
FOLD_ROWS = {0: 77, 1: 77, 2: 76, 3: 76, 4: 76}
SELECTION = [0, 1, 2]
VALIDATION = [3, 4]
INCUMBENT_ID = "sentence-transformers/all-MiniLM-L6-v2"
# The ladder's rung 3, to the row: 359/382 grouped.
INCUMBENT_CORRECT = {0: 71, 1: 69, 2: 76, 3: 69, 4: 74}

FAILURES: list[str] = []
CHECKS = 0


def check(case: str, condition: bool, message: str) -> None:
    global CHECKS
    CHECKS += 1
    if not condition:
        FAILURES.append(f"{case}: {message}")


def accuracy(correct: dict[int, int], fold_rows: dict[int, int], folds) -> float:
    return sum(correct[f] for f in folds) / sum(fold_rows[f] for f in folds)


def build_sweep(
    model_id: str,
    correct: dict[int, int],
    *,
    incumbent_correct: dict[int, int] | None = None,
    folds: tuple[int, ...] = FOLDS,
    fold_rows: dict[int, int] | None = None,
    grouped_pooled: float | None = None,
    rowwise_pooled: float = 0.9424,
    status: str = "ok",
) -> dict:
    """A survey-sweep/1 artefact with the fields the reporter reads, and consistent arithmetic."""
    fold_rows = dict(fold_rows or FOLD_ROWS)
    incumbent_correct = dict(incumbent_correct or INCUMBENT_CORRECT)
    folds = tuple(folds)
    per_fold = [
        {
            "fold": fold,
            "n_rows": fold_rows[fold],
            "train_rows": 195,
            "correct": correct[fold],
            "accuracy": round(correct[fold] / fold_rows[fold], 4),
            "fit_seconds": 80.0,
            "latency_ms": {"p50": 9.0, "p95": 11.0, "mean": 9.5},
        }
        for fold in folds
    ]
    pooled = accuracy(correct, fold_rows, folds)
    halves = {}
    for name, wanted in (("selection", SELECTION), ("validation", VALIDATION)):
        present = [fold for fold in wanted if fold in folds]
        value = accuracy(correct, fold_rows, present) if present else None
        mean_fold = (
            sum(correct[f] / fold_rows[f] for f in present) / len(present) if present else None
        )
        halves[name] = {
            "folds": wanted,
            "folds_run": present,
            "complete": len(present) == len(wanted),
            "missing_folds": [f for f in wanted if f not in present],
            "n_folds": len(present),
            "n_rows": sum(fold_rows[f] for f in present),
            "grouped_accuracy": None if value is None else round(value, 4),
            "mean_fold_accuracy": None if mean_fold is None else round(mean_fold, 4),
            "per_fold_accuracy": {
                str(f): round(correct[f] / fold_rows[f], 4) for f in present
            },
            "baseline_per_fold_accuracy": {
                str(f): round(incumbent_correct[f] / fold_rows[f], 4) for f in present
            },
            "baseline_mean_fold_accuracy": (
                round(
                    sum(incumbent_correct[f] / fold_rows[f] for f in present) / len(present), 4
                )
                if present
                else None
            ),
        }
    payload = {
        "schema": "survey-sweep/1",
        "phase": 4,
        "model_id": model_id,
        "status": status,
        "error": None if status == "ok" else "fixture failure",
        "failed_stage": None if status == "ok" else "load",
        "measured_at": "2026-11-23T15:04:50-0500",
        "environment": {
            "hostname": "fixture",
            "platform": "fixture",
            "python": "3.12.14",
            "cpu_count": 12,
            "torch": "2.15.0+cpu",
            "torch_threads": 10,
            "loadavg_1m": 1.0,
        },
        "load": {"status": "ok", "load_ms": 300.0, "error": None},
        "model": {"params": 22713216, "embedding_dim": 384, "max_seq_length": 256,
                  "license": "apache-2.0", "snapshot_bytes": 91578415},
        "bench": {"n_rows": 382, "n_families": 110, "n_labels": 18, "n_reference_rows": 200},
        "fold_assignments": {"5": {"n_folds": 5, "rows_per_fold": [fold_rows[f] for f in folds]}},
        "recipe": {"name": "rung-3-setfit", "seed": 0},
        "folds_run": list(folds),
        "halves": halves,
        "runs": [
            {
                "rung": "r3-setfit",
                "protocol": "grouped",
                "fold_count": len(folds),
                "n_folds": len(folds),
                "complete": len(folds) == len(FOLDS),
                "folds": list(folds),
                "fold_rows": [fold_rows[f] for f in folds],
                "n_rows": sum(fold_rows[f] for f in folds),
                "per_fold_accuracy": [round(correct[f] / fold_rows[f], 4) for f in folds],
                "per_fold": per_fold,
                "accuracy": round(pooled if grouped_pooled is None else grouped_pooled, 4),
                "ece": 0.5,
                "none_recall": 0.84,
                "false_none_rate": 0.0056,
                "ledger": {"__none__": {"n": 25, "correct": 21, "recall": 0.84}},
                "latency_ms": {"p50": 9.0, "p95": 11.0, "mean": 9.5},
                "fold_spread": {"mean": pooled, "min": 0.9, "max": 1.0, "sd": 0.04},
            },
            {
                "rung": "r3-setfit",
                "protocol": "rowwise",
                "fold_count": None,
                "per_fold_accuracy": [],
                "accuracy": rowwise_pooled,
                "none_recall": 0.84,
                "latency_ms": {"p50": 9.0, "p95": 11.0, "mean": 9.5},
            },
        ],
        "latency_passes": {
            "passes": 5,
            "p50_ms": 9.711,
            "p95_ms": 13.637,
            "conditions": {"torch_threads": 10, "loadavg_1m": 1.0},
        },
        "incumbent": {
            "rung": "r3-setfit-minilm",
            "encoder": INCUMBENT_ID,
            "grouped_accuracy": 0.9398,
            "rowwise_accuracy": 0.9424,
            "per_fold_accuracy": [
                round(incumbent_correct[f] / fold_rows[f], 4) for f in sorted(folds)
            ],
            "latency_p50_ms": 7.944,
            "none_recall": 0.84,
        },
        "determinism": {"checked": True, "identical_predictions": True, "rows_differing": 0},
        "provenance": {"reproduces_artifact": True},
    }
    return payload


def build_probe(
    model_id: str,
    *,
    p50: float | None = 6.33,
    p95: float | None = 8.01,
    status: str = "ok",
    error: str | None = None,
    failed_stage: str | None = None,
    torch_threads: int = 10,
    embed_per_row: float = 1.818,
    params: int = 22713216,
    dim: int = 384,
    max_seq: int = 256,
    licence: str | None = "apache-2.0",
    snapshot_bytes: int = 91578415,
    rss_delta: int = 29794304,
    passes: int = 5,
) -> dict:
    failed = status == "error"
    return {
        "schema": "survey-probe/1",
        "phase": 3,
        "model_id": model_id,
        "status": status,
        "error": error,
        "failed_stage": failed_stage,
        "measured_at": "2026-11-23T15:19:05-0500",
        "environment": {
            "hostname": "fixture",
            "platform": "Linux-fixture",
            "python": "3.12.14",
            "cpu_count": 12,
            "torch": "2.15.0+cpu",
            "torch_threads": torch_threads,
            "cuda_available": False,
            "loadavg_1m": 1.0,
        },
        "corpus": {"source": "fixture", "n_rows": 382, "truncated": False},
        "load": {
            "status": "error" if failed else "ok",
            "load_ms": None if failed else 389.3,
            "import_ms": None if failed else 3198.5,
            "error": error if failed else None,
        },
        "model": None
        if failed
        else {
            "params": params,
            "embedding_dim": dim,
            "max_seq_length": max_seq,
            "license": licence,
            "license_source": "fixture cardData",
            "license_error": None,
            "gated": False,
            "pipeline_tag": "sentence-similarity",
            "snapshot_path": f"<cache>/{model_id}",
            "snapshot_bytes": snapshot_bytes,
            "snapshot_files": 11,
            "snapshot_source": "fixture",
            "snapshot_error": None,
        },
        "embeddings": None
        if failed
        else {
            "n_rows": 382,
            "dim": dim,
            "dtype": "float32",
            "wall_ms": 694.5,
            "ms_per_row": embed_per_row,
            "mean_l2_norm": 1.0,
            "min_l2_norm": 1.0,
            "max_l2_norm": 1.0,
            "finite": True,
        },
        "memory": {
            "readings": {"start": 1, "after_import": 2, "before_load": 3, "after_load": 4,
                         "after_embed": 5, "after_latency": 6},
            "method": "fixture rss",
            "deltas_bytes": {"after_load_minus_before_load": None if failed else rss_delta},
        },
        "latency": None
        if failed
        else {
            "passes": passes,
            "n_sentences_per_pass": 382,
            "warmup_calls": 8,
            "pass_p50_ms": [p50] * passes if p50 is not None else [],
            "pass_p95_ms": [p95] * passes if p95 is not None else [],
            "p50_ms": p50,
            "p95_ms": p95,
            "p50_spread_ms": 0.1,
            "method": "fixture",
        },
        "conditions": {
            "protocol": "fixture",
            "passes_requested": passes,
            "warmup_calls": 8,
            "batch_size_single_call": 1,
            "embed_batch_size": 32,
            "latency_source": "fixture",
            "passes_recorded": 0 if failed else passes,
        },
        "stage_errors": {},
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run(script: str, args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [PYTHON, str(HERE / script), *args],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        timeout=300,
    )


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def phase4(case: str, root: Path, payloads: dict[str, dict], extra: list[str] | None = None):
    sweeps = root / "sweeps"
    for name, payload in payloads.items():
        write_json(sweeps / f"{name}.json", payload)
    out = root / "out"
    result = run(
        "report_phase4.py",
        ["--sweeps", str(sweeps / "*.json"), "--out-dir", str(out), *(extra or [])],
    )
    return result, out, load(out / "phase4-table.json"), (
        (out / "phase4-table.md").read_text(encoding="utf-8")
        if (out / "phase4-table.md").exists()
        else None
    )


def phase3(case: str, root: Path, payloads: dict[str, dict], extra: list[str] | None = None):
    probes = root / "probes"
    for name, payload in payloads.items():
        write_json(probes / f"probe-{name}.json", payload)
    out = root / "out"
    result = run(
        "report_phase3.py",
        ["--probes", str(probes / "probe-*.json"), "--out-dir", str(out), *(extra or [])],
    )
    return result, out, load(out / "phase3-table.json"), (
        (out / "phase3-table.md").read_text(encoding="utf-8")
        if (out / "phase3-table.md").exists()
        else None
    )


def row_of(payload: dict | None, model_id: str) -> dict | None:
    if not payload:
        return None
    for row in payload.get("models") or payload.get("rows") or []:
        if row.get("model_id") == model_id:
            return row
    return None


# --------------------------------------------------------------------------- cases


def case_beats_both_halves(tmp: Path) -> None:
    name = "p4 beats both halves"
    challenger = "fixture/beats-both-halves"
    # Selection: +2, +2, 0 rows. Validation: +2, +1 rows. Same sign, |mean| > sd.
    correct = {0: 73, 1: 71, 2: 76, 3: 71, 4: 75}
    result, _out, payload, md = phase4(
        name,
        tmp / "beats",
        {
            "incumbent": build_sweep(INCUMBENT_ID, dict(INCUMBENT_CORRECT)),
            "challenger": build_sweep(challenger, correct),
        },
    )
    check(name, result.returncode == 0, f"exit {result.returncode}: {result.stderr.strip()}")
    verdict = (payload or {}).get("verdict") or {}
    check(name, verdict.get("form") == "win", f"verdict form is {verdict.get('form')!r}")
    check(
        name,
        (verdict.get("winner") or {}).get("model_id") == challenger,
        "the selection-half winner is not the challenger",
    )
    check(name, "beats the incumbent by more than the fold spread" in (md or ""), "md lacks the win text")
    delta = (verdict.get("winner") or {}).get("validation_delta") or {}
    check(name, delta.get("mean_points") is not None, "no validation-half delta reported")
    check(
        name,
        delta.get("mean_points") is not None
        and delta.get("sd_points") is not None
        and abs(delta["mean_points"]) > delta["sd_points"],
        f"|mean| {delta.get('mean_points')} does not clear sd {delta.get('sd_points')}",
    )
    check(name, delta.get("folds_improved") == 2, "folds improved is not 2/2")
    check(
        name,
        payload is not None and payload["falsifier"] is not None,
        "no falsifier named",
    )
    check(
        name,
        "not a selection criterion" in (md or ""),
        "md does not label pooled accuracy as not a selection criterion",
    )
    falsifier = (payload or {}).get("falsifier") or {}
    check(name, falsifier.get("rows_changed") == 1, "falsifier is not a one-row change")
    print(f"  {name}: form={verdict.get('form')} delta={delta.get('mean_points')}±{delta.get('sd_points')}")


def case_selection_wins_validation_loses(tmp: Path) -> None:
    name = "p4 selection wins, validation loses"
    challenger = "fixture/selection-only"
    correct = {0: 76, 1: 74, 2: 76, 3: 69, 4: 70}
    result, _out, payload, md = phase4(
        name,
        tmp / "nontransfer",
        {
            "incumbent": build_sweep(INCUMBENT_ID, dict(INCUMBENT_CORRECT)),
            "challenger": build_sweep(challenger, correct),
        },
    )
    check(name, result.returncode == 0, f"exit {result.returncode}: {result.stderr.strip()}")
    verdict = (payload or {}).get("verdict") or {}
    check(name, verdict.get("form") == "ranking_does_not_transfer", f"form is {verdict.get('form')!r}")
    check(name, "the ranking does not transfer" in (md or ""), "md lacks the does-not-transfer text")
    check(name, verdict.get("top_differs") is True, "top_differs is not reported as true")
    check(
        name,
        (verdict.get("winner") or {}).get("model_id") == challenger,
        "the selection winner is not the challenger",
    )
    check(
        name,
        (verdict.get("validation_top") or {}).get("model_id") == INCUMBENT_ID,
        "the validation top is not the incumbent",
    )
    check(
        name,
        "Selection half (folds [0, 1, 2])" in (md or "")
        and "Validation half (folds [3, 4])" in (md or ""),
        "both orderings are not shown",
    )
    check(name, "incumbent for stability" in (md or "") or "for stability" in (md or ""), "no incumbent recommendation")
    print(f"  {name}: form={verdict.get('form')} selection={verdict.get('winner', {}).get('model_id')} validation={verdict.get('validation_top', {}).get('model_id')}")


def case_one_fold_disagrees(tmp: Path) -> None:
    name = "p4 one fold disagrees (inside sd)"
    challenger = "fixture/one-fold-disagrees"
    # fold 3: 76/76 (69+7); fold 4: 73/76 (74-1). Deltas +9.21 and -1.32 -> mean 3.95 < sd 7.44.
    correct = {0: 72, 1: 70, 2: 76, 3: 76, 4: 73}
    result, _out, payload, _md = phase4(
        name,
        tmp / "onerow",
        {
            "incumbent": build_sweep(INCUMBENT_ID, dict(INCUMBENT_CORRECT)),
            "challenger": build_sweep(challenger, correct),
        },
    )
    check(name, result.returncode == 0, f"exit {result.returncode}: {result.stderr.strip()}")
    verdict = (payload or {}).get("verdict") or {}
    check(name, verdict.get("form") != "win", f"a fold spread this wide produced a win ({verdict.get('form')})")
    delta = ((row_of(payload, challenger) or {}).get("paired_validation_half")) or {}
    check(
        name,
        delta.get("mean_points") is not None
        and delta.get("sd_points") is not None
        and abs(delta["mean_points"]) <= delta["sd_points"],
        f"the fixture did not land inside the sd: {delta.get('mean_points')}±{delta.get('sd_points')}",
    )
    check(
        name,
        (row_of(payload, challenger) or {}).get("clears_validation_spread") is False,
        "clears_validation_spread is not false",
    )
    print(f"  {name}: form={verdict.get('form')} delta={delta.get('mean_points')}±{delta.get('sd_points')}")


def case_magnitude_disagrees(tmp: Path) -> None:
    name = "p4 one fold agrees in sign, not size (inside sd)"
    challenger = "fixture/wide-magnitude"
    # fold 3: +7 rows (+9.21), fold 4: +1 row (+1.32). Both up, mean 5.26 < sd 5.58.
    correct = {0: 72, 1: 70, 2: 76, 3: 76, 4: 75}
    result, _out, payload, _md = phase4(
        name,
        tmp / "magnitude",
        {
            "incumbent": build_sweep(INCUMBENT_ID, dict(INCUMBENT_CORRECT)),
            "challenger": build_sweep(challenger, correct),
        },
    )
    check(name, result.returncode == 0, f"exit {result.returncode}: {result.stderr.strip()}")
    verdict = (payload or {}).get("verdict") or {}
    delta = ((row_of(payload, challenger) or {}).get("paired_validation_half")) or {}
    check(name, verdict.get("form") != "win", f"an inside-sd delta produced a win ({verdict.get('form')})")
    check(
        name,
        delta.get("sign_consistent") is True and abs(delta.get("mean_points") or 0) <= (delta.get("sd_points") or 0),
        f"the fixture did not land inside the sd: {delta}",
    )
    print(f"  {name}: form={verdict.get('form')} delta={delta.get('mean_points')}±{delta.get('sd_points')}")


def case_tie_at_zero(tmp: Path) -> None:
    name = "p4 tie at zero"
    challenger = "fixture/tie-at-zero"
    result, _out, payload, md = phase4(
        name,
        tmp / "tie",
        {
            "incumbent": build_sweep(INCUMBENT_ID, dict(INCUMBENT_CORRECT)),
            "challenger": build_sweep(challenger, dict(INCUMBENT_CORRECT)),
        },
    )
    check(name, result.returncode == 0, f"exit {result.returncode}: {result.stderr.strip()}")
    verdict = (payload or {}).get("verdict") or {}
    check(name, verdict.get("form") == "incumbent_stands", f"a tie produced {verdict.get('form')!r}")
    check(name, "nothing beats the incumbent; the incumbent stands" in (md or ""), "md lacks the incumbent-stands text")
    row = row_of(payload, challenger) or {}
    check(name, (row.get("paired_validation_half") or {}).get("mean_points") == 0.0, "the delta mean is not 0.0")
    check(name, row.get("wins_validation") is False, "a zero delta was reported as a win")
    ranking = (payload or {}).get("ranking_selection") or []
    check(
        name,
        ranking and ranking[0]["model_id"] == INCUMBENT_ID,
        f"the tie-break did not keep the incumbent first: {ranking[:1]}",
    )
    print(f"  {name}: form={verdict.get('form')} top={ranking[0]['model_id'] if ranking else None}")


def case_pooled_is_not_a_criterion(tmp: Path) -> None:
    name = "p4 pooled accuracy does not decide"
    challenger = "fixture/pooled-poisoned"
    correct = {0: 73, 1: 71, 2: 76, 3: 71, 4: 75}
    clean = build_sweep(challenger, correct)
    poisoned = build_sweep(challenger, correct, grouped_pooled=0.0, rowwise_pooled=0.0)
    incumbent = build_sweep(INCUMBENT_ID, dict(INCUMBENT_CORRECT))
    result_a, _a, plain, _mda = phase4(name, tmp / "pooled-clean", {"incumbent": incumbent, "challenger": clean})
    result_b, _b, poisoned_payload, mdb = phase4(
        name, tmp / "pooled-poisoned", {"incumbent": incumbent, "challenger": poisoned}
    )
    check(name, result_a.returncode == 0 and result_b.returncode == 0, "a run failed")
    verdict_a = (plain or {}).get("verdict") or {}
    verdict_b = (poisoned_payload or {}).get("verdict") or {}
    check(
        name,
        verdict_a.get("form") == verdict_b.get("form"),
        f"moving the pooled number changed the verdict: {verdict_a.get('form')} -> {verdict_b.get('form')}",
    )
    check(
        name,
        verdict_a.get("ranking_selection") == verdict_b.get("ranking_selection"),
        "moving the pooled number changed the selection ranking",
    )
    row = row_of(poisoned_payload, challenger) or {}
    check(
        name,
        (row.get("pooled_all_folds") or {}).get("not_a_selection_criterion") is True,
        "the JSON does not label pooled accuracy as not a selection criterion",
    )
    check(name, "0.0000" in (mdb or ""), "the poisoned pooled number is not shown in the table")
    print(f"  {name}: form unchanged ({verdict_a.get('form')}) with pooled 0.0000")


def case_missing_fold(tmp: Path) -> None:
    name = "p4 missing fold refuses"
    partial = build_sweep("fixture/missing-fold", {0: 73, 1: 71, 2: 76, 3: 71}, folds=(0, 1, 2, 3))
    result, out, payload, md = phase4(
        name,
        tmp / "missing",
        {"incumbent": build_sweep(INCUMBENT_ID, dict(INCUMBENT_CORRECT)), "partial": partial},
    )
    check(name, result.returncode == 2, f"exit {result.returncode}, expected 2")
    check(name, payload is None and md is None, "a refused sweep still wrote a table")
    check(
        name,
        "missing fold" in result.stderr and "not a full sweep" in result.stderr,
        f"the refusal does not name the missing fold: {result.stderr.strip()}",
    )
    check(name, not (out / "phase4-table.json").exists(), "phase4-table.json exists after a refusal")
    print(f"  {name}: exit=2 msg={result.stderr.strip().splitlines()[0][:110]}")


def case_single_model(tmp: Path) -> None:
    name = "p4 single model refuses"
    result, out, payload, md = phase4(
        name, tmp / "single", {"incumbent": build_sweep(INCUMBENT_ID, dict(INCUMBENT_CORRECT))}
    )
    check(name, result.returncode == 2, f"exit {result.returncode}, expected 2")
    check(name, payload is None and md is None, "a one-model 'sweep' was emitted")
    check(
        name,
        "at least two" in result.stderr,
        f"the refusal does not say a sweep needs two models: {result.stderr.strip()}",
    )
    print(f"  {name}: exit=2 msg={result.stderr.strip().splitlines()[0][:110]}")


def case_unpicked_model_clears(tmp: Path) -> None:
    name = "p4 unpicked model clears the spread"
    winner_by_selection = "fixture/selection-winner"
    elsewhere = "fixture/validation-winner"
    result, _out, payload, md = phase4(
        name,
        tmp / "unpicked",
        {
            "incumbent": build_sweep(INCUMBENT_ID, dict(INCUMBENT_CORRECT)),
            "a": build_sweep(winner_by_selection, {0: 76, 1: 74, 2: 76, 3: 70, 4: 74}),
            "b": build_sweep(elsewhere, {0: 72, 1: 70, 2: 76, 3: 76, 4: 76}),
        },
    )
    check(name, result.returncode == 0, f"exit {result.returncode}: {result.stderr.strip()}")
    verdict = (payload or {}).get("verdict") or {}
    check(
        name,
        verdict.get("form") == "ranking_does_not_transfer",
        f"an unpicked model clearing the spread produced {verdict.get('form')!r}",
    )
    check(
        name,
        (verdict.get("winner") or {}).get("model_id") == winner_by_selection,
        "the selection winner is not the model that topped selection",
    )
    check(
        name,
        elsewhere in (verdict.get("clearers") or []),
        f"the clearer is not named: {verdict.get('clearers')}",
    )
    print(f"  {name}: form={verdict.get('form')} clearers={verdict.get('clearers')}")


def case_failed_load_probe(tmp: Path) -> None:
    name = "p3 failed load appears with its error"
    message = "OSError: fixture/model is not a valid model identifier (404 Client Error)"
    result, _out, payload, md = phase3(
        name,
        tmp / "probe-fail",
        {
            "good": build_probe(INCUMBENT_ID),
            "broken": build_probe(
                "fixture/does-not-exist", status="error", error=message, failed_stage="load",
                p50=None, p95=None,
            ),
        },
    )
    check(name, result.returncode == 0, f"exit {result.returncode}: {result.stderr.strip()}")
    broken = row_of(payload, "fixture/does-not-exist") or {}
    check(name, bool(broken), "the failed model is missing from the table")
    check(name, broken.get("decision") == "drop", f"decision is {broken.get('decision')!r}")
    check(name, message in (broken.get("decision_reason") or ""), "the error text is not in the row")
    check(name, message in (md or ""), "the error text is not printed in the markdown")
    check(name, "## Failures" in (md or ""), "no failures section")
    print(f"  {name}: decision={broken.get('decision')} error-in-md={message in (md or '')}")


def case_over_ceiling(tmp: Path) -> None:
    name = "p3 over the ceiling drops with the threshold"
    result, _out, payload, md = phase3(
        name,
        tmp / "probe-slow",
        {
            "incumbent": build_probe(INCUMBENT_ID, p50=6.33),
            "slow": build_probe("fixture/slow", p50=12.0, p95=20.0, embed_per_row=3.5),
        },
    )
    check(name, result.returncode == 0, f"exit {result.returncode}: {result.stderr.strip()}")
    slow = row_of(payload, "fixture/slow") or {}
    fast = row_of(payload, INCUMBENT_ID) or {}
    check(name, slow.get("decision") == "drop", f"the slow model was {slow.get('decision')!r}")
    check(name, fast.get("decision") == "pass", f"the incumbent was {fast.get('decision')!r}")
    check(
        name,
        slow.get("threshold_ms") == 9.495 or slow.get("threshold_ms") == 9.49,
        f"threshold is {slow.get('threshold_ms')!r}",
    )
    check(name, "9.50" in (md or ""), "the ceiling is not printed in the markdown")
    print(f"  {name}: slow={slow.get('decision')} incumbent={fast.get('decision')} ceiling={slow.get('threshold_ms')}")


def case_conflicting_probes(tmp: Path) -> None:
    name = "p3 conflicting probes are not merged"
    result, _out, payload, md = phase3(
        name,
        tmp / "probe-conflict",
        {
            "a": build_probe("fixture/same-model", p50=6.0, torch_threads=10),
            "b": build_probe("fixture/same-model", p50=14.0, torch_threads=4),
        },
    )
    check(name, result.returncode == 0, f"exit {result.returncode}: {result.stderr.strip()}")
    conflicts = (payload or {}).get("conflicts") or []
    check(name, len(conflicts) == 1, f"{len(conflicts)} conflicts recorded, expected 1")
    check(name, "CONFLICT" in result.stderr, "the conflict was not announced on stderr")
    rows = [row for row in (payload or {}).get("rows") or [] if row.get("model_id") == "fixture/same-model"]
    check(name, len(rows) == 2, f"{len(rows)} rows for the conflicted model, expected 2")
    check(name, all(row.get("decision") == "unresolved" for row in rows), "a conflicted row was decided")
    check(name, "## Conflicting probes" in (md or ""), "no conflicts section")
    check(name, "torch_threads" in (md or ""), "the differing field is not named")
    check(name, "6.00" in (md or "") and "14.00" in (md or ""), "both probes' latencies are not shown")
    print(f"  {name}: rows={len(rows)} decisions={[row.get('decision') for row in rows]}")


def case_duplicate_identical(tmp: Path) -> None:
    name = "p3 identical duplicate is named, not doubled"
    result, _out, payload, _md = phase3(
        name,
        tmp / "probe-dup",
        {
            "a": build_probe(INCUMBENT_ID, p50=6.33),
            "b": build_probe(INCUMBENT_ID, p50=6.40),
        },
    )
    check(name, result.returncode == 0, f"exit {result.returncode}: {result.stderr.strip()}")
    rows = [row for row in (payload or {}).get("rows") or [] if row.get("model_id") == INCUMBENT_ID]
    check(name, len(rows) == 1, f"{len(rows)} rows for the duplicated model, expected 1")
    notes = " ".join((payload or {}).get("notes") or [])
    check(name, "identical conditions" in notes, "the duplicate is not named in the notes")
    check(name, not (payload or {}).get("conflicts"), "identical conditions were reported as a conflict")
    print(f"  {name}: rows={len(rows)} note={'identical conditions' in notes}")


def case_unreadable_sweep(tmp: Path) -> None:
    name = "p4 unreadable sweep refuses"
    sweeps = tmp / "unreadable" / "sweeps"
    write_json(sweeps / "incumbent.json", build_sweep(INCUMBENT_ID, dict(INCUMBENT_CORRECT)))
    write_json(
        sweeps / "challenger.json", build_sweep("fixture/fine", {0: 73, 1: 71, 2: 76, 3: 71, 4: 75})
    )
    (sweeps / "truncated.json").write_text("{not json", encoding="utf-8")
    out = tmp / "unreadable" / "out"
    result = run("report_phase4.py", ["--sweeps", str(sweeps / "*.json"), "--out-dir", str(out)])
    check(name, result.returncode == 2, f"exit {result.returncode}, expected 2")
    check(
        name,
        "JSONDecodeError" in result.stderr or "cannot read" in result.stderr,
        f"the refusal does not name the parse error: {result.stderr.strip()[:120]}",
    )
    check(name, not (out / "phase4-table.json").exists(), "a table was written despite the refusal")
    print(f"  {name}: exit=2 msg={result.stderr.strip().splitlines()[0][:100]}")


def case_failed_sweep_is_listed(tmp: Path) -> None:
    name = "p4 failed sweep is listed, not ranked"
    broken = build_sweep("fixture/broken", {0: 1, 1: 1, 2: 1, 3: 1, 4: 1}, status="error")
    broken["error"] = "RuntimeError: tokenizer exploded"
    broken["failed_stage"] = "load"
    result, _out, payload, md = phase4(
        name,
        tmp / "failed-sweep",
        {
            "incumbent": build_sweep(INCUMBENT_ID, dict(INCUMBENT_CORRECT)),
            "challenger": build_sweep("fixture/challenger", {0: 73, 1: 71, 2: 76, 3: 71, 4: 75}),
            "broken": broken,
        },
    )
    check(name, result.returncode == 0, f"exit {result.returncode}: {result.stderr.strip()}")
    failed = (payload or {}).get("failed_sweeps") or []
    check(name, [entry["model_id"] for entry in failed] == ["fixture/broken"], f"failed list is {failed}")
    ranked = (payload or {}).get("models") or []
    check(
        name,
        all(model["model_id"] != "fixture/broken" for model in ranked),
        "the failed sweep was ranked",
    )
    check(
        name,
        "Failed sweeps" in (md or "") and "tokenizer exploded" in (md or ""),
        "the failed sweep and its error are not in the markdown",
    )
    print(f"  {name}: failed={[e['model_id'] for e in failed]} ranked={len(ranked)}")


def case_partial_sweep_is_listed(tmp: Path) -> None:
    name = "p4 partial sweep is listed with what is missing"
    partial = build_sweep(
        "fixture/partial", {0: 71, 1: 69, 2: 76, 3: 69, 4: 74}, status="partial"
    )
    partial["failed_stage"] = "fold 2"
    partial["missing"] = ["fold 2", "rowwise"]
    result, _out, payload, md = phase4(
        name,
        tmp / "partial-sweep",
        {
            "incumbent": build_sweep(INCUMBENT_ID, dict(INCUMBENT_CORRECT)),
            "challenger": build_sweep("fixture/challenger", {0: 73, 1: 71, 2: 76, 3: 71, 4: 75}),
            "partial": partial,
        },
    )
    check(name, result.returncode == 0, f"exit {result.returncode}: {result.stderr.strip()}")
    failed = (payload or {}).get("failed_sweeps") or []
    entry = next((e for e in failed if e["model_id"] == "fixture/partial"), None)
    check(name, entry is not None, f"the partial sweep is not in the failed list: {failed}")
    check(name, entry is not None and entry["status"] == "partial", f"entry is {entry}")
    check(
        name,
        all(model["model_id"] != "fixture/partial" for model in (payload or {}).get("models") or []),
        "the partial sweep was ranked",
    )
    check(
        name,
        "`fixture/partial`" in (md or "") and "fold 2, rowwise" in (md or ""),
        "the markdown does not name the partial sweep and what is missing",
    )
    print(f"  {name}: failed={[e['model_id'] for e in failed]} missing={entry and entry.get('missing')}")


def case_incumbent_not_swept(tmp: Path) -> None:
    name = "p4 incumbent not swept uses the artefact's incumbent block"
    result, _out, payload, md = phase4(
        name,
        tmp / "no-incumbent",
        {
            "bge": build_sweep("BAAI/bge-small-en-v1.5", {0: 73, 1: 71, 2: 76, 3: 71, 4: 75}),
            "e5": build_sweep("intfloat/e5-small-v2", {0: 72, 1: 70, 2: 76, 3: 76, 4: 73}),
        },
    )
    check(name, result.returncode == 0, f"exit {result.returncode}: {result.stderr.strip()}")
    incumbent = (payload or {}).get("incumbent") or {}
    check(name, incumbent.get("model_id") == INCUMBENT_ID, f"incumbent is {incumbent.get('model_id')}")
    check(name, incumbent.get("swept_here") is False, "the reference incumbent claims to have been swept")
    check(
        name,
        abs((incumbent.get("validation_pooled") or 0) - 0.9408) < 1e-6,
        f"validation pooled is {incumbent.get('validation_pooled')}, expected 0.9408",
    )
    check(name, "not swept here" in (md or ""), "the table does not mark the reference row")
    check(
        name,
        ((payload or {}).get("verdict") or {}).get("form") in FORM_FORMS,
        "no verdict form",
    )
    print(f"  {name}: form={(payload or {}).get('verdict', {}).get('form')}")


FORM_FORMS = {"win", "incumbent_stands", "ranking_does_not_transfer"}


CASES = [
    ("p4 beats both halves", case_beats_both_halves),
    ("p4 selection wins, validation loses", case_selection_wins_validation_loses),
    ("p4 one fold disagrees (inside sd)", case_one_fold_disagrees),
    ("p4 magnitude disagreement (inside sd)", case_magnitude_disagrees),
    ("p4 tie at zero", case_tie_at_zero),
    ("p4 pooled accuracy does not decide", case_pooled_is_not_a_criterion),
    ("p4 missing fold refuses", case_missing_fold),
    ("p4 single model refuses", case_single_model),
    ("p4 unpicked model clears the spread", case_unpicked_model_clears),
    ("p4 unreadable sweep refuses", case_unreadable_sweep),
    ("p4 failed sweep is listed, not ranked", case_failed_sweep_is_listed),
    ("p4 partial sweep is listed with what is missing", case_partial_sweep_is_listed),
    ("p4 incumbent not swept uses the incumbent block", case_incumbent_not_swept),
    ("p3 failed load appears with its error", case_failed_load_probe),
    ("p3 over the ceiling drops with the threshold", case_over_ceiling),
    ("p3 conflicting probes are not merged", case_conflicting_probes),
    ("p3 identical duplicate is named, not doubled", case_duplicate_identical),
]


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="survey-report-test-"))
    print(f"fixtures: {root}")
    print(f"python:   {PYTHON}")
    print("")
    passed = 0
    for label, fn in CASES:
        before = len(FAILURES)
        try:
            fn(root)
        except Exception as exc:  # noqa: BLE001 - a crashing case is a failing case
            FAILURES.append(f"{label}: raised {type(exc).__name__}: {exc}")
        if len(FAILURES) == before:
            passed += 1
            print(f"PASS  {label}")
        else:
            print(f"FAIL  {label}")
            for failure in FAILURES[before:]:
                print(f"        {failure}")
    print("")
    print(f"{passed}/{len(CASES)} cases passed, {CHECKS} checks, {len(FAILURES)} failure(s)")
    if FAILURES:
        print("")
        for failure in FAILURES:
            print(f"  - {failure}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
