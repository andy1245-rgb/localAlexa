"""Phase 4 report: rank on the selection half, quote the validation half, emit one verdict.

    .venv-embed/bin/python experiments/survey/report_phase4.py \
        --sweeps 'results/survey-2026-11-23/sweeps/*.json'

Reads the artefacts ``setfit_sweep.py`` writes and nothing else. The brief
(``docs/base-model-survey-brief-2026-11-23.md``, "Phase 4 — the honest sweep")
allows exactly three verdicts, and this script can emit exactly those three:

1. ``model X beats the incumbent by more than the fold spread`` — only when the
   selection-half winner's per-fold delta on the **validation** half has
   ``|mean| > sd`` and the same sign in every validation fold. Both numbers are
   printed.
2. ``nothing beats the incumbent; the incumbent stands`` — when no model's
   validation-half delta clears its own spread and the two halves agree on the
   top model.
3. ``the ranking does not transfer`` — when the selection half's pick is not the
   model the validation half points at, or when a model that was *not* picked
   clears the spread on validation. Both orderings are printed. The
   recommendation is the incumbent, for stability.

The rule is applied in that order, and the order is on the record so a reader can
disagree with a stated precedence instead of guessing one.

**Pooled-across-all-folds accuracy never enters a decision.** It appears in the
table labelled "not a selection criterion", because it is the number a reader is
most likely to mistake for the decision number, and it is the number the ladder
(``docs/ladder-results-2026-11-10.md``, rung 6) demonstrated is a measure of
template coverage rather than of a classifier.

A sweep set that cannot support a verdict is refused rather than reported: fewer
than two models, a model missing a fold, models scored on different fold sets, or
an incumbent that cannot be identified all exit 2 with the reason and write
nothing.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev

SCHEMA = "survey-phase4-table/1"
GENERATOR = "experiments/survey/report_phase4.py"
DEFAULT_OUT_DIR = Path(__file__).resolve().parents[2] / "results" / "survey-2026-11-23"
DEFAULT_INCUMBENT = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_SELECTION_FOLDS = (0, 1, 2)
DEFAULT_VALIDATION_FOLDS = (3, 4)

VERDICT_WIN = "win"
VERDICT_INCUMBENT = "incumbent_stands"
VERDICT_NO_TRANSFER = "ranking_does_not_transfer"

FORM_LABEL = {
    VERDICT_WIN: "model X beats the incumbent by more than the fold spread",
    VERDICT_INCUMBENT: "nothing beats the incumbent; the incumbent stands",
    VERDICT_NO_TRANSFER: "the ranking does not transfer",
}


def headline(verdict: dict) -> str:
    """The brief's form, with the placeholder filled in when there is a model to name."""
    if verdict["form"] == VERDICT_WIN:
        return (
            f"model `{verdict['winner']['model_id']}` beats the incumbent by more than the fold "
            "spread"
        )
    return FORM_LABEL[verdict["form"]]


class Refusal(Exception):
    """The sweep set cannot support a verdict. The message is the whole answer."""


def expand_patterns(patterns: list[str]) -> list[Path]:
    found: list[Path] = []
    seen: set[str] = set()
    for pattern in patterns:
        hits = glob.glob(pattern)
        if not hits and Path(pattern).is_file():
            hits = [pattern]
        for hit in hits:
            path = Path(hit)
            key = str(path.resolve()) if path.exists() else str(path)
            if key in seen or not path.is_file():
                continue
            seen.add(key)
            found.append(path)
    return sorted(found, key=lambda p: str(p))


def load_json(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:  # noqa: BLE001
        raise Refusal(f"cannot read sweep artefact {path}: {type(exc).__name__}: {exc}") from exc
    if not isinstance(payload, dict):
        raise Refusal(f"sweep artefact {path} is not a JSON object")
    return payload


def grouped_run(payload: dict) -> dict | None:
    for run in payload.get("runs") or []:
        if run.get("protocol") == "grouped":
            return run
    return None


def rowwise_run(payload: dict) -> dict | None:
    for run in payload.get("runs") or []:
        if run.get("protocol") == "rowwise":
            return run
    return None


def fold_table(payload: dict) -> tuple[dict[int, float], dict[int, int], list[str]]:
    """Exact per-fold accuracy and row counts, from the per-fold detail where present."""
    problems: list[str] = []
    run = grouped_run(payload)
    if run is None:
        raise Refusal(f"{payload.get('model_id')}: no grouped run in the artefact")
    accuracies: dict[int, float] = {}
    rows: dict[int, int] = {}
    for detail in run.get("per_fold") or []:
        fold = int(detail["fold"])
        n_rows = int(detail["n_rows"])
        correct = int(detail["correct"])
        accuracies[fold] = correct / n_rows if n_rows else 0.0
        rows[fold] = n_rows
    listed = run.get("per_fold_accuracy") or []
    folds_run = [int(f) for f in (run.get("folds_scored") or payload.get("folds_run") or [])]
    by_fold = run.get("per_fold_accuracy_by_fold") or {}
    fold_rows = [int(n) for n in (run.get("fold_rows") or [])]
    for index, fold in enumerate(folds_run):
        if fold in accuracies:
            continue
        if str(fold) in by_fold:
            accuracies[fold] = float(by_fold[str(fold)])
            if index < len(fold_rows):
                rows[fold] = fold_rows[index]
        elif index < len(listed) and listed[index] is not None:
            accuracies[fold] = float(listed[index])
            if index < len(fold_rows):
                rows[fold] = fold_rows[index]
    if not accuracies:
        raise Refusal(f"{payload.get('model_id')}: no per-fold accuracy in the artefact")
    if not rows:
        problems.append(
            f"{payload.get('model_id')}: per-fold row counts absent, half means cannot be pooled"
        )
    return accuracies, rows, problems


def pool(accuracies: dict[int, float], rows: dict[int, int], folds: list[int]) -> float | None:
    total_rows = sum(rows.get(fold, 0) for fold in folds)
    if not total_rows:
        return None
    correct = sum(accuracies[fold] * rows[fold] for fold in folds if fold in rows)
    return correct / total_rows


def mean_of(accuracies: dict[int, float], folds: list[int]) -> float | None:
    present = [fold for fold in folds if fold in accuracies]
    if not present:
        return None
    return mean(accuracies[fold] for fold in present)


def paired(deltas: list[float]) -> dict:
    values = [100.0 * d for d in deltas]
    n = len(values)
    if not n:
        return {"n_folds": 0, "mean_points": None, "sd_points": None, "min_points": None,
                "max_points": None, "folds_improved": 0, "folds_worse": 0, "folds_tied": 0,
                "sign_consistent": False, "spread_defined": False}
    sd = stdev(values) if n > 1 else None
    return {
        "n_folds": n,
        "mean_points": round(mean(values), 2),
        "sd_points": None if sd is None else round(sd, 2),
        "min_points": round(min(values), 2),
        "max_points": round(max(values), 2),
        "folds_improved": sum(1 for v in values if v > 0),
        "folds_worse": sum(1 for v in values if v < 0),
        "folds_tied": sum(1 for v in values if v == 0),
        "sign_consistent": all(v > 0 for v in values) or all(v < 0 for v in values),
        "spread_defined": sd is not None,
    }


def ordering(models: list[dict], half: str, incumbent_id: str) -> list[dict]:
    """Rank by that half's pooled grouped accuracy, with a stated, stable tie-break."""
    def key(model: dict):
        block = model[half]
        pooled = block["pooled"]
        mean_fold = block["mean_fold"]
        return (
            -(pooled if pooled is not None else -1.0),
            -(mean_fold if mean_fold is not None else -1.0),
            0 if model["model_id"] == incumbent_id else 1,
            model["model_id"],
        )

    return sorted(models, key=key)


def decide(models: list[dict], incumbent: dict, selection_folds: list[int],
           validation_folds: list[int]) -> dict:
    """The whole selection: one function, so a perturbation re-runs all of it."""
    for model in models:
        model["selection"] = {
            "folds": selection_folds,
            "pooled": pool(model["per_fold"], model["fold_rows"], selection_folds),
            "mean_fold": mean_of(model["per_fold"], selection_folds),
            "n_rows": sum(model["fold_rows"].get(fold, 0) for fold in selection_folds),
        }
        model["validation"] = {
            "folds": validation_folds,
            "pooled": pool(model["per_fold"], model["fold_rows"], validation_folds),
            "mean_fold": mean_of(model["per_fold"], validation_folds),
            "n_rows": sum(model["fold_rows"].get(fold, 0) for fold in validation_folds),
        }
        model["delta_by_fold"] = {
            fold: model["per_fold"][fold] - incumbent["per_fold"][fold]
            for fold in sorted(model["per_fold"])
            if fold in incumbent["per_fold"]
        }
        model["paired_all"] = paired([model["delta_by_fold"][f] for f in sorted(model["delta_by_fold"])])
        model["paired_validation"] = paired(
            [model["delta_by_fold"][f] for f in validation_folds if f in model["delta_by_fold"]]
        )
        model["paired_selection"] = paired(
            [model["delta_by_fold"][f] for f in selection_folds if f in model["delta_by_fold"]]
        )
        validation_delta = model["paired_validation"]
        model["clears_validation_spread"] = bool(
            validation_delta["spread_defined"]
            and validation_delta["mean_points"] is not None
            and validation_delta["sd_points"] is not None
            and abs(validation_delta["mean_points"]) > validation_delta["sd_points"]
            and validation_delta["sign_consistent"]
        )
        model["wins_validation"] = bool(
            model["clears_validation_spread"] and validation_delta["mean_points"] > 0
        )
        model["loses_validation"] = bool(
            model["clears_validation_spread"] and validation_delta["mean_points"] < 0
        )

    ranking_selection = ordering(models, "selection", incumbent["model_id"])
    ranking_validation = ordering(models, "validation", incumbent["model_id"])
    winner = ranking_selection[0]
    validation_top = ranking_validation[0]

    clearers = [model for model in models if model["wins_validation"]]
    clearers_excluding_winner = [m for m in clearers if m["model_id"] != winner["model_id"]]
    top_differs = validation_top["model_id"] != winner["model_id"]

    if winner["wins_validation"]:
        form = VERDICT_WIN
    elif clearers_excluding_winner or top_differs:
        form = VERDICT_NO_TRANSFER
    else:
        form = VERDICT_INCUMBENT

    winner_validation_delta = winner["paired_validation"]
    return {
        "form": form,
        "label": FORM_LABEL[form],
        "winner": {
            "model_id": winner["model_id"],
            "selection_pooled": winner["selection"]["pooled"],
            "selection_mean_fold": winner["selection"]["mean_fold"],
            "validation_pooled": winner["validation"]["pooled"],
            "validation_mean_fold": winner["validation"]["mean_fold"],
            "validation_delta": winner_validation_delta,
            "selection_delta": winner["paired_selection"],
        },
        "validation_top": {
            "model_id": validation_top["model_id"],
            "validation_pooled": validation_top["validation"]["pooled"],
        },
        "incumbent": {
            "model_id": incumbent["model_id"],
            "validation_pooled": incumbent["validation"]["pooled"],
            "validation_mean_fold": incumbent["validation"]["mean_fold"],
            "selection_pooled": incumbent["selection"]["pooled"],
        },
        "ranking_selection": [
            {"model_id": m["model_id"], "pooled": m["selection"]["pooled"],
             "mean_fold": m["selection"]["mean_fold"]}
            for m in ranking_selection
        ],
        "ranking_validation": [
            {"model_id": m["model_id"], "pooled": m["validation"]["pooled"],
             "mean_fold": m["validation"]["mean_fold"]}
            for m in ranking_validation
        ],
        "clearers": [m["model_id"] for m in clearers],
        "top_differs": top_differs,
    }


def verdict_text(verdict: dict, validation_folds: list[int]) -> str:
    winner = verdict["winner"]
    delta = winner["validation_delta"]
    if verdict["form"] == VERDICT_WIN:
        return (
            f"model {winner['model_id']} beats the incumbent by more than the fold spread — "
            f"validation-half paired delta {delta['mean_points']:+.2f} ± {delta['sd_points']:.2f} "
            f"points on folds {validation_folds}, "
            f"improved in {delta['folds_improved']} of {delta['n_folds']} folds, "
            f"sign consistent ({'all up' if delta['folds_improved'] == delta['n_folds'] else 'mixed'}), "
            f"|mean| {abs(delta['mean_points']):.2f} > sd {delta['sd_points']:.2f}."
        )
    if verdict["form"] == VERDICT_INCUMBENT:
        if delta["mean_points"] is None:
            detail = "no model has a validation-half delta at all"
        elif not delta["spread_defined"]:
            detail = (
                f"{winner['model_id']}'s validation-half delta is {delta['mean_points']:+.2f} points, "
                "but a spread needs at least two validation folds"
            )
        else:
            detail = (
                f"the selection-half winner {winner['model_id']} moves "
                f"{delta['mean_points']:+.2f} ± {delta['sd_points']:.2f} points on the validation "
                f"half, which is inside its own spread"
            )
        return f"nothing beats the incumbent; the incumbent stands — {detail}."
    return (
        "the ranking does not transfer — the selection half picks "
        f"{verdict['winner']['model_id']} "
        f"({verdict['winner']['selection_pooled']:.4f}) and the validation half points at "
        f"{verdict['validation_top']['model_id']} "
        f"({verdict['validation_top']['validation_pooled']:.4f}). "
        "Recommend the incumbent for stability; do not crown a model on the rows that would have "
        "selected it."
    )


def find_falsifier(models: list[dict], incumbent: dict, selection_folds: list[int],
                   validation_folds: list[int], verdict: dict) -> dict | None:
    """What specific number would change this verdict, in one of two shapes.

    First the smallest single-fold row change on the validation half that flips it —
    exhaustive, so when it answers, it answers exactly ("fold 4 at 73/76 instead of
    74/76"). A win is usually one or two rows from the other side.

    When nothing single-fold flips it, the verdict needs rows on *every* validation
    fold at once (that is how a mean gets outside its own sd), so the second shape
    reports the smallest uniform row change per validation fold that flips it, in the
    direction that would break the current verdict.
    """
    exact = _single_fold_flip(models, incumbent, selection_folds, validation_folds, verdict)
    if exact is not None:
        return {"kind": "single_fold", **exact}
    direction = -1 if verdict["form"] == VERDICT_WIN else 1
    return _uniform_flip(models, incumbent, selection_folds, validation_folds, verdict, direction)


def _patched(models: list[dict], model_id: str, changes: dict[int, int]) -> list[dict]:
    patched = [
        {
            **model,
            "per_fold": dict(model["per_fold"]),
            "fold_rows": dict(model["fold_rows"]),
        }
        for model in models
    ]
    for model in patched:
        if model["model_id"] != model_id:
            continue
        for fold, correct in changes.items():
            model["per_fold"][fold] = correct / model["fold_rows"][fold]
    return patched


def _single_fold_flip(models: list[dict], incumbent: dict, selection_folds: list[int],
                      validation_folds: list[int], verdict: dict) -> dict | None:
    candidates: list[tuple[int, str, int, int, int]] = []
    for model in models:
        for fold in validation_folds:
            if fold not in model["per_fold"] or fold not in model["fold_rows"]:
                continue
            n_rows = model["fold_rows"][fold]
            actual = round(model["per_fold"][fold] * n_rows)
            for correct in range(0, n_rows + 1):
                if correct == actual:
                    continue
                candidates.append((abs(correct - actual), model["model_id"], fold, correct, n_rows))
    candidates.sort()
    for distance, model_id, fold, correct, n_rows in candidates:
        flipped = decide(
            _patched(models, model_id, {fold: correct}),
            incumbent,
            selection_folds,
            validation_folds,
        )
        if flipped["form"] != verdict["form"]:
            return {
                "model_id": model_id,
                "fold": fold,
                "correct": correct,
                "n_rows": n_rows,
                "accuracy": round(correct / n_rows, 4),
                "actual_accuracy": round(
                    next(m for m in models if m["model_id"] == model_id)["per_fold"][fold], 4
                ),
                "rows_changed": distance,
                "new_form": flipped["form"],
                "new_label": FORM_LABEL[flipped["form"]],
                "new_headline": headline(flipped),
            }
    return None


def _uniform_flip(models: list[dict], incumbent: dict, selection_folds: list[int],
                  validation_folds: list[int], verdict: dict, direction: int) -> dict | None:
    """Smallest `r` such that `r` more (or fewer) correct rows in every validation fold flips it."""
    for model in models:
        max_rows = max(
            (model["fold_rows"][fold] for fold in validation_folds if fold in model["fold_rows"]),
            default=0,
        )
        for rows in range(1, max_rows + 1):
            changes: dict[int, int] = {}
            out_of_range = False
            for fold in validation_folds:
                if fold not in model["per_fold"] or fold not in model["fold_rows"]:
                    out_of_range = True
                    break
                n_rows = model["fold_rows"][fold]
                correct = round(model["per_fold"][fold] * n_rows) + direction * rows
                if correct < 0 or correct > n_rows:
                    out_of_range = True
                    break
                changes[fold] = correct
            if out_of_range:
                break
            flipped = decide(
                _patched(models, model["model_id"], changes),
                incumbent,
                selection_folds,
                validation_folds,
            )
            if flipped["form"] != verdict["form"]:
                return {
                    "kind": "uniform_fold",
                    "model_id": model["model_id"],
                    "rows_per_fold": rows,
                    "direction": direction,
                    "new_form": flipped["form"],
                    "new_label": FORM_LABEL[flipped["form"]],
                    "new_headline": headline(flipped),
                }
    return None


def falsifier_text(falsifier: dict | None, validation_folds: list[int]) -> str:
    """One sentence naming the number that would change the verdict."""
    if falsifier is None:
        return (
            "no row change on the validation half flips this verdict: every accuracy between 0 and "
            "1 was tried one fold at a time, and no uniform per-fold change does either."
        )
    if falsifier.get("kind") == "single_fold":
        return (
            f"move {falsifier['model_id']}'s fold {falsifier['fold']} accuracy from "
            f"{falsifier['actual_accuracy']:.4f} to {falsifier['accuracy']:.4f} "
            f"({falsifier['correct']}/{falsifier['n_rows']}, {falsifier['rows_changed']} row"
            f"{'' if falsifier['rows_changed'] == 1 else 's'}) and the verdict becomes "
            f"'{falsifier.get('new_headline') or falsifier['new_label']}'."
        )
    direction = "more" if falsifier["direction"] > 0 else "fewer"
    return (
        f"give {falsifier['model_id']} {falsifier['rows_per_fold']} {direction} correct row"
        f"{'' if falsifier['rows_per_fold'] == 1 else 's'} in every validation fold "
        f"{validation_folds} and the verdict becomes "
        f"'{falsifier.get('new_headline') or falsifier['new_label']}'."
    )


def render_markdown(
    models: list[dict],
    verdict: dict,
    falsifier: dict | None,
    incumbent: dict,
    notes: list[str],
    warnings: list[str],
    inputs: list[Path],
    selection_folds: list[int],
    validation_folds: list[int],
    generated: str,
    failed: list[dict] | None = None,
) -> str:
    out: list[str] = []
    winner = verdict["winner"]
    delta = winner["validation_delta"]

    out.append("# Phase 4 — the honest sweep")
    out.append("")
    out.append(
        f"Generated {generated} by `{GENERATOR}` from {len(inputs)} sweep artefact(s). "
        "Selection is folds "
        f"{selection_folds} (the selection half); the answer is quoted on folds "
        f"{validation_folds} (the validation half), which selected nothing."
    )
    out.append("")
    out.append("## Verdict")
    out.append("")
    out.append(f"**{headline(verdict)}**")
    out.append("")
    out.append(verdict_text(verdict, validation_folds))
    out.append("")
    out.append("| number | value |")
    out.append("|---|---|")
    out.append(
        f"| selection-half winner (ranked on folds {selection_folds} only) | "
        f"`{winner['model_id']}` |"
    )
    out.append(
        f"| winner's selection-half pooled grouped accuracy | "
        f"{winner['selection_pooled']:.4f} |"
    )
    out.append(
        f"| winner's validation-half pooled grouped accuracy | "
        f"{winner['validation_pooled']:.4f} |"
    )
    out.append(
        f"| incumbent's validation-half pooled grouped accuracy | "
        f"{verdict['incumbent']['validation_pooled']:.4f} |"
    )
    out.append(
        "| winner's validation-half paired delta vs incumbent | "
        + (
            f"{delta['mean_points']:+.2f} ± {delta['sd_points']:.2f} points "
            f"(folds {validation_folds}, improved {delta['folds_improved']}/{delta['n_folds']}) |"
            if delta["mean_points"] is not None and delta["sd_points"] is not None
            else "— |"
        )
    )
    out.append(
        f"| |mean| > sd on the validation half | "
        + (
            f"{'yes' if abs(delta['mean_points']) > delta['sd_points'] else 'no'} "
            f"({abs(delta['mean_points']):.2f} vs {delta['sd_points']:.2f}) |"
            if delta["mean_points"] is not None and delta["sd_points"] is not None
            else "— |"
        )
    )
    out.append(
        f"| sign consistent across validation folds | {'yes' if delta['sign_consistent'] else 'no'} |"
    )
    out.append("")
    out.append(
        "**Not a selection criterion:** pooled-across-all-folds accuracy (the `grouped pooled` and "
        "`rowwise pooled` columns below) entered none of these decisions. It is the number the "
        "ladder's rung 6 showed measures template coverage, not a classifier, and the same rows "
        "that selected the winner would be scoring it."
    )
    out.append("")

    out.append("## The three forms, and which one fired")
    out.append("")
    out.append(
        "Checked in this order, and the first that fires is the verdict: (1) the selection-half "
        "winner clears its own validation spread with one consistent sign; (2) no such win, and "
        "the halves disagree about which model is best; (3) neither, so the incumbent stands. "
        "The order is stated because a verdict is only checkable against a stated rule."
    )
    out.append("")
    out.append("| form | condition in this script | fired |")
    out.append("|---|---|---|")
    out.append(
        "| a model beats the incumbent by more than the fold spread | selection-half winner's "
        "validation paired delta has \\|mean\\| > sd and one consistent sign | "
        f"{'**yes**' if verdict['form'] == VERDICT_WIN else 'no'} |"
    )
    out.append(
        "| the ranking does not transfer | no such win, and the validation half's top model is not "
        "the selection half's pick (or a model that was not picked clears the spread) | "
        f"{'**yes**' if verdict['form'] == VERDICT_NO_TRANSFER else 'no'} |"
    )
    out.append(
        "| nothing beats the incumbent; the incumbent stands | no model clears its own spread and "
        "both halves agree on the top model | "
        f"{'**yes**' if verdict['form'] == VERDICT_INCUMBENT else 'no'} |"
    )
    out.append("")

    out.append("## Rankings")
    out.append("")
    out.append(f"### Selection half (folds {selection_folds}) — the only ordering that selects")
    out.append("")
    out.append("| # | model | pooled | mean of folds |")
    out.append("|---:|---|---:|---:|")
    for index, entry in enumerate(verdict["ranking_selection"], start=1):
        out.append(
            f"| {index} | `{entry['model_id']}` | {entry['pooled']:.4f} | {entry['mean_fold']:.4f} |"
        )
    out.append("")
    out.append(f"### Validation half (folds {validation_folds}) — reporting only, selects nothing")
    out.append("")
    out.append("| # | model | pooled | mean of folds |")
    out.append("|---:|---|---:|---:|")
    for index, entry in enumerate(verdict["ranking_validation"], start=1):
        out.append(
            f"| {index} | `{entry['model_id']}` | {entry['pooled']:.4f} | {entry['mean_fold']:.4f} |"
        )
    out.append("")
    if verdict["top_differs"]:
        out.append(
            "The two orderings disagree on the top model: "
            f"selection says `{winner['model_id']}`, validation says "
            f"`{verdict['validation_top']['model_id']}`. That disagreement is the finding, not a "
            "detail — it is why no model is recommended on the strength of these folds."
        )
        out.append("")
    else:
        out.append(
            f"Both halves rank `{winner['model_id']}` first, so the ordering itself transferred. "
            "Whether the *size* of the win clears the fold spread is the verdict above."
        )
        out.append("")

    out.append("## Per model")
    out.append("")
    out.append(
        "| model | sel pooled | sel mean/fold | val pooled | val mean/fold | paired Δ all folds | "
        "sd | folds improved | `__none__` recall | p50 ms | grouped pooled (**not a selection "
        "criterion**) | rowwise pooled (**not a selection criterion**) |"
    )
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for model in sorted(models, key=lambda m: -(m["selection"]["pooled"] or -1.0)):
        all_delta = model["paired_all"]
        out.append(
            "| `{model}` | {sel_pooled} | {sel_mean} | {val_pooled} | {val_mean} | {delta} | {sd} | "
            "{improved}/{n} | {none} | {p50} | {grouped} | {rowwise} |".format(
                model=model["model_id"]
                + ("" if model.get("swept_here", True) else " (reference, not swept here)"),
                sel_pooled=_f(model["selection"]["pooled"], 4),
                sel_mean=_f(model["selection"]["mean_fold"], 4),
                val_pooled=_f(model["validation"]["pooled"], 4),
                val_mean=_f(model["validation"]["mean_fold"], 4),
                delta=_signed(all_delta["mean_points"]),
                sd=_f(all_delta["sd_points"], 2),
                improved=all_delta["folds_improved"],
                n=all_delta["n_folds"],
                none=_f(model.get("none_recall"), 2),
                p50=_f(model.get("p50_ms"), 2),
                grouped=_f(model.get("grouped_pooled"), 4),
                rowwise=_f(model.get("rowwise_pooled"), 4),
            )
        )
    out.append("")
    out.append(
        "*`paired Δ all folds` is the ladder's statistic: the mean and sd of the per-fold "
        "difference against the incumbent, in points, over every fold that ran — not the difference "
        "of two pooled means. A mean smaller than its own sd is not a win; fold difficulty varies by "
        "more than the effect being measured.*"
    )
    out.append("")

    out.append("## Per-fold accuracy and per-fold delta")
    out.append("")
    folds = sorted({fold for model in models for fold in model["per_fold"]})
    header = "| model | " + " | ".join(f"f{fold}" for fold in folds) + " |"
    out.append("### Accuracy")
    out.append("")
    out.append(header)
    out.append("|---|" + "---:|" * len(folds))
    for model in sorted(models, key=lambda m: -(m["selection"]["pooled"] or -1.0)):
        cells = " | ".join(_f(model["per_fold"].get(fold), 4) for fold in folds)
        out.append(f"| `{model['model_id']}` | {cells} |")
    out.append("")
    out.append("### Delta vs the incumbent, points")
    out.append("")
    out.append(header)
    out.append("|---|" + "---:|" * len(folds))
    for model in sorted(models, key=lambda m: -(m["selection"]["pooled"] or -1.0)):
        cells = " | ".join(
            _signed(None if fold not in model["delta_by_fold"] else 100.0 * model["delta_by_fold"][fold])
            for fold in folds
        )
        out.append(f"| `{model['model_id']}` | {cells} |")
    out.append("")
    out.append(
        f"Selection folds {selection_folds}, validation folds {validation_folds}. "
        f"The incumbent is `{incumbent['model_id']}`; its own row is the zero line."
    )
    out.append("")

    out.append("## What would falsify this")
    out.append("")
    out.append(falsifier_text(falsifier, validation_folds))
    out.append("")
    out.append(
        "That is the specific number to check: the per-fold rows are in the table above, so the "
        "arithmetic can be redone by hand."
    )
    out.append("")

    if warnings:
        out.append("## Warnings")
        out.append("")
        for warning in warnings:
            out.append(f"- {warning}")
        out.append("")

    if failed:
        out.append("## Failed sweeps — not ranked")
        out.append("")
        out.append(
            "These artefacts carry `status: error` or `status: partial`: they are rows, not "
            "gaps, and they were excluded from the ranking rather than counted as a zero. A "
            "partial row carries the folds that did measure; `missing` names the rest."
        )
        out.append("")
        for entry in failed:
            out.append(f"- `{entry['model_id']}` — status `{entry.get('status')}`, "
                       f"stage `{entry.get('failed_stage')}`")
            if entry.get("missing"):
                out.append(f"  missing: {', '.join(str(m) for m in entry['missing'])}")
            out.append("")
            out.append("  ```")
            out.append(f"  {str(entry.get('error')).strip()}")
            out.append("  ```")
        out.append("")

    if notes:
        out.append("## Notes")
        out.append("")
        for note in notes:
            out.append(f"- {note}")
        out.append("")

    out.append("## Inputs")
    out.append("")
    for path in inputs:
        out.append(f"- `{path}`")
    out.append("")
    return "\n".join(out)


def _f(value, digits: int) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)


def _signed(value) -> str:
    if value is None:
        return "—"
    return f"{value:+.2f}"


def _incumbent_reference(per_fold, expected_folds: list[int], encoder: str) -> dict[int, float]:
    """The incumbent's per-fold accuracy from an artefact's `incumbent` block, list or mapping."""
    if isinstance(per_fold, dict):
        resolved = {int(key): float(value) for key, value in per_fold.items()}
    else:
        resolved = {fold: float(value) for fold, value in zip(expected_folds, per_fold)}
    missing = [fold for fold in expected_folds if fold not in resolved]
    if missing:
        raise Refusal(
            f"the `incumbent` block names `{encoder}` but has no accuracy for fold(s) {missing}; "
            "nothing can be paired against a partial incumbent."
        )
    return resolved


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 4: rank on the selection half, quote the validation half, emit one verdict."
        )
    )
    parser.add_argument(
        "--sweeps",
        nargs="+",
        action="extend",
        required=True,
        help="glob pattern(s) for sweep JSONs, e.g. 'results/survey-2026-11-23/sweeps/*.json'",
    )
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help=f"where phase4-table.md and phase4-table.json are written (default {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--incumbent",
        default=DEFAULT_INCUMBENT,
        help=f"model id of the incumbent everything is paired against (default {DEFAULT_INCUMBENT})",
    )
    parser.add_argument(
        "--selection-folds",
        default=",".join(str(f) for f in DEFAULT_SELECTION_FOLDS),
        help="folds that may select (default 0,1,2)",
    )
    parser.add_argument(
        "--validation-folds",
        default=",".join(str(f) for f in DEFAULT_VALIDATION_FOLDS),
        help="folds the answer is quoted on (default 3,4)",
    )
    args = parser.parse_args(argv)

    try:
        selection_folds = [int(part) for part in args.selection_folds.split(",") if part.strip()]
        validation_folds = [int(part) for part in args.validation_folds.split(",") if part.strip()]
    except ValueError:
        print("--selection-folds and --validation-folds take comma-separated integers", file=sys.stderr)
        return 2
    if not selection_folds or not validation_folds:
        print("both halves need at least one fold", file=sys.stderr)
        return 2
    overlap = set(selection_folds) & set(validation_folds)
    if overlap:
        print(f"the halves overlap on folds {sorted(overlap)}; that would select on the answer", file=sys.stderr)
        return 2

    inputs = expand_patterns(args.sweeps)
    if not inputs:
        print(f"no sweep artefacts matched {args.sweeps!r} — nothing to report", file=sys.stderr)
        return 2

    notes: list[str] = []
    warnings: list[str] = []
    models: list[dict] = []
    failed: list[dict] = []
    seen_ids: dict[str, str] = {}
    expected_folds = sorted(set(selection_folds) | set(validation_folds))
    fold_rows_reference: dict[int, int] | None = None

    try:
        for path in inputs:
            payload = load_json(path)
            model_id = payload.get("model_id") or path.stem.replace("__", "/")
            if model_id in seen_ids:
                raise Refusal(
                    f"two sweeps for model id `{model_id}`: "
                    f"`{seen_ids[model_id]}` and `{path}`. Rank one of them, not both."
                )
            seen_ids[model_id] = str(path)
            if payload.get("status") not in (None, "ok"):
                failed.append(
                    {
                        "model_id": model_id,
                        "path": str(path),
                        "status": payload.get("status"),
                        "failed_stage": payload.get("failed_stage"),
                        "missing": payload.get("missing"),
                        "error": payload.get("error"),
                    }
                )
                continue

            accuracies, rows, problems = fold_table(payload)
            folds_run = sorted(int(f) for f in (payload.get("folds_run") or sorted(accuracies)))
            missing = [fold for fold in expected_folds if fold not in accuracies]
            extra = [fold for fold in folds_run if fold not in expected_folds]
            if missing or extra:
                detail = []
                if missing:
                    detail.append(f"missing fold(s) {missing}")
                if extra:
                    detail.append(f"unexpected fold(s) {extra}")
                raise Refusal(
                    f"`{model_id}` is not a full sweep: {'; '.join(detail)}. "
                    f"folds_run={folds_run}, expected folds {expected_folds} "
                    f"({path}). A partial run is not a sweep, and a sweep with a hole in it "
                    "cannot be ranked against the others."
                )
            if problems:
                raise Refusal(f"`{model_id}`: {problems[0]} ({path})")
            if fold_rows_reference is None:
                fold_rows_reference = {fold: rows[fold] for fold in expected_folds}
            elif any(fold_rows_reference[fold] != rows[fold] for fold in expected_folds):
                raise Refusal(
                    f"`{model_id}` was scored on different fold sizes than the other models "
                    f"({[rows[fold] for fold in expected_folds]} vs "
                    f"{[fold_rows_reference[fold] for fold in expected_folds]}); the fold "
                    "assignment is not shared, so the numbers are not comparable."
                )

            run_grouped = grouped_run(payload) or {}
            run_rowwise = rowwise_run(payload) or {}
            halves = payload.get("halves") or {}
            latency_passes = payload.get("latency_passes") or {}
            p50 = latency_passes.get("p50_ms")
            p50_source = "latency_passes.p50_ms (dedicated pass)"
            if p50 is None:
                p50 = (run_grouped.get("latency_ms") or {}).get("p50")
                p50_source = "runs[grouped].latency_ms.p50 (in-fold timing — context, not a measurement)"
            none_recall = run_grouped.get("none_recall")
            if none_recall is None:
                none_recall = ((run_grouped.get("ledger") or {}).get("__none__") or {}).get("recall")

            model = {
                "model_id": model_id,
                "path": str(path),
                "measured_at": payload.get("measured_at"),
                "per_fold": {fold: accuracies[fold] for fold in expected_folds},
                "fold_rows": {fold: rows[fold] for fold in expected_folds},
                "grouped_pooled": run_grouped.get("accuracy"),
                "rowwise_pooled": run_rowwise.get("accuracy"),
                "none_recall": none_recall,
                "false_none_rate": run_grouped.get("false_none_rate"),
                "ece": run_grouped.get("ece"),
                "p50_ms": p50,
                "p50_source": p50_source,
                "artifact_halves": {
                    half: {
                        "pooled": (halves.get(half) or {}).get("grouped_accuracy"),
                        "mean_fold": (halves.get(half) or {}).get("mean_fold_accuracy"),
                    }
                    for half in ("selection", "validation")
                },
                "environment": payload.get("environment") or {},
                "payload_incumbent": payload.get("incumbent") or {},
            }
            models.append(model)

        if len(models) < 2:
            names = ", ".join(f"`{m['model_id']}`" for m in models) or "none"
            failed_text = ""
            if failed:
                failed_text = " Failed sweeps: " + "; ".join(
                    f"`{f['model_id']}` ({f['failed_stage']}: {f['error']})" for f in failed
                )
            raise Refusal(
                f"{len(models)} rankable model(s) in the sweep set ({names}). A sweep needs at "
                "least two — the incumbent and a challenger — before a ranking means anything."
                + failed_text
            )

        incumbent_model = next(
            (model for model in models if model["model_id"] == args.incumbent), None
        )
        if incumbent_model is None:
            source = models[0]["payload_incumbent"]
            encoder = source.get("encoder")
            per_fold = source.get("per_fold_accuracy")
            rooted = models[0]["fold_rows"]
            if not encoder or not per_fold:
                raise Refusal(
                    f"the incumbent `{args.incumbent}` is not in the sweep set and no artefact "
                    "carries an `incumbent` block with its per-fold accuracy, so nothing can be "
                    "paired against it."
                )
            incumbent_model = {
                "model_id": encoder,
                "path": models[0]["path"],
                "per_fold": _incumbent_reference(per_fold, expected_folds, encoder),
                "fold_rows": dict(rooted),
                "reference_only": True,
            }
            notes.append(
                f"the incumbent `{encoder}` was not swept here; its per-fold accuracy is read from "
                f"the `incumbent` block of `{models[0]['path']}` (the ladder's rung 3, rounded to "
                "4 dp there) and its row counts from the same fold assignment."
            )
        models_for_decision = (
            models if incumbent_model in models else [*models, incumbent_model]
        )
        incumbent_for_decision = next(
            model for model in models_for_decision if model["model_id"] == incumbent_model["model_id"]
        )

        verdict = decide(models_for_decision, incumbent_for_decision, selection_folds, validation_folds)
        falsifier = find_falsifier(
            models_for_decision, incumbent_for_decision, selection_folds, validation_folds, verdict
        )

        for model in models:
            recomputed = model["artifact_halves"]
            for half, folds in (("selection", selection_folds), ("validation", validation_folds)):
                shipped = recomputed[half]["pooled"]
                ours = model[half]["pooled"]
                if shipped is not None and ours is not None and abs(shipped - ours) > 5e-4:
                    warnings.append(
                        f"`{model['model_id']}`: {half}-half pooled accuracy recomputed as "
                        f"{ours:.4f} but the artefact reports {shipped:.4f}; this report uses the "
                        "recomputed value from the per-fold rows."
                    )
        for model in models:
            env = model["environment"]
            if env.get("loadavg_1m") is not None and env["loadavg_1m"] > (env.get("cpu_count") or 1):
                warnings.append(
                    f"`{model['model_id']}` ran at loadavg {env['loadavg_1m']} on "
                    f"{env.get('cpu_count')} CPUs — its latency is context, not a measurement."
                )
    except Refusal as refusal:
        print(f"REFUSED: {refusal}", file=sys.stderr)
        print("no phase4-table.md or phase4-table.json written", file=sys.stderr)
        stale = [
            str(candidate)
            for candidate in (Path(args.out_dir) / "phase4-table.md", Path(args.out_dir) / "phase4-table.json")
            if candidate.exists()
        ]
        if stale:
            print(
                "an earlier run left " + ", ".join(stale)
                + " in place — it does not describe this sweep set; delete it before quoting it",
                file=sys.stderr,
            )
        return 2

    generated = datetime.now().astimezone().replace(microsecond=0).isoformat()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "phase4-table.json"
    md_path = out_dir / "phase4-table.md"

    for model in models_for_decision:
        model.pop("payload_incumbent", None)
        model["swept_here"] = not model.get("reference_only", False)

    payload = {
        "schema": SCHEMA,
        "generated_at": generated,
        "generator": GENERATOR,
        "command": " ".join(["report_phase4.py", *sys.argv[1:]]),
        "rule": {
            "selection_folds": selection_folds,
            "validation_folds": validation_folds,
            "rank_on": "selection-half pooled grouped accuracy only",
            "quote_on": "validation-half paired per-fold delta vs the incumbent",
            "win_condition": "|mean| > sd of the validation-half paired deltas, one consistent sign",
            "verdict_order": [
                "win: the selection-half winner clears its own validation spread",
                "ranking does not transfer: no such win, and the halves disagree on the top model "
                "(or a model that was not picked clears the spread)",
                "incumbent stands: no model clears its spread and the halves agree",
            ],
            "pooled_all_folds": "not a selection criterion — reported, never decided on",
        },
        "claimed_schema": "survey-sweep/1",
        "inputs": [str(path) for path in inputs],
        "incumbent": {
            "model_id": incumbent_for_decision["model_id"],
            "swept_here": not incumbent_for_decision.get("reference_only", False),
            "selection_pooled": round(incumbent_for_decision["selection"]["pooled"], 4),
            "validation_pooled": round(incumbent_for_decision["validation"]["pooled"], 4),
            "validation_mean_fold": round(incumbent_for_decision["validation"]["mean_fold"], 4),
            "per_fold_accuracy": {
                str(fold): round(value, 4) for fold, value in incumbent_for_decision["per_fold"].items()
            },
        },
        "verdict": {
            "form": verdict["form"],
            "label": verdict["label"],
            "headline": headline(verdict),
            "text": verdict_text(verdict, validation_folds),
            "winner": verdict["winner"],
            "validation_top": verdict["validation_top"],
            "top_differs": verdict["top_differs"],
            "clearers": verdict["clearers"],
        },
        "falsifier": falsifier,
        "ranking_selection": verdict["ranking_selection"],
        "ranking_validation": verdict["ranking_validation"],
        "models": [
            {
                "model_id": model["model_id"],
                "path": model["path"],
                "swept_here": model.get("swept_here", True),
                "measured_at": model.get("measured_at"),
                "selection": model["selection"],
                "validation": model["validation"],
                "per_fold_accuracy": {
                    str(fold): round(value, 4) for fold, value in model["per_fold"].items()
                },
                "fold_rows": {str(fold): rows for fold, rows in model["fold_rows"].items()},
                "per_fold_delta_points": {
                    str(fold): round(100.0 * value, 2)
                    for fold, value in model["delta_by_fold"].items()
                },
                "paired_all_folds": model["paired_all"],
                "paired_selection_half": model["paired_selection"],
                "paired_validation_half": model["paired_validation"],
                "clears_validation_spread": model["clears_validation_spread"],
                "wins_validation": model["wins_validation"],
                "loses_validation": model["loses_validation"],
                "none_recall": model.get("none_recall"),
                "false_none_rate": model.get("false_none_rate"),
                "ece": model.get("ece"),
                "p50_ms": model.get("p50_ms"),
                "p50_source": model.get("p50_source"),
                "pooled_all_folds": {
                    "grouped": model.get("grouped_pooled"),
                    "rowwise": model.get("rowwise_pooled"),
                    "not_a_selection_criterion": True,
                },
            }
            for model in sorted(models_for_decision, key=lambda m: -(m["selection"]["pooled"] or -1.0))
        ],
        "failed_sweeps": failed,
        "warnings": warnings,
        "notes": notes,
    }
    json_path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    md_path.write_text(
        render_markdown(
            models_for_decision,
            verdict,
            falsifier,
            incumbent_for_decision,
            notes,
            warnings,
            inputs,
            selection_folds,
            validation_folds,
            generated,
            failed,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"phase 4: {len(models)} model(s) from {len(inputs)} sweep artefact(s)")
    print(f"selection folds {selection_folds} -> winner `{verdict['winner']['model_id']}` "
          f"at {verdict['winner']['selection_pooled']:.4f} pooled")
    print(f"validation folds {validation_folds} -> "
          f"`{verdict['validation_top']['model_id']}` at "
          f"{verdict['validation_top']['validation_pooled']:.4f} pooled")
    print(f"VERDICT: {headline(verdict)}")
    print(verdict_text(verdict, validation_folds))
    print(f"FALSIFIER: {falsifier_text(falsifier, validation_folds)}")
    for warning in warnings:
        print(f"warning: {warning}")
    for entry in failed:
        print(f"failed sweep (not ranked): {entry['model_id']} at {entry.get('failed_stage')}: {entry.get('error')}")
    for note in notes:
        print(f"note: {note}")
    print(f"wrote {md_path}")
    print(f"wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
