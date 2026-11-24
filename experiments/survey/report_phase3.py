"""Phase 3 report: every probe JSON as one row, plus the pre-registered drop rule.

    .venv-embed/bin/python experiments/survey/report_phase3.py \
        --probes 'results/survey-2026-11-23/probe-*.json'

Reads the artefacts ``probe_model.py`` writes and nothing else. One row per
model: what it is (params, dim, max sequence length, licence, snapshot bytes),
what it costs resident, what it costs per row, what it costs per utterance, and
the decision the brief pre-registers: a model that cannot load is out, and a
model that cannot answer inside the latency ceiling is out.

The ceiling is a **ratio against the incumbent's measured p50**, not an absolute
millisecond count, because the absolute number moves with the box's load (the
same probe read 6.33 ms at loadavg 8.3 and 8.62 ms at loadavg 9.4). The ratio
default is 1.5; with the incumbent at 6.330 ms that is 9.50 ms, inside the
brief's ~10 ms budget. Both numbers are printed so the ratio is never the only
thing on record.

Two probes of the same model id are compared on the fields that make two
measurements comparable (host, interpreter, torch, thread count, corpus, passes,
warmup, batch size). Identical conditions are a repeat measurement and the
duplicate is named, not dropped silently. Different conditions are a conflict:
both rows are printed, the differing fields are listed, and the model's decision
becomes ``unresolved`` — merging them would invent a number neither probe
measured.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from datetime import datetime
from pathlib import Path

SCHEMA = "survey-phase3-table/1"
GENERATOR = "experiments/survey/report_phase3.py"
DEFAULT_OUT_DIR = Path(__file__).resolve().parents[2] / "results" / "survey-2026-11-23"
DEFAULT_RATIO = 1.5
DEFAULT_INCUMBENT = "sentence-transformers/all-MiniLM-L6-v2"

PERMISSIVE_LICENCES = {
    "apache-2.0",
    "mit",
    "bsd-3-clause",
    "bsd-2-clause",
    "bsd",
    "isc",
    "cc0-1.0",
    "cc-by-4.0",
    "unlicense",
    "ofl-1.1",
    "artistic-2.0",
    "openrail",
    "openrail++",
    "bigscience-openrail-m",
    "creativeml-openrail-m",
}
NON_PERMISSIVE_LICENCES = {
    "gpl-2.0",
    "gpl-3.0",
    "agpl-3.0",
    "lgpl-2.1",
    "lgpl-3.0",
    "cc-by-nc-4.0",
    "cc-by-nc-sa-4.0",
    "cc-by-nd-4.0",
    "cc-by-sa-4.0",
}

MIB = 1024.0 * 1024.0


def expand_patterns(patterns: list[str]) -> list[Path]:
    """Glob every pattern, keep file hits, sort for a stable report order."""
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


def load_probe(path: Path) -> tuple[dict | None, str | None]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:  # noqa: BLE001 - the message is the row
        return None, f"{type(exc).__name__}: {exc}"
    if not isinstance(payload, dict):
        return None, f"expected a JSON object, got {type(payload).__name__}"
    return payload, None


def condition_signature(row: dict) -> dict:
    env = row.get("environment") or {}
    conditions = row.get("conditions") or {}
    corpus = row.get("corpus") or {}
    return {
        "hostname": env.get("hostname"),
        "platform": env.get("platform"),
        "python": env.get("python"),
        "torch": env.get("torch"),
        "torch_threads": env.get("torch_threads"),
        "passes_requested": conditions.get("passes_requested"),
        "passes_recorded": conditions.get("passes_recorded"),
        "warmup_calls": conditions.get("warmup_calls"),
        "embed_batch_size": conditions.get("embed_batch_size"),
        "corpus_rows": corpus.get("n_rows"),
        "corpus_truncated": corpus.get("truncated"),
    }


def differing_conditions(signatures: list[dict]) -> dict[str, list]:
    """Field name -> the distinct values seen, only where they disagree."""
    out: dict[str, list] = {}
    if not signatures:
        return out
    for field in signatures[0]:
        values = [sig.get(field) for sig in signatures]
        if len({json.dumps(value, sort_keys=True) for value in values}) > 1:
            out[field] = values
    return out


def mb(value) -> float | None:
    return None if value is None else round(float(value) / MIB, 1)


def human_params(value) -> str:
    if value is None:
        return "—"
    if value >= 1_000_000_000:
        return f"{value / 1e9:.2f}B"
    if value >= 1_000_000:
        return f"{value / 1e6:.2f}M"
    if value >= 1_000:
        return f"{value / 1e3:.1f}k"
    return str(value)


def cell(value, digits: int = 2) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)


EMPTY_ROW = {
    "model_id": None,
    "path": None,
    "status": None,
    "load_status": None,
    "load_ms": None,
    "import_ms": None,
    "error": None,
    "failed_stage": None,
    "stage_errors": {},
    "measured_at": None,
    "params": None,
    "params_human": "—",
    "embedding_dim": None,
    "max_seq_length": None,
    "licence": None,
    "licence_source": None,
    "licence_error": None,
    "gated": None,
    "pipeline_tag": None,
    "snapshot_bytes": None,
    "snapshot_mb": None,
    "snapshot_files": None,
    "snapshot_error": None,
    "rss_delta_bytes": None,
    "rss_mb": None,
    "rss_total_bytes": None,
    "rss_total_mb": None,
    "rss_method": None,
    "embed_ms_per_row": None,
    "embed_wall_ms": None,
    "embed_rows": None,
    "p50_ms": None,
    "p95_ms": None,
    "p50_spread_ms": None,
    "passes": None,
    "conditions": None,
    "loadavg_1m": None,
    "decision": None,
    "decision_reason": None,
    "threshold_ms": None,
    "failed_load_error": None,
}


def build_row(path: Path, payload: dict | None, error: str | None) -> dict:
    row = dict(EMPTY_ROW)
    row["path"] = str(path)
    if payload is None:
        stem = path.stem
        guessed = stem[len("probe-") :] if stem.startswith("probe-") else stem
        row.update(
            {
                "model_id": guessed.replace("__", "/"),
                "status": "unreadable",
                "load_status": "unreadable",
                "error": error,
                "failed_stage": "read",
            }
        )
        return row

    model = payload.get("model") or {}
    memory = payload.get("memory") or {}
    readings = memory.get("readings") or {}
    deltas = memory.get("deltas_bytes") or {}
    embeddings = payload.get("embeddings") or {}
    latency = payload.get("latency") or {}
    load = payload.get("load") or {}
    env = payload.get("environment") or {}

    status = payload.get("status")
    load_status = load.get("status")
    failed = None
    if status == "error" or load_status == "error":
        failed = (
            payload.get("error")
            or load.get("error")
            or (payload.get("stage_errors") or {}).get(payload.get("failed_stage") or "")
        )

    row.update(
        {
            "model_id": payload.get("model_id") or path.stem.replace("__", "/"),
            "status": status,
            "load_status": load_status,
            "load_ms": load.get("load_ms"),
            "import_ms": load.get("import_ms"),
            "error": payload.get("error") or load.get("error"),
            "failed_stage": payload.get("failed_stage"),
            "stage_errors": payload.get("stage_errors") or {},
            "measured_at": payload.get("measured_at"),
            "params": model.get("params"),
            "params_human": human_params(model.get("params")),
            "embedding_dim": model.get("embedding_dim") or embeddings.get("dim"),
            "max_seq_length": model.get("max_seq_length"),
            "licence": model.get("license"),
            "licence_source": model.get("license_source"),
            "licence_error": model.get("license_error"),
            "gated": model.get("gated"),
            "pipeline_tag": model.get("pipeline_tag"),
            "snapshot_bytes": model.get("snapshot_bytes"),
            "snapshot_mb": mb(model.get("snapshot_bytes")),
            "snapshot_files": model.get("snapshot_files"),
            "snapshot_error": model.get("snapshot_error"),
            "rss_delta_bytes": deltas.get("after_load_minus_before_load"),
            "rss_mb": mb(deltas.get("after_load_minus_before_load")),
            "rss_total_bytes": readings.get("after_latency"),
            "rss_total_mb": mb(readings.get("after_latency")),
            "rss_method": memory.get("method"),
            "embed_ms_per_row": embeddings.get("ms_per_row"),
            "embed_wall_ms": embeddings.get("wall_ms"),
            "embed_rows": embeddings.get("n_rows"),
            "p50_ms": latency.get("p50_ms"),
            "p95_ms": latency.get("p95_ms"),
            "p50_spread_ms": latency.get("p50_spread_ms"),
            "passes": latency.get("passes"),
            "conditions": condition_signature(payload),
            "loadavg_1m": env.get("loadavg_1m"),
            "failed_load_error": failed,
        }
    )
    return row


def decide(row: dict, threshold_ms: float | None, ratio: float) -> None:
    """Apply the pre-registered rule in place. Order matters: load, then latency."""
    row["threshold_ms"] = None if threshold_ms is None else round(threshold_ms, 3)
    row["p50_ceiling_ratio"] = ratio
    if row["status"] == "unreadable":
        row["decision"] = "drop"
        row["decision_reason"] = f"probe artefact unreadable: {row['error']}"
        return
    if row["status"] == "error" or row["load_status"] == "error":
        stage = row.get("failed_stage") or "load"
        text = row.get("failed_load_error") or row.get("error") or "load failed"
        row["decision"] = "drop"
        row["decision_reason"] = f"load failed at stage '{stage}': {text}"
        return
    if row["p50_ms"] is None:
        row["decision"] = "drop"
        row["decision_reason"] = "no latency measurement in the probe"
        return
    if threshold_ms is None:
        row["decision"] = "unresolved"
        row["decision_reason"] = "no ceiling available (incumbent p50 unknown)"
        return
    if row["p50_ms"] <= threshold_ms:
        row["decision"] = "pass"
        row["decision_reason"] = (
            f"p50 {row['p50_ms']:.3f} ms <= ceiling {threshold_ms:.3f} ms"
        )
    else:
        row["decision"] = "drop"
        row["decision_reason"] = (
            f"p50 {row['p50_ms']:.3f} ms > ceiling {threshold_ms:.3f} ms "
            f"({ratio:.2f} x incumbent p50)"
        )


def licence_warnings(rows: list[dict]) -> list[str]:
    warnings: list[str] = []
    for row in rows:
        licence = row.get("licence")
        normalized = str(licence).strip().lower() if licence else None
        if normalized is None:
            detail = row.get("licence_error") or "no licence field in the model card"
            warnings.append(
                f"{row['model_id']}: licence unread ({detail}) — check by hand before shipping"
            )
        elif normalized in NON_PERMISSIVE_LICENCES:
            warnings.append(
                f"{row['model_id']}: licence '{licence}' is not permissive; the brief requires one that is"
            )
        elif normalized not in PERMISSIVE_LICENCES:
            warnings.append(
                f"{row['model_id']}: licence '{licence}' is not on the permissive allow-list "
                "this report knows — check by hand"
            )
    return warnings


def render_markdown(
    rows: list[dict],
    conflicts: list[dict],
    notes: list[str],
    warnings: list[str],
    rule: dict,
    inputs: list[Path],
    generated: str,
) -> str:
    out: list[str] = []
    out.append("# Phase 3 — probe table")
    out.append("")
    out.append(
        f"Generated {generated} by `{GENERATOR}` from {len(inputs)} probe artefact(s). "
        "Numbers are the probe's own; nothing here re-measures."
    )
    out.append("")
    out.append("## The pre-registered drop rule")
    out.append("")
    if rule["threshold_ms"] is None:
        out.append(
            f"- A model that cannot load is **out** (the brief: prove it rather than trust it).\n"
            f"- Latency ceiling: **unavailable** — no probe for the incumbent "
            f"`{rule['incumbent_model_id']}` and no `--incumbent-p50-ms` given, so no model "
            f"can be judged against one."
        )
    else:
        out.append(
            f"- A model that cannot load is **out** (the brief: prove it rather than trust it).\n"
            f"- Latency ceiling: **{rule['threshold_ms']:.2f} ms** = "
            f"{rule['p50_ceiling_ratio']:.2f} × the incumbent's measured p50 "
            f"{rule['incumbent_p50_ms']:.3f} ms (`{rule['incumbent_model_id']}`, probe p50). "
            f"A model whose p50 exceeds the ceiling is dropped, at phase 3, before any training."
        )
        if rule["threshold_note"]:
            out.append("")
            out.append(f"  {rule['threshold_note']}")
    out.append("")
    out.append("## The table")
    out.append("")
    header = (
        "| model | load | params | dim | max_seq | licence | snapshot MB | RSS MB | "
        "embed ms/row | p50 ms | p95 ms | decision | against ceiling |"
    )
    out.append(header)
    out.append("|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---|---|")
    for row in rows:
        licence = row.get("licence") or "—"
        if not licence and row.get("licence_error"):
            licence = "unreadable"
        label = row.get("label") or row["model_id"]
        name = f"`{row['model_id']}`"
        if label != row["model_id"]:
            name += f" ({label.replace(row['model_id'], '').strip().strip('()')})"
        out.append(
            "| {label} | {load} | {params} | {dim} | {seq} | {licence} | {snap} | {rss} | "
            "{per_row} | {p50} | {p95} | **{decision}** | {reason} |".format(
                label=name,
                load=cell(row.get("load_status")),
                params=row.get("params_human") or "—",
                dim=cell(row.get("embedding_dim"), 0),
                seq=cell(row.get("max_seq_length"), 0),
                licence=str(licence),
                snap=cell(row.get("snapshot_mb"), 1),
                rss=cell(row.get("rss_mb"), 1),
                per_row=cell(row.get("embed_ms_per_row"), 3),
                p50=cell(row.get("p50_ms"), 2),
                p95=cell(row.get("p95_ms"), 2),
                decision=row.get("decision") or "—",
                reason=row.get("decision_reason") or "—",
            )
        )
    out.append("")
    out.append("*RSS MB is `memory.deltas_bytes.after_load_minus_before_load` — the model's own "
               "footprint, taken after torch and sentence-transformers were already imported, so "
               "the ~500 MB those cost is not in the column. MB here is MiB (1024² B).*")
    out.append("")
    out.append(
        "*The rule compares **p50** only — that is what `--p50-ceiling-ratio` names, and the "
        "brief's budget is a median. p95 is reported beside it because a model whose median fits "
        "and whose tail does not is the row a reader will want to look at, but it changes no "
        "decision here.*"
    )
    out.append("")
    out.append("## Decisions, one line each")
    out.append("")
    passed = [row for row in rows if row.get("decision") == "pass"]
    dropped = [row for row in rows if row.get("decision") == "drop"]
    unresolved = [row for row in rows if row.get("decision") == "unresolved"]
    out.append(
        f"- **pass** ({len(passed)}): "
        + (", ".join(f"`{row['model_id']}`" for row in passed) if passed else "none")
    )
    out.append(
        f"- **drop** ({len(dropped)}): "
        + (", ".join(f"`{row['model_id']}`" for row in dropped) if dropped else "none")
    )
    if unresolved:
        out.append(
            f"- **unresolved** ({len(unresolved)}): "
            + ", ".join(f"`{row.get('label') or row['model_id']}`" for row in unresolved)
        )
    out.append("")

    failures = [
        row
        for row in rows
        if row["status"] in {"error", "unreadable"} or row.get("load_status") == "error"
    ]
    if failures:
        out.append("## Failures")
        out.append("")
        out.append("These are rows, not gaps: a model that did not load is reported with its error.")
        out.append("")
        for row in failures:
            out.append(f"- `{row['model_id']}` — stage `{row.get('failed_stage')}`")
            out.append("")
            out.append("  ```")
            out.append(f"  {str(row.get('failed_load_error') or row.get('error')).strip()}")
            out.append("  ```")
        out.append("")

    if conflicts:
        out.append("## Conflicting probes — not merged")
        out.append("")
        out.append(
            "The same model id was probed under different conditions. Both rows are printed and "
            "no decision is issued for either: averaging them, or keeping whichever came first, "
            "would be a number neither probe measured."
        )
        out.append("")
        for conflict in conflicts:
            out.append(f"### `{conflict['model_id']}`")
            out.append("")
            out.append(
                f"{len(conflict['paths'])} probes differ on: "
                + ", ".join(f"`{field}`" for field in conflict["differing_fields"])
            )
            out.append("")
            out.append("| field | " + " | ".join(f"probe {i + 1}" for i in range(len(conflict["paths"]))) + " |")
            out.append("|---|---" + "|---" * (len(conflict["paths"]) - 1) + "|")
            for field, values in conflict["differing_values"].items():
                cells = " | ".join("—" if value is None else str(value) for value in values)
                out.append(f"| `{field}` | {cells} |")
            out.append("")
            for index, (path, row) in enumerate(zip(conflict["paths"], conflict["rows"])):
                out.append(
                    f"- probe {index + 1}: `{path}` — measured {row.get('measured_at')}, "
                    f"p50 {cell(row.get('p50_ms'), 3)} ms, loadavg {cell(row.get('loadavg_1m'), 2)}"
                )
            out.append("")

    out.append("## Conditions each number was measured under")
    out.append("")
    out.append("| model | host | torch | threads | loadavg | passes |")
    out.append("|---|---|---|---:|---:|---:|")
    for row in rows:
        conditions = row.get("conditions") or {}
        out.append(
            "| `{label}` | {host} | {torch} | {threads} | {load} | {passes} |".format(
                label=row.get("label") or row["model_id"],
                host=conditions.get("hostname") or "—",
                torch=conditions.get("torch") or "—",
                threads=cell(conditions.get("torch_threads"), 0),
                load=cell(row.get("loadavg_1m"), 2),
                passes=cell(row.get("passes"), 0),
            )
        )
    out.append("")
    out.append(
        "p50 across rows is only comparable when the box was as quiet for one as for the other. "
        "The ladder's lesson (`docs/ladder-results-2026-11-10.md`) is that a latency taken while "
        "something else trained is context, not a measurement."
    )
    out.append("")

    if warnings:
        out.append("## Warnings (advisory — they do not change a decision)")
        out.append("")
        for warning in warnings:
            out.append(f"- {warning}")
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Phase 3: probe JSONs in, one table plus the pre-registered drop decision out."
    )
    parser.add_argument(
        "--probes",
        nargs="+",
        action="extend",
        required=True,
        help="glob pattern(s) for probe JSONs, e.g. 'results/survey-2026-11-23/probe-*.json'",
    )
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help=f"where phase3-table.md and phase3-table.json are written (default {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--p50-ceiling-ratio",
        type=float,
        default=DEFAULT_RATIO,
        help=(
            "drop a model whose p50 exceeds this ratio times the incumbent's measured p50 "
            f"(default {DEFAULT_RATIO})"
        ),
    )
    parser.add_argument(
        "--incumbent",
        default=DEFAULT_INCUMBENT,
        help=f"model id whose probe p50 sets the ceiling (default {DEFAULT_INCUMBENT})",
    )
    parser.add_argument(
        "--incumbent-p50-ms",
        type=float,
        default=None,
        help="override the incumbent p50 instead of reading it from a probe (ms)",
    )
    args = parser.parse_args(argv)

    if args.p50_ceiling_ratio <= 0:
        print("--p50-ceiling-ratio must be positive", file=sys.stderr)
        return 2

    inputs = expand_patterns(args.probes)
    if not inputs:
        print(
            f"no probe artefacts matched {args.probes!r} — nothing to report",
            file=sys.stderr,
        )
        return 2

    rows: list[dict] = []
    notes: list[str] = []
    for path in inputs:
        payload, error = load_probe(path)
        row = build_row(path, payload, error)
        row["probe_error"] = error
        row["label"] = row["model_id"]
        rows.append(row)

    by_model: dict[str, list[dict]] = {}
    for row in rows:
        by_model.setdefault(row["model_id"], []).append(row)

    conflicts: list[dict] = []
    kept: list[dict] = []
    for model_id, group in by_model.items():
        if len(group) == 1:
            kept.append(group[0])
            continue
        # Collapse exact repeats first: same model, same conditions, measured twice.
        by_conditions: dict[str, list[dict]] = {}
        for row in group:
            key = json.dumps(row.get("conditions"), sort_keys=True)
            by_conditions.setdefault(key, []).append(row)
        representatives: list[dict] = []
        for members in by_conditions.values():
            members.sort(key=lambda row: (row.get("measured_at") or "", row["path"]))
            representatives.append(members[0])
            for duplicate in members[1:]:
                notes.append(
                    f"`{model_id}` was probed again under identical conditions: kept "
                    f"`{members[0]['path']}` ({members[0].get('measured_at')}), ignored "
                    f"`{duplicate['path']}` ({duplicate.get('measured_at')}) — a repeat "
                    "measurement, not a conflict"
                )
                print(
                    f"note: {model_id} probed again under identical conditions; "
                    f"using {members[0]['path']}",
                    file=sys.stderr,
                )
        representatives.sort(key=lambda row: row["path"])
        if len(representatives) == 1:
            kept.extend(representatives)
            continue

        differing = differing_conditions([row.get("conditions") or {} for row in representatives])
        conflicts.append(
            {
                "model_id": model_id,
                "paths": [row["path"] for row in representatives],
                "rows": representatives,
                "differing_fields": list(differing),
                "differing_values": differing,
            }
        )
        for index, row in enumerate(representatives, start=1):
            row["label"] = f"{model_id} (probe {index})"
            row["decision"] = "unresolved"
            row["decision_reason"] = (
                "same model id, different measurement conditions — see Conflicts"
            )
            row["threshold_ms"] = None
            kept.append(row)
        print(
            f"CONFLICT: {model_id} was probed under {len(representatives)} different sets of "
            f"conditions (of {len(group)} probes); every one is printed and none is decided.",
            file=sys.stderr,
        )
        for index, row in enumerate(representatives, start=1):
            print(f"  probe {index}: {row['path']}", file=sys.stderr)
            for field, values in differing.items():
                print(f"    {field}: {values[index - 1]!r}", file=sys.stderr)

    incumbent_rows = [row for row in kept if row["model_id"] == args.incumbent]
    incumbent_p50 = None
    incumbent_source = None
    if args.incumbent_p50_ms is not None:
        incumbent_p50 = float(args.incumbent_p50_ms)
        incumbent_source = "--incumbent-p50-ms"
    elif incumbent_rows:
        candidate = incumbent_rows[0]
        if candidate.get("p50_ms") is not None:
            incumbent_p50 = float(candidate["p50_ms"])
            incumbent_source = f"probe p50 for `{args.incumbent}`"
            if len(incumbent_rows) > 1:
                notes.append(
                    f"the incumbent `{args.incumbent}` was probed more than once; the ceiling "
                    f"uses {incumbent_source} from `{candidate['path']}`"
                )

    threshold_ms = None
    threshold_note = None
    if incumbent_p50 is not None:
        threshold_ms = args.p50_ceiling_ratio * incumbent_p50
        if args.p50_ceiling_ratio == DEFAULT_RATIO:
            relation = "inside" if threshold_ms <= 10.0 else "above"
            threshold_note = (
                f"The ratio default is {DEFAULT_RATIO}: half again the incumbent's p50, which lands "
                f"at {threshold_ms:.2f} ms here — {relation} the brief's ~10 ms CPU budget. The "
                "ratio carries the measuring box's speed with it, which is why it is a ratio and "
                "not an absolute: pass `--p50-ceiling-ratio` to move it. The pre-registered rule "
                "is the one in this file."
            )
    else:
        notes.append(
            f"no probe for the incumbent `{args.incumbent}` in the input set and no "
            "`--incumbent-p50-ms` override: the latency rule cannot be applied, so no model is "
            "passed or dropped on speed (load failures are still drops)."
        )

    for row in kept:
        row["p50_ceiling_ratio"] = args.p50_ceiling_ratio
        if row.get("decision") == "unresolved":
            row["threshold_ms"] = None if threshold_ms is None else round(threshold_ms, 3)
            continue
        decide(row, threshold_ms, args.p50_ceiling_ratio)

    kept.sort(
        key=lambda row: (
            row.get("p50_ms") is None,
            row.get("p50_ms") if row.get("p50_ms") is not None else 0.0,
            str(row.get("label")),
        )
    )

    warnings = licence_warnings(kept)
    hosts = sorted(
        {
            (row.get("conditions") or {}).get("hostname")
            for row in kept
            if row.get("p50_ms") is not None and (row.get("conditions") or {}).get("hostname")
        }
    )
    if len(hosts) > 1:
        warnings.append(
            f"this probe set spans {len(hosts)} hosts ({', '.join(hosts)}): a ceiling derived on "
            "one host is not a ceiling on another, and the p50 column is not comparable across "
            "them — split the set by host before trusting a drop"
        )
    generated = datetime.now().astimezone().replace(microsecond=0).isoformat()

    rule = {
        "statement": (
            "drop on load failure; drop when p50 exceeds the ceiling; the ceiling is "
            "--p50-ceiling-ratio x the incumbent's measured probe p50"
        ),
        "p50_ceiling_ratio": args.p50_ceiling_ratio,
        "incumbent_model_id": args.incumbent,
        "incumbent_p50_ms": None if incumbent_p50 is None else round(incumbent_p50, 3),
        "incumbent_p50_source": incumbent_source,
        "threshold_ms": None if threshold_ms is None else round(threshold_ms, 3),
        "threshold_note": threshold_note,
    }

    counts = {
        "probe_files": len(inputs),
        "models": len(kept),
        "pass": sum(1 for row in kept if row.get("decision") == "pass"),
        "drop": sum(1 for row in kept if row.get("decision") == "drop"),
        "unresolved": sum(1 for row in kept if row.get("decision") == "unresolved"),
        "load_failures": sum(
            1
            for row in kept
            if row["status"] in {"error", "unreadable"} or row.get("load_status") == "error"
        ),
        "conflicts": len(conflicts),
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "phase3-table.json"
    md_path = out_dir / "phase3-table.md"

    payload = {
        "schema": SCHEMA,
        "generated_at": generated,
        "generator": GENERATOR,
        "command": " ".join(["report_phase3.py", *sys.argv[1:]]),
        "rule": rule,
        "counts": counts,
        "inputs": [str(path) for path in inputs],
        "rows": kept,
        "conflicts": [
            {
                "model_id": conflict["model_id"],
                "paths": conflict["paths"],
                "differing_fields": conflict["differing_fields"],
                "differing_values": conflict["differing_values"],
            }
            for conflict in conflicts
        ],
        "warnings": warnings,
        "notes": notes,
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(
        render_markdown(kept, conflicts, notes, warnings, rule, inputs, generated) + "\n",
        encoding="utf-8",
    )

    print(f"phase 3: {counts['models']} model(s) from {counts['probe_files']} probe file(s)")
    if threshold_ms is None:
        print("ceiling: unavailable (no incumbent p50)")
    else:
        print(
            f"ceiling: {threshold_ms:.2f} ms "
            f"({args.p50_ceiling_ratio:.2f} x {incumbent_p50:.3f} ms incumbent p50)"
        )
    for row in kept:
        print(f"  {row.get('decision'):10s} {row.get('label')}: {row.get('decision_reason')}")
    if conflicts:
        print(f"conflicts: {len(conflicts)} model id(s) probed under different conditions — see {md_path}")
    print(f"wrote {md_path}")
    print(f"wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
