"""Render the dell321 results as the markdown tables they come back as.

The box returns JSON; the operator reads tables. One renderer, so the numbers in the doc and the
numbers in the artifact cannot drift.

    .venv/bin/python experiments/dell321/render_report.py
"""

from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[2] / "results" / "dell321"


def pct(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.2f}%"


def job1_table(payload: dict) -> str:
    gen = payload["generation"]
    lines = [
        "| arm | references | + paraphrases | paired Δ | folds W/L/T | fold sd |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for name, block in payload["arms"].items():
        base, expanded = block["references"], block["expanded"]
        paired = expanded["paired_vs_references"]
        lines.append(
            f"| {name} | {pct(base['accuracy'])} | {pct(expanded['accuracy'])} | "
            f"{paired['mean_points']:+.2f} ± {paired['sd_points']:.2f} | "
            f"{paired['folds_better']}/{paired['folds_worse']}/{paired['folds_tied']} | "
            f"{pct(base['spread']['sd'])} → {pct(expanded['spread']['sd'])} |"
        )
    lines += [
        "",
        "**Generation and filtering**",
        "",
        "| | |",
        "|---|---:|",
        f"| candidates generated | {gen['candidates']} |",
        f"| distinct | {gen['distinct']} |",
        f"| kept after both filters | {gen['kept']} |",
        f"| dropped, adds nothing (< diversity floor) | {gen['dropped_adds_nothing']} |",
        f"| dropped, leaks (too close to a held-out row) | {gen['dropped_leaks']} |",
        "",
        "Diversity of the kept rows (1 − max 3-gram Jaccard against the reference pool):",
        "",
        "| n | min | p10 | p50 | p90 | max | mean |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        "| {n} | {min} | {p10} | {p50} | {p90} | {max} | {mean} |".format(
            **gen["diversity_of_kept"]),
        "",
        "Kept rows by transformation: " + ", ".join(
            f"`{name}` {count}" for name, count in
            sorted(gen["transformations_kept"].items(), key=lambda kv: -kv[1])),
        "",
        "Examples:",
        "",
    ]
    lines += [f"* {row['text']}  \n  `{'+'.join(row['transformations'])}`"
              for row in gen["examples"][:10]]
    return "\n".join(lines)


def job2_table(payload: dict) -> str:
    flat = payload["flat"]
    lines = [
        f"Flat incumbent out-of-fold: **{pct(flat['accuracy'])}** "
        f"(fold sd {pct(statistics_sd(flat['per_fold']))}).",
        "",
        "| proposal | buckets | stage A acc | errors within a bucket | top-1 | top-2 | "
        "p50 top-1 | p50 top-2 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for proposal, variants in payload["buckets"].items():
        for name, block in variants.items():
            top1, top2 = block["top1"], block["top2"]
            lines.append(
                f"| {proposal} `{name}` | {len(block['bucket_sizes'])} | "
                f"{pct(block['top1']['stage_a_accuracy'])} | "
                f"{pct(block['within_bucket_share'])} | "
                f"{pct(top1['fold_mean'])} | {pct(top2['fold_mean'])} | "
                f"{top1['latency_ms']['p50_ms']:.2f} ms | {top2['latency_ms']['p50_ms']:.2f} ms |"
            )
    lines += ["", "Latency of the flat path, same measurement: "
              f"**{flat['latency_ms']['p50_ms']:.2f} ms p50**, "
              f"{flat['latency_ms']['p95_ms']:.2f} ms p95.", ""]
    lines += ["**The buckets, as the data proposed them**", ""]
    for proposal, variants in payload["buckets"].items():
        for name, block in variants.items():
            groups: dict[int, list[str]] = {}
            for label, bucket in block["assignment"].items():
                groups.setdefault(bucket, []).append(label)
            rendered = " · ".join("{" + ", ".join(sorted(labels)) + "}"
                                  for _, labels in sorted(groups.items()))
            lines.append(f"* `{proposal}` / `{name}`: {rendered}")
    lines += ["", "**Confusions, ranked (out-of-fold)**", "",
              "| true | predicted | n |", "|---|---|---:|"]
    lines += [f"| `{pair['true']}` | `{pair['pred']}` | {pair['n']} |"
              for pair in payload["confusion"]["pairs"][:10]]
    return "\n".join(lines)


def statistics_sd(values) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5


def main() -> int:
    job1 = json.loads((RESULTS / "job1.json").read_text())
    job2 = json.loads((RESULTS / "job2.json").read_text())
    body = "\n\n".join([
        "# Job 1 — paraphrase families\n", job1_table(job1),
        "# Job 2 — the two-stage bucket\n", job2_table(job2),
    ])
    target = RESULTS / "tables.md"
    target.write_text(body + "\n", encoding="utf-8")
    print(body)
    print(f"\nwrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
