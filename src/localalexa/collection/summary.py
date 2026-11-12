"""The weekly summary: counts, ten lines, ten seconds.

No rates that need explaining, no plots, no verdicts — the brief asks for counts
and the point of counts is that the operator can read them before the coffee
lands. What is included is the four things asked for (rows, share below the gate,
the ten most confused pairs, auto-accepted labels) plus the two counts that make
the yield honest: corrections, and rows that produced no signal at all.

The confused pairs come from the reject pile spelled exactly as the router saw
them: ``top → runner-up``, ordered, because the direction is what a
disambiguation pattern gets written against.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .records import RequestRecord

WEEK_SECONDS = 7 * 24 * 3600
TOP_PAIRS = 10


@dataclass(frozen=True)
class Summary:
    days: float
    rows: int
    sessions: int
    below_gate: int
    gate_share_pct: float
    tier_counts: Mapping[str, int]
    confused_pairs: Sequence[tuple[str, str, int]]
    labels_total: int
    labels_by_source: Mapping[str, int]
    corrections: int
    silent_rows: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "days": self.days,
            "rows": self.rows,
            "sessions": self.sessions,
            "below_gate": self.below_gate,
            "gate_share_pct": round(self.gate_share_pct, 1),
            "tier_counts": dict(self.tier_counts),
            "confused_pairs": [
                {"top": top, "runner_up": second, "n": n} for top, second, n in self.confused_pairs
            ],
            "labels_total": self.labels_total,
            "labels_by_source": dict(self.labels_by_source),
            "corrections": self.corrections,
            "silent_rows": self.silent_rows,
        }


def confused_pairs(
    records: Iterable[RequestRecord], limit: int = TOP_PAIRS
) -> list[tuple[str, str, int]]:
    """The most frequent top-two candidate pairs among rows below the gate."""
    counts: dict[tuple[str, str], int] = {}
    for record in records:
        if not record.gate or len(record.candidates) < 2:
            continue
        key = (record.candidates[0].label, record.candidates[1].label)
        counts[key] = counts.get(key, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [(top, second, n) for (top, second), n in ranked[:limit]]


def _as_dict(row: Any) -> Mapping[str, Any]:
    """A label row as a mapping, whether it arrived as a dict or a ``LabelRow``."""
    return row if isinstance(row, Mapping) else row.as_dict()


def summarise(
    records: Sequence[RequestRecord],
    labels: Sequence[Mapping[str, Any]] | None = None,
    *,
    days: float = 7.0,
    now: float | None = None,
    limit_pairs: int = TOP_PAIRS,
) -> Summary:
    """Count one window of traffic. ``labels`` are derived label rows, if any."""
    reference = time.time() if now is None else now
    cutoff = reference - days * 86400
    window = [row for row in records if row.ts >= cutoff]

    below = [row for row in window if row.gate]
    tiers = {
        "tier0_regex": sum(1 for row in window if row.tier == 0),
        "tier1_router": sum(1 for row in window if row.tier == 1),
        "tier2_escalated": len(below),
    }
    label_rows = [
        _as_dict(row) for row in (labels or ()) if float(_as_dict(row).get("ts", 0)) >= cutoff
    ]
    by_source: dict[str, int] = {}
    for row in label_rows:
        source = str(row.get("source", "unknown"))
        by_source[source] = by_source.get(source, 0) + 1

    return Summary(
        days=days,
        rows=len(window),
        sessions=len({row.session for row in window}),
        below_gate=len(below),
        gate_share_pct=100.0 * len(below) / len(window) if window else 0.0,
        tier_counts=tiers,
        confused_pairs=confused_pairs(window, limit_pairs),
        labels_total=len(label_rows),
        labels_by_source=dict(sorted(by_source.items())),
        corrections=by_source.get("correction", 0),
        silent_rows=max(0, len(window) - len({str(r.get("request_id")) for r in label_rows})),
    )


def render(summary: Summary) -> str:
    """The ten-second read."""
    lines = [
        f"last {summary.days:g} days · {summary.rows} rows · {summary.sessions} sessions",
        f"below the gate {summary.below_gate} ({summary.gate_share_pct:.1f}%) · "
        f"tier0 {summary.tier_counts['tier0_regex']} / tier1 {summary.tier_counts['tier1_router']} / "
        f"escalated {summary.tier_counts['tier2_escalated']}",
        f"labels {summary.labels_total} · auto-accepted {sum(v for k, v in summary.labels_by_source.items() if k.startswith('auto_accept'))}"
        f" · corrections {summary.corrections} · rows with no signal {summary.silent_rows}",
    ]
    if summary.confused_pairs:
        lines.append("most confused:")
        lines.extend(
            f"  {top} → {second}  {n}" for top, second, n in summary.confused_pairs
        )
    else:
        lines.append("most confused: none (no row fell below the gate)")
    return "\n".join(lines)
