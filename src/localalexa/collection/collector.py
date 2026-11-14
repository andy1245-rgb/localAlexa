"""The collector: a tap on the request path, and the signals it emits as it goes.

``record``
    append the row, before the caller has its answer. Nothing else happens here.

``settle``
    read the rows written since the last settle *from the log file*, run the
    correction detector and the auto-accept rule over what is new, and append the
    label or correction events those produce.

Settling reads the log rather than an in-memory request list on purpose: the log
is the record, so a row written by a previous process — or by a second writer —
is still found. The collector keeps a byte offset and reads forward, so the cost
is proportional to new rows, not to the size of the file.

Everything here appends. Nothing rewrites. The first version of this module
rebuilt the derived files on every settle, which measured at 1.6 ms with 50 rows
in the tail and 15.6 ms with 500 — an order of magnitude more than the routing
decision it was sitting next to. Appending one line is a microsecond and the
reduction ("one label per request, strongest wins") moved to ``derive`` where it
runs on demand.

What the live path does *not* derive is the second-opinion signal: scoring the
whole tail through another model costs milliseconds. ``collect derive`` adds it
offline, and ``meta.json`` next to the files records which signals produced them.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from ..cascade import CompiledPatterns
from ..data import Registry
from ..router import Choice
from .autoaccept import AutoAcceptRule, decide as decide_auto
from .corrections import Correction, CorrectionRule, classify, previous_acted_on
from .derive import label_for
from .log import RequestLog
from .opinion import SecondOpinion
from .records import EffectRecord, LabelRow, RequestRecord, to_line
from .tap import CollectionTap

TAIL_ROWS = 2000
CORRECTIONS_NAME = "corrections.jsonl"
LABELS_NAME = "labels.jsonl"
REJECTS_NAME = "rejects.jsonl"
META_NAME = "meta.json"


@dataclass(frozen=True)
class SettleReport:
    """What one settle did, so the caller can see the work it paid for."""

    new_rows: int
    corrections: int
    labels: int
    wrote: tuple[str, ...]
    derive_ms: float


class Collector:
    """Records requests, then follows the log and emits the signals it finds."""

    def __init__(
        self,
        log: RequestLog | str | Path | None = None,
        *,
        registry: Registry | None = None,
        compiled: CompiledPatterns | None = None,
        sim_threshold: float = 0.45,
        conf_threshold: float = 0.35,
        correction_rule: CorrectionRule = CorrectionRule(),
        auto_rule: AutoAcceptRule = AutoAcceptRule(),
        aliases: Mapping[str, str] | None = None,
        opinion: SecondOpinion | None = None,
        tail_rows: int = TAIL_ROWS,
        clock=time.time,
    ) -> None:
        self.log = log if isinstance(log, RequestLog) else RequestLog(log)
        self.correction_rule = correction_rule
        self.auto_rule = auto_rule
        self.aliases = aliases
        self.opinion = opinion
        self.tail_rows = tail_rows
        self.root = self.log.root
        self.tap = CollectionTap(
            self.log,
            registry=registry,
            compiled=compiled
            or (CompiledPatterns.from_registry(registry) if registry is not None else None),
            sim_threshold=sim_threshold,
            conf_threshold=conf_threshold,
            clock=clock,
        )
        self._row_offset = 0
        self._effect_offset = 0
        self._records: list[RequestRecord] = []
        self._emitted: set[tuple[str, str]] = set()
        self._corrections: list[Correction] = []
        self._rejects_seen: set[str] = set()
        self._label_count = 0
        self._load()

    def _load(self) -> None:
        """Pick up what the log already holds, so a restart loses nothing."""
        rows, self._row_offset = self.log.read_from(0)
        self._absorb(rows)
        _, self._effect_offset = self.log.effects_from(0)
        self._load_emitted()

    def _load_emitted(self) -> None:
        for row in RequestLog._iter(self.root / LABELS_NAME):
            self._emitted.add((str(row.get("request_id")), str(row.get("source"))))
            self._label_count += 1
        for row in RequestLog._iter(self.root / CORRECTIONS_NAME):
            self._emitted.add((str(row.get("earlier_id")), "correction"))
            self._corrections.append(
                Correction(
                    earlier_id=str(row["earlier_id"]),
                    later_id=str(row["later_id"]),
                    delta_seconds=float(row.get("delta_seconds", 0.0)),
                    rule=row.get("rule", {}),
                    evidence=row.get("evidence", {}),
                )
            )
        for row in RequestLog._iter(self.root / REJECTS_NAME):
            self._rejects_seen.add(str(row.get("id")))

    def _absorb(self, rows: Sequence[dict]) -> None:
        for row in rows:
            if row.get("kind") != "request":
                continue
            try:
                self._records.append(RequestRecord.from_dict(row))
            except (KeyError, TypeError, ValueError):
                continue
        if len(self._records) > self.tail_rows:
            self._records = self._records[-self.tail_rows :]

    @property
    def records(self) -> list[RequestRecord]:
        return list(self._records)

    @property
    def errors(self) -> int:
        return self.log.errors

    def record(self, text: str, choice: Choice | None = None, *, route_ms: float | None = None) -> RequestRecord:
        """Append one request row. The hot path: nothing derived happens here."""
        return self.tap.observe(text, choice, route_ms=route_ms)

    def append_effect(self, effect: EffectRecord) -> None:
        self.log.append_effect(effect)

    def settle(self) -> SettleReport:
        """Catch up with the log and emit the signals the new rows produced."""
        started = time.perf_counter()
        rows, self._row_offset = self.log.read_from(self._row_offset)
        effects, self._effect_offset = self.log.effects_from(self._effect_offset)
        self._absorb(rows)
        new_rows = len([row for row in rows if row.get("kind") == "request"])

        wrote: list[str] = []
        emitted_corrections = 0
        emitted_labels = 0

        for record in self._records[-new_rows:] if new_rows else []:
            correction = self._correction_for(record)
            if correction is not None:
                self._append(CORRECTIONS_NAME, correction.as_dict())
                self._corrections.append(correction)
                emitted_corrections += 1
                earlier = self._find(correction.earlier_id)
                if earlier is not None and self._emit(
                    label_for(
                        earlier,
                        earlier.label,
                        -1,
                        "strong",
                        "correction",
                        {
                            "corrected_by": correction.later_id,
                            "delta_seconds": round(correction.delta_seconds, 3),
                            "rule": dict(correction.rule),
                            "evidence": dict(correction.evidence),
                        },
                    )
                ):
                    emitted_labels += 1

            acceptance = decide_auto(record, self.auto_rule, self.aliases)
            if acceptance is not None and self._emit(
                label_for(
                    record,
                    acceptance.label,
                    1,
                    "auto",
                    f"auto_accept_{acceptance.rule}",
                    {"rule_version": acceptance.rule_version, "detail": dict(acceptance.detail)},
                )
            ):
                emitted_labels += 1

            if record.gate and record.id not in self._rejects_seen:
                self._append(REJECTS_NAME, _reject_row(record))
                self._rejects_seen.add(record.id)

        for effect in effects:
            if effect.get("kind") != "effect":
                continue
            try:
                parsed = EffectRecord.from_dict(effect)
            except (KeyError, TypeError, ValueError):
                continue
            target = self._find(parsed.request_id)
            if target is None or target.label is None:
                continue
            source = "effect" if parsed.verified else "effect_failed"
            if self._emit(
                label_for(
                    target,
                    target.label,
                    1 if parsed.verified else -1,
                    "weak",
                    source,
                    {"check": parsed.check, "detail": parsed.detail},
                )
            ):
                emitted_labels += 1

        if emitted_corrections:
            wrote.append(CORRECTIONS_NAME)
        if emitted_labels:
            wrote.append(LABELS_NAME)
        self._write_meta()
        return SettleReport(
            new_rows=new_rows,
            corrections=len(self._corrections),
            labels=self._label_count,
            wrote=tuple(wrote),
            derive_ms=round((time.perf_counter() - started) * 1000.0, 3),
        )

    def _correction_for(self, record: RequestRecord) -> Correction | None:
        window = self.correction_rule.window_seconds
        earlier = previous_acted_on(self._records, record, window)
        if earlier is None or earlier.id == record.id or earlier.text == record.text:
            return None
        if (earlier.id, "correction") in self._emitted:
            return None
        evidence = classify(earlier, record, self.correction_rule)
        if evidence is None:
            return None
        return Correction(
            earlier_id=earlier.id,
            later_id=record.id,
            delta_seconds=record.ts - earlier.ts,
            rule=self.correction_rule.as_dict(),
            evidence=evidence,
        )

    def _find(self, request_id: str) -> RequestRecord | None:
        for record in reversed(self._records):
            if record.id == request_id:
                return record
        return None

    def _emit(self, label: LabelRow) -> bool:
        key = (label.request_id, label.source)
        if key in self._emitted:
            return False
        self._append(LABELS_NAME, label.as_dict())
        self._emitted.add(key)
        self._label_count += 1
        return True

    def _append(self, name: str, row: Mapping) -> None:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, to_line(row).encode("utf-8"))
        finally:
            os.close(fd)

    def _write_meta(self) -> None:
        payload = {
            "collection_root": str(self.root),
            "records_in_view": len(self._records),
            "correction_rule": self.correction_rule.as_dict(),
            "auto_accept_rule": self.auto_rule.as_dict(),
            "view": "append-only label events; reduce with `localalexa collect derive`",
            "signals_used": ["correction", "effect", "auto_accept_same_class", "auto_accept_margin"]
            + (["opinion_agreement"] if self.opinion is not None else []),
        }
        path = self.root / META_NAME
        path.parent.mkdir(parents=True, exist_ok=True)
        body = json.dumps(payload, indent=2) + "\n"
        if not path.is_file() or path.read_text(encoding="utf-8") != body:
            path.write_text(body, encoding="utf-8")
            try:
                path.chmod(0o600)
            except OSError:
                pass

    def close(self) -> None:
        self.log.close()

    def __enter__(self) -> "Collector":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


def _reject_row(record: RequestRecord) -> dict:
    return {
        "id": record.id,
        "ts": round(record.ts, 6),
        "at": record.at,
        "session": record.session,
        "text": record.text,
        "confidence": round(record.confidence, 6),
        "top_sim": round(record.top_sim, 6),
        "margin": round(record.margin, 6),
        "candidates": [c.as_dict() for c in record.candidates],
    }
