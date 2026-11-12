"""public_corpus — a public dataset on disk as rows localAlexa's labels can score.

Two files per dataset, both under the gitignored `data/raw/<slug>/`:

``normalized.tsv``   ``split <TAB> text <TAB> intent``, one row per utterance, written by
                     `sources.py` from whatever shape the dataset actually ships.
``mapping.csv``      ``dataset_intent,localalexa_tool,confidence,rationale`` — one row per
                     intent, the recon subagent's mapping.

Nothing here ships the rows. The dataset is a path on this box; the repository carries the
licence, the counts and the mapping, which is what `docs/public-data-2026-11-10.md` is for.

An intent whose tool is outside the 17 + ``__none__`` is a load error, not a silent drop:
the mapping file is the instrument, and a typo in it would otherwise read as a smaller
dataset instead of as a mistake.
"""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from localbench import LOCALALEXA, HRow

RAW = LOCALALEXA / "data" / "raw"
CONFIDENCE_ORDER = ("high", "medium", "low")

# The corpus's own out-of-scope label is the abstain class by definition, so it does not need a
# mapping row. Nothing else may arrive without one: an unjoined intent is a broken mapping file,
# and the error below is what turns that into a refusal instead of a smaller dataset.
DEFAULT_INTENTS = {
    "oos": ("__none__", "high", "the corpus's own out-of-scope label is the abstain class"),
}


class CorpusError(ValueError):
    """A dataset, a mapping or a join that cannot be scored honestly."""


@dataclass(frozen=True)
class PublicRow:
    id: str
    text: str
    intent: str
    split: str
    tool: str
    confidence: str


@dataclass(frozen=True)
class PublicCorpus:
    slug: str
    train: tuple[PublicRow, ...]
    test: tuple[PublicRow, ...]
    intents: tuple[str, ...]
    mapping: dict[str, tuple[str, str, str]]
    row_counts: dict[str, int]

    def rows(self, confidence: str = "all") -> tuple[PublicRow, ...]:
        if confidence == "all":
            return self.train + self.test
        keep = set(CONFIDENCE_ORDER[: CONFIDENCE_ORDER.index(confidence) + 1])
        return tuple(r for r in self.train + self.test if r.confidence in keep)

    def stats(self) -> dict:
        """What the doc's table quotes: rows, mapped rows, unmapped rows, intents, splits."""
        all_rows = self.train + self.test
        none_rows = [r for r in all_rows if r.tool == "__none__"]
        unmapped_rows = len(none_rows)
        intents_to_none = {r.intent for r in none_rows}
        by_tool = Counter(r.tool for r in all_rows)
        return {
            "n_rows": len(all_rows),
            "n_train": len(self.train),
            "n_test": len(self.test),
            "n_intents": len(self.intents),
            "n_intents_to_none": len(intents_to_none),
            "n_rows_to_none": unmapped_rows,
            "pct_rows_to_none": round(100 * unmapped_rows / len(all_rows), 1) if all_rows else 0.0,
            "pct_rows_mapped": round(100 * (len(all_rows) - unmapped_rows) / len(all_rows), 1)
            if all_rows else 0.0,
            "by_tool": dict(sorted(by_tool.items(), key=lambda kv: (-kv[1], kv[0]))),
            "by_confidence": dict(sorted(Counter(r.confidence for r in all_rows).items())),
            "splits": dict(sorted(Counter(r.split for r in all_rows).items())),
        }


def _read_normalized(path: Path) -> list[tuple[str, str, str]]:
    if not path.exists():
        raise CorpusError(f"{path} is missing — sources.py has to normalize this dataset first")
    out = []
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh, delimiter="\t")
        header = next(reader, None)
        if header != ["split", "text", "intent"]:
            raise CorpusError(f"{path}: header is {header}, expected ['split', 'text', 'intent']")
        for line, row in enumerate(reader, start=2):
            if len(row) != 3:
                raise CorpusError(f"{path}:{line}: {len(row)} fields, expected 3")
            split, text, intent = (f.strip() for f in row)
            if not text:
                raise CorpusError(f"{path}:{line}: empty text")
            out.append((split, text, intent))
    if not out:
        raise CorpusError(f"{path}: no rows")
    return out


def _read_mapping(path: Path, allowed: Sequence[str]) -> dict[str, tuple[str, str, str]]:
    """Intent -> (tool, confidence, rationale), with `none` read as the abstain label.

    The recon brief asks for `none` in the mapping file because that is the word the sentence
    uses; the label itself is `__none__`. Both spellings are accepted and one label comes out.
    """
    aliases = {"none": "__none__", "__none__": "__none__"}
    if not path.exists():
        raise CorpusError(f"{path} is missing — the recon stage writes it")
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        cols = reader.fieldnames or []
        want = ["dataset_intent", "localalexa_tool", "confidence", "rationale"]
        if cols[:4] != want:
            raise CorpusError(f"{path}: columns are {cols}, expected {want}")
        out: dict[str, tuple[str, str, str]] = {}
        for line, row in enumerate(reader, start=2):
            intent = (row["dataset_intent"] or "").strip()
            tool = aliases.get((row["localalexa_tool"] or "").strip(),
                               (row["localalexa_tool"] or "").strip())
            conf = (row["confidence"] or "").strip().lower()
            if not intent:
                raise CorpusError(f"{path}:{line}: blank intent")
            if tool not in allowed:
                raise CorpusError(
                    f"{path}:{line}: {intent!r} maps to {tool!r}, which is not one of the "
                    f"{len(allowed)} localAlexa labels — the mapping file is the instrument")
            if conf not in CONFIDENCE_ORDER:
                raise CorpusError(f"{path}:{line}: confidence {conf!r} not in {CONFIDENCE_ORDER}")
            if intent in out:
                raise CorpusError(f"{path}:{line}: duplicate intent {intent!r}")
            out[intent] = (tool, conf, (row["rationale"] or "").strip())
    if not out:
        raise CorpusError(f"{path}: no intents")
    return out


def load(slug: str, options: Sequence[str], name: str | None = None) -> PublicCorpus:
    """Read one dataset directory and join rows to the mapping, refusing a lossy join."""
    root = RAW / slug
    raw_rows = _read_normalized(root / "normalized.tsv")
    mapping = _read_mapping(root / "mapping.csv", options)
    intents_in_data = {intent for _, _, intent in raw_rows}
    missing = sorted(intents_in_data - set(mapping) - set(DEFAULT_INTENTS))
    if missing:
        raise CorpusError(
            f"{slug}: {len(missing)} intent(s) appear in the data with no mapping row, "
            f"e.g. {missing[:5]} — an unjoined intent would silently shrink the dataset")
    joined: list[PublicRow] = []
    for i, (split, text, intent) in enumerate(raw_rows):
        tool, conf, _ = mapping[intent] if intent in mapping else DEFAULT_INTENTS[intent]
        joined.append(PublicRow(id=f"{slug}-{i:05d}", text=text, intent=intent, split=split,
                                tool=tool, confidence=conf))
    train = tuple(r for r in joined if r.split == "train")
    test = tuple(r for r in joined if r.split == "test")
    if not train or not test:
        raise CorpusError(
            f"{slug}: split counts are train={len(train)} test={len(test)} — the transfer "
            f"measurement needs both sides")
    return PublicCorpus(slug=slug, train=train, test=test,
                        intents=tuple(sorted(mapping)), mapping=mapping,
                        row_counts={"train": len(train), "test": len(test)})


def as_harness_rows(rows: Iterable[PublicRow], options: Sequence[str]) -> list[HRow]:
    """Mapped public rows as harness rows: the label is the localAlexa tool, the group the intent."""
    out = []
    for r in rows:
        out.append(HRow(id=r.id, text=r.text, label=r.tool, group=r.intent,
                        options=tuple(options), meta={"intent": r.intent,
                                                      "confidence": r.confidence}))
    return out


def pairs(corpus: PublicCorpus, confidence: str = "all") -> list[tuple[str, str]]:
    """(text, label) pairs for a training side, deduplicated on text.

    Two intents mapping to the same tool are the same label here, so an identical utterance
    under both would sit in the pool twice and weight itself. Duplicates are counted, not
    hidden: `dedup_report` says how many were dropped.
    """
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for row in corpus.rows(confidence):
        if row.text in seen:
            continue
        seen.add(row.text)
        out.append((row.text, row.tool))
    return out


def dedup_report(corpus: PublicCorpus, confidence: str = "all") -> dict:
    rows = corpus.rows(confidence)
    kept = pairs(corpus, confidence)
    return {"n_rows": len(rows), "n_unique_texts": len(kept),
            "n_duplicate_texts": len(rows) - len(kept),
            "by_tool": dict(sorted(Counter(label for _, label in kept).items(),
                                   key=lambda kv: (-kv[1], kv[0])))}


def cap_per_label(rows: Sequence[tuple[str, str]], cap: int, seed: int = 0) -> list[tuple[str, str]]:
    """At most `cap` rows per label, taking a stable pseudo-random sample.

    The ladder's rung 6 is the reason this exists: every uncapped slot fill made the grouped
    number worse, and the capped variant was the one that did not. Order is
    ``sha256(seed|label|text)`` so the sample does not move between runs.
    """
    import hashlib

    buckets: dict[str, list[tuple[str, str]]] = {}
    for text, label in rows:
        buckets.setdefault(label, []).append((text, label))
    out: list[tuple[str, str]] = []
    for label in sorted(buckets):
        bucket = sorted(buckets[label],
                        key=lambda p: hashlib.sha256(f"{seed}|{label}|{p[0]}".encode()).hexdigest())
        out.extend(bucket[:cap])
    return out
