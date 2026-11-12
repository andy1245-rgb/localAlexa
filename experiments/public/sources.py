"""sources — each dataset's on-disk shape turned into one table.

`normalized.tsv` is ``split <TAB> text <TAB> intent``. Nothing downstream parses a dataset's
native format, so a loader change cannot silently alter a row count.

The four shapes, and the split rule applied to each:

``CLINC150``  ``data_full.json``: train/val/test keyed arrays. val folds into train (the corpus
              is 100/20/30 per intent, so leaving val out would throw away 3,000 rows for no
              reason); ``oos_*`` rows keep their ``oos`` intent and train/test split.
``HWU64``     ``split/kfold1/{train,test}/<scenario>_<intent>.csv`` — the suite's own k-fold 1,
              so the split is the benchmark's, not mine.
``SNIPS``     the recon's ``rows.jsonl`` (train/test already assigned by file).
``ATIS``      the Rasa-format ``train.json`` / ``test.json``; ``intent`` is the label.

Rows are written in file order and never deduplicated here: the dedupe that matters is on the
training side and is counted there, where it can be reported.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

from public_corpus import RAW, CorpusError

OOS = "oos"


def _write(root: Path, rows: Iterable[tuple[str, str, str]]) -> dict:
    target = root / "normalized.tsv"
    counts: dict[str, int] = {}
    rows = list(rows)
    if not rows:
        raise CorpusError(f"{root.name}: normalized nothing")
    with target.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t", lineterminator="\n")
        writer.writerow(["split", "text", "intent"])
        for split, text, intent in rows:
            text = " ".join(text.split())
            if not text:
                continue
            writer.writerow([split, text, intent])
            counts[split] = counts.get(split, 0) + 1
    return {"path": str(target), "rows": sum(counts.values()), "splits": counts}


def clinc150() -> dict:
    raw = json.loads((RAW / "CLINC150" / "data_full.json").read_text(encoding="utf-8"))
    rows: list[tuple[str, str, str]] = []
    for key, split in (("train", "train"), ("val", "train"),
                       ("oos_train", "train"), ("oos_val", "train"),
                       ("test", "test"), ("oos_test", "test")):
        for text, intent in raw[key]:
            rows.append((split, text, intent))
    return _write(RAW / "CLINC150", rows)


def hwu64() -> dict:
    rows: list[tuple[str, str, str]] = []
    base = RAW / "HWU64" / "split" / "kfold1"
    # The split files carry the annotated utterance, not the master CSV's `answer_normalised`,
    # so the filled annotation is the text: `answer_from_anno`, falling back to the two raw
    # columns. Empty cells exist in this corpus and are dropped by _write's blank check.
    text_columns = ("answer_normalised", "answer_from_anno", "answer_annotation", "answer")
    for split in ("train", "test"):
        files = sorted((base / split).glob("*.csv"))
        if not files:
            raise CorpusError(f"HWU64: no csv files under {base / split}")
        for path in files:
            with path.open(encoding="utf-8", newline="") as fh:
                for entry in csv.DictReader(fh, delimiter=";"):
                    text = next((entry[c] for c in text_columns if (entry.get(c) or "").strip()),
                                "")
                    if text:
                        rows.append((split, text, path.stem))
    return _write(RAW / "HWU64", rows)


def snips() -> dict:
    rows: list[tuple[str, str, str]] = []
    with (RAW / "SNIPS" / "rows.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            entry = json.loads(line)
            rows.append((entry["split"], entry["text"], entry["intent"]))
    return _write(RAW / "SNIPS", rows)


def atis() -> dict:
    rows: list[tuple[str, str, str]] = []
    for name, split in (("train.json", "train"), ("test.json", "test")):
        raw = json.loads((RAW / "ATIS" / name).read_text(encoding="utf-8"))
        for entry in raw["rasa_nlu_data"]["common_examples"]:
            rows.append((split, entry["text"], entry["intent"]))
    return _write(RAW / "ATIS", rows)


NORMALIZERS = {"CLINC150": clinc150, "HWU64": hwu64, "SNIPS": snips, "ATIS": atis}


def main(argv: list[str]) -> int:
    names = argv or list(NORMALIZERS)
    for name in names:
        if name not in NORMALIZERS:
            raise SystemExit(f"unknown dataset {name!r}; known: {list(NORMALIZERS)}")
        print(name, json.dumps(NORMALIZERS[name]()))
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
