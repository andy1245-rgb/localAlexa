"""transfer — the four measurements the brief names, in one place, on one fold assignment.

The model is `sklearn_text|vectorizer=tfidf_word|clf=linsvc`: rung 0 of the classifier ladder,
the one whose grouped accuracy is the 89.79% (re-measured here as 90.05%, one row apart — see
`docs/public-data-2026-11-10.md`). Every arm uses the same fold map over the 382 rows' 110
phrasing families, the same option set, and the same scoring functions, because the only
question is what happens when public rows are added to the training side.

Four measurements:

``ceiling``      protocol L: train on the reference pool (+ public rows), score the 382 rows
``public_only``  train on public rows mapped to localAlexa's labels, score the 382 rows
``reverse``      train on the 382 rows, score the public test rows mapped to those labels
``controls``     the reference-only arms at two classifiers, so a null is not one model's opinion

    python experiments/public/transfer.py all
    python experiments/public/transfer.py baseline
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

import localbench  # noqa: E402  (also puts the harness on sys.path)
import public_corpus  # noqa: E402
from bench.harness import metrics  # noqa: E402
from bench.harness.protocol import build  # noqa: E402
import bench.harness.predictors  # noqa: F401,E402  (registers the arms)

from report import format_table  # noqa: E402

RESULTS = localbench.LOCALALEXA / "results" / "public"
DATASETS = ("CLINC150", "HWU64", "SNIPS", "ATIS")
DATASET_SLUGS = {"CLINC150": "CLINC150", "HWU64": "HWU64", "SNIPS": "SNIPS", "ATIS": "ATIS"}

MODELS = {
    "svc": "sklearn_text|vectorizer=tfidf_word|clf=linsvc",
    "logreg": "sklearn_text|vectorizer=tfidf_word|clf=logreg|C=100",
}

# The second classifier is a check on the null, not a second sweep: logreg on 40k rows is an
# order of magnitude slower than the svc and the question it answers is "is this one solver's
# opinion". So it runs on the reference arm and on the two arms that a recommendation would
# actually pick.
CONTROL_MODEL_ARMS = ("refs", "union", "union_cap100", "union_toolonly", "HWU64",
                      "HWU64_toolonly")


def load_corpora(bench: localbench.Bench) -> dict[str, public_corpus.PublicCorpus]:
    return {slug: public_corpus.load(slug, bench.options) for slug in DATASETS}


def union_pairs(corpora, confidence: str = "all") -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for slug in DATASETS:
        out.extend(public_corpus.pairs(corpora[slug], confidence))
    return out


def union_capped(corpora, cap: int, confidence: str = "all") -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for slug in DATASETS:
        out.extend(public_corpus.cap_per_label(public_corpus.pairs(corpora[slug], confidence), cap))
    return out


def tool_only(pairs_: Sequence[tuple[str, str]]) -> list[tuple[str, str]]:
    """Drop the rows mapped to the abstain class.

    Three of the four corpora are 85–100% out-of-scope for localAlexa's 17 tools. Those rows
    are not useless — they are the *right* answer for utterances this router should refuse —
    but they arrive in the tens of thousands against a reference pool that carries 25, so an
    uncapped union is a class-prior experiment before it is a supervision experiment. This arm
    and the cap are how the two are told apart.
    """
    return [(text, label) for text, label in pairs_ if label != "__none__"]


def cap_label(pairs_: Sequence[tuple[str, str]], label: str, cap: int,
              seed: int = 0) -> list[tuple[str, str]]:
    """`pairs_` with at most `cap` rows of one label, the rest passed through unchanged."""
    import hashlib

    matching = [(t, l) for t, l in pairs_ if l == label]
    others = [(t, l) for t, l in pairs_ if l != label]
    matching.sort(key=lambda p: hashlib.sha256(f"{seed}|{label}|{p[0]}".encode()).hexdigest())
    return others + matching[:cap]


def fold_spread(per_fold: Sequence[dict]) -> dict:
    accs = [p["accuracy"] for p in per_fold]
    return {"min": min(accs), "max": max(accs), "range": round(max(accs) - min(accs), 4),
            "mean": round(sum(accs) / len(accs), 4)}


def ceiling_arm(bench, model: str, extra: Sequence[tuple[str, str]]) -> dict:
    """Protocol L with `extra` appended to every fold's reference pool."""
    recs, per_fold = localbench.grouped_refs(lambda: build(MODELS[model]), bench, extra)
    return {"records": recs, "per_fold": per_fold, "summary": localbench.summarise(recs),
            "fold_spread": fold_spread(per_fold),
            "n_train_per_fold": [p["n_train"] for p in per_fold]}


def public_only_arm(bench, model: str, pairs_: Sequence[tuple[str, str]]) -> dict:
    """Train on public rows only, score all 382 localAlexa rows, report per family fold.

    A corpus that maps to a single label cannot train a router at all, and that is a result
    worth recording rather than an error to swallow: the arm is marked refused with the reason.
    """
    labels = {label for _, label in pairs_}
    if len(labels) < 2:
        return {"records": [], "per_fold": [], "fold_spread": None, "n_train": len(pairs_),
                "summary": {"n": 0, "correct": 0, "accuracy": None, "none_recall": None,
                            "n_none": 0, "wilson95": None},
                "refused": f"one label only ({sorted(labels)}) — nothing to discriminate"}
    train = localbench.pool_rows(pairs_, bench.options)
    recs = localbench.fit_and_score("public_only", lambda: build(MODELS[model]), train, bench.rows)
    by_fold = []
    for fold in range(localbench.FOLDS):
        sub = metrics.accuracy([r for r, k in zip(recs, bench.fold_of_row) if k == fold])
        by_fold.append({"fold": fold, "n_test": sub["n"], "accuracy": round(sub["accuracy"], 4),
                        "correct": sub["correct"]})
    return {"records": recs, "per_fold": by_fold, "summary": localbench.summarise(recs),
            "fold_spread": fold_spread(by_fold), "n_train": len(train)}


def reverse_arm(bench, model: str, corpus: public_corpus.PublicCorpus) -> dict:
    """Train on the 382 rows, score that corpus's mapped test rows."""
    train = localbench.pool_rows(
        [(r["text"], r["tool"]) for r in json.loads(
            (localbench.LOCALALEXA / "data" / "routing_382.json").read_text())],
        bench.options)
    test = public_corpus.as_harness_rows(corpus.test, bench.options)
    recs = localbench.fit_and_score("reverse", lambda: build(MODELS[model]), train, test)
    by_intent: dict[str, list] = {}
    for r in recs:
        by_intent.setdefault(r.group, []).append(r)
    per_intent = {intent: {"n": len(v), "accuracy": round(metrics.accuracy(v)["accuracy"], 4),
                           "label": v[0].label}
                  for intent, v in sorted(by_intent.items())}
    return {"records": recs, "summary": localbench.summarise(recs), "n_train": len(train),
            "per_intent": per_intent}


def compare(reference: dict, treatment: dict) -> dict | None:
    """Paired on the same 382 rows: fixed/broke, exact McNemar, folds improved."""
    if not treatment.get("records"):
        return None
    p = metrics.paired(reference["records"], treatment["records"])
    ref_folds = reference["per_fold"]
    got_folds = treatment["per_fold"]
    return {
        "delta_accuracy": round(treatment["summary"]["accuracy"]
                                - reference["summary"]["accuracy"], 4),
        "fixed": p["fixed"], "broke": p["broke"], "mcnemar_p": round(p["p"], 4),
        "folds_better": sum(1 for a, b in zip(ref_folds, got_folds) if b["correct"] > a["correct"]),
        "folds_worse": sum(1 for a, b in zip(ref_folds, got_folds) if b["correct"] < a["correct"]),
        "folds_tied": sum(1 for a, b in zip(ref_folds, got_folds) if b["correct"] == a["correct"]),
    }


def run_all() -> dict:
    bench = localbench.load()
    corpora = load_corpora(bench)
    out: dict = {
        "bench": {"n_rows": len(bench.rows), "n_families": len(set(bench.families)),
                  "n_labels": len(bench.options), "folds": localbench.FOLDS,
                  "quoted_baseline": 0.8979, "model": MODELS},
        "datasets": {slug: {"stats": c.stats(), "dedup": public_corpus.dedup_report(c),
                            "train_rows_mapped": len(public_corpus.pairs(c)),
                            "test_rows": len(c.test)}
                     for slug, c in corpora.items()},
    }

    for model in MODELS:
        ceilings: dict[str, dict] = {}
        print(f"[{model}] reference arm", flush=True)
        base = ceiling_arm(bench, model, [])
        ceilings["refs"] = base
        variants: dict[str, list[tuple[str, str]]] = {}
        for slug in DATASETS:
            variants[slug] = public_corpus.pairs(corpora[slug])
            variants[f"{slug}_toolonly"] = tool_only(variants[slug])
        union = union_pairs(corpora)
        variants["union"] = union
        variants["union_high"] = union_pairs(corpora, "high")
        variants["union_cap100"] = union_capped(corpora, 100)
        variants["union_cap25"] = union_capped(corpora, 25)
        variants["union_toolonly"] = tool_only(union)
        variants["union_none25"] = cap_label(union, "__none__", 25)
        variants["HWU64_cap25"] = public_corpus.cap_per_label(
            public_corpus.pairs(corpora["HWU64"]), 25)
        for name, extra in variants.items():
            if not extra:
                continue  # a corpus with no tool-mapped row is the reference arm again
            if name not in CONTROL_MODEL_ARMS and model != "svc":
                continue
            print(f"  {model} {name}: {len(extra)} extra rows", flush=True)
            ceilings[name] = ceiling_arm(bench, model, extra)
            print(f"    -> {ceilings[name]['summary']['accuracy']:.4f}", flush=True)
        out[f"ceiling_{model}"] = {
            "arms": {name: {"summary": arm["summary"], "per_fold": arm["per_fold"],
                            "n_train_per_fold": arm["n_train_per_fold"]}
                     for name, arm in ceilings.items()},
            "vs_refs": {name: compare(base, arm) for name, arm in ceilings.items()
                        if name != "refs" and compare(base, arm) is not None},
        }
        out[f"_records_{model}"] = ceilings

    # the two transfer directions, on the same model as the baseline
    public_only = {slug: public_only_arm(bench, "svc", public_corpus.pairs(corpora[slug]))
                   for slug in DATASETS}
    public_only["union"] = public_only_arm(bench, "svc", union_pairs(corpora))
    out["public_only"] = {
        "arms": {name: {"summary": arm["summary"], "per_fold": arm["per_fold"],
                        "n_train": arm["n_train"], "refused": arm.get("refused")}
                 for name, arm in public_only.items()},
        "vs_refs": {name: compare(out["_records_svc"]["refs"], arm)
                    for name, arm in public_only.items()
                    if compare(out["_records_svc"]["refs"], arm) is not None},
    }

    reverse = {slug: reverse_arm(bench, "svc", corpora[slug]) for slug in DATASETS}
    out["reverse"] = {slug: {"summary": arm["summary"], "n_train": arm["n_train"]}
                      for slug, arm in reverse.items()}
    out["reverse"]["union"] = localbench.summarise(
        [r for arm in reverse.values() for r in arm["records"]])
    out["reverse_per_intent"] = {slug: arm["per_intent"] for slug, arm in reverse.items()}

    for key in [k for k in out if k.startswith("_records_")]:
        del out[key]
    return out


def baseline_only() -> dict:
    bench = localbench.load()
    arms = {model: ceiling_arm(bench, model, []) for model in MODELS}
    return {"bench": {"n_rows": len(bench.rows), "n_families": len(set(bench.families))},
            "arms": {m: {"summary": a["summary"], "per_fold": a["per_fold"]}
                     for m, a in arms.items()}}


def main(argv: Sequence[str]) -> int:
    stage = argv[0] if argv else "all"
    if stage == "baseline":
        payload = baseline_only()
        print(json.dumps(payload, indent=1))
        return 0
    if stage != "all":
        raise SystemExit(f"unknown stage {stage!r}; known: baseline, all")
    payload = run_all()
    print(format_table(payload))
    RESULTS.mkdir(parents=True, exist_ok=True)
    target = RESULTS / "transfer.json"
    target.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(f"\nwrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
