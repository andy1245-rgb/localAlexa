"""Job 1 — more phrasings, not more slot fills: deterministic paraphrase families.

The corpus is 110 templates times slot fills, and more fills stopped helping at 16 per label.
What is missing is *sentence shape*, so this generates new shapes from the reference pool by
deterministic transformation — no model, fully reproducible, each row labelled with the
transformations that produced it — then filters by lexical diversity against what already
exists, then retrains on the same grouped folds and asks whether the family-held-out number
moves.

Two filters, and they are different questions:

* **adds nothing** — a generated row that is nearly identical to a reference is a slot fill
  wearing a hat. Dropped, and the count reported.
* **leaks** — a generated row that is nearly identical to a *held-out evaluation* row would be
  the evaluator handing itself the answer. Dropped, counted, and reported per fold.

The diversity distribution is written out in full, because "we generated 5,000 rows" says
nothing if they are all one sentence.

    python job1_paraphrase_families.py --out results/dell321/job1.json

Reads ``data/routing_382.json`` and ``data/registry.json`` from the tree it is run in.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))
sys.path.insert(0, str(HERE))

from localalexa.cascade import CompiledPatterns  # noqa: E402
from localalexa.data import (  # noqa: E402
    NONE_LABEL,
    load_registry,
    load_routing_set,
    phrasing_families,
    template_index,
)
from localalexa.ladder.folds import DEFAULT_FOLDS, build_folds  # noqa: E402
from localalexa.ladder.pools import fold_pool, reference_pool  # noqa: E402
from localalexa.variants import Fuser, fit_plain  # noqa: E402

# ------------------------------------------------------------------ transformations

POLITE_PREFIXES = ("could you ", "can you ", "would you ", "please ")
CONTEXTS = (
    "in the kitchen, ",
    "in the bedroom, ",
    "in the living room, ",
    "right now, ",
    "for me, ",
    "before bed, ",
    "after work, ",
    "quick, ",
    "hey, ",
    "okay so ",
)
CONTRACTIONS = {
    # "what are" -> "what's" is deliberately absent: the contraction is ambiguous ("what's left"
    # is "what is"), and expanding it back produced "what are left on my countdown"
    "what is": "what's",
    "do not": "don't",
    "does not": "doesn't",
    "it is": "it's",
    "i am": "i'm",
    "cannot": "can't",
    "can not": "can't",
    "let us": "let's",
}
EXPANSIONS = {v: k for k, v in CONTRACTIONS.items()}
# curated, verb-level synonyms: generic enough that no intent changes
SYNONYMS = (
    ("turn on", "switch on"),
    ("turn off", "switch off"),
    ("switch on", "turn on"),
    ("switch off", "turn off"),
    ("set a timer", "start a timer"),
    ("start a timer", "set a timer"),
    ("cancel", "stop"),
    ("stop", "cancel"),
    ("get rid of", "remove"),
    ("add", "put"),
    ("take", "get"),
    ("show me", "give me"),
    ("give me", "show me"),
    ("how much", "what's the level of"),
    ("how many", "what count of"),
)
LEADING_PREP = re.compile(r"^((?:in|on|at|for|near|by|inside|outside)\b[^,]*),\s*(.+)$", re.I)


def t_polite(text: str) -> str | None:
    lowered = text.lower()
    if any(lowered.startswith(p) for p in POLITE_PREFIXES):
        return None
    # a row that already opens with a context clause keeps it first: "in the kitchen, could you
    # ..." rather than "could you in the kitchen, ..."
    match = LEADING_PREP.match(text)
    if match:
        head, tail = match.group(1), match.group(2).rstrip("?.")
        return f"{head}, could you {tail}?"
    if any(lowered.startswith(c.rstrip(", ")) for c in CONTEXTS):
        head, _, tail = text.partition(", ")
        return f"{head}, could you {tail.rstrip('?.')}?"
    return "could you " + text.rstrip("?.") + "?"


def t_drop_polite(text: str) -> str | None:
    lowered = text.lower()
    for prefix in POLITE_PREFIXES:
        if lowered.startswith(prefix):
            stripped = text[len(prefix):].strip()
            return stripped[:1].upper() + stripped[1:]
    return None


def t_context(text: str) -> str | None:
    lowered = text.lower()
    if any(lowered.startswith(c) for c in CONTEXTS) or LEADING_PREP.match(text):
        return None
    # deterministic choice so a rerun reproduces the corpus exactly
    return CONTEXTS[sum(map(ord, text)) % len(CONTEXTS)] + text


def t_reorder(text: str) -> str | None:
    match = LEADING_PREP.match(text)
    if not match:
        return None
    head, tail = match.group(1), match.group(2)
    return tail.rstrip("?.") + " " + head.strip()


def t_contract(text: str) -> str | None:
    for full, short in CONTRACTIONS.items():
        if re.search(rf"\b{re.escape(full)}\b", text, re.I):
            return re.sub(rf"\b{re.escape(full)}\b", short, text, count=1, flags=re.I)
    return None


def t_expand(text: str) -> str | None:
    for short, full in EXPANSIONS.items():
        if re.search(rf"\b{re.escape(short)}\b", text, re.I):
            return re.sub(rf"\b{re.escape(short)}\b", full, text, count=1, flags=re.I)
    return None


def t_synonym(text: str) -> str | None:
    lowered = text.lower()
    for source, target in SYNONYMS:
        if source in lowered:
            return re.sub(rf"\b{re.escape(source)}\b", target, text, count=1, flags=re.I)
    return None


TRANSFORMS = (
    ("polite", t_polite),
    ("drop_polite", t_drop_polite),
    ("context", t_context),
    ("reorder", t_reorder),
    ("contract", t_contract),
    ("expand", t_expand),
    ("synonym", t_synonym),
)


def generate(labelled, rounds: int = 2) -> list[dict]:
    """Apply every transform, then re-apply to the results for ``rounds`` generations.

    The label travels with the row: a paraphrase of a reference keeps that reference's intent,
    which is the assumption the whole job rests on — a transformation that changed the intent
    would be teaching the model a wrong label.
    """
    born: list[dict] = [{"text": text, "label": label, "transformations": (), "generation": 0}
                        for text, label in labelled]
    frontier = list(born)
    for generation in range(1, rounds + 1):
        new: list[dict] = []
        for row in frontier:
            for name, fn in TRANSFORMS:
                produced = fn(row["text"])
                if produced and produced != row["text"]:
                    new.append({"text": produced,
                                "label": row["label"],
                                "transformations": row["transformations"] + (name,),
                                "generation": generation})
        born.extend(new)
        frontier = new
    return born


def ngrams(text: str, n: int = 3) -> set[tuple[str, ...]]:
    words = re.findall(r"[a-z0-9']+", text.lower())
    if len(words) < n:
        return {tuple(words)} if words else set()
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def jaccard(left: set, right: set) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def max_similarity(row: set, others) -> float:
    return max((jaccard(row, other) for other in others), default=0.0)


# ------------------------------------------------------------------ recipes


def recipe_linear(registry, pool):
    del registry  # the word model does not look at the registry
    model = fit_plain(pool)
    return model.predict


def recipe_fusion(registry, pool):
    from localalexa.router import SemanticRouter

    router = SemanticRouter(registry)
    router.predict("warm up")
    fused = Fuser(("linear", "regex", "router")).fit(registry, pool, router=router)
    return fused.predict


def evaluate(recipe, registry, rows, folds, extra_pool_for_fold, name) -> dict:
    per_fold = []
    started = time.perf_counter()
    for indices, held in folds:
        pool = extra_pool_for_fold(indices, held)
        predict = recipe(registry, pool)
        correct = sum(1 for i in indices if predict(rows[i].text) == rows[i].tool)
        per_fold.append({"n": len(indices), "correct": correct,
                         "accuracy": round(correct / len(indices), 4)})
    total = sum(f["correct"] for f in per_fold)
    return {
        "name": name,
        "accuracy": round(total / len(rows), 4),
        "correct": total,
        "n": len(rows),
        "per_fold": [f["accuracy"] for f in per_fold],
        "spread": {
            "mean": round(statistics.mean(f["accuracy"] for f in per_fold), 4),
            "sd": round(statistics.stdev(f["accuracy"] for f in per_fold), 4),
        },
        "wall_seconds": round(time.perf_counter() - started, 1),
    }


def paired(a, b) -> dict:
    deltas = [x - y for x, y in zip(a["per_fold"], b["per_fold"])]
    return {
        "mean_points": round(100 * statistics.mean(deltas), 2),
        "sd_points": round(100 * statistics.stdev(deltas), 2) if len(deltas) > 1 else 0.0,
        "folds_better": sum(1 for d in deltas if d > 0),
        "folds_worse": sum(1 for d in deltas if d < 0),
        "folds_tied": sum(1 for d in deltas if d == 0),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--diversity-floor", type=float, default=0.35,
                        help="minimum 1 - max 3-gram Jaccard against the existing pool")
    parser.add_argument("--leak-ceiling", type=float, default=0.80,
                        help="drop a generated row that is this similar to a held-out row")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--recipes", default="linear,fusion")
    parser.add_argument("--out", default="results/dell321/job1.json")
    args = parser.parse_args(argv)

    registry = load_registry()
    rows = load_routing_set()
    families = phrasing_families(rows, template_index(registry))
    assignment = build_folds(families, DEFAULT_FOLDS)
    index = {i: fold for i, fold in
             ((i, assignment.fold_of(families[i])) for i in range(len(rows)))}
    fold_indices = [[] for _ in range(DEFAULT_FOLDS)]
    for i, fold in index.items():
        fold_indices[fold].append(i)
    fold_groups = [(idx, {families[i] for i in idx}) for idx in fold_indices]

    base = reference_pool(registry)
    print(f"references {len(base)}  rows {len(rows)}  folds {DEFAULT_FOLDS}")

    # ---- generate
    generated = generate(base, rounds=args.rounds)
    unique: dict[str, dict] = {}
    for row in generated:
        unique.setdefault(row["text"], row)
    print(f"generated {len(generated)} candidates, {len(unique)} distinct")

    # ---- diversity against the pool, leakage against the evaluation rows
    pool_grams = [ngrams(text) for text, _ in base]
    eval_grams = [ngrams(row.text) for row in rows]
    kept, dropped_same, dropped_leak = [], 0, 0
    overlaps, leak_overlaps = [], []
    for text, row in unique.items():
        grams = ngrams(text)
        similar = max_similarity(grams, pool_grams)
        overlaps.append(similar)
        if 1.0 - similar < args.diversity_floor:
            dropped_same += 1
            continue
        leak = max_similarity(grams, eval_grams)
        leak_overlaps.append(leak)
        if leak > args.leak_ceiling:
            dropped_leak += 1
            continue
        kept.append(row)
    print(f"kept {len(kept)}  dropped (adds nothing) {dropped_same}  dropped (leaks) {dropped_leak}")

    def distribution(values):
        ordered = sorted(values)
        if not ordered:
            return {}
        return {
            "n": len(ordered),
            "min": round(ordered[0], 3),
            "p10": round(ordered[int(0.1 * len(ordered))], 3),
            "p50": round(statistics.median(ordered), 3),
            "p90": round(ordered[int(0.9 * len(ordered))], 3),
            "max": round(ordered[-1], 3),
            "mean": round(statistics.mean(ordered), 3),
        }

    transform_counts = Counter(name for row in kept for name in row["transformations"])
    print(f"diversity of kept rows: {distribution([1.0 - max_similarity(ngrams(r['text']), pool_grams) for r in kept])}")

    # ---- the experiment: same folds, expanded pool
    def baseline_pool(indices, held):
        return fold_pool(registry, held)

    def expanded_pool(indices, held):
        base_pool = fold_pool(registry, held)
        held_grams = [ngrams(rows[i].text) for i in indices]
        extra = []
        for row in kept:
            # the generated row must not resemble this fold's held-out rows either: the global
            # filter was against the whole evaluation set, the fold filter is the one that matters
            if max_similarity(ngrams(row["text"]), held_grams) > args.leak_ceiling:
                continue
            extra.append((row["text"], row["label"]))
        return base_pool + extra

    payload = {
        "config": {
            "diversity_floor": args.diversity_floor,
            "leak_ceiling": args.leak_ceiling,
            "rounds": args.rounds,
            "folds": DEFAULT_FOLDS,
            "recipes": args.recipes.split(","),
        },
        "generation": {
            "candidates": len(generated),
            "distinct": len(unique),
            "kept": len(kept),
            "dropped_adds_nothing": dropped_same,
            "dropped_leaks": dropped_leak,
            "similarity_to_pool": distribution(overlaps),
            "similarity_to_eval": distribution(leak_overlaps),
            "diversity_of_kept": distribution(
                [1.0 - max_similarity(ngrams(r["text"]), pool_grams) for r in kept]),
            "transformations_kept": dict(transform_counts),
            "examples": [{"text": r["text"], "transformations": list(r["transformations"])}
                         for r in random.Random(0).sample(kept, min(12, len(kept)))],
        },
        "arms": {},
    }

    available = {"linear": recipe_linear, "fusion": recipe_fusion}
    for recipe_name in args.recipes.split(","):
        if recipe_name not in available:
            continue
        recipe = available[recipe_name]
        base_result = evaluate(
            lambda registry, pool: recipe(registry, pool), registry, rows, fold_groups,
            baseline_pool, f"{recipe_name}:references")
        expanded_result = evaluate(
            lambda registry, pool: recipe(registry, pool), registry, rows, fold_groups,
            expanded_pool, f"{recipe_name}:references+paraphrases")
        expanded_result["paired_vs_references"] = paired(expanded_result, base_result)
        payload["arms"][recipe_name] = {"references": base_result,
                                         "expanded": expanded_result}
        print(f"{recipe_name}: references {base_result['accuracy']:.4f} "
              f"(sd {base_result['spread']['sd']:.4f}) -> expanded "
              f"{expanded_result['accuracy']:.4f} (sd {expanded_result['spread']['sd']:.4f}) "
              f"paired {expanded_result['paired_vs_references']['mean_points']:+.2f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
