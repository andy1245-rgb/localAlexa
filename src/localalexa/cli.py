"""Command line: reproduce the numbers in the README.

    python -m localalexa eval --split all        # row-wise, family-held-out, strict
    python -m localalexa regex                   # tier 0 alone, no model needed
    python -m localalexa classic                 # the three TF-IDF configs
    python -m localalexa cascade                 # full cascade, needs embeddings
    python -m localalexa ladder --rungs all      # the classifier ladder, docs/classifier-ladder

Every subcommand reads the shipped data under ``data/`` and writes nothing.
"""

from __future__ import annotations

import argparse
import json

from .classifiers import MODEL_SPECS, train_and_score
from .data import NONE_LABEL, load_registry, load_routing_set
from .evaluate import family_held_out, row_wise


def _render(payload: dict[str, object]) -> str:
    return json.dumps(payload, indent=2)


def _cmd_eval(args: argparse.Namespace) -> int:
    from .evaluate import measure

    payload = measure(
        model=args.model,
        splits=("full",) if args.split == "all" else (args.split,),
        bootstrap_resamples=args.bootstrap,
    )
    print(_render(payload))
    return 0


def _cmd_regex(_: argparse.Namespace) -> int:
    from .cascade import CompiledPatterns, regex_tier_metrics

    registry = load_registry()
    rows = load_routing_set()
    metrics = regex_tier_metrics(rows, CompiledPatterns.from_registry(registry))
    metrics.pop("_hits")
    print(_render(metrics))
    return 0


def _cmd_classic(args: argparse.Namespace) -> int:
    registry = load_registry()
    rows = load_routing_set()
    train = [
        (ref, tool.id) for tool in registry.tools for ref in tool.references
    ] + [(ref, NONE_LABEL) for ref in registry.none_references]
    results = [
        train_and_score(name, train, [r.text for r in rows], [r.tool for r in rows])
        for name in MODEL_SPECS
    ]
    print(_render(results))
    return 0


def _cmd_cascade(args: argparse.Namespace) -> int:
    import sys

    from .cascade import run_cascade
    from .router import SemanticRouter

    registry = load_registry()
    rows = load_routing_set()
    try:
        router = SemanticRouter(registry)
        router.predict("warmup query")
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    payload = run_cascade(
        rows,
        registry,
        router.predict_batch,
        sim_threshold=args.sim,
        conf_threshold=args.conf,
    )
    print(_render(payload))
    return 0


def _cmd_ladder(args: argparse.Namespace) -> int:
    from .ladder.report import format_table, run_ladder, write_artifact
    from .ladder.catalog import available_rungs

    rungs = available_rungs(args.rungs)
    fold_counts = tuple(int(f) for f in args.folds.split(","))
    payload = run_ladder(
        rungs,
        fold_counts=fold_counts,
        protocols=tuple(args.protocols.split(",")),
        keep_predictions=args.keep_predictions,
    )
    print(format_table(payload))
    if args.out:
        print(f"\nwrote {write_artifact(payload, args.out)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="localalexa", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    eval_p = sub.add_parser("eval", help="accuracy under the row-wise and held-out splits")
    eval_p.add_argument(
        "--split", choices=("rowwise", "family", "strict", "all"), default="all"
    )
    eval_p.add_argument("--model", default="tfidf-word+linearsvc", choices=sorted(MODEL_SPECS))
    eval_p.add_argument("--bootstrap", type=int, default=2000)
    eval_p.set_defaults(func=_cmd_eval)

    regex_p = sub.add_parser("regex", help="tier-0 regex coverage and accuracy")
    regex_p.set_defaults(func=_cmd_regex)

    classic_p = sub.add_parser("classic", help="the three TF-IDF configurations")
    classic_p.set_defaults(func=_cmd_classic)

    cascade_p = sub.add_parser("cascade", help="regex -> router -> abstain cascade")
    cascade_p.add_argument("--sim", type=float, default=0.45)
    cascade_p.add_argument("--conf", type=float, default=0.35)
    cascade_p.set_defaults(func=_cmd_cascade)

    ladder_p = sub.add_parser("ladder", help="the classifier ladder (rungs 0-6)")
    ladder_p.add_argument("--rungs", default="all", help="comma list of rung names, or 'all'")
    ladder_p.add_argument("--folds", default="5", help="comma list of grouped fold counts")
    ladder_p.add_argument(
        "--protocols", default="grouped,rowwise", help="comma list of grouped,rowwise"
    )
    ladder_p.add_argument("--keep-predictions", action="store_true")
    ladder_p.add_argument("--out", default=None, help="artifact path to write")
    ladder_p.set_defaults(func=_cmd_ladder)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
