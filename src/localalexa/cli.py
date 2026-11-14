"""Command line: reproduce the numbers in the README.

    python -m localalexa eval --split all        # row-wise, family-held-out, strict
    python -m localalexa regex                   # tier 0 alone, no model needed
    python -m localalexa classic                 # the three TF-IDF configs
    python -m localalexa cascade                 # full cascade, needs embeddings
    python -m localalexa ladder --rungs all      # the classifier ladder, docs/classifier-ladder
    python -m localalexa collect summary         # the ten-second weekly read
    python -m localalexa collect derive          # replay the log into labels/rejects
    python -m localalexa collect bench           # what logging costs the request path
    python -m localalexa ask "turn off the hall light"   # one request through the real path

Every subcommand except ``collect`` reads the shipped data under ``data/`` and
writes nothing; ``collect`` reads and writes the local log under
``$LOCALALEXA_COLLECT`` (default ``<repo>/var/collect``) and never touches the
network.
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


def _cmd_ask(args: argparse.Namespace) -> int:
    from .service import Assistant, default_predictor

    registry = load_registry()
    predictor, source = default_predictor(registry)
    assistant = Assistant(
        registry,
        predictor=predictor,
        log=args.root,
        sim_threshold=args.sim,
        conf_threshold=args.conf,
        settle=not args.no_settle,
        record_requests=not args.no_log,
    )
    try:
        response = assistant.handle(args.text)
        print(f"predictor: {source}")
        print(response.describe())
        if response.record_id:
            print(f"recorded {response.record_id} in {assistant.collector.log.path}")
        if response.settled is not None:
            wrote = ", ".join(response.settled.wrote) or "nothing"
            print(
                f"settled {response.settled.new_rows} row(s) in {response.settled.derive_ms} ms: "
                f"{response.settled.corrections} correction(s), "
                f"{response.settled.labels} label(s), wrote {wrote}"
            )
    finally:
        assistant.close()
    return 0


def _cmd_collect(args: argparse.Namespace) -> int:
    from .collection import derive as replay
    from .collection import summarise, render, measure_append, measure_tap
    from .collection.corrections import CorrectionRule
    from .collection.autoaccept import AutoAcceptRule
    from .collection.log import RequestLog, collect_root, write_jsonl
    from .collection.records import EffectRecord, RequestRecord

    if args.action == "bench":
        append = measure_append(args.iterations)
        tap = measure_tap(max(args.iterations // 4, 50))
        print(json.dumps({"append_us": append, "observe_us": tap}, indent=2))
        return 0

    log = RequestLog(args.root)
    try:
        records = [RequestRecord.from_dict(row) for row in log.rows() if row.get("kind") == "request"]
        effects = [EffectRecord.from_dict(row) for row in log.effects() if row.get("kind") == "effect"]
    finally:
        log.close()

    if args.action == "derive":
        from .collection.opinion import SecondOpinion

        derived = replay(
            records,
            effects,
            correction_rule=CorrectionRule(window_seconds=args.window),
            auto_rule=AutoAcceptRule(margin_threshold=args.margin),
            opinion=None if args.no_opinion else SecondOpinion(load_registry()),
        )
        root = collect_root(args.root)
        written = [
            write_jsonl(root / "rejects.jsonl", list(derived.reject_pile)),
            write_jsonl(root / "corrections.jsonl", [c.as_dict() for c in derived.corrections]),
            write_jsonl(root / "labels.jsonl", [row.as_dict() for row in derived.events]),
            write_jsonl(root / "labels.reduced.jsonl", [row.as_dict() for row in derived.labels]),
        ]
        print(json.dumps({"stats": dict(derived.stats), "wrote": [str(p) for p in written]}, indent=2))
        return 0

    root = collect_root(args.root)
    reduced = root / "labels.reduced.jsonl"
    journal = root / "labels.jsonl"
    labels_path = reduced if reduced.is_file() else journal
    labels: list[dict] = []
    if labels_path.is_file():
        labels = [
            json.loads(line)
            for line in labels_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if labels_path == journal:
            from .collection.derive import reduce_labels
            from .collection.records import LabelRow

            labels = [
                row.as_dict()
                for row in reduce_labels(
                    LabelRow(
                        request_id=row["request_id"],
                        ts=float(row["ts"]),
                        text=row["text"],
                        label=row["label"],
                        polarity=int(row["polarity"]),
                        strength=row["strength"],
                        source=row["source"],
                        detail=row.get("detail", {}),
                    )
                    for row in labels
                )
            ]

    summary = summarise(records, labels, days=args.days)
    print(json.dumps(summary.as_dict(), indent=2) if args.json else render(summary))
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

    ask_p = sub.add_parser("ask", help="route one utterance through the real request path")
    ask_p.add_argument("text", help="the utterance")
    ask_p.add_argument("--root", default=None, help="collection root (default $LOCALALEXA_COLLECT or var/collect)")
    ask_p.add_argument("--sim", type=float, default=0.45)
    ask_p.add_argument("--conf", type=float, default=0.35)
    ask_p.add_argument("--no-log", action="store_true", help="do not record the request")
    ask_p.add_argument("--no-settle", action="store_true", help="record but do not derive yet")
    ask_p.set_defaults(func=_cmd_ask)

    collect_p = sub.add_parser("collect", help="hands-off collection: log, derive, summarise")
    collect_p.add_argument("action", choices=("derive", "summary", "bench"))
    collect_p.add_argument("--root", default=None, help="collection root (default $LOCALALEXA_COLLECT or var/collect)")
    collect_p.add_argument("--days", type=float, default=7.0, help="summary window in days")
    collect_p.add_argument("--window", type=float, default=45.0, help="correction window, seconds")
    collect_p.add_argument("--margin", type=float, default=0.05, help="auto-accept margin threshold")
    collect_p.add_argument("--iterations", type=int, default=2000, help="bench: appends to measure")
    collect_p.add_argument("--json", action="store_true", help="summary: machine-readable")
    collect_p.add_argument(
        "--no-opinion", action="store_true", help="derive: skip the offline second-opinion signal"
    )
    collect_p.set_defaults(func=_cmd_collect)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
