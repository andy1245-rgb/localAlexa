"""Rung 4, the gate: is the answer even inside the top 3?

A cross-encoder can only re-rank candidates it is given. If the linear model's
mistakes are usually a wrong label that is *not* in its top 3, no re-ranker at
any quality recovers them, and the rung is refuted before a model is downloaded.

This measures top-k recall of the truth on the grouped folds, for the two linear
rungs that matter, and writes the confusion pairs so the "confusable pair" claim
can be checked rather than assumed.

    .venv/bin/python experiments/ladder/top3_diagnostic.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from localalexa.ladder.pools import fold_pool
from localalexa.ladder.predictors import top_k
from localalexa.ladder.rungs import rung_0_baseline, rung_2_logreg_c100
from localalexa.ladder.run import load_bench


def diagnose(rung, bench, ks=(1, 2, 3, 5)) -> dict[str, object]:
    hits = {k: 0 for k in ks}
    pairs: dict[str, int] = {}
    rows = []
    for indices in bench.folds.indices(list(bench.families)):
        held = {bench.families[i] for i in indices}
        pool = fold_pool(bench.registry, held)
        model = rung.build().fit([t for t, _ in pool], [l for _, l in pool])
        texts = [bench.rows[i].text for i in indices]
        truth = [bench.rows[i].tool for i in indices]
        ranked = top_k(model, texts, k=max(ks))
        for text, want, order in zip(texts, truth, ranked):
            for k in ks:
                if want in order[:k]:
                    hits[k] += 1
            if order[0] != want:
                rows.append({"text": text, "truth": want, "top": order, "in_top3": want in order[:3]})
                key = f"{want} -> {order[0]}"
                pairs[key] = pairs.get(key, 0) + 1
    total = len(bench.rows)
    return {
        "rung": rung.name,
        "n_rows": total,
        "top_k_recall": {str(k): round(hits[k] / total, 4) for k in ks},
        "errors": len(rows),
        "errors_in_top3": sum(1 for r in rows if r["in_top3"]),
        "errors_outside_top3": sum(1 for r in rows if not r["in_top3"]),
        "top_confusions": dict(sorted(pairs.items(), key=lambda kv: -kv[1])[:12]),
        "error_rows": rows,
        "one_minus_recall_at_1": round(1 - hits[1] / total, 4),
        "one_minus_recall_at_3": round(1 - hits[3] / total, 4),
        "ceiling_gain_points_if_reranker_perfect": round(
            100.0 * (hits[3] - hits[1]) / total, 2
        ),
    }


def main() -> int:
    bench = load_bench(5)
    out = []
    for rung in (rung_0_baseline(), rung_2_logreg_c100()):
        block = diagnose(rung, bench)
        out.append(block)
        print(
            f"{block['rung']:22s} top1={block['top_k_recall']['1']:.3f} "
            f"top2={block['top_k_recall']['2']:.3f} top3={block['top_k_recall']['3']:.3f} "
            f"top5={block['top_k_recall']['5']:.3f} | errors={block['errors']} "
            f"in_top3={block['errors_in_top3']} outside={block['errors_outside_top3']} "
            f"| perfect reranker ceiling={block['ceiling_gain_points_if_reranker_perfect']:+.2f} pts"
        )
        for pair, n in list(block["top_confusions"].items())[:6]:
            print(f"    {n:3d}  {pair}")
    target = Path(__file__).resolve().parents[2] / "results" / "ladder" / "top3_diagnostic.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
