"""Experiment: does a Cactus hybrid checkpoint's confidence predict its own mistakes?

The gate is not accuracy and not the model card's benchmarks. It is the AUROC of the
checkpoint's `confidence` against whether the prediction was right, on the same 382 rows
the rest of this repo scores. Base needle3 scored **0.523** (confidence barely separates
right from wrong, so it cannot gate a handoff). The bar here is **0.8**.

Protocol
--------
1. one prompt asking for exactly one of the 18 registry ids, greedy decoding, so the
   answer is a decision rather than a sample;
2. `generate(..., return_confidence=True)` from the checkpoint's remote code, giving
   `confidence = 1 - p_wrong` as structured data (never parsed out of the text);
3. predicted id = the first registry id appearing as a token in the output; anything
   else counts as NO_CALL, and `__none__` rows are correct when the answer is `__none__`
   or NO_CALL — the same rule the needle3 run used;
4. AUROC of confidence against correctness, reported over all rows and over the 17 tool
   rows only.

    .venv-hybrid/bin/python experiments/optional/exp_cactus_hybrid.py --limit 40
    .venv-hybrid/bin/python experiments/optional/exp_cactus_hybrid.py            # all 382

Writes ``results/cactus_hybrid_results.json``. Local only, $0, no API.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

HERE = Path(__file__).resolve().parents[2]
DATA = HERE / "data"
RESULTS = HERE / "results"

MODEL_ID = "Cactus-Compute/gemma-4-e2b-it-hybrid"
NONE_ID = "__none__"
AUROC_GATE = 0.8
BASE_NEEDLE3_AUROC = 0.523

PROMPT = """You are the intent classifier of an offline voice assistant. Choose exactly one \
intent id for the utterance. Reply with the id alone, nothing else.

Intent ids: {options}

Utterance: {text}
Intent id:"""


def load_rows(limit: int = 0, stride: int = 1) -> tuple[list[dict], tuple[str, ...]]:
    rows = json.loads((DATA / "routing_382.json").read_text(encoding="utf-8"))
    registry = json.loads((DATA / "registry.json").read_text(encoding="utf-8"))
    options = tuple(tool["id"] for tool in registry["tools"]) + (NONE_ID,)
    if stride > 1:
        # systematic sample across the file, which is ordered by intent block, so a stride
        # keeps every label represented instead of taking the first two intents only
        rows = rows[::stride]
    if limit:
        rows = rows[:limit]
    return rows, options


def parse_answer(text: str, options: tuple[str, ...]) -> str:
    """The first option id that appears as a token; NO_CALL otherwise."""
    lowered = text.strip().lower()
    best = None
    for option in options:
        match = re.search(rf"(?<![a-z0-9_]){re.escape(option.lower())}(?![a-z0-9_])", lowered)
        if match and (best is None or match.start() < best[0]):
            best = (match.start(), option)
    if best is not None:
        return best[1]
    if re.search(r"\b(no_call|none|no action)\b", lowered):
        return NONE_ID
    return "NO_CALL"


def auroc_ci(
    pairs: list[tuple[float, bool]], *, resamples: int = 2000, seed: int = 0
) -> tuple[float, float] | None:
    """Percentile interval for the AUROC, resampling rows."""
    import random

    if not pairs:
        return None
    rng = random.Random(seed)
    draws = []
    for _ in range(resamples):
        sample = [pairs[rng.randrange(len(pairs))] for _ in range(len(pairs))]
        value = auroc(sample)
        if value is not None:
            draws.append(value)
    if not draws:
        return None
    draws.sort()
    return (round(draws[int(0.025 * len(draws))], 3), round(draws[int(0.975 * len(draws))], 3))


def auroc(pairs: list[tuple[float, bool]]) -> float | None:
    """P(score of a correct row > score of an incorrect row), ties counted as half."""
    positives = [score for score, correct in pairs if correct]
    negatives = [score for score, correct in pairs if not correct]
    if not positives or not negatives:
        return None
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            wins += 1.0 if positive > negative else 0.5 if positive == negative else 0.0
    return wins / (len(positives) * len(negatives))


def _write(path: Path, per_row: list[dict], args, options, *, partial: bool = False) -> None:
    """Persist what has been scored so far, with the metrics it supports."""
    scored = [(row["confidence"], row["correct"]) for row in per_row if row["confidence"] is not None]
    tools_only = [(row["confidence"], row["correct"]) for row in per_row
                  if row["confidence"] is not None and row["true"] != NONE_ID]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "config": {
                    "model": MODEL_ID,
                    "limit": args.limit,
                    "stride": args.stride,
                    "max_new_tokens": args.max_new_tokens,
                    "partial": partial,
                },
                "n_rows": len(per_row),
                "n_scored": len(scored),
                "n_errors": sum(1 for row in per_row if not row["correct"]),
                "accuracy": round(sum(1 for row in per_row if row["correct"]) / len(per_row), 4)
                if per_row else None,
                "auroc_all_rows": None if auroc(scored) is None else round(auroc(scored), 4),
                "auroc_tool_rows": None if auroc(tools_only) is None else round(auroc(tools_only), 4),
                "auroc_ci95": None if auroc_ci(tools_only) is None else list(auroc_ci(tools_only)),
                "per_row": per_row,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def verdict(auroc_all: float | None, auroc_tools: float | None, n_rows: int,
            n_errors: int) -> dict:
    """The gate, stated so that the number a user would see leads and nothing hides behind it.

    Both figures must clear, and the **abstain-inclusive** one is the one quoted first. A
    confidence that separates right from wrong only when the abstain class is excluded is a
    confidence that fails on the rows where a voice assistant most needs it: deciding whether
    to act at all. The 17-label figure is reported beside it, never instead of it.
    """
    if auroc_all is None or auroc_tools is None:
        return {
            "clears": False,
            "evaluable": False,
            "headline": "abstain-inclusive AUROC",
            "verdict": (
                f"AUROC undefined: {n_errors} wrong out of {n_rows} rows. With too few errors "
                f"the gate cannot be evaluated either way — neither a pass nor a refutation"
            ),
        }
    clears = auroc_all >= AUROC_GATE and auroc_tools >= AUROC_GATE
    failing = []
    if auroc_all < AUROC_GATE:
        failing.append(f"abstain-inclusive {auroc_all:.3f}")
    if auroc_tools < AUROC_GATE:
        failing.append(f"17-label {auroc_tools:.3f}")
    return {
        "clears": bool(clears),
        "evaluable": True,
        "headline": "abstain-inclusive AUROC",
        "verdict": (
            f"AUROC {auroc_all:.3f} over all rows (17 labels alone: {auroc_tools:.3f}) — both "
            f"clear {AUROC_GATE}, so the confidence can gate a handoff"
            if clears
            else f"AUROC {auroc_all:.3f} over all rows (17 labels alone: {auroc_tools:.3f}) — "
            f"below {AUROC_GATE} on {', '.join(failing)}; it cannot gate a handoff"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, default=0, help="score at most N rows")
    parser.add_argument("--stride", type=int, default=1,
                        help="systematic sample: take every Nth row (the file is ordered by "
                             "intent block, so stride keeps every label represented)")
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--out", default=str(RESULTS / "cactus_hybrid_results.json"))
    parser.add_argument("--recompute", default="",
                        help="rebuild the metrics and the verdict from a saved per_row list, "
                             "without loading the model")
    args = parser.parse_args(argv)

    if args.recompute:
        saved = json.loads(Path(args.recompute).read_text(encoding="utf-8"))
        per_row = saved["per_row"]
        tool_pairs = [(row["confidence"], row["correct"]) for row in per_row
                      if row["confidence"] is not None and row["true"] != NONE_ID]
        all_pairs = [(row["confidence"], row["correct"]) for row in per_row
                     if row["confidence"] is not None]
        errors = sum(1 for row in per_row if not row["correct"])
        tool_auroc = auroc(tool_pairs)
        payload = dict(saved)
        payload.update({
            "n_rows": len(per_row),
            "n_errors": errors,
            "accuracy": round(sum(1 for row in per_row if row["correct"]) / len(per_row), 4),
            "auroc_all_rows": None if auroc(all_pairs) is None else round(auroc(all_pairs), 4),
            "auroc_tool_rows": None if tool_auroc is None else round(tool_auroc, 4),
            "auroc_ci95": None if auroc_ci(tool_pairs) is None else list(auroc_ci(tool_pairs)),
            "auroc_all_ci95": None if auroc_ci(all_pairs) is None else list(auroc_ci(all_pairs)),
            "gate": verdict(
                None if auroc(all_pairs) is None else round(auroc(all_pairs), 4),
                None if tool_auroc is None else round(tool_auroc, 4),
                len(per_row),
                errors,
            ),
        })
        Path(args.out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({key: payload[key] for key in (
            "n_rows", "n_errors", "accuracy", "auroc_tool_rows", "auroc_ci95",
            "auroc_all_rows", "auroc_all_ci95", "gate")}, indent=2))
        print(f"wrote {args.out}")
        return 0

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows, options = load_rows(args.limit, args.stride)
    option_list = ", ".join(options)
    print(f"rows {len(rows)}  options {len(options)}  model {MODEL_ID}")

    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True, dtype="auto"
    ).eval()
    print(f"loaded in {time.perf_counter() - started:.1f}s  "
          f"dtype {next(model.parameters()).dtype}")

    out_path = Path(args.out)
    per_row = []
    for index, row in enumerate(rows):
        prompt = PROMPT.format(options=option_list, text=row["text"])
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
        began = time.perf_counter()
        try:
            out = model.generate(
                **inputs,
                return_confidence=True,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
            )
            generated = out.sequences[0][inputs["input_ids"].shape[-1]:]
            answer = tokenizer.decode(generated, skip_special_tokens=True)
            confidence = out.confidence
        except Exception as exc:  # a probe failure is a row-level fact, not a crash
            answer, confidence = "", None
            print(f"  row {index}: generate failed: {type(exc).__name__}: {exc}")
        latency = (time.perf_counter() - began) * 1000.0
        predicted = parse_answer(answer, options)
        correct = predicted == row["tool"] or (
            row["tool"] == NONE_ID and predicted in (NONE_ID, "NO_CALL")
        )
        per_row.append(
            {
                "i": index,
                "text": row["text"],
                "true": row["tool"],
                "family": row.get("family"),
                "pred": predicted,
                "answer": " ".join(answer.split())[:80],
                "correct": bool(correct),
                "confidence": None if confidence is None else float(confidence),
                "latency_ms": round(latency, 1),
            }
        )
        if (index + 1) % 10 == 0:
            # write what exists so far: an hour of CPU should not be lost to a kill
            _write(out_path, per_row, args, options, partial=True)
            print(f"  [checkpoint] {index + 1} rows written")
        if (index + 1) % 10 == 0 or index + 1 == len(rows):
            done = index + 1
            print(f"  {done}/{len(rows)}  last: {predicted} (want {row['tool']}) "
                  f"conf {per_row[-1]['confidence']}  {latency:.0f} ms")

    scored = [(row["confidence"], row["correct"]) for row in per_row
              if row["confidence"] is not None]
    tools_only = [(row["confidence"], row["correct"]) for row in per_row
                  if row["confidence"] is not None and row["true"] != NONE_ID]
    payload = {
        "config": {
            "model": MODEL_ID,
            "rows": len(rows),
            "limit": args.limit,
            "stride": args.stride,
            "max_new_tokens": args.max_new_tokens,
            "decoding": "greedy",
            "prompt": PROMPT,
            "confidence_api": "generate(return_confidence=True) -> out.confidence",
            "prediction_rule": "first registry id as a token; NO_CALL otherwise; "
            "__none__ correct on NO_CALL",
            "torch": torch.__version__,
        },
        "n_rows": len(per_row),
        "n_scored": len(scored),
        "accuracy": round(sum(1 for row in per_row if row["correct"]) / len(per_row), 4)
        if per_row else None,
        "auroc_all_rows": None if auroc(scored) is None else round(auroc(scored), 4),
        "auroc_tool_rows": None if auroc(tools_only) is None else round(auroc(tools_only), 4),
        "auroc_ci95": None if auroc_ci(tools_only) is None else list(auroc_ci(tools_only)),
        "labels_covered": sorted({row["true"] for row in per_row}),
        "auroc_gate": AUROC_GATE,
        "base_needle3_auroc": BASE_NEEDLE3_AUROC,
        "latency_ms_p50": sorted(row["latency_ms"] for row in per_row)[len(per_row) // 2]
        if per_row else None,
        "per_row": per_row,
    }
    payload["gate"] = verdict(
        payload["auroc_all_rows"],
        payload["auroc_tool_rows"],
        payload["n_rows"],
        sum(1 for row in per_row if not row["correct"]),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\naccuracy {payload['accuracy']}")
    print(f"AUROC abstain-inclusive {payload['auroc_all_rows']} "
          f"(17 labels alone {payload['auroc_tool_rows']})")
    print(f"gate {AUROC_GATE}: {payload['gate']['verdict']}")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
