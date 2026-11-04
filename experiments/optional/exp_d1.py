"""Exp F — LiquidAI/d1-3B as a direct tool-choice decision model.

d1-3B is a 3.1B multimodal *decision* model (LFM2.5-VL-3B base) that answers
noul/choice/score questions off its distribution with zero generated tokens.
llama.cpp (>= b11500) serves it natively at POST /v1/systemone.

This probes it the same way exp_classic/exp_router/exp_needle/exp_qwen probe the
other tiers: one choice question per utterance over the 17 tools + __none__,
scored against data/test.json (382 rows). Writes results/d1_results.json.

Usage:
  python exp_d1.py [base_url]            # default http://127.0.0.1:8080
  python exp_d1.py http://127.0.0.1:8080 --limit 60
"""
import json
import statistics
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REG = json.loads((HERE / "registry.json").read_text(encoding="utf-8"))
TEST = json.loads((HERE / "data" / "test.json").read_text(encoding="utf-8"))
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)

BASE = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].startswith("http") \
    else "http://127.0.0.1:8080"
LIMIT = None
if "--limit" in sys.argv:
    LIMIT = int(sys.argv[sys.argv.index("--limit") + 1])

# Short, human-readable option descriptions (the llama.cpp blog notes bare labels
# hurt routing quality; descriptions roughly doubled Julia-1's accuracy).
TOOL_DESC = {
    "timer_create": "start a countdown timer",
    "timer_cancel": "cancel or stop a running timer",
    "timer_status": "check how much time is left on a timer",
    "reminder_create": "create a reminder for a date or time",
    "reminder_cancel": "cancel or delete a reminder",
    "shoppinglist_add": "add an item to the shopping or grocery list",
    "shoppinglist_remove": "remove an item from the shopping list",
    "shoppinglist_view": "read out the shopping list",
    "thermostat_set": "set the thermostat to a specific temperature",
    "thermostat_offset": "make it warmer or colder by a relative amount",
    "light_set": "turn lights on or off, or dim them",
    "light_query": "ask whether lights are currently on or off",
    "docs_query": "search the internal documentation",
    "projects_query": "ask about project status or summaries",
    "email_unread": "check unread email or the inbox",
    "calendar_today": "ask about the calendar or schedule",
    "research_deep": "do deep web research on a topic",
    "__none__": "no tool fits: small talk or a general knowledge question",
}
CRITERIA = {t["id"]: TOOL_DESC.get(t["id"], t["id"]) for t in REG["tools"]}
CRITERIA["__none__"] = TOOL_DESC["__none__"]
VALID = set(CRITERIA)

QUESTION = {
    "type": "choice",
    "instructions": "Which single tool should handle the user command?",
    "criteria": CRITERIA,
}


def systemone(state, timeout=120):
    body = json.dumps({"state": state, "questions": {"route": QUESTION}}).encode()
    req = urllib.request.Request(
        BASE + "/v1/systemone", body, {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        payload = json.loads(r.read())
    return (time.perf_counter() - t0) * 1000, payload


def pct(ms, q):
    if not ms:
        return 0.0
    return round(sorted(ms)[min(len(ms) - 1, int(len(ms) * q) - 1)], 1)


def main():
    rows = TEST if LIMIT is None else TEST[:LIMIT]
    N = len(rows)
    print(f"=== d1-3B tool-choice vs {BASE} (n={N}) ===", flush=True)

    # warmup / shape check
    try:
        dt, payload = systemone(rows[0]["text"])
        a = payload["answers"]["route"]
        print(f"warmup ok: {dt:.0f}ms choice={a.get('choice')} "
              f"conf={a.get('confidence')} usage={payload.get('usage')}", flush=True)
    except Exception as e:
        print(f"WARMUP FAILED: {type(e).__name__}: {e}", flush=True)
        print("tip: llama-server -m models/d1-3B-Q4_K_M.gguf --mmproj "
              "models/mmproj-d1-3B-Q8_0.gguf --port 8080", flush=True)
        sys.exit(1)

    lat, preds, golds, confs, texts = [], [], [], [], []
    errors = 0
    t_start = time.time()
    for i, row in enumerate(rows):
        try:
            dt, payload = systemone(row["text"])
        except Exception as e:
            errors += 1
            preds.append("__ERROR__")
            golds.append(row["tool"])
            confs.append(0.0)
            texts.append(row["text"])
            continue
        a = payload["answers"]["route"]
        pred = a.get("choice")
        if pred not in VALID:
            pred = "INVALID:" + str(pred)[:20]
        lat.append(dt)
        preds.append(pred)
        golds.append(row["tool"])
        confs.append(float(a.get("confidence", 0.0)))
        texts.append(row["text"])
        if (i + 1) % 40 == 0:
            run = statistics.mean(p == g for p, g in zip(preds, golds))
            print(f"  {i+1}/{N}  running-acc {run:.3f}  "
                  f"mean-lat {statistics.mean(lat) if lat else 0:.0f}ms", flush=True)

    total_s = time.time() - t_start
    acc = statistics.mean(p == g for p, g in zip(preds, golds))

    per_family = {}
    tool_fam = {t["id"]: t["family"] for t in REG["tools"]}
    tool_fam["__none__"] = "none"
    for g, p in zip(golds, preds):
        per_family.setdefault(tool_fam.get(g, "?"), []).append(p == g)
    per_family = {k: round(statistics.mean(v), 3) for k, v in per_family.items()}

    per_tool = {}
    for g, p in zip(golds, preds):
        per_tool.setdefault(g, []).append(p == g)
    per_tool = {k: round(statistics.mean(v), 3) for k, v in per_tool.items()}

    none_mask = [g == "__none__" for g in golds]
    n_none = sum(none_mask)
    none_detect = (sum(p == "__none__" for p, m in zip(preds, none_mask) if m) / n_none
                   if n_none else None)
    real_n = N - n_none
    false_none = (sum(p == "__none__" for p, m in zip(preds, none_mask) if not m)
                  / real_n if real_n else None)

    confusion = Counter((p, g) for p, g in zip(preds, golds) if p != g)

    conf_arr = confs
    cut = {}
    for c in (0.5, 0.7, 0.9):
        idx = [i for i, cc in enumerate(conf_arr) if cc >= c]
        cut[str(c)] = {
            "coverage": round(len(idx) / N, 3),
            "acc": round(statistics.mean(preds[i] == golds[i] for i in idx), 3)
            if idx else None,
        }

    fails = [{"text": t, "true": g, "pred": p}
             for t, g, p in zip(texts, golds, preds) if g != p][:12]

    results = {
        "model": "LiquidAI/d1-3B-GGUF:Q4_K_M",
        "endpoint": BASE + "/v1/systemone",
        "n": N,
        "accuracy": round(acc, 4),
        "per_family_accuracy": per_family,
        "per_tool_accuracy": per_tool,
        "none_detection": round(none_detect, 3) if none_detect is not None else None,
        "false_none_rate": round(false_none, 3) if false_none is not None else None,
        "confidence_cutoffs": cut,
        "latency_p50_ms": pct(lat, 0.50),
        "latency_p95_ms": pct(lat, 0.95),
        "latency_mean_ms": round(statistics.mean(lat), 1) if lat else None,
        "total_wall_s": round(total_s, 1),
        "errors": errors,
        "top_confusion": [{"pred": p, "true": t, "n": c}
                          for (p, t), c in confusion.most_common(10)],
        "failures": fails,
    }
    tag = "d1_results" if LIMIT is None else f"d1_results_limit{LIMIT}"
    (OUT / f"{tag}.json").write_text(json.dumps(results, indent=1), encoding="utf-8")

    print(f"\n=== d1-3B zero-shot tool choice (n={N}, errors={errors}) ===")
    print(f"accuracy: {acc:.3f}   none-detect: {results['none_detection']}   "
          f"false-none: {results['false_none_rate']}")
    print(f"latency p50/p95/mean: {results['latency_p50_ms']}/"
          f"{results['latency_p95_ms']}/{results['latency_mean_ms']} ms   "
          f"total {total_s:.0f}s")
    print(f"per-family: {per_family}")
    print("confidence cutoffs:", json.dumps(cut))
    print("top confusions:", json.dumps(results["top_confusion"][:6]))
    for f in fails[:6]:
        print(f"  MISS true={f['true']:18s} pred={f['pred']:18s} <- {f['text'][:55]}")
    print(f"wrote {OUT / (tag + '.json')}")


if __name__ == "__main__":
    main()
