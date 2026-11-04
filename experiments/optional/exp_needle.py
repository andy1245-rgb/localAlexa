#!/usr/bin/env python
"""Experiment: needle3 (Cactus Compute "cactus" tier, tiny on-device tool-calling
model) as a DIRECT tool-choice classifier on the localAlexa router test set.

Same 382-utterance test set as the embedding-router baseline (exp_router.py /
router_lib.py).  Protocol:
  1. one needle tool per registry tool (17) + a no_action opt-out tool
  2. warm up the model with one agent.run() first
  3. one Needle.complete() per utterance (API has no batching; native
     needle_complete takes a single text).  agent.reset() before every call:
     the engine keeps conversation state across completions, which we verified
     leaks context (a chitchat utterance after timer utterances produced a
     timer call).  reset() costs ~1 ms.
  4. predicted tool = name of first call object in response["function_calls"],
     "NO_CALL" if the envelope carries no call objects.  (This needle version's
     run() returns executed *results* and runs a second completion round after
     execution, so complete() is the clean single-decision classifier surface.)
  5. "__none__" rows count as correct iff prediction is no_action or NO_CALL.

Also computes, in-memory and read-only, the MiniLM embedding-router baseline
(router_lib.predict_batch) on the same rows for the viability comparison.

Writes results/needle_results.json.  ASCII-only console output, f-strings only.
"""
import os
os.environ["NEEDLE_TELEMETRY"] = "0"  # no telemetry network calls during bench

import json
import time
from collections import Counter
from pathlib import Path

import needle

HERE = Path(__file__).resolve().parent
REGISTRY = json.loads((HERE / "registry.json").read_text(encoding="utf-8"))
ROWS = json.loads((HERE / "data" / "test.json").read_text(encoding="utf-8"))
RESULTS_DIR = HERE / "results"

NONE_ID = "__none__"
WARMUP_TEXT = "set a timer for 3 minutes"
SUBSET_WALL_LIMIT_S = 1200.0  # if projected eval wall time exceeds 20 min -> subset
CONFUSABLE_PAIRS = [
    ("timer_create", "reminder_create"),
    ("timer_cancel", "reminder_cancel"),
    ("thermostat_set", "thermostat_offset"),
    ("light_set", "light_query"),
    ("research_deep", "docs_query"),
]


# ---------------------------------------------------------------- tools -----
# One needle tool per registry id; description is one line derived from the
# registry references/purpose.  Params are simple strs; params do not matter
# for scoring, the model only needs to PICK the tool.
@needle.tool
def timer_create(duration: str):
    "Start a countdown timer for a duration."
    return {"ok": True}


@needle.tool
def timer_cancel():
    "Cancel or stop a running timer or countdown."
    return {"ok": True}


@needle.tool
def timer_status():
    "Check how much time is left on the running timer."
    return {"ok": True}


@needle.tool
def reminder_create(task: str, when: str):
    "Create a reminder to do a task at a specific time."
    return {"ok": True}


@needle.tool
def reminder_cancel(task: str):
    "Cancel or delete an existing reminder."
    return {"ok": True}


@needle.tool
def shoppinglist_add(item: str):
    "Add an item to the shopping list."
    return {"ok": True}


@needle.tool
def shoppinglist_remove(item: str):
    "Remove an item from the shopping list."
    return {"ok": True}


@needle.tool
def shoppinglist_view():
    "Read out what is on the shopping list."
    return {"ok": True}


@needle.tool
def thermostat_set(temp: str):
    "Set the thermostat to a specific temperature in degrees."
    return {"ok": True}


@needle.tool
def thermostat_offset(direction: str):
    "Make the temperature relatively warmer or colder."
    return {"ok": True}


@needle.tool
def light_set(room: str, state: str):
    "Turn lights or lamps on or off, or change their brightness."
    return {"ok": True}


@needle.tool
def light_query(room: str):
    "Check whether lights or lamps are on or off."
    return {"ok": True}


@needle.tool
def docs_query(topic: str):
    "Search the documentation for how something works."
    return {"ok": True}


@needle.tool
def projects_query(project: str):
    "Report the status of my projects and repos."
    return {"ok": True}


@needle.tool
def email_unread():
    "Check unread emails and the inbox."
    return {"ok": True}


@needle.tool
def calendar_today(day: str):
    "Show calendar events for a day or span."
    return {"ok": True}


@needle.tool
def research_deep(topic: str):
    "Research a topic in depth and compare the options."
    return {"ok": True}


@needle.tool
def no_action():
    "Call this when the user's request does not match any other tool."
    return {"ok": True}


ALL_TOOLS = [timer_create, timer_cancel, timer_status, reminder_create,
             reminder_cancel, shoppinglist_add, shoppinglist_remove,
             shoppinglist_view, thermostat_set, thermostat_offset, light_set,
             light_query, docs_query, projects_query, email_unread,
             calendar_today, research_deep, no_action]
VALID_IDS = {t["id"] for t in REGISTRY["tools"]}
assert {fn.__name__ for fn in ALL_TOOLS} == VALID_IDS | {"no_action"}


# --------------------------------------------------------------- helpers ----
def percentile(vals, p):
    """Nearest-rank percentile of a list of floats."""
    if not vals:
        return float("nan")
    s = sorted(vals)
    idx = min(max(int(round(p / 100.0 * len(s) + 0.5)) - 1, 0), len(s) - 1)
    return s[idx]


def is_none_pred(pred):
    return pred in ("no_action", "NO_CALL")


def row_correct(true, pred):
    if true == NONE_ID:
        return is_none_pred(pred)
    return pred == true


# ------------------------------------------------------------------ main ----
def main():
    try:
        from needle.agent.fetch import ENGINE_VERSIONS
        engine_ver = ENGINE_VERSIONS[3]
    except Exception:
        engine_ver = "3.0.1"

    print("== 0. SETUP ==")
    print(f"needle pkg {needle.__version__} (engine needle3 v{engine_ver}) | "
          f"{len(ROWS)} test rows | {len(REGISTRY['tools'])} tools + no_action")

    agent = needle.Needle(tools=ALL_TOOLS)

    # ---- warmup: one full run() first (also warms the model) ----
    t0 = time.perf_counter()
    warm = agent.run(WARMUP_TEXT)
    warm_ms = (time.perf_counter() - t0) * 1000.0
    wcalls = [c.get("name") for c in (warm.get("function_calls") or [])]
    print(f"warmup run(): {warm_ms:.0f} ms  type={warm.get('type')}  "
          f"calls={wcalls}  results={warm.get('results')}")
    print("note: run() executes tools then runs a SECOND completion round; "
          "this bench uses single-shot complete() per row instead")

    # ---- calibration: 3 fresh single-shot completes -> runtime estimate ----
    calib = []
    for t in ("set a timer for 9 minutes", "what's the population of Japan",
              "add oatmeal to the shopping list"):
        agent.reset()
        t0 = time.perf_counter()
        agent.complete(t)
        calib.append((time.perf_counter() - t0) * 1000.0)
    est_ms = sum(calib) / len(calib)
    est_total_s = len(ROWS) * est_ms / 1000.0
    print(f"calibration completes: {[f'{x:.0f}ms' for x in calib]} -> "
          f"est {est_total_s:.0f}s for {len(ROWS)} rows")

    subset_mode = est_total_s > SUBSET_WALL_LIMIT_S
    if subset_mode:
        keep, seen = [], Counter()
        for r in ROWS:
            if r["tool"] == NONE_ID:
                keep.append(r)
            elif seen[r["tool"]] < 10:
                keep.append(r)
                seen[r["tool"]] += 1
        eval_rows = keep
        print(f"PROJECTED > {SUBSET_WALL_LIMIT_S:.0f}s -> STRATIFIED SUBSET: "
              f"{len(eval_rows)} rows (10 per tool + all none rows)")
    else:
        eval_rows = ROWS
        print(f"under limit -> FULL test set, {len(eval_rows)} rows "
              f"(API has no batching; one completion per utterance)")

    # ---- main eval loop ----
    records = []
    t_start = time.perf_counter()
    for i, r in enumerate(eval_rows):
        agent.reset()  # engine keeps cross-completion state; reset = ~1 ms
        t0 = time.perf_counter()
        try:
            resp = agent.complete(r["text"])
            dt_ms = (time.perf_counter() - t0) * 1000.0
            calls = resp.get("function_calls") or []
            sup = resp.get("suppressed_calls") or []
            pred = str(calls[0].get("name")) if calls else "NO_CALL"
            conf = resp.get("confidence")
        except Exception as exc:  # keep going; count as error row
            dt_ms = (time.perf_counter() - t0) * 1000.0
            pred, conf, sup = "ERROR", None, []
            print(f"  row {i} ERROR: {exc}")
        records.append({
            "i": i, "text": r["text"], "true": r["tool"], "family": r["family"],
            "pred": pred, "correct": row_correct(r["tool"], pred),
            "latency_ms": round(dt_ms, 2), "confidence": conf,
            "n_suppressed": len(sup),
        })
        if (i + 1) % 50 == 0 or i + 1 == len(eval_rows):
            done = records
            acc = sum(1 for x in done if x["correct"]) / len(done)
            lat = sum(x["latency_ms"] for x in done) / len(done)
            print(f"  [{i + 1:3d}/{len(eval_rows)}]  running acc={acc:.1%}  "
                  f"mean latency={lat:.0f} ms")
    wall_s = time.perf_counter() - t_start

    # ---- metrics ----
    n = len(records)
    n_corr = sum(1 for x in records if x["correct"])
    acc = n_corr / n
    tool_rows = [x for x in records if x["true"] != NONE_ID]
    none_rows = [x for x in records if x["true"] == NONE_ID]
    acc_tool = (sum(1 for x in tool_rows if x["correct"]) / len(tool_rows)
                if tool_rows else None)
    none_hit = sum(1 for x in none_rows if x["correct"])
    none_acc = none_hit / len(none_rows) if none_rows else None
    none_breakdown = Counter(x["pred"] for x in none_rows)
    false_none = sum(1 for x in tool_rows if is_none_pred(x["pred"]))
    false_none_rate = false_none / len(tool_rows) if tool_rows else None
    lat = [x["latency_ms"] for x in records]
    p50, p95 = percentile(lat, 50), percentile(lat, 95)
    mean_lat = sum(lat) / len(lat)

    fams = sorted({x["family"] for x in records})
    fam_stats = {}
    for f in fams:
        rows_f = [x for x in records if x["family"] == f]
        fam_stats[f] = {
            "n": len(rows_f),
            "n_correct": sum(1 for x in rows_f if x["correct"]),
            "acc": sum(1 for x in rows_f if x["correct"]) / len(rows_f)}

    per_tool = {}
    for t in sorted({x["true"] for x in records}):
        rows_t = [x for x in records if x["true"] == t]
        per_tool[t] = {
            "n": len(rows_t),
            "n_correct": sum(1 for x in rows_t if x["correct"]),
            "acc": sum(1 for x in rows_t if x["correct"]) / len(rows_t)}

    errors = [x for x in records if not x["correct"]]
    conf_pairs = Counter((x["true"], x["pred"]) for x in errors)
    top10 = conf_pairs.most_common(10)

    pair_reports = []
    for a, b in CONFUSABLE_PAIRS:
        ra = [x for x in records if x["true"] == a]
        rb = [x for x in records if x["true"] == b]
        if not ra or not rb:
            continue
        acc_a = sum(1 for x in ra if x["correct"]) / len(ra)
        acc_b = sum(1 for x in rb if x["correct"]) / len(rb)
        a_as_b = sum(1 for x in ra if x["pred"] == b)
        b_as_a = sum(1 for x in rb if x["pred"] == a)
        both = ra + rb
        acc_both = sum(1 for x in both if x["correct"]) / len(both)
        pair_reports.append({"pair": [a, b], "n_a": len(ra), "acc_a": acc_a,
                             "n_b": len(rb), "acc_b": acc_b,
                             "a_predicted_as_b": a_as_b,
                             "b_predicted_as_a": b_as_a,
                             "combined_acc": acc_both})

    failures = errors[:10]

    # ---- embedding-router baseline, in-memory, read-only ----
    router_res = None
    print("\n== baseline: MiniLM embedding router (in-memory, same rows) ==")
    try:
        import router_lib
        texts = [r["text"] for r in eval_rows]
        t0 = time.perf_counter()
        rp = router_lib.predict_batch(texts, tau=0.08, low_sim_floor=0.35)
        r_ms_row = (time.perf_counter() - t0) * 1000.0 / len(texts)
        r_preds = [p["top_tool"] for p in rp]
        r_corr = [rp_["top_tool"] == truth for rp_, truth in zip(rp, [r["tool"] for r in eval_rows])]
        r_none = [r_["tool"] for r_ in eval_rows]
        r_none_hit = sum(1 for c, t in zip(r_corr, r_none) if t == NONE_ID and c)
        r_none_n = sum(1 for t in r_none if t == NONE_ID)
        r_tool_n = len(r_none) - r_none_n
        r_false_none = sum(1 for p, t in zip(r_preds, r_none)
                           if t != NONE_ID and p == NONE_ID)
        router_res = {
            "model": router_lib.MODEL_NAME,
            "accuracy": sum(r_corr) / len(r_corr),
            "none_accuracy": (r_none_hit / r_none_n) if r_none_n else None,
            "false_none_rate": (r_false_none / r_tool_n) if r_tool_n else None,
            "batch_ms_per_row": round(r_ms_row, 2),
            "n_rows": len(texts),
        }
        print(f"  router: acc={router_res['accuracy']:.1%}  "
              f"none-detect={router_res['none_accuracy']:.1%}  "
              f"false-none={router_res['false_none_rate']:.1%}  "
              f"batch={router_res['batch_ms_per_row']} ms/row")
    except Exception as exc:
        print(f"  embedding-router baseline unavailable: {exc}")

    classic = None
    cpath = RESULTS_DIR / "classic_results.json"
    if cpath.exists():
        try:
            cl = json.loads(cpath.read_text(encoding="utf-8"))
            best = max(cl, key=lambda m: m["accuracy"])
            classic = {"model": best["model"], "accuracy": best["accuracy"],
                       "none_detection": best["none_detection"],
                       "latency_p50_ms": best["latency_p50_ms"]}
            print(f"  classic best (from classic_results.json): "
                  f"{classic['model']} acc={classic['accuracy']:.1%} "
                  f"none={classic['none_detection']:.0%} "
                  f"p50={classic['latency_p50_ms']} ms")
        except Exception as exc:
            print(f"  could not read classic_results.json: {exc}")

    # ---- console report ----
    mode = "STRATIFIED SUBSET" if subset_mode else "FULL SET"
    print(f"\n== 1. ACCURACY ({mode}, n={n}) ==")
    print(f"  overall:               {n_corr}/{n} = {acc:.1%}")
    print(f"  excl. none-class:      "
          f"{sum(1 for x in tool_rows if x['correct'])}/{len(tool_rows)} = "
          f"{acc_tool:.1%}")
    print(f"  none detection:        {none_hit}/{len(none_rows)} = {none_acc:.1%} "
          f"({dict(none_breakdown)})")
    print(f"  false-none rate:       {false_none}/{len(tool_rows)} = "
          f"{false_none_rate:.1%}")

    print("\n== 2. PER-FAMILY ACCURACY ==")
    for f in fams:
        s = fam_stats[f]
        print(f"  {f:12s} n={s['n']:3d}  {s['n_correct']:3d} correct  "
              f"acc={s['acc']:6.1%}")

    print("\n== 3. TOP-10 CONFUSION PAIRS (true -> predicted) ==")
    if top10:
        for (tt, pp), c in top10:
            ex = next((x["text"] for x in errors
                       if x["true"] == tt and x["pred"] == pp), "")[:44]
            print(f"  {tt:18s} -> {pp:18s} n={c:2d}   e.g. \"{ex}\"")
    else:
        print("  (none)")

    print("\n== 4. CONFUSABLE PAIRS ==")
    for pr in pair_reports:
        a, b = pr["pair"]
        print(f"  {a:18s} acc={pr['acc_a']:6.1%} ({pr['n_a']:2d}) | "
              f"{b:18s} acc={pr['acc_b']:6.1%} ({pr['n_b']:2d}) | "
              f"cross: {a}->{b}:{pr['a_predicted_as_b']} "
              f"{b}->{a}:{pr['b_predicted_as_a']} | "
              f"combined={pr['combined_acc']:.1%}")

    worst5 = sorted(per_tool.items(), key=lambda kv: (kv[1]["acc"], -kv[1]["n"]))[:5]
    print("\n== 5. WORST 5 TOOLS ==")
    for t, s in worst5:
        print(f"  {t:20s} {s['n_correct']:2d}/{s['n']:2d} = {s['acc']:6.1%}")

    print("\n== 6. LATENCY (single-shot complete(), warm model) ==")
    print(f"  p50={p50:.1f} ms   p95={p95:.1f} ms   mean={mean_lat:.1f} ms")
    print(f"  total eval wall time: {wall_s:.1f} s for {n} rows "
          f"({1000.0 * wall_s / n:.0f} ms/row incl. reset + scoring)")

    print("\n== 7. FAILURE EXAMPLES (first 10) ==")
    for x in failures:
        print(f"  \"{x['text'][:58]}\"")
        print(f"      pred={x['pred']}  true={x['true']}  "
              f"conf={x['confidence'] if x['confidence'] is not None else 'n/a'}")

    # ---- FINDINGS ----
    r_acc = router_res["accuracy"] if router_res else None
    best_classic = classic["accuracy"] if classic else None
    top_pairs_txt = "; ".join(f"{tt} -> {pp} (x{c})" for (tt, pp), c in top10[:3])
    none_chitchat = none_breakdown.get("NO_CALL", 0) + none_breakdown.get("no_action", 0)
    misrouted = len(none_rows) - none_chitchat
    ratio = (p50 / max(router_res["batch_ms_per_row"], 0.01)
             if router_res else 0.0)
    viable = (r_acc is not None and acc > r_acc and none_acc is not None
              and none_acc >= 0.6)
    print("\n== FINDINGS ==")
    print(f"* Accuracy verdict: needle3 zero-shot hits {acc:.1%} overall "
          f"({acc_tool:.1%} excl. none, {none_acc:.1%} none-detect, "
          f"{false_none_rate:.1%} false-none)"
          + (f" vs embedding-router {r_acc:.1%} on the same rows"
             if r_acc is not None else "")
          + (f" and best classic baseline {best_classic:.1%}" if best_classic else "")
          + ".")
    print(f"* Latency verdict: p50={p50:.0f} ms / p95={p95:.0f} ms per decision "
          f"on CPU"
          + (f" vs router {router_res['batch_ms_per_row']} ms/row batched "
             f"(~{ratio:.0f}x slower)"
             if router_res else "")
          + f"; full {n}-row eval {wall_s:.0f} s - usable for voice, not free.")
    if r_acc is not None:
        if acc > r_acc:
            detail = (f"it beats the MiniLM reference router on accuracy "
                      f"({acc:.1%} vs {r_acc:.1%}) with no fine-tuning, but "
                      f"costs ~{ratio:.0f}x its latency")
        else:
            detail = (f"it trails the MiniLM reference router by "
                      f"{100 * (r_acc - acc):.1f} pts ({acc:.1%} vs {r_acc:.1%}) "
                      f"at ~{ratio:.0f}x the latency, zero-shot")
    else:
        detail = "no router comparison available"
    print(f"* Cactus-tier viability: needle3 "
          f"{'IS viable' if viable else 'is NOT viable'} as the learned tier "
          f"zero-shot: {detail}.")
    if top10:
        print(f"* Biggest failure mode: sibling-tool confusion - "
              f"{top_pairs_txt}; none-rows misrouted to real tools "
              f"{misrouted}/{len(none_rows)}, worst tool {worst5[0][0]} at "
              f"{worst5[0][1]['acc']:.0%}.")
    else:
        print("* Biggest failure mode: none (zero errors).")
    print(f"* Recommendation: "
          + (f"keep the embedding router as the fast default tier and promote "
             f"needle3 to a confirmation/escalation tier on low-confidence or "
             f"confusable-pair utterances ({p50:.0f} ms p50 is acceptable for "
             f"voice), then fine-tune it on this registry to close the "
             f"none/false-none gap." if viable else
             f"stay with the embedding router as the learned tier; needle3 "
             f"zero-shot does not justify the latency here - fine-tune or use "
             f"it only as an escalation tier."))

    # ---- write JSON ----
    RESULTS_DIR.mkdir(exist_ok=True)
    out = {
        "config": {
            "needle_package": needle.__version__,
            "engine": f"needle3 v{engine_ver}",
            "api_surface": "Needle.complete() single-shot decision; "
                           "agent.reset() before every call (engine keeps "
                           "cross-completion state)",
            "warmup": "one agent.run() call",
            "prediction_rule": "function_calls[0].name else NO_CALL; "
                               "__none__ correct iff no_action or NO_CALL",
            "batching": "not supported by API - one completion per utterance",
            "determinism": "greedy decoding: repeated identical calls with "
                           "reset() in between return identical predictions "
                           "(verified pre-run)",
            "subset_mode": subset_mode,
            "n_rows": n,
            "n_tools": len(REGISTRY["tools"]) + 1,
            "tool_descriptions": {fn.__name__: (fn.__doc__ or "").strip()
                                  for fn in ALL_TOOLS},
        },
        "accuracy": {
            "overall": acc, "n_correct": n_corr, "n": n,
            "excl_none": acc_tool, "n_excl_none": len(tool_rows),
            "none_accuracy": none_acc, "none_n": len(none_rows),
            "none_pred_breakdown": dict(none_breakdown),
            "false_none_count": false_none, "false_none_rate": false_none_rate,
            "per_family": fam_stats, "per_tool": per_tool,
        },
        "confusion": {
            "n_errors": len(errors),
            "top10_pairs": [{"true": tt, "pred": pp, "count": c}
                            for (tt, pp), c in top10],
            "confusable_pairs": pair_reports,
        },
        "latency": {
            "p50_ms": round(p50, 2), "p95_ms": round(p95, 2),
            "mean_ms": round(mean_lat, 2),
            "total_eval_wall_s": round(wall_s, 2),
            "ms_per_row_incl_overhead": round(1000.0 * wall_s / n, 1),
        },
        "failures": failures,
        "per_row": records,
        "embedding_router_baseline": router_res,
        "classic_baseline_best": classic,
    }
    jpath = RESULTS_DIR / "needle_results.json"
    jpath.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {jpath}")


if __name__ == "__main__":
    main()
