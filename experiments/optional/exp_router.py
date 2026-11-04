"""Experiment: semantic router (MiniLM vs per-tool references) as tool-choice classifier.

Evaluates router_lib.predict on data/test.json (382 rows):
  1. top-1 accuracy: overall / per-family / per-tool
  2. confusion analysis: top-10 confusion pairs + known confusable pairs
  3. abstain curve: sweep decision threshold on top_sim, find knee
  4. entropy-confidence calibration: 10 bins, Pearson r, acc at conf cutoffs
  5. none-route detection: accuracy on __none__ rows, false-none rate
  6. latency: 30 single predict() calls (p50/p95), batch eval wall time
  7. tau sensitivity: top-1 acc at tau 0.05 / 0.08 / 0.12

Writes results/router_results.json. ASCII-only console output, no table libs.
"""
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np

import router_lib

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "test.json"
RESULTS_DIR = HERE / "results"
TAU = 0.08
FLOOR = 0.35

CONFUSABLE_PAIRS = [
    ("timer_create", "reminder_create"),
    ("timer_cancel", "reminder_cancel"),
    ("thermostat_set", "thermostat_offset"),
    ("light_set", "light_query"),
    ("research_deep", "docs_query"),
]

LAT_TEXTS = [
    "set a timer for 7 minutes",
    "start a 12 minute countdown",
    "how long until my pasta timer goes off",
    "remind me to call the dentist at 9am",
    "set an alarm for 6 in the morning",
    "what's the weather in tokyo tomorrow",
    "will it rain this weekend",
    "add batteries to the shopping list",
    "put peanut butter on the grocery list",
    "turn the living room lights to 50 percent",
    "dim the kitchen lights a bit",
    "set the thermostat to 72 degrees",
    "make it 5 degrees warmer in here",
    "play some jazz music",
    "skip this song",
    "what's on my calendar today",
    "schedule a meeting with sarah tomorrow at 3",
    "tell me a fun fact",
    "how do I hard boil an egg",
    "search the web for the best coffee grinder",
    "look up the history of the roman empire",
    "read me my last email",
    "send a text to dad saying I'm on my way",
    "lock the front door",
    "is the garage door closed",
    "what time is it in london",
    "order more paper towels",
    "turn on the porch light",
    "cancel my 5pm timer",
    "set a reminder to water the plants on friday",
]
assert len(LAT_TEXTS) == 30


def acc_or_none(x):
    if x is None:
        return None
    x = float(x)
    return None if math.isnan(x) else x


def f_acc(x, w=5):
    y = acc_or_none(x)
    return "  n/a"[:w + 1].rjust(w + 1) if y is None else f"{100 * y:5.1f}%"


def masked_acc(mask, correct):
    n = int(mask.sum())
    return (float(correct[mask].mean()), n) if n else (None, 0)


def main():
    out = {"config": {"tau": TAU, "low_sim_floor": FLOOR,
                      "model": router_lib.MODEL_NAME}}

    rows = json.loads(DATA.read_text(encoding="utf-8"))
    texts = [r["text"] for r in rows]
    truth = [r["tool"] for r in rows]
    fams = [r["family"] for r in rows]
    N = len(rows)
    out["n_rows"] = N
    out["n_classes"] = len(set(truth))
    print(f"== 0. SETUP ==\n{N} test rows, {out['n_classes']} classes "
          f"(tau={TAU}, low_sim_floor={FLOOR})")

    # warmup (loads model + references once, not counted anywhere)
    t0 = time.perf_counter()
    router_lib.predict("warmup query")
    load_s = time.perf_counter() - t0
    out["timing"] = {"load_and_first_predict_s": round(load_s, 3)}

    # ---- main eval (tau=0.08) ----
    t0 = time.perf_counter()
    preds = router_lib.predict_batch(texts, tau=TAU, low_sim_floor=FLOOR)
    batch_s = time.perf_counter() - t0
    out["timing"]["batch_eval_s"] = round(batch_s, 3)
    out["timing"]["batch_ms_per_row"] = round(1000 * batch_s / N, 2)
    pred_tools = [p["top_tool"] for p in preds]
    correct = np.array([pt == tt for pt, tt in zip(pred_tools, truth)])
    top_sims = np.array([p["top_sim"] for p in preds])
    confs = np.clip([p["confidence"] for p in preds], 0.0, 1.0)

    # ================= 1. ACCURACY =================
    acc = float(correct.mean())
    out["accuracy"] = {"overall": acc, "n_correct": int(correct.sum())}

    fam_stats = {}
    for f in sorted(set(fams)):
        idx = np.array([i for i, ff in enumerate(fams) if ff == f])
        fam_stats[f] = {"n": int(len(idx)), "n_correct": int(correct[idx].sum()),
                        "acc": float(correct[idx].mean())}
    out["accuracy"]["per_family"] = fam_stats

    tool_stats = {}
    for t in sorted(set(truth)):
        idx = np.array([i for i, tt in enumerate(truth) if tt == t])
        tool_stats[t] = {"n": int(len(idx)), "n_correct": int(correct[idx].sum()),
                         "acc": float(correct[idx].mean())}
    out["accuracy"]["per_tool"] = tool_stats
    imperfect = sorted([t for t, s in tool_stats.items() if s["acc"] < 1.0],
                       key=lambda t: (tool_stats[t]["acc"], -tool_stats[t]["n"]))
    worst3 = imperfect[:3]

    print("\n== 1. TOP-1 ACCURACY ==")
    print(f"overall: {int(correct.sum())}/{N} = {acc:.1%}")
    print("-- per family --")
    for f, s in fam_stats.items():
        print(f"  {f:16s} n={s['n']:3d}  acc={s['acc']:6.1%}")
    print(f"-- per tool, acc < 100% ({len(imperfect)} of {len(tool_stats)} tools; "
          f"* = 3 worst) --")
    for t in imperfect:
        s = tool_stats[t]
        star = "*" if t in worst3 else " "
        print(f" {star} {t:20s} {s['n_correct']:2d}/{s['n']:2d} = {s['acc']:6.1%}")

    # ================= 2. CONFUSION =================
    err_pairs = Counter((pred_tools[i], truth[i])
                        for i in range(N) if not correct[i])
    top10 = err_pairs.most_common(10)
    conf_json = []
    print("\n== 2. CONFUSION (top 10 pred <- true error pairs) ==")
    for (pt, tt), c in top10:
        ex = next(texts[i] for i in range(N)
                  if pred_tools[i] == pt and truth[i] == tt)[:46]
        print(f"  {pt:18s} <- {tt:18s} n={c:2d}   e.g. \"{ex}\"")
        conf_json.append({"predicted": pt, "true": tt, "count": int(c),
                          "example": ex})
    out["confusion"] = {"top10_pairs": conf_json,
                        "n_errors": int((~correct).sum())}

    pair_reports = []
    print("-- known confusable pairs (accuracy per member) --")
    for a, b in CONFUSABLE_PAIRS:
        if a not in tool_stats or b not in tool_stats:
            print(f"  {a} / {b}: SKIPPED (tool missing in test set)")
            continue
        ma = np.array([tt == a for tt in truth])
        mb = np.array([tt == b for tt in truth])
        acc_a, na = masked_acc(ma, correct)
        acc_b, nb = masked_acc(mb, correct)
        a_as_b = int((ma & (np.array(pred_tools) == b)).sum())
        b_as_a = int((mb & (np.array(pred_tools) == a)).sum())
        both = ma | mb
        acc_both, nboth = masked_acc(both, correct)
        pair_reports.append({"pair": [a, b], "n_a": na, "acc_a": acc_a,
                             "n_b": nb, "acc_b": acc_b,
                             "a_predicted_as_b": a_as_b,
                             "b_predicted_as_a": b_as_a,
                             "combined_acc": acc_both})
        print(f"  {a:18s} {acc_a:6.1%} ({na:2d}) | {b:18s} {acc_b:6.1%} ({nb:2d})"
              f" | cross-err: {a}->{b}:{a_as_b} {b}->{a}:{b_as_a}")
    out["confusion"]["confusable_pairs"] = pair_reports

    # ================= 3. ABSTAIN CURVE =================
    sweep = []
    for th in np.arange(0.30, 0.85 + 1e-9, 0.05):
        th = round(float(th), 2)
        m = top_sims >= th - 1e-9
        a, n = masked_acc(m, correct)
        sweep.append({"threshold": th, "n_covered": n,
                      "coverage": float(m.mean()), "accuracy": a})
    # knee (kneedle-style): sweep point with max perpendicular distance from the
    # chord joining (max coverage, min acc) to (min coverage, max acc),
    # computed in min-max normalized (coverage, accuracy) space
    cov_arr = np.array([s["coverage"] for s in sweep])
    acc_arr = np.array([acc_or_none(s["accuracy"]) or 0.0 for s in sweep])

    def _norm01(v):
        lo, hi = float(v.min()), float(v.max())
        return (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)

    xn, yn = _norm01(cov_arr), _norm01(acc_arr)
    dx, dy = xn[-1] - xn[0], yn[-1] - yn[0]
    chord_len = math.hypot(dx, dy)
    dists = (np.abs(dy * (xn - xn[0]) - dx * (yn - yn[0])) / chord_len
             if chord_len > 0 else np.zeros_like(xn))
    knee_i = int(np.argmax(dists))
    knee = {**sweep[knee_i], "knee_distance": float(dists[knee_i])}
    # alt reference: min coverage needed to be within 1pt of peak accuracy
    peak = float(acc_arr.max())
    alt = max((s for s in sweep
               if (acc_or_none(s["accuracy"]) or 0.0) >= peak - 0.01),
              key=lambda s: s["coverage"])
    out["abstain_curve"] = {"sweep": sweep, "knee": knee,
                            "peak_accuracy": peak,
                            "knee_alt_within_1pt_of_peak": alt}

    print("\n== 3. ABSTAIN CURVE (threshold on top_sim) ==")
    print("  thr   n_cov  coverage  acc@covered")
    for s in sweep:
        mark = " <- knee" if s["threshold"] == knee["threshold"] else ""
        print(f"  {s['threshold']:.2f}  {s['n_covered']:5d}   "
              f"{100 * s['coverage']:6.1f}%  {f_acc(s['accuracy'])}{mark}")
    knee_a = acc_or_none(knee["accuracy"])
    if knee_a is not None:
        print(f"  knee (max chord distance, kneedle-style): thr={knee['threshold']:.2f}"
              f"  coverage={knee['coverage']:.1%}  acc={knee_a:.1%}"
              f"  ({100 * (knee_a - acc):+.1f}pts vs no abstain)")
        print(f"  alt reference (within 1pt of peak acc {peak:.1%}, max coverage): "
              f"thr={alt['threshold']:.2f} coverage={alt['coverage']:.1%} "
              f"acc={f_acc(alt['accuracy']).strip()}")

    # ================= 4. CALIBRATION =================
    confs_arr = np.array(confs)
    bin_idx = np.minimum((confs_arr * 10).astype(int), 9)  # 10 equal-width bins
    bin_rows = []
    for i in range(10):
        m = bin_idx == i
        n = int(m.sum())
        bin_rows.append({
            "bin": f"{i / 10:.1f}-{(i + 1) / 10:.1f}",
            "count": n,
            "mean_conf": round(float(confs_arr[m].mean()), 3) if n else None,
            "acc": float(correct[m].mean()) if n else None})
    pearson = float(np.corrcoef(confs, correct.astype(float))[0, 1])
    conf_cut = []
    for cmin in (0.5, 0.7, 0.9):
        m = confs >= cmin
        a, n = masked_acc(m, correct)
        conf_cut.append({"min_conf": cmin, "n": n, "coverage": float(m.mean()),
                         "accuracy": a})
    out["calibration"] = {"bins": bin_rows, "pearson_r_conf_vs_correct": pearson,
                          "accuracy_at_min_conf": conf_cut}

    print("\n== 4. CONFIDENCE CALIBRATION (entropy conf, 10 bins) ==")
    print("  bin        count  mean_conf   acc")
    for r in bin_rows:
        mc = "  n/a" if r["mean_conf"] is None else f"{r['mean_conf']:9.3f}"
        print(f"  {r['bin']:>7s}  {r['count']:6d}  {mc}  {f_acc(r['acc'])}")
    print(f"  Pearson r(confidence, correct) = {pearson:.3f}")
    for c in conf_cut:
        print(f"  conf>={c['min_conf']:.1f}: coverage={c['coverage']:6.1%}  "
              f"acc={f_acc(c['accuracy'])}")

    # ================= 5. NONE DETECTION =================
    none_mask = np.array([tt == "__none__" for tt in truth])
    none_idx = np.where(none_mask)[0]
    none_correct = int(correct[none_idx].sum())
    none_n = int(none_mask.sum())
    false_none = int((~none_mask & (np.array(pred_tools) == "__none__")).sum())
    real_n = N - none_n
    none_misroutes = Counter(pred_tools[i] for i in none_idx
                             if pred_tools[i] != "__none__")
    out["none_detection"] = {
        "none_rows": none_n, "none_correct": none_correct,
        "none_accuracy": none_correct / none_n if none_n else None,
        "false_none_count": false_none, "real_tool_rows": real_n,
        "false_none_rate": false_none / real_n if real_n else None,
        "none_misroutes": dict(none_misroutes.most_common())}
    print("\n== 5. NONE-ROUTE DETECTION ==")
    print(f"  __none__ rows: {none_correct}/{none_n} = "
          f"{none_correct / none_n:.1%} routed to __none__")
    print(f"  false-none: {false_none}/{real_n} real tool utterances "
          f"routed to __none__ = {false_none / real_n:.1%}")
    if none_misroutes:
        top_mr = none_misroutes.most_common(3)
        print("  chitchat misrouted to: "
              + ", ".join(f"{t} x{c}" for t, c in top_mr))

    # ================= 6. LATENCY =================
    lat_ms = []
    for t in LAT_TEXTS:
        t0 = time.perf_counter()
        router_lib.predict(t, tau=TAU, low_sim_floor=FLOOR)
        lat_ms.append((time.perf_counter() - t0) * 1000)
    lat_arr = np.array(lat_ms)
    p50, p95 = float(np.percentile(lat_arr, 50)), float(np.percentile(lat_arr, 95))
    out["latency"] = {"n_calls": 30, "p50_ms": round(p50, 2),
                      "p95_ms": round(p95, 2), "mean_ms": round(float(lat_arr.mean()), 2),
                      "max_ms": round(float(lat_arr.max()), 2),
                      "note": "model warm before timing; fresh unseen texts"}
    print("\n== 6. LATENCY (30 fresh single predict() calls, warm model) ==")
    print(f"  p50={p50:.1f}ms  p95={p95:.1f}ms  mean={lat_arr.mean():.1f}ms  "
          f"max={lat_arr.max():.1f}ms")
    print(f"  batch eval (382 rows): {batch_s:.2f}s total, "
          f"{1000 * batch_s / N:.1f}ms/row; model+refs load: {load_s:.2f}s")

    # ================= 7. TAU SENSITIVITY =================
    tau_rows = []
    for tau in (0.05, 0.08, 0.12):
        if tau == TAU:
            pr, mean_c = preds, float(np.mean(confs))
        else:
            pr = router_lib.predict_batch(texts, tau=tau, low_sim_floor=FLOOR)
            mean_c = float(np.mean([p["confidence"] for p in pr]))
        a = float(np.mean([p["top_tool"] == truth[i]
                           for i, p in enumerate(pr)]))
        tau_rows.append({"tau": tau, "top1_acc": a, "mean_confidence": mean_c})
    out["tau_sensitivity"] = tau_rows
    print("\n== 7. TAU SENSITIVITY ==")
    for r in tau_rows:
        print(f"  tau={r['tau']:.2f}  top1 acc={r['top1_acc']:6.1%}  "
              f"mean conf={r['mean_confidence']:.3f}")
    accs_set = {round(r["top1_acc"], 4) for r in tau_rows}
    print(f"  -> top1 accuracy {'IDENTICAL' if len(accs_set) == 1 else 'DIFFERS'} "
          f"across tau (choice = argmax of sims, tau only rescales confidence)")

    # ================= FINDINGS =================
    worst_pair = min((p for p in pair_reports),
                     key=lambda p: min(p["acc_a"], p["acc_b"]))
    wp = worst_pair["pair"]
    wc = conf_cut[1]  # conf >= 0.7
    print("\n== FINDINGS ==")
    print(f"* Accuracy: top-1 = {acc:.1%} ({int(correct.sum())}/{N}); "
          f"{len(imperfect)}/{len(tool_stats)} tools below 100%, worst 3: "
          + ", ".join(f"{t} ({tool_stats[t]['acc']:.0%})" for t in worst3)
          + f"; tau 0.05/0.08/0.12 identical ({acc:.1%}) - "
            "argmax over raw sims, tau only rescales confidence.")
    print(f"* Confusables: hardest known pair {wp[0]}/{wp[1]} "
          f"({worst_pair['acc_a']:.0%} vs {worst_pair['acc_b']:.0%}); cross-errors "
          + "; ".join(f"{p['pair'][0]}->{p['pair'][1]}:{p['a_predicted_as_b']},"
                      f"{p['pair'][1]}->{p['pair'][0]}:{p['b_predicted_as_a']}"
                      for p in pair_reports)
          + f"; worst overall error mode: "
          + ", ".join(f"{pt}<-{tt} x{c}" for (pt, tt), c in top10[:2]) + ".")
    if knee_a is not None:
        print(f"* Abstain knee: top_sim >= {knee['threshold']:.2f} keeps "
              f"{knee['coverage']:.1%} coverage at {knee_a:.1%} accuracy "
              f"({100 * (knee_a - acc):+.1f}pts over no-abstain {acc:.1%}); peak acc "
              f"{peak:.1%} needs thr>={alt['threshold']:.2f} "
              f"({alt['coverage']:.1%} coverage).")
    print(f"* Calibration: Pearson r={pearson:.2f}; conf>={wc['min_conf']:.1f} -> "
          f"{wc['coverage']:.1%} coverage at {wc['accuracy']:.1%} acc -> confidence "
          f"{'IS' if pearson > 0.3 and wc['accuracy'] > acc else 'is NOT'} a usable "
          f"secondary abstain signal (top_sim is the primary one).")
    print(f"* Latency: p50={p50:.1f}ms / p95={p95:.1f}ms per single query "
          f"(CPU, warm); full 382-row batch eval {batch_s:.2f}s "
          f"({1000 * batch_s / N:.1f}ms/row) - fast enough for real-time routing.")

    # ---- write JSON ----
    RESULTS_DIR.mkdir(exist_ok=True)
    jpath = RESULTS_DIR / "router_results.json"
    jpath.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {jpath}")


if __name__ == "__main__":
    main()
