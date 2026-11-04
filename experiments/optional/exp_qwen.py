"""Exp E — Qwen3.5-0.8B as a direct tool-choice classifier (CPU probe).

Stratified subset: 8 utterances per tool + all 25 none rows.
Greedy decoding, tool ids only in output. Measures accuracy + latency.
This is the 'big small-model' tier probe: no fine-tuning, zero-shot.
"""
import json
import os
import random
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
REG = json.loads((HERE / "registry.json").read_text(encoding="utf-8"))
TEST = json.loads((HERE / "data" / "test.json").read_text(encoding="utf-8"))
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)

PER_TOOL = 4
NONE_CAP = 10
random.Random(7).shuffle(TEST)
by_tool = {}
for r in TEST:
    by_tool.setdefault(r["tool"], []).append(r)
subset = []
for tool, rows in by_tool.items():
    take = rows[:NONE_CAP] if tool == "__none__" else rows[:PER_TOOL]
    subset.extend(take)

tool_lines = []
for t in REG["tools"]:
    desc = t["references"][0].rstrip("?").strip().lower()
    tool_lines.append(f"{t['id']}: {desc}")
tool_lines.append("no_action: request matches no tool")
TOOL_BLOCK = "\n".join(tool_lines)
VALID = {t["id"] for t in REG["tools"]} | {"no_action"}

PROMPT = """You are a command classifier for a home assistant. Pick the single best tool for the user command. Reply with ONLY the tool id, nothing else.

Tools:
{tools}

Command: {text}
Tool id:"""


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = "Qwen/Qwen3.5-0.8B"
    print(f"loading {name} on CPU ...", flush=True)
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype="float32")
    model.eval()
    try:
        torch_threads = min(8, os.cpu_count() or 4)
        import torch
        torch.set_num_threads(torch_threads)
        print(f"torch threads: {torch_threads}", flush=True)
    except Exception:
        pass

    lat, preds, golds, texts = [], [], [], []
    t_start = time.time()
    for i, row in enumerate(subset):
        prompt = PROMPT.format(tools=TOOL_BLOCK, text=row["text"])
        messages = [{"role": "user", "content": prompt}]
        try:
            chat_text = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
        except TypeError:
            chat_text = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(chat_text, return_tensors="pt")
        t0 = time.perf_counter()
        out = model.generate(**inputs, max_new_tokens=8, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        dt = (time.perf_counter() - t0) * 1000
        text_out = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        pred = text_out.strip().split()[0] if text_out.strip() else "NO_OUTPUT"
        pred = pred.strip('",.:;')
        if pred not in VALID:
            # try to find any valid tool id inside the output
            for v in VALID:
                if v in text_out:
                    pred = v
                    break
            else:
                pred = "INVALID:" + text_out.strip()[:20]
        preds.append(pred)
        golds.append(row["tool"])
        texts.append(row["text"])
        lat.append(dt)
        if (i + 1) % 20 == 0:
            acc_so_far = np.mean([p == g for p, g in zip(preds, golds)])
            print(f"  {i+1}/{len(subset)}  running-acc {acc_so_far:.3f}  "
                  f"mean-lat {np.mean(lat):.0f}ms", flush=True)

    total_s = time.time() - t_start
    lat = np.array(lat)
    golds_a, preds_a = np.array(golds), np.array(preds)
    acc = float((golds_a == preds_a).mean())

    per_family = {}
    for t, g, p in zip(texts, golds, preds):
        fam = next((t2["family"] for t2 in REG["tools"] if t2["id"] == g), "none")
        per_family.setdefault(fam, []).append(p == g)
    per_family = {k: round(float(np.mean(v)), 3) for k, v in per_family.items()}

    none_mask = golds_a == "__none__"
    tool_mask = ~none_mask
    none_detect = float((preds_a[none_mask] == "no_action").mean()) if none_mask.any() else None
    false_none = float((preds_a[tool_mask] == "no_action").mean())

    fails = [{"text": t, "true": g, "pred": p}
             for t, g, p in zip(texts, golds, preds) if g != p][:10]

    results = {
        "model": name,
        "n": len(subset),
        "accuracy": round(acc, 4),
        "per_family_accuracy": per_family,
        "none_detection": round(none_detect, 3),
        "false_none_rate": round(false_none, 3),
        "latency_p50_ms": round(float(np.percentile(lat, 50)), 1),
        "latency_p95_ms": round(float(np.percentile(lat, 95)), 1),
        "total_wall_s": round(total_s, 1),
        "failures": fails,
    }
    (OUT / "qwen_results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")

    print(f"\n=== Qwen3.5-0.8B zero-shot tool choice (n={len(subset)}) ===")
    print(f"accuracy: {acc:.3f}   none-detect: {results['none_detection']}   "
          f"false-none: {results['false_none_rate']}")
    print(f"latency p50/p95: {results['latency_p50_ms']}/{results['latency_p95_ms']} ms   "
          f"total: {total_s:.0f}s")
    print(f"per-family: {per_family}")
    for f in fails[:5]:
        print(f"  MISS true={f['true']:18s} pred={f['pred']:18s} <- {f['text'][:60]}")
    print(f"wrote {OUT / 'qwen_results.json'}")


if __name__ == "__main__":
    main()
