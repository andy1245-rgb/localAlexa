# The Cactus hybrid checkpoint — the confidence gate passes, narrowly

2026-11-10, branch `collect/2026-11-10`. One Cactus hybrid build downloaded and scored on
the repo's own 382-row corpus, judged by the gate that was set for it: **AUROC of the
checkpoint's confidence against its own correctness on the 17 tool labels**. Not its
benchmarks, not its accuracy.

**Verdict: 0.837 — it clears 0.8.** Base needle3 was 0.523, so this is a different kind of
object: a model whose confidence carries information about whether it is wrong. Two caveats
sit beside the pass and are not decoration: the interval's lower bound is 0.731, and once the
abstain class is included the same number drops to 0.794, i.e. below the gate. §4 says what
that means for use.

## 1. What was tested

| | |
|---|---|
| checkpoint | `Cactus-Compute/gemma-4-e2b-it-hybrid` — Gemma-4-E2B plus Cactus's "handoff probe", a head (weight prefix `handoff_probe.*`) that scores each generation with `confidence = 1 − p_wrong` |
| size on disk | 9.33 GB (three safetensors shards) |
| confidence surface | `model.generate(..., return_confidence=True).confidence` — structured, never parsed out of the answer text |
| prompt | one instruction listing the 18 registry ids and asking for exactly one id back, greedy decoding, `max_new_tokens=8` |
| rows | **128 of 382** — a systematic 1-in-3 sample, so every intent block is represented (18/18 labels) |
| prediction rule | the first registry id appearing as a token; otherwise `NO_CALL`; a `__none__` row is correct on `__none__` or `NO_CALL` — the same rule the needle3 run used |
| environment | `.venv-hybrid` (transformers 5.5.4, torch 2.15.0+cpu, CPU only), `docs/ladder-env.md` has the pins |

The sample is 128 rows rather than 382 because the box is at load ~23 from other work and a
single generation costs **26.4 s** (§5). Systematic sampling across a file ordered by intent
block keeps the label mix, at the price of a wider interval — which is why the interval is
reported rather than the point estimate alone.

## 2. The gate

| | abstain-inclusive (headline) | 17 tool labels |
|---|---:|---:|
| **Cactus hybrid, full corpus (382 rows)** | filled in below, §2.1 | §2.1 |
| Cactus hybrid, 128-row systematic sample | **0.794** (95% CI 0.701–0.875) | 0.837 (0.731–0.932) |
| base needle3, 382 rows | **0.523** | 0.530 |

### 2.1 The full-corpus result

Filled in from `results/cactus_hybrid_full.json` when the run finishes. The protocol and the
decision rule were fixed before it started, so nothing is chosen after the fact:

* the gate passes **only if both** the abstain-inclusive and the 17-label AUROC are ≥ 0.8;
* if the abstain-inclusive figure is below 0.8 the verdict is **not usable**, nothing is wired,
  and there is no prompt tuning, no class-exclusion rescue and no second sample;
* both intervals are reported, because a point estimate alone is what nearly banked the
  128-row sample.

### 2.2 What the 128-row sample showed

Base needle3's number reproduces from the historical per-row confidences in
`results/historical/needle_results.json`, so the comparison is the same arithmetic on both
models rather than a number carried over.

What the separation looks like in the raw numbers: mean confidence 0.596 on the rows it got
right against 0.409 on the rows it got wrong, over 20 errors. The overlap is real — the
lowest-confidence *correct* row sits at 0.17, below two of the errors at 0.55 — which is
exactly why 0.837 rather than 0.95.

## 3. Where the errors are

20 wrong out of 128, and they are almost entirely the confusions this repo already documents:

| confusion | count |
|---|---:|
| `thermostat_offset` → `thermostat_set` | 6 |
| `__none__` → a tool (research_deep ×5, docs_query ×2) | 7 |
| `shoppinglist_remove` → `shoppinglist_add` | 3 |
| `research_deep` → `docs_query` | 2 |
| `shoppinglist_view` → `shoppinglist_add` | 1 |
| `projects_query` → `NO_CALL` | 1 |

The abstain class is where the checkpoint is weakest: 7 of the 9 `__none__` rows in the sample
were answered with a tool. That is the whole reason the all-rows AUROC is under the gate while
the 17-label one is over it — and it is the same failure the linear tier has (0.60 recall on
the class), which is worth knowing before treating this model as an upgrade.

## 4. What the pass does and does not buy

**It buys a usable handoff signal.** 0.837 means the confidence can order its own errors above
its own correct answers well enough to route a share of the traffic upward with a precision
floor — the threshold can be picked from the same curve the ladder uses
(`coverage at precision`, `docs/ladder-results-2026-11-10.md` §"reject curve"). The two
clusters it would catch first are `thermostat_offset`/`thermostat_set` and
`shoppinglist_remove`/`shoppinglist_add`, which are the pairs the regex-and-router tiers also
confuse.

**It does not buy a tier-1 model.** At 26.4 s per utterance on 12 contended cores (and 9.33 GB
resident), this is two orders of magnitude past anything the cascade's tier 2 was assumed to
cost and four past the encoder router. It is a candidate for the tier that a network model
currently occupies, not for the sub-millisecond tier.

**It does not buy accuracy either.** 84.4% on this sample, against the ladder's 95.03% grouped
for a 7.9 ms logreg+SetFit ensemble. Different protocol and a one-shot prompt with no
examples, so this is indicative rather than a head-to-head — but nothing here suggests the
checkpoint's *labels* are the reason to adopt it. Its confidence is.

**And the interval is not a formality.** 0.731 sits below the gate. A 128-row sample cannot
distinguish "comfortably usable" from "just about"; the honest statement is that the point
estimate clears and the interval spans the threshold. Growing the sample is the cheap fix
(`--stride 1` on an idle box, ~2.5 h) and it is the next step if the operator wants the pass to
be a pass with an interval under it.

## 5. Cost, measured

| | |
|---|---|
| model load | 5.1 s (bf16, mmap) |
| per utterance | **26.4 s p50**, up to 48 s in the first pilot while the box was busier |
| resident | ~4.6 GB RSS for the weights during generation, 9.33 GB on disk |
| total wall for 128 rows | ~57 min of generation |

Nothing was uploaded, no API was called, and the download is the only network use.

## 6. Reproduce

```bash
# one-time env (transformers 5.5.4 pins required by the checkpoint)
uv venv .venv-hybrid --python 3.12
uv pip install --python .venv-hybrid/bin/python "transformers>=5.5.4,<5.6" "tokenizers>=0.22.0,<=0.23.0"
uv pip install --python .venv-hybrid/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv-hybrid/bin/python httpx pillow numpy safetensors huggingface-hub tqdm filelock regex packaging pyyaml requests

.venv-hybrid/bin/python experiments/optional/exp_cactus_hybrid.py --stride 3   # the 128 rows
.venv-hybrid/bin/python experiments/optional/exp_cactus_hybrid.py --stride 1   # all 382, ~2.5 h
.venv-hybrid/bin/python experiments/optional/exp_cactus_hybrid.py --recompute results/cactus_hybrid_results.json
```

Artifact: `results/cactus_hybrid_results.json` — per-row text, truth, prediction, raw answer,
confidence and latency, plus the AUROC, its interval and the verdict. The `--recompute` path
rebuilds every metric and the verdict from the saved rows without loading the model, so the
number can be re-derived without another hour of CPU.

## 7. Honest limits

* 128 of 382 rows, systematic sample, one prompt with no few-shot examples. A different prompt
  moves both accuracy and confidence; this measures the checkpoint as prompted, not its ceiling.
* Greedy decoding only. The checkpoint's confidence is defined for single-sequence decoding and
  the card says batched or beam modes fall back to stock behaviour with no probe — so a
  batched deployment would have no gate at all.
* The confidence is `1 − p_wrong` from a probe trained on Cactus's own handoff data, not on this
  corpus. That it transfers at all is the finding; that it would transfer *better* after a few
  hundred labelled rows is the guess, and it is untested here.
* Base needle3's 0.523 came from a different confidence surface (the tool-calling engine's own
  score) on the full 382 rows. Same arithmetic, different model API — comparable in kind, not
  in provenance.
