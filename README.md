# localAlexa

Offline intent classification for a voice assistant. A spoken command has to become an
operation (`timer_create`, `thermostat_set`, …) before anything runs, and doing that with an
LLM means paying a model call per utterance. This routes locally in well under a
millisecond and escalates only when the cheap tiers are unsure. Software prototype: the hardware
side (ESP32/C firmware, mic array, wake word, on-device deployment) is a later phase, is **not**
in this repo, and has not been built.

## Architecture

```
utterance
  tier 0   hand-written per-tool regex                                    ~0 ms, free
  tier 1   reference-similarity router (MiniLM)  or TF-IDF + LinearSVC   0.6–2 ms
  tier 2   abstain -> escalate to a real LLM                             only when unsure
```

Tier 0 answers when exactly one tool's patterns fire. Everything else reaches tier 1, which
returns a label, top similarity, margin and `confidence = 1 − H(p)/ln(K)` over a softmax of
the per-tool maximum similarities. Under either threshold the row escalates rather than
guessing, so the expensive tier only sees admitted uncertainty.

## Measured results

382 utterances, 110 phrasing families, 18 labels (17 tools + a none route). Rows come from
the same templates the registry trains on, so the split decides the number:

| split | training sees | accuracy | 95% CI | vs row-wise |
|---|---|---|---|---|
| row-wise | the 200 registry references | **93.2%** (356/382) | 88.0–97.6 | — |
| phrasing-family held out | same pool minus the held-out family's slot fills | 90.3% (345/382) | 85.0–95.2 | −2.9 |
| phrasing-family held out, strict | rebuilt from the other 109 families' fills | **74.9%** (286/382) | 66.7–83.1 | −18.3 |

93.2% is template-contaminated: 29 of the 200 training references are literal slot fills of
the templates that generate the test rows. The strict split is the generalisation estimate;
the middle row shows that dropping one template's slot fills barely moves anything, since the
same tool's other references are paraphrases from the same grammar. Intervals resample whole
phrasing families (110 clusters), not rows — rows inside a family are not independent.

### The classifier ladder (2026-11-10)

Six rungs were built and measured on one grouped fold assignment against the
93.19% / 0.61 ms baseline — vectoriser union, calibration, SetFit, cross-encoder
re-rank, ensembles, augmentation. Nothing beat 93.19% at the 0.61 ms budget; the
rung that beats it by a real margin costs 7.85 ms. The full result, with the
reject curves and the reasons, is `docs/ladder-results-2026-11-10.md`; the
artifacts are `results/ladder/`. Latency is one utterance, warm, idle box, median
of five passes.

| what | grouped acc | rowwise acc | p50 ms | `__none__` recall |
|---|---:|---:|---:|---:|
| baseline: word TF-IDF + LinearSVC | 89.79% | 93.19% | 0.39 | 0.60 |
| word TF-IDF + LogisticRegression | 90.84% | 93.46% | 0.38 | 0.68 |
| SetFit (MiniLM contrastive + head) | 93.98% | 94.24% | 7.75 | 0.84 |
| MiniLM cross-encoder re-rank of top 3 | 92.15% | 93.19% | 11.79 | 0.48 |
| ensemble: logreg + SetFit, standardised | 95.03% | 96.86% | 7.85 | 0.88 |
| + all registry slot fills (augmentation) | 86.65% | 96.34% | 0.78 | 0.44 |

Three things that table says and an accuracy column hides. The logistic rung takes
ECE from 0.735 to 0.128 for the same 0.38 ms (the LinearSVC's "confidence" was a
ranking wearing a probability's clothes). The ensemble improves in 5 of 5 folds
and lifts coverage at 97.5% precision from 71.5% to 91.9%. And 96.34% is not a
result — it is the row-wise protocol being fed the generator's own slot fills,
which is why the honest number beside it is 86.65%.

Tier 0 alone: 76.2% coverage at 83.8% first-hit accuracy, 12% of covered rows ambiguous, 1 of
25 chitchat rows falsely matched, 67 tool rows unmatched. Tier-1 latency: 0.61 ms p50 for
LinearSVC, 13–17 ms for the torch MiniLM router (CPU, warm) — re-measure with
`localalexa classic`. The 3B model's 91.36% in `results/historical/d1_results.json` is **not**
quoted: it was measured on a different task whose labels do not overlap these operations.

## Run it

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -e . --group dev
.venv/bin/python -m pytest                      # 61 tests
.venv/bin/python -m localalexa eval --split all # the table above, ~20 s
.venv/bin/python -m localalexa classic          # the three TF-IDF configurations
.venv/bin/python -m localalexa regex            # tier 0 detail, no model needed
.venv/bin/python -m localalexa cascade          # needs pip install 'localalexa[embeddings]'
.venv/bin/python -m localalexa ladder --rungs all --folds 5   # the ladder, rungs 0-6
```

Rungs 3–4 need torch and a transformer encoder; `docs/ladder-env.md` has the
second venv, the pinned versions, and why the paraphrase half of rung 6 could not
be measured on this box.

`data/routing_382.json` and `data/registry.json` ship with the repo, so every number reproduces
offline. CLINC150 / Banking77 / HWU64 are not vendored; loading them is optional.
`docs/ladder-results-2026-11-10.md` is the six-rung experiment and its verdict;
`docs/RESEARCH.md` is the prior research with a dated correction note on top;
`results/historical/` holds the original experiment JSONs unedited; `experiments/optional/`
holds the scripts that need a downloaded model or a llama.cpp server. Apache-2.0.

## What I'd do next

1. Grow the reference pool past 200 utterances — the strict split is the signal that 12
   references per tool is not enough.
2. ~~Give the LinearSVC tier a calibrated abstain~~ — answered in the ladder: swap in
   `LogisticRegression` on the same features (ECE 0.735 → 0.128, sub-millisecond, no
   new dependency). Choosing the abstain threshold on out-of-fold predictions rather
   than on the scored rows is the remaining half of this item.
3. Attack the known confusions (`light_query`/`light_set`, `timer_*` read/write) with negation
   and disambiguation patterns in tier 0.
4. int8-ONNX the MiniLM router, then re-measure whether tier 1 earns its memory; firmware last.
