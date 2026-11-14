# The cascade against one classifier — rerun grouped, and the answer changes

2026-11-10, branch `collect/2026-11-10`. Two operator decisions are implemented here:
`SMART_ASSUMED_ACC` is gone (§1), and the cascade-vs-single-classifier comparison is
re-measured grouped and without the leak (§2). The earlier verdict used the row-wise
93.19% for the single classifier and a tier-1 model trained on the very slot fills it was
scored on. Both flatter the cheap tiers. Measured honestly the verdict is **not** the same.

## 1. The end-to-end number is a function now, not a constant

`SMART_ASSUMED_ACC = 0.95` is deleted. Tier-2 accuracy is a parameter of the measurement:

```
e2e(x) = (auto_correct + x · escalated) / n
```

so `cascade_metrics` reports, instead of one number with a decimal in it:

* `e2e_curve_pct` — end-to-end accuracy at tier-2 accuracies 0.80 … 1.00;
* `e2e_at_measured_tier2_pct` — the value at whatever tier-2 accuracy you actually have;
* `tier2_breakeven` — the tier-2 accuracy needed to match a stated target, solved for,
  with a verdict of *already reached* / *needed* / *unreachable at this escalation rate*.

The old 73.8% is still in the test suite, now as `e2e_curve_pct["0.95"] == 73.8`: the same
arithmetic, labelled as the assumption it always was. `tests/test_cascade.py` grows four
cases for the break-even: reachable, already reached, unreachable, and "nothing escalates
so the number cannot move".

## 2. The comparison, measured on grouped folds

Same 5 folds over the 110 phrasing families as the ladder; tier 1 refitted per fold on the
reference pool minus the slot fills of the held-out families; 382 rows.

| | correct | accuracy |
|---|---:|---:|
| **best single classifier** (word logreg C=100) | **347/382** | **90.84%** |
| tier 1 alone, under the leak (the old basis) | 357/382 | 93.46% |
| tier 0 regex alone | 234/256 | 91.41% |

The leak is worth **7.05 points** to tier 1. That is the whole reason the earlier verdict is
suspect: the gate looked harmless because the model it was gating had been shown the answers.

### At the production gate (confidence ≥ 0.35), the cascade loses

| | value |
|---|---:|
| coverage | 94.0% |
| accuracy on what it answers | 89.97% |
| end-to-end at tier-2 = 0.95 | 90.3% |
| break-even vs 90.84% | **1.0435 — unreachable** |

No tier-2 model, not even a perfect one, makes this operating point beat a single
classifier. The gate keeps so much traffic that tier 2 barely exists: 23 of 382 rows.

### But the gate is the constraint, not the tier-2 model

Sweeping the gate changes the answer. At each tier-2 accuracy, the best gate and what it
reaches:

| tier-2 accuracy | best gate | coverage | end-to-end | vs 90.84% |
|---|---:|---:|---:|---|
| 0.80 | 0.70 | 83% | 90.3% | loses |
| **0.85** | 0.70 | 83% | **91.1%** | **beats** |
| 0.90 | 0.70 | 83% | 92.0% | beats |
| 0.95 | 0.70 | 83% | 92.9% | beats |
| 0.97 | 0.90 | 73% | 93.4% | beats |
| 1.00 | 0.90 | 73% | 94.2% | beats |

So: **the cascade does beat one classifier, and the number that was blocking it was the
gate threshold (0.35), not the expensive tier.** With the gate at 0.70 the cascade answers
83% of traffic itself and beats the 90.84% single classifier as soon as tier 2 reaches 85%.
At 0.35 it cannot win at any tier-2 accuracy.

### The 91.9 is real, and it is the router variant

The operator's figure reproduces exactly when tier 1 is the production encoder router
rather than the linear model (`--tier1 router`, `results/compare/cascade-vs-classifier-router.json`):

| tier-1 model | answering every row | on the rows the gate routes | under the leak | cascade at its best gate |
|---|---:|---:|---:|---:|
| word logreg C=100 | 90.84% | 86.41% | 93.46% | 92.9% (gate 0.70, tier 2 at 0.95) |
| MiniLM router | 86.65% | 89.90% | 93.46% | **91.9%** (gate 0.80, tier 2 at 0.90) |

Both rows are now in the artifacts, so the claim can be checked instead of remembered, and
91.9% is beaten against the 90.84% single classifier in the same file
(`beats_one_classifier: true`). The oracle-gate ceiling — perfect escalation, same cheap
tiers — is 98.1% at tier-2 0.80 and 100% at 1.00, i.e. the headroom is in *deciding what to
escalate*, not in the model.

## 3. What this changes

1. **Ship the gate at ~0.70, not 0.35.** It is the single highest-value change available:
   it turns an operating point that cannot win into one that wins whenever tier 2 is at
   85% or better. It also means less traffic to the expensive tier, which is the point.
2. **The criterion for a tier-2 model is now a number.** At gate 0.70 and 83% coverage it
   must be ≥85% accurate on the escalated rows to beat a single classifier. Above that it
   buys points; below it, escalate nothing and run the classifier.
3. **Do not quote 93.46% for tier 1.** That is the leak. The honest figure is 86.41% on the
   rows the gate routes to it, 90.84% when it answers everything.
4. **The earlier "the cascade cannot beat one classifier" is withdrawn**, along with the
   reasoning that produced it. It was true of the operating point and false of the design.

## 4. Reproduce

```bash
.venv/bin/python -m localalexa compare --tier1 logreg   # 90.84% bar, gate sweep
.venv/bin/python -m localalexa compare --tier1 router   # needs 'localalexa[embeddings]'
.venv/bin/python -m localalexa cascade --tier2 0.92 --target 347   # e2e at a measured tier 2
.venv/bin/python -m pytest tests/test_compare.py tests/test_cascade.py -q
```

Artifacts: `results/compare/cascade-vs-classifier-{logreg,router}.json`. The gate sweep and
the per-gate tier accuracies are in each, so an operating point can be re-read without
re-running anything.

## 5. Caveats

* The tier-2 accuracies in the tables are a *parameter sweep*, not a measurement: no LLM has
  been scored on this corpus. The break-even column is the honest form of that gap — it says
  which accuracies would matter instead of pretending to know one.
* Tier 2 is modelled as a constant accuracy across escalated rows. A real model is better on
  some intents than others, and its errors will not be independent of the cheap tiers', so
  the true curve is optimistic at the top end. The break-even is a floor, not a forecast.
* The regex tier is deterministic, so its 91.41% on 256 rows is not affected by the leak and
  was never wrong in the first place; what moved was tier 1, and with it the cascade.
