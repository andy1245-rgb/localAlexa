# Classifier ladder — is there more in localAlexa than one linear model?

2026-11-10. The operator asked the right question: the last measurement said "use TF-IDF + LinearSVC",
but that was one point on a curve nobody has walked. This document says which rungs are worth
climbing, in expected-gain order, and what would make each result untrustworthy.

## Where we are

| model | accuracy | p50 latency | source |
|---|---:|---:|---|
| TF-IDF(word) + LinearSVC | **93.19%** | **0.61 ms** | `results/historical/classic_results.json` |
| sentence-transformers MiniLM router | 86.65% | 57.8 ms | `results/historical/cascade_results.json` |
| needle3 (Cactus, base) | 59.42% | 345 ms | `results/historical/needle_results.json` |

382 rows, 17 intents, `group` available (family), a `none` class with **0.6 detection** for the
linear model. Everything below is judged against 93.19% at 0.61 ms, and against the reject curve, not
just the accuracy.

**The binding constraint is data: ~22 examples per intent.** Say that in every result, because it
predicts which rungs will work.

## The rungs, in order

### 1. Vectoriser union — cheap, expect +1–3 points (minutes)

`FeatureUnion` of word 1–2 grams and `char_wb` 3–5 grams. Short imperative utterances
("turn on the kitchen lights") are exactly the case where character n-grams help. Free, in-process,
same latency order. **This should be the new baseline** before anything more exotic is tried.

### 2. Calibrated probabilities, and the reject curve (an hour)

`LinearSVC` has no probabilities; wrap it in `CalibratedClassifierCV(method="sigmoid")`, or use
`LogisticRegression` on the same features. Then report the thing that actually matters for a voice
assistant: **coverage at precision 0.9**, and accuracy on the rejected residue. The current
`none_accuracy` of 0.6 is the real defect — 40% of "this isn't a command" gets acted on.

This rung may beat every fancier model on the metric you care about, because a well-calibrated
linear model with a good threshold often beats a stronger model with a bad one.

### 3. Few-shot fine-tune — the honest version of "fine-tuning" (an afternoon, CPU)

Not a full fine-tune: at 382 rows that overfits. Use the **SetFit** recipe — contrastive fine-tuning
of a sentence encoder on pairs generated from the labelled set, then a light classifier head. It is
designed for 8–16 examples per class, which is exactly this corpus, and it is the method that
actually has a chance of beating TF-IDF here. CPU-feasible on a 22M-parameter encoder.

Report it against rung 2, on the same grouped folds, with the same reject curve. If it wins by less
than the fold-to-fold spread, it has not won.

### 4. Cross-encoder re-rank of the top 3 (adds latency, buys accuracy on confusable pairs)

Take the linear model's top 3, re-score them with a small cross-encoder. Latency goes from 0.6 ms to
tens of ms — still trivial for a server, probably too slow for an ESP32. Only worth it if the
confusion matrix shows the errors are *within* the top 3 rather than out of it. Check that first.

### 5. Ensembles and stacking — last, and only with a correlation report

Voting or stacking only helps if the members **fail on different rows**. Compute the pairwise error
correlation (and the disagreement rate) across rungs 1–4 before building any ensemble, and report it.
An ensemble of models that agree on their mistakes is a slower model with the same accuracy.

### 6. Data augmentation — the highest-ceiling rung, and the least glamorous

Generate paraphrases per intent with a local model (llama.cpp on this box, $0), keep only those the
current model is *unsure* about, have a human spot-check a sample, and add them. At 22 examples per
class this is where the real headroom is. It is also the only rung that raises the ceiling for
*every* other rung.

## Method rules — non-negotiable

- **Grouped CV by family**, never a random split. Utterances from one family in both train and test
  is how a classifier reports 97% and ships 88%. The fold assignment goes in the artifact.
- **Report the spread across folds**, not a single number, and refuse to call a win smaller than it.
- **Every rung reports**: accuracy, per-intent ledger, p50/p95 latency, ECE, coverage@precision 0.9,
  and accuracy on the rejected residue.
- **Same folds for every rung.** A ladder measured on different splits is not a ladder.
- **Do not change `cascade.py` semantics** and do not touch `SMART_ASSUMED_ACC` — replacing that
  constant is a separate operator decision that is already on his todo list.

## Where this lands

`~/toolInterfacer/bench` is being made into a shared harness with dataset adapters and a predictor
protocol (`docs/harness-spec-2026-11-10.md` there). Build the ladder first with this repo's own
grouped-CV code so the result exists, then **port the winners onto the harness** so the numbers are
reproducible in the same place as everything else. Do not block on the harness.

## Hard limits

Branch commits only, never push, no paid API calls, no data deletion. If a package needs installing
(`setfit`, `sentence-transformers`), install it in a project venv and record the version — that is
free and allowed.
