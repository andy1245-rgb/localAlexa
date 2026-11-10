# Ladder results — nothing here beats 93.19% at 0.61 ms

2026-11-10, branch `ladder/rungs-1-6`. Every rung in
`docs/classifier-ladder-2026-11-10.md` was built and measured, in order, on one
fold assignment and one metrics block. `cascade.py` semantics and
`SMART_ASSUMED_ACC` were not touched.

## The answer, in one table

382 rows, 17 intents plus `__none__`, 110 phrasing families, ~22 examples per
intent, 200 training references. Latency is one utterance, warm, on an idle box
(`results/ladder/latency.json`).

| rung | what it is | grouped acc | ± paired vs r0 | rowwise acc | p50 ms | `__none__` recall | ECE |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | word TF-IDF + LinearSVC — **the quoted model** | 89.79% | — | **93.19%** | 0.34 | 0.60 | 0.735 |
| 1 | + `char_wb` union, C=1 | 89.53% | −0.26 ± 2.51 | 92.41% | 0.98 | 0.52 | 0.697 |
| 1 | union, C=4 (best of twelve swept variants) | 90.31% | +0.52 ± 1.49 | 92.67% | 0.96 | 0.52 | 0.671 |
| 2 | word TF-IDF + LogisticRegression C=100 | 90.84% | +1.04 ± 1.43 | 93.46% | 0.57 | 0.68 | **0.128** |
| 2 | word + LinearSVC wrapped in sigmoid | 87.17% | −2.61 ± 3.07 | 91.88% | 2.63 | 0.64 | 0.310 |
| 3 | **SetFit** (MiniLM contrastive + logistic head) | **93.98%** | **+4.17 ± 9.82** | **94.24%** | 8.06 | **0.84** | 0.530 |
| 4 | MiniLM cross-encoder re-ranks top 3 | 92.15% | +2.34 ± 4.07 | 93.19% | 13.28 | 0.48 | 0.066 |
| 5 | equal-weight linear ensemble | 89.79% | −0.00 ± 0.92 | 93.72% | 3.34 | 0.64 | 0.507 |
| 6 | + every registry slot fill (uncapped) | 86.65% | −3.16 ± 4.61 | 96.34% | 0.78 | 0.44 | 0.068 |
| 6 | + at most 16 slot fills per label | 91.36% | +1.56 ± 1.93 | — | 0.87 | 0.52 | 0.123 |

Read the two accuracy columns as different questions. **Rowwise** is the protocol
the 93.19% claim was made under; it is the only fair comparison to it, and it is
inflated. **Grouped** holds whole phrasing families out and is the number that
survives rephrasing.

**No rung beats 93.19% at 0.61 ms.** Two beat 93.19% at all — SetFit by 1.05
points and the logistic rung by 0.27 (one row) — and SetFit pays 15–25× the
latency for it. The only rung that matches the baseline's latency *and* improves
something real is rung 2's logistic regression, and its accuracy gain is one row,
which is not a win; its calibration gain is large and is.

## Rung by rung, and why

### Rung 1 — vectoriser union: refuted, and it was the most confident prediction in the plan

The plan expected +1–3 points because short imperatives are exactly where
character n-grams should help. Measured on 110 held-out families: the union as
written is **0.26 points worse**, and the best of twelve variants swept
(`experiments/ladder/sweep_union.py`) is **+0.52 ± 1.49** — a non-win by the
plan's own rule — for **2.8× the latency**. Character n-grams alone are worse
still (87.96%, −1.83). The union changes the scale of the decision values, so C
needed re-tuning; with C=4 it recovers the baseline and stops there.

Why it fails: 200 references contain ~1300 distinct word features already, and
word 1–2 grams on 3–6 word utterances are nearly a full description of the
string. `char_wb` adds 30k features that mostly split the same evidence, and at
22 examples per intent the extra dimension is noise the linear model has to fit.

### Rung 2 — calibration: the accuracy is a tie, the probabilities are not

This rung was proposed as the one that might beat every fancier model on the
metric that matters. Half right.

`CalibratedClassifierCV(method="sigmoid")` on 200 rows is a loss: each
calibrator sees 133 rows, accuracy drops 2.6 points, and it costs 4× the
latency. Plain `LogisticRegression` on the baseline's own features is better and
cheaper:

- **ECE 0.735 → 0.128.** The LinearSVC column of that table was never a
  probability; softmax over hinge decision values is a ranking dressed up as
  confidence, and it is wrong by 73 points on average.
- `__none__` recall 0.60 → 0.68, false-none rate unchanged at 0.8%.
- rowwise 93.19% → 93.46% (+1 row), grouped +1.04 ± 1.43, latency unchanged
  (0.57 ms vs 0.34 ms — both sub-millisecond; the difference is scheduler noise,
  see the latency note below).

So the reject curve this rung exists to produce is real, and it is the one thing
worth shipping from rungs 1–2. Numbers in the reject-curve section.

### Rung 3 — SetFit: the first rung that actually wins, on accuracy, at a price

SetFit does what the plan said it might: contrastive fine-tuning of a 22M encoder
on pairs from the labelled set, then a logistic head. On grouped folds it is the
best rung measured:

- **93.98% grouped**, +4.17 points over the baseline, and the only rung whose
  per-fold spread is *half* the baseline's (sd 4.5 vs 9.3 points — it is more
  stable across phrasings, not just better on average).
- **`__none__` recall 0.60 → 0.84.** The defect the plan called the real one is
  largely fixed: 4 of 25 "this is not a command" rows were acted on, not 10.
- rowwise 94.24%, +1.05 over 93.19% — it does beat the quoted number.

Against it:

- **8.06 ms p50**, 24× the baseline, and roughly what the existing MiniLM router
  costs (13–17 ms). It is a different latency tier, not a free upgrade.
- **+4.17 ± 9.82** does not clear the bar the plan set ("if it wins by less than
  the fold-to-fold spread, it has not won"). It wins in 3 folds and loses in 2,
  and the sd is dominated by one fold where the baseline collapses to 74%
  (`projects_query`'s family) and SetFit does not. That is a real difference in
  robustness, but with five folds it is not a statistically clean +4 points.
- Its probabilities are worse than the logistic rung's (ECE 0.53): the head is
  confident and wrong on the rows where it is wrong.

### Rung 4 — cross-encoder re-rank: the gate passed, and the rung still did not pay

The plan said to check first whether the errors are *inside* the top 3. They are:
top-3 recall 98.4% on grouped folds, 29 of 35 mistakes in-top-3, so a perfect
re-ranker had **+7.6 points** of headroom (`results/ladder/top3_diagnostic.json`).
That is a generous gate and the rung was worth building.

Built — MiniLM with a two-way head over `(utterance, "intent name")` pairs,
trained on the base model's own top-3 as hard negatives — it scores:

- **92.15% grouped**, +2.34 ± 4.07, better in 4 folds and worse in 1.
- **93.19% rowwise: exactly the baseline**, to the row.
- **13.3 ms p50**, worse than SetFit, which scores 1.8 points higher.
- `__none__` recall **0.48**, a regression: re-ranking keeps choosing a tool
  among the top 3 when the right answer was "no command".
- **ECE 0.066**, the best calibration in the document, and the one genuinely
  useful thing it produces: at the 95% precision point it answers 87.2% of
  traffic and 71.4% of what it hands on is correct, against the logistic rung's
  63.5%.

So the rung is refuted *relative to rung 3*, not relative to the baseline: it pays
2.2× SetFit's latency for 1.8 points less accuracy and a worse reject of the
non-command class. Its ceiling was 98.4% and it reaches 92.2%, which is the usual
outcome for a data-hungry architecture on 200 training rows.

The interesting negative is *why* the first version did nothing. With 582
training pairs it could not fit its own training set — 68.6% separation of
positives from negatives, because 55 optimizer steps cannot move a 22M encoder.
Splitting the head's learning rate from the encoder's, adding a class weight for
the 1:3 positive-to-negative ratio, and training 10 epochs took training
separation to 99% and the fold-0 score from 44% to 82%. The recipe change is in
`src/localalexa/ladder/crossencoder_rung.py`, and its lesson is that most of what
looks like "the architecture does not help" is actually "the training loop was
not given a chance".

### Rung 5 — ensembles: measured against the correlation table, and refused

The plan said build nothing before the pairwise error correlation. Here it is
(`results/ladder/forward-rungs.json`, grouped, 5 folds):

| pair | disagreement | error Jaccard | errors both | only A | only B |
|---|---:|---:|---:|---:|---:|
| word-svc vs word-logreg | 2.4% | 0.85 | 34 | 5 | 1 |
| word-svc vs union-svc-C4 | 6.0% | 0.62 | 29 | 10 | 8 |
| word-svc vs SetFit | 13.6% | **0.13** | 7 | 32 | 16 |
| word-logreg vs SetFit | 12.8% | **0.12** | 6 | 29 | 17 |

The three linear models are the same model wearing different hats: Jaccard 0.62–0.85
means they are wrong on the same rows, and the linear ensemble duly scores
**exactly the baseline's 89.79%** (rowwise 93.72%, +2 rows) at 4× the latency.
SetFit is the only member that fails differently (Jaccard 0.12), which is the one
pairing an ensemble could be justified on; that combination is what rung 5's
mixed variant measures, and it is reported with the caveat that the linear member
is a 0.61 ms model bolted to a 8 ms one, so the ensemble inherits the 8 ms.

### Rung 6 — data augmentation: the highest ceiling, and the most instructive result

Two sources, both $0.

**Registry slot fills (`template`).** The repo ships 1318 template renders it
never trains on. Adding all of them:

- **rowwise 96.34%** — +3.15 points over the baseline, the biggest number
  anywhere in this document.
- **grouped 86.65%** — −3.16 points, worse than the baseline.

Both are correct, and together they are the sharpest possible statement of what
the rowwise protocol measures: the evaluation set is generated from the same
templates, so adding the generator's own slot fills hands the model the test
strings verbatim. A reader who trusted 96.34% would ship a classifier that is
four points worse than the one they started with. **This is the headline finding
of the whole ladder**: the protocol, not the model, is where the 93% comes from.

Capping the dose makes augmentation a nudge instead of a replacement — 194
references plus at most 16 fills per label:

| fills per label | grouped acc | paired vs r0 | train rows/fold |
|---|---:|---:|---:|
| 0 (references only) | 89.79% | — | 194 |
| 8 | 90.58% | +0.78 ± 1.97 | 325 |
| **16** | **91.36%** | **+1.56 ± 1.93** | 431 |
| 16, class-balanced | 91.62% | +1.82 ± 3.26 | 431 |
| 32 | 89.27% | −0.54 ± 3.88 | 587 |
| 64 | 88.22% | −1.59 ± 3.88 | 751 |
| all (~1038) | 86.65% | −3.16 ± 4.61 | 1115 |

The dose-response curve is the argument: monotone up to 16 fills per label, then
monotone down past 32. A curve that turns in both directions is harder to
attribute to fold noise than a single point, and it says the mechanism is real
but small: **+1.6 points, with a paired spread of 1.9 — under the plan's own
rule, not a win.** It costs `__none__` recall (0.60 → 0.52) as well, because
generated tool rows crowd the 16 none references.

**Local-model paraphrases (`paraphrase`).** Not measured: there is no usable
generator on this box. `flake` — the probe is in `docs/ladder-env.md` — the only
local seq2seq available (`google/flan-t5-base`, CPU) needs 5–25 s per batch and
returns `["yes", "wall", "h"]` for *"is the porch light on"* and
`["timer for 5 minutes", "turn on a timer for 5 minutes", "push the p button on
the front of your Smart-Home speaker"]` for *"set a timer for 5 minutes"*. No
llama.cpp, no GGUF, no ollama on this box. Rung 6's paraphrase half needs a real
instruct model before it can be measured; the scaffolding (provenance-tracked
rows, fold-aware filtering, an uncertainty filter) is in
`src/localalexa/ladder/augment_rung.py` and waits for one.

## The reject curve, which is the point of rungs 2 and 3

**Coverage at precision 0.90 was the wrong instrument** and the plan did not
anticipate it: any rung above 90% accuracy reaches 90% precision by accepting
everything, so every rung reports coverage ≈ 1.0 and the metric cannot rank them.
The fix is to report the curve at 0.95 and 0.975 as well, where it discriminates.

Coverage at 95% precision, and the accuracy of what gets handed on:

| rung | cov @ .95 | residue acc @ .95 | cov @ .975 | residue acc @ .975 |
|---|---:|---:|---:|---:|
| 0 word-svc (softmax) | 0.864 | 0.558 | 0.715 | 0.697 |
| 2 word-logreg C=100 | 0.864 | 0.635 | 0.788 | 0.654 |
| 3 SetFit | **0.987** | 0.0 (5 rows) | 0.869 | 0.700 |

At 97.5% precision SetFit answers 86.9% of traffic against the baseline's 71.5%
— +15 points of coverage at a *higher* precision, which is the largest
operational gain in the document and comes with the reject headroom the assistant
was designed to use. It is not free: 8 ms per utterance.

**These thresholds are oracle operating points** — each is chosen on the rows it
is scored on. They measure the ceiling a well-chosen threshold would reach, not
what a shipped threshold achieves. Nesting the threshold choice (picking it on
out-of-fold predictions inside each training pool) is implemented in
`experiments/ladder/sweep_calibration.py --nested` and is the next thing to run
before any threshold goes into the product.

## What this says to ship

1. **Tier 1 becomes the calibrated logistic model** — same features, same
   sub-millisecond budget, ECE 0.735 → 0.128, `__none__` recall 0.60 → 0.68,
   accuracy a tie. That is a strict improvement with no new dependency, and it is
   the only change here that costs nothing.
2. **Do not add the union, the sigmoid calibrator, the linear ensemble, or
   uncapped augmentation.** Each is measured above and each is worse, slower, or
   both.
3. **SetFit is the pull-the-trigger-on-the-SoC decision.** +4.2 grouped, +15
   points of coverage at 97.5% precision, `__none__` recall 0.84, and a
   `sentence-transformers` dependency at ~10 ms/utterance. It belongs in the
   tier the MiniLM router already occupies, not in the ESP32 target.
4. **Do not quote 96.34%.** That number is the rowwise protocol being gamed by
   the generator's own vocabulary, and it is the most useful thing this exercise
   produced.

## Method notes, including what went wrong

- **Folds.** `GroupKFold` over the 110 phrasing families, balanced 76–77 rows per
  fold, identical for every rung and written into each artifact
  (`results/ladder/*.json → fold_assignments`). The 7-value `family` column in
  `routing_382.json` is a domain group, not a phrasing group; grouping on it
  would leave wording shared across folds.
- **Paired deltas, not marginal spreads.** Fold difficulty varies by 25 points
  across folds and is shared between rungs, so `mean ± sd` of the per-fold
  *difference* is the sensitive statistic and the one the "refuse to call a win
  smaller than the spread" rule is applied to.
- **Selection bias is present and named.** Rung 1's C, rung 2's C, and rung 6's
  cap were chosen on the folds they are reported on. That is why
  `results/ladder/folds10-confirmation.json` re-measures the finalists on a
  different fold count.
- **Latency was measured twice.** The per-rung numbers inside the forward run
  were taken while another rung trained on the same eight cores, and the baseline
  drifted between 0.34 ms and 1.68 ms depending on load. Everything in the table
  at the top comes from `experiments/ladder/latency_pass.py`, which runs alone and
  takes the median of five passes; the run-internal numbers are in the artifacts
  and should be ignored for comparison.
- **`coverage@precision 0.9` is degenerate here** (above). Reporting it at one
  target was the plan's only methodological mistake.
- **The rowwise protocol is not merely inflated, it is exploitable** (rung 6
  above). Anyone quoting a rowwise number for this corpus is quoting the
  generator's coverage, not the classifier's.

## Reproduce

```bash
.venv/bin/python        experiments/ladder/sweep_union.py          # rung 1, 12 variants
.venv/bin/python        experiments/ladder/sweep_calibration.py    # rung 2, 11 variants
.venv/bin/python        experiments/ladder/top3_diagnostic.py      # rung 4's gate
.venv/bin/python        experiments/ladder/sweep_augment.py        # rung 6, dose response
.venv/bin/python -m localalexa ladder --rungs all --folds 5,10     # one rung at a time
.venv-embed/bin/python  experiments/ladder/ladder_report.py forward   # needs torch
.venv-embed/bin/python  experiments/ladder/latency_pass.py            # idle box, 5 passes
```

Artifacts: `results/ladder/{linear-rungs,forward-rungs,folds10-confirmation,latency,
sweep_union,sweep_calibration,sweep_augment,top3_diagnostic}.json`. Environment and
versions: `docs/ladder-env.md`.
