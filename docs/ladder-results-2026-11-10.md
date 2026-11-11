# Ladder results — 20× the latency buys +5.2 points; 0.61 ms buys nothing

2026-11-10, branch `ladder/rungs-1-6`. Every rung in
`docs/classifier-ladder-2026-11-10.md` was built and measured, in order, on one
fold assignment and one metrics block. `cascade.py` semantics and
`SMART_ASSUMED_ACC` were not touched.

## The answer

**Nothing beat 93.19% at 0.61 ms.** One rung beat it — by one row, which is not a
win. The rung that beats 93.19% meaningfully needs 7.8 ms, 20× the tier-1 budget:

| rung | what it is | grouped acc | paired vs r0 | rowwise acc | p50 ms | `__none__` | ECE |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | word TF-IDF + LinearSVC — **the quoted model** | 89.79% | — | **93.19%** | **0.39** | 0.60 | 0.735 |
| 1 | + `char_wb` union, C=1 | 89.53% | −0.26 ± 2.51 | 92.41% | 0.98 | 0.52 | 0.697 |
| 1 | union, C=4 (best of twelve swept) | 90.31% | +0.52 ± 1.49 | 92.67% | 0.96 | 0.52 | 0.671 |
| 2 | word TF-IDF + LogisticRegression C=100 | 90.84% | +1.04 ± 1.43 | 93.46% | **0.38** | 0.68 | **0.128** |
| 2 | word + LinearSVC, sigmoid-calibrated | 87.17% | −2.61 ± 3.07 | 91.88% | 2.63 | 0.64 | 0.310 |
| 3 | SetFit (MiniLM contrastive + logistic head) | 93.98% | +4.17 ± 9.82 | 94.24% | 7.75 | 0.84 | 0.530 |
| 4 | MiniLM cross-encoder re-ranks top 3 | 92.15% | +2.34 ± 4.07 | 93.19% | 11.79 | 0.48 | 0.066 |
| 5 | linear-only ensemble | 89.79% | −0.00 ± 0.92 | 93.72% | 3.34 | 0.64 | 0.507 |
| 5 | **ensemble: logreg + SetFit, row-standardised** | **95.03%** | **+5.21 ± 5.87** | **96.86%** | 7.85 | **0.88** | 0.226 |
| 6 | + every registry slot fill (uncapped) | 86.65% | −3.16 ± 4.61 | 96.34% | 0.78 | 0.44 | 0.068 |
| 6 | + at most 16 slot fills per label | 91.36% | +1.56 ± 1.93 | — | 0.87 | 0.52 | 0.123 |

Latency is one utterance, warm, idle box, median of five passes
(`results/ladder/latency.json`). The baseline re-measures at **0.387 ms**, not the
0.61 ms in the README — that number predates this machine state and the honest
move is to quote both. Every rung's relative cost is what matters here.

Read the two accuracy columns as different questions. **Rowwise** is the protocol
the 93.19% claim was made under, so it is the only fair comparison to it, and it
is inflated. **Grouped** holds whole phrasing families out and is the number that
survives rephrasing.

Three sentences that carry the result:

1. **At the 0.61 ms budget, the ladder produces no accuracy win.** Rung 2 ties
   (93.46%, one row) and its real contribution is calibration, not accuracy.
2. **At 7.8 ms, the mixed ensemble wins clearly**: +5.21 grouped, +3.67 rowwise,
   `__none__` recall 0.60 → 0.88, and every one of the five folds improved.
3. **The 96.34% at the bottom of the table is a trap**, not a result — see rung 6.

## Rung by rung, and why

### Rung 1 — vectoriser union: refuted, and it was the plan's most confident prediction

The plan expected +1–3 points because short imperatives are exactly where
character n-grams should help. On 110 held-out families the union as written is
**0.26 points worse**, and the best of twelve variants swept
(`experiments/ladder/sweep_union.py`) is **+0.52 ± 1.49** — a non-win by the
plan's own rule — for **2.5× the latency**. Character n-grams alone: 87.96%
(−1.83). The union rescales the decision values, so C needed re-tuning; with C=4
it recovers the baseline and stops there.

Why it fails: 200 references already yield ~1300 word features, and word 1–2
grams over 3–6 word utterances nearly describe the whole string. `char_wb` adds
30k features that split the same evidence finer, and at 22 examples per intent
that extra dimension is noise the linear model has to fit.

### Rung 2 — calibration: the accuracy is a tie, the probabilities are not

Half right. `CalibratedClassifierCV(method="sigmoid")` on 200 rows is a loss —
each calibrator sees 133 rows, accuracy drops 2.6 points, latency quadruples.
Plain `LogisticRegression` on the baseline's own features is better *and* cheaper:

- **ECE 0.735 → 0.128.** The old column was never a probability: softmax over
  hinge decision values is a ranking dressed as confidence, wrong by 73 points on
  average. Anything downstream that thresholds it — the cascade, the smart tier's
  cost model — was reading a number that did not mean what its name said.
- **`__none__` recall 0.60 → 0.68** at an unchanged 0.8% false-none rate.
- **0.379 ms vs the baseline's 0.387 ms**: free. Grouped +1.04 ± 1.43, rowwise +1
  row.

This is the only change in the whole exercise that costs nothing and improves
something real.

### Rung 3 — SetFit: the first rung that wins, at a price

Contrastive fine-tuning of a 22M encoder on pairs from the labelled set, then a
logistic head. Best single model measured:

- **93.98% grouped**, +4.17 points, and the per-fold spread halves (sd 4.5 vs
  9.3 points) — it is *more stable across phrasings*, not just better on average.
- **`__none__` recall 0.60 → 0.84**: 4 of 25 non-commands acted on instead of 10.
- rowwise **94.24%**, +1.05 over 93.19%.
- **Replicated on a different fold count**: 93.98% and +4.41 ± 11.97 at 10 folds
  (`results/ladder/folds10-confirmation.json`). Two independent fold
  assignments, the same 359/382 rows. This is the most reliable number in the
  document.

Against it: **7.75 ms**, 20× the baseline and about what the existing MiniLM
router costs (13–17 ms) — a different latency tier, not an upgrade. Its
probabilities are poor (ECE 0.53): confident, and wrong where it is wrong.
Its mean win is also smaller than its fold-to-fold spread (+4.17 vs 9.82), which
by the plan's rule means it has not won; the honest reading is that the *size* is
uncertain while the *direction* is not (3 folds up, 2 down, and the two losses are
the folds where the baseline was already at 91–99%).

### Rung 4 — cross-encoder re-rank: the gate passed, and the rung still did not pay

The gate was checked first, as the plan required: top-3 recall is 98.4%, 29 of 35
mistakes are in-top-3, so a perfect re-ranker had **+7.6 points** of headroom
(`results/ladder/top3_diagnostic.json`). Generous gate; worth building.

Built — MiniLM with a two-way head over `(utterance, "intent name")` pairs,
trained on the base model's own top-3 as hard negatives — it scores:

- **92.15% grouped**, +2.34 ± 4.07, better in 4 folds of 5.
- **93.19% rowwise: exactly the baseline, to the row.**
- **11.79 ms**, 1.5× SetFit, which scores 1.8 points higher.
- **`__none__` recall 0.48** — a regression: asked to choose among three tools, it
  keeps choosing one when the right answer was "no command".
- **ECE 0.066**, the best calibration here, and the one useful artefact: at 95%
  precision it answers 87.2% of traffic and 71.4% of what it hands on is correct.

Refuted relative to rung 3, not to the baseline: 1.8 points less accuracy for 1.5×
the latency. Its ceiling was 98.4% and it reaches 92.2%, the usual outcome for a
data-hungry architecture on 200 rows.

The instructive part is why the first version did nothing. With 582 pairs it could
not fit its own training set — 68.6% separation of positives from negatives,
because 55 optimizer steps cannot move a 22M encoder. A separate learning rate for
the head, a class weight for the 1:3 positive:negative ratio, and 10 epochs took
training separation to 99% and fold 0 from 44% to 82%. Most of what looks like
"the architecture does not help" is "the training loop was not given a chance".

### Rung 5 — ensembles: the correlation table said one pairing was worth it, and it was

The plan said build nothing before the pairwise error correlation. Here it is
(grouped, 5 folds; full table in `results/ladder/rung5-ensemble.json`):

| pair | disagreement | error Jaccard | errors both | only A | only B |
|---|---:|---:|---:|---:|---:|
| word-svc vs word-logreg | 2.4% | 0.85 | 34 | 5 | 1 |
| word-svc vs union-svc-C4 | 6.0% | 0.62 | 29 | 10 | 8 |
| word-svc vs SetFit | 13.6% | **0.13** | 7 | 32 | 16 |
| logreg vs SetFit | 12.8% | **0.12** | 6 | 29 | 17 |

The three linear models are one model in three hats: Jaccard 0.62–0.85, wrong on
the same rows. Averaging them gives **exactly the baseline's accuracy** (89.79%
grouped, 93.72% rowwise) at 8× the latency — the plan's prediction, confirmed to
the decimal.

SetFit is the only member that fails differently, and the pairing the table argues
for is the one that wins. `StandardisedEnsemble` — a row-standardised average of
word-logreg and SetFit, because a probability average would let the sharp linear
model drown the one whose errors are worth having — scores:

- **95.03% grouped**, +5.21 points, and it improves in **5 of 5 folds**
  (+1.3, +3.9, +2.6, +1.3, +15.6); a 5/5 sign test is p = 0.03. The sd of the
  deltas is still 5.87 because the *size* of the win varies with how bad the
  baseline is in that fold; the direction does not. By the plan's rule the size
  has not won; by any reasonable reading of the direction, it has.
- **96.86% rowwise**, +3.67 over the quoted 93.19%.
- **`__none__` recall 0.88**, false-none 0.84% — the best reject behaviour of any
  rung, and the two rows that improved most are exactly the confusable pair
  (`light_query`/`light_set`, 0.84 each against 0.80/0.92 for the baseline).
- Per-fold sd **3.7 points**, the lowest here (baseline 9.3).
- **7.85 ms p50**, essentially SetFit's latency plus nothing: the linear member is
  free next to the encoder.

This is the rung to adopt if, and only if, tier 1 is allowed to cost what tier 1
already costs in the MiniLM router.

### Rung 6 — data augmentation: the highest ceiling, and the most instructive result

Two sources, both $0.

**Registry slot fills.** The repo ships 1318 template renders it never trains on.
Adding all of them:

- **rowwise 96.34%** — +3.15 over the baseline, the biggest number in this
  document.
- **grouped 86.65%** — −3.16, worse than the baseline.

Both are correct, and together they are the sharpest available statement of what
the rowwise protocol measures: the evaluation set is generated from the same
templates, so adding the generator's own slot fills hands the model the test
strings verbatim. A reader who trusted 96.34% would ship a classifier four points
worse than the one they started with. **This is the most useful finding in the
ladder**: at this corpus's scale the 93% comes from the protocol, not the model,
and any rowwise number here is a measure of template coverage.

Capping the dose turns augmentation into a nudge instead of a replacement:

| fills per label | grouped acc | paired vs r0 | train rows/fold |
|---|---:|---:|---:|
| 0 (references only) | 89.79% | — | 194 |
| 8 | 90.58% | +0.78 ± 1.97 | 325 |
| **16** | **91.36%** | **+1.56 ± 1.93** | 431 |
| 16, class-balanced | 91.62% | +1.82 ± 3.26 | 431 |
| 32 | 89.27% | −0.54 ± 3.88 | 587 |
| 64 | 88.22% | −1.59 ± 3.88 | 751 |
| all (~1038) | 86.65% | −3.16 ± 4.61 | 1115 |

The dose–response curve is the argument: monotone up to 16 fills per label, then
monotone down past 32. A curve that turns in both directions is harder to
attribute to fold noise than a single point, so the mechanism is real — but small:
**+1.6 points against a 1.9-point paired spread.** Under the plan's rule, not a
win. It also costs `__none__` recall (0.60 → 0.52): generated tool rows crowd the
16 none references, which is the same mechanism that makes the uncapped version
fail.

**Local-model paraphrases.** Not measured: there is no usable generator on this
box. `google/flan-t5-base` on CPU needs 5–25 s per batch and returns
`["yes", "wall", "h"]` for *"is the porch light on"* and a description of a
smart speaker's button for *"set a timer for 5 minutes"* (full probe in
`docs/ladder-env.md`). No llama.cpp, no ollama, no GGUF files here. So rung 6's
paraphrase half is **unmeasured, with the reason** — not a negative result about
augmentation. The machinery is built and fold-aware
(`src/localalexa/ladder/augment_rung.py`): every generated row carries the
reference it came from, a fold drops the paraphrases of any reference it dropped,
and the intended filter is "keep only rows the current model is unsure about". It
needs an instruct model and a human spot-check of a 40-row sample — a step no
model can substitute for.

## The reject curve, which is what rungs 2 and 3 exist for

**Coverage at precision 0.90 was the wrong instrument**, and the plan did not
anticipate it: any rung above 90% accuracy reaches 90% precision by accepting
everything, so every rung reports coverage ≈ 1.0 and the metric cannot rank them.
Reporting the curve at 0.95 and 0.975 as well is the fix.

| rung | cov @ .95 | residue acc @ .95 | cov @ .975 | residue acc @ .975 |
|---|---:|---:|---:|---:|
| 0 word-svc (softmax) | 0.864 | 0.558 | 0.715 | 0.697 |
| 2 word-logreg C=100 | 0.864 | 0.635 | 0.788 | 0.654 |
| 3 SetFit | 0.987 | — (5 rows) | 0.869 | 0.700 |
| 4 cross-encoder | 0.872 | 0.714 | 0.531 | 0.860 |
| 5 ensemble z | **1.000** | — (0 rows) | **0.919** | 0.645 |

At 97.5% precision the ensemble answers 91.9% of traffic where the baseline
answers 71.5% — **+20 points of coverage at a higher precision**, which is the
largest operational gain in the document, and it is the gain the cascade was
built to exploit. SetFit alone gets +15.

**These thresholds are oracle operating points** — each is chosen on the rows it
is scored on. They measure the ceiling a well-chosen threshold would reach, not
what a shipped threshold achieves. Nested selection (picking the floor on
out-of-fold predictions inside each training pool) is implemented in
`experiments/ladder/sweep_calibration.py --nested` and is the next thing to run
before any threshold enters the product.

## What this says to ship

1. **Tier 1 becomes the calibrated logistic model.** Same features, same
   0.38 ms, ECE 0.735 → 0.128, `__none__` recall 0.60 → 0.68, accuracy a tie. No
   new dependency, no new failure mode. This is the only free win and it is the
   one to take tonight.
2. **If tier 1 may cost what the MiniLM router already costs, ship the ensemble.**
   logreg + SetFit, row-standardised: +5.21 grouped, +3.67 rowwise, `__none__`
   0.88, +20 points of 97.5%-precision coverage, 7.85 ms. It improves in every
   fold. This is the answer to "is there more in localAlexa than one linear
   model?" — yes, and it lives in the encoder tier.
3. **Do not add** the union, the sigmoid calibrator, the linear-only ensemble, the
   cross-encoder, or uncapped augmentation. Each is measured above and each is
   worse, slower, or both.
4. **Do not quote 96.34%.** That is the rowwise protocol fed the generator's own
   vocabulary.

## Method notes, including what went wrong

- **Folds.** `GroupKFold` over the 110 phrasing families, 76–77 rows per fold,
  identical for every rung and written into every artifact
  (`results/ladder/*.json → fold_assignments`). The 7-value `family` column in
  `routing_382.json` is a *domain* group, not a phrasing group; grouping on it
  shares wording across folds. Confirmed the hard way — see the harness note.
- **Paired deltas, not marginal spreads.** Fold difficulty varies by 25 points and
  is shared between rungs, so `mean ± sd` of the per-fold *difference* is the
  sensitive statistic. Where the direction is unanimous (rung 5: 5/5 folds) and
  where it is not (rung 3: 3/5) is reported explicitly rather than reduced to one
  number.
- **Selection bias is present and named.** Rung 1's C, rung 2's C and rung 6's cap
  were chosen on the folds they are reported on.
  `results/ladder/folds10-confirmation.json` re-measures the finalists on a
  different fold count: rung 2 +0.52 (vs +1.04), SetFit +4.41 (vs +4.17). Both
  hold.
- **Latency was measured twice.** The per-rung numbers inside the forward runs
  were taken while another rung trained on all eight cores; the baseline drifted
  between 0.34 and 1.68 ms depending on load. The table uses
  `experiments/ladder/latency_pass.py`, which runs alone, repeats, and takes the
  median of five passes. Ignore the run-internal latency fields in the artifacts
  for comparisons.
- **`coverage@precision 0.9` is degenerate on this corpus** (above). Reporting it
  at one target was the plan's only methodological mistake.
- **The rowwise protocol is not merely inflated, it is exploitable** (rung 6).
  Anyone quoting a rowwise number for this corpus is quoting the generator's
  coverage, not the classifier's.

## The harness port, which is blocked on one line

`~/toolInterfacer/bench` already ships a `localalexa_intents` dataset adapter, so
the port looked like a spec string. It is not:

```
$ cd ~/toolInterfacer/bench && python3 -m harness.run \
    --dataset localalexa_intents --cv 5 \
    --arms "r0=sklearn_text|vectorizer=tfidf_word|clf=linsvc;r2=...|clf=logreg" --reference r0
grouped CV  r0: 5 folds over 7 groups / 18 labels, stratified by label
  seed 0: out-of-fold 0/382 = 0.0%
    WARNING fold(s) [0,1,2,3,4] hold out a label the training side never sees: ...
```

`harness/rows.py::localalexa_intents` sets `group = r["family"]`, the seven-value
domain group. Five folds over seven domain groups withhold a whole domain each, so
every fold contains intents the training side has never seen and nothing is
predictable — 0/382 out-of-fold, which is the metric showing the *grouping* is
wrong rather than the arm. The adapter needs the phrasing family (110 groups),
derived the way `localalexa.data.phrasing_families` does it.

I did not patch that repo: it is on another branch (`overnight/2026-11-10`) and
the plan says not to block on the harness. For the port, rung 2 is expressible
today as `sklearn_text|vectorizer=tfidf_word|clf=logreg` (plus a `c=` spec key for
C=100); rung 3 needs a `setfit_text` predictor the harness does not have.

## Reproduce

```bash
.venv/bin/python        experiments/ladder/sweep_union.py          # rung 1, 12 variants
.venv/bin/python        experiments/ladder/sweep_calibration.py    # rung 2, 11 variants
.venv/bin/python        experiments/ladder/top3_diagnostic.py      # rung 4's gate
.venv/bin/python        experiments/ladder/sweep_augment.py        # rung 6, dose response
.venv/bin/python -m localalexa ladder --rungs all --folds 5        # one rung at a time
.venv-embed/bin/python  experiments/ladder/ladder_report.py rung5      # needs torch
.venv-embed/bin/python  experiments/ladder/ladder_report.py confirm   # 10-fold re-check
.venv-embed/bin/python  experiments/ladder/latency_pass.py            # idle box, 5 passes
```

Artifacts: `results/ladder/{linear-rungs,rung4-crossencoder,rung5-ensemble,
folds10-confirmation,latency,sweep_union,sweep_calibration,sweep_augment,
top3_diagnostic}.json`. Environment, pinned versions and the generator probe:
`docs/ladder-env.md`.
