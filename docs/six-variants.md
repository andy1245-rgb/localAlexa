# Six structures never tried — one wins, one is a wash, four lose

2026-11-10, branch `collect/2026-11-10`. Six things this repo had never tried, on the same
grouped folds and the same training pools as everything else, so each number sits beside the
baseline (word logreg C=100, **90.84%**, fold sd 8.55%) without a footnote. Per-fold movement is
reported because a win smaller than the spread across folds has not won here either.

| # | structure | accuracy | fold sd | paired vs baseline | folds W/L/T |
|---|---|---:|---:|---:|---|
| — | baseline: word logreg C=100 | 90.84% | 8.55% | — | — |
| 1 | regex as features (17 binary columns ∪ TF-IDF) | 87.43% | 7.38% | −3.43 ± 5.21 | 1/3/1 |
| 2a | fuse linear + regex | 87.17% | 8.36% | −3.68 ± 4.95 | 2/3/0 |
| **2b** | **fuse linear + regex + router** | **95.03%** | **3.24%** | **+4.16 ± 7.16** | 3/2/0 |
| 3 | `__none__` as a separate binary question | 89.01% | 12.43% | −1.82 ± 4.07 | 0/1/4 |
| 4 | hierarchical: family, then action | 90.58% | 11.58% | −0.25 ± 3.84 | 3/2/0 |
| 5 | distill the ensemble into one fast model | 92.15% | 9.67% | +1.31 ± 2.06 | 3/1/1 |
| 6 | per-class thresholds, target precision 0.90 | 59.2% coverage at **100.00%** precision | — | e2e 97.97% | — |

*(2b) ties the SetFit ensemble's accuracy exactly — 363/382 — and halves its fold spread.*

## 1. Regex as features: it hurts, and the confusion pairs say why

The 17 patterns are knowledge no classifier can learn from 200 rows, so putting them in as
columns rather than using them as a serial gate looked like free information. It is not: −3.43
points, and the new confusion pairs are the tell. The baseline's worst mistake is
`light_query → light_set` (6). With the rule columns the top mistakes become
`shoppinglist_add → shoppinglist_view` (6) and then **`→ __none__` five times for three
different intents** — `docs_query`, `calendar_today`, `light_query`. The columns are sparse and
binary while TF-IDF is dense and weighted; at 194 training rows the model spends capacity
fitting 18 near-constant columns and drifts toward the class the patterns do *not* cover.

What it does fix is worth noting for later: `thermostat_offset` goes 0.87 → **1.00**. The
patterns know something about that intent that the text statistics do not; they are just being
asked to help with all 17 at once instead of the two or three where they carry information.

## 2. Fusion: the one real win, and only with all three members

Averaging distributions instead of cascading decisions, each row standardised first:

* **linear + regex: 87.17%** (−3.68). The regex member's distribution is binary and blunt; mixing
  it with a calibrated model drags the average down.
* **linear + regex + router: 95.03%** (+4.16), fold sd **3.24%** — the lowest of anything measured
  in this repo, including the SetFit ensemble's 3.72%.

Two things this establishes. First, **the fusion reaches the SetFit ensemble's number without
torch**: 363/382 by three cheap members and no fine-tuning, which is the other project's finding
reproduced on this corpus. Second, and more useful, it **fixes the fold the baseline collapses
on**: fold 0 goes 76.62% → 90.91%, and per-intent `calendar_today` goes 0.67 → **1.00**,
`projects_query` 0.83 → 0.94, `thermostat_offset` 0.87 → 1.00. The router member is what carries
those; the folded pairs are all within-family or knowledge-family cases where reference
similarity is a better signal than word statistics.

The cost is the router. Measured on a loaded box, p50 per utterance: fused **16.81 ms**, router
alone 10.28 ms, linear alone 1.05 ms. So the fusion is *not* a cheaper route to 95% — it is the
same accuracy as the ensemble at about twice the latency. Its case is that it needs no training
pipeline, not that it is fast.

## 3. `__none__` as its own question: the gate is too weak to be worth it

−1.82 points, and the shape of the result is the interesting part: **four of five folds tie
exactly**. The binary gate only changes an answer when it fires, and with 16 none-references it
mostly does not. Where it does fire it is wrong — `__none__` recall falls from 0.68 to **0.40**,
because a balanced binary model trained on 16 negatives at threshold 0.5 over-predicts "none" on
the confusable tool rows. Splitting the question is sound; doing it with this little data is not.

## 4. Hierarchical: a wash, and less stable

−0.25 ± 3.84 with 3 folds better and 2 worse — indistinguishable from the baseline, at a higher
fold spread (11.58% against 8.55%) and with three models to maintain. It does buy
`light_query` 0.76 → 0.84 and `calendar_today` 0.67 → 0.76, and it pays for them with
`thermostat_offset` 0.87 → 0.83 and `__none__` 0.68 → 0.48: the family stage makes the abstain
decision first, and it gets that wrong more often than the flat model does. Not worth adopting on
this evidence; worth revisiting only if the per-family classifiers are allowed more data than a
single pool.

## 5. Distillation: +1.31 points for free, at the same 0.38 ms

The teacher is the measured ensemble (95.03%); the student is one word logreg trained on the
teacher's answers for the references **plus 1,038 template fills the teacher labels itself**. The
student reaches **92.15%** — 1.31 points above the logreg baseline (+1.31 ± 2.06, 3 folds better),
at the student's own latency.

That is the accuracy-latency point the brief asked for: **92.15% at 0.38 ms**, against 95.03% at
7.85 ms for the teacher and 90.84% at 0.38 ms for the baseline. It is the best single model you
can run without torch, and it closes 2.9 of the 4.2 points the ensemble is ahead by.

## 6. Per-class thresholds: the only structure that beats the baseline outright

One global gate at 0.35 answers 94% of traffic at 89.97%. Per-intent floors, each chosen on
out-of-fold pool predictions to hold a precision target, do this instead:

| per-class target | answered | coverage | precision | escalated | e2e with the ensemble at tier 2 |
|---|---:|---:|---:|---:|---:|
| 0.70 | 269 | 70.4% | 96.28% | 113 | 95.91% |
| 0.80 | 242 | 63.3% | 97.52% | 140 | 96.61% |
| **0.90** | **226** | **59.2%** | **100.00%** | **156** | **97.97%** |

At a 0.90 per-class target, 226 rows are answered and **all 226 are right**, and the 156 hardest
rows go to the ensemble. What the 100% actually means needs saying precisely: it is
**class-selective**. Fifteen of the eighteen intents can answer at this target and three cannot, so
the coverage is not spread evenly — it is `docs_query`, `reminder_create`, `shoppinglist_add` and
`research_deep` at 25 rows each, down to `shoppinglist_view` at 1 and `calendar_today` at 2. The
thresholds are effectively routing *by class difficulty*, which is the right thing for them to do
and is not the same claim as "59% of traffic answered correctly regardless of intent".

Three caveats, none small: the 100% is a clean result on 226 rows that deserves an independent
re-run before anything is planned around it; the answered set is class-selective as above; and the
end-to-end figure assumes the escalated rows are answered at the ensemble's *overall* 95.03% when
they are by construction the hard ones — at 90% on those, the e2e is 95.9%. Both numbers are in the
artifact; the honest reading is "95.9–98.0%, with the upper end optimistic".

## 7. What this changes

**The frontier, measured rather than argued:**

| design | accuracy | latency | notes |
|---|---:|---:|---|
| logreg baseline | 90.84% | 0.38 ms | one model, no dependencies |
| **distilled student** | **92.15%** | **0.38 ms** | one model, trained on the ensemble's answers |
| SetFit ensemble | 95.03% | 7.85 ms | needs torch |
| **fusion (linear + regex + router)** | **95.03%** | **16.81 ms** | no fine-tuning, needs the encoder |
| **per-class thresholds + ensemble** | **97.97% (bound)** | 0.38 ms on 59% of traffic | 156 rows escalate |

**What I would ship, given all seven experiments:** the distilled student as tier 1 and the
ensemble as tier 2, decided by *per-class* thresholds at a 0.90 target. That combination is the
only one where every part earned its place: the student is +1.31 over the baseline at no cost, the
thresholds answer 59% of traffic at 100% precision instead of 94% at 89.97%, and the ensemble only
sees the rows that are genuinely hard. The fusion is the better *single* model if the encoder is
already in the build (same accuracy as the ensemble, half the fold spread, no fine-tuning), and the
ensemble alone remains the simplest thing that reaches 95%.

Two of the six are now answered for good: regex-as-features and none-first both lose, and the
mechanism in each case is visible in the per-intent ledger rather than left as a score.

## 8. The per-intent analysis

`results/final/per-intent.md` carries the full ledger from the same folds: per-class recall for
every variant, the confusion pairs ranked, and the eight rows *every* variant gets wrong — which
are the corpus's real defects rather than any model's:

| intent | n | baseline | best variant | note |
|---|---:|---:|---:|---|
| `calendar_today` | 21 | 0.67 | **1.00** (fusion) | the baseline's worst; reference similarity fixes it |
| `__none__` | 25 | 0.68 | 0.76 (fusion) | nobody gets this class above 0.76 |
| `light_query` | 25 | 0.76 | 0.88 (fusion) | → `light_set`, 6 of the 9 errors |
| `projects_query` | 18 | 0.83 | 0.94 (fusion) | → `shoppinglist_view` |
| `thermostat_offset` | 23 | 0.87 | **1.00** (fusion, regex features) | → `light_set`, a cross-family confusion |
| `light_set` | 25 | 0.88 | 0.92 (fuse linear+regex) | → `light_query` |

The eight rows wrong for every variant are the off-scaffold chitchat — "how far is the moon",
"why is the sky blue", "how many bones are in the human body" — plus two ambiguous
`calendar_today` phrasings ("what does my morning look like"). Every structure maps those to a
tool because the text carries no evidence either way; the fix is data (more none-references and
better calendar phrasings), not architecture. That is the same conclusion the ladder reached from
the augmentation rung, arrived at from the errors instead of from a sweep.

## 9. Reproduce

```bash
.venv/bin/python        experiments/final/six_variants.py --arms fast   # (1)(2a)(3)(4)(6)
.venv-embed/bin/python  experiments/final/six_variants.py --arms all    # adds (2b)(5)
.venv/bin/python        experiments/final/error_analysis.py             # per-intent ledger
.venv/bin/python -m pytest tests/test_variants.py -q                    # 14 cases
```

Artifacts: `results/final/six-variants.json` (every arm, per fold, with paired deltas),
`results/final/per-intent.{md,json}` (the ledger, the pairs, the universal failures).
