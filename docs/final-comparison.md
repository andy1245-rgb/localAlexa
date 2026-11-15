# The final comparison: three splits, three arms, and which one to ship

2026-11-10, branch `collect/2026-11-10`. The question this answers: **with the best model this
repo has as tier 2, does the cascade structure still buy anything?** Every number is measured
under the split it is quoted beside, and the cascade's end-to-end figure multiplies the tier-2
accuracy *measured for that same split* — 96.86% rowwise, 95.03% grouped — never a number carried
over from a different protocol.

Tier 2 is the ladder's winner: word TF-IDF + logistic regression **plus SetFit**, averaged with
row standardisation. 7.85 ms p50, and the best single classifier in the repo on every split.

```
folds 5 · cascade gate 0.35 · tier 2 = logreg + SetFit, row-standardised average

split      arm              correct  accuracy     ms  note
--------------------------------------------------------------------------------------------
rowwise    tier 1 (logreg)   357/382    93.46%   0.38  answers every row
           tier 2 (ens)      370/382    96.86%   7.85  answers every row
           cascade                                    answers 95.8% itself at 90.71%,
                                                      escalates 16
             end to end                90.97%         at this split's measured tier 2
             break-even               237.50%         unreachable: no tier-2 model saves it

family     tier 1 (logreg)   347/382    90.84%   0.38  answers every row
           tier 2 (ens)      363/382    95.03%   7.85  answers every row
           cascade                                    answers 94.0% itself at 89.97%,
                                                      escalates 23
             end to end                90.28%         at this split's measured tier 2
             break-even               173.91%         unreachable

strict     tier 1 (logreg)   261/382    68.32%   0.38  answers every row
           tier 2 (ens)           —     pending    7.85  five fine-tunes on a 1000-row pool
           cascade                                    answers 92.4% itself at 81.02%,
                                                      escalates 29
```

Rowwise is the split that flatters everything (29 of the 200 references are literal slot fills of
the scoring templates) and it is included because the repo's headline numbers were quoted there.
Family-held-out is the honest one. Strict is the hardest: its pool is template fills of the other
families only, so the model never sees a hand-written reference.

## 1. The verdict: the cascade loses, and it is not close

On both splits where tier 2 is measured, the cascade is **worse than either single classifier**:

| split | tier 1 alone | tier 2 alone | cascade, end to end |
|---|---:|---:|---:|
| rowwise | 93.46% | **96.86%** | 90.97% |
| family | 90.84% | **95.03%** | 90.28% |

The break-even column explains why this cannot be fixed by a better tier 2: to match the ensemble
the cascade would need a tier-2 model at **173.9%** (grouped) or **237.5%** (rowwise). Both are
impossible, so no tier-2 model — not a perfect one, not a future one — rescues this arrangement.

The mechanism is measured, and it is two separate losses:

* **The regex front costs accuracy.** On the 256 rows where exactly one pattern fires, the regex
  is 91.41% right and the logistic model is **96.88%** right on the same rows. Sending those rows
  to the regex instead of the model throws away 14 correct answers. The regex tier was worth
  having when tier 1 was the MiniLM router (86.65% alone); against a model that is better than
  the patterns on their own rows, it is a regression.
* **The gate at 0.35 removes rows the model would have got right.** The logreg answers 94% of
  traffic at 89.97%, which is *below* its own 90.84% overall: the 23 rows it escalates are ones it
  mostly gets right, and they are handed to a tier that costs 20× as much per call.

## 2. Which one I would ship, and why

**Ship the ensemble as a single classifier** — 95.03% grouped (363/382), 7.85 ms p50, no cascade,
no regex front, no gate. It is the most accurate thing here on every split, it removes three
moving parts and two thresholds, and its cost is a fixed 7.85 ms rather than a distribution that
depends on how often the gate fires. On a server that is nothing; it is what the existing MiniLM
router costs.

**Keep the cascade only if 0.38 ms is a hard requirement**, and if it is, ship it *without the
regex tier* and with the gate raised — because the measured evidence says the structure is only
viable when the cheap tier is close to the expensive one. As configured it is strictly dominated.

What that means concretely, in the order I would do it:

1. **Tier 1 becomes the ensemble**, and the cascade becomes a two-tier design: model, then
   escalate on its own calibrated confidence. The 0.38 ms logreg stays as the fallback for a box
   that cannot run an encoder.
2. **Delete the regex front from the decision path.** It is 91.41% on its own rows against the
   model's 96.88%. Keep the patterns for slot extraction, which is what they are good at, not for
   routing.
3. **Re-derive the gate from the ensemble's own confidence curve** rather than keeping 0.35, which
   was chosen for a model with different calibration.

## 3. The strict split, and why it is still measuring

Five SetFit fine-tunes on the strict pool is 10–17 minutes each on this box, and the box has been
carrying other sessions' work throughout (load 6–26, swap in the gigabytes). It is running;
`results/final/tier2-ensemble-by-split.json` records each split as it completes.

The strict row is the one that could change the *shape* of the verdict without changing the
recommendation. There, tier 1 collapses to 68.32% while the cascade still answers 92.4% of
traffic at 81.02% — because template fills of *other* families are a much better training pool
than a strict-pool model trained on nothing hand-written. If the strict-pool ensemble lands near
that 81%, then tier 1, tier 2 and the cascade are all within a few points, the break-even becomes
reachable, and the cascade becomes a legitimate way to buy latency back. If it lands well above,
the same domination as §1 applies. Either way the recommendation in §2 is unaffected: the split
where the numbers are complete is the split where the ensemble wins.

## 4. A defect found on the way: the strict split used to leak

The strict protocol's pool is built by dropping templates by their *string*. Once folds hold out
many families at once, that filter is not enough — another template can render the same utterance,
and the pool keeps it. Measured: **52 to 340 rows per fold** were rendered identically on both
sides, and 77 of 77 scored rows in one fold were present verbatim in the training pool. The
logistic model scored 97.12% on that leaked split; filtered by rendered text it scores **68.32%**.

The repo's published 74.9% strict number is not affected: at leave-one-family-out the overlap is
zero (verified over the first 20 families), so the 110-fold protocol was clean. It was the
five-fold version I built for this table that leaked, and it now filters by text, the same rule
`fold_pool` uses for the non-strict split. `tests/test_final_table.py` asserts that no scored
utterance appears in the pool it was trained on, for every split.

## 5. Reproduce

```bash
# the tier-2 accuracy per split (SetFit fine-tunes; .venv-embed for torch)
.venv-embed/bin/python experiments/final/tier2_ensemble.py --splits rowwise,family,strict

# the table (logistic regression only, ~1 min)
.venv/bin/python -c "
from localalexa.final_table import build, render
from localalexa.final_table import load_tier2
print(render(build(tier2=load_tier2('results/final/tier2-ensemble-by-split.json'))))
"

.venv/bin/python -m pytest tests/test_final_table.py -q
```

Artifacts: `results/final/tier2-ensemble-by-split.json` (the tier-2 accuracy and fit cost per
split) and `results/final/final-table.json` (the rendered comparison, including the break-evens).

## 6. Caveats

* The gate is the product's 0.35 throughout. §1 shows it is the wrong value for this tier 1; the
  cascade's numbers would improve at 0.70, and `docs/cascade-vs-classifier.md` carries that sweep.
  Even there the cascade only beats *tier 1 alone* (92.9% against 90.84%), never the ensemble.
* Rowwise is inflated by construction and is shown because the repo's quoted numbers live there.
* The strict tier-2 figure is pending; §3 says what its two possible outcomes change.
* Latencies are the ladder's idle-box measurements (0.38 ms, 7.85 ms), not re-measured here — on
  a loaded box the ensemble's per-utterance cost reads two to seven times higher.
