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
           tier 2 (ens)      300/382    78.53%   7.85  answers every row
           cascade                                    answers 92.4% itself at 81.02%,
                                                      escalates 29
             end to end                80.83%         at this split's measured tier 2
             break-even                48.28%         reachable — and it is reached
```

Rowwise is the split that flatters everything (29 of the 200 references are literal slot fills of
the scoring templates) and it is included because the repo's headline numbers were quoted there.
Family-held-out is the honest one. Strict is the hardest: its pool is template fills of the other
families only, so the model never sees a hand-written reference.

## 1. The verdict: it depends entirely on how far ahead tier 2 is

| split | tier 1 alone | tier 2 alone | cascade, end to end | break-even vs tier 2 |
|---|---:|---:|---:|---:|
| rowwise | 93.46% | **96.86%** | 90.97% | 237.5% — unreachable |
| family | 90.84% | **95.03%** | 90.28% | 173.9% — unreachable |
| strict | 68.32% | 78.53% | **80.83%** | 48.3% — reached |

Two of the three splits say the cascade is dead, and the third says it wins. The difference is not
noise, and it is not the tier-2 model — it is the **gap** between the cheap path and the expensive
one:

* where tier 2 is far ahead (rowwise: +3.4 points on everything; family: +4.2), the cascade's cheap
  path answers 94–96% of traffic *worse* than the model would have, and no tier-2 accuracy rescues
  it — the break-evens are 174% and 238%, i.e. impossible;
* where the two are close (strict: tier 1 68.32%, tier 2 78.53%), the cascade's cheap path answers
  92.4% of traffic at **81.02%** — better than the ensemble's 78.53% on everything — and escalating
  the remaining 29 rows takes it to **80.83%**, beating both single classifiers by 2.3 points for
  0.38 ms on the rows it keeps.

So the one-sentence rule this table establishes: **the cascade earns its structure exactly when its
cheap tiers are at least as accurate as the expensive model on the rows they claim.** On today's
reference pool they are not (89.97% against 95.03%), so it loses by 4.8 points. In the strict
setting, where nothing hand-written is available, the cheap path is genuinely competitive with the
ensemble and the cascade is the better design.

Two measured causes for the loss, which are also the two things to fix before shipping the cascade
anywhere:

* **The regex front costs accuracy.** On the 256 rows where exactly one pattern fires, the regex
  is 91.41% right and the logistic model is **96.88%** on the same rows. Sending those rows to the
  regex throws away 14 correct answers. The front was worth having when tier 1 was the MiniLM
  router (86.65% alone); against a model better than the patterns on their own rows, it is a
  regression.
* **The gate at 0.35 removes rows the model would have got right.** On the family split the logreg
  answers 94% of traffic at 89.97%, *below* its own 90.84% overall: the 23 rows it escalates are
  ones it mostly gets right, and they go to a tier that costs 20× per call.

## 2. Which one I would ship, and why

**Ship the ensemble as a single classifier** — 95.03% family-held-out (363/382), 96.86% rowwise,
7.85 ms p50, no regex front, no gate, no cascade. It is the most accurate thing here on the two
splits that represent deployment, it removes three moving parts and two thresholds, and its cost is
a fixed 7.85 ms rather than a distribution that depends on how often a gate fires. That is about
what the existing MiniLM router already costs.

**Keep the logreg as the fallback** for a box that cannot run an encoder: 90.84% at 0.38 ms, and it
is one model rather than three tiers.

**Do not ship the cascade as configured**, and if a cascade is wanted for latency reasons, ship it
with two changes the table already justifies: drop the regex front (it costs 14 rows) and re-derive
the gate from the ensemble's own confidence curve rather than keeping 0.35, which was chosen for a
model with different calibration.

The cascade becomes the right answer again in the world the strict row describes — where the model
has little to learn from and its edge over the cheap tier is a few points. That is an argument for
keeping the code, not for shipping it: the strict setting is not the deployment, and the right
response to it is more reference data, which the ladder measured as worth +1.6 points against the
cascade's +2.3 in a setting nobody deploys.

## 3. Why the strict row matters even though it loses the argument

Five SetFit fine-tunes on strict pools of 1000–1200 rows cost 37 minutes of CPU on this box
(`results/final/tier2-ensemble-by-split.json` records each fit). It was worth it: it is the only
split where the cascade wins, and without it the table would have produced a confident
recommendation from two splits that all point one way. The mechanism it exposes — the gap between
the tiers, not the tier-2 accuracy — is the part that will still be true when the models change.

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
* The strict split's tier-2 figure (78.53%) comes from five fine-tunes with a 37-minute total fit
  cost, and it is the noisiest number in the table for that reason — 300/382, versus 363/382 for
  the family split.
* Latencies are the ladder's idle-box measurements (0.38 ms, 7.85 ms), not re-measured here — on
  a loaded box the ensemble's per-utterance cost reads two to seven times higher.
