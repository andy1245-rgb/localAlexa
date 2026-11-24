# Phase 4 — the honest sweep

Generated 2026-11-24T04:18:31-05:00 by `experiments/survey/report_phase4.py` from 4 sweep artefact(s). Selection is folds [0, 1, 2] (the selection half); the answer is quoted on folds [3, 4] (the validation half), which selected nothing.

## Verdict

**the ranking does not transfer**

the ranking does not transfer — the selection half picks sentence-transformers/all-MiniLM-L6-v2 (0.9391) and the validation half points at Snowflake/snowflake-arctic-embed-xs (0.9868). Recommend the incumbent for stability; do not crown a model on the rows that would have selected it.

| number | value |
|---|---|
| selection-half winner (ranked on folds [0, 1, 2] only) | `sentence-transformers/all-MiniLM-L6-v2` |
| winner's selection-half pooled grouped accuracy | 0.9391 |
| winner's validation-half pooled grouped accuracy | 0.9408 |
| incumbent's validation-half pooled grouped accuracy | 0.9408 |
| winner's validation-half paired delta vs incumbent | +0.00 ± 0.00 points (folds [3, 4], improved 0/2) |
| |mean| > sd on the validation half | no (0.00 vs 0.00) |
| sign consistent across validation folds | no |

**Not a selection criterion:** pooled-across-all-folds accuracy (the `grouped pooled` and `rowwise pooled` columns below) entered none of these decisions. It is the number the ladder's rung 6 showed measures template coverage, not a classifier, and the same rows that selected the winner would be scoring it.

## The three forms, and which one fired

Checked in this order, and the first that fires is the verdict: (1) the selection-half winner clears its own validation spread with one consistent sign; (2) no such win, and the halves disagree about which model is best; (3) neither, so the incumbent stands. The order is stated because a verdict is only checkable against a stated rule.

| form | condition in this script | fired |
|---|---|---|
| a model beats the incumbent by more than the fold spread | selection-half winner's validation paired delta has \|mean\| > sd and one consistent sign | no |
| the ranking does not transfer | no such win, and the validation half's top model is not the selection half's pick (or a model that was not picked clears the spread) | **yes** |
| nothing beats the incumbent; the incumbent stands | no model clears its own spread and both halves agree on the top model | no |

## Rankings

### Selection half (folds [0, 1, 2]) — the only ordering that selects

| # | model | pooled | mean of folds |
|---:|---|---:|---:|
| 1 | `sentence-transformers/all-MiniLM-L6-v2` | 0.9391 | 0.9394 |
| 2 | `ibm-granite/granite-embedding-30m-english` | 0.9391 | 0.9394 |
| 3 | `Snowflake/snowflake-arctic-embed-xs` | 0.8435 | 0.8433 |
| 4 | `sentence-transformers/paraphrase-MiniLM-L3-v2` | 0.8391 | 0.8390 |

### Validation half (folds [3, 4]) — reporting only, selects nothing

| # | model | pooled | mean of folds |
|---:|---|---:|---:|
| 1 | `Snowflake/snowflake-arctic-embed-xs` | 0.9868 | 0.9868 |
| 2 | `sentence-transformers/all-MiniLM-L6-v2` | 0.9408 | 0.9408 |
| 3 | `sentence-transformers/paraphrase-MiniLM-L3-v2` | 0.9408 | 0.9408 |
| 4 | `ibm-granite/granite-embedding-30m-english` | 0.9276 | 0.9276 |

The two orderings disagree on the top model: selection says `sentence-transformers/all-MiniLM-L6-v2`, validation says `Snowflake/snowflake-arctic-embed-xs`. That disagreement is the finding, not a detail — it is why no model is recommended on the strength of these folds.

## Per model

| model | sel pooled | sel mean/fold | val pooled | val mean/fold | paired Δ all folds | sd | folds improved | `__none__` recall | p50 ms | grouped pooled (**not a selection criterion**) | rowwise pooled (**not a selection criterion**) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `ibm-granite/granite-embedding-30m-english` | 0.9391 | 0.9394 | 0.9276 | 0.9276 | -0.53 | 3.03 | 2/5 | 0.92 | 6.32 | 0.9346 | 0.9293 |
| `sentence-transformers/all-MiniLM-L6-v2` | 0.9391 | 0.9394 | 0.9408 | 0.9408 | +0.00 | 0.00 | 0/5 | 0.84 | 11.70 | 0.9398 | 0.9424 |
| `Snowflake/snowflake-arctic-embed-xs` | 0.8435 | 0.8433 | 0.9868 | 0.9868 | -3.92 | 10.48 | 2/5 | 0.88 | 6.70 | 0.9005 | 0.9031 |
| `sentence-transformers/paraphrase-MiniLM-L3-v2` | 0.8391 | 0.8390 | 0.9408 | 0.9408 | -6.03 | 8.56 | 1/5 | 0.72 | 3.67 | 0.8796 | 0.8953 |

*`paired Δ all folds` is the ladder's statistic: the mean and sd of the per-fold difference against the incumbent, in points, over every fold that ran — not the difference of two pooled means. A mean smaller than its own sd is not a win; fold difficulty varies by more than the effect being measured.*

## Per-fold accuracy and per-fold delta

### Accuracy

| model | f0 | f1 | f2 | f3 | f4 |
|---|---:|---:|---:|---:|---:|
| `ibm-granite/granite-embedding-30m-english` | 0.9351 | 0.8831 | 1.0000 | 0.9342 | 0.9211 |
| `sentence-transformers/all-MiniLM-L6-v2` | 0.9221 | 0.8961 | 1.0000 | 0.9079 | 0.9737 |
| `Snowflake/snowflake-arctic-embed-xs` | 0.9091 | 0.8182 | 0.8026 | 0.9868 | 0.9868 |
| `sentence-transformers/paraphrase-MiniLM-L3-v2` | 0.8701 | 0.8442 | 0.8026 | 0.9474 | 0.9342 |

### Delta vs the incumbent, points

| model | f0 | f1 | f2 | f3 | f4 |
|---|---:|---:|---:|---:|---:|
| `ibm-granite/granite-embedding-30m-english` | +1.30 | -1.30 | +0.00 | +2.63 | -5.26 |
| `sentence-transformers/all-MiniLM-L6-v2` | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 |
| `Snowflake/snowflake-arctic-embed-xs` | -1.30 | -7.79 | -19.74 | +7.89 | +1.32 |
| `sentence-transformers/paraphrase-MiniLM-L3-v2` | -5.19 | -5.19 | -19.74 | +3.95 | -3.95 |

Selection folds [0, 1, 2], validation folds [3, 4]. The incumbent is `sentence-transformers/all-MiniLM-L6-v2`; its own row is the zero line.

## What would falsify this

move Snowflake/snowflake-arctic-embed-xs's fold 3 accuracy from 0.9868 to 0.8947 (68/76, 7 rows) and the verdict becomes 'nothing beats the incumbent; the incumbent stands'.

That is the specific number to check: the per-fold rows are in the table above, so the arithmetic can be redone by hand.

## Inputs

- `results/survey-2026-11-23/sweeps/Snowflake__snowflake-arctic-embed-xs.json`
- `results/survey-2026-11-23/sweeps/ibm-granite__granite-embedding-30m-english.json`
- `results/survey-2026-11-23/sweeps/sentence-transformers__all-MiniLM-L6-v2.json`
- `results/survey-2026-11-23/sweeps/sentence-transformers__paraphrase-MiniLM-L3-v2.json`

