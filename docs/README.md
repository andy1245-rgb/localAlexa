# docs/ — what is in here

Sixteen documents: measurement records, the research they rest on, and the briefs that
commissioned them. **A brief is not a result, and a plan is not a measurement** — the two tables
below are kept apart for exactly that reason.

Several of these records are negative: a gate that was failed, an approach that was measured and
retired, a verdict that was withdrawn. They are marked **negative** where that is what they are.
Keeping them is the point — the front page's honest limits section cites them.

All five briefs and plans below have been closed by a record in this directory.

## Measurement results

| file | what it records |
|---|---|
| [`final-comparison.md`](final-comparison.md) | **Negative for the cascade.** Three splits × three arms with the best model as tier 2, the break-even per split, and the shipping decision: ensemble alone, logreg fallback, cascade not shipped. |
| [`cascade-vs-classifier.md`](cascade-vs-classifier.md) | **A withdrawn verdict, then a negative.** The cascade re-measured grouped and without the leak, the gate sweep at 0.35 → 0.70, and the withdrawal of the earlier "cannot beat one classifier" verdict. |
| [`ladder-results-2026-11-10.md`](ladder-results-2026-11-10.md) | The six-rung ladder as measured — five rungs negative, one win: nothing beats the baseline at 0.61 ms, the ensemble wins at 7.85 ms, uncapped augmentation is a trap. |
| [`six-variants.md`](six-variants.md) | **Four of six negative.** Six further structures — regex-as-features, fusion, a separate `none` question, hierarchical, distillation, per-class thresholds — one wins, one is a wash, four lose. |
| [`public-data-2026-11-10.md`](public-data-2026-11-10.md) | **Negative: none adopted.** Four public intent corpora parsed and measured on this corpus's folds; none adopted as training data, and the label space is the reason. |
| [`cactus-hybrid-gate.md`](cactus-hybrid-gate.md) | **Negative: failed its own gate.** One Cactus hybrid checkpoint against an AUROC gate: 0.794 abstain-inclusive against a 0.8 bar, on 128 of 382 rows, not wired. |
| [`data-collection.md`](data-collection.md) | The hands-off collection design as implemented: corrections are negatives, silence is never a label, what one append costs the request path. |
| [`dell321-overnight-2026-11-23.md`](dell321-overnight-2026-11-23.md) | **One win, one negative.** Two jobs run overnight on the CPU-only remote box: paraphrase families help the weak model, the two-stage bucket loses. Artefacts in `results/dell321/`. |
| [`base-model-survey-2026-11-23.md`](base-model-survey-2026-11-23.md) | **Negative for every candidate.** Four phases — 84 sourced rows over five families, a shortlist of five with the eliminations, six probes on the target box, four encoders swept on the ladder's own folds. The ranking does not transfer between the halves that select and the halves that report, so the incumbent stands; latency is not linear in parameter count, and the absolute 10 ms budget was broken by the incumbent itself. Evidence in `results/survey-2026-11-23/`. |
| [`RESEARCH.md`](RESEARCH.md) | The prior research this repo started from — fine-tune bases, classic ML, inference stacks, vision chains — with a dated correction note on top: the 93.2% in its section 3 is a row-wise figure. |

## Environment

| file | what it records |
|---|---|
| [`ladder-env.md`](ladder-env.md) | The two virtualenvs, pinned versions, why the later rungs need torch, and the generator probe that closed one rung as unmeasurable. |

## Briefs and plans

| file | status |
|---|---|
| [`classifier-ladder.md`](classifier-ladder.md) | Written before any rung was built; closed by `ladder-results-2026-11-10.md`. The plan, not a measurement. |
| [`data-collection-brief-2026-11-10.md`](data-collection-brief-2026-11-10.md) | The corrected collection design, written after the operator's objection that "no correction" is not a positive; closed by `data-collection.md`. |
| [`public-data-brief-2026-11-10.md`](public-data-brief-2026-11-10.md) | What the public-data question had to answer and in what order; closed by `public-data-2026-11-10.md`. |
| [`dell321-overnight-brief-2026-11-23.md`](dell321-overnight-brief-2026-11-23.md) | The two overnight jobs, their schedules and rules; closed by `dell321-overnight-2026-11-23.md`. |
| [`base-model-survey-brief-2026-11-23.md`](base-model-survey-brief-2026-11-23.md) | Four phases for finding a better base model than SetFit-on-MiniLM, with the constraints and the three acceptable verdicts; closed by `base-model-survey-2026-11-23.md`. The plan, not a measurement. |
