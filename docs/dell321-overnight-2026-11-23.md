# Overnight on dell321: paraphrases help the weak model, the two-stage bucket loses

2026-11-23, branch `collect/2026-11-10`. Both jobs from
`docs/dell321-overnight-brief-2026-11-23.md` ran on `dell321` over ssh, CPU only, 8 logical
cores, Python 3.11.9 / scikit-learn 1.8.0 / numpy 2.4.4. Nothing ran against the sshfs mount:
the corpus and `src/localalexa` were copied with `scp` to `C:\Users\andre\alexa-finetune\`, the
`.git` directory was never copied, and only JSON came back (`results/dell321/job1.json`,
`job2.json`; the tables below are rendered from them by
`experiments/dell321/render_report.py`).

**The box's flat baseline is 90.31%, not the 90.84% this machine measures** — one row of drift
from the scikit-learn version, the same effect `docs/public-data-2026-11-10.md` documented. Every
comparison below is internal to dell321.

## Job 1 — paraphrase families: +2.34 for the weak model, −0.79 for the strong one

Deterministic transformations of the 200 references (politeness add/drop, context clause,
adverbial reorder, contraction expand/contract, curated verb synonyms), two rounds, each row
carrying the transformations that produced it. Then a diversity filter against the existing pool
and a separate leak filter against the *held-out* rows, and a retrain on the same five grouped
folds.

| arm | references | + paraphrases | paired Δ | folds W/L/T | fold sd |
|---|---:|---:|---:|---|---:|
| word logreg C=100 | 90.31% | **92.67%** | **+2.34 ± 4.63** | 3/1/1 | 8.04% → 5.52% |
| fusion (linear + regex + router) | **95.03%** | 94.24% | −0.79 ± 1.99 | 1/2/2 | 2.82% → 4.39% |

| generation | |
|---|---:|
| candidates generated | 1,637 |
| distinct | 1,513 |
| kept after both filters | **878** |
| dropped, adds nothing (below the diversity floor) | 633 |
| dropped, leaks (too close to a held-out row) | 2 |

Diversity of the kept rows (1 − max 3-gram Jaccard against the reference pool):

| n | min | p10 | p50 | p90 | max | mean |
|---:|---:|---:|---:|---:|---:|---:|
| 878 | 0.364 | 0.400 | 0.500 | 0.750 | 1.000 | 0.560 |

Kept rows by transformation: `context` 646, `polite` 548, `synonym` 155, `expand` 62, `reorder` 60,
`contract` 5, `drop_polite` 5.

**What this says.** New *shapes* do move the family-held-out number — for the model that needs
them. The word model gains 2.34 points and its fold spread falls from 8.04% to 5.52%, which is the
signature of more surface variety in the training pool. The fusion gains nothing and loses 0.79: it
already has a reference-similarity member, and a paraphrase of a reference is exactly what that
member already covers. **The ceiling was not "more rows", it was "more rows the incumbent has no
way to see" — and which rows those are depends on the recipe.**

Two limits, stated because they bound the claim: 878 rows is a third of the candidates, so the
generator is mostly producing things the pool already contains; and the +2.34 ± 4.63 does not clear
the fold spread by this repo's own rule, although 3 folds improve and the spread halves. The
direction is consistent with the augmentation rung's earlier finding (+1.6 at 16 fills per label);
this is the first time shape rather than volume paid.

## Job 2 — the two-stage bucket: it loses, and the premise is wrong

Buckets proposed three ways: agglomerative clustering of the labels by their out-of-fold confusion
distance, the registry's own families (the operator's hypothesis, scored for the first time), and
the single strongest confusion pair isolated.

| proposal | buckets | stage A acc | errors within a bucket | top-1 | top-2 | p50 top-1 | p50 top-2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| flat incumbent | — | — | — | **90.31%** | — | **0.43 ms** | — |
| agglomerative k=3 | 3 | 98.95% | 100.00% | 89.54% | 89.80% | 0.40 ms | 0.61 ms |
| agglomerative k=4 | 4 | 98.95% | 100.00% | 89.54% | 89.80% | 0.42 ms | 1.00 ms |
| agglomerative k=5 | 5 | 99.48% | 100.00% | 90.06% | 90.06% | 0.61 ms | 0.68 ms |
| registry families | 7 | 95.29% | **43.24%** | 90.09% | 90.61% | 0.61 ms | 0.98 ms |
| strongest pair isolated | 2 | 98.95% | 78.38% | 91.11% | **91.37%** | 0.65 ms | 0.99 ms |

**The diagnosis does not survive the confusion matrix.** "The confusions are within-family" is
false on this model's out-of-fold predictions: with the registry's seven families as buckets, only
**43.24%** of errors stay inside a bucket. The ranked confusions are `light_query → light_set` (7,
same family), then `__none__ → light_query` (4, cross), `thermostat_offset → light_set` (3, cross),
`projects_query → shoppinglist_view` (3, cross), `calendar_today → __none__` (3, cross). The
hierarchy the data actually has is a hub — `light_query`/`light_set` absorbing strays from other
families — not a tree.

That is also why agglomerative clustering degenerates: most label pairs never confuse at all
(distance 1.0), so the method peels off singletons and leaves one bucket holding 13–14 of the 18
labels. Its two-stage score is *below* flat (89.54%) because stage B then trains on a subset of the
pool and loses the cross-bucket evidence that made the flat model work, while stage A contributes
almost nothing (98.95% at bucket level).

The only hierarchy that beats flat is the trivial one — isolate `{light_query, light_set}` and let
everything else stay together: **91.37% at top-2, +1.06 points**, for 2.3× the latency (0.99 ms
against 0.43 ms). That is smaller than the fold spread (6.78%), so by this repo's rule it has not
won either; it is a pair-specific fix, not an architecture.

**Verdict on job 2: the two-stage bucket loses.** For the operator's decision this closes the
question: there is no useful hierarchy in these confusions to exploit. The 26 rows the other
project gained from fusion came from *combining signals*, not from *splitting labels* — which is
consistent with this repo's own fusion result (§ the six variants: fusion 95.03%, +4.16).

## What I would do with this

1. **Keep the paraphrase generator and point it at the weak recipe.** It is deterministic, it is
   cheap, and it is the first expansion in this repo that moved a held-out number for a model that
   needed it. An LLM paraphrase pass is now worth trying, with the same diversity and leak filters,
   because the deterministic set is visibly hitting its own ceiling (633 of 1,637 candidates were
   too similar to keep).
2. **Drop the two-stage bucket from the roadmap.** The confusions are not within-family, stage A is
   nearly perfect and therefore nearly useless, and the only winning split is "isolate the light
   pair", which a per-class threshold already does more cheaply.
3. **Re-run the paraphrase arm against the fusion with the paraphrase *pool* restricted to intents
   the fusion gets wrong** — that is the untried version of this experiment: shape was worth +2.34
   where the model lacked it, and the fusion's 4 wrong-most intents (`light_query`,
   `__none__`, `shoppinglist_add`, `shoppinglist_remove`) are exactly the ones whose phrasings are
   already dense.

## Reproduce

```bash
# on dell321 (paths as copied)
ssh dell321 "cd C:\Users\andre\alexa-finetune; \$env:PYTHONPATH='C:\Users\andre\alexa-finetune\src'; python job1_paraphrase_families.py --recipes linear,fusion --out results/dell321/job1.json"
ssh dell321 "cd C:\Users\andre\alexa-finetune; \$env:PYTHONPATH='C:\Users\andre\alexa-finetune\src'; python job2_two_stage_bucket.py --sizes 3,4,5 --out results/dell321/job2.json"

# here, from the retrieved JSON
.venv/bin/python experiments/dell321/render_report.py
```

Artifacts: `results/dell321/job1.json`, `results/dell321/job2.json` (per-fold numbers, the
generation and filter distributions, every bucket assignment with its evidence, and the latency of
each path), `results/dell321/tables.md`.

## Limits

* 878 kept rows from 1,637 candidates: the deterministic generator saturates quickly, which is why
  the diversity distribution is reported rather than the count.
* The transformations are not grammar-checked beyond two fixes made during this run (`what are` →
  `what's` produced "what are left on my countdown"; a context clause applied to an already-framed
  request produced "could you quick, am I free"). Some kept rows are awkward — "could you what is
  my agenda for today?" — and awkward is not the same as wrong, but it is noise the model has to
  absorb.
* Two-stage latency is measured per utterance on one fold's models, 60 utterances, on a box that
  was otherwise idle; the flat path is measured the same way in the same run, so the ratio is the
  trustworthy number, not the absolute milliseconds.
* Everything here is the word model, the fusion, and sklearn; no model larger than MiniLM was
  used, and no API was called.
