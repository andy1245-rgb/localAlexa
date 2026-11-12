# Public intent data — it does not raise the ceiling, and the reason is the label space

2026-11-10, branch `public-data/2026-11-10`. Four public corpora were parsed on this box,
mapped onto localAlexa's 17 tools by four parallel subagents, and then measured in one place
on one fold assignment. `cascade.py` semantics and `SMART_ASSUMED_ACC` were not touched.

**Recommendation: adopt none of the four — as training data.** The evidence is below and it
is not close: every configuration that adds public rows comes out at or below the
reference-only baseline, the best case by 1.8 points, and the only thing that measurably
improves is the abstain class. Two of the four corpora cannot even train a router under this
mapping — they collapse to a single label. What is worth keeping is not the supervision but
the *negative* material, and §9 says what to do with it.

## 1. The numbers in one table

Grouped CV, five folds over the 382 rows' 110 phrasing families, rung-0 model
(`word TF-IDF + LinearSVC`, the model the 89.79% was measured with). Δ is against the
re-measured reference arm, not the quoted number (§2).

| arm (rows added to the training pool) | added | acc | Δ | p | folds W/L/T | `__none__` recall |
|---|---:|---:|---:|---:|---:|---:|
| **refs — reference arm** | 0 | **90.05%** | — | — | — | 0.60 |
| union, everything | 54,510 | 67.28% | −22.77 | <0.0001 | 1/4/0 | 1.00 |
| union, high-confidence mappings only | 46,749 | 67.54% | −22.51 | <0.0001 | 1/4/0 | 1.00 |
| CLINC150 alone | 23,695 | 66.23% | −23.82 | <0.0001 | 1/4/0 | 1.00 |
| SNIPS alone | 14,306 | 70.68% | −19.37 | <0.0001 | 0/5/0 | 1.00 |
| HWU64 alone | 11,036 | 73.04% | −17.01 | <0.0001 | 1/4/0 | 0.96 |
| union, tool-mapped rows only (no abstain rows) | 2,862 | 72.77% | −17.28 | <0.0001 | 0/5/0 | 0.08 |
| union, abstain rows capped at 25 | 2,887 | 74.08% | −15.97 | <0.0001 | 0/5/0 | 0.28 |
| CLINC150, tool-mapped rows only | 1,200 | 79.06% | −10.99 | <0.0001 | 0/5/0 | 0.24 |
| HWU64, tool-mapped rows only | 1,662 | 78.80% | −11.25 | <0.0001 | 1/4/0 | 0.12 |
| union, 100 rows per label per corpus | 1,600 | 78.27% | −11.78 | <0.0001 | 1/4/0 | 0.72 |
| ATIS alone | 5,473 | 84.82% | −5.23 | 0.0022 | 1/4/0 | 1.00 |
| **HWU64, 25 rows per label** | 175 | 87.96% | −2.09 | 0.13 | 1/3/1 | 0.56 |
| **union, 25 rows per label per corpus** | 400 | **88.22%** | **−1.83** | 0.31 | 1/4/0 | 0.88 |

The last two rows are the fair test — 25 rows per label is the only cap in the same order as
the reference pool's 10–12 per tool, and it is the configuration a sane adoption would use. It
is 1.8 points *down*, one fold better and four worse, and McNemar cannot tell that apart from
noise on 382 rows (p = 0.31). The ceiling did not move.

The second classifier agrees, at the six arms it was run on: reference 90.31%, HWU64 72.77%,
HWU64 tool-only 79.84%, union 70.16%, union capped at 100 79.58%, union tool-only 76.18%. No
arm improves under either model, so this is not one solver's opinion.

**Read the uncapped rows as a class-prior experiment, not a supervision experiment.** Three of
the four corpora are 85–100% out of scope for localAlexa's tools, so an uncapped union hands
the model 52,229 abstain rows against 175 tool references. It predicts `__none__` (recall
1.00) and loses 23 points. That is arithmetic, and it is why the capped arms exist.

## 2. The 89.79% baseline does not reproduce on this box — it is one row stale

The brief's comparison target is 89.79%. Re-running the ladder's own code
(`rung_0_baseline`, `evaluate_grouped`, `build_folds`, 5 folds) on this box yields **90.05%
(344/382)**; the recorded artifact says 343/382. Same folds, same pools, same code:
`git worktree` at the commit that wrote the artifact (`a1ef942`) re-measures 0.9005 too; a
row-by-row diff against my harness replica over all 382 rows shows **zero** disagreements for
rung 0, and rung 2's per-fold accuracies match exactly. So the difference is the environment —
a different scikit-learn wheel between the run that wrote the number and the one installed now —
not the fold map, not the pool, not the model.

Every comparison in this doc is against the re-measured 90.05%, because that is the same
process on the same box as the arms it is compared to. Against the *quoted* 89.79% the two
capped arms read −1.57 and −1.83 points, and the conclusion is unchanged either way. Worth
fixing in the ladder docs at some point: the number moves by a row between environments, and
that is larger than any effect this experiment found.

## 3. Licence, per dataset, read from each dataset's own page

| dataset | licence | SPDX | redistributable here? | read from |
|---|---|---|---|---|
| CLINC150 | CC BY 3.0 Unported | `CC-BY-3.0` | **yes**, with attribution: "to Reproduce the Work, to incorporate the Work into one or more Collections" / "to Distribute and Publicly Perform the Work" (§3a, §3c); §4a–§4b require the licence URI and credit to travel with the copies | `clinc/oos-eval` `LICENSE` |
| HWU64 | Creative Commons Attribution 4.0 International | `CC-BY-4.0` | **yes**, with attribution: "reproduce and Share the Licensed Material, in whole or in part" / "produce, reproduce, and Share Adapted Material" (§2a); §3a requires creator identification and a licence notice | `xliuhw/NLU-Evaluation-Data` `README.md` and `LICENSE` |
| SNIPS | CC0 1.0 Universal | `CC0-1.0` | **yes**, unconditionally: §2 Waiver — "overtly, fully, permanently, irrevocably and unconditionally waives, abandons, and surrenders all of Affirmer's Copyright and Related Rights … for any purpose whatsoever". No attribution condition; the README *asks* for a citation, which is not a licence term | `sonos/nlu-benchmark` root `LICENSE` |
| ATIS | LDC User Agreement for Non-Members | none (not an SPDX licence) | **no.** "User agrees to use the LDC Databases received under this Agreement only for non-commercial linguistic education, research and technology development" and "User shall not otherwise publish, retransmit, disclose, display, copy, reproduce or redistribute the LDC Databases to others outside of User's Research Group" | `catalog.ldc.upenn.edu/LDC94S19` and `LDC95S26` |

One caveat that matters more than it looks: the ATIS bytes actually downloaded come from
`howl-anderson/ATIS_dataset`, which carries **no licence at all** (no `LICENSE`, GitHub reports
`license: null`). The corpus descends from the LDC releases, so the LDC agreement is the terms
that apply and redistribution is prohibited. ATIS is usable as a local measurement and must
never be committed — not the rows, not a sample, not a fixture.

No dataset rows are committed by this branch. `data/raw/` is gitignored and stays that way.
The repository gets this document: licence strings, counts, the mapping table, and the recipe.

## 4. Rows, splits, and the size of the intersection

| dataset | rows | train / test | intents | intents → `none` | **rows → `none`** | tool-mapped rows |
|---|---:|---|---:|---:|---:|---:|
| CLINC150 | 23,700 | 18,200 / 5,500 | 150 | 143 | 22,500 (94.9%) | 1,200 |
| HWU64 | 11,036 | 9,960 / 1,076 | 64 | 54 | 9,374 (84.9%) | 1,662 |
| SNIPS | 14,484 | 13,784 / 700 | 7 | 7 | 14,484 (100.0%) | **0** |
| ATIS | 5,871 | 4,978 / 893 | 26 | 26 | 5,871 (100.0%) | **0** |

Split rules, so a reader can rebuild them: CLINC150 is the corpus's own train/val/test with val
folded into train and the `oos_*` blocks kept as the `none` class (that is 1,200 real
out-of-scope utterances, the single most valuable thing in any of these four); HWU64 is the
suite's own `KFold_1`, not a split of mine; SNIPS is train-full/validate as shipped; ATIS is the
Rasa `train.json`/`test.json` from the mirror. Counts were re-derived locally from the parsed
files and match each subagent's report. The training arms use every mapped row of a corpus,
train and test splits alike, deduplicated on text; the reverse direction (§7) is the only one
that scores a corpus's own test split, and it never trains on it.

The intersection is the story: **9 of 17 tools receive zero rows from all four corpora
combined** — `timer_cancel`, `timer_status`, `reminder_cancel`, `thermostat_set`,
`thermostat_offset`, `light_query`, `docs_query`, `projects_query`, `research_deep`. The task
the brief poses ("more ways of saying the same thing") is only available for 8 labels, and for
five of those the available material is thinner than the references already in the registry.

## 5. The mapping table

Rows per tool per corpus. The mapping files are `data/raw/<dataset>/mapping.csv`, one row per
intent, written by the recon stage and validated on load against the 18-label option set — a
tool id outside it is a load error, not a silent drop.

| tool | CLINC150 | HWU64 | SNIPS | ATIS | total |
|---|---:|---:|---:|---:|---:|
| timer_create | 150 | 0 | 0 | 0 | 150 |
| timer_cancel | 0 | 0 | 0 | 0 | **0** |
| timer_status | 0 | 0 | 0 | 0 | **0** |
| reminder_create | 300 | 0 | 0 | 0 | 300 |
| reminder_cancel | 0 | 0 | 0 | 0 | **0** |
| shoppinglist_add | 150 | 194 | 0 | 0 | 344 |
| shoppinglist_remove | 0 | 194 | 0 | 0 | 194 |
| shoppinglist_view | 150 | 194 | 0 | 0 | 344 |
| thermostat_set | 0 | 0 | 0 | 0 | **0** |
| thermostat_offset | 0 | 0 | 0 | 0 | **0** |
| light_set | 150 | 692 | 0 | 0 | 842 |
| light_query | 0 | 0 | 0 | 0 | **0** |
| docs_query | 0 | 0 | 0 | 0 | **0** |
| projects_query | 0 | 0 | 0 | 0 | **0** |
| email_unread | 0 | 194 | 0 | 0 | 194 |
| calendar_today | 300 | 194 | 0 | 0 | 494 |
| research_deep | 0 | 0 | 0 | 0 | **0** |
| `__none__` | 22,500 | 9,374 | 14,484 | 5,871 | 52,229 |
| **rows** | 23,700 | 11,036 | 14,484 | 5,871 | **55,091** |

### What the failures say about the label design

Five gaps, in order of how much they cost, all of them found by asking where the mapping has no
answer rather than by asking where it has a good one:

1. **There is no alarm concept.** HWU64's `alarm_set` / `alarm_remove` / `alarm_query` (505 rows)
   are the closest thing in any corpus to `timer_create` + `reminder_create` and they map to
   nothing, because an alarm is a recurring time trigger and localAlexa has neither a
   repeating reminder nor an alarm. `reminder_create` is the near-miss and mapping it there
   would be wrong: the reference set for `reminder_create` is "remind me to X at Y", not
   "wake me at 6:30 every weekday". This is a missing tool, not a missing label.
2. **There is no calendar write and no reminder/alarm view.** `calendar_set` / `calendar_remove`
   (388 HWU64 rows) have nowhere to go — localAlexa can read today's calendar but not write to
   it — and `alarm_query` plus CLINC150's `reminder` (view) have nowhere either, even though
   `timer_status` and `light_query` exist as views for their families. The 17 tools are
   asymmetric across families: timers and lights have create/set *and* status/query, reminders
   and calendar have neither.
3. **`light_set` is one tool where the data has five.** HWU64 splits lighting into
   `iot_hue_lighton` / `lightoff` / `lightup` / `lightdim` / `lightchange` (692 rows, mapped at
   `low`/`low`/`low`/`high`/`high`). On/off is one axis; brightness and colour are another. A
   17-way head cannot express "dimmer", and it sees five intents with conflicting targets for
   one label. This is the strongest case in the data for splitting `light_set` into a power
   tool and a level/colour tool — and it is also why mapping those 692 rows into this label is
   part of what made the HWU64 arms worse.
4. **One dataset intent, two local tools.** CLINC150's `shopping_list_update` covers add and
   remove in one utterance; localAlexa splits them. Conversely `timer_create` / `timer_cancel` /
   `timer_status` are three labels where CLINC150 has one creation-only `timer` intent, and
   `calendar_today` is spread across CLINC150's `calendar` and `meeting_schedule`. Three-to-one
   and one-to-two splits in both directions are a sign the local split is drawn on the *verb*
   while the public corpora draw it on the *object*.
5. **`general_commandstop` is a false-positive factory.** HWU64's bare "stop" / "cancel" /
   "never mind" is meta-dialogue; under localAlexa's grammar a bare "stop" most plausibly means
   `timer_cancel`. Mapped to `none` at medium confidence, and flagged here because a model that
   learns it will refuse a command the operator meant.

And the two that map nothing at all, which is a statement about scope rather than a defect:
SNIPS (music, restaurants, weather, media catalogues) and ATIS (flight booking) are domains
localAlexa has no tool for. They are not bad data. They are data about a different product.

## 6. Direction one: train on public, evaluate on the 382 rows

The appendix-1 model, trained only on the public rows mapped to localAlexa's labels, scored on
all 382 rows:

| train set | rows | accuracy | Δ vs reference |
|---|---:|---:|---:|
| union | 54,510 | 37.43% | −52.62 |
| HWU64 | 11,036 | 30.10% | −59.95 |
| CLINC150 | 23,695 | 27.49% | −62.56 |
| SNIPS | 14,306 | **refused** — one label only (`__none__`) | — |
| ATIS | 5,473 | **refused** — one label only (`__none__`) | — |

The two refusals are the harness behaving correctly: a one-class fit is a degenerate model, and
`sklearn_text` refuses rather than emitting a constant. They are also the result. Two of four
corpora carry exactly one localAlexa label under this mapping, so they cannot train a router at
all, in any direction.

The other two land at 27–37% against a model that gets 90% from 200 references — the mapping
throws away so much of the label space that the resulting classifier is barely better than
guessing among 18 classes. That is not a model-quality statement; it is the same label-coverage
fact as §4, measured.

## 7. Direction two: train on the 382 rows, evaluate on the public rows

Trained on all 382 rows, scored on each corpus's held-out test rows (mapped labels only):

| test set | rows | accuracy | on rows that should abstain | on rows with a real tool |
|---|---:|---:|---:|---:|
| CLINC150 | 5,500 | 39.78% | 39.4% of 5,260 | 47.9% of 240 |
| HWU64 | 1,076 | 39.41% | 37.7% of 914 | 48.8% of 162 |
| SNIPS | 700 | 33.71% | 33.7% of 700 | — |
| ATIS | 893 | 11.20% | 11.2% of 893 | — |
| union | 8,169 | 36.09% | — | — |

The headline number is misleading on its own, so read the fourth column: on rows whose correct
answer is abstain, a model trained on localAlexa's 382 rows abstains on **11–40%** of them. It
routes the rest to a tool. localAlexa's `__none__` class is 25 chitchat utterances ("tell me a
joke", "how are you doing today"), and it does not generalise to a request that is out of scope
because it is about a domain the assistant does not have — a flight, a restaurant, the weather.
Those get sent to `calendar_today` and `research_deep` because that is what the surface form
resembles.

This is the single most useful thing the four corpora produced, and it costs no training data
at all — see §9.

## 8. What the shared harness gave, and where it got in the way

`~/toolInterfacer/bench/harness/` **is** usable from here, and this experiment uses it: `Row`
and its option-set validation, `protocol.build` with the `sklearn_text` predictor (rung 0's
pipeline as a spec string), and `metrics.score` / `accuracy` / `paired` (exact McNemar) /
`wilson_ci`. Every number above except the fold assignment comes from those functions, which is
what makes them comparable to the harness's own artifacts.

Three findings about it, in decreasing order of how much they cost me:

1. **`run.py --cv` cannot score this dataset.** The `localalexa_intents` adapter sets
   `group=r["family"]`, which is the *tool family* (7 values: timers, reminders, shopping,
   climate, lights, knowledge, none) — not the phrasing family. Five folds over seven groups
   holds out whole families, so every fold asks the model to predict labels it has never seen:
   `run.py --dataset localalexa_intents --cv 5` reports **0.0%** with a five-fold
   "holds out a label the training side never sees" warning. The adapter's own docstring says
   "`group` is the intent family from the row", so this is a naming collision rather than a
   typo — but grouped CV on this corpus means the 110 phrasing families, and the file
   `data/routing_382.json` does not carry them; they have to be recovered from the registry's
   templates, as `localalexa.ladder.folds` does. A one-line note in the adapter pointing at
   `template_index` would save the next reader this afternoon.
2. **`grouped_cv`'s fold rule is not the ladder's.** It is hashed and label-stratified; the
   ladder uses `GroupKFold` over families. On this corpus the two differ by more than 20 rows
   (the harness says so itself), so using `grouped_cv` would have moved the baseline and made
   the comparison meaningless. The fix was four lines of fold loop in `localbench.py` reusing
   `localalexa.ladder.folds.build_folds`, and calling the harness only for fitting and scoring.
   That worked exactly as intended — the seam is the right one — but "one fold assignment"
   required owning the loop.
3. **No adapter reads these corpora, and `generic_csv` is the wrong shape for them.** These are
   JSON/TSV with their own splits and an intent vocabulary that has to be mapped before a row
   has a label at all. I normalized each into `split <TAB> text <TAB> intent` and built `Row`s
   directly; `generic_csv` would have needed a pre-joined file anyway and cannot express the
   intent→label mapping that is the actual deliverable of the recon stage.

## 9. What would change the answer, and what is worth keeping

The bottleneck is not volume. It is that the label space is 8 of 150 + 10 of 64 + 0 of 7 + 0 of
26, and that nine tools have no external paraphrase at all. Two concrete ways to revisit:

- **As a paraphrase source for the 8 labels that do overlap, not as pool rows.** The ladder's
  rung 6 already showed the pattern: uncapped augmentation of any kind made the grouped number
  worse, capped augmentation did not. Adding 25 rows per label here reproduced that shape
  exactly (−1.83 points, indistinguishable from noise). The untried variant is to treat HWU64's
  `light_set` rows as a *new phrasing family* and hold them out in the fold map, so the model
  has to generalise to them rather than being scored on the families it trained on. That is a
  different experiment from this one and it is the only version of "adopt HWU64" I would still
  put money on being able to help — it needs the mapping split into power/level first (§5, gap 3).
- **As an out-of-scope benchmark, which needs no training data and no licence risk.** Direction
  two shows the abstain path is the weak part: 39% of out-of-scope requests are refused. Take
  the 5,260 CLINC150 `oos`+unmapped test rows and the 700 SNIPS test rows (both
  redistributable), point the shipped cascade at them, and measure `__none__` recall and the
  false-tool rate. That is a regression test for the thing that is actually broken, it costs
  one file of text, and it does not touch the classifier at all. Keep ATIS out of it — its
  licence forbids redistribution.

Neither of those is "adopt these datasets", and both are cheaper than the ceiling work this
brief set out to do.

## 10. Reproduction

```
python experiments/public/sources.py                  # data/raw/<ds>/normalized.tsv
python experiments/public/transfer.py baseline        # the reference arm, both classifiers
python experiments/public/transfer.py all             # every arm; writes results/public/transfer.json
python experiments/public/report.py                   # the artifact above as the tables in §1, §6, §7
```

`experiments/public/localbench.py` is the protocol (the ladder's fold map and reference pool,
scored through the harness). `public_corpus.py` joins a corpus to its `mapping.csv` and refuses
an unjoined intent. `sources.py` is the four normalizers. `transfer.py` is the arms and the
artifact, `report.py` renders it. The harness checkout is found as a sibling
directory or through `TOOLINTERFACER_ROOT`. Rows live only under gitignored `data/raw/`, and
the run is `$0`: everything came from public repos over `https`, no API, no mirror service,
nothing paid.

Runtime is a few minutes; the two uncapped union arms dominate it, because fitting an 18-class
linear model over 54,510 rows is the only expensive thing in the experiment.
