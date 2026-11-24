# localAlexa

Intent classification for a voice assistant that runs on the CPU you already have: one
utterance of text becomes one of 18 operations, in well under a millisecond, with no model
call and no network in the inference path.

![Licence: Apache-2.0](https://img.shields.io/badge/licence-Apache--2.0-blue)
![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)

This is the software layer only — the utterance arrives as text. Microphone, wake word,
ESP32 firmware and on-device deployment are not in this repository and have not been built.

## What it does

One string in, one operation and a confidence out, plus the candidates it considered:

```console
$ python -m localalexa ask "turn off the hall light" --no-log
predictor: linear stand-in (the semantic router needs sentence-transformers: pip install 'localalexa[embeddings]')
light_set (tier 0) · confidence 1.000 · no router call
  candidates: light_set 1.00
```

A hand-written pattern claimed that one at zero cost. When no pattern fires, the cheap model
answers, and it can be unsure:

```console
$ python -m localalexa ask "how far is the moon" --no-log
predictor: linear stand-in (the semantic router needs sentence-transformers: pip install 'localalexa[embeddings]')
light_query (tier 1) · confidence 0.474 · 2.4 ms
  candidates: light_query 0.58, docs_query 0.14, timer_status 0.09
```

That second answer is wrong — chitchat has no route, and the linear tier picks the nearest
tool rather than abstaining. The `2.4 ms` in that line is the full request path on this box,
not the isolated latency measurement below.

The labels are 17 tools plus a `none` route: timers, lights, thermostat, shopping lists,
calendar, reminders, documents, research and projects. The design is a three-tier cascade:

| tier | mechanism | measured cost |
|---|---|---|
| 0 | hand-written regex per tool | ~0 ms |
| 1 | word TF-IDF + logistic regression, or MiniLM reference similarity | 0.38 ms p50 |
| 2 | escalate to something expensive | not shipped — see below |

Tier 0 covers 76.2% of evaluation rows at 83.8% first-hit accuracy, 12% of what it covers is
ambiguous, and it misses 67 tool rows outright, so it is a filter rather than an answerer.

## The measurement, and the caveat that goes with each number

382 utterances, 110 phrasing families, 18 labels, about 22 training references per intent.
The evaluation rows are generated from the same templates the registry trains on, so **the
split decides the number**. Read this table before any accuracy on this page — model is the
default word TF-IDF + LinearSVC, reproduced by `localalexa eval --split all`:

| split | what training sees | accuracy | 95% CI | vs row-wise |
|---|---|---:|---|---:|
| row-wise | the 200 registry references | **93.19%** (356/382) | 88.0–97.6 | — |
| phrasing-family held out | that pool minus the held-out family's slot fills | 90.31% (345/382) | 85.0–95.2 | −2.9 |
| phrasing-family held out, strict | rebuilt from the other 109 families' template fills | **74.87%** (286/382) | 66.7–83.1 | −18.3 |

Row-wise is template-contaminated: 29 of the 200 training references are literal slot fills of
the templates that generate the scoring rows, so it measures template overlap as much as it
measures understanding. Intervals resample whole phrasing families (110 clusters), not rows —
rows inside a family are not independent.

### One classifier at 0.38 ms

Calibrated logistic regression on the baseline's own features answers every row at **90.84%
grouped** (347/382) in **0.38 ms p50**, and that figure comes with its conditions attached:
one utterance, warm model, idle box, median of five passes, on the CPU of the machine that ran
the ladder, whose model is not recorded in the artefact (`results/ladder/latency.json`) — so treat
it as this box's number, not a portable one, especially since the baseline drifts between 0.34 and
1.68 ms when other work shares the cores, which is why the idle-box pass and the training runs are
kept apart (`docs/ladder-results-2026-11-10.md`). What the change buys
besides speed is calibration: expected calibration error 0.735 → 0.128 for the same cost, and
`none` recall 0.60 → 0.68 (`results/ladder/linear-rungs.json`).

The encoder alternatives cost 20×: a SetFit model at 93.98% grouped for 7.75 ms, and a
row-standardised logreg + SetFit ensemble at 95.03% for 7.85 ms.

### The cascade was measured and lost

The cascade is the design this repo was built around, and with the best model in the repo as
tier 2 it does not beat that model. On the family-held-out split:

| arm | accuracy | correct | latency |
|---|---:|---:|---:|
| tier 1 alone (logistic regression) | 90.84% | 347/382 | 0.38 ms |
| tier 2 alone (logreg + SetFit ensemble) | **95.03%** | 363/382 | 7.85 ms |
| **the cascade, end to end** (gate 0.35) | 90.28% | — | — |

It loses for a reason that is not a knob. The cheap path answers 94.0% of traffic itself at
89.97% — worse than the model it is fronting — so the remaining 23 rows barely matter: the
tier-2 accuracy needed to match the 95.03% arm is **174%**, and 104% to match the single
classifier. No tier-2 model at any quality rescues this operating point. The gate, not the
expensive tier, is the constraint.
Swept at 0.70 the cascade does beat tier 1 alone (92.9% at a tier-2 accuracy of 0.95) and still
never reaches the ensemble. One split disagrees — on the strict split, where the two cheap
tiers land within 10 points of each other, the cascade wins 80.83% to 78.53%, which is an
argument for keeping the code and not for deploying it.

**Shipped decision, from the same comparison: run the ensemble as a single classifier, keep the
logreg as the 0.38 ms fallback, do not ship the cascade as configured.** Record and full
three-split, three-arm table: `docs/final-comparison.md`; gate sweep and the withdrawn earlier
verdict: `docs/cascade-vs-classifier.md`. The run artefacts behind those tables are not
distributed in this repository.

### Tier 2 has been tried once

One Cactus hybrid checkpoint was tested for the only property that makes a tier-2 handoff worth
anything — whether its own confidence knows when it is wrong. Abstain-inclusive AUROC **0.794**
against a 0.8 bar (0.837 if the `none` rows are excluded, which is the exclusion that makes a
gate useless), on 128 of 382 rows. Nothing was wired. `docs/cactus-hybrid-gate.md`.

## How it was measured

Every rung was scored on one `GroupKFold` assignment over the 110 phrasing families, with the
fold assignment written into every artefact, and wins are reported as paired per-fold deltas
with the spread beside them — a gain smaller than the fold-to-fold spread is not called a win
here. Where a hyperparameter or a threshold was chosen on the folds it is scored on, the
document says so. Latency was measured in a separate pass, alone on the box, because the
latencies recorded inside training runs drift with load. Row-wise accuracy is reported only
because the repo's earlier claims live there, and one rung demonstrates why: feeding the model
every registry slot fill scores 96.34% row-wise and 86.65% grouped — the 96.34% is the
protocol, not the model.

The six-rung ladder and its verdicts are `docs/ladder-results-2026-11-10.md`; the plan it was
pre-registered against is `docs/classifier-ladder.md`; six further structures are in
`docs/six-variants.md`; the environment, pinned versions and the generator probe that closed
one rung as unmeasurable are `docs/ladder-env.md`.

## Quickstart

Needs Python 3.12 or newer (the `python3` on a stock Ubuntu is older — check it) and either
`uv` or `pip`. Nothing else: the base install is numpy and scikit-learn. From the repository
root:

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -e .
.venv/bin/python -m localalexa ask "turn off the hall light" --no-log
.venv/bin/python -m localalexa eval --split all         # the three-split table above
.venv/bin/python -m localalexa regex                    # tier 0 alone, no model needed
.venv/bin/python -m localalexa compare --tier1 logreg   # cascade vs one classifier, grouped
```

The remaining subcommands reproduce the research records rather than the front page:
`ladder --rungs all` (the six-rung ladder; the encoder rungs print a skip line until the
extras are installed), `collect summary|derive|bench` (the local request log), and `cascade`
(the regex → router → abstain path, which needs the extras).
`tier2 --measured results/cactus_hybrid_results.json` joins the cascade to the committed
per-row measurement of the Cactus checkpoint, while `classic` reruns the three TF-IDF
configurations.

The test suite is 177 cases and takes a couple of minutes:

```bash
uv pip install --python .venv/bin/python pytest
.venv/bin/python -m pytest
```

The encoder tiers are optional and need `sentence-transformers`, which brings torch, plus a
one-time model download:
`uv pip install --python .venv/bin/python -e '.[embeddings]'`. Everything above runs offline
against the data committed here.

## What it does not do

- **No audio.** No wake word, no speech recognition, no microphone array, no firmware. The
  utterance reaches `ask` as a string.
- **No tier-2 model ships.** The cascade's tier-2 figures are a parameter sweep over assumed
  accuracies, not a measured LLM, and the one checkpoint that was measured failed its gate.
- **18 labels, one domain.** A smart-home tool set of 17 operations plus a `none` route. It is
  not a general assistant and the label space is a design choice, not a limitation to be
  papered over.
- **The honest accuracy is 74.87%, not 93%.** Strict family-held-out, with no hand-written
  reference for the held-out wording. The binding constraint is data: about 22 references per
  intent, and every rung that tried to substitute architecture for data lost.
- **Not a service.** Single process, no authentication, no packaging beyond this source tree,
  no CI, and the collection log is local-only and gitignored.

## Repository layout

| path | what is in it |
|---|---|
| `src/localalexa/` | the package: classifiers, cascade, scoring, CLI, collection, and the ladder harness |
| `tests/` | the suite, including the fold-leak assertions the measurement depends on |
| `data/` | the shipped corpus and tool registry — every number here reproduces from these two files |
| `docs/` | research records, measurement results and experiment briefs; indexed in `docs/README.md` |
| `experiments/` | the scripts that produced the results, including `experiments/optional/` for the ones needing a downloaded model |
| `results/` | committed artefacts: the ladder JSONs, historical records, the Cactus sample, the public-corpus transfer and the dell321 runs |
| `checkpoints/` | gitignored SetFit scratch output — model weights are not distributed |

## Licence and attribution

Apache-2.0, copyright 2026 Andrew Radu — the full text is in `LICENSE`. The corpus and registry
in `data/` are part of the project under the same terms; CLINC150, HWU64, SNIPS and ATIS are
not vendored and loading them is optional — all four were measured as substitutes for the
shipped corpus and none was adopted as training data (`docs/public-data-2026-11-10.md`).
Third-party work this builds on: scikit-learn for the linear models, SetFit and
`sentence-transformers/all-MiniLM-L6-v2` for the encoder rungs (downloaded, never vendored), and
the Cactus hybrid checkpoint that was measured and rejected. Prior research notes, including a
dated correction to an early number, are in `docs/RESEARCH.md`.
