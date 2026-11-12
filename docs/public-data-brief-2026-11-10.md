# Public intent data — does it raise localAlexa's ceiling?

2026-11-10. Operator approved exploring public intent datasets, and asked for an orchestrator with
parallel subagents on it. This is the brief.

## The problem it is meant to solve

localAlexa's corpus is 382 template-generated utterances across 110 phrasing families. Grouped
cross-validation already costs it 3.4 points against the rowwise number, and the strict
family-held-out estimate is worse. **No amount of slot filling fixes that**, because the missing
thing is not more examples — it is more *ways of saying the same thing*. Public intent corpora exist
for exactly this reason and contain what this corpus lacks: real speakers, real paraphrases,
real noise.

## What must be answered, in this order

1. **Licence, per dataset.** This repository is public and its README is read by people outside the
   project. A dataset we cannot redistribute must never be committed, and one we can must say so.
   Verify the licence from the dataset's own page — never from memory, never from a blog post.
2. **Label space.** Which of the dataset's intents map onto localAlexa's 17 tools? Where does the
   mapping fail, and does that failure reveal a gap in the label design (two tools that should be
   one, one tool that should be two)?
3. **The transfer measurement.** Two directions, both on the shared harness with grouped folds:
   - **train on the public data mapped to localAlexa's labels, evaluate on localAlexa's 382 rows**
   - **train on localAlexa's 382 rows, evaluate on the held-out public rows**
   A model that only works in one direction is telling you the corpora differ in a way that matters.
4. **The ceiling question.** With the external data added, does the grouped-CV accuracy on
   localAlexa's own rows improve — and by how much over the 89.79% grouped baseline? If it does not
   improve, say so plainly and stop: the bottleneck is then the label design or the task, not the
   data.

## Orchestration

Run the per-dataset recon **in parallel** — one subagent per dataset, asking each the identical set of
questions, so the answers are comparable:

- CLINC150 (~23k utterances, 150 intents)
- HWU64 / NLU benchmark suite (~25k, 64 intents)
- SNIPS (~14k, 7 intents)
- ATIS (~5k, 21 intents, older and noisier — include it because it is cheap and it is the standard
  low-water mark)

Use `fleet-manager fanout` for that stage — one template, four tasks, collected automatically. Each
subagent must return: the licence verbatim, the row count and split, the intent list, whether the raw
data downloaded and parsed here, three sample rows (public data, so quoting is fine), and a first-pass
mapping to the 17 tools with the rows that map to nothing counted.

Then **do the modelling yourself**, in one place, using one harness and one fold assignment. Do not
let four subagents each invent a metric; the whole point is comparability.

## Constraints

- **$0.** These datasets are free; nothing here needs a paid call.
- **Never commit the datasets.** They are large and mostly not ours to redistribute. Commit the
  licence, the row counts, the mapping table and the recipe — not the rows. `check:corpus`-style
  discipline applies even though this repo has no such guard: the corpus stays out.
- **Branch commits only, never push.**
- **The shared harness is `~/toolInterfacer/bench/harness/`** — use it if it is usable from here, and
  say so if it is not, because that is a finding about the harness.
- Do not touch `cascade.py` semantics or `SMART_ASSUMED_ACC`. Both are the operator's decisions.

## Deliverable

`docs/public-data-2026-11-10.md`: licence table, mapping table with the unmappable rows counted, the
two transfer measurements, the grouped-CV comparison against 89.79%, and a recommendation stated as
one of — *adopt these datasets*, *adopt one and say why*, or *adopt none, and here is the evidence*.
