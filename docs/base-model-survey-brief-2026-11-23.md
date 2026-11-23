# Four phases: find the right base model for fine-tuned intent classification

2026-11-10 (rev 2026-11-23). The operator's instruction: research base models for classifier
fine-tuning in **four phases**, phase 1 broad landscape, phase 2 narrowed to what we actually need,
and run the phases with parallel subagents. Phase 3 and 4 are defined here because he left them to
judgement.

Read `docs/ladder-results-2026-11-10.md` first. The incumbent is **SetFit on MiniLM-L6 (22M)**:
93.98% grouped, +4.17, `__none__` recall 0.84, 7.75 ms. That is the model to beat. The ensemble
(logreg + SetFit, 95.03% grouped) is the system to beat, at 7.85 ms.

## Constraints, stated once so every phase uses them

- **22 examples per intent**, 110 phrasing families, 18 labels. Fine-tuning at this size is on the
  edge; a bigger model is not automatically a better one.
- **Latency budget**: the linear path is 0.38 ms, the ensemble 7.85 ms. A candidate that cannot
  answer inside ~10 ms on CPU is out, whatever its accuracy — this may end up on a Raspberry Pi.
- **CPU only** where measurement happens. The target box (`dell321`) has an Intel Core Ultra 7 258V,
  torch 2.11.0+cpu, transformers 5.6.2, Python 3.11.9, no CUDA, an Arc 140V iGPU that nothing here
  will use.
- **Licence must be permissive**, because the repo is public and credits its dependencies.
- **Grouped CV by family**, never a random split. Report fold spread, never a bare mean.
- **Selection on one half, validation on the other.** Picking the best of N by mean accuracy on the
  same rows is how a sweep produces a number that does not transfer. This project has already been
  burned by exactly that twice in the sibling repo.

## Phase 1 — the landscape (parallel subagents)

One subagent per family, each returning a table with **the source it read and the date**, not
recollection:

| subagent | covers |
|---|---|
| A | sentence encoders for SetFit: MiniLM L6/L12, mpnet-base, and what has replaced them |
| B | modern embedding families: BGE, E5, GTE, Jina, Nomic, Stella — v2/v3 variants, sizes, dims |
| C | small decoder LLMs used as classifiers: Qwen3-0.6B/1.7B, SmolLM, Gemma-3-270M, and their fine-tune cost on CPU |
| D | ONNX / quantised / distilled options, and anything built for CPU-first or edge inference |
| E | SetFit itself: is it still the best few-shot recipe in 2026, or has something superseded it |

Each row: name, params, embedding dim, max sequence length, licence, whether it supports contrastive
fine-tuning, published speed, and the date the claim was read. **Mark vendor claims as vendor
claims.** Do not recommend in phase 1 — only describe.

## Phase 2 — narrow to what we need

Take phase 1 and apply the constraints. Deliver a **shortlist of five**, each with one sentence on
why it survives and one on what could disqualify it. Explicitly list what was eliminated and why —
the eliminations are half the value, and they stop anyone re-proposing a 7B model for a Pi.

## Phase 3 — verify capability on the actual box

For each shortlisted model, on `dell321`, prove it rather than trust it: download, load, embed the
382 rows, measure median and p95 latency, and record model size on disk and resident memory. No
fine-tuning yet. A model that cannot be downloaded, or that runs at 200 ms/utterance on CPU, is out
now rather than after a night of training.

## Phase 4 — the honest sweep

Fine-tune the survivors with the same recipe (SetFit contrastive + logistic head), same folds, same
metrics as the ladder: grouped accuracy, rowwise, per-fold spread, `__none__` recall, latency. Select
on one half, validate on the other, and report whether the winner **transfers**.

**The verdict has three acceptable forms**: a model beats MiniLM by more than the fold spread; nothing
beats it and the incumbent stands; or the ranking does not transfer between halves, in which case say
so and recommend the incumbent for stability. All three are results. A fourth form — "the best model
scored highest" — is not.

## How to run it

Phases 1 and 2 are research and can run fully in parallel. Phase 3 needs the box and can run in
parallel across models. Phase 4 trains, so it is serial per model but may run two at a time on the
8 cores.

Every phase writes its own section in `docs/base-model-survey-2026-11-23.md`. Nothing is committed
until phase 4 closes; the document carries all four sections so a reader sees the eliminations and
not only the winner.
