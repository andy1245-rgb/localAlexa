# Evidence: the four-phase base-model survey, 2026-11-23

Every number in [`docs/base-model-survey-2026-11-23.md`](../../docs/base-model-survey-2026-11-23.md)
comes from a file in this directory. Nothing here was typed by hand; each file is the output of a
script under `experiments/survey/`, and the script that produced it is named beside it.

## Publication note

Host names and absolute paths were **normalised for publication**: the measurement machine appears
as `probe-box` and user home directories as `~`. Nothing else was altered — no number, no model
name, no verdict, no timestamp. The substitution is mechanical and applies only to those two
strings; if you re-run the scripts on your own machine the files will contain your own paths.

## Phase 1 — the landscape

| file | what it is |
|---|---|
| `phase1-A.md` … `phase1-E.md` | one survey per family: sentence encoders, modern embedding families, small decoder LLMs, ONNX/quantised/edge, and the SetFit recipe itself |
| `phase1-2-audit.md` | the audit of those rows against their sources — every row carrying a URL and a read date |
| `phase1-E-verdict.md` | whether SetFit is still the right few-shot recipe in 2026 |
| `doc-review.md` | the review of the assembled document against the brief |

## Phase 2 — narrowing

`phase2.md` — the shortlist of five, and **the eliminations with reasons**, which is half the value:
a reader should be able to see why a 1.7B model was not considered.

## Phase 3 — capability on the actual box

| file | what it is |
|---|---|
| `phase3-env.md` | the environment as measured: interpreter, torch, processor, memory |
| `phase3-table.json` | the summary table — load, 382 rows, p50/p95, snapshot size, resident memory, per candidate |
| `probe-<model>.json` | one file per probe, unaggregated, so any row in the table can be traced to its own measurement |
| `determinism/` | repeated runs of the same probe, used to bound the measurement span — two MiniLM canaries bracketed a 4% spread, which is the error bar on every comparison here |

## Phase 4 — the sweep

| file | what it is |
|---|---|
| `sweeps/` | per-encoder, per-fold output from `experiments/survey/setfit_sweep.py` |
| `sweeps/logs/` | the raw training logs, kept because a failed fold is only visible there |
| `sweeps-prev/` | the earlier pass, retained so the two runs can be compared rather than one being quoted |

## What the evidence shows, in one paragraph

The selection half ranks the incumbent first (0.9391, tied with granite at 0.9391/0.9394); the
validation half ranks `arctic-embed-xs` first at 0.9868, after it came third of four on the selection
half, 9.6 points back. **The ranking does not transfer between halves, so no candidate is supported**,
and the incumbent stands. Two estimates also failed against measurement: latency is not linear in
parameter count across architectures (granite at 30.3M measured faster than arctic at 22.6M), and the
absolute 10 ms budget is already broken by the incumbent on this box (15.02 ms), so the binding rule
in any future comparison must be a ratio, not an absolute.

That is the honest form of a negative result: not "we could not find anything", but "we looked
properly, here is everything we produced, and here is why nothing survived".
