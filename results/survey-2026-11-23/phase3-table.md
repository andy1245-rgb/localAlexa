# Phase 3 — probe table

Generated 2026-11-24T01:40:28-05:00 by `experiments/survey/report_phase3.py` from 7 probe artefact(s). Numbers are the probe's own; nothing here re-measures.

## The pre-registered drop rule

- A model that cannot load is **out** (the brief: prove it rather than trust it).
- Latency ceiling: **22.53 ms** = 1.50 × the incumbent's measured p50 15.021 ms (`sentence-transformers/all-MiniLM-L6-v2`, probe p50). A model whose p50 exceeds the ceiling is dropped, at phase 3, before any training.

  The ratio default is 1.5: half again the incumbent's p50, which lands at 22.53 ms here — above the brief's ~10 ms CPU budget. The ratio carries the measuring box's speed with it, which is why it is a ratio and not an absolute: pass `--p50-ceiling-ratio` to move it. The pre-registered rule is the one in this file.

## The table

| model | load | params | dim | max_seq | licence | snapshot MB | RSS MB | embed ms/row | p50 ms | p95 ms | decision | against ceiling |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---|---|
| `sentence-transformers/all-MiniLM-L6-v2` (probe 2) | ok | 22.71M | 384 | 256 | apache-2.0 | 87.3 | 28.4 | 1.818 | 6.33 | 8.01 | **unresolved** | same model id, different measurement conditions — see Conflicts |
| `sentence-transformers/paraphrase-MiniLM-L3-v2` | ok | 17.39M | 384 | 128 | apache-2.0 | 67.0 | 21.7 | 1.029 | 8.39 | 12.80 | **pass** | p50 8.387 ms <= ceiling 22.532 ms |
| `ibm-granite/granite-embedding-30m-english` | ok | 30.30M | 384 | 512 | apache-2.0 | 60.3 | 50.6 | 1.696 | 11.70 | 13.58 | **pass** | p50 11.696 ms <= ceiling 22.532 ms |
| `Snowflake/snowflake-arctic-embed-xs` | ok | 22.57M | 384 | 512 | — | 87.1 | 34.2 | 2.231 | 12.13 | 17.67 | **pass** | p50 12.126 ms <= ceiling 22.532 ms |
| `sentence-transformers/all-MiniLM-L6-v2` (probe 1) | ok | 22.71M | 384 | 256 | apache-2.0 | 87.3 | 21.5 | 1.788 | 15.02 | 21.62 | **unresolved** | same model id, different measurement conditions — see Conflicts |
| `BAAI/bge-small-en-v1.5` | ok | 33.36M | 384 | 512 | mit | 128.3 | 45.3 | 3.364 | 28.19 | 38.57 | **drop** | p50 28.189 ms > ceiling 22.532 ms (1.50 x incumbent p50) |

*RSS MB is `memory.deltas_bytes.after_load_minus_before_load` — the model's own footprint, taken after torch and sentence-transformers were already imported, so the ~500 MB those cost is not in the column. MB here is MiB (1024² B).*

*The rule compares **p50** only — that is what `--p50-ceiling-ratio` names, and the brief's budget is a median. p95 is reported beside it because a model whose median fits and whose tail does not is the row a reader will want to look at, but it changes no decision here.*

## Decisions, one line each

- **pass** (3): `sentence-transformers/paraphrase-MiniLM-L3-v2`, `ibm-granite/granite-embedding-30m-english`, `Snowflake/snowflake-arctic-embed-xs`
- **drop** (1): `BAAI/bge-small-en-v1.5`
- **unresolved** (2): `sentence-transformers/all-MiniLM-L6-v2 (probe 2)`, `sentence-transformers/all-MiniLM-L6-v2 (probe 1)`

## Conflicting probes — not merged

The same model id was probed under different conditions. Both rows are printed and no decision is issued for either: averaging them, or keeping whichever came first, would be a number neither probe measured.

### `sentence-transformers/all-MiniLM-L6-v2`

2 probes differ on: `hostname`, `platform`, `python`, `torch`, `torch_threads`

| field | probe 1 | probe 2 |
|---|---|---|
| `hostname` | probe-box | andrew-Lenovo-7-14IAL7 |
| `platform` | Windows-10-10.0.26200-SP0 | Linux-6.8.0-149-generic-x86_64-with-glibc2.35 |
| `python` | 3.11.9 | 3.12.14 |
| `torch` | 2.11.0+cpu | 2.15.0+cpu |
| `torch_threads` | 8 | 10 |

- probe 1: `results/survey-2026-11-23/probe-canary-start-MiniLM-L6.json` — measured 2026-11-24T01:31:45-0500, p50 15.021 ms, loadavg —
- probe 2: `results/survey-2026-11-23/probe-sentence-transformers__all-MiniLM-L6-v2.json` — measured 2026-11-23T15:19:05-0500, p50 6.330 ms, loadavg 8.29

## Conditions each number was measured under

| model | host | torch | threads | loadavg | passes |
|---|---|---|---:|---:|---:|
| `sentence-transformers/all-MiniLM-L6-v2 (probe 2)` | andrew-Lenovo-7-14IAL7 | 2.15.0+cpu | 10 | 8.29 | 5 |
| `sentence-transformers/paraphrase-MiniLM-L3-v2` | probe-box | 2.11.0+cpu | 8 | — | 5 |
| `ibm-granite/granite-embedding-30m-english` | probe-box | 2.11.0+cpu | 8 | — | 5 |
| `Snowflake/snowflake-arctic-embed-xs` | probe-box | 2.11.0+cpu | 8 | — | 5 |
| `sentence-transformers/all-MiniLM-L6-v2 (probe 1)` | probe-box | 2.11.0+cpu | 8 | — | 5 |
| `BAAI/bge-small-en-v1.5` | probe-box | 2.11.0+cpu | 8 | — | 5 |

p50 across rows is only comparable when the box was as quiet for one as for the other. The ladder's lesson (`docs/ladder-results-2026-11-10.md`) is that a latency taken while something else trained is context, not a measurement.

## Warnings (advisory — they do not change a decision)

- Snowflake/snowflake-arctic-embed-xs: licence unread (no licence field in the model card) — check by hand before shipping
- this probe set spans 2 hosts (probe-box, andrew-Lenovo-7-14IAL7): a ceiling derived on one host is not a ceiling on another, and the p50 column is not comparable across them — split the set by host before trusting a drop

## Notes

- `sentence-transformers/all-MiniLM-L6-v2` was probed again under identical conditions: kept `results/survey-2026-11-23/probe-canary-start-MiniLM-L6.json` (2026-11-24T01:31:45-0500), ignored `results/survey-2026-11-23/probe-canary-end-MiniLM-L6.json` (2026-11-24T01:36:55-0500) — a repeat measurement, not a conflict
- the incumbent `sentence-transformers/all-MiniLM-L6-v2` was probed more than once; the ceiling uses probe p50 for `sentence-transformers/all-MiniLM-L6-v2` from `results/survey-2026-11-23/probe-canary-start-MiniLM-L6.json`

## Inputs

- `results/survey-2026-11-23/probe-L3-v2.json`
- `results/survey-2026-11-23/probe-arctic-xs.json`
- `results/survey-2026-11-23/probe-bge-small-en-v15.json`
- `results/survey-2026-11-23/probe-canary-end-MiniLM-L6.json`
- `results/survey-2026-11-23/probe-canary-start-MiniLM-L6.json`
- `results/survey-2026-11-23/probe-granite-30m.json`
- `results/survey-2026-11-23/probe-sentence-transformers__all-MiniLM-L6-v2.json`

