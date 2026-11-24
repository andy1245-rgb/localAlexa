# Survey harness — phase 3 probe and phase 4 SetFit sweep

Two runnable scripts that measure a Hugging Face encoder the way the ladder
measured MiniLM-L6, so a sweep across base models is one command per model, and
a batch runner that does the whole list serially.

Nothing here is a new experiment design. It is the ladder's measurement, applied
to a different encoder:

| what | where it comes from |
|---|---|
| the 382-row corpus and the registry | `localalexa.data.load_routing_set` / `load_registry` |
| the 5-fold phrasing-family assignment | read out of `results/ladder/rung5-ensemble.json` |
| each fold's training pool (contamination filter) | `localalexa.ladder.pools.fold_pool` |
| the SetFit recipe (pairs, epochs, iterations, batch, seed) | `localalexa.ladder.setfit_rung` constants |
| accuracy / ECE / reject curve / `__none__` recall / ledger / bootstrap CI | `localalexa.ladder.run.block_metrics` |
| single-utterance latency | `localalexa.ladder.run.timed_latencies`, and `latency_pass.py`'s pass shape |
| the JSON field names | `results/ladder/*.json` |

## Requirements

Both scripts need `torch` (CPU build), `sentence-transformers`, `setfit`,
`datasets`, `scikit-learn`, `numpy`, `huggingface_hub`, and the repo installed
(`pip install -e .`). `psutil` is used when present and is not required.

```bash
# this checkout's torch environment
.venv-embed/bin/python experiments/survey/probe_model.py --help
```

On `probe-box`, use an interpreter that has those packages — not the system Python
unless it already does:

```bat
py -3.11 -m venv .venv-survey
.venv-survey\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv-survey\Scripts\python -m pip install setfit sentence-transformers scikit-learn numpy
.venv-survey\Scripts\python -m pip install -e .
.venv-survey\Scripts\python experiments\survey\probe_model.py --model sentence-transformers/all-MiniLM-L6-v2 --out results\survey-2026-11-23\probe-minilm.json
```

Paths are built from `pathlib` and every separator is `os.path.join`-equivalent,
so the same command works under `cmd`, PowerShell and bash. Each script is one
process, single-threaded at the script level (torch's own thread pool is what it
is and is recorded per row), and CPU only — no CUDA code path exists.

Three modules are shared by both scripts and are where the failure handling
lives: `survey_env.py` (conditions, memory, atomic write), `survey_load.py`
(online load, then the local cache) and `survey_dtype.py` (the float32 cast and
its assertion). Everything below describes what they do and what the artefacts
record about it.

---

## Phase 3 — `probe_model.py`

```bash
.venv-embed/bin/python experiments/survey/probe_model.py \
    --model sentence-transformers/all-MiniLM-L6-v2 \
    --out results/survey-2026-11-23/probe-sentence-transformers__all-MiniLM-L6-v2.json
```

| flag | default | meaning |
|---|---|---|
| `--model` | required | Hugging Face model id |
| `--out` | required | JSON path to write (parent dirs created) |
| `--passes` | 5 | latency repeats; the reported number is the median of the per-pass medians |
| `--limit` | all 382 | embed only the first N rows (smoke tests) |
| `--batch-size` | 32 | batch size for the bulk embed (latency is always one row per call) |
| `--offline` | off | load from the local cache only, no online attempt (see "Loading: online, then the cache") |

What it does, in order: import `sentence-transformers` (timed), read the host
facts, load the corpus through the repo's loader, load the model (timed), read
the model card (licence) and the snapshot's size on disk, embed all 382 rows,
sample resident memory, then measure latency.

Latency is measured the way `experiments/ladder/latency_pass.py` measured it:
one utterance per call, warm, one process, `--passes` repeats, and the reported
`p50_ms` is the **median of the per-pass medians**. The row carries the
conditions (`torch_threads`, `loadavg_1m`, `python`, `torch` version) because a
number without them is not a measurement.

The latency ceiling that decides whether a probed model is worth a phase-4 fit is
`--p50-ceiling-ratio` × the incumbent's measured p50, and that ratio **defaults
to 1.5** (`report_phase3.py`, `DEFAULT_RATIO`) — half again MiniLM-L6's own p50,
not an absolute millisecond budget.

A model that cannot be imported, downloaded or loaded still writes a row:
`status: "error"`, `failed_stage`, and the error text. The script exits 1 and
does not print a stack trace.

### Probe JSON schema

```json
{
  "schema": "survey-probe/1",
  "phase": 3,
  "model_id": "sentence-transformers/all-MiniLM-L6-v2",
  "status": "ok | partial | error",
  "error": null,
  "failed_stage": null,
  "measured_at": "2026-11-23T14:54:42-0500",
  "environment": {"hostname": "...", "platform": "...", "cpu_count": 12,
                  "torch": "2.15.0+cpu", "torch_threads": 10, "cuda_available": false,
                  "python": "3.12.14", "transformers": "...", "setfit": "...",
                  "sentence_transformers": "...", "numpy": "...", "sklearn": "...",
                  "huggingface_hub": "...", "datasets": "...", "psutil": "...",
                  "loadavg_1m": 8.942},
  "corpus": {"source": "localalexa.data.load_routing_set (...)", "n_rows": 382, "truncated": false},
  "load": {"status": "ok", "load_ms": 382.1, "import_ms": 2104.0, "error": null,
           "mode": "online", "forced": false, "offline_reason": null,
           "attempts": [{"mode": "online", "ok": true, "error": null, "seconds": 0.396}],
           "dtype": {"requested": "float32", "device": "cpu",
                     "before": ["float32"], "after": ["float32"], "coerced": false,
                     "embeddings": {"torch": "float32", "numpy": "float32"}}},
  "model": {"params": 22713216, "embedding_dim": 384, "max_seq_length": 256,
            "license": "apache-2.0", "license_source": "huggingface_hub.model_info cardData",
            "license_error": null, "gated": false, "pipeline_tag": "sentence-similarity",
            "snapshot_path": "/home/.../snapshots/1110a2...", "snapshot_bytes": 91578415,
            "snapshot_files": 11, "snapshot_source": "...", "snapshot_error": null},
  "embeddings": {"n_rows": 382, "dim": 384, "dtype": "float32", "wall_ms": 668.2,
                 "ms_per_row": 1.749, "mean_l2_norm": 1.0, "min_l2_norm": 1.0,
                 "max_l2_norm": 1.0, "finite": true},
  "memory": {"readings": {"start": 33046528, "after_import": 532197376,
                          "before_load": 534355968, "after_load": 565379072,
                          "after_embed": 650059776, "after_latency": 650440704},
             "method": "psutil.Process().memory_info().rss (current)",
             "deltas_bytes": {"after_load_minus_before_load": 30769152,
                              "after_embed_minus_after_load": 84680704}},
  "latency": {"passes": 5, "n_sentences_per_pass": 382, "warmup_calls": 8,
              "pass_p50_ms": [4.803, 8.078, 8.623, 8.998, 8.937],
              "pass_p95_ms": [7.313, 15.642, 14.928, 12.419, 13.951],
              "p50_ms": 8.623, "p95_ms": 13.951, "p50_spread_ms": 4.195,
              "method": "single utterance per call, warm, one process, median of the per-pass medians"},
  "conditions": {"protocol": "...", "passes_requested": 5, "warmup_calls": 8,
                 "batch_size_single_call": 1, "embed_batch_size": 32,
                 "latency_source": "...", "passes_recorded": 5, "loadavg_1m_at_end": 9.368},
  "stage_errors": {}
}
```

Fields that need a word of explanation:

- `load.mode` is which path produced the encoder: `online`, or
  `offline-fallback` when the cached retry did (and when `--offline` was passed).
  `load.attempts` has one entry per attempt, so a fallback is two lines and a
  forced offline load is one. A probe whose load failed carries the same fields
  plus the `LoadFailure` naming both attempts.
- `load.dtype` is the pre-fit cast, recorded rather than assumed:
  `before`/`after` are the distinct parameter dtypes of the body, `coerced` says
  whether anything was cast, and `embeddings.torch` is what a real forward pass
  handed back. `embeddings.numpy` is recorded beside it only to show the two can
  disagree — for a bf16 body numpy refuses the tensor outright, which is why the
  assertion reads the tensor.
- `memory.method` names the source of every reading, because the fallbacks are
  not equal: `psutil` and `/proc/self/status VmRSS` and Windows `psapi
  WorkingSetSize` are *current* resident bytes; `ru_maxrss` is a **peak** and is
  labelled as such. `SURVEY_DISABLE_PSUTIL=1` forces the fallback path so it can
  be exercised.
- `memory.deltas_bytes.after_load_minus_before_load` is the model's footprint.
  `before_load` is taken after torch and sentence-transformers are imported, so
  the 500 MB those cost is in `after_import_minus_start`, not in the model's
  delta.
- `snapshot_bytes` is the size of the downloaded snapshot, following the
  snapshot's symlinks into the Hub's blob store and counting each inode once.
- `embeddings.mean_l2_norm` is the encoder's own output geometry — some models
  (MiniLM-L6 among them) normalise inside the module chain, and that is worth
  knowing before blaming the head for anything.
- `status` is `partial` when a stage other than load failed and the rest of the
  row still measured something.

---

## Phase 4 — `setfit_sweep.py`

```bash
.venv-embed/bin/python experiments/survey/setfit_sweep.py \
    --model sentence-transformers/all-MiniLM-L6-v2 \
    --out results/survey-2026-11-23/sweeps/sentence-transformers__all-MiniLM-L6-v2.json
```

| flag | default | meaning |
|---|---|---|
| `--model` | required | Hugging Face model id |
| `--out` | required | JSON path to write |
| `--folds` | `all` | comma-separated fold ids, e.g. `0,1` for a partial run |
| `--artifact` | `results/ladder/rung5-ensemble.json` | where the fold assignment is read from |
| `--passes` | 5 | repeats for the dedicated latency pass (`0` disables it) |
| `--no-rowwise` | off | skip the extra 200-reference fit |
| `--no-determinism-check` | off | skip the second fit of the first fold |
| `--no-normalize-embeddings` | off | feed the head raw body embeddings (provenance only) |
| `--keep-predictions` | off | include the per-fold prediction vectors |
| `--offline` | off | load every encoder from the local cache only; no online attempt |

What it runs:

Before any fold is touched: the encoder is **loaded** through
`survey_load.load_with_fallback` (online, then the local cache — `load.mode`) and
**cast to float32** with the cast asserted (`load.dtype`). Both are described
under "Loading: online, then the cache" and "The encoder's dtype" below; a
failure in either is a fatal stage, not a fold.

1. **Fold assignment, read and verified.** `fold_assignments["5"]` from the
   artefact above. The script checks the corpus has 382 rows, that the family
   sets are identical, that every fold holds 76–77 rows and that the recomputed
   row counts equal the artefact's. Any mismatch → exit 2, an error row with
   `refused: true`, and no training. Regenerating the folds is only used as a
   cross-check (`regenerated_matches_artifact`), never as the source.
2. **Grouped folds.** For each fold: `fold_pool` drops the references that
   render into the held-out families, a fresh `SurveySetFitPredictor` is fitted,
   latencies are timed one utterance at a time, and predictions are pooled.
3. **Metrics.** `block_metrics` over the pooled rows — the same accuracy, ECE,
   reject curve, `__none__` recall, false-none rate, per-intent ledger and
   cluster-bootstrap CI the ladder reports, under the same field names.
4. **Rowwise** (unless `--no-rowwise`): the ladder's `evaluate_rowwise`, i.e.
   train on all 200 references and score all 382 rows.
5. **Latency pass** on the last fold's fitted model, over all 382 rows, in the
   `latency_pass.py` shape (run alone, repeat, median).
6. **Determinism receipt**: the first fold is fitted a second time from scratch
   and the two prediction vectors are diffed.
7. **Halves**: folds 0–2 (selection) and folds 3–4 (validation), each with its
   own pooled grouped accuracy.

The recipe, written into every artefact under `recipe`: SetFit contrastive pairs
from the fold's training pool, `num_iterations=8`, `num_epochs=1`,
`batch_size=16`, `seed=0`, and SetFit's logistic head on **L2-normalised** body
embeddings. The constants are imported from `localalexa.ladder.setfit_rung`; the
one thing this script cannot inherit is `normalize_embeddings`, which
`SetFitPredictor` does not expose, so `survey_setfit.py` writes out that
`SetFitModel.from_pretrained(..., normalize_embeddings=True)` call and keeps the
rest identical. Both settings were measured on MiniLM fold 0 before choosing:
same accuracy to four decimals (0.9221), so the choice is not buying a number —
it is making the head's geometry explicit.

### Sweep JSON schema

Ladder field names are marked **→ ladder**; the rest is survey-specific.

```json
{
  "schema": "survey-sweep/1",
  "phase": 4,
  "model_id": "...",
  "status": "ok | partial | error",
  "error": null,
  "failed_stage": null,
  "failed_category": null,
  "traceback": null,
  "missing": [],
  "stage_errors": {},
  "stage_order": ["setup", "environment", "load", "dtype", "model_facts", "latency_pass",
                  "rowwise", "halves", "comparisons", "determinism", "provenance"],
  "measured_at": "...",
  "environment": {"...": "same block as the probe"},
  "load": {"status": "ok", "load_ms": 1420.0, "error": null,
           "mode": "online | offline-fallback", "forced": false, "offline_reason": null,
           "attempts": [{"mode": "online", "ok": true, "error": null, "seconds": 3.1}],
           "dtype": {"requested": "float32", "device": "cpu", "before": ["float32"],
                     "after": ["float32"], "coerced": false,
                     "embeddings": {"torch": "float32", "numpy": "float32"}},
           "fit_modes": ["online"]},
  "model": {"params": 22713216, "embedding_dim": 384, "max_seq_length": 256,
            "license": "apache-2.0", "snapshot_bytes": 91578415},
  "bench": {"n_rows": 382, "n_families": 110, "n_labels": 18, "n_reference_rows": 200},  **→ ladder**
  "fold_assignments": {"5": {"source": ".../rung5-ensemble.json", "key": "fold_assignments.5",
                             "n_folds": 5, "n_families": 110, "rows_per_fold": [77,77,76,76,76],
                             "verified": {"...": "..."},
                             "regenerated_matches_artifact": true,
                             "family_fold": {...}}}                                     **→ ladder**
  "recipe": {"name": "rung-3-setfit", "encoder": "...", "num_iterations": 8,
             "num_epochs": 1, "batch_size": 16, "normalize_embeddings": true,
             "head": "setfit.LogisticRegression on body embeddings", "seed": 0},
  "seeds": {"setfit": 0, "torch": 0, "numpy": 0, "sklearn": 0, "note": "..."},
  "folds_run": [0, 1, 2, 3, 4],
  "halves": {"selection":  {"folds": [0,1,2], "folds_run": [0,1,2], "complete": true,
                            "n_folds": 3, "n_rows": 230, "grouped_accuracy": 0.9391,
                            "mean_fold_accuracy": 0.9394, "per_fold_accuracy": {"0": 0.9221},
                            "baseline_per_fold_accuracy": {...},
                            "baseline_mean_fold_accuracy": 0.9394, "definition": "..."},
             "validation": {"folds": [3,4], "n_rows": 152, "grouped_accuracy": 0.9408, "...": "..."},
             "rule": "select on folds 0-2, quote folds 3-4"},
  "runs": [
    {"rung": "r3-setfit-minilm", "note": "...", "protocol": "grouped", "fold_count": 5,   **→ ladder**
     "n_folds": 5, "complete": true, "folds": [0,1,2,3,4], "folds_scored": [0,1,2,3,4],
     "train_rows": [194,199,195,195,194],
     "fold_rows": [77,77,76,76,76], "n_rows": 382, "latency_ms": {"p50": 8.22, "p95": 11.504},
     "per_fold_accuracy": [0.9221, 0.8961, 1.0, 0.9079, 0.9737],
     "per_fold_accuracy_by_fold": {"0": 0.9221, "1": 0.8961, "2": 1.0, "3": 0.9079, "4": 0.9737},
     "fold_spread": {"mean": 0.94, "min": 0.8961, "max": 1.0, "sd": 0.0448},
     "per_fold": [{"fold": 0, "n_rows": 77, "train_rows": 194, "correct": 71,
                   "accuracy": 0.9221, "fit_seconds": 57.76, "latency_ms": {"p50": 8.145}}],
     "accuracy": 0.9398, "ece": 0.53, "mean_confidence": 0.3843,                       **→ ladder**
     "reject_curve": {...}, "none_recall": 0.84, "false_none_rate": 0.0056,              **→ ladder**
     "ledger": {...}, "cluster_bootstrap_ci95": [0.8909, 0.9823]},                       **→ ladder**
    {"rung": "...", "protocol": "rowwise", "fold_count": null, "accuracy": 0.9424, "...": "..."}
  ],
  "latency_passes": {"passes": 5, "p50_ms": 7.662, "p95_ms": 10.821, "pass_p50_ms": [...],
                     "conditions": {"measured_on": "...", "torch_threads": 10, "loadavg_1m": 0.697}},
  "comparisons": [
    {"rung": "...", "baseline": "r3-setfit-minilm", "fold_count": 5, "protocol": "grouped",
     "paired": {"mean_points": 0.0, "sd_points": 0.0, "min_points": 0.0, "max_points": 0.0,
                "folds_better": 0, "folds_worse": 0, "folds_tied": 5},
     "baseline_source": ".../rung5-ensemble.json"},
    {"rung": "...", "baseline": "r3-setfit-minilm", "protocol": "rowwise",
     "paired": null, "delta_points": 0.0}
  ],
  "incumbent": {"rung": "r3-setfit-minilm", "encoder": "sentence-transformers/all-MiniLM-L6-v2",
                "source": ".../rung5-ensemble.json", "grouped_accuracy": 0.9398,
                "rowwise_accuracy": 0.9424, "per_fold_accuracy": [0.9221, 0.8961, 1.0, 0.9079, 0.9737],
                "latency_p50_ms": 7.944, "none_recall": 0.84},
  "determinism": {"checked": true, "fold": 0, "identical_predictions": true, "rows_differing": 0,
                  "max_confidence_abs_diff": 0.0, "accuracy_first": 0.9221,
                  "accuracy_second": 0.9221},
  "provenance": {"reproduces_artifact": true, "per_fold_delta_from_artifact_points": [0.0]},
  "trainer_output_dir": "/tmp/surveysweep-6ydl2spy",
  "skipped": [],
  "protocol_notes": {"grouped": "...", "rowwise": "...", "reject_curve": "...", "selection": "..."}
}
```

How to read it:

- **`halves.*.grouped_accuracy`** is the number to compare across models. Pick
  the winner on `halves.selection`, then quote that model's
  `halves.validation` — which no model in the sweep was chosen on. The halves
  are disjoint family sets by construction of the family-grouped assignment.
- **`comparisons[0].paired`** is the per-fold difference against MiniLM-L6 on
  the same folds: `mean_points` ± `sd_points`, with `folds_better/worse/tied`.
  A mean smaller than its own sd is not a win; that rule is the ladder's.
- **`folds_run` / `complete`** say whether the row is the whole sweep or the
  partial one used to prove the machinery. A partial row is still honest — every
  metric is computed only over the folds that ran — but it is not a sweep.
  `folds_scored` (and `per_fold_accuracy_by_fold`) name the folds that actually
  produced a number, so a row reduced to four folds can never be read as if its
  `per_fold_accuracy` list were folds 0-3 of five.
- **`trainer_output_dir`** is where the fits in this process wrote their trainer
  output (see "Temp directories" below); it is recorded so a Windows run that
  dies on the filesystem says where it was writing.
- **`latency_passes.p50_ms`** is the latency figure to quote. The per-fold
  `latency_ms` inside `runs[0]` is taken during the sweep, so anything else
  training on the same box lands in it: this same artefact, produced at loadavg
  10, read 9.711 ms for the dedicated pass and 446.057 ms p95 pooled in-fold,
  against 7.662 ms and 11.504 ms for the run recorded above at loadavg 0.7. Same
  lesson, same fix as `docs/ladder-results-2026-11-10.md`: the pass that runs
  alone is the number, the in-fold timings are context.
  `latency_passes.conditions.loadavg_1m` is the box's load average when the run
  started (`environment.loadavg_1m`).
- **`determinism.identical_predictions: true`** is the receipt that a re-fit of
  the same fold produced the same rows. `false` means the two fits disagreed and
  the row's numbers should not be trusted at face value.

### The encoder's dtype — cast once, on the body, before anything is fitted

`ibm-granite/granite-embedding-30m-english` ships bfloat16 weights. Loaded on CPU
they stay bfloat16, the SetFit head is a float32 logistic regression, and the
first pass through the body raises

```
RuntimeError: Found dtype Float but expected BFloat16
```

Reproduced twice on that box. The artefact of the first attempt said
`failed_stage: "harness"` — the generic label for a failure that named neither
the model nor the dtype. Three things changed.

1. **The cast is its own stage, on the encoder, immediately after the load.**
   There is no CUDA path here, so it is `model_body.float()` on the CPU body — the
   thing whose weights are bf16 — and every later fit in that process inherits
   it. It is deliberately *not* inside the loss: a `to(float32)` there would hide
   which tensor disagreed, and it would change the numbers of one model quietly
   while every other row of the sweep stayed as it was.
2. **The cast is asserted, not assumed.** `load.dtype` records the distinct
   parameter dtypes before and after, and the dtype a real forward pass
   (`encode(..., convert_to_tensor=True)`) hands the head. Those are two different
   facts and the row now has both. The phase-3 probe had only the second, read
   through numpy, and it read `embeddings.dtype: float32` for granite while that
   model's fit raised the bf16 error: a forward pass that happens to run in
   float32 says nothing about the dtypes of the weights, and numpy cannot
   describe a bf16 tensor either — measured here, `np.asarray` on one raises `Got
   unsupported ScalarType BFloat16`. So the probe now writes `load.dtype` too,
   and the two cases the assertion can catch are a cast that did not take and a
   body whose parameters are float32 while its embeddings are not.
3. **A cast that does not take fails as `dtype`, with the dtype in the message.**
   `failed_stage: "dtype"`, and `error` reads e.g. `encoder is still bfloat16
   after the cast to float32 (loaded as bfloat16); the SetFit head trains in
   float32`. A checkpoint that already loads in float32 is untouched and records
   `coerced: false`, so the cast is visible in the row whether or not it did
   anything.

`test_dtype_offline.py` drives the four cases against a stub body with real torch
dtypes: the cast that takes, the cast that refuses, a `float()` that returns
without casting anything, and parameters float32 with bf16 embeddings.

### Loading: online, then the cache

`RuntimeError: Cannot send a request, as the client has been closed.` hit all
four models in one run at `failed_stage: load`, seconds apart, while
`huggingface.co:443` was demonstrably reachable and every one of those models was
already in the local HF cache. The client that died (`httpx`) is process-wide —
the closed pool is shared — so the first blink fails every load after it. A
cached model must never fail a sweep because the network blinked.

So `survey_load.load_with_fallback` loads online once, and on a Hub-shaped error
sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` and retries once from the
cache. `load.mode` says which path produced the model:

| `load.mode` | when |
|---|---|
| `online` | the first attempt succeeded; nothing was switched |
| `offline-fallback` | the online attempt failed, or `--offline` was passed, and the cached path produced the model |

Three details that are easy to get wrong, and are the reason this is a module
rather than a `try/except` around the loader:

- **The environment variable alone does nothing after the import.**
  `huggingface_hub.constants.HF_HUB_OFFLINE` is computed once, at module import;
  a process that has already imported the Hub ignores a later `os.environ` write,
  and by the time a model is being loaded it has. Both are set — the variable and
  the constant (`transformers` imports `is_offline_mode` from the Hub, and that
  function reads the constant at call time, so one patch reaches both) — and the
  loader also passes `local_files_only=True`, so the retry does not depend on a
  library internal staying where it is across versions.
- **The switch is sticky for the rest of the process.** Once the pool has been
  closed, an online attempt per fit is four more failures rather than four more
  chances; after one fallback every later load in that process starts on the
  cached path. `load.forced` records whether `--offline` asked for it up front
  and `load.offline_reason` carries the error that caused it.
- **A non-Hub error is not retried.** A bad model id or a bad config fails the
  same way from the cache, and retrying it costs a second and buries the real
  message. `load.attempts` records however many attempts there were, each with
  its `exception_type`, its `hub_error` verdict and its wall time.

When either attempt fails irrecoverably the row still names what was tried:
`error` is always a `LoadFailure`, listing every attempt and the original error
beside its path — one attempt for a non-Hub error or a forced offline load, two
for a fallback that also failed, e.g. `could not load 'x' (online: RuntimeError:
Cannot send a request, as the client has been closed.; offline-fallback: OSError:
no such file in the cache)`. `load.attempts` carries the same lines in structured
form. `--offline` is on `probe_model.py`, `setfit_sweep.py` and `run_survey.py`,
and it is what the target box should use for re-runs: every model the survey
measures is in its cache already.

### Partial sweeps — `status`, `traceback`, `missing`

Every stage is attempted through `sweep_stages.StageBook`, which records the
stage name, the error text, the exception type, the wall time and the **full
traceback** when it raises. The row's `status` is then one of three things, and
the difference is not cosmetic:

| `status` | means | what is in the artefact |
|---|---|---|
| `ok` | every stage that was asked for ran; every fold measured | the full sweep |
| `partial` | something raised or could not be measured, but folds did | the fold numbers that survived, plus `missing` and `stage_errors` |
| `error` | no fold was measured at all | `failed_stage`, `error` and `traceback` |

- **`traceback`** is the full traceback text of the reported failure (top-level),
  and every failed stage carries its own under `stage_errors[<stage>].traceback`.
  This is the field whose absence cost a whole sweep: the first Windows run wrote
  `failed_stage: "harness"`, `OSError: [Errno 22] Invalid argument`, and no way
  to tell which of the seven fits raised it.
- **`failed_stage`** names the stage, or `fold <n>` for a fold that could not be
  fitted. A fatal stage (`load`, `dtype`, a bug in the assembly, which reports as
  `harness`) outranks a fold failure, which outranks an optional stage — the
  failure that changed which numbers exist is the one reported.
- **`failed_category`** is the taxonomy beside it, and it is what makes the two
  live failures readable. `fold 0` alone said nothing about granite; `fold 0` +
  `dtype` says "the pre-fit cast, on the first fold":

| `failed_category` | `failed_stage` says | means | fatal |
|---|---|---|---|
| `load` | `load` | the encoder could not be loaded, online or from the cache | yes |
| `dtype` | `dtype`, or `fold <n>` | it loaded in a dtype the float32 head cannot train with, or the cast did not take | yes / no |
| `fit` | `fold <n>` | that fold's contrastive fine-tune plus head fit raised | no |
| `rowwise` | `rowwise` | the 200-reference rowwise fit raised | no |
| `determinism` | `determinism` | the second fit of the first fold raised | no |
| `latency` | `latency_pass` | the dedicated latency pass raised | no |
| `harness` | `harness` | a bug in the assembly itself | yes |
| `fold_assignment` | `fold_assignment` | the artefact's folds do not match the corpus (exit 2) | yes |

  The fold id is kept in front of the category because the fold is what a reader
  needs; `stage_errors.folds.<n>.category` carries the same bucket, so a fold
  failure can be filtered without parsing strings. `sweep_stages.category_of_stage`
  is the one place the mapping lives.
- **`missing`** lists every part that did not run: failed stages and failed folds
  (`"fold 0"`). It is empty on an `ok` row.
- **`skipped`** is a different thing and stays in its own list: a stage the
  operator turned off (`--no-rowwise`, `--no-determinism-check`, `--passes 0`) is
  not a failure and does not make a row partial.
- **Skippable vs fatal.** `environment`, `model_facts`, the latency pass, the
  rowwise fit, the halves, the comparisons, the determinism refit and the
  provenance block are all attempted and recorded but never abort the run; a
  failed fold is recorded and the remaining folds still run. Only `load` and
  `dtype` are fatal, because nothing downstream — a fit, a metric, a latency — is
  measurable without an encoder that loaded and trains. A half whose folds are
  incomplete says so (`complete: false`, `missing_folds: [0]`) instead of quoting
  a pooled number as if it covered the half.
- **Exit code** follows the artefact: `0` only for `ok`, `1` for `partial` or
  `error`, `2` for a refused fold assignment. `run_survey.py` carries a partial
  row into `summary.json` with its `missing` list and prints it on stderr, so a
  partial sweep is never mistaken for a completed one.

The Windows-simulation test (`test_sweep_partial.py`) drives `sweep()` with a
stub predictor and injects `OSError(EINVAL)` into the trainer's per-fit output
directory, the rowwise fit, the latency pass and the determinism refit; in every
case the artefact comes out `partial`, the stage is named, the traceback is
present and the surviving fold numbers are there.

### Temp directories

A fit writes about 260 MB of trainer output to disk even though nothing reads it
back: `SentenceTransformerTrainer` saves the body on `on_train_end` whether or not
checkpointing is on, and `_save_checkpoint` adds `optimizer.pt`, `scheduler.pt`
and `rng_state.pth` beside the weights. `survey_setfit` therefore creates **one**
directory per process (`%TEMP%/surveysweep-*`, or `/tmp` here), gives each fit a
fresh `fit-NN` subdirectory inside it, clears the previous fit's subdirectory when
it starts the next one, and removes the whole root once, at process exit.

That replaces `tempfile.mkdtemp()` per fit plus `shutil.rmtree(...,
ignore_errors=True)` in a `finally`. A five-fold sweep with the rowwise and
determinism fits is seven create/write/list/delete cycles, against one in the
smoke run — and on Windows every one of those is a chance to delete a directory
whose files a trainer, an mmap or a cache lock is still holding, with the failure
hidden by `ignore_errors`. The peak is now two fits on disk (about 520 MB) rather
than one at a time with no bound on the churn, and no fit ever deletes the tree
it has just written or writes into a tree it is deleting. The one removal that
must not leak is the exit-time one.

---

## Runner — `run_survey.py`

```bash
# the whole list, one model at a time (the default)
.venv-embed/bin/python experiments/survey/run_survey.py \
    --models sentence-transformers/all-MiniLM-L6-v2,BAAI/bge-small-en-v1.5,intfloat/e5-small-v2 \
    --out-dir results/survey-2026-11-23/sweeps

# or from a file, one id per line, # comments allowed
.venv-embed/bin/python experiments/survey/run_survey.py \
    --models-file results/survey-2026-11-23/models.txt

# two at a time, only when there are cores to spare
.venv-embed/bin/python experiments/survey/run_survey.py \
    --models-file results/survey-2026-11-23/models.txt --parallel 2
```

| flag | default | meaning |
|---|---|---|
| `--models` | — | comma-separated ids, repeatable |
| `--models-file` | — | file of ids, one per line, `#` comments |
| `--out-dir` | `results/survey-2026-11-23/sweeps` | per-model JSON, logs and `summary.json` |
| `--parallel` | 1 | models at a time; 2 is opt-in, because phase 4 trains |
| `--folds`, `--passes`, `--artifact`, `--no-rowwise`, `--skip-determinism-check`, `--no-normalize-embeddings`, `--offline` | — | passed straight through |
| `--python` | this interpreter | interpreter used for each sweep |
| `--skip-existing` | off | keep an existing `ok` artefact instead of re-training |

Serial by default, one **process** per model: an encoder's allocator, thread pool
and temp checkpoints cannot leak into the next row, and a model that crashes
takes down its own interpreter instead of the sweep. `--parallel 2` starts two
of those processes at once — the brief allows it on eight cores, but two
CPU-bound fine-tunes on one box distort each other's latency numbers, which is
why it is a flag and not the default.

Per model it writes `<slug>.json` (the sweep artefact), `logs/<slug>.log` (the
sweep's stdout), and at the end `summary.json`:

```json
{"schema": "survey-runner/1", "started_at": "...", "finished_at": "...",
 "folds": "all", "parallel": 1, "artifact": ".../rung5-ensemble.json",
 "models": [{"model_id": "...", "status": "ok", "exit_code": 0,
             "grouped_accuracy": 0.9398, "rowwise_accuracy": 0.9424,
             "selection_grouped_accuracy": 0.9391, "validation_grouped_accuracy": 0.9408,
             "delta_mean_points": 0.0, "delta_sd_points": 0.0,
             "p50_ms": 8.22, "p50_ms_idle_pass": 7.662, "determinism_identical": true,
             "failed_stage": null, "failed_category": null, "load_mode": "online",
             "json": "...", "log": "...", "wall_seconds": 833.0, "error": null}]}
```

Exit code is 1 if any model did not finish `ok`, but the `summary.json` is
written either way — a failed model is a row with an error, not a gap.

---

## Worked example — MiniLM-L6, measured on this linux box

Nothing below is a projection: these are the artefacts the verification run
wrote, and the commands that produced them.

```bash
.venv-embed/bin/python experiments/survey/probe_model.py \
    --model sentence-transformers/all-MiniLM-L6-v2 \
    --out results/survey-2026-11-23/probe-sentence-transformers__all-MiniLM-L6-v2.json

.venv-embed/bin/python experiments/survey/setfit_sweep.py \
    --model sentence-transformers/all-MiniLM-L6-v2 \
    --out results/survey-2026-11-23/sweeps/sentence-transformers__all-MiniLM-L6-v2.json
```

**Phase 3 probe** (`probe-sentence-transformers__all-MiniLM-L6-v2.json`), on
`probe-box-b`, python 3.12.14, torch 2.15.0+cpu, 10 torch threads:

| field | value |
|---|---|
| load | ok, 389.3 ms (import of `sentence-transformers` 3198.5 ms) |
| params / dim / max_seq_length | 22,713,216 / 384 / 256 |
| licence | `apache-2.0` (hub cardData) |
| snapshot on disk | 91,578,415 B in 11 files |
| RSS delta across the load | +29,794,304 B (28.4 MiB) |
| embed 382 rows | 694.5 ms (1.818 ms/row), dim 384, `mean_l2_norm` 1.0 |
| latency, 5 passes | **p50 6.330 ms**, p95 8.014 ms, per-pass p50s 6.172-6.464, spread 0.292 |
| conditions | loadavg 8.29 at start, 8.70 at end |

**Phase 4 sweep** (`sweeps/sentence-transformers__all-MiniLM-L6-v2.json`):

| fold | rows | train rows | correct | accuracy | fit s |
|---|---:|---:|---:|---:|---:|
| 0 | 77 | 194 | 71 | 0.9221 | 57.76 |
| 1 | 77 | 199 | 69 | 0.8961 | 65.44 |
| 2 | 76 | 195 | 76 | 1.0000 | 60.91 |
| 3 | 76 | 195 | 69 | 0.9079 | 63.27 |
| 4 | 76 | 194 | 74 | 0.9737 | 60.12 |

| metric | this sweep | ladder rung 3 |
|---|---:|---:|
| grouped accuracy | **0.9398** (359/382) | 0.9398 |
| fold spread (sd) | 0.0448 | 0.0448 |
| rowwise accuracy | **0.9424** (360/382) | 0.9424 |
| paired delta vs MiniLM (this is MiniLM) | +0.00 ± 0.00, 5/5 tied | — |
| `__none__` recall | 0.84 | 0.84 |
| false-none rate | 0.0056 | 0.0056 |
| ECE | 0.53 | 0.53 |
| cluster bootstrap CI95 | [0.8909, 0.9823] | [0.8909, 0.9823] |
| selection half (folds 0-2) | 0.9391 pooled, 0.9394 mean-of-folds (230 rows) | 0.9394 mean-of-folds |
| validation half (folds 3-4) | 0.9408 pooled, 0.9408 mean-of-folds (152 rows) | 0.9408 mean-of-folds |
| latency, dedicated pass | p50 7.662 ms, p95 10.821 ms (5 passes, spread 1.497) | p50 7.944 ms |
| determinism | fold 0 refit from scratch: identical predictions, 0/77 rows differ | — |

Read the two accuracy columns as a provenance check: the survey's recipe is the
ladder's recipe, and on MiniLM it reproduces rung 3 **to the row** — every fold,
both protocols, `__none__` recall, ECE and the bootstrap interval. That is the
strongest available evidence that a different encoder's row is comparable, and
it is the first thing to re-check if the recipe is ever touched.

The latency row is the box, not the code. This run was taken at loadavg 0.7, and
its dedicated pass (7.662 ms) sits 0.28 ms from the ladder's 7.944 ms; the same
artefact produced by the same recipe on the same folds at loadavg 10 read
9.711 ms for that pass and 446.057 ms p95 in-fold. The probe shows the same
effect on its own: 8.623 ms at loadavg 9.4, 6.330 ms once the machine quietened.
The recorded conditions are what make that readable; a sweep on `probe-box`
should be quoted with its own.

### Determinism, checked twice

- **in-process**: `determinism.identical_predictions: true`, `rows_differing: 0`,
  `max_confidence_abs_diff: 0.0` for a fold-0 refit inside the sweep.
- **cross-process**: the fold-0-only sweep was run twice as two separate
  interpreters (`determinism/minilm-fold0-runA.json`, `...runB.json`):

  ```
  A status ok acc 0.9221 per_fold [0.9221] n_rows 77 folds_run [0]
  B status ok acc 0.9221 per_fold [0.9221] n_rows 77 folds_run [0]
  identical predictions: True   differing rows: 0
  mean_confidence A/B: 0.3843 / 0.3843
  ```

  Same number, same 77 labels, same mean confidence, from two processes. Seeds
  are in every artefact under `seeds` (`setfit`/`torch`/`numpy`/`sklearn` = 0);
  the fold assignment has no seed because it is read, not generated.

## Self-tests — three of them, none needing a model or a network

```bash
.venv-embed/bin/python experiments/survey/test_report.py         # 17 cases, 88 checks
.venv-embed/bin/python experiments/survey/test_sweep_partial.py  # 60 checks
.venv-embed/bin/python experiments/survey/test_dtype_offline.py  # 13 cases, 92 checks
```

| file | covers | imports |
|---|---|---|
| `test_report.py` | `report_phase3.py` / `report_phase4.py` against synthetic artefacts: the win form, the ranking that does not transfer, a delta inside its own sd, the missing-fold refusal, a failed probe, the latency ceiling, duplicate probes | stdlib only |
| `test_sweep_partial.py` | a stage that raises writes `partial`, names the stage and keeps the fold numbers; injected `OSError(EINVAL)` in four stages; the atomic artefact write; the trainer-output root lifecycle | numpy, no torch |
| `test_dtype_offline.py` | the float32 cast and its assertion (bf16, a refused cast, a no-op `float()`, float32 params with bf16 embeddings); the `dtype` stage; the fold categories; both load paths including the sticky switch and `--offline` | torch |

All three are meant to run on the box that is being measured, not only here —
they are the cheapest way to find out that its interpreter is missing something
or that a Hub client behaves differently, before a fourteen-minute sweep says so
less precisely. Exit code 0 only when every check passes.

## What is verified, and what is not

Verified on this box (linux, `.venv-embed`, the real 382 rows):

- `probe_model.py` end to end on `sentence-transformers/all-MiniLM-L6-v2`,
  including the licence read, the snapshot size, the RSS delta and 5 latency
  passes.
- `probe_model.py`'s failure path on a non-existent id: `status: "error"`,
  `failed_stage: "load"`, the Hub's error text in the row, exit 1, no stack
  trace.
- `setfit_sweep.py` end to end, all 5 folds plus the rowwise fit, the latency
  pass and the determinism refit, reproducing rung 3 to the row.
- The fold-assignment guard: an artefact with one family moved (or one family
  removed) is refused with exit 2, a `refused: true` error row and no training.
- `run_survey.py`'s skip path and its `summary.json` on an existing `ok`
  artefact.
- The stage bookkeeping, against injected `OSError(EINVAL)` in four stages
  (`test_sweep_partial.py`, 60 checks): the artefact comes out `partial`, names
  the stage, carries the traceback, keeps the fold numbers that were measured,
  and turns into `error` only when every fold failed. The corpus, the fold
  assignment and the metrics in those cases are the real ones; only the encoder
  is a stub.
- The dtype contract (`test_dtype_offline.py`, 13 cases / 92 checks, PASS per
  case): a bf16 body cast and asserted; a cast that refuses; a `float()` that
  returns without casting anything; parameters float32 with bf16 embeddings; the
  `dtype` stage turning a whole sweep into `error` with `failed_stage: "dtype"`
  and the dtype string in the message; and the fold categories `dtype`, `fit` and
  `load` on a fold that raised inside its fit.
- The load paths, same file: online; online-fails-then-cache, with
  `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE` set in-process, the
  `huggingface_hub.constants.HF_HUB_OFFLINE` constant patched, and the switch
  sticky for the rest of the process; `--offline` making no online call at all;
  both attempts failing into a `LoadFailure` that names both; and a non-Hub error
  not being retried.
- The full 5-fold sweep after these changes
  (`results/survey-2026-11-23/verify/minilm-5fold-rerun.json`): grouped **0.9398
  (359/382)**, rowwise **0.9424 (360/382)**, per-fold identical to rung 3,
  `determinism.identical_predictions: true`, `load.mode: "online"` and
  `load.dtype.coerced: false` — the cast is a no-op on a float32 checkpoint,
  which is the property that keeps MiniLM's row exactly where it was.

Not verified here, and why:

- **`probe-box` itself.** Nothing in this directory has been run on the Windows
  box *by this checkout*. The three completed Windows sweeps in
  `results/survey-2026-11-23/sweeps/` were written by the previous revision of
  this harness; nothing added here — the `dtype` stage, the offline fallback,
  `--offline`, `failed_category` — has executed on Windows yet. The code avoids
  POSIX-only calls (`os.getloadavg` is guarded, the memory reader falls back to
  `psapi`, paths come from `pathlib`, the artefact is written through a
  same-directory temp file and an `os.replace`), but the Windows memory fallback
  and the Windows command lines above are reasoned, not executed. Say so before
  quoting a `probe-box` number as if this harness had produced it.
- **The bf16 checkpoint itself.** No bf16 encoder has been through the real code
  path. The cast is proven against stub bodies with real torch dtypes, and
  against MiniLM, which is float32 and therefore untouched. `granite-embedding-30m-english`
  is the one model that needs the cast, and only a run on that box can say the
  fix holds there.
- **A real dead Hub client.** Both load paths are exercised with stubs, and the
  sticky switch, the two environment variables and the patched constant are
  asserted, but no run here has watched a real closed `httpx` pool fall back to
  the cache.
- **The Windows `OSError: [Errno 22]` itself.** What is proven and what is only
  inference is set out in the section below; the short version is that the
  raising call was never identified, because the traceback was discarded and the
  failure correlates with a dropped ssh session rather than with anything in the
  measurement path.
- **The `ru_maxrss` memory fallback** is only reached if psutil is absent *and*
  `/proc` is absent; `SURVEY_DISABLE_PSUTIL=1` exercises the `/proc` branch
  (which was checked) but not the peak-vs-current caveat on Windows.
- **`run_survey.py --parallel 2`** was not exercised here. It starts two
  independent sweep processes, which is the same call as `--parallel 1` twice;
  the untested part is only the scheduling.
- **A real sweep of several base models** with the current revision. The three in
  `sweeps/` predate it. The next command is the runner with the phase-2 shortlist
  in `--models-file`.

What still needs a Windows run, in the order it should be done:

1. `setfit_sweep.py --model ibm-granite/granite-embedding-30m-english --offline`
   — the bf16 case, end to end. Expect `load.mode: "offline-fallback"`,
   `load.dtype.before: ["bfloat16"]`, `after: ["float32"]`, `coerced: true`, and
   five folds instead of `failed_stage: "harness"`.
2. A short `--folds 0` sweep of any model with the network blocked — expect the
   online attempt to fail, the cached retry to succeed, and
   `load.mode: "offline-fallback"` with the online error in `load.offline_reason`.
3. The three-model runner, detached (Task Scheduler / `start "" /b`), to confirm
   `failed_category` and the `dtype` stage cost nothing on the models that already
   worked.

## The Windows `Errno 22` — what is proven, and what is only inference

The first full sweep on the Windows box wrote `"failed_stage": "harness"`,
`"error": "OSError: [Errno 22] Invalid argument"`, and threw the traceback away.
Three full sweeps on that same box have since completed —
`sentence-transformers/all-MiniLM-L6-v2` (grouped 0.9398, rowwise 0.9424),
`sentence-transformers/paraphrase-MiniLM-L3-v2` (0.8796 / 0.8953) and
`snowflake/snowflake-arctic-embed-xs` (0.9005 / 0.9031) — each with all five
folds, the rowwise fit, the determinism refit (`identical_predictions: true`) and
the latency pass, written to `results/survey-2026-11-23/sweeps/` with
`platform: Windows-10-10.0.26200-SP0`. They were launched under Task Scheduler
rather than from a live ssh session; that is the operator's account of how they
were run, and it is the only difference between them and the run that died.

That splits the explanation in two, and the two are not equally strong.

### Proven

- **A full sweep completes on that box.** Three encoders, five folds each, all
  seven fits, the latency pass and the determinism refit, on the same Windows 10
  install and the same CPU-only torch as the run that failed. The measurement
  path — folds, pool, recipe, metrics, artefact write — is not what broke.
- **The error carried no path.** `describe_error` renders `Type: message`, and
  every Python-level path call on Windows appends the operand: a failed `open`,
  `mkdir`, `rmdir`, `rename` or `listdir` prints `[Errno 22] Invalid argument:
  'C:\...'`. The recorded string has none, so the raiser was one of the calls
  that fails with an errno and no filename — `mmap`/`CreateFileMapping`,
  `fsync`/`_commit`, `msvcrt.locking`, a socket write, `WriteFile` on a closed
  handle, or a C extension that dropped the path. That excludes the whole
  `open`/`mkdir`/`rmtree`/`replace` family, and it is the strongest fact there is.
- **Every `Errno 22` on record came from a run whose ssh connection dropped
  mid-run**, and no run that stayed connected produced one. That is an
  observation over the failing runs, not an experiment: nothing was reproduced
  deliberately.

### Inference, not proven

- **A torn-down stdio handle.** The sweep writes progress to stdout and the SetFit
  trainer writes its own progress bars; if the ssh session dies the handles point
  at a closed pipe, and a write to one of those is exactly the kind of call that
  raises `EINVAL` with no filename. It fits every observation — the failures only
  happened when the connection died, and the error carried no path — and it has
  never been isolated, because provoking it means killing a channel at a chosen
  moment.
- **A Hub client closed by the interruption.** `RuntimeError: Cannot send a
  request, as the client has been closed.` is the recorded form of that same
  interruption from the other side (it is the failure the offline fallback above
  exists for), and a closed pool raising from a socket call would also appear
  without a filename. A sweep loads the encoder seven times — once per fit plus
  the probe — so an interruption mid-run has seven chances to land there.
- **The fit-output-directory cycle** was fixed, not convicted. It was the only
  filesystem cycle the full sweep performed that the smoke run did not, and it is
  no longer reachable in the shape it had: one root per process, a fresh `fit-NN`
  per fit, one removal at exit. Nothing proves that cycle raised the error.

### What would settle it

- **The next failing run names the call.** `failed_stage` now separates `load`,
  `dtype`, `fit`, `rowwise`, `determinism` and `latency`, and both the reported
  failure and every stage record carry their own `traceback`.
- **`audit_fit_io.py --fits 2`** on that box shows the per-fit OS calls from the
  table below, if the raiser turns out to be in that cycle.
- **Launch it detached.** On Windows, run the sweep from Task Scheduler (or
  `start "" /b`, or `pythonw.exe`) rather than in the foreground of an ssh
  session: the run takes ten to fifteen minutes, and a dropped channel is the
  only environment in which this error has ever appeared. That is how the three
  completed sweeps were run.

### What the evidence excludes

Two facts narrowed it before any code was touched, and they are worth keeping:

**The error carries no path.** `describe_error` renders `Type: message`, and every
Python-level path call on Windows attaches the operand: a failed `open`, `mkdir`,
`rmdir`, `rename` or `listdir` prints `[Errno 22] Invalid argument: 'C:\...'`.
The recorded string has no path, so the raiser was one of the calls that fail with
an errno and no filename — `mmap`/`CreateFileMapping`, `fsync`/`_commit`,
`msvcrt.locking` (which `filelock` uses, and `EINVAL` is a documented errno for
it), a socket call, or a C extension that dropped the path. That single
observation excludes the whole `open`/`mkdir`/`rmtree`/`replace` family.

**The only full-sweep-only filesystem cycle is the trainer's own output tree.**
`experiments/survey/audit_fit_io.py` records every audited OS call per fit and
diffs fit 1 against fit 2. Measured on this box: fit 1 opens 6,714 paths (the
model cache, its download and the datasets map), and every *later* fit is a
compact cycle in a fresh `surveysweep-*` directory — 16 `os.mkdir`, 66 `open`,
16 `os.listdir`, 15 `os.remove`, 4 `os.rmdir` — writing 260 MB (weights,
`optimizer.pt`, `scheduler.pt`, `rng_state.pth`), listing it, and deleting the
tree. That happens once in the smoke run and seven times in a full sweep;
everything else the full sweep adds over the smoke run is arithmetic (the
latency loop, the metrics, the refits).

Ranked, with the reason each is or is not reachable in the full-sweep path:

| # | operation | reachable? | why |
|---|---|---|---|
| 1 | create / write 260 MB / list / delete the trainer output tree, per fit | **yes** | 7 cycles instead of 1; the only filesystem cycle the full sweep adds. Fixed: one root per process, the previous fit's subdirectory cleared, one removal at exit |
| 2 | `mmap` of a file an earlier fit wrote (weights, arrow cache) | **yes, in principle** | C-level, no filename in the error. Not reproduced here; the fix keeps a fresh `fit-NN` per fit so no fit maps a path another fit wrote |
| 3 | `msvcrt.locking` via `filelock` on an HF cache lock | **yes, in principle** | `EINVAL` is a documented `msvcrt.locking` errno and it raises without a filename; the audit trace shows no `.lock` opens in the fit-2 delta, so it is unexercised, not excluded |
| 4 | `os.fsync` on a handle that is not a disk file | no | the harness never fsyncs; `huggingface_hub` only fsyncs while downloading |
| 5 | `os.replace` across volumes | no | the artefact writer stages in the target's own directory; nothing else renames |
| 6 | `os.fsync`/`os.replace` on a directory | no | no such call exists here, and the new writer adds none |
| 7 | `numpy.memmap` on a path being rewritten | no | nothing in this harness or the ladder imports `numpy.memmap` |
| 8 | `tempfile` delete-on-close, then reopen | no | nothing uses `NamedTemporaryFile`; `mkstemp` is used once, in the artefact writer |
| 9 | subprocess / multiprocessing / signal | no | `run_survey.py` runs `subprocess.run` once per model in the *parent*; torch's DataLoader is at `num_workers=0` |
| 10 | console encoding on a redirect | **yes, as a different error** | the three progress lines the smoke run skips carried `…`; on a redirected stdout under a `cp437` locale that is `UnicodeEncodeError`. Fixed: ASCII progress, `errors="replace"`, and `PYTHONIOENCODING=utf-8:replace` for the child |
| 11 | `%z` in `time.strftime` | not the cause | `measured_at` in the failing artefact reads `...-0500`, so `%z` works on that box |
| 12 | a generated path landing on `con`/`nul`/`aux`, or carrying an invalid character | no | `mkdtemp` is prefixed `surveysweep-`, and `model_slug` strips `/` and `\` |

To identify it on the box rather than by inference: `failed_stage` and
`traceback` now name the call, and `audit_fit_io.py --fits 2` shows the
operations it was one of. Until a run supplies that, the honest statement is: a
network interruption during an ssh-launched sweep is proven to correlate; a
torn-down stdio handle or a Hub client closed by that interruption is the
inference; the fit-output tree is fixed but was never shown to be the cause.
