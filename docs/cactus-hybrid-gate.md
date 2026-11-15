# The Cactus hybrid checkpoint — below the gate on the figure that decides it

2026-11-10, branch `collect/2026-11-10`. One Cactus hybrid build scored on the repo's own
corpus, judged by the gate that was set for it: **AUROC of the checkpoint's confidence against
its own correctness**. Not its benchmarks, not its accuracy.

**The number that leads is the abstain-inclusive one.** A confidence that separates right from
wrong only after the `__none__` rows are removed is a confidence that fails on the rows a voice
assistant most needs it — the ones where the question is whether to act at all. Both figures
must clear 0.8; the 17-label figure is reported beside the abstain-inclusive one, never instead
of it. That rule lives in the measurement code (`verdict()` in
`experiments/optional/exp_cactus_hybrid.py`), not only in this document.

**Verdict: not usable, and nothing was wired.** Abstain-inclusive AUROC **0.794** (95% CI
0.701–0.875) over the 128 rows measured — below the gate. The 17-label figure clears at 0.837,
which is exactly the half-result the headline rule exists to stop anyone banking. It is *not* a
refutation: the interval spans the gate, and the full-corpus run that would have settled it was
started and died at row 30.

| | abstain-inclusive (headline) | 17 tool labels | rows |
|---|---:|---:|---:|
| **Cactus hybrid — the result that stands** | **0.794** (0.701–0.875) | 0.837 (0.731–0.932) | 128 |
| Cactus hybrid — full-corpus attempt | not evaluable | not evaluable | **30 of 382, killed** |
| base needle3 (same corpus, same rule) | **0.523** | 0.530 | 382 |

## 1. What was tested

| | |
|---|---|
| checkpoint | `Cactus-Compute/gemma-4-e2b-it-hybrid` — Gemma-4-E2B plus Cactus's "handoff probe", a head (weight prefix `handoff_probe.*`) that scores each generation with `confidence = 1 − p_wrong` |
| size on disk | 9.33 GB (three safetensors shards) |
| confidence surface | `model.generate(..., return_confidence=True).confidence` — structured, never parsed out of the answer text |
| prompt | one instruction listing the 18 registry ids and asking for exactly one id back, greedy decoding, `max_new_tokens=8` |
| rows measured | **128 of 382**, a systematic 1-in-3 sample, 18/18 labels represented |
| rows attempted | 382 (`--stride 1`); the run died at row 30 — §2.1 |
| prediction rule | the first registry id appearing as a token; otherwise `NO_CALL`; a `__none__` row is correct on `__none__` or `NO_CALL` — the rule the needle3 run used |
| environment | `.venv-hybrid` (transformers 5.5.4, torch 2.15.0+cpu, CPU only); pins in `docs/ladder-env.md` |
| cost | $0: one Hugging Face download, no API, nothing uploaded |

## 2. The gate

Base needle3's number is recomputed from the historical per-row confidences in
`results/historical/needle_results.json`, so both models are measured by the same arithmetic
rather than one number being carried over from a different analysis.

What the 128 rows look like underneath: 108 right, 20 wrong; mean confidence 0.596 on the rows it
got right against 0.409 on the rows it got wrong. The overlap is real — the lowest-confidence
*correct* row sits at 0.17, below two of the errors at 0.55 — which is why neither figure is
anywhere near 0.95.

### 2.1 The full-corpus run: started, and killed at 30 rows

Started with the same protocol and `--stride 1`, and it died at row 30: no python process left,
the row counter frozen, only the shell wrappers remaining. The machine at the time:

* **11.9 GB of swap in use** of 50 GB, against 15.7 GB of RAM and 5.9 GB available;
* the checkpoint is **9.33 GB**, and it has to be resident for the probe to score generations;
* `journalctl` records `OOM killer enabled`, and the process left no traceback — an
  out-of-memory kill, not a probe failure;
* the 30 saved rows are all correct (0 errors), so they add accuracy but **no AUROC**: with no
  errors there is nothing to separate.

This is an environment limit, not a result. The box cannot hold this checkpoint alongside the
other work running on it, and the attempt was not repeated: an idle machine with ≥16 GB free
would settle it in 2.8–4 h at the measured 26–38 s per row (`--stride 1`). Until then the
128-row sample is the only measurement, and its headline figure is below the gate.

### 2.2 Why the sample is not being rescued

The brief's rule for this gate was: if the full run holds above 0.8 on both figures, wire it;
if it drops below, say so and stop. The full run did not complete, so the rule is applied to the
evidence that exists: the headline figure is below 0.8, therefore **not usable, not wired**.
No prompt tuning, no re-scoring the sample without the abstain rows, no second sample, no
third-party benchmark quoted instead. If a later run on an idle box comes back above 0.8 on both
figures, the work is already in place: `localalexa tier2` joins the cascade to any per-row
measurement file and reports the coverage and precision curve (§6).

## 3. Where the errors are

20 wrong out of 128, almost entirely the confusions this repo already documents:

| confusion | count |
|---|---:|
| `thermostat_offset` → `thermostat_set` | 6 |
| `__none__` → a tool (`research_deep` ×5, `docs_query` ×2) | 7 |
| `shoppinglist_remove` → `shoppinglist_add` | 3 |
| `research_deep` → `docs_query` | 2 |
| `shoppinglist_view` → `shoppinglist_add` | 1 |
| `projects_query` → `NO_CALL` | 1 |

The abstain class is where the checkpoint is weakest: 7 of the 9 `__none__` rows in the sample
were answered with a tool. That single fact is what pulls the headline figure under the gate
while the 17-label figure clears it, and it is the same failure the linear tier has (0.60 recall
on that class). A confidence that cannot tell "no command" apart from a command is not a
handoff gate for this product.

## 4. What this means

* **Not wired as tier 2.** The plumbing exists and is tested (`src/localalexa/tier2.py`,
  `localalexa tier2 --measured …`), and it is pointed at nothing.
* **The model is accurate but not adjudicating.** 84.4% on the sample, against the ladder's
  95.03% grouped for a 7.9 ms logreg+SetFit ensemble — different protocol, one-shot prompt, so
  indicative rather than head-to-head. Its labels are not why anyone would adopt it.
* **Cost rules it out of the cheap tier anyway.** 26.4 s per utterance and 9.33 GB resident, on
  a machine that could not finish the run. Even with a passing gate this is a remote-tier
  candidate, not a tier-1 one.
* **The thing worth keeping is the method.** AUROC of a model's own confidence against its own
  correctness, with the abstain class in the denominator, is a cheap and decisive test — and it
  cost two hours and no money. It is the test to run on the next checkpoint before anything else
  about it is believed.

## 5. Cost, measured

| | |
|---|---|
| model load | 3–6 s (bf16, mmap) |
| per utterance | **26.4 s p50** in the sample; 32–39 s per row in the full attempt |
| resident | ~4.6 GB RSS during generation, 9.33 GB on disk |
| 128-row sample, end to end | ~57 min of generation |
| full-corpus attempt | 30 rows in ~3.5 h of job time, then killed |

## 6. Reproduce

```bash
# one-time env (the checkpoint pins transformers)
uv venv .venv-hybrid --python 3.12
uv pip install --python .venv-hybrid/bin/python "transformers>=5.5.4,<5.6" "tokenizers>=0.22.0,<=0.23.0"
uv pip install --python .venv-hybrid/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv-hybrid/bin/python httpx pillow numpy safetensors huggingface-hub tqdm filelock regex packaging pyyaml requests

# the measurement (128-row sample: ~1 h; full corpus: 2.8-4 h on an idle box with >=16 GB free)
.venv-hybrid/bin/python experiments/optional/exp_cactus_hybrid.py --stride 3
.venv-hybrid/bin/python experiments/optional/exp_cactus_hybrid.py --stride 1

# every metric and the verdict re-derived from the saved rows, no model loaded
.venv-hybrid/bin/python experiments/optional/exp_cactus_hybrid.py --recompute results/cactus_hybrid_results.json

# the tier-2 join, ready for a measurement that passes the gate (not run against this model)
.venv/bin/python -m localalexa tier2 --measured results/cactus_hybrid_results.json
```

Artifacts: `results/cactus_hybrid_results.json` (the 128-row sample, with per-row text, truth,
prediction, raw answer, confidence and latency) and `results/cactus_hybrid_full.json` (the 30
rows the killed run saved, kept as the record of the attempt).

## 7. Honest limits

* 128 of 382 rows, systematic sample, one prompt with no few-shot examples. A different prompt
  moves both accuracy and confidence; this measures the checkpoint as prompted, not its ceiling.
* Greedy decoding only. The card states the probe runs only for single-sequence decoding, so a
  batched deployment would have no confidence at all.
* The probe is trained on Cactus's own handoff data, not on this corpus. That it transfers at
  all is the finding; that it would transfer better after a few hundred labelled rows is a guess,
  untested here.
* Base needle3's 0.523 came from a different confidence surface (the tool-calling engine's own
  score) on the full 382 rows — the same arithmetic, a different model API.
* The full-corpus attempt is one data point about this box's memory, not a claim that the model
  cannot be measured; it can, on a machine that can hold it.
