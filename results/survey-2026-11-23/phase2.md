# Phase 2 — narrowing the landscape to five

2026-11-23, branch `survey/base-model-2026-11-23`. This stage **eliminates and shortlists**; it does
not pick a winner. Phase 4 decides.

Constraints are taken verbatim from `docs/base-model-survey-brief-2026-11-23.md` §"Constraints,
stated once so every phase uses them". The incumbent and the numbers to beat are from
`docs/ladder-results-2026-11-10.md`. Every row argued from below is cited as `A`/`B`/`C`/`D`/`E`
(family table) with the model name, since the tables are keyed by name rather than row number.
Where a number does not exist in the phase-1 tables or in those two documents it is reported as
missing and the decision is made without it.

**The universe, counted.** The five phase-1 tables contain **85 rows** describing **74 distinct
model artefacts**. Eleven rows re-name something already counted: five are the same repo appearing
in two tables (`nomic-ai/modernbert-embed-base` A+B, `Snowflake/snowflake-arctic-embed-xs` A+D,
`mixedbread-ai/mxbai-embed-xsmall-v1` A+D, `jinaai/jina-embeddings-v2-small-en` B+D,
`google/embeddinggemma-300m` A+D), two are ONNX re-exports of a model counted elsewhere
(`Xenova/all-MiniLM-L6-v2`, `Xenova/bge-small-en-v1.5`), and four family-E rows are recipes whose
*body* is counted elsewhere (SetFit on `paraphrase-mpnet-base-v2`, SetFit on
`all-MiniLM-L6-v2`, frozen MiniLM-L6 + kNN, FastFit-large's `roberta-large` body). All 85 rows are
accounted for in §2 or §3.

---

## 1. The filter, applied in order

Filters are applied in the order the brief states them. A model that fails two constraints is
attributed to the **first** one that kills it, so the counts below sum to 74 exactly and no model is
double-counted. The funnel:

| # | filter | rule, in one line | in | removed | out |
|---|---|---|---:|---:|---:|
| F1 | latency ceiling | params above the ceiling's arithmetic equivalent are out unmeasured | 74 | 50 | 24 |
| F2 | permissive licence | must be Apache-2.0 or MIT **and** declared at the artefact | 24 | 4 | 20 |
| F3 | CPU-only | must run under `torch 2.11.0+cpu`, no CUDA, no iGPU | 20 | **0** | 20 |
| F4 | parameter / embedding budget at 22 examples per intent | must fit a trainable capacity/dim budget | 20 | **0** | 20 |
| F5 | availability as a loadable sentence encoder | must load with tokenizer + pooling module, no hand-built head | 20 | 7 | 13 |
| — | shortlisting (a selection cut, not a constraint) | five rows covering distinct hypotheses | 13 | 8 | **5** |

### F1 — latency ceiling: removed 50 of 74

**Rule.** The brief sets ~10 ms per utterance on CPU, batch 1. One model has been measured on this
protocol: the incumbent at **6.33 ms p50 / 8.01 ms p95** (22.7M params, 8-core dev box, 5 passes,
warm, idle). Phase 3's probe artefact
(`results/survey-2026-11-23/probe-sentence-transformers__all-MiniLM-L6-v2.json`) reproduces it at
**6.33 ms p50 / 8.014 ms p95** over the real 382 rows, so the anchor is a number I can use twice.

Converting a latency ceiling into a parameter ceiling requires a scaling law, and I use the most
favourable one available for the candidate — **linear in parameters** — anchored on that 6.33 ms:

> N_max = 22.7M × (10 ms ÷ 6.33 ms) = **35.9M parameters**

A model above 35.9M is eliminated **without measurement**, because even the optimistic law exceeds
the ceiling. This is a floor, not a prediction: the box supplies a second measured point, and it says
latency grows *faster* than params. In phase 3's probe, `paraphrase-MiniLM-L3-v2` (17.4M) ran at
8.152 ms and `all-MiniLM-L6-v2` (22.7M) at 12.478 ms under identical conditions — a param ratio of
0.767 for a latency ratio of 0.653, which fits an exponent of ≈1.6, not 1.0. Using 1.6 instead of 1.0
moves the ceiling to **≈25M parameters**. I deliberately did **not** apply the 1.6 form as a filter,
because both of those probe numbers are batch-1 smoke tests that the phase-3 document itself says must
not be read against the ladder's protocol. So the 35.9M ceiling is the *permissive* cut, and the
steeper exponent is carried forward as the basis for §4's predictions instead. A model in the
25–35.9M band is not eliminated here; it is routed to the probe with a pre-registered drop threshold.

The 50 models this removed, with the number that fails each, are in §3.1 grouped by family. The
short version: every 768-dim and 1024-dim encoder, every decoder LLM, and every model whose body is
larger than 36M. That is the whole of family C (10 of 10), 31 of 36 rows in family B, 5 of 14 in
family A, and 3 of the 8 family-E rows.

### F2 — permissive licence: removed 4 of 24

**Rule.** The brief requires a permissive licence because the repo is public and credits its
dependencies. I applied it strictly at the artefact: **Apache-2.0 or MIT, declared in the
repository** (frontmatter field, `LICENSE` file, or card body). A model with *no declared licence* is
out — not because it is certainly non-permissive, but because a public repo cannot credit what
carries no licence, and "no licence" is legally the most restrictive state a model can be in.

Removed: `sentence-transformers/multi-qa-MiniLM-L6-cos-v1`, `sentence-transformers/msmarco-MiniLM-L6-cos-v5`,
`sentence-transformers/msmarco-MiniLM-L12-cos-v5` (all three: "none declared — no LICENSE file in the
repo and no licence tag", family A) and `TaylorAI/gte-tiny` ("not stated — no licence tag and no
LICENSE file", family D). All four were otherwise inside the parameter ceiling at 22.7M/22.7M/33.4M/
22.7M, so this filter is doing real work rather than rubber-stamping F1.

**One borderline pass, recorded because it is a metadata gap rather than a legal one.**
`Snowflake/snowflake-arctic-embed-xs` is recorded in family A as "not declared in the repo (no LICENSE
file, no licence tag); the arctic-embed paper states the five arctic-embed models are released under
Apache-2", and in family D as "Apache-2.0 (card body: 'Arctic is licensed under the Apache-2'; no
licence field in frontmatter)". The two rows disagree about *where* the statement lives, not about
what it says. I pass it, and phase 3 must resolve it by reading the `LICENSE` file in the resolved
snapshot — see §4 and §5.

Nothing else in the surviving 24 relies on a non-permissive licence. Gemma Terms, CC-BY-NC-4.0 and
the Qwen Research License all appeared only on models that F1 had already removed
(`google/embeddinggemma-300m`, `BAAI/bge-multilingual-gemma2`, `google/gemma-3-270m`,
`jinaai/jina-embeddings-v3/v4/v5-*`), which is why they are attributed to F1 in §3 rather than here.

### F3 — CPU-only: **removed 0 of 20**

**Rule.** Must run under the measured stack — `torch 2.11.0+cpu`, `transformers 5.6.2`,
`sentence-transformers 5.4.1`, Python 3.11.9, no CUDA, and the Arc 140V iGPU unused
(`phase3-env.md` §(c)). Anything requiring a CUDA-only kernel, a GPU-only quantisation format, or a
triton/long-context kernel was to be dropped.

**This constraint removed nothing, and that is worth saying plainly.** Every model that reached it is
a plain `sentence-transformers` encoder with a CPU path in the box's own stack; the reason none died
here is that F1's parameter ceiling had already excluded everything whose *only* documented speed is a
GPU figure. The constraint is not idle, though — it does three things that are not eliminations:

- it fixes the arithmetic of F1 (CPU latency, not GPU throughput, is the quantity that gets compared);
- it disqualifies every published speed claim in the phase-1 tables except two. Family B records that
  **35 of 36** models publish no latency, throughput or per-utterance figure of any kind; the
  surviving claims are relative to GPU or to an unnamed baseline and cannot be used as evidence;
- it makes `model_args.trust_remote_code: true` a *risk* rather than a filter, which is the honest
  classification for `jinaai/jina-embeddings-v2-small-en`'s ALiBi attention (§3.5).

### F4 — parameter / embedding budget at 22 examples per intent: **removed 0 of 20**

**Rule.** The corpus gives 22 examples per intent over 18 labels — 200 reference rows, 194 train rows
per fold (`docs/ladder-results-2026-11-10.md` §"Method notes"). The budget I applied: the model must
be small enough that ~200 rows can still move it, and the embedding must be narrow enough that the
logistic head is not the binding constraint — I set dim ≤ 512 and ≤ 4× the incumbent's parameters.

**This removed nothing either, and the reason is arithmetic, not luck.** F1's 35.9M ceiling is
already *tighter* than any capacity ceiling I could defend for 200 rows, so the parameter half of this
budget does no independent work. The dim half (≤ 512) bites only at 768 and 1024, and every one of
those models is above 35.9M and therefore already gone. The dim ceiling is not a coincidence — it is
the same fact seen from the embedding side.

What F4 *does* establish, and what §2 relies on, is a property of the surviving set rather than a
count: **all five shortlisted models are 384-dimensional.** Every delta phase 4 measures will
therefore be capacity-or-objective, never a dimension change, because no model with a different
dimension survived both F1 and the shortlisting cut.

The brief's `max_seq` concern does not bite at all here. The ladder describes the corpus as **3–6 word
utterances** (rung 1 analysis: "word 1–2 grams over 3–6 word utterances nearly describe the whole
string"), so the shortest surviving limit — 128 tokens on `paraphrase-MiniLM-L3-v2` — truncates
nothing, and `max_seq` is recorded in §2 as a property, not applied as a filter.

### F5 — availability as a loadable sentence encoder, or trivially wrappable: removed 7 of 20

**Rule.** It must load under the pinned stack as a **trainable** encoder with a tokenizer and a
pooling module, so that phase 4's recipe (SetFit contrastive + logistic head) applies without building
a model first. `trust_remote_code` is tolerated as a recorded risk; hand-built pooling heads and
weight-less artefacts are not.

Removed:

| model | family | why it is not a loadable trainable encoder |
|---|---|---|
| `minishlab/potion-base-8M`, `potion-base-32M`, `potion-retrieval-32M` | D | Static `model2vec` models — "no transformer encoder to fine-tune; distillation recipe only". There is nothing for the contrastive stage to train. |
| `sentence-transformers/static-retrieval-mrl-en-v1` | D | `StaticEmbedding` module only, "0 active params" — same reason, and additionally a 31.25M-entry lookup table with no weights to adapt. |
| `huawei-noah/TinyBERT_General_4L_312D` | D | Raw BERT checkpoint — "no sentence-transformers modules or pooling head; a head and contrastive training would have to be built". Its 15.7M is *derived*, not published. |
| `Xenova/all-MiniLM-L6-v2`, `Xenova/bge-small-en-v1.5` | D | ONNX weights only, no PyTorch/ST modules — "contrastive training must happen on the base repo, then re-export". These are deployment formats of the incumbent and of `bge-small-en-v1.5`, not candidates. |

The weakest kill in the funnel is TinyBERT: a `Transformer` + `Pooling` wrapper is mechanically
possible, so "trivially wrappable" is a judgement I am making against family D's own verdict. I
removed it because entering a sweep with an unreleased artefact — derived parameter count, no pooling
head, no published sentence-embedding evaluation — means phase 4 would be measuring my wrapper as much
as the model.

### The shortlisting cut: 13 → 5, and why it is not a sixth filter

The five constraints leave **13** models, all 384-or-512-dim and all under the parameter ceiling:
`all-MiniLM-L6-v2`, `all-MiniLM-L12-v2`, `paraphrase-MiniLM-L6-v2`, `arctic-embed-xs`,
`arctic-embed-s`, `mxbai-embed-xsmall-v1`, `bge-small-en-v1.5`, `e5-small-v2`, `gte-small`,
`bge-micro-v2`, `paraphrase-MiniLM-L3-v2`, `granite-embedding-30m-english`,
`jina-embeddings-v2-small-en`.

Cutting 13 to 5 is a *selection* decision, not a constraint: the brief asks for five rows, so the
eight that passed everything are cut on coverage grounds (each is disposed of individually in §3.5,
with the reason). The rule I used: **each slot must change what phase 4 can conclude.** Four distinct
axes exist at this size — capacity down (does a 3-layer encoder keep SetFit's gain?), capacity up
(does 33.4M pay for itself inside the ceiling?), same-size-different-objective (does better pretraining
data beat a bigger model?), and different-tokenizer (is the untested axis at 30M the tokeniser rather
than the weights?). One slot for each, plus the mandated incumbent row.

---

## 2. Survivors: exactly five

| # | model id | params | dim | max_seq | licence | why it survives | what could disqualify it |
|---|---|---|---:|---:|---|---|---|
| 1 | `sentence-transformers/all-MiniLM-L6-v2` (**incumbent, reference row**) | 22.7M (22,713,216 measured) | 384 | 256 | apache-2.0 | Mandated as the reference: it is the only candidate with a measured accuracy on this corpus (359/382 grouped = 0.9398, 360/382 rowwise, `__none__` 0.84) and a measured latency on this protocol (6.33 ms p50), so it is the origin every other row is scored from. | Nothing can disqualify it as the reference; the only failure mode is that its probe does not reproduce 6.33 ms, which indicts the harness rather than the model. |
| 2 | `sentence-transformers/paraphrase-MiniLM-L3-v2` | 17.4M | 384 | 128 | apache-2.0 | The cleanest capacity test available: same 384 dim, same tokenizer and lineage, same ST/SetFit recipe, three layers instead of six — so any delta is attributable to depth alone — and it is already downloaded and measured on the box at 8.152 ms single-sentence. | If SetFit's contrastive gain depends on encoder depth, a 3-layer body may buy nothing while scoring below the 0.38 ms linear rung, in which case the row is a cost with no benefit. |
| 3 | `Snowflake/snowflake-arctic-embed-xs` | 22.6M | 384 | 512 | Apache-2.0 (stated in card body and paper; absent from frontmatter) | The only candidate at or below the incumbent's parameter count with twice the usable `max_seq` (512 vs 256) **and** a documented infoNCE contrastive objective — the closest match to SetFit's own contrastive stage, so it is the swap least likely to break the recipe. | Its licence statement lives in the card body and the paper rather than a `LICENSE` file in the repo; if the resolved snapshot has none, it falls to the same rule that killed `gte-tiny`. |
| 4 | `BAAI/bge-small-en-v1.5` | 33.4M | 384 | 512 | MIT | The highest-upside row the ceiling still admits, and the best-supported CPU deployment path in the whole landscape: family D carries fp32/int8/q4f16 ONNX exports of exactly this model (133.1 / 33.8 / 36.2 MB) and a 17.4M distilled sibling, so a win here transfers to a Pi without re-derivation. | It is predicted at 11.7 ms (fitted exponent) against a 10 ms ceiling, with a 9.3 ms optimistic floor — the probe may drop it before phase 4 ever trains it. |
| 5 | `ibm-granite/granite-embedding-30m-english` | 30.3M | 384 | 512 | apache-2.0 | The only surviving row carrying a published speed claim to falsify ("twice as fast as other models with similar embedding dimensions" — VENDOR CLAIM, no baseline named, no ms) at a 6-layer/30.3M shape with a **RoBERTa tokenizer**, i.e. a different tokenisation of 3–6-word imperatives, the one untested axis left at this size. | Predicted at 10.0 ms (fitted) / 8.5 ms (floor), it sits exactly on the ceiling — the row most likely to die in phase 3 for a reason that has nothing to do with its accuracy. |

**Properties of the set, stated because they constrain what phase 4 can conclude.**

- All five are **384-dim**. Dim is held constant across every comparison, deliberately (F4).
- All five are **Apache-2.0 or MIT** with a declared licence, and all five load as plain
  `sentence-transformers` models with a pooling module. Only `arctic-embed-xs` needs its licence
  provenance pinned; none needs `trust_remote_code`.
- Parameter range **17.4M–33.4M**, i.e. 0.77× to 1.47× the incumbent. The shortlist deliberately does
  not span an order of magnitude, because at 200 training rows the evidence does not support spending
  a slot above 1.5× the incumbent — see the "no larger model" argument in §5.
- Latency range is predicted at **4.1–11.7 ms**, which brackets the ceiling rather than sitting under
  it. One row (§4's #5) is predicted to fail, and the probe is designed to catch it.
- **No sixth row is listed here.** The one model for which a case can genuinely be made is argued in
  §3.5 and still not shipped, per the brief.

---

## 3. Eliminations, with the reason for each

### 3.1 Died on F1 — the latency ceiling (50 models)

Every number in this section is `6.33 ms × (params ÷ 22.7M)` — **an estimate, labelled as such**, using
the most favourable linear scaling of the incumbent's measured 6.33 ms p50 on the 8-core dev box
(warm, idle, 5 passes, batch 1, 382 rows). Real latency is expected to be worse, because the box's own
L3/L6 probe pair fits an exponent of ≈1.6. Where a family's members all fail for the same reason I
give the family once, with each member's parameter count and its own floor.

**BGE above small — 5 models.** `bge-base-en-v1.5` 109.5M (≥30.5 ms), `bge-large-en-v1.5` 335.1M
(≥93.4 ms), `bge-m3` 568M derived from a 2271 MB fp32 bin (≥158 ms), `bge-en-icl` 7.11B (≥1.98 s),
`bge-multilingual-gemma2` 9.24B (≥2.58 s, and non-permissive Gemma terms as well). All are encoders
of an architecture family whose only member inside the budget is already on the shortlist.

**E5 — 6 models.** `e5-base-v2` 109.5M (≥30.5 ms), `e5-large-v2` 335.1M (≥93.4 ms),
`multilingual-e5-small` 117.7M (≥32.8 ms — note "small" here is 117.7M, five times the incumbent),
`multilingual-e5-base` 278.0M (≥77.5 ms), `multilingual-e5-large` 559.9M (≥156 ms),
`e5-mistral-7b-instruct` 7.11B (≥1.98 s). The multilingual line is not a candidate at all: the corpus
is English imperatives.

**GTE — 7 models.** `gte-base` 109.5M (≥30.5 ms), `gte-large` 335.1M (≥93.4 ms),
`gte-base-en-v1.5` 137M (≥38.2 ms), `gte-large-en-v1.5` 434M (≥121 ms),
`gte-multilingual-base` 305M (≥85.1 ms — its "10× faster" claim is vs a 1.5B decoder and is relative
only), `gte-modernbert-base` 149M (≥41.6 ms), `gte-Qwen2-1.5B-instruct` 1.78B (≥496 ms). The 8192-token
context these carry is worth nothing on a 6-word utterance.

**Jina — 6 models.** `jina-embeddings-v2-base-en` 137.4M (≥38.3 ms), `jina-embeddings-v3` 572.3M
(≥160 ms), `jina-embeddings-v4` 3.75B (≥1.05 s), `jina-embeddings-v5-text-nano` 239M (≥66.7 ms),
`jina-embeddings-v5-text-small` 677M (≥189 ms), `jina-embeddings-v5-omni-small` ~1.74B (≥485 ms).
Four of the six are additionally **CC-BY-NC-4.0** and one is the **Qwen Research License** — both
non-commercial, so they would have died at F2 anyway. The v5-nano row is the sharpest loss here: 239M
is the closest any non-permissive model came to the ceiling.

**Nomic — 4 models.** `nomic-embed-text-v1` 136.7M (≥38.2 ms), `nomic-embed-text-v1.5` 136.7M
(≥38.2 ms), `nomic-embed-text-v2-moe` 475M total / 305M active (≥133 ms, and needs `megablocks` +
`trust_remote_code` — a MoE that activates only 64% of its weights is still 13× the incumbent's
memory), `nomic-ai/modernbert-embed-base` 149M (≥41.6 ms, counted once despite two rows).

**Stella — 4 models.** `stella-base-en-v2` — parameters **not published**; derived at ≈50M from the
card's 0.2 GB at fp32 (≥13.9 ms), which is above the ceiling but rests on an unstated precision, so
this row is eliminated on the *stronger* ground that it ships no `modules.json` and its pooling is only
in a card snippet (F5's rule). `stella_en_400M_v5` 400M (≥111 ms), `stella_en_1.5B_v5` 1.54B (≥429 ms),
`stella-large-zh-v3-1792d` ≈325M derived from 1.3 GB (≥90.6 ms) — the last is also Chinese-only per
its card and 1792-dim.

**Sentence-transformers' own larger models — 4 models.** `all-mpnet-base-v2` 109.5M (≥30.5 ms),
`paraphrase-mpnet-base-v2` 109.5M (≥30.5 ms — this is SetFit's *default* body, and it is 4.8× the
incumbent), `all-distilroberta-v1` 82.1M (≥22.9 ms), `google/embeddinggemma-300m` 302.9M (≥84.5 ms,
also gated behind manual approval and non-permissive Gemma terms).

**Other encoders — 1 model.** `distilbert-base-uncased` 67.0M (≥18.7 ms) — and it is an MLM
checkpoint with no sentence-transformers modules, so it would have died at F5 as well.

**Decoder LLMs — all 10 of family C.** `Qwen3-0.6B` 751.6M (≥210 ms), `Qwen3-0.6B-Base` 596.0M
(≥166 ms), `Qwen3-1.7B` 2.03B (≥566 ms), `Qwen3-1.7B-Base` 1.72B (≥480 ms),
`Qwen3-Embedding-0.6B` 595.8M (≥166 ms), `SmolLM2-135M` 134.5M (≥37.5 ms), `SmolLM2-360M` 361.8M
(≥101 ms), `SmolLM2-1.7B` 1.71B (≥477 ms), `SmolLM3-3B` 3.08B (≥859 ms), `Gemma 3 270M`/`270M-IT`
268.1M (≥74.8 ms, also non-permissive and gated). **The smallest member of this entire family,
`SmolLM2-135M` at 134.5M, is 5.9× the incumbent and has a ≥37.5 ms floor — 3.8× the ceiling.** This
is the group the eliminations section exists for: no decoder LLM of any size, in any family phase 1
read, is within an order of magnitude of the budget, and none of them has a documented contrastive
recipe in the first place (family C records "none documented" for 8 of the 10). Note
`SmolLM3-3B` also fails on `max_seq` economics: its 65,536-token context is four orders of magnitude
more than a 6-word utterance.

**Non-encoder bodies reached through family E — 3 models.** `FacebookAI/roberta-large` 355.4M
(≥99.1 ms, reached twice: as FastFit-large's body and as PET's backbone), `bigscience/T0_3B` 2.85B
(≥795 ms, T-Few), `google/flan-t5-xxl` 11.27B (≥3.14 s — and its own author-claimed throughput is
"exceeding 1 second per input", i.e. 100× the ceiling *per input*).

### 3.2 Died on F2 — licence (4 models)

| model | family | number that fails it |
|---|---|---|
| `sentence-transformers/multi-qa-MiniLM-L6-cos-v1` | A | no licence declared — "no LICENSE file in the repo and no licence tag"; 22.7M/384, otherwise inside every budget |
| `sentence-transformers/msmarco-MiniLM-L6-cos-v5` | A | no licence declared — "no LICENSE file, no licence tag"; 22.7M/384 |
| `sentence-transformers/msmarco-MiniLM-L12-cos-v5` | A | no licence declared — "no LICENSE file, no licence tag"; 33.4M/384 |
| `TaylorAI/gte-tiny` | D | no licence declared — "no licence tag and no LICENSE file in the repo"; 22.7M/384, quantised ONNX shipped |

Three of the four are MiniLM-L6-shaped with 384 dims, i.e. exactly the profile that would otherwise
have competed for a shortlist slot. They are out on paper, not on merit, and no measurement is needed.

### 3.3 Died on F3 — CPU-only: none

No model reached F3 that could not run on CPU under the box's own stack. See §1 F3 for why that is a
statement about F1 having done the work, not about the constraint being weak.

### 3.4 Died on F4 — parameter / embedding budget: none

No model reached F4 outside the budget; F1's ceiling is stricter. The only models the *dim* half would
have caught are the 768- and 1024-dim rows, all of them above 35.9M and gone at F1. See §1 F4.

### 3.5 Died on F5 — not a loadable trainable encoder (7 models)

`minishlab/potion-base-8M` (7.6M), `potion-base-32M` (32.3M), `potion-retrieval-32M` (32.3M) — static
`model2vec` bag-of-tokens models, no encoder to contrastively fine-tune; their "up to 500× faster on
CPU" claim is the most attractive number in the entire landscape and it is unusable here, because the
only recipe this project has measured is a fine-tune of an encoder.
`sentence-transformers/static-retrieval-mrl-en-v1` — `StaticEmbedding` only, 0 active parameters.
`huawei-noah/TinyBERT_General_4L_312D` (15.7M derived) — raw checkpoint, no pooling head, no published
parameter count and no sentence-embedding evaluation; wrapping it would mean phase 4 measures the
wrapper. `Xenova/all-MiniLM-L6-v2`, `Xenova/bge-small-en-v1.5` — ONNX-only re-exports of the incumbent
and of shortlist row #4; they cannot be trained and are deployment formats, not candidates.

### 3.6 Recipes from family E, which are not weights

Family E's eight rows are recipes; four of them name a body counted above, so they appear here rather
than in the parameter tables.

- **SetFit v1.2.0 — survives, as the instrument, not as a candidate.** It is phase 4's recipe and it is
  already measured on this corpus (0.9398 grouped / 0.9424 rowwise on the incumbent body, 359/382 and
  360/382). Its *default* body `paraphrase-mpnet-base-v2` is eliminated at F1 (109.5M, ≥30.5 ms); the
  recipe will be run on the five shortlisted bodies instead.
- **FastFit (small 110M / large 355M)** — eliminated: the small config's body is 109.5M (≥30.5 ms) and
  the large config's is 355.4M (≥99.1 ms), the repo is **archived read-only since 2026-10-19**, and its
  128-d projection with token-level similarity means the ladder's metric block does not transfer
  unmodified. Its "about 1 millisecond" is an AUTHOR CLAIM in a paper, on unstated hardware.
- **Frozen ST embeddings + cosine kNN on MiniLM-L6** — eliminated as a candidate because nothing is
  trained, so it is not a base model at all; the ladder's nearest measured equivalent is rung 2
  (word TF-IDF + logistic regression, 90.84% grouped at 0.38 ms), which is already the tier-1 model.
- **PET on `roberta-large`** — eliminated: 355.4M (≥99.1 ms), and the paper's own training cost is
  ≈30 min per classifier for 250 steps on one GTX 1080 Ti, ≈2 h for the final classifier — a GPU price
  for a CPU-only box, with no inference latency published.
- **T-Few on `T0_3B`** — eliminated: 2.85B (≥795 ms) and ≈700 s to train on a ≥40 GB GPU.
- **flan-t5-xxl in-context prompting** — eliminated: 11.27B (≥3.14 s), "throughput exceeding 1 second
  per input" (AUTHOR CLAIM). This is the row the brief's "stop anyone re-proposing a 7B model" note is
  aimed at; it is 11B.

### 3.7 Survived all five constraints, cut at shortlisting (8 models)

These are **not** constraint failures. They are inside the latency ceiling, permissively licensed,
CPU-loadable, and 384-or-512-dim; they lost a coverage slot to a model making the same argument
better.

| model | params / dim / max_seq | why it is not one of the five |
|---|---|---|
| `sentence-transformers/all-MiniLM-L12-v2` | 33.4M / 384 / 128 (card says 256) | The incumbent's own family, one layer deeper, and **dominated inside its own cells**: it is the sixth 33.4M/384 model and it cannot separate depth from depth-plus-more-layers. Its row also contradicts itself on `max_seq` (`sentence_bert_config.json` 128 vs card 256), which is a metadata defect I would have to resolve before trusting any other cell. |
| `sentence-transformers/paraphrase-MiniLM-L6-v2` | 22.7M / 384 / 128 | Same size and architecture as the incumbent, shorter `max_seq` (128 vs 256), and its card states no training objective ("card states no training objective"). It is the incumbent with strictly less information; running it would spend a slot to confirm that. |
| `Snowflake/snowflake-arctic-embed-s` | 33.2M / 384 / 512 | The 33.4M-class sibling of shortlist row #3 — same family, same objective, same licence question. Taking both spends two slots on one hypothesis, and the *xs* is the more interesting one because it matches the incumbent's parameter count exactly. |
| `mixedbread-ai/mxbai-embed-xsmall-v1` | 24.1M / 384 / 4096 | **The closest near-miss.** It is built on `all-MiniLM-L6-v2` itself, so it is the most controlled objective-only swap in the landscape, and its 24.1M predicts ~7.0 ms. Cut because `arctic-embed-xs` occupies the same "same size, better objective" axis at a lower parameter count, and mxbai brings an AnglE-specific training config that would be a second change in the same experiment. |
| `intfloat/e5-small-v2` | 33.4M / 384 / 512 | Fills the same 33.4M/384/512 cell as `bge-small-en-v1.5` but adds a **mandatory `query:`/`passage:` prefix contract** on top of the recipe — two changes in one experiment, and the prefix has to be applied at both training and inference or the row is measured wrong. |
| `thenlper/gte-small` | 33.4M / 384 / 512 | Third member of the same cell, with no distinguishing property in any phase-1 column beyond being third. If `bge-small-en-v1.5` loses on latency, the evidence that its cell is worth revisiting comes from that row's failure, not from running a near-identical one. |
| `TaylorAI/bge-micro-v2` | 17.4M / 384 / 512 | Same slot as `paraphrase-MiniLM-L3-v2` and arguably a better model, but L3-v2 wins the slot because it shares the **incumbent's tokenizer and lineage**, so a delta is attributable to depth alone; bge-micro-v2 changes the pretraining data and the depth at once, and its result would be uninterpretable. |
| `jinaai/jina-embeddings-v2-small-en` | 32.7M / **512** / 8192 | See the sixth-model case below. |

**The sixth model, and the case for it.** `jinaai/jina-embeddings-v2-small-en` is genuinely defensible:
32.7M is inside the ceiling (~9.1 ms floor), Apache-2.0, plain `Transformer → Pooling → Normalize`
modules, and it is **the only surviving model that varies the embedding dimension** — 512 against the
shortlist's uniform 384. If anyone believes dim is a live hypothesis at this corpus size, this is the
row that tests it, and no other row can. I still ship five, for two reasons that are both in the
tables: (a) 512 dim doubles the logistic head's feature count at 194 training rows per fold, where the
brief's binding worry is data rather than capacity; and (b) it needs `model_args.trust_remote_code:
true` for its ALiBi attention, which under `transformers 5.6.2` is an untested load path — and
`phase3-env.md` §(f.5) already records that even SetFit *training* under `transformers 5` is untested,
so stacking a custom-modelling-code dependency on top of an unproven training path risks losing the
slot to an environment failure rather than to a result. The operator can promote it by swapping it for
any of rows #2, #5 in §2 without disturbing the funnel.

---

## 4. What phase 3 must prove about each survivor, and the order to probe them in

**The instrument** is `experiments/survey/probe_model.py` under the box's pinned stack, and it must
report, per model: `load_ms`, `params` (verified, not copied from the card), `embedding_dim`,
`max_seq_length`, `snapshot_bytes`, RSS before load / after load / after embed, and **p50 and p95 over
the real 382 rows** in `data/routing_382.json`, single utterance per call, warm, 5 passes. Manufactured
corroboration that the instrument works: `probe-sentence-transformers__all-MiniLM-L6-v2.json` already
reproduces 6.33 ms p50 / 8.014 ms p95 on 382 rows over 5 passes, matching the incumbent's stated
number.

Two further facts each row must yield, because §5 says they are unmeasured: the **`LICENSE` file in
the resolved snapshot** (for `arctic-embed-xs`), and the **tokenizer vocabulary size** (for all five —
not published in any phase-1 table, and the variable most likely to explain a latency surprise on
3–6-word inputs, where the embedding lookup is a large share of the forward pass).

### The pre-registered decision rule

Written before any number is seen, so the probe is a measurement rather than a judgement call.

1. **Drop without training if p50 > 10.0 ms**, the brief's ceiling, measured on the box over the 382
   rows, warm, batch 1, median of 5 passes. No exceptions and no "it's only 11".
2. **The absolute rule is cross-checked by a relative one: drop if p50 > 1.6 × the incumbent's
   same-protocol p50 on the same run.** 1.6 is 10.0 ÷ 6.33 — the same rule expressed in the box's own
   units, so a box that runs uniformly slower does not silently eliminate the whole shortlist.
3. **If the incumbent itself exceeds 10.0 ms on the same run, the harness is wrong, not the models.**
   Fix the protocol before dropping anything. This is the reason the incumbent is probed first: it is
   the canary. Note the box has already produced 12.478 ms for the incumbent under a batch-1
   unquiesced smoke test (`phase3-env.md` §(d)) — that number is below the drop threshold's protection
   and above the ceiling, which is exactly why rule 3 exists and why the smoke test's protocol must
   not be reused.
4. **Drop if the model does not load, does not embed all 382 rows, or reports a dim other than 384.**
   A wrong dim means the wrong artefact was resolved, and any latency from it is a measurement of a
   different model.
5. RSS and `snapshot_bytes` are **recorded but not thresholded**: the brief gives no memory budget for
   the Pi, so there is no number to fail. Missing, and I make no decision on it.
6. **Dropping a row is a phase-3 result, not a phase-2 failure**, and it must be written down with its
   number. If #4 is dropped, the shortlist loses its only "more capacity" arm and phase 4 becomes a
   within-class comparison — that consequence is accepted now, in advance, rather than renegotiated
   after the number appears.

### Predicted latency per survivor, and the threshold that kills it

All predictions are **estimates**, from `6.33 ms × (params ÷ 22.7M)^1.6` — the exponent fitted to the
box's own L3/L6 probe pair (param ratio 0.767 → latency ratio 0.653), anchored on the incumbent's
measured 6.33 ms. Each is bounded by the optimistic linear floor in the same units.

| # | model | params | predicted p50 (fitted, ^1.6) | optimistic floor (linear) | threshold — drop without training above |
|---|---|---:|---:|---:|---|
| 1 | `all-MiniLM-L6-v2` | 22.7M | 6.33 ms — **not an estimate, the measured anchor; the probe must reproduce it** | 6.33 ms | none; it is the canary (rule 3) |
| 2 | `paraphrase-MiniLM-L3-v2` | 17.4M | **4.1 ms** (ratio 0.653, measured on this box) | 4.9 ms | 10.0 ms — expected to pass with wide margin |
| 3 | `snowflake-arctic-embed-xs` | 22.6M | **6.3 ms** | 6.3 ms | 10.0 ms — expected to pass |
| 4 | `granite-embedding-30m-english` | 30.3M | **10.0 ms** | 8.5 ms | 10.0 ms — **the marginal row**; its own vendor "2× faster" claim, if it means anything, must land it nearer 8.5 |
| 5 | `bge-small-en-v1.5` | 33.4M | **11.7 ms** | 9.3 ms | 10.0 ms — **predicted to fail**; only a measurement below 10.0 saves it |

Read the spread honestly: the shortlist is predicted to lose one of five before phase 4, and its
highest-upside row is the one predicted to lose. That is a property of the constraint, not of the
selection — the 10 ms ceiling and a 33M-parameter encoder are only compatible if latency grows
sub-linearly, and the box's own two-point probe says it does not.

### The order to probe them in, and why

1. **`all-MiniLM-L6-v2`** — cached on the box already (`phase3-env.md` §(f.1)), zero download, and it
   is the canary for rule 3. Nothing else should run until the incumbent reproduces 6.33 ms.
2. **`paraphrase-MiniLM-L3-v2`** — already downloaded and on disk from `07_download_proof.py`, so it
   costs no bandwidth; it is the fastest predicted row and it exercises the harness at the low end.
3. **`snowflake-arctic-embed-xs`** — first real download, and the only row at the incumbent's parameter
   count, so its number is the cleanest same-size latency comparison the probe will produce.
4. **`ibm-granite/granite-embedding-30m-english`** — the marginal row, ordered after the three that
   should pass so a failure here does not block them.
5. **`BAAI/bge-small-en-v1.5`** — last, because it is the row most likely to be dropped and the other
   four should already be measured and banked if it is.

The logic is two rules: **zero-download models first** (they shake out the harness at no bandwidth
cost, and the box has no `HF_TOKEN`, so anonymous pulls are rate-limited — `phase3-env.md` §(f.11)),
and **the most-likely-to-fail model last**, so a drop removes a row rather than a queue.

---

## 5. What phase 1 could not establish

Every one of these is a cell a survivor's row rests on that is a vendor claim, unpublished, or an
artefact of metadata rather than a measurement. Phase 3 must produce the number rather than trust it.

1. **Every parameter count is metadata, not a measurement.** Family B's methodology note is explicit:
   params are the Hugging Face API's `safetensors.total`, "a direct read of the published checkpoint
   metadata". A weight count is not a FLOP count: `granite-embedding-30m-english` (30.3M, 6 layers,
   RoBERTa tokenizer) and `bge-small-en-v1.5` (33.4M, BERT tokenizer) can differ in real latency for
   reasons the parameter ratio cannot see. **The `params → latency` law in §1 and §4 is an estimate on
   top of an estimate and is the first thing the probe replaces with a measurement.** For the record,
   the two counts for the incumbent differ by source: 22,713,216 measured on this repo versus
   22,713,728 in family E's `safetensors` cell — 512 parameters, immaterial, but a reminder that these
   are counts of stored tensors.
2. **Almost no CPU latency exists to inherit.** Family B: **35 of 36** models "publish no latency,
   throughput or per-utterance figure of any kind." Family C: no CPU latency, per-utterance cost or
   CPU fine-tuning cost for **any** of its ten entries. Family A publishes exactly two speed claims and
   both are relative ("5 times faster", "18,000 GPU / 750 CPU queries per sec" with the hardware and
   batch unstated). So four of the five shortlisted rows have **no published latency at all** — their
   §4 predictions are pure estimation, and the only measured points in this document are the
   incumbent's and `paraphrase-MiniLM-L3-v2`'s.
3. **`granite-embedding-30m-english`'s speed claim may be unfalsifiable as stated.** "Twice as fast as
   other models with similar embedding dimensions" is a VENDOR CLAIM with no baseline named and no ms.
   Phase 3 can place granite in a ranking of the five; it **cannot** confirm or refute the "2×" claim,
   because there is no named comparator to be twice as fast as. If that claim was part of the reason
   the row survived, the honest reading is that it survived on its tokenizer and shape, not its speed.
4. **`arctic-embed-xs`'s licence is not cleanly declared.** Family A records it as undeclared in the
   repo with an Apache-2 statement in the paper; family D records an Apache-2 statement in the card
   body and no frontmatter field. A card body and a paper are not a `LICENSE` file. Phase 3 must read
   the resolved snapshot, and if there is no licence there the row is disqualified on the identical
   rule that removed `TaylorAI/gte-tiny` — with the difference that a substitution is cheap, since
   `arctic-embed-s` and `mxbai-embed-xsmall-v1` are §3.7 alternates.
5. **`max_seq` cells contradict themselves and cannot be trusted without reading the file.**
   `all-MiniLM-L12-v2`'s card says 256 while its `sentence_bert_config.json` says 128; the same split
   appears on `all-distilroberta-v1` (card 128, config 512) and `nomic-embed-text-v1.5` (config 2048,
   card 8192 via RoPE scaling). No shortlisted row is affected today — the corpus is 3–6 words — but
   the general lesson is that a `max_seq` in a phase-1 table is a claim about a file, not a reading of
   it.
6. **"Contrastive fine-tune support" is a claim about format, not about capability.** For all five
   survivors it means "ships `Transformer`+`Pooling` modules and a tokenizer", which is what SetFit
   needs to *accept* the model. It is not evidence that contrastive fine-tuning helps at 22 examples
   per intent — the only such evidence in this repo is one model, and it is `+4.17 ± 9.82` grouped on
   fold spread, i.e. a direction rather than a size.
7. **No accuracy number exists for any of the four non-incumbent survivors, and none can exist before
   phase 4.** Nothing in phase 1 measured a classifier; nothing in phase 3 will. Every accuracy claim
   in this document about row #2–#5 is therefore **absent by design**, and no ordering in §2 should be
   read as a prediction of accuracy — the rows are ordered for the probe, not ranked.
8. **The recipe itself is untested on this stack.** `phase3-env.md` §(f.5) records that `setfit 1.2.0`
   under `transformers 5.6.2` **imports** cleanly but whether it **trains** is untested, and §(f.4)
   records that `pyproject.toml` declares `requires-python >= 3.12` against a 3.11.9 box. This is a
   risk to all five rows equally and it is phase 4's first gate, not a property of any survivor.
9. **The anchor's conditions are recorded loosely.** The 6.33 ms anchor is stated as an 8-core idle dev
   box, while the probe artefact that reproduces it records 12 logical CPUs and `loadavg_1m` 8.286 at
   start, 8.704 at end. If 6.33 ms was taken under mild load, a genuinely idle box would be faster,
   which moves the 35.9M ceiling *up* and makes F1 conservative — in the direction that risks losing a
   model that would have passed. `bge-small-en-v1.5` is the row that risk sits on, and rule 2 of §4's
   threshold list (relative to the incumbent, same run) is what protects it from a mis-scaled anchor.
10. **The reason no model above 1.5× the incumbent was shortlisted is not a constraint — it is an
    absence of evidence.** The ladder's only attempt at a larger-capacity architecture was rung 4, and
    what it demonstrated was that a 22M encoder could not be moved by 55 optimizer steps (68.6%
    positive/negative separation on 582 pairs) until the learning rate and class weighting were fixed.
    Nothing in phase 1 measures whether 100M parameters helps at 200 rows, because nothing in phase 1
    measured *anything*. So the ceiling on this shortlist is a latency claim, and if phase 4 finds no
    gain among these five, that is a result about the 10 ms budget, not a refutation of larger base
    models.
