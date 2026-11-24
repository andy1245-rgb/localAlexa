# Independent audit — phase 1 and phase 2

2026-11-23 branch `survey/base-model-2026-11-23`. Auditor had no hand in any of the material below.

**Read:** `docs/base-model-survey-brief-2026-11-23.md`; `results/survey-2026-11-23/phase1-{A,B,C,D,E}.md`;
`results/survey-2026-11-23/phase2.md`. Cross-checked against `docs/ladder-results-2026-11-10.md`,
`results/ladder/latency.json`, `results/ladder/setfit-rung.json`, `data/routing_382.json`, the phase-3
probe artefacts in the same directory, and live sources.

**Live verification performed** (not recollection): Hugging Face model API (`safetensors.total`,
`cardData.license`, sibling file lists), raw `sentence_bert_config.json` / `config.json` / `README.md`
per model, `https://sbert.net/docs/sentence_transformer/pretrained_models.html`, arXiv abs pages, and
`?blobs=true` file sizes for the ONNX exports. Every cited URL in every phase-1 row resolved.

**Row census, counted by script, not by eye:**

| table | data rows | rows w/o source URL | rows w/o date read | rows with all 9 columns |
|---|---:|---:|---:|---:|
| phase1-A.md | 14 | 0 | 0 | 14 |
| phase1-B.md | 36 | 0 | 0 | 36 |
| phase1-C.md | 10 | 0 | 0 | 10 |
| phase1-D.md | 16 | 0 | 0 | 16 |
| phase1-E.md | 8 | 0 | 0 | 8 |
| **total** | **84** | **0** | **0** | **84** |

84, not the 85 phase 2 claims (line 13). That single number is the root of the funnel defects in §4.

---

## 1. Evidence discipline in phase 1 — **PASS**, with one defect

**What I verified rather than assumed.** Zero rows lack a source URL, zero lack a date, every row
carries all nine columns, and every date is `2026-11-23`. Of 346 digit-bearing cells, 120 carry an
explicit in-cell provenance note (`safetensors.total`, `card`, `config.json`, `paper §5.1`,
`derived: … ÷ 4`, `VENDOR CLAIM`); the other 226 are bare values (`384`, `512`, `MIT`, `22.7M`) that
inherit the row's URL column. That is the table style the brief asked for — one source and one date
per row — not 226 unsourced claims.

**Two rows per table checked against live content, model id and at least one numeric claim:**

| table | row | claim in the file | what the source says now |
|---|---|---|---|
| A | A1 (`phase1-A.md:8`) | 22.7M, max_seq 256, "5 times faster" | `safetensors.total` 22,713,728; `sentence_bert_config.json` `max_seq_length: 256`; sbert.net verbatim: "all-MiniLM-L6-v2 is 5 times faster and still offers good quality" |
| A | A10 (`:17`) | 22.6M, 512, "not declared in the repo … paper states Apache-2" | 22,565,376; `max_seq_length: 512`; `LICENSE` → **HTTP 404**; README line 3068 "Arctic is licensed under the Apache-2" |
| B | B1 (`:25`) | 33.4M, 384, MIT, 512 | 33,360,512; `license: mit`; dim 384; `max_seq_length: 512` |
| B | B19 (`:43`) | 305M, apache-2.0, "10x increase in inference speed" | 305,369,089; `license: apache-2.0` |
| C | C1 (`:9`) | 751,632,384 | 751,632,384 |
| C | C6 (`:15`) | 134,515,008 | 134,515,008 |
| D | D3 (`:12`) | 17.4 M, 128, apache-2.0 | 17,390,336; `max_seq_length: 128`; `license: apache-2.0` |
| D | D6 (`:15`) | 30.3 M, 384, "twice as fast as other models with similar embedding dimensions" | 30,295,296; README line 183 carries that sentence verbatim; `license: apache-2.0` in frontmatter |
| E | E1 (`:11`) | body 109,486,978 | `sentence-transformers/paraphrase-mpnet-base-v2` = 109,486,978 exactly |
| E | E7 (`:17`) | 2,849,757,184 | `bigscience/T0_3B` = 2,849,757,184 exactly |

**The evidence that settled "recollection vs read" for me** is the `max_seq` column. Several values are
counter-intuitive and every one is correct against `sentence_bert_config.json`:
`msmarco-MiniLM-L6-cos-v5` **384** (not 512), `msmarco-MiniLM-L12-cos-v5` **384**,
`all-mpnet-base-v2` **384** (its sibling `paraphrase-mpnet-base-v2` **512**),
`paraphrase-MiniLM-L6-v2` **128**, `all-MiniLM-L12-v2` **128** with the card text at line 115 saying
"longer than 256 word pieces is truncated" — the A2 cell says exactly that contradiction and it is real.
ONNX byte counts also reproduce exactly: `Xenova/all-MiniLM-L6-v2` 90.4 / 23.0 / 22.8 MB and
`Xenova/bge-small-en-v1.5` 133.1 / 33.8 / 36.2 MB are the published file sizes to the decime.
Nobody writes 384 for msmarco from memory.

**The one defect.** `phase1-E.md:11` — the incumbent's own speed cell:

> a 2026 preprint reports SetFit at 8 examples/class "in ~2ms" (AUTHOR CLAIM, body not stated)

That is a number in the `published speed` column with **no URL and no identifier** — not the arXiv id,
not the title, not the venue. It is the only numeric claim in 84 rows that a reader cannot follow. It
also sits on the row that anchors the whole survey.

**Verdict:** PASS on the discipline the brief asked for; one untraceable cell.

---

## 2. Vendor claims marked — **PARTIAL**

The tables apply `VENDOR CLAIM` (and `AUTHOR CLAIM` in E) to **performance** numbers, consistently and
usefully: A1, A7, A8, A9, B19, C10, D6, D9, D10, D11, D12, D13, D14, D15, E5, E8 all carry the tag,
and I spot-checked four of them against the source wording (A1 "5 times faster", D6 "twice as fast",
A8/A9 "18,000 / 750" and "11,000 / 400" — all verbatim on sbert.net). So the convention is real.

**2a. Author-supplied speed numbers carried without the tag** — all in family E, which defines
`AUTHOR CLAIM` and then does not use it on three of its own rows:

| file:line | model | unmarked number |
|---|---|---|
| `phase1-E.md:11` | SetFit v1.2.0 (mpnet body) | "training ≈30 s per split on p3.2xlarge (16 GB GPU) vs ≈700 s for T-Few 3B (paper §5, Table 5)" — tagged nowhere in the cell, while the same cell tags the "~2ms" preprint |
| `phase1-E.md:16` | PET / roberta-large | "training ≈30 min per PET classifier for 250 steps on one GTX 1080 Ti, ≈2 h for the final 5000-step classifier (paper appendix D)" |
| `phase1-E.md:17` | T-Few / T0_3B | "≈700 s to train on p4d.24xlarge (≥40 GB GPU) and ≈1 order of magnitude more inference FLOPs than SetFit-mpnet" |

The contrast is inside one table: E5 and E8 tag their author claims, E1/E6/E7 do not. E5's
"4K tokens (FastFit §5.1, AUTHOR CLAIM)" is also a claim about a **third party's** model (flan-t5) by
FastFit's authors, so it is not the same thing the tag means elsewhere.

**2b. Vendor-supplied parameter counts carried without the tag.** The brief says "mark vendor claims as
vendor claims"; phase 1's own gloss (`phase1-A.md:3`) is "a number published by the model's own
author/vendor". By that definition these are vendor numbers and none carries the tag:
`phase1-B.md:41,42` ("137M (card table)", "434M (card table)"), `:50,51,52` ("239M (card)",
"677M (card)", "~1.74B (card)"), `:58,59` ("400M (repo name)", "1.5B (repo name)"),
`phase1-C.md:9,10` ("0.44B non-embedding (card)"), `:13` ("card states 0.6B, 28 layers"),
`:18` ("blog breaks this into 170M embeddings + 100M transformer blocks"),
`phase1-A.md:21` ("card says 300M"), `phase1-D.md:22` ("0 active params (card: …)").

This is a defensible narrowing (params are checkable metadata, speeds are not) but it is not what the
brief asked and it is not stated as a rule anywhere. A reader cannot tell whether the absence of the tag
means "independently verified" or "vendor-stated but harmless".

**2c. Cases tagged as vendor where the source is not the model's author** — none found. Every
`VENDOR CLAIM` I traced resolved to the checkpoint's own publisher, except the E5 case above and one
labelling inconsistency: D14 and D15 tag claims that come from **papers** (`arXiv:1909.10351`,
`arXiv:1910.01108`) as `VENDOR CLAIM` while family E would call the same thing `AUTHOR CLAIM`.

**Verdict:** PARTIAL. The tag exists and is applied; it is applied to one class of claim only, and
family E contradicts itself on three rows.

---

## 3. No recommendations in phase 1 — **PASS**

I grepped all five files for ranking and recommendation language (`recommend|best|better|prefer|should|
winner|supersede|cleanest|worth|promising|beats|candidate|instead of`) and read every hit in context.
Every hit is descriptive:

- `phase1-A.md:14` "trained only on inputs up to 250 tokens" — a card fact.
- `phase1-B.md:37` "says >4096 is not recommended" — reporting the card.
- `phase1-D.md:6` and `phase1-E.md:7` — "Descriptive only — no ranking", the authors' own declaration.
- `phase1-B.md:21` — "No claim here about whether fine-tuning is a good idea at this corpus size."

No sentence in any phase-1 table ranks models, names a winner, or steers a choice. Family D opens with
"No ranking" and family C closes by disclaiming capability inferences ("not claimed to be incapable of
it — only that nothing published says how"). The discipline held.

One consequence to note, because it lands in check 7: family E was *asked* a ranking question by the
brief ("is it still the best few-shot recipe in 2026, or has something superseded it") and avoided
answering it rather than answering it. That is compliance with "no recommendations" and a gap against
the brief at the same time.

**Verdict:** PASS.

---

## 4. The funnel arithmetic — **FAIL**

### 4a. The entry count is wrong

Phase 2 line 13: "The five phase-1 tables contain **85 rows**". There are **84** (14 + 36 + 10 + 16 + 8,
counted by script over the five files). Line 20 repeats it: "All 85 rows are accounted for in §2 or §3."
They are — 84 of them.

### 4b. The distinct count and the rename list

Line 13 says 74 distinct from 85 rows with 11 re-naming rows. The 11 are: five same-repo duplicates
(`modernbert-embed-base` A13/B32, `arctic-embed-xs` A10/D7, `mxbai-embed-xsmall-v1` A12/D9,
`jina-embeddings-v2-small-en` B22/D8, `embeddinggemma-300m` A14/D16 — all five correct), two ONNX
re-exports (correct), and "four family-E rows are recipes whose *body* is counted elsewhere (SetFit on
`paraphrase-mpnet-base-v2`, SetFit on `all-MiniLM-L6-v2`, frozen MiniLM-L6 + kNN, **FastFit-large's
`roberta-large` body**)".

The last one is wrong. `FacebookAI/roberta-large` is counted in no A–D table; it enters phase 1 only as
a family-E body, and phase 2 itself treats it as a distinct artefact at line 300 ("`FacebookAI/roberta-large`
355.4M (≥99.1 ms, reached twice: as FastFit-large's body and as PET's backbone)"). Meanwhile the row
that *is* a duplicate and is omitted from the list is E3, **FastFit small**, whose body is
`paraphrase-mpnet-base-v2` — the same body as E1 and the same repo as A5. Two compensating errors that
preserve the number 11 while the membership is wrong.

Net: 84 rows − 11 re-naming rows = 73 checkpoints named by rows, +1 for `Gemma 3 270M` and `270M-IT`
both sitting inside row C10 = **74 artefacts, or 72 if the Gemma variants count as one model** (which is
how phase 2 treats them at line 296). Phase 2's 74 is reachable only on the split-Gemma reading while
its row count says 85. The two numbers cannot both be right.

### 4c. Every phase-1 row *is* dispositioned — that part passes

I checked all 84 rows against §2 and §3: each maps to exactly one disposition, none is orphaned, none
appears in both a survivor slot and an elimination with contradictory reasons. The multi-references are
consistent (`paraphrase-mpnet-base-v2` at §3.1 and §3.6 both eliminated; `all-MiniLM-L6-v2` at §2 row 1
and §3.6 as the instrument; `roberta-large` at §3.1 and §3.6). And no model in phase 2 is absent from
phase 1 — I scanned every backticked identifier in phase 2 against the five tables and the only unmatched
tokens are file paths (`data/routing_382.json`, the two docs, the branch name).

### 4d. The per-family counts in §1 contradict §3.1

Line 66–67: "family C (10 of 10), **31 of 36 rows in family B**, 5 of 14 in family A, and **3 of the 8
family-E rows**".

- Family B: §3.1 itself lists **32** B-family models (BGE 5 + E5 6 + GTE 7 + Jina 6 + Nomic 4 + Stella 4),
  which is every B row except B1, B7, B14, B22. 32, not 31. The 31 is only reached by silently excluding
  B32 (`modernbert-embed-base`) from the B count while counting it in A's 5 — but §3.1 lists it under
  Nomic with the note "counted once despite two rows", i.e. it *is* in the B list.
- Family E: "3 of the 8 **rows**" is not a row count. The E bodies eliminated at F1 are reached by four
  rows (E4 and E6 both carry `roberta-large`) or by three distinct bodies (`roberta-large`, `T0_3B`,
  `flan-t5-xxl`). Either way "3 of the 8 rows" is right on neither basis.

### 4e. Does 74 − eliminations = 5? Only as printed, and only by double-counting

The printed chain is: 74 → −50 → 24 → −4 → 20 → −0 → 20 → −0 → 20 → −7 → 13 → −8 → 5.

The endpoints are right. **The middle is not.** F5's "removed 7 of 20" (line 138) removes
`potion-base-8M`, `potion-base-32M`, `potion-retrieval-32M`, `static-retrieval-mrl-en-v1`,
`TinyBERT_General_4L_312D` — **five distinct models** — plus `Xenova/all-MiniLM-L6-v2` and
`Xenova/bge-small-en-v1.5`, which line 15–16 says are "ONNX re-exports of a model counted elsewhere" and
therefore are not in the distinct count at all. You cannot subtract them from a population they were
never added to. So either they are distinct models (and the rename list is 9, not 11, and the count is 76),
or the F5 removal is 5 and the population was 18.

The self-consistent repair, which lands on the same 13 and the same 5:

> 84 rows → 22 distinct trainable candidates survive F1 (I enumerated them: 9 from A, 4 from B, 9 from D)
> → 4 die at F2 → 18 → F3 0 → F4 0 → 5 distinct die at F5 → **13** → 8 cut at shortlisting → **5**.

Proof that 22 and not 24 is the right post-F1 figure, from phase 2's own lists: 13 survivors + 5 distinct
F5 kills + 4 F2 kills = 22. The 24 requires the Xenova pair to be inside the population, which §1 says
it is not.

### 4f. One mis-attributed kill

`stella-base-en-v2` is counted inside F1's 50 (line 288) but its stated reason is F5's rule: "eliminated
on the *stronger* ground that it ships no `modules.json` and its pooling is only in a card snippet". It
is absent from §3.5's list. The count survives; the attribution does not.

**Verdict:** FAIL. The two endpoints (13 → 5) are correct and reproducible; every intermediate number
between the 84 rows and the 13 is inflated by two, and three of the prose counts contradict §3.1.

---

## 5. The latency filter's arithmetic — **PASS on the 50 recomputations, FAIL on the anchor**

### 5a. Every printed estimate recomputes

I recomputed all 50 using phase 2's own stated law, `6.33 ms × (params ÷ 22.7M)`, from the phase-1
parameter counts. **50 of 50 match to the printed precision.** Six, so the reader can check:

| model | phase-1 params | printed | recomputed `6.33 × p/22.7` | match |
|---|---:|---:|---:|---|
| `BAAI/bge-base-en-v1.5` | 109.5M | ≥30.5 ms | 30.54 | ✓ |
| `intfloat/multilingual-e5-small` | 117.7M | ≥32.8 ms | 32.81 | ✓ |
| `Alibaba-NLP/gte-large-en-v1.5` | 434M | ≥121 ms | 121.0 | ✓ |
| `jinaai/jina-embeddings-v5-text-small` | 677M | ≥189 ms | 188.8 | ✓ |
| `Qwen/Qwen3-1.7B` | 2030M | ≥566 ms | 566.1 | ✓ |
| `sentence-transformers/all-distilroberta-v1` | 82.1M | ≥22.9 ms | 22.90 | ✓ |

The four shortlist predictions also recompute exactly on the `^1.6` form: L3-v2 4.14 (printed 4.1),
arctic-xs 6.29 (6.3), granite 10.05 (10.0), bge-small 11.74 (11.7). Nothing is fudged.

### 5b. No model was eliminated *close enough* to the ceiling for the estimate to be the problem

The nearest F1 casualty above 35.9M is `distilbert-base-uncased` at 67.0M — **1.87× the ceiling**, with
an estimate of 18.7 ms against a 10 ms budget. The next are `all-distilroberta-v1` (82.1M) and
`embeddinggemma-300m` (302.9M). Nothing sits in 36–50M. **On the elimination side the estimate is not
load-bearing anywhere.**

### 5c. The problem is the opposite direction: rows *admitted* by the charitable law

Seven of the thirteen models that survive all five constraints sit in a 30.3M–33.4M band where the
document's own two laws disagree about whether they belong:

| model | params | linear floor (admits) | `^1.6` fitted (rejects) |
|---|---:|---:|---:|
| `granite-embedding-30m-english` | 30.3M | 8.5 ms | 10.0 ms |
| `snowflake-arctic-embed-s` | 33.2M | 9.3 ms | 11.6 ms |
| `BAAI/bge-small-en-v1.5` | 33.4M | 9.3 ms | 11.7 ms |
| `all-MiniLM-L12-v2`, `intfloat/e5-small-v2`, `thenlper/gte-small` | 33.4M | 9.3 ms | 11.7 ms |
| `jinaai/jina-embeddings-v2-small-en` | 32.7M | 9.1 ms | 11.5 ms |

F1 admits them on the linear law (line 51) and §4 predicts them out on the `^1.6` law (lines 413–417).
Under `^1.6` the admission ceiling is **30.2M**, which also rejects `granite-embedding-30m-english`
(predicted 10.05 ms) — i.e. one of the five survivors fails the document's own better-fitting law as
well as the linear one. The doc says it chose the permissive cut deliberately (line 60), but a filter
applied permissively to admission and strictly to prediction is not one filter.

### 5d. The anchor is not the number in the brief, and §4's headline prediction contradicts §2

The brief states the incumbent at **7.75 ms**; `results/ladder/latency.json` gives rung 3
(`r3-setfit-minilm`) `p50_ms: 7.75` over 5 passes on 382 rows under the documented protocol. Phase 2
never mentions 7.75 anywhere (0 occurrences) and anchors on **6.33 ms** — which comes from
`probe-sentence-transformers__all-MiniLM-L6-v2.json`, an **encoder-only embed** run on
`probe-box` with `torch 2.15.0+cpu`, `python 3.12.14`, 12 logical CPUs, `loadavg_1m` 8.286
→ 8.704. Three consequences the document does not draw:

1. The ceiling is anchor-dependent. `22.7 × 10 / 7.75` = **29.3M**, and at 29.3M four of the five
   survivors (`bge-small` 33.4M, `granite` 30.3M) and most of §3.7's eight alternates would fail F1
   outright. The 35.9M ceiling is not a property of the model; it is a property of which number was
   picked.
2. Line 43 describes the anchor as "22.7M params, **8-core dev box**, 5 passes, warm, **idle**". The
   artefact it cites says 12 cores and loadavg 8.3–8.7. §5.9 notices this; §1's framing is left standing.
3. The 6.33 figure is under a dependency stack the brief does not pin (torch 2.15 / py3.12 against the
   box's torch 2.11.0+cpu / py3.11.9). §4 requires the instrument to run "under the box's pinned stack";
   the artefact that supposedly proves the instrument works does not meet §4's own spec.

**The mixed-anchor error, stated precisely.** Line 414 predicts `paraphrase-MiniLM-L3-v2` at "**4.1 ms**
(ratio 0.653, measured on this box)". The ratio 0.653 is `8.152 / 12.478` — a pair measured against a
**12.478 ms** anchor. Multiplying it by a 6.33 ms anchor divides the ratio and the anchor by two
different worlds. Applied to the anchor it was actually measured against, the same two numbers predict
`12.478 × 0.653` = **8.15 ms**. §2 row 2, four pages earlier, says the same model "is already downloaded
and measured on the box at 8.152 ms single-sentence". So phase 2 quotes 8.152 as a virtue for the row
and 4.1 as its prediction, and the two cannot both be what the row is expected to do.

**Corroboration from the box, offered as evidence the auditor found and phase 2 predates**
(`phase2.md` mtime 2026-11-24 01:30; probes written 01:32–01:36, `probe-box`, `torch 2.11.0+cpu`,
py 3.11.9, 8 CPUs):

| artefact | model | p50 |
|---|---|---:|
| `probe-canary-end-MiniLM-L6.json` | `all-MiniLM-L6-v2` | **15.575 ms** (6.33 predicted) |
| `probe-L3-v2.json` | `paraphrase-MiniLM-L3-v2` | **8.387 ms** (4.1 predicted; §2 quoted 8.152) |
| `probe-arctic-xs.json` | `snowflake-arctic-embed-xs` | **12.126 ms** (6.3 predicted) |
| `probe-granite-30m.json` | `granite-embedding-30m-english` | **11.696 ms** (10.0 predicted) |
| `probe-bge-small-en-v15.json` | `BAAI/bge-small-en-v1.5` | **28.189 ms** (11.7 predicted) |

Under §4's own pre-registered rule 1 (drop if p50 > 10.0 ms) four of five survivors fail; under rule 3
the incumbent itself exceeds 10 ms, so the harness is the finding, not the models. That is a phase-3
result and outside this audit's scope — but it is the reason the anchor in §1 must be fixed before the
number is copied into the final document.

**Verdict:** PASS on the arithmetic inside the filter (50/50 recompute, no marginal elimination);
FAIL on what the filter is anchored to and on using two different laws for admission and prediction.

---

## 6. The five survivors against the brief's constraints — **PARTIAL (3 clean, 2 resting on weak ground)**

Brief constraints: permissive licence, CPU-only inference, available as a loadable encoder, params
within the latency budget. Verified live per survivor:

| # | model | licence (frontmatter?) | CPU | loadable encoder | params | box p50 |
|---|---|---|---|---|---|---|
| 1 | `all-MiniLM-L6-v2` | `apache-2.0` ✓ frontmatter | ✓ | ✓ `modules.json` + `sentence_bert_config.json` | 22.7M ✓ | 15.575 ms |
| 2 | `paraphrase-MiniLM-L3-v2` | `apache-2.0` ✓ frontmatter | ✓ | ✓ | 17.4M ✓ | 8.387 ms |
| 3 | `snowflake-arctic-embed-xs` | **no frontmatter field, no `LICENSE` (404), card prose only** | ✓ | ✓ | 22.6M ✓ | 12.126 ms |
| 4 | `BAAI/bge-small-en-v1.5` | `mit` ✓ frontmatter | ✓ | ✓ | 33.4M — inside the linear ceiling, outside the `^1.6` and outside the ladder-anchored ceiling | 28.189 ms |
| 5 | `ibm-granite/granite-embedding-30m-english` | `apache-2.0` ✓ frontmatter | ✓ | ✓ | 30.3M — predicted 10.05 ms, i.e. exactly on the ceiling | 11.696 ms |

**6a. The licence this check exists to find.** `Snowflake/snowflake-arctic-embed-xs` — shortlist row 3,
`phase2.md:184` — is the one survivor whose licence is not in the model card frontmatter. I confirmed
the state directly: `cardData.license` is null, there is no `license:` tag, `GET /resolve/main/LICENSE`
returns **404**, and the only statement is README line 3068, "Arctic is licensed under the Apache-2."
Phase 2 admits all of this (line 184, §5.4) and ships the row anyway, deferring the check to phase 3.

The doc's characterisation of the A/D discrepancy is also softer than the facts. Line 80 says the two
phase-1 rows "disagree about *where* the statement lives, not about what it says". `phase1-A.md:17` says
"**not declared in the repo** (no LICENSE file, no licence tag)" — which is exactly what the HF API and
the 404 confirm — and attributes the Apache-2 statement to the **paper**. `phase1-D.md:16` attributes it
to the **card body**. Both are true of different documents; neither describes the repository as
declaring a licence. The card-body reading is the one that saves the row, and it is the one family A did
not make.

I checked whether this was applied consistently against the four F2 kills: `TaylorAI/gte-tiny`,
`multi-qa-MiniLM-L6-cos-v1` and `msmarco-MiniLM-L6-cos-v5` have READMEs that are **silent** on licence
(verified by grep) while `arctic-embed-xs` speaks. So F2's rule — "frontmatter field, `LICENSE` file, or
card body" — was applied consistently. The gap is that the brief's licence constraint plus a public repo
wants a file, and this row has prose.

**6b. Row 4 does not satisfy the brief's latency constraint on the document's own better law.** The brief
says "a candidate that cannot answer inside ~10 ms on CPU is **out**, whatever its accuracy". Row 4 is
predicted at 11.7 ms by the fit phase 2 prefers, admitted at 9.3 ms by the fit phase 2 disclaims, and
measured at 28.2 ms on the box. Its justification ("highest-upside row the ceiling still admits") is a
statement about the permissive ceiling, not about the constraint.

**6c. Row 5's speed evidence is disclaimed in the same document.** `phase2.md:186` survives the row
partly on "the only surviving row carrying a published speed claim to falsify". §5.3 then says the claim
is unfalsifiable — "there is no named comparator to be twice as fast as" — and that "the honest reading
is that it survived on its tokenizer and shape, not its speed". Slightly overstated: the granite card's
own table (line 197) names `bge-small-en-v1.5` among "other models with similar embedding dimensions",
so a comparator is nameable even if the speed claim does not name one. The RoBERTa-tokenizer claim does
check out — the card says "based on an encoder-only RoBERTa like transformer architecture".

**6d. Rows 1–3 pass on paper.** Row 2 (`paraphrase-MiniLM-L3-v2`, 17.4M, apache-2.0 frontmatter,
`max_seq_length: 128`) is the only survivor that clears every constraint on every basis, including the
box's own measurement. Row 1 is the mandated reference. Row 3 passes only on the card-prose licence.

**Verdict:** PARTIAL — 3 of 5 clean; #3 rests on a licence with no frontmatter field and no `LICENSE`
file; #4 and #5 survive on the maximum-charitable reading of the latency law.

---

## 7. What the brief required that is missing — **PARTIAL**

The brief asked each phase-1 row for: name, params, embedding dim, max sequence length, licence,
contrastive fine-tuning support, published speed, and the date the claim was read.

**All eight columns exist in all five tables**, plus the source URL. That part is complete — no column is
missing and none is systematically empty of *attempts*.

| column | state across 84 rows |
|---|---|
| name, source URL, date read | 84/84 |
| params | 84/84 populated; 3 are derived rather than published (`phase1-B.md:28` bge-m3 568M from a 2271 MB fp32 bin, `:60` stella-large-zh from 1.3 GB, `phase1-D.md:20` TinyBERT from a 62.7 MB bin) and 2 are explicitly "not published" (`:57`, `:60`) with a size given instead |
| embedding dim | 84/84 |
| max seq len | 83/84 — `phase1-E.md:17` T-Few is "not published (paper reports median combined input+target length of 103 tokens)" |
| licence | 84/84 stated; 4 rows state the absence of one (`A:14,15,16`, `D:14`) |
| contrastive support | 84/84, though family B's own preamble admits it grades **format** ("the module chain in `modules.json`"), not capability — phase 2 §5.6 catches this honestly |
| published speed | 84/84 populated, and **effectively empty**: no row in any family publishes a CPU per-utterance latency or throughput in ms. Family B: 35 of 36 "not published"; family C: 10 of 10; A: 12 of 14 "not published" + 2 relative claims; D: 8 VENDOR CLAIMs, none in ms; E: author claims in seconds/minutes of *training*, none of inference on CPU |

That last row is the real gap and it is not phase 1's fault — but it means the brief's
"published speed" column supplies **zero** usable inputs to a filter whose entire first stage is a
latency ceiling. Every number in §3.1 and §4 is an extrapolation from one anchor. Both documents say so;
the final document must say it in the same sentence as the shortlist, not four pages later.

**Missing deliverables, as distinct from missing columns:**

1. **Family E never answers its own question.** The brief assigned E "is it still the best few-shot
   recipe in 2026, or has something superseded it". `phase1-E.md` is a table of eight recipes plus a
   provenance note; it contains no sentence about whether SetFit is still the best and none about what
   would supersede it. Phase 2 §3.6 says SetFit "survives, as the instrument" — an assumption, not a
   finding. Nothing anywhere establishes that the recipe being carried into phase 4 is still the right one.
2. **No CPU inference cost for any of the ten decoder LLMs in family C**, which the brief specifically
   asked that family for ("their fine-tune cost on CPU"). `phase1-C.md` says so explicitly at line 32 and
   is right to; but the brief's question is then unanswered, and phase 2 §3.1's "no decoder LLM of any
   size … is within an order of magnitude of the budget" is therefore an inference from parameter counts
   alone, on a family whose decode loop is not parameter-proportional at all.
3. **`docs/base-model-survey-2026-11-23.md` does not exist.** The brief requires every phase to write its
   section there so a reader sees the eliminations and not only the winner. Phase 1 and 2 live only under
   `results/survey-2026-11-23/`. The brief says nothing is committed until phase 4 closes, so this may be
   intentional sequencing — but as of this audit the consolidated document has no phase-1 or phase-2
   section in it.

**Verdict:** PARTIAL — columns complete, content gaps in two of the five assigned questions.

---

## What to fix before this goes in the final document

Ordered by severity. Items 1–4 change numbers a reader will check; 5–8 change claims; 9–11 are accuracy
polish.

1. **Correct the funnel's entry count and re-derive the middle (check 4).** `phase2.md:13` "85 rows" →
   **84**. Then either fix the "removed 7 of 20" at F5 (line 138) to 5 distinct models, or state the
   population as including the two ONNX re-exports — not both. The defensible chain is
   **84 rows → 22 trainable candidates after F1 → 18 after F2 → 13 after F5 → 5**. The 13 and the 5 do
   not move; ten intermediate numbers do. Also fix line 66 "31 of 36 rows in family B" → **32**, and
   line 67 "3 of the 8 family-E rows" (say which basis: 4 rows or 3 bodies).

2. **Reconcile the 6.33 ms anchor with the 7.75 ms the brief and the ladder both give (check 5d).** The
   35.9M ceiling is `22.7 × 10 / 6.33`; at the ladder's 7.75 it is **29.3M**, and four of the five
   survivors sit above that. Say which anchor the filter uses and why, state that the 6.33 artefact is
   encoder-only on a different host and a different torch/python than the pinned box, and drop "8-core …
   idle" from line 43 (the artefact says 12 CPUs, loadavg 8.3–8.7). Without this, the single number the
   whole elimination stage rests on is unverifiable from the brief backwards.

3. **Fix the mixed-anchor prediction for `paraphrase-MiniLM-L3-v2` (check 5d).** Line 414's 4.1 ms applies
   a ratio measured against 12.478 ms to a 6.33 ms anchor; the same pair yields **8.15 ms**, which is what
   §2 row 2 already quotes (8.152) and what the box measured (8.387). Either prediction is defensible on
   its own terms; 4.1 and 8.152 in the same document is not.

4. **Stop using two laws for one filter (check 5c).** F1 admits on the linear floor and §4 predicts on
   `^1.6`. Under `^1.6` the admission ceiling is 30.2M and `granite-embedding-30m-english` (predicted
   10.05 ms) is out too. State the law once, apply it to both admission and prediction, and if the
   permissive cut is kept for admission say plainly that it is a courtesy to the probe, not a pass.

5. **Resolve `arctic-embed-xs`'s licence or demote the row (check 6a).** No frontmatter field, no
   `LICENSE` file (404), card prose only — I re-verified all three. It is the one survivor that does not
   meet "licence must be permissive" on the artefact as the other four do. Either read the resolved
   snapshot and cite the line, or swap in `arctic-embed-s` (33.2M, `apache-2.0` in **frontmatter**) and
   say so. Do not carry a prose licence into a public repo's credits.

6. **Say out loud that `published speed` contributed nothing (check 7).** 0 of 84 rows publish a CPU
   per-utterance latency. Every ms in §3.1 and §4 descends from one anchor. Put that sentence next to the
   shortlist, where a reader meets the five names, not in §5 after they have read 400 lines.

7. **Answer family E's question or record it as unanswered (check 7).** The brief asked whether SetFit is
   still the best few-shot recipe in 2026. Neither phase 1 nor phase 2 says. If the answer is "we did not
   establish it", write that and note what would (a FastFit head-to-head is the obvious candidate, and
   phase 2 already eliminated FastFit on latency without checking whether it wins on accuracy at
   identical cost).

8. **Tag the three unmarked author claims and pick one vocabulary (check 2a/2c).** `phase1-E.md:11`
   (30 s / 700 s), `:16` (30 min / 2 h), `:17` (700 s, FLOPs ratio) — tag them `AUTHOR CLAIM` as E5 and
   E8 already do, or reword to "the paper reports, untagged". Then choose one of `VENDOR CLAIM` /
   `AUTHOR CLAIM` and use it in D14/D15, where paper claims are currently labelled vendor.

9. **Give the "~2 ms" preprint an identifier or delete it (check 1).** `phase1-E.md:11` — the only
   numeric claim in 84 rows a reader cannot follow, on the anchor row.

10. **Fix the F5 attribution of `stella-base-en-v2` (check 4f).** Counted in F1 (line 288), reasoned as an
    F5 kill. Move it or move the reason; the count is right either way.

11. **Three precision fixes.** `phase2.md:184` "the only candidate at or below the incumbent's parameter
    count with twice the usable `max_seq`" — `TaylorAI/bge-micro-v2` is 17.4M / 512 / MIT and is inside
    the 13; the sentence is only unique once the infoNCE clause is added. `phase2.md:291` "239M is the
    closest any non-permissive model came to the ceiling" — 239M is 6.7× the ceiling; "closest" is true
    and "sharpest loss" is not. `phase2.md` §5.3 "no named comparator" — the granite card's own table
    names `bge-small-en-v1.5` as a peer, so a comparator exists even though the speed claim does not
    name one.

**What holds.** The phase-1 tables are real reads, not recollection: 84 rows, 84 sources, 84 dates, every
parameter count I could reach matched to the digit including three counter-intuitive `max_seq` values and
six ONNX byte counts, and 50 of 50 latency estimates recompute. Phase 1 contains no recommendation. The
shortlist of five and the 13-model cut are the right numbers. The defects are in the funnel's bookkeeping
and in the one anchor every latency number descends from — both fixable without moving a single survivor.
