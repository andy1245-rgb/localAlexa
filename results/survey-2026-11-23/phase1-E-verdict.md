# Phase 1-E, answered: is SetFit still the best few-shot recipe in 2026?

2026-11-24, branch `survey/base-model-2026-11-23`. Answers the question the brief assigned family E and
phase 1 did not answer (`phase1-2-audit.md` item 7). Everything below is a reading of a source cited
with the date read; nothing is a measurement I made, and where no source exists the gap is stated.

## The verdict

**At 22 examples per class, 18 classes, CPU, a ≤36M body and a ~10 ms budget, SetFit is still the right
recipe — on the narrow ground that it is the only one of the seven that fits, not on the ground that it
is the most accurate.** Two recipes beat SetFit on published accuracy: **FastFit**, by +1.2 to +2.7
points at equal body, and **plain fine-tuning of the encoder with a linear head**, by +2.0 points at
10-shot. Both wins were measured at bodies of 110M and 355M — 4.8× and 15× this budget — and **neither
has ever been run on a body that fits inside 10 ms**. So: nothing has superseded SetFit *at this scale
and budget*; something plausibly does *if the budget moves*.

What has changed since 2022 is not a recipe but maintenance. **SetFit is the only member of family E
still under active development**: `setfit` 1.2.0 shipped 2026-09-04 (https://pypi.org/pypi/setfit/json,
read 2026-11-24), the repo is not archived and was pushed 2026-11-17
(https://api.github.com/repos/huggingface/setfit, read 2026-11-24). FastFit — its only published
accuracy rival — is **archived read-only** (`archived: true`, last push 2026-10-10,
https://api.github.com/repos/IBM/fastfit, read 2026-11-24) with no PyPI release since `fast-fit` 1.2.1
on 2024-05-08 (https://pypi.org/pypi/fast-fit/json, read 2026-11-24).

## The seven, against the four questions

| recipe | CPU inference cost | decoder at inference? | maintained (release read 2026-11-24) | compared to SetFit by someone other than SetFit's authors? |
|---|---|---|---|---|
| **SetFit** — contrastive pairs + logistic head | **7.75 ms p50** on this corpus (`docs/ladder-results-2026-11-10.md`, read 2026-11-24): 1 encoder forward + head | no | **yes** — 1.2.0, 2026-09-04 (https://pypi.org/pypi/setfit/json) | is the baseline in both comparisons below |
| **FastFit** — batch contrastive + token-level similarity, no head | "about 1 millisecond" per input, AUTHOR CLAIM, hardware unstated (https://arxiv.org/abs/2404.12365 §5.3, read 2026-11-24). At equal body its graph is one encoder forward **plus** the token-similarity term of Eq. 2 — SetFit's cost ± a head | no | **no** — archived; last release 2024-05-08 (https://pypi.org/pypi/fast-fit/json; https://api.github.com/repos/IBM/fastfit) | **yes** — the only equal-body head-to-head that exists |
| **Frozen ST + cosine kNN / prototypical** (Snell, https://arxiv.org/abs/1703.05175, read 2026-11-24; implemented as "all-MiniLM-L6-v2 embeddings with cosine kNN", https://arxiv.org/abs/2608.20371 §3.2 — nothing is trained) | 1 encoder forward + cosine over ~200 vectors; not timed separately in the source | no | n/a — no trained artefact to maintain | **yes** — https://arxiv.org/abs/2608.20371 Table 1 (read 2026-11-24): **ST+kNN 86.2 vs fine-tuned RoBERTa 95.9** (ATIS), **75.1 vs 89.1** (CLINC150); the 5-shot 1-NN variant collapses to **28.1** on ATIS |
| **PET** — cloze patterns + full fine-tune | no published latency; needs a 355M forward → ≥99.1 ms by phase 2's estimate (`phase2.md` §3.1) | no | **no** — last push 2023-06-12, no PyPI (https://api.github.com/repos/timoschick/pet) | only by SetFit's authors: "+8.6 and 1.7 points" over GPT-3 and PET, AUTHOR CLAIM (https://arxiv.org/abs/2209.11055 §5, read 2026-11-24) |
| **T-Few** — (IA)³ on T0-3B | ≈795 ms estimated at 2.85B; ≈700 s to train on ≥40 GB GPU, AUTHOR CLAIM (https://arxiv.org/abs/2205.05638; https://arxiv.org/abs/2209.11055 §5) | **yes** (encoder-decoder) | **no** — last push 2023-09-06 (https://api.github.com/repos/r-three/t-few) | only by SetFit's authors, same AUTHOR CLAIM |
| **In-context prompting** of a decoder | **1.624–11.795** for 7 LLMs vs **0.030** for SetFit in the same table, units unstated, ratio given in prose as SetFit "about 56 times" faster (https://arxiv.org/abs/2410.01627 Table 1, §3.4, read 2026-11-24); Claude Haiku 981 ms p50 / 1787 ms p95 (https://arxiv.org/abs/2608.20371 Table 5) | **yes** | n/a — the model, not the recipe | **yes, and it is the strongest challenge** — see below |
| **Plain fine-tune + linear head** | identical to SetFit by construction: same encoder forward, a linear head instead of a logistic one. No separate measurement exists | no | n/a — no separate package | **yes** — the "Classifier" rows of https://arxiv.org/abs/2404.12365 Table 1 |

## What the evidence actually says

**The only equal-body head-to-head is FastFit's Table 1.** FewMany average accuracy, same bodies both
sides (MPNet 110M = "S", RoBERTa-large 355M = "L"), read 2026-11-24 from
https://arxiv.org/html/2404.12365:

| | FastFit S | SetFit S | Δ | FastFit L | SetFit L | Δ | Classifier S | Classifier L |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 5-shot | 80.8 | 78.7 | **+2.1** | 83.2 | 81.2 | **+2.0** | 71.3 | 78.7 |
| 10-shot | 84.6 | 83.4 | **+1.2** | 86.5 | 83.8 | **+2.7** | 80.5 | 85.8 |

1. **FastFit's margin shrinks with shots at the small body**: +2.1 → +1.2. We are at 22/class, past both
   columns, and the direction of travel points at parity.
2. **SetFit's win over plain fine-tuning is a small-body effect, which is exactly our regime.** At 110M
   SetFit beats the plain classifier by **+7.4** (5-shot) and **+2.9** (10-shot); at 355M that falls to
   **+2.5** and then **reverses to −2.0**. SetFit's contrastive stage is what makes a *small* encoder
   work at few shots — the same thing this repo measures as +4.17 grouped on a 22.7M body.
3. **The cost evidence favours FastFit and is the only training cost the sources support.** FastFit small
   (ST) averaged **1 m 16 s** across the eight FewMany sets at 5-shot vs SetFit small (ST) **12 m 44 s**,
   and **2 m 28 s** vs **49 m 38 s** at 10-shot (https://arxiv.org/html/2404.12365 Table 15, computed
   from their table, read 2026-11-24).

**On MASSIVE — a benchmark FastFit's authors did not curate — the same magnitude holds**: FastFit S 68.1
/ L 73.4 vs SetFit S 65.4 at 5-shot, 73.4 / 76.0 vs 72.4 at 10-shot
(https://arxiv.org/html/2404.12365 Table 4). SetFit has no large multilingual sentence-transformer arm
there, and the paper says so; the experiment is still the FastFit authors'.

**The strongest case against SetFit is in-context prompting, and it costs ~56× the latency.** On six
real deployed-TODS intent datasets — the closest published regime to this repo — SetFit averages **0.600
F1** against **0.736** for Claude v3 Haiku, and SetFit plus its own authors' negative augmentation only
reaches **0.658**, still ~8 points behind (https://arxiv.org/abs/2410.01627 Table 1 and §3.4, read
2026-11-24). That is a real defeat by a method needing no training. It is also unrunnable here: no
sub-1B decoder in the survey has a published CPU classification number, and the smallest, `SmolLM2-135M`
at 134.5M, has a ≥37.5 ms floor by phase 2's own arithmetic (`phase2.md` §3.1). **A method that wins by
13.6 F1 points at 56× the latency is not a successor at a 10 ms budget; it is the reason the budget is
the question.**

**Frozen embeddings + kNN is decisively out, from a source with no stake in SetFit.** 86.2 vs 95.9
(ATIS) and 75.1 vs 89.1 (CLINC150) against a plain fine-tuned `roberta-base`, with the 5-shot 1-NN
variant at 28.1 on ATIS (https://arxiv.org/abs/2608.20371 Table 1, read 2026-11-24) — a 9.7 to 14.0
point deficit and a collapse at low shots. That paper tests RoBERTa, not SetFit, so it ranks the
*approach* SetFit's stage one replaces, not SetFit itself.

**PET and T-Few are author-claimed comparisons only.** The +1.7-over-PET and 30 s-vs-700 s figures are
SetFit's authors measuring their own method on their own eleven datasets
(https://arxiv.org/abs/2209.11055 §5, read 2026-11-24). No independent party has re-run either. Both
need a 355M+ or decoder-side body and miss the budget regardless.

## The audit's criticism, addressed

Phase 2 eliminated FastFit on cost — "the small config's body is 109.5M (≥30.5 ms) and the large
config's is 355.4M (≥99.1 ms) … archived read-only" (`phase2.md` §3.6) — and did not check accuracy.
Three findings:

1. **The comparison exists and phase 2 cited the paper without it.** It quotes FastFit's "about 1
   millisecond" speed claim but not Table 1, the same-body 5/10-shot accuracy table. That table is the
   survey's only evidence that anything beats SetFit on accuracy, and it was not named.
2. **The latency reason was the body's, not the recipe's.** ≥99.1 ms is the cost of `roberta-large`;
   ≥30.5 ms is MPNet. FastFit's inference graph is one encoder forward plus a token-similarity term —
   SetFit's order. **No FastFit result exists on any body below 110M**: its smallest is MPNet 110M, and
   every number it reports is 5- or 10-shot, never 22. **FastFit's accuracy at this budget is therefore
   unmeasured, and the elimination stands on cost alone** — which is what phase 2 said, and I add no
   accuracy claim to it.
3. **Phase 2's non-accuracy reasons survive untouched.** The repo is archived (confirmed above), and its
   128-d projection with token-level similarity means the ladder's metrics block does not transfer
   unmodified. Those are integration costs, and they are real.

## What would falsify this

Any one of these:

1. **FastFit on a ≤36M body at 22/class inside 10 ms, beating rung 3's 359/382 grouped by more than the
   fold spread.** Rung 3's gain is +4.17 ± 9.82 grouped, so by the ladder's own rule the bar is a
   unanimous fold direction, not a point estimate (`docs/ladder-results-2026-11-10.md`, read 2026-11-24).
   FastFit's +1.2 to +2.7 over SetFit at equal body sits inside that band. Nobody has run this.
2. **A plain fine-tune + linear head measured here at ≤36M and 22-shot.** It has never been a rung — the
   ladder runs 0 word-svc, 1 union, 2 logreg, 3 SetFit, 4 cross-encoder, 5 ensembles, 6 augmentation, and
   no plain-FT arm. It is the cheapest falsifier available: identical inference cost to SetFit by
   construction, a fraction of its training cost, and it is the only method with a published win over
   SetFit at all (85.8 vs 83.8 at 10-shot, large body).
3. **The cost model itself is wrong.** Phase 2's filter descends from one 6.33 ms anchor and a
   params→latency law. https://arxiv.org/abs/2608.20371 Table 5 (read 2026-11-24) reports `roberta-base`
   at **2.4 ms p50** "local" where that law predicts 34.9 ms — a 14× disagreement on a protocol the paper
   does not fully specify. If a 125M encoder really answers in 2.4 ms on CPU, FastFit's MPNet is inside
   the budget and the answer changes from "SetFit, because nothing else fits" to "unresolved".

## What could not be established

- **No CPU per-utterance latency exists for FastFit, PET, T-Few or any in-context decoder.** FastFit's
  "about 1 millisecond" is an AUTHOR CLAIM on unstated hardware cited beside an LLM's "exceeding 1
  second per input" and nothing else (https://arxiv.org/abs/2404.12365 §5.3). SetFit's own CPU number is
  not in the SetFit paper either; the only 7.75 ms in existence is this repo's ladder measurement.
- **The "~2 ms at 8 examples/class" cell in `phase1-E.md:11` remains untraceable** (audit item 9). I
  checked the 2026 preprint the table cites for it, https://arxiv.org/abs/2608.20371 (read 2026-11-24),
  and **it never mentions SetFit** — its systems are TF-IDF+LR, ST+kNN, fine-tuned RoBERTa and Claude
  Haiku. The *8 examples* anchor is real (https://arxiv.org/abs/2209.11055 §1: "with only 8 labeled
  examples in the Customer Reviews … dataset, SetFit is competitive with fine-tuning on the full training
  set"); the *2 ms* is in no source I read. Delete it rather than re-cite it.
- **No independent replication of any SetFit comparison exists in the sources read.** Every head-to-head
  above is SetFit's authors against rivals (AUTHOR CLAIM), FastFit's authors against SetFit (author claim
  on their own benchmark, plus MASSIVE), or a third party that omits SetFit. The one genuinely
  third-party evaluation including SetFit (https://arxiv.org/abs/2410.01627) runs it on MPNet against
  cloud LLMs. **Nobody has published a neutral multi-recipe few-shot benchmark at ~22 examples per
  class.** Closing that is one experiment, not a literature search — and the two cheapest arms are the
  two named in the falsifier above.
