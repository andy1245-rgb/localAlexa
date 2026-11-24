# Adversarial review — `docs/base-model-survey-2026-11-23.md`

2026-11-24. Reviewer did not write the document. Scope: the document under review (544 lines, cited by line
number as it stands), the brief, and the ground-truth artefacts: `phase4-table.{md,json}`,
`phase3-table.{md,json}`, the seven `probe-*.json`, the four `sweeps/*.json` plus `sweeps-prev/*.json` and
`verify/minilm-5fold-rerun.json`, `results/ladder/rung5-ensemble.json`, `docs/ladder-results-2026-11-10.md`,
`phase1-{A..E}.md`, `phase1-E-verdict.md`, `phase2.md`, `phase1-2-audit.md`, `phase3-env.md`,
`experiments/survey/README.md`, `experiments/survey/report_phase4.py`. Nothing was edited except this file.

---

## 0. Verdict fidelity — the artefact and the prose, side by side

Artefact (`phase4-table.json` → `verdict`):

> `"form": "ranking_does_not_transfer"`, `"label": "the ranking does not transfer"`, `"text": "the ranking does
> not transfer — the selection half picks sentence-transformers/all-MiniLM-L6-v2 (0.9391) and the validation
> half points at Snowflake/snowflake-arctic-embed-xs (0.9868). Recommend the incumbent for stability; do not
> crown a model on the rows that would have selected it."`

Document, §4.2 (line 377):

> "**The selection half picks `all-MiniLM-L6-v2` at 0.9391, tied with granite at 0.9391 (mean-of-folds 0.9394
> each). The validation half puts `snowflake-arctic-embed-xs` top at 0.9868 — +4.6 points over the incumbent's
> 0.9408 there, derived as 0.9868 − 0.9408 — after that model finished LAST on the selection half at 0.8435.**"

The §4.2 body is faithful and *better than the artefact* on one point (it states the granite tie, which the
artefact's own verdict text omits). The abstract and the title are not. Findings 1–4 below are the places the
prose is stronger, weaker, or differently worded than the artefact in a way that changes the conclusion.

---

## 1. BLOCKING

### B1 — Title states the verdict form that did not fire · line 1

**Text:** `# Base-model survey: the incumbent stands, and not because it won`

**Wrong:** the artefact's form is `ranking_does_not_transfer`; the form the title names
(`incumbent_stands` — "nothing beats the incumbent; the incumbent stands") is the *falsifier's* result, not the
result. The document says so itself in two places: line 404–405, "move arctic-xs's fold 3 accuracy from 0.9868
to 0.8947 … and the verdict becomes 'nothing beats the incumbent; the incumbent stands'", and line 480, "The
sweep produced form 2". A reader who reads only the title and the abstract concludes the sweep established that
nothing beat the incumbent; the artefact says the two halves disagree and that is why nothing is crowned.

**Fix:** retitle to the fired form, e.g. `# Base-model survey: the ranking does not transfer — keep the
incumbent for stability`.

### B2 — Arctic did not finish last on the selection half · lines 11, 379, 480–482

**Text (line 11):** "the other half ranked first the model that finished *last* among the selectors
(arctic-embed-xs, 0.9868 …)"
**Text (line 379):** "after that model finished LAST on the selection half at 0.8435"
**Text (line 482):** "from a model last on the 230 rows that did"
**Text (line 380):** "decided it for the row that was worst on the three folds that did"

**Wrong:** `phase4-table.json` → `ranking_selection` is MiniLM 0.9391, granite 0.9391, **arctic 0.8435**,
**L3-v2 0.8391**. Arctic is third of four; `paraphrase-MiniLM-L3-v2` is last, and it is also the worst. The
claim is repeated four times and it is the abstract's punchline, so it is the sentence most readers will carry
away. It also weakens the real finding: "the validation-half winner was third on the half that selects" is the
same verdict with fewer adjectives and no false number.

**Fix:** "third of four on the selection half (0.8435, behind the tied leaders and ahead of L3-v2 at 0.8391)"
in all four places; delete "worst".

### B3 — "all five … with a declared licence" contradicts the row above it and §3 · lines 238–239

**Text:** "All five are 384-dim, so no phase-4 delta can be a dimension change; all five are Apache-2.0 or MIT
with a **declared licence**"

**Wrong:** §2.3's own row 3 (line 234) says of the same model: "Apache-2.0 stated in card body and paper,
absent from frontmatter", and its disqualifier column reads "its licence is prose, not a file; on the rule that
killed `gte-tiny` it falls". §3 (line 283) puts `snowflake-arctic-embed-xs` "**out on licence**" and line 323
records `license: null`, `license_source: null`, `GET /resolve/main/LICENSE` → 404. `phase3-table.md`'s advisory
warning says "licence unread (no licence field in the model card)". This is exactly the class of error the brief
forbids: a model described as satisfying the licence constraint when the artefact says its licence is unreadable.

**Fix:** "four of five carry a licence file (Apache-2.0 or MIT); `arctic-embed-xs` carries card prose only and is
out on the rule that killed `gte-tiny`."

### B4 — Abstract claims an elimination-free result · line 9

**Text:** "**No model in this survey beat the incumbent, and no model in it was properly ruled out.**"

**Wrong:** both halves fail. "No model beat the incumbent" is contradicted nine lines later by the document's
own line 19 ("the one that beats it on validation fails the licence constraint"), and "no model … was properly
ruled out" is contradicted by §2.4 (62 rows removed at F1, 4 at F2, 5 at F5) and by §3 (line 319: bge-small
"out on latency … 1.88×", line 323: arctic "out on licence"). The only reading under which "no model beat the
incumbent" is true is pooled-across-all-folds — which line 398 says "entered no decision". So the headline uses
the one statistic the document forbids as a summary of the result.

**Fix:** "No model beat the incumbent on the half that selects. The model that beats it on the half that reports
fails the licence constraint, and the two models that would have been cheap upgrades did not transfer."

---

## 2. HIGH

### H1 — The selection-half tie is stated; the tie-break is not · lines 377, 383

**Text (line 377):** "The selection half picks `all-MiniLM-L6-v2` at 0.9391, tied with granite at 0.9391
(mean-of-folds 0.9394 each)"
**Text (line 383):** "It is the only model first or tied-first on the half that selects"

**Wrong:** "picks" is an artefact of a sort order. `experiments/survey/report_phase4.py:197–209` ranks on
`(pooled, mean_fold, incumbent-first, model_id)` — the tie is broken *by being the incumbent*, and the document
never says so. A reader cannot tell whether MiniLM won anything or was named by the script. The verdict is in
fact invariant to the tie-break (granite's validation delta is −1.32 ± 5.58, clears no spread; the validation
top is arctic either way), which is a strengthening the document does not take.

**Fix:** add one clause: "the artefact breaks that tie incumbent-first, then by `model_id`; had granite been
picked instead, the verdict would be unchanged — granite clears no validation spread and the validation top is
arctic either way."

### H2 — §4.2's form-2 condition drops a disjunct the verdict depends on · line 374

**Text:** "| the ranking does not transfer | no such win, and the halves disagree about the top model | **yes** |"

**Wrong:** the artefact's rule (`phase4-table.json` → `rule.verdict_order[1]`, and `phase4-table.md`'s
three-forms table) is "no such win, and the validation half's top model is not the selection half's pick **(or a
model that was not picked clears the spread)**". The document silently omits the second disjunct, and drops
`clearers: []` — the field that shows *why* form 2 fired rather than form 1. A reader cannot reproduce the
decision from the document alone.

**Fix:** quote the condition in full and add "no other model cleared its validation spread (`clearers: []`)".

### H3 — Three numbering schemes, one label · lines 12, 371, 374, 480

**Text (line 12):** "That is the brief's **third** acceptable form — the ranking does not transfer"
**Text (line 371):** "| the brief's three forms | condition | fired |", with "the ranking does not transfer" as
row **two**
**Text (line 480):** "The sweep produced **form 2**"

**Wrong:** the brief orders the forms (1) beats by more than the spread, (2) nothing beats it / incumbent
stands, (3) ranking does not transfer. The reporter orders them (1) win, (2) does not transfer, (3) incumbent
stands. The document uses the brief's numbering in the abstract, the reporter's numbering in a table it labels
"the brief's three forms", and the reporter's numbering again in note 11 — so the same verdict is "third" and
"form 2" in one document.

**Fix:** present the table in the brief's order and add "the reporter tests them in its own order; the form that
fired is the one the brief lists third."

---

## 3. MEDIUM

### M1 — "72 if they count as one model" does not follow · line 185

**Text:** "| entry | 84 rows | — | 74 distinct artefacts: 11 rows re-name something counted elsewhere, +1 for
`Gemma 3 270M` and `270M-IT` inside one row **(72 if they count as one model)** |"

**Wrong:** 84 − 11 = 73 distinct artefacts; adding one for the split Gemma row gives 74; counting the two Gemma
names as one model gives back 73, not 72. The figure is inherited verbatim from `phase1-2-audit.md:168`, which
has the same off-by-one. Every other number in the funnel checks out (84 → 22 → 18 → 13 → 5, and 51 + 11 + 22 =
84 — verified line by line against §2.4's group counts), which makes this one stand out.

**Fix:** "(73 if they count as one model)".

### M2 — "Four of five low" — all five are low · line 461

**Text:** "L3-v2 4.1 → **8.387**, arctic-xs 6.3 → **12.126**, granite 10.0 → **11.696**, bge-small 11.7 →
**28.189**, and the anchor itself 6.33 → **15.021**. **Four of five low**, one by 2.4×"

**Wrong:** every one of the five measured values is above its prediction (8.387>4.1, 12.126>6.3, 11.696>10.0,
28.189>11.7, 15.021>6.33). "Four of five" contradicts the list it introduces and contradicts §3 (line 299–300):
"§2's four predictions were all low … and none high."

**Fix:** "All five low, one by 2.4×".

### M3 — "The family's other five rows" names six · line 116

**Text:** "The family's other five rows are the same artefacts counted elsewhere — `snowflake-arctic-embed-xs`
(A10), `mxbai-embed-xsmall-v1` (A12), `google/embeddinggemma-300m` (A14), `jina-embeddings-v2-small-en` (B15),
and `potion-base-32M` / `potion-retrieval-32M` (32.3M, 512, MIT, static …)"

**Wrong:** that is four re-names plus two potion rows = six. `phase1-D.md` has 16 data rows; §1.4 lists 10 and
the remainder is 6, which is what makes "10 of 16" true.

**Fix:** "The family's other six rows", and say four of them are re-names.

### M4 — The one method-note number with no artefact · line 415

**Text:** "Banner-exchange timeouts, then full loss, then recovery **220 s** later."

**Wrong:** I could not trace 220 s to any artefact. `phase3-env.md` records the ssh invocation patterns but no
flap; `experiments/survey/README.md` documents dropped ssh sessions and Errno 22 correlates (lines 805–820) with
no duration. Every other number in note 1–12 is traceable (278-byte artefact, `failed_stage: harness`, dtype
trap, seven fits — `README.md:460`, `:798`).

**Fix:** cite the log the 220 s came from, or replace with "recovered on retry" and drop the number.

### M5 — The offline fallback was requested, not exercised · lines 436–437, 433

**Text:** "…sticky for the rest of the process, recorded as `load.mode`. **All four phase-4 artefacts carry
`load.mode: offline-fallback` with a single successful cached attempt.**"

**Wrong:** all four sweeps carry `"forced": true, "offline_reason": "--offline"` in the same `load` block
(verified in each of the four `sweeps/*.json`). So the offline path was *selected by flag*, not entered after a
live failure — and the document's own source says so: `README.md:748–750`, "no run here has watched a real
closed `httpx` pool fall back to the cache." The sentence reads as evidence that the failure path was fixed and
proven. Line 433's mechanism ("the closed `httpx` pool is shared, so the first blink fails every load after
it") is likewise an inference, and `README.md:727–733` files the equivalent under inference.

**Fix:** "All four phase-4 artefacts ran `--offline`, so they record `mode: offline-fallback` with
`forced: true`; the live-failure path has not been exercised (README says so)".

### M6 — A Windows mechanism stated as fact · line 421

**Text:** "**2. Windows OpenSSH kills `Start-Process` children when the session's job object dies.** A fit
launched from an ssh session inherits that session's job, so a dropped connection kills the training — which
reads exactly like 'the model hung'."

**Wrong:** the source this note rests on is explicit that the mechanism is not established —
`README.md:751–754`: "the raising call was never identified, because the traceback was discarded and the failure
correlates with a dropped ssh session"; `README.md:807–820` separates "What is proven" from "Inference, not
proven". The document states the job-object account flatly.

**Fix:** prefix "on the evidence available this is inference, not proof:" and cite README's split.

### M7 — Garbled attribution of 8.15 vs 8.387 · line 222

**Text:** "the same pair against its own anchor gives **8.15 ms**, which is what phase 2 quotes for that row
elsewhere **and what the box measured (8.387 ms)**"

**Wrong:** 8.15 ms is an *estimate* (12.478 × 0.653); the box measured 8.387 ms
(`probe-L3-v2.json` → `latency.p50_ms`). As written the clause asserts 8.15 is the measurement and then
contradicts itself in the parenthesis.

**Fix:** "…gives 8.15 ms — the figure phase 2 quotes for that row elsewhere; the box measured 8.387 ms."

### M8 — "Every millisecond in this section is an estimate" · line 212

**Text:** "Every millisecond in this section — the 35.9M ceiling, the 50 printed floors, the four predictions —
is an **estimate** descended from that one 6.33 ms anchor."

**Wrong:** §2.3 (lines 232–233) carries two numbers the document treats as measurements: the 6.33 ms anchor
itself and L3-v2 "measured at 8.152 ms single-sentence". The claim is right about the estimates and wrong about
the scope of "every".

**Fix:** "Every millisecond in this section that came from the parameter law is an estimate…".

### M9 — "Seven recipes" is never seven in §1.5 · line 148

**Text:** "at 22 examples per class, 18 classes, CPU, a ≤36M body and a ~10 ms budget, SetFit is the only one
**of the seven** recipes that fits"

**Wrong:** §1.5 lists eight rows covering six recipes (SetFit ×2, FastFit ×2, ST+NN, PET, T-Few, in-context
prompting). "Seven" is reachable only by counting plain fine-tuning with a linear head, which is never a row in
§1.5 and is never named here. A stranger counting the table gets six.

**Fix:** name the seven explicitly in one clause, or write "six".

### M10 — 8.152 ms is quoted without its protocol · line 233

**Text:** "already on disk, measured at 8.152 ms single-sentence"

**Wrong:** 8.152 ms is the phase-3 workspace smoke run — 96 calls over 32 sentences, `median_ms_per_sentence:
8.1518` in `phase3-env.md:204` — a different protocol from the probe that measures the same model at 8.387 ms
and the incumbent at 15.021 ms. It is traceable, but as printed it reads as a measurement on the phase-3
protocol, and it is the number used to justify the shortlist's cleanest test.

**Fix:** "measured at 8.152 ms in the 96-call smoke run (`phase3-env.md`); the phase-3 probe reads 8.387 ms."

---

## 4. LOW

### L1 — Two roundings of the same figure in one document · lines 350 and 390

**Text (line 350):** "| `granite-embedding-30m-english` | … | **0.9391** / 0.9394 | 0.9276 / **0.9277** |"
**Text (line 390):** "0.9276 v 0.9408, a **−1.32**-point gap"

**Wrong:** both values exist upstream — the sweep's own `halves.validation.mean_fold_accuracy` is 0.9277, and
`report_phase4.py` recomputes 0.9276 from the 4-dp fold accuracies (`phase4-table.md` prints 0.9276). So the
document prints a figure its own later sentence rounds the other way, and a reader checking `phase4-table.md`
will not find 0.9277.

**Fix:** pick one (0.9276 matches the generated table) and say which artefact it comes from.

### L2 — One model, two parameter counts · lines 83 and 249

**Text (line 83):** "| `NovaSearch/stella_en_400M_v5` | **435.2M** fp32 safetensors |"
**Text (line 249):** "`stella_en_400M_v5` **400M** → ≥111 ms"

**Wrong:** the floor is computed from the repo-name count, not the safetensors count the document itself
reports two sections earlier; 6.33 × 435.2/22.7 = 121 ms. In a document whose §1 promises `safetensors.total`
unless flagged `(card)` or `derived`, using 400M unflagged is a provenance slip.

**Fix:** "435.2M → ≥121 ms", or note that the floor uses the repo-name count.

### L3 — Rowwise is a credential in §2.3 and meaningless in §4.2 · lines 232 and 399–401

**Text (line 232):** "the only candidate with measured accuracy on this corpus (359/382 = 0.9398 grouped,
**360/382 rowwise**, `__none__` 0.84)"
**Text (line 399):** "the rowwise evaluation set is generated from the same templates as the training data, so
a rowwise number here measures template coverage, not a classifier"

**Wrong:** the same statistic is offered as evidence for the incumbent in phase 2 and declared uninformative in
phase 4, with no note that §2.3 predates the finding.

**Fix:** drop "360/382 rowwise" from §2.3, or add "(a rowwise number this survey later shows measures template
coverage — §4.2)".

### L4 — Three forward references a stranger cannot resolve · lines 139, 141, 146

**Text:** "(audit item 9)"; "(audit item 8)"; "(audit item 7)"

**Wrong:** `phase1-2-audit.md` is first introduced at line 178 ("The independent audit (`phase1-2-audit.md`,
check 4)"). At lines 139–146 "the audit" has no antecedent, so §1.5 reads as a document that references a
document the reader has not been told about.

**Fix:** at line 139, one clause: "an independent audit of phases 1–2 (`results/survey-2026-11-23/
phase1-2-audit.md`, §5)".

### L5 — ECE never expanded · line 364

**Text:** "0.9424 rowwise, `__none__` 0.84, **ECE 0.53**"

**Fix:** "expected calibration error 0.53" on first use.

### L6 — A causal claim from a correlational comparison · line 153

**Text:** "the contrastive stage **is what makes** a small encoder work, the same effect this repo measures as
+4.17 grouped on a 22.7M body"

**Wrong:** the evidence is an accuracy comparison (SetFit vs plain fine-tuning at 110M and 355M) with no
ablation of the contrastive stage; the repo's +4.17 is SetFit-vs-linear-baseline, not contrastive-vs-not. The
sentence is inherited from `phase1-E-verdict.md:52`, where it is equally unsupported.

**Fix:** "consistent with the contrastive stage being what makes a small encoder work at few shots" — or cite
SetFit's own ablation.

### L7 — "~56× the latency" carries units the source does not have · line 159

**Text:** "**The strongest challenge is in-context prompting, at ~56× the latency**"

**Wrong:** `phase1-E-verdict.md:34` records the source's table as "**1.624–11.795** for 7 LLMs vs **0.030** for
SetFit in the same table, **units unstated**, ratio given in prose as SetFit 'about 56 times' faster". The
document converts a prose ratio with unstated units into a latency multiple.

**Fix:** "at ~56× by the paper's own prose ratio (units unstated)".

### L8 — p50 column has two sources in the artefacts, and the document names neither · line 348

**Wrong:** each sweep carries two latency blocks — `runs[].latency_ms.p50` (granite **7.864**) and
`latency_passes.p50_ms` (granite **6.318**). §4.1 prints 6.32 without saying which pass it is, and §5 note 9
shows the spread is large enough to matter (9.694–15.575 for one encoder). `phase4-table.json` names its source
(`latency_passes.p50_ms (dedicated pass)`); the document does not.

**Fix:** add "(dedicated latency pass)" to the column header.

---

## 5. Check-by-check — what I verified, and what passed

**1. Verdict fidelity — FAILS on packaging, passes on substance.** §4.2's body matches `verdict.form`,
`winner.model_id`, 0.9391/0.9408/0.9868 and the "+4.6 points" derivation exactly, and correctly labels folds 0–2
selection (230 rows) and 3–4 validation (152 rows). Fails at B1 (title), B2 (last/worst), H2 (rule text), H3
(numbering). The document is *stronger* than the artefact in one good way — it states the granite tie the
artefact's verdict sentence omits — and weaker in another: it never says the tie-break is incumbent-first.

**2. Number provenance — 47 numbers traced; 8 defects.** Verified against source files: the funnel (84 → 22 →
18 → 13 → 5; 62 = 51 + 11; 62 + 22 = 84); 9 A / 4 B / 9 D survivors; family B 32 of 36; the 5+6+7 and 6+4+4 and
5 group counts; 35.9M = 22.7×10/6.33 and 29.3M = 22.7×10/7.75; 30.2M = 22.7×(10/6.33)^(1/1.6); the L3-v2 mixed
anchor (0.767^1.6 = 0.653; 6.33×0.653 = 4.14; 12.478×0.653 = 8.15); all twelve "≥N ms" floors in §2.4
(recomputed each at 6.33 × params/22.7); distilbert 1.87× the ceiling; every canary and probe latency, p95,
ms/row, snapshot and RSS in §3 against the probe JSONs; per-pass spreads 0.055–1.406; 24.03/24.92/22.53 and the
3.7% span; 34% parameter gap and 1.92×/3.4× spans; all twenty per-fold accuracies and all fifteen per-fold
deltas in §4; every selection/validation pooled and mean-of-fold figure; the 68/76 = 0.8947 falsifier and 7
rows; 359/382 and 360/382; 0.77×–1.47×; rung 3's 93.98% / +4.17 ± 9.82 / 94.24% / 7.75 ms / 0.84 / ECE 0.530;
rung 5's 95.03% / 7.85; rung 6's +3.15/−3.16; `fold_assignments["5"]` rows_per_fold 77/77/76/76/76; the 278-byte
granite artefact with `failed_stage: "harness"`; the six-session p50 series 6.33/8.222/9.694/11.702/15.021/
15.575; 31.57 GB RAM and 8 cores. Defects: B2, M1, M2, M3, M4, M5 (load.mode), L1, L2. Numbers I could not
trace: **220 s** (M4) and nothing else.

**3. The forbidden verdict — PASSES, with one caveat.** Line 398 states it explicitly: "**Pooled-across-all-folds
accuracy entered no decision**: neither `grouped pooled` nor `rowwise pooled` is in the ranking", with the rung-6
reason and the sentence "The rows that selected a winner cannot also be the rows that score it." Note 11 (lines
478–483) names "the best model scored highest" as inadmissible and refuses the 0.9868 temptation. The caveat is
B4: the abstract's headline is only true pooled, which is the number the body forbids.

**4. Selection/validation integrity — PASSES on use, FAILS on tie-break disclosure.** The document never selects
on folds 3–4 and never confirms on folds 0–2 by name; §4.1's caption says folds 3–4 "selected nothing", §4.2
says "Two folds that selected nothing decided an ordering". The numerator of the finding — arctic is the
validation top and the selection-half third — is correctly reported as the *disagreement*, not as evidence for
arctic. The tie at 0.9391/0.9394 is stated (line 377); the tie-break is not (H1). Recommendation of the
incumbent rests on the selecting half plus the ladder reproduction, which matches the artefact.

**5. Unsupported prose — 4 items.** M6 (job-object mechanism), M5 (httpx pool mechanism + the reading of
`load.mode`), L6 ("the contrastive stage is what makes…"), L7 (56×). Everything else that looked unsupported
traced: "8-core idle" → `phase2.md:43,494`; "four of the five survivors" → `phase1-2-audit.md:280–281` (whose
own internal contradiction between "four" and the two named models the document reports honestly); "seven fits"
→ `README.md:460`; `results/` gitignored → confirmed by the Artefacts note.

**6. Licence and constraint consistency — FAILS at B3, passes on the five-vs-four explanation.** §4 opening says
"Four models: bge-small was dropped in phase 3; arctic-xs was carried with its licence defect recorded", and §3's
"The two drops" states the bge case (1.88× the incumbent) and the arctic case (`license: null`, LICENSE 404,
prose only) correctly. No model is credited with a licence the artefacts don't show — except in the one summary
sentence at line 238.

**7. Phase 3 vs phase 2 — PASSES.** The contradiction is stated where it matters and in both directions: the
ordering failure at §3 item 1 ("`granite` at 30.3M measured 11.696 ms — faster than `arctic-embed-xs` at 22.6M …
a model with 34% more parameters answering sooner"), the magnitude failure in the same paragraph ("§2's four
predictions were all low … and none high"), the dead absolute ceiling at item 2 ("the incumbent itself measures
15.021 ms here"), and the estimate/measurement distinction is repeated in §2.2(b), §2.2(c) and §5 note 8. The
estimate is not kept as a measurement anywhere except the ambiguity at M7 and the scope error at M8.

**8. The brief's checklist.** Phase 1 — **present**: nine columns including source and date, 84 rows, vendor
claims tagged, no recommendation (§1.6 is flagged as supplied afterwards). Phase 2 — **present**: five rows with
one sentence each way (lines 232–236) plus explicit eliminations grouped by constraint (§2.4). Phase 3 —
**partial**: load, 382 rows, p50 and p95, snapshot MB and RSS MB are all present; download is only implicit (a
snapshot size, no download line), and the download proof lives in `phase3-env.md`, not here. Phase 4 —
**present**: same recipe (`num_iterations=8, num_epochs=1, batch_size=16, seed=0`, verified in all four sweeps),
same folds (`fold_assignments["5"]`), same metrics, select-on-one-half, and a transfer statement in §4.2.

**9. Internal contradictions — 6 pairs.** B1 (title vs §4.2/note 11), B3 (line 238 vs line 234/§3), B4 (line 9
vs line 19/§2.4/§3), B2 (line 379 "LAST" vs `phase4-table.json` order), M2 (line 461 vs line 299), L1 (0.9277 vs
0.9276), H3 ("third" vs "form 2"), M8 ("every millisecond … estimate" vs §2.3). Abstract against body: three of
these (B1, B2, B4) are abstract-level and all three change what a reader concludes.

**10. Reading as a stranger — FAILS at the first line.** The first paragraph body (lines 9–20) does give the
honest conclusion and its caveat (6.33 ms anchor from another host, 15.02/15.58 ms canaries, the 10 ms budget
already broken), and that is the document's best writing. But the title contradicts it (B1), the first sentence
overclaims (B4), and the punchline is false (B2). Remaining stranger problems, all minor: "audit item N" before
the audit is introduced (L4), ECE unexpanded (L5), "rung 3/5/6" and "the ladder" used before definition (the
ladder doc is cited inline, so acceptable), "seven recipes" that the tables do not show (M9), and a
`## Reproduce` block whose commands assume a `.venv-embed` and a checkout of `experiments/survey/` that a reader
of the committed doc will not have — worth one sentence.

---

## 6. Verdict

**FIX-FIRST** — blocking: **B1** (title states the verdict form that did not fire, contradicting
`phase4-table.json` and the document's own note 11), **B2** (arctic reported as last/worst on the selection half
at lines 11, 379, 380, 482; L3-v2 is last and worst), **B3** ("all five … with a declared licence", contradicted
by §2.3 row 3 and §3's licence drop), **B4** ("no model in it was properly ruled out", contradicted by §2.4 and
§3). Secondary but should land in the same pass: **H1** (state the incumbent-first tie-break), **H2** (restore
the omitted disjunct and `clearers: []`), **H3** (one numbering for the forms), **M2** and **M1** (two counts
that do not add up), **M5** (the offline fallback was forced, not exercised). The verdict itself — "the ranking
does not transfer; recommend the incumbent for stability" — is correctly derived from the artefact and correctly
argued; nothing in this review asks for a different conclusion.
