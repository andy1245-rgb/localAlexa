# Overnight on dell321: paraphrase families, and the two-stage bucket

2026-11-23. Operator's call: the ceiling is not label redesign, it is **multiple steps — general
buckets, then specific ones** — and the data needs more *phrasings*, not more slot fills. Both run
tonight on `dell321`, CPU only, and both are measured on the same grouped folds as everything else.

## Box facts, verified today

```
ssh dell321  (PowerShell 5.1 default; use wsl -d Ubuntu -- bash -lc "..." for POSIX)
Python 3.11.9 · torch 2.11.0+cpu · transformers 5.6.2 · pip 25.3 · 211.6 GB free
No CUDA. Intel Arc 140V iGPU present and NOT to be used - torch-xpu is a rabbit hole, stay on CPU.
8 logical cores. `&&` does not exist in PowerShell; `2>nul` creates a file named nul, use 2>$null.
```

Never touch the sshfs mount from either side. Everything lives in `C:\Users\andre\alexa-finetune\`
and comes back as small JSON plus a markdown table.

## Job 1 — paraphrase families, not fills

The corpus is 110 families × slot fills. More fills stopped helping at 16 per label. What is missing
is **new sentence shapes**, and the safe way to get them without importing a generator's style:

1. **Deterministic transformations first** — no model, fully reproducible: imperative ⇄ question form,
   add/remove politeness, add/remove a leading context clause ("in the kitchen, …"), reorder
   adverbials, swap synonyms from a small curated lexicon per slot type, expand/contract contractions.
   Each transformation is labelled so a row records which ones produced it.
2. **Diversity filter** — measure lexical overlap (distinct n-gram ratio) between each generated row
   and the existing families; drop anything below the diversity floor. **Report the distribution of
   that overlap**, because "we generated 5,000 rows" is meaningless if they are all the same sentence.
3. **Then the experiment**: retrain the current best recipe with the expanded corpus, same grouped
   folds, and report whether the family-held-out number improves. Adding data that only matches new
   phrasings of known families will show up here and nowhere else — that is the point of the test.
4. An LLM paraphrase pass is **out of scope tonight**. If deterministic transformations show the
   direction is right, that is the follow-up and it can use a proper generator then.

## Job 2 — the two-stage bucket

The operator's diagnosis: confusions are within-family, so the task is hierarchical. Build it:

- **Stage A**: coarse bucket over the 18 labels. Propose the buckets from the data — cluster the
  intents by confusion in the current model's out-of-fold predictions, not by intuition — and report
  the bucket assignment with its evidence.
- **Stage B**: within-bucket classifier, only over that bucket's labels.
- **Compare, same folds**: (i) flat 18-way incumbent, (ii) two-stage, (iii) two-stage where stage A
  emits top-2 buckets and stage B runs on the union. Report accuracy, per-fold spread, and the
  latency of the full two-stage path.
- **The failure mode to watch**: stage A errors are unrepairable, and hierarchical models often lose
  to flat ones when the hierarchy is approximate. If it loses, say so plainly — that is a result and
  it closes the question.

## Rules

- **Measure, do not assert.** Every table row carries the command or script that produced it.
- **Grouped folds, spread reported.** A win smaller than the spread has not won.
- Copy the corpus and scripts to the box; **do not run training against the sshfs mount**, and do not
  copy the repo's `.git`.
- Bring back JSON and markdown only. Commits happen here, on a branch, never pushed.
- If something takes longer than the night, leave a resumable checkpoint and say where.
