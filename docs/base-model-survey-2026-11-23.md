# Base-model survey: the ranking does not transfer — keep the incumbent for stability

2026-11-23, branch `survey/base-model-2026-11-23`. Four phases as the brief divided them
(`docs/base-model-survey-brief-2026-11-23.md`): a landscape, a narrow cut, a measurement on the target box, a
fine-tuned sweep. The incumbent is SetFit on `all-MiniLM-L6-v2` — **0.9398 grouped, +4.17 ± 9.82 against the
linear baseline, `__none__` recall 0.84, 7.75 ms** — and the system to beat is rung 5's ensemble, logreg +
SetFit standardised, **0.9503 at 7.85 ms** (`docs/ladder-results-2026-11-10.md`).

**No model beat the incumbent on the half that selects, and the shortlist was not ruled out on evidence —
the 62 rows F1 removed fell to a parameter-derived estimate, not a measurement — but bge-small was then ruled
out by measurement and arctic-embed-xs by licence.** Four encoders survived to the sweep. The half that
selected ranked the incumbent first (0.9391, tied with granite); the other half ranked first a model that came
third of four among the selectors (arctic-embed-xs, 0.9868, +4.6 points over the incumbent there, 9.6 points
behind it on the half that selected). That is the brief's third acceptable form — the ranking does not
transfer — so the recommendation is to keep the incumbent for stability and treat every other row as unproven
at 22 examples per intent. The caveat that makes even this weak: the survey's whole latency constraint descends
from one **6.33 ms** encoder-only measurement taken on a *different* host than the target box, whose own two
canaries read **15.02 ms and 15.58 ms** for the same encoder. The brief's ~10 ms budget is already broken by
the incumbent itself, and the parameter ceiling that removed 62 of 84 phase-1 rows was calibrated against a
number from another machine. Of the two models that tied the incumbent on the selection half, one then lost
on validation; the one that beats it on validation fails the licence constraint. Nothing here says
MiniLM-L6 is the best base model. It says nothing else was proved better.

---

## 1. Phase 1 — the landscape

84 rows over five families, **every row read 2026-11-23** (the `read` column repeats it because the brief
required it per row). `VENDOR CLAIM` marks a number whose only source is the model's publisher, `AUTHOR
CLAIM` the same inside a paper, `not published` that no number was found. URLs are shown without their
`https://` prefix. `params` is Hugging Face's `safetensors.total` unless the cell says `(card)` or `derived`;
`contrastive` means "documented, or a plain `Transformer`+`Pooling` chain SetFit can accept" — format, never
whether fine-tuning helps at 22 examples.

### 1.1 Family A — sentence encoders for SetFit (14 rows, all listed)

| name | params | dim | max_seq | licence | contrastive | published speed | source | read |
|---|---|---:|---:|---|---|---|---|---|
| `sentence-transformers/all-MiniLM-L6-v2` (**incumbent**) | 22.7M | 384 | 256 | Apache-2.0 | yes | VENDOR CLAIM: "5 times faster" than all-mpnet-base-v2; no absolute ms | huggingface.co/sentence-transformers/all-MiniLM-L6-v2 ; sbert.net/docs/sentence_transformer/pretrained_models.html | 2026-11-23 |
| `sentence-transformers/all-MiniLM-L12-v2` | 33.4M | 384 | 128 (config; card 256) | Apache-2.0 | yes | not published | huggingface.co/sentence-transformers/all-MiniLM-L12-v2 | 2026-11-23 |
| `sentence-transformers/all-mpnet-base-v2` | 109.5M | 768 | 384 | Apache-2.0 | yes | not published (it is the baseline in the "5× faster" claim) | huggingface.co/sentence-transformers/all-mpnet-base-v2 | 2026-11-23 |
| `sentence-transformers/paraphrase-MiniLM-L6-v2` | 22.7M | 384 | 128 | Apache-2.0 | yes (card states no objective) | not published | huggingface.co/sentence-transformers/paraphrase-MiniLM-L6-v2 ; huggingface.co/docs/setfit/en/index | 2026-11-23 |
| `sentence-transformers/paraphrase-mpnet-base-v2` (SetFit's default body) | 109.5M | 768 | 512 | Apache-2.0 | yes (card states no objective) | not published | huggingface.co/sentence-transformers/paraphrase-mpnet-base-v2 | 2026-11-23 |
| `sentence-transformers/all-distilroberta-v1` | 82.1M | 768 | 512 (config; card 128) | Apache-2.0 | yes | not published | huggingface.co/sentence-transformers/all-distilroberta-v1 | 2026-11-23 |
| `sentence-transformers/multi-qa-MiniLM-L6-cos-v1` | 22.7M | 384 | 512 | **none declared** (no LICENSE, no tag) | yes | VENDOR CLAIM: 18,000 GPU / 750 CPU queries per sec (hardware, batch unstated) | huggingface.co/sentence-transformers/multi-qa-MiniLM-L6-cos-v1 ; sbert.net/docs/… | 2026-11-23 |
| `sentence-transformers/msmarco-MiniLM-L6-cos-v5` | 22.7M | 384 | 384 | **none declared** | yes | VENDOR CLAIM: 18,000 GPU / 750 CPU q/s | huggingface.co/sentence-transformers/msmarco-MiniLM-L6-cos-v5 ; sbert.net/docs/… | 2026-11-23 |
| `sentence-transformers/msmarco-MiniLM-L12-cos-v5` | 33.4M | 384 | 384 | **none declared** | yes | VENDOR CLAIM: 11,000 GPU / 400 CPU q/s | huggingface.co/sentence-transformers/msmarco-MiniLM-L12-cos-v5 ; sbert.net/docs/… | 2026-11-23 |
| `Snowflake/snowflake-arctic-embed-xs` | 22.6M | 384 | 512 | not in frontmatter, no `LICENSE` (404); card body says Apache-2 | yes (infoNCE) | not published | huggingface.co/Snowflake/snowflake-arctic-embed-xs ; arxiv.org/abs/2405.05374 | 2026-11-23 |
| `Snowflake/snowflake-arctic-embed-s` | 33.2M | 384 | 512 | Apache-2.0 | yes (infoNCE) | not published | huggingface.co/Snowflake/snowflake-arctic-embed-s ; arxiv.org/abs/2405.05374 | 2026-11-23 |
| `mixedbread-ai/mxbai-embed-xsmall-v1` | 24.1M | 384 (MRL) | 4096 | Apache-2.0 | yes (AnglE, built on MiniLM-L6) | VENDOR CLAIM: binary quantisation "32× efficiency" at 93.9% of performance | huggingface.co/mixedbread-ai/mxbai-embed-xsmall-v1 | 2026-11-23 |
| `nomic-ai/modernbert-embed-base` | 149M | 768 (MRL 256) | 8192 | Apache-2.0 | yes | not published | huggingface.co/nomic-ai/modernbert-embed-base | 2026-11-23 |
| `google/embeddinggemma-300m` | 302.9M (card 300M) | 768 (MRL) | 2048 | Gemma Terms of Use, gated | not stated | not published | huggingface.co/google/embeddinggemma-300m ; arxiv.org/abs/2509.20354 | 2026-11-23 |

### 1.2 Family B — modern embedding families (21 of 36 rows; the other 15 are the same families larger or non-permissive)

Omitted: `bge-multilingual-gemma2`, `e5-large-v2`, `multilingual-e5-base/large`, `gte-base`, `gte-large`,
`gte-large-en-v1.5`, `jina-embeddings-v2-base-en`, `jina-embeddings-v5-text-small`,
`jina-embeddings-v5-omni-small`, `nomic-embed-text-v1`, `modernbert-embed-base` (same repo as A13),
`stella-base-en-v2`, `stella_en_1.5B_v5`, `stella-large-zh-v3-1792d`. All 84 rows, including these, keep a
disposition in §2.4. **35 of this family's 36 rows publish no latency, throughput or per-utterance figure of
any kind**; the one exception is marked below.

| name | params | dim | max_seq | licence | contrastive | published speed | source | read |
|---|---|---:|---:|---|---|---|---|---|
| `BAAI/bge-small-en-v1.5` | 33.4M | 384 | 512 | MIT | yes | not published | huggingface.co/BAAI/bge-small-en-v1.5 | 2026-11-23 |
| `BAAI/bge-base-en-v1.5` | 109.5M | 768 | 512 | MIT | yes | not published | huggingface.co/BAAI/bge-base-en-v1.5 | 2026-11-23 |
| `BAAI/bge-large-en-v1.5` | 335.1M | 1024 | 512 | MIT | yes | not published | huggingface.co/BAAI/bge-large-en-v1.5 | 2026-11-23 |
| `BAAI/bge-m3` | 568M (derived: 2271 MB fp32 ÷ 4) | 1024 | 8192 | MIT | yes | not published | huggingface.co/BAAI/bge-m3 | 2026-11-23 |
| `BAAI/bge-en-icl` | 7.11B | 4096 | 32768 | Apache-2.0 | yes (Mistral-7B as embedder) | not published | huggingface.co/BAAI/bge-en-icl | 2026-11-23 |
| `intfloat/e5-small-v2` | 33.4M | 384 | 512 | MIT | yes (`query:`/`passage:` prefixes required) | not published | huggingface.co/intfloat/e5-small-v2 | 2026-11-23 |
| `intfloat/e5-base-v2` | 109.5M | 768 | 512 | MIT | yes (prefixes) | not published | huggingface.co/intfloat/e5-base-v2 | 2026-11-23 |
| `intfloat/multilingual-e5-small` | 117.7M | 384 | 512 | MIT | yes | not published | huggingface.co/intfloat/multilingual-e5-small | 2026-11-23 |
| `intfloat/e5-mistral-7b-instruct` | 7.11B | 4096 | 32768 (card uses 4096) | MIT | yes (last-token pooling) | not published | huggingface.co/intfloat/e5-mistral-7b-instruct | 2026-11-23 |
| `thenlper/gte-small` | 33.4M | 384 | 512 | MIT | yes | not published | huggingface.co/thenlper/gte-small | 2026-11-23 |
| `Alibaba-NLP/gte-base-en-v1.5` | 137M (card; 136.8M safetensors) | 768 | 8192 | Apache-2.0 | yes (CLS) | not published | huggingface.co/Alibaba-NLP/gte-base-en-v1.5 | 2026-11-23 |
| `Alibaba-NLP/gte-multilingual-base` | 305M | 768 | 8192 | Apache-2.0 | yes | VENDOR CLAIM: "a 10x increase in inference speed" vs a 1.5B decoder, relative only | huggingface.co/Alibaba-NLP/gte-multilingual-base | 2026-11-23 |
| `Alibaba-NLP/gte-modernbert-base` | 149M | 768 | 8192 | Apache-2.0 | yes | not published | huggingface.co/Alibaba-NLP/gte-modernbert-base | 2026-11-23 |
| `Alibaba-NLP/gte-Qwen2-1.5B-instruct` | 1.78B | 1536 | 32000 | Apache-2.0 | yes | not published | huggingface.co/Alibaba-NLP/gte-Qwen2-1.5B-instruct | 2026-11-23 |
| `jinaai/jina-embeddings-v2-small-en` | 32.7M | **512** | 8192 (ALiBi) | Apache-2.0 | yes (`trust_remote_code`) | not published | huggingface.co/jinaai/jina-embeddings-v2-small-en | 2026-11-23 |
| `jinaai/jina-embeddings-v3` | 572.3M | 1024 (MRL) | 8192 | **CC-BY-NC-4.0** | yes (5 task LoRAs) | not published | huggingface.co/jinaai/jina-embeddings-v3 | 2026-11-23 |
| `jinaai/jina-embeddings-v4` | 3.75B | 2048 (MRL) | 32768 | **Qwen Research License** (from `LICENSE`) | yes | not published | huggingface.co/jinaai/jina-embeddings-v4 | 2026-11-23 |
| `jinaai/jina-embeddings-v5-text-nano` | 239M (card); 211.8M bf16 | 768 (MRL) | 8192 | **CC-BY-NC-4.0** | yes (EuroBERT-210m, LoRA) | not published | huggingface.co/jinaai/jina-embeddings-v5-text-nano | 2026-11-23 |
| `nomic-ai/nomic-embed-text-v1.5` | 136.7M | 768 (MRL) | 8192 (config 2048 + RoPE) | Apache-2.0 | yes | not published | huggingface.co/nomic-ai/nomic-embed-text-v1.5 | 2026-11-23 |
| `nomic-ai/nomic-embed-text-v2-moe` | 475M total / 305M active | 768 (MRL) | 512 | Apache-2.0 | yes (needs `megablocks`) | not published | huggingface.co/nomic-ai/nomic-embed-text-v2-moe | 2026-11-23 |
| `NovaSearch/stella_en_400M_v5` | 435.2M fp32 safetensors | 1024 (MRL) | 512 | MIT | yes (prompt required) | not published | huggingface.co/NovaSearch/stella_en_400M_v5 | 2026-11-23 |

### 1.3 Family C — small decoder LLMs as classifiers (7 of 10 rows)

| name | params | dim | max_seq | licence | contrastive | published speed | source | read |
|---|---|---:|---:|---|---|---|---|---|
| `Qwen/Qwen3-0.6B` | 751.6M total / 0.44B non-embedding | 1024 | 32768 | Apache-2.0 | none documented | not published | huggingface.co/Qwen/Qwen3-0.6B ; …/raw/main/config.json | 2026-11-23 |
| `Qwen/Qwen3-0.6B-Base` | 596.0M | 1024 | 32768 | Apache-2.0 | indirectly (base of Qwen3-Embedding-0.6B, InfoNCE) | not published | huggingface.co/Qwen/Qwen3-0.6B-Base ; github.com/QwenLM/Qwen3-Embedding/…/technical_report.pdf | 2026-11-23 |
| `Qwen/Qwen3-Embedding-0.6B` | 595.8M | 1024 (MRL to 32) | 32768 | Apache-2.0 | yes, first-party | not published | huggingface.co/Qwen/Qwen3-Embedding-0.6B ; …technical_report.pdf | 2026-11-23 |
| `Qwen/Qwen3-1.7B` | 2.03B total / 1.4B non-embedding | 2048 | 32768 | Apache-2.0 | none documented | not published | huggingface.co/Qwen/Qwen3-1.7B ; …/config.json | 2026-11-23 |
| `HuggingFaceTB/SmolLM2-135M` | 134.5M | 576 | 8192 | Apache-2.0 | none documented | not published (card says "on-device", no figure) | huggingface.co/HuggingFaceTB/SmolLM2-135M ; …/config.json | 2026-11-23 |
| `HuggingFaceTB/SmolLM3-3B` | 3.08B | 2048 | 65536 | Apache-2.0 | none documented | not published | huggingface.co/HuggingFaceTB/SmolLM3-3B ; …/config.json | 2026-11-23 |
| `google/gemma-3-270m` / `-it` | 268.1M (170M embeddings + 100M blocks) | 640 | 32768 | **Gemma Terms of Use**, gated | none documented | VENDOR CLAIM: INT4 QAT "0.75% of the battery for 25 conversations" on a Pixel 9 Pro; nothing on CPU | ai.google.dev/gemma/docs/core/model_card_3 ; developers.googleblog.com/en/introducing-gemma-3-270m/ | 2026-11-23 |

Omitted: `Qwen3-1.7B-Base`, `SmolLM2-360M`, `SmolLM2-1.7B`. **No source in this family publishes a CPU
latency, a per-utterance cost, or a CPU fine-tuning cost for any of its ten entries** — the question the brief
assigned it.

### 1.4 Family D — ONNX / quantised / distilled / CPU-first (10 of 16 rows)

| name | params | dim | max_seq | licence | contrastive | published speed | source | read |
|---|---|---:|---:|---|---|---|---|---|
| `Xenova/all-MiniLM-L6-v2` (fp32 90.4 / int8 23.0 / uint8 22.8 MB) | 22.7M (base) | 384 | 256 | Apache-2.0 | **no** — ONNX weights only | not published | huggingface.co/Xenova/all-MiniLM-L6-v2 | 2026-11-23 |
| `Xenova/bge-small-en-v1.5` (fp32 133.1 / int8 33.8 / q4f16 36.2 MB) | 33.4M | 384 | 512 | MIT | **no** — ONNX weights only | not published | huggingface.co/Xenova/bge-small-en-v1.5 | 2026-11-23 |
| `sentence-transformers/paraphrase-MiniLM-L3-v2` (int8 ONNX 17.5 MB vs 69.0 fp32) | 17.4M | 384 | 128 | Apache-2.0 | yes | not published | huggingface.co/sentence-transformers/paraphrase-MiniLM-L3-v2 | 2026-11-23 |
| `TaylorAI/bge-micro-v2` | 17.4M | 384 | 512 | MIT | yes | not published | huggingface.co/TaylorAI/bge-micro-v2 | 2026-11-23 |
| `TaylorAI/gte-tiny` (int8 ONNX 22.9 MB) | 22.7M | 384 | 512 | **not stated** — no tag, no LICENSE | yes | not published | huggingface.co/TaylorAI/gte-tiny | 2026-11-23 |
| `ibm-granite/granite-embedding-30m-english` | 30.3M | 384 | 512 | Apache-2.0 | yes | VENDOR CLAIM: "twice as fast as other models with similar embedding dimensions" (no ms, no baseline named) | huggingface.co/ibm-granite/granite-embedding-30m-english | 2026-11-23 |
| `minishlab/potion-base-8M` (Model2Vec static) | 7.6M | 256 | none (bag of tokens) | MIT | **no** — static, no encoder | VENDOR CLAIM: "up to 500 times faster on CPU" | huggingface.co/minishlab/potion-base-8M ; github.com/MinishLab/model2vec | 2026-11-23 |
| `sentence-transformers/static-retrieval-mrl-en-v1` | 0 active params (31.25M-entry table) | 1024 (MRL 256) | inf | Apache-2.0 | **no** — `StaticEmbedding` only | VENDOR CLAIM: "397x faster on CPU" than all-mpnet-base-v2, i7-13700K | huggingface.co/sentence-transformers/static-retrieval-mrl-en-v1 | 2026-11-23 |
| `huawei-noah/TinyBERT_General_4L_312D` | ≈15.7M (derived: 62.7 MB fp32 ÷ 4) | 312 | 512 | Apache-2.0 | **no** — raw BERT, no head | AUTHOR CLAIM: "7.5x smaller and 9.4x faster" than BERT-base | huggingface.co/huawei-noah/TinyBERT_General_4L_312D ; arxiv.org/abs/1909.10351 | 2026-11-23 |
| `distilbert-base-uncased` | 67.0M | 768 | 512 | Apache-2.0 | **no** — MLM checkpoint | AUTHOR CLAIM: "60% faster" than BERT-base | huggingface.co/distilbert-base-uncased ; arxiv.org/abs/1910.01108 | 2026-11-23 |

The family's other five rows are the same artefacts counted elsewhere — `snowflake-arctic-embed-xs` (A10),
`mxbai-embed-xsmall-v1` (A12), `google/embeddinggemma-300m` (A14), `jina-embeddings-v2-small-en` (B15), and
`potion-base-32M` / `potion-retrieval-32M` (32.3M, 512, MIT, static, same recipe as `potion-base-8M`).

On granite's speed claim: the sentence names no baseline, but the card's own comparison table names
`bge-small-en-v1.5` among "other models with similar embedding dimensions", so a comparator is nameable even
though the claim does not name one. The claim is unverifiable as written, not incomparable.

### 1.5 Family E — recipes, not weights (all 8 rows)

| name | params | dim | max_seq | licence | contrastive | published speed | source | read |
|---|---|---:|---:|---|---|---|---|---|
| SetFit 1.2.0, default body `paraphrase-mpnet-base-v2` | 109.5M | 768 | 256 | Apache-2.0 | yes (Siamese pairs + logistic head) | AUTHOR CLAIM: ≈30 s per split on p3.2xlarge vs ≈700 s for T-Few 3B (paper §5); no CPU figure | huggingface.co/sentence-transformers/paraphrase-mpnet-base-v2 ; arxiv.org/abs/2209.11055 ; pypi.org/project/setfit/ | 2026-11-23 |
| SetFit 1.2.0, incumbent body `all-MiniLM-L6-v2` | 22.7M | 384 | 256 | Apache-2.0 | yes | not published; this repo measures 7.75 ms p50 | huggingface.co/sentence-transformers/all-MiniLM-L6-v2 ; arxiv.org/abs/2209.11055 | 2026-11-23 |
| FastFit, small (MPNet body, 128-d projection, no head) | 110M; body 109.5M | 768 → 128 | 128 | Apache-2.0; **archived read-only 2026-10-19** | yes (batch contrastive over texts + class names) | AUTHOR CLAIM: "about 1 millisecond" per input, hardware unstated (§5.3) | arxiv.org/abs/2404.12365 ; github.com/IBM/fastfit | 2026-11-23 |
| FastFit, large (`roberta-large` body) | 355M; body 355.4M | 1024 → 128 | 128 | Apache-2.0 package (archived); body MIT | yes | same AUTHOR CLAIM | arxiv.org/abs/2404.12365 ; huggingface.co/FacebookAI/roberta-large | 2026-11-23 |
| Frozen ST embeddings + cosine kNN ("ST+NN"), nothing trained | 22.7M | 384 | 256 | Apache-2.0 (body) | **no** — encoder frozen | not published; the paper's table has TF-IDF+LR <0.1 ms, roberta-base 2.4 ms, Claude Haiku 981 ms p50 | arxiv.org/abs/2608.20371 ; huggingface.co/sentence-transformers/all-MiniLM-L6-v2 | 2026-11-23 |
| PET (`roberta-large`, cloze patterns, soft-labelled fine-tune) | 355.4M | 1024 | 256 | Apache-2.0; body MIT | no | AUTHOR CLAIM: ≈30 min per classifier for 250 steps on one GTX 1080 Ti, ≈2 h for the final (appendix D) | arxiv.org/abs/2001.07676 ; github.com/timoschick/pet | 2026-11-23 |
| T-Few ((IA)³ on `T0_3B`) | 2.85B | 2048 | not published | MIT (code); body Apache-2.0 | no — LM-loss adapters | AUTHOR CLAIM: ≈700 s on a ≥40 GB GPU; ≈1 order of magnitude more inference FLOPs than SetFit-mpnet | arxiv.org/abs/2205.05638 ; huggingface.co/bigscience/T0_3B | 2026-11-23 |
| in-context prompting, `flan-t5-xxl`, no gradient training | 11.27B | 4096 | 4K tokens (AUTHOR CLAIM) | Apache-2.0 | no | AUTHOR CLAIM: "throughput exceeding 1 second per input" | arxiv.org/abs/2404.12365 ; huggingface.co/google/flan-t5-xxl | 2026-11-23 |

Two provenance defects carried here rather than dropped: the incumbent's speed cell in `phase1-E.md` cited a
2026 preprint for SetFit "in ~2ms" at 8 examples/class with **no identifier or URL**, and the preprint it must
have meant (`arxiv.org/abs/2608.20371`) never mentions SetFit — so it is not re-cited (audit item 9). And the
three author-supplied training figures above were untagged in the source table; they carry `AUTHOR CLAIM`
here (audit item 8).

### 1.6 Family E's verdict: is SetFit still the right recipe in 2026?

The brief asked and phase 1 E did not answer — it returned eight rows and no sentence about which to use
(audit item 7). The answer is in `results/survey-2026-11-23/phase1-E-verdict.md`, and it is **yes, on narrow
grounds**: at 22 examples per class, 18 classes, CPU, a ≤36M body and a ~10 ms budget, SetFit is the only one
of the seven recipes that fits.

- **FastFit beats SetFit by 1.2–2.7 points at equal body on the only equal-body comparison that exists**:
  +2.1 (5-shot) and +1.2 (10-shot) at the 110M body, +2.0 and +2.7 at 355M (FewMany averages, FastFit's own
  Table 1, `arxiv.org/html/2404.12365`). SetFit beats *plain fine-tuning* by +7.4 and +2.9 at 110M and that
  reverses at 355M (−2.0) — the contrastive stage is what makes a small encoder work, the same effect this
  repo measures as +4.17 grouped on a 22.7M body.
- **FastFit is eliminated on cost alone.** Smallest body MPNet 110M, largest RoBERTa-large 355M, every number
  5- or 10-shot: it has **never run below 110M parameters or above 10-shot**, so its accuracy at 22-shot is
  unmeasured. The repo is archived read-only (2026-10-19, last release `fast-fit` 1.2.1 on 2024-05-08) while
  `setfit` 1.2.0 shipped 2026-09-04 and the repo was pushed 2026-11-17.
- **The strongest challenge is in-context prompting, at ~56× the latency**: 0.736 F1 for Claude v3 Haiku
  against SetFit's 0.600 on six deployed TODS intent datasets (`arxiv.org/abs/2410.01627`) — unrunnable here,
  since the smallest decoder in the survey (`SmolLM2-135M`, 134.5M) has a ≥37.5 ms floor. **Frozen embeddings
  + kNN is decisively out** on a source with no stake in SetFit: 86.2 v 95.9 (ATIS) and 75.1 v 89.1
  (CLINC150) against a plainly fine-tuned `roberta-base`, 5-shot 1-NN at 28.1 (`2608.20371` Table 1). PET and
  T-Few are compared only by SetFit's own authors.

What would falsify it: a FastFit run at ≤36M and 22/class inside 10 ms beating rung 3 by more than its fold
spread (+4.17 ± 9.82 means a unanimous fold direction, not a point estimate), a plain fine-tune + linear head
measured here (never a rung), or evidence the cost model is wrong — `2608.20371` reports `roberta-base` at 2.4
ms p50 where it predicts 34.9 ms. §1 describes only. §2 eliminates.

---

## 2. Phase 2 — narrowed to five

### 2.1 The funnel, with the audit's corrected arithmetic

Phase 2 as written printed 85 rows walking 74 → 24 → 20 → 20 → 20 → 13 → 5. The independent audit
(`phase1-2-audit.md`, check 4) counted the five phase-1 tables by script: **84** rows, a middle inflated by
two throughout (the two ONNX re-exports were subtracted from a population they were never added to), and one
mis-attributed kill. The numbers below are the audit's repair, not phase 2's. The endpoints do not move and no
survivor changes.

| stage | in | out | what it removed |
|---|---:|---:|---|
| entry | 84 rows | — | 74 distinct artefacts: 11 rows re-name something counted elsewhere, +1 for `Gemma 3 270M` and `270M-IT` inside one row (73 if they count as one model) |
| F1 parameter ceiling | 84 | **22** | 62 rows. The 22 are distinct **trainable** candidates: 9 from A, 4 from B, 9 from D |
| F2 permissive licence | 22 | 18 | 4 |
| F3 CPU-only | 18 | 18 | 0 — F1 had already removed everything whose only speed figure is a GPU one |
| F4 parameter/embedding budget | 18 | 18 | 0 — F1's ceiling is stricter than any capacity budget defensible at 200 rows |
| F5 loadable trainable encoder | 18 | **13** | 5 distinct models |
| shortlisting (a selection cut, not a constraint) | 13 | **5** | 8 that passed every constraint and lost a coverage slot |

Two prose counts had to be corrected and are stated correctly from here: family B is **32** of 36 rows
removed at F1 (not 31), and family E's F1 casualties are **three distinct bodies reached by four rows**
(`roberta-large` twice, `T0_3B`, `flan-t5-xxl`), not "3 of the 8 rows". `stella-base-en-v2` is counted in F1
but reasoned as an F5 kill; the count survives and it is listed under F5 in §2.4.

### 2.2 Three things the audit established that change how this section must be read

**(a) The ceiling descends from one anchor that is not the brief's incumbent number.** F1 converts ~10 ms
into a parameter ceiling using a measured p50, and the number it used — **6.33 ms** — is an *encoder-only*
embed of 382 rows on `andrew-Lenovo-7-14IAL7` (12 logical CPUs, torch 2.15.0+cpu, python 3.12.14, loadavg
8.29→8.70), not the SetFit rung the brief names at **7.75 ms**. So the ceiling is `22.7M × 10/6.33` =
**35.9M** or, on the brief's own anchor, `22.7M × 10/7.75` = **29.3M**. The audit says "four of the five
survivors" sit above 29.3M while naming two; on the shortlist's own counts I can reproduce **two** (granite
30.3M, bge-small 33.4M), plus five of the 33.4M-band alternates §2.4 cuts at shortlisting. Phase 2 also
described that artefact as an "8-core, idle" box; its own file says 12 CPUs and loadavg 8.3–8.7.

**(b) `published speed` contributed nothing to the filter it was collected for.** **0 of 84 phase-1 rows
publish a CPU per-utterance latency in milliseconds.** The nearest things are two CPU throughput claims with
hardware and batch unstated (750 and 400 queries/sec), two relative claims (397×, 500×) and one untraceable
"~2 ms". Every millisecond in this section — the 35.9M ceiling, the 50 printed floors, the four predictions —
is an **estimate descended from that one 6.33 ms anchor**.

**(c) Two extrapolation laws were used in the same section and they contradict each other.** F1 admits on
`6.33 × (params ÷ 22.7M)`, linear, the most charitable law — 35.9M. Phase 2's own §4 then predicts on
`6.33 × (params ÷ 22.7M)^1.6`, the exponent fitted to the box's own L3-v2/MiniLM smoke pair (param ratio
0.767 → latency ratio 0.653), under which the admission ceiling is **30.2M** and granite (30.3M) is out as
well. A filter applied
permissively to admission and strictly to prediction is not one filter. Worse, the L3-v2 prediction of 4.1 ms
applied a ratio measured against a **12.478 ms** anchor to a **6.33 ms** one; the same pair against its own
anchor gives **8.15 ms**, which is what phase 2 quotes for that row elsewhere and what the box measured
(8.387 ms).

The consequence was predictable and is why §3 is a measurement rather than a second estimate: the shortlist
was predicted to lose one of five, and the box lost one *and* invalidated the absolute rule.

### 2.3 The five survivors

| # | model | params | dim | max_seq | licence | why it survives | what could disqualify it |
|---|---|---:|---:|---|---|---|---|
| 1 | `all-MiniLM-L6-v2` (reference) | 22.7M | 384 | 256 | Apache-2.0 | the only candidate with measured accuracy on this corpus (359/382 = 0.9398 grouped, 360/382 rowwise, `__none__` 0.84) and a measured latency on this protocol | nothing disqualifies it as the reference; the failure mode is that its probe does not reproduce the anchor, which indicts the harness |
| 2 | `paraphrase-MiniLM-L3-v2` | 17.4M | 384 | 128 | Apache-2.0 | the cleanest capacity test: same dim, tokenizer and lineage, three layers instead of six, so a delta is attributable to depth alone; already on disk, measured at 8.152 ms single-sentence | if SetFit's gain depends on depth, a 3-layer body buys nothing for a real cost |
| 3 | `snowflake-arctic-embed-xs` | 22.6M | 384 | 512 | Apache-2.0 stated in card body and paper, absent from frontmatter | the only candidate at the incumbent's parameter count with twice the usable `max_seq` **and** a documented infoNCE objective — the swap least likely to break the recipe | its licence is prose, not a file; on the rule that killed `gte-tiny` it falls |
| 4 | `bge-small-en-v1.5` | 33.4M | 384 | 512 | MIT | the highest-upside row the permissive ceiling admitted, with fp32/int8/q4f16 ONNX exports already published, so a win transfers to a Pi without re-derivation | predicted 11.7 ms against a 10 ms budget, 9.3 ms on the optimistic floor — the row most likely to die before it trains |
| 5 | `granite-embedding-30m-english` | 30.3M | 384 | 512 | Apache-2.0 | the only survivor carrying a published speed claim to falsify, at a 6-layer shape with a **RoBERTa tokenizer** — a different tokenisation of 3–6-word imperatives, the one untested axis left at this size | predicted 10.0 ms (fitted) / 8.5 ms (floor), i.e. exactly on the ceiling |

All five are 384-dim, so no phase-4 delta can be a dimension change; four of five declare a licence in the
card frontmatter (Apache-2.0 or MIT), while `snowflake-arctic-embed-xs` declares none there — card prose only —
and is out on the rule that killed `TaylorAI/gte-tiny`; the parameter range is 0.77×–1.47× the incumbent.
**No accuracy number exists for survivors 2–5, and none could exist before phase 4.**

### 2.4 Eliminations, grouped by the constraint that killed them

| constraint | models | the number that killed them |
|---|---:|---|
| **F1 — parameter ceiling** (`6.33 ms × params/22.7M` against ~10 ms; 35.9M linear, 29.3M on the ladder's 7.75 ms) | **62 rows** | every 768- and 1024-dim encoder, every decoder LLM, every body above 36M |
| · decoder LLMs, all 10 of family C | 10 | smallest `SmolLM2-135M` 134.5M → ≥37.5 ms (5.9× the incumbent); largest `SmolLM3-3B` 3.08B → ≥859 ms. 8 of the 10 have no documented contrastive recipe either |
| · BGE above small · E5 · GTE | 5 + 6 + 7 | `bge-base` 109.5M → ≥30.5 ms, `bge-large` 335.1M → ≥93.4 ms, `bge-en-icl` 7.11B → ≥1.98 s · `e5-base-v2` 109.5M → ≥30.5 ms, `multilingual-e5-small` 117.7M → ≥32.8 ms ("small" is 5× the incumbent) · `gte-base` 109.5M → ≥30.5 ms, `gte-large-en-v1.5` 434M → ≥121 ms |
| · Jina · Nomic · Stella | 6 + 4 + 4 | `jina-v2-base-en` 137.4M → ≥38.3 ms, `jina-v3` 572.3M → ≥160 ms (also CC-BY-NC), `jina-v4` 3.75B → ≥1.05 s (Qwen Research License) · `nomic-v1.5` 136.7M → ≥38.2 ms, `v2-moe` 475M/305M active → ≥133 ms + `megablocks` · `stella_en_400M_v5` 400M → ≥111 ms, `stella_en_1.5B_v5` 1.54B → ≥429 ms |
| · sentence-transformers' larger models + `distilbert-base-uncased` | 5 | `all-mpnet-base-v2` and `paraphrase-mpnet-base-v2` 109.5M → ≥30.5 ms (the latter is SetFit's *default* body, 4.8× the incumbent); `all-distilroberta-v1` 82.1M → ≥22.9 ms; `embeddinggemma-300m` 302.9M → ≥84.5 ms and gated; `distilbert` 67.0M → ≥18.7 ms |
| · non-encoder bodies reached through family E | 3 bodies / 4 rows | `roberta-large` 355.4M → ≥99.1 ms (FastFit-large and PET); `T0_3B` 2.85B → ≥795 ms; `flan-t5-xxl` 11.27B → ≥3.14 s whose own author-claimed throughput is ">1 s per input" |
| **F2 — permissive licence** | **4** | `multi-qa-MiniLM-L6-cos-v1`, `msmarco-MiniLM-L6-cos-v5` and `msmarco-MiniLM-L12-cos-v5` (22.7M / 22.7M / 33.4M) and `TaylorAI/gte-tiny` (22.7M): **no licence declared at all** — no tag, no `LICENSE`. Three are MiniLM-shaped at 384 dim and would otherwise have competed for a slot |
| **F3 — CPU-only** | 0 | nothing reached it that could not run under torch 2.11.0+cpu; F1 had already removed everything whose only documented speed is a GPU number. It still disqualifies every published speed claim except two, and reclassifies `trust_remote_code` (jina-v2-small-en's ALiBi) as a risk rather than a filter |
| **F4 — parameter/embedding budget** | 0 | F1's ceiling is stricter than a ≤512-dim / ≤4× budget at 200 rows. What F4 establishes is a property: all five survivors are 384-dim |
| **F5 — not a loadable trainable encoder** | **5** | `potion-base-8M` 7.6M, `potion-base-32M` and `potion-retrieval-32M` 32.3M — static bag-of-tokens, nothing to contrastively train, despite the landscape's most attractive speed claim ("up to 500× faster on CPU"); `static-retrieval-mrl-en-v1` — `StaticEmbedding`, 0 active params; `TinyBERT_General_4L_312D` 15.7M **derived**, raw checkpoint, no pooling head, no published sentence-embedding evaluation (the weakest kill in the funnel, taken because a sweep would measure the wrapper). `stella-base-en-v2` is counted in F1 but dies on this rule |
| **shortlisting** — passed everything, lost a coverage slot | 8 | `all-MiniLM-L12-v2` 33.4M (dominated in its own cell; its row also contradicts itself on `max_seq` 128 v 256) · `paraphrase-MiniLM-L6-v2` 22.7M (the incumbent with strictly less information) · `arctic-embed-s` 33.2M (same hypothesis as the xs, higher count) · `mxbai-embed-xsmall-v1` 24.1M (**closest near-miss** — built on MiniLM-L6 itself, but arctic-xs occupies the same axis lower) · `e5-small-v2` and `gte-small` 33.4M (same cell as bge-small; one adds a mandatory prefix contract) · `bge-micro-v2` 17.4M (changes depth and pretraining data at once, so its result would be uninterpretable) · `jina-embeddings-v2-small-en` 32.7M (the only row that varies dim — 512 — and the only one needing `trust_remote_code` under transformers 5.6.2) |

No elimination rests on a marginal call: the nearest F1 casualty above the ceiling is `distilbert-base-uncased`
at 67.0M, 1.87× the ceiling, and nothing sits in the 36–50M band. The estimate was never load-bearing on the
elimination side — only on the admission side, which is §3's business.

Read the F1 counts two ways, because they differ on purpose. Phase 2 printed **50** model-level removals and
the audit recomputed all 50 of its printed estimates exactly; the groups above total **51** because family E's
three killed bodies occupy four rows. Against the 84-row census F1 accounts for **62** rows: those 51, plus 11
rows that could never have been candidates — the two ONNX re-exports, the five rows that re-name a model
counted in another family, and family E's four recipe rows whose body is counted in A. 62 + 22 = 84.

---

## 3. Phase 3 — proved on the box

The box: **`dell321`**, Intel Core Ultra 7 258V, **8 logical cores**, 31.57 GB RAM, **Python 3.11.9**, **torch
2.11.0+cpu** (`torch.version.cuda is None`), **transformers 5.6.2**, sentence-transformers 5.4.1, setfit 1.2.0,
no CUDA, Arc 140V iGPU unused. Protocol: `experiments/survey/probe_model.py`, one process, one utterance per
call, 8 warmup calls, 5 passes over all 382 rows, p50 = median of the per-pass medians. Per-pass p50 spreads
were 0.055–1.406 ms.

| model | load | params | dim | max_seq | licence | snapshot MB | RSS MB | ms/row | p50 ms | p95 ms | decision |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---|
| `paraphrase-MiniLM-L3-v2` | ok | 17.39M | 384 | 128 | apache-2.0 | 67.0 | 21.7 | 1.029 | **8.387** | 12.80 | **pass** |
| `all-MiniLM-L6-v2` (canary A) | ok | 22.71M | 384 | 256 | apache-2.0 | 87.3 | 21.5 | 1.788 | **15.021** | 21.62 | canary — sets the ceiling |
| `all-MiniLM-L6-v2` (canary B, 5 min later) | ok | 22.71M | 384 | 256 | apache-2.0 | 87.3 | 21.3 | 2.224 | **15.575** | 26.39 | canary — repeat, not a conflict |
| `snowflake-arctic-embed-xs` | ok | 22.57M | 384 | 512 | **none** (card field empty) | 87.1 | 34.2 | 2.231 | **12.126** | 17.67 | pass on latency; **out on licence** |
| `granite-embedding-30m-english` | ok | 30.30M | 384 | 512 | apache-2.0 | 60.3 | 50.6 | 1.696 | **11.696** | 13.58 | **pass** |
| `bge-small-en-v1.5` | ok | 33.36M | 384 | 512 | mit | 128.3 | 45.3 | 3.364 | **28.189** | 38.57 | **drop — 1.88× the incumbent** |
| *anchor row, other host* — `all-MiniLM-L6-v2` | ok | 22.71M | 384 | 256 | apache-2.0 | 87.3 | 28.4 | 1.818 | 6.330 | 8.01 | not comparable: `andrew-Lenovo-7-14IAL7`, torch 2.15.0+cpu, py 3.12.14, 12 CPUs, loadavg 8.29 |

Six probes on the box — two of them the same model, which is what a canary is for — and one on a different
host. All six loaded, all returned 384 dims and finite normalised embeddings, and every one embedded all 382
rows in under 1.4 s. RSS MB is the delta across the load only, taken after torch and sentence-transformers
were imported, so the ~500 MB those cost is not in the column.

**What the measurements settled that the estimates had wrong, in both directions.**

1. **Parameter count does not order latency across architectures and tokenisers.** `granite` at **30.3M
   measured 11.696 ms — faster than `arctic-embed-xs` at 22.6M (12.126 ms)**, a model with 34% more
   parameters answering sooner. `paraphrase-MiniLM-L3-v2` at **17.4M was fastest of all at 8.387 ms**, and
   the shortlist's deepest parameter gap (17.4M → 33.4M, 1.92×) produced a 3.4× latency gap (8.387 →
   28.189 ms). The params→latency law that admitted the shortlist is not a law, and §2's four predictions
   were all low (4.1 → 8.387, 6.3 → 12.126, 10.0 → 11.696, 11.7 → 28.189) and none high.
2. **The absolute ceiling was dead before the first model was probed.** The brief's rule was "out above ~10
   ms"; the incumbent itself measures **15.021 ms** here. Every survivor but L3-v2 exceeded 10 ms, and so
   did the canary — which under phase 2's own pre-registered rule 3 means the harness or the box is the
   finding, not the models.

**The ceiling actually used, and why.** The binding rule is a **ratio**: drop above **1.6× the incumbent's
same-session p50** — phase 2's §4 rule 2, written before any number existed. Against canary A that is
**24.03 ms**, against canary B **24.92 ms**. The two canaries bracket a **3.7% span** (15.021 → 15.575 ms),
which is what makes a ratio usable: within a session the box is stable, across sessions it is not (§5,
note 9).
`report_phase3.py` defaults to a 1.5 ratio (**22.53 ms**) and prints that as the ceiling; **no decision
changes between 1.5 and 1.6**, and both numbers are recorded rather than one. The absolute ~10 ms is kept as a
**fact about the box, not a filter**: on this CPU one MiniLM-L6 forward pass costs 15 ms per utterance, so a
10 ms budget is unachievable at the incumbent's own size by the incumbent's own recipe, and filtering on it
would have dropped every row including the reference.

**The two drops.**

- **`bge-small-en-v1.5` — out on latency.** p50 **28.189 ms = 1.88×** the incumbent's 15.021 ms, with the
  largest p95 (38.57 ms) and per-row embed cost (3.364 ms) in the set. It was the shortlist's highest-upside
  row and the one phase 2 predicted would fail; the prediction was right and the margin 2.4× larger than
  predicted.
- **`snowflake-arctic-embed-xs` — out on licence, swept anyway.** The probe reports `license: null` and
  `license_source: null`: no `license:` field in the card frontmatter, and `GET /resolve/main/LICENSE`
  returns **404** (audit re-verified). The only statement is README prose, "Arctic is licensed under the
  Apache-2." That is the identical rule that removed `TaylorAI/gte-tiny` at F2, so the row fails the brief's
  permissive-licence constraint. It is swept anyway **so the number exists** if the operator overrules the
  call; its phase-4 row is a measurement, not a candidate.

---

## 4. Phase 4 — the honest sweep

Same recipe as the ladder's rung 3, imported rather than retyped: SetFit contrastive pairs from the fold's
training pool (`num_iterations=8`, `num_epochs=1`, `batch_size=16`, `seed=0`), logistic head on
**L2-normalised** body embeddings, the same five `GroupKFold` family-held-out folds read out of
`results/ladder/rung5-ensemble.json` (`fold_assignments["5"]`, rows per fold 77/77/76/76/76, verified against
the corpus before training), same metrics block. Four models: bge-small was dropped in phase 3; arctic-xs was
carried with its licence defect recorded. Every sweep ran `status: ok`, all five folds, `missing: []`, with an
in-process determinism receipt (fold 0 refitted from scratch, `rows_differing: 0`).

### 4.1 Per-model results

Folds 0–2 are the selection half (230 rows); folds 3–4 are the validation half (152 rows) and selected
nothing.

| model | f0 | f1 | f2 | f3 | f4 | grouped pooled | rowwise pooled | `__none__` | p50 ms | selection (0-2) pooled / mean-of-folds | validation (3-4) pooled / mean-of-folds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `all-MiniLM-L6-v2` (**incumbent**) | 0.9221 | 0.8961 | 1.0000 | 0.9079 | 0.9737 | **0.9398** | 0.9424 | 0.84 | 11.70 | **0.9391** / 0.9394 | **0.9408** / 0.9408 |
| `granite-embedding-30m-english` | 0.9351 | 0.8831 | 1.0000 | 0.9342 | 0.9211 | 0.9346 | 0.9293 | 0.92 | 6.32 | **0.9391** / 0.9394 | 0.9276 / 0.9277 |
| `snowflake-arctic-embed-xs` | 0.9091 | 0.8182 | 0.8026 | 0.9868 | 0.9868 | 0.9005 | 0.9031 | 0.88 | 6.70 | 0.8435 / 0.8433 | **0.9868** / 0.9868 |
| `paraphrase-MiniLM-L3-v2` | 0.8701 | 0.8442 | 0.8026 | 0.9474 | 0.9342 | 0.8796 | 0.8953 | 0.72 | 3.67 | 0.8391 / 0.8390 | 0.9408 / 0.9408 |

Paired per-fold deltas against the incumbent — the ladder's statistic, the mean and sd of the per-fold
*difference*, not the difference of two pooled means:

| model | Δ all folds (points) | sd | folds improved | per-fold deltas, f0→f4 | Δ on validation folds |
|---|---:|---:|---:|---|---|
| `granite-embedding-30m-english` | −0.53 | 3.03 | 2/5 | +1.30, −1.30, 0.00, +2.63, −5.26 | +2.63, −5.26 |
| `snowflake-arctic-embed-xs` | −3.92 | 10.48 | 2/5 | −1.30, −7.79, −19.74, +7.89, +1.32 | +7.89, +1.32 |
| `paraphrase-MiniLM-L3-v2` | −6.03 | 8.56 | 1/5 | −5.19, −5.19, −19.74, +3.95, −3.95 | +3.95, −3.95 |

The incumbent's row is the zero line, and the survey's recipe reproduces rung 3 **to the row**: 0.9398
grouped, per-fold [0.9221, 0.8961, 1.0, 0.9079, 0.9737], 0.9424 rowwise, `__none__` 0.84, ECE 0.53 —
`provenance.reproduces_artifact: true`, every per-fold delta from the artefact 0.0.

### 4.2 The verdict: the ranking does not transfer

Checked in the reporter's stated order; the first form that fires is the verdict. The brief and the reporter
number the same three forms differently — the brief lists a win, then nothing beating it, then the ranking not
transferring; `report_phase4.py` tests a win, then does-not-transfer, then incumbent-stands — so the form that
fired is the brief's third and the reporter's second check, and it is named here by title rather than number.

| form (in the order the reporter tests them) | condition | fired |
|---|---|---|
| a model beats MiniLM by more than the fold spread | selection-half winner's validation paired delta has \|mean\| > sd and one consistent sign | **no** (0.00 vs 0.00) |
| the ranking does not transfer | no such win, and the halves disagree about the top model **or** a model that was not picked clears the spread (`clearers: []` here) | **yes** |
| nothing beats it, the incumbent stands | no model clears its spread **and** both halves agree on the top model | no |

**The selection half picks `all-MiniLM-L6-v2` at 0.9391, tied with granite at 0.9391 (mean-of-folds 0.9394
each). The validation half puts `snowflake-arctic-embed-xs` top at 0.9868 — +4.6 points over the incumbent's
0.9408 there, derived as 0.9868 − 0.9408 — after that model came third of four on the selection half at
0.8435, 9.6 points behind the incumbent.** Two folds that selected nothing decided an ordering, and decided it
for the row that was third of four on the three folds that did. The tie at the top is broken by the script,
not by a measurement: `report_phase4.py` ranks on pooled accuracy, then mean-of-folds, then **incumbent-first**,
then `model_id`, so `all-MiniLM-L6-v2` is named on the third key of an exact tie — had granite been named
instead the verdict would not move, since granite clears no validation spread (−1.32 ± 5.58) and the
validation top is arctic either way.

Operationally: **recommend the incumbent for stability.** It is the only model first or tied-first on the
half that selects and near the top on the half that reports, it reproduces the ladder exactly, and it is
already in production. There is no case here for swapping it out, and none for calling it the best model
available.

Two things make even this verdict generous:

1. **Granite ties the incumbent on the selection half and then loses on validation** — 0.9276 v 0.9408, a
   −1.32-point gap derived from those two numbers — with 2/5 folds improved and a delta smaller than its own
   sd (−0.53 ± 3.03). The row that looked like the cheap safe upgrade on the selecting folds is worse on the
   reporting folds; the transfer failure happens inside a single model.
2. **The model that causes the transfer failure is the one that fails the licence constraint.** arctic-xs is
   the validation-half winner and carries no `LICENSE` file, so the halves' disagreement is not even about a
   shippable model — the survey's most interesting phase-4 number belongs to a row phase 3 removed on paper.

**Pooled-across-all-folds accuracy entered no decision**: neither `grouped pooled` nor `rowwise pooled` is in
the ranking. The reason is the ladder's rung 6 — the rowwise evaluation set is generated from the same
templates as the training data, so a rowwise number here measures template coverage, not a classifier
(`+3.15 rowwise at −3.16 grouped` when rung 6 added the generator's own slot fills). The rows that selected a
winner cannot also be the rows that score it.

What would falsify this: move arctic-xs's fold 3 accuracy from 0.9868 to 0.8947 (68/76 — 7 rows) and the
verdict becomes "nothing beats the incumbent; the incumbent stands". Seven rows is the whole distance between
the two forms, which is a fair summary of how much this sweep establishes.

---

## 5. Method notes, including what went wrong

Each note says what was measured, what it cost, and what the fix was. Each changed an artefact this document
cites.

**1. The box's ssh flapped mid-survey.** Banner-exchange timeouts, then full loss, then recovery on retry.
Cost: every probe and sweep is a 1–15 minute unattended remote run, so a drop at the wrong moment loses the
run *and* the log that would explain it. Fix: one `ssh` per call with `-o ConnectTimeout=5` inside a local
`timeout`, every artefact written to disk **on the box** and pulled afterwards, nothing that matters held only
in a session's stdout.

**2. Windows OpenSSH kills `Start-Process` children when the session's job object dies.** A fit launched from
an ssh session inherits that session's job, so a dropped connection kills the training — which reads exactly
like "the model hung". Cost: a killed sweep and a log that says nothing. Fix: run through **Task Scheduler**,
so the job belongs to the machine and the exit code is readable afterwards.

**3. Two hung `scp.exe -t` receivers locked a script file.** Cost: the upload had failed, so the scheduled
task ran a stale, partially-written script and **exited 1 with no log at all** — no artefact, no error text.
Fix: kill the stuck receivers, upload to a fresh filename per revision, and verify size/hash after the
transfer instead of trusting `scp`'s silence.

**4. The Hugging Face client died process-wide.** `RuntimeError: Cannot send a request, as the client has been
closed.` hit all four models seconds apart while `huggingface.co:443` was reachable and every model was
already cached — the closed `httpx` pool is shared, so the first blink fails every load after it. Cost: a
whole sweep run at `failed_stage: load` for models sitting on disk. Fix: `survey_load.load_with_fallback` —
one online attempt, then `HF_HUB_OFFLINE=1` + `TRANSFORMERS_OFFLINE=1` + `local_files_only=True` and one
cached retry, sticky for the rest of the process, recorded as `load.mode`. All four phase-4 artefacts ran
`--offline`, so they record `mode: offline-fallback` with `forced: true` and `offline_reason: "--offline"`:
the forced offline path, not a fallback that recovered from a failed online attempt — no run here has watched
a real closed `httpx` pool fall back to the cache (`experiments/survey/README.md` says so).

**5. Granite's bf16 weights against an fp32 head.** `RuntimeError: Found dtype Float but expected BFloat16`,
twice, the first recorded as `failed_stage: "harness"` in a 278-byte artefact with no fold numbers
(`sweeps-prev/ibm-granite__granite-embedding-30m-english.json`). The trap underneath: the phase-3 probe read
`embeddings.dtype: float32` for granite — a float32 forward pass says nothing about the dtypes of the
weights, and numpy cannot even describe a bf16 tensor. Fix: the cast is its own stage on the encoder right
after the load, **asserted** (`load.dtype.before ["bfloat16"] → after ["float32"], coerced: true`), and a cast
that does not take fails as `dtype` with the dtype in the message.

**6. The harness's catch-all hid which stage raised.** The first Windows run wrote `failed_stage: "harness"`,
`OSError: [Errno 22] Invalid argument`, no traceback — seven fits in the sweep and no way to tell which died.
Fix: named stages (`load`, `dtype`, `fit` per fold, `rowwise`, `determinism`, `latency`), each through a
`StageBook` recording the stage, error, exception type, wall time and **full traceback**, plus a
`failed_category` taxonomy; `status` is `ok`/`partial`/`error`, `missing` lists what did not run, exit codes
follow the artefact (0/1/2). The granite failure above is now unambiguously `dtype`.

**7. Stale harness copies on the box ran the old code silently.** A run that should have exercised the new
stages reproduced the old behaviour because the executed file was the previous upload. Cost: a sweep's time
and a false "the fix did not work". Fix: a fresh filename per revision (see 3), a provenance line at process
start, and trusting the artefact's own `schema`/`stage_order` over the local tree.

**8. Estimates are not measurements, and the error went both ways.** Predicted (fitted) versus measured p50:
L3-v2 4.1 → **8.387**, arctic-xs 6.3 → **12.126**, granite 10.0 → **11.696**, bge-small 11.7 → **28.189**,
and the anchor itself 6.33 → **15.021**. All five low, one by 2.4×, and the ordering the parameter law was
built to explain (arctic-xs faster than granite) came out backwards. The lesson is not that the estimates
were bad but that the filter's first stage was a latency ceiling with **no measured latency to work from**
(§2.2b) — and the document said so at the time.

**9. The same encoder's latency is not reproducible across sessions; its accuracy is.** Three independent
sweeps of the incumbent — two on the box (02:33, 03:57) and one on the Linux host (03:07,
`verify/minilm-5fold-rerun.json`) — all produced **0.9398 grouped with per-fold [0.9221, 0.8961, 1.0, 0.9079,
0.9737]**. The same encoder's p50 across those sessions and protocols reads **6.33** (probe, Linux, loadavg
8.29), **8.222** (sweep, Linux), **9.694** and **11.702** (sweep, box), **15.021** and **15.575** (probe, box).
That is why §3 decides on a same-session canary-bracketed ratio and why §4's p50 column is **reported, not
used**: a cross-session millisecond comparison here is noise dressed as a result.

**10. The audit's funnel arithmetic, corrected.** 84 rows, not 85; 22 trainable candidates after F1 (9 A, 4
B, 9 D), not 24; 18 after F2; 13 after F5; 5 after shortlisting. Every intermediate number between the rows
and the 13 was inflated by two, by subtracting the two ONNX re-exports from a population they were never in.
The 13 and the 5 never moved and no survivor changed, but ten printed numbers did — they are correct in §2.1.

**11. "The best model scored highest" was never a verdict, and is not one here.** The brief allowed three
forms and named a fourth as inadmissible. The sweep produced the ranking-does-not-transfer form — the brief's
third, the reporter's second check — and the tempting alternative was to quote arctic-xs's 0.9868 because it
is the highest number in the document. It is 0.9868 on 152 rows that selected nothing, from a model third of
four on the 230 rows that did, with no licence file. The number is real; it means nothing about what to ship.

**12. Phase 1's gaps, recorded rather than patched.** Family E was asked a question and returned a table
(answer supplied afterwards, §1.6). Family C was asked for CPU fine-tune cost and no source publishes one for
any of its ten entries. One family-E cell cited a "~2 ms" SetFit figure with no identifier, and the preprint
it must have meant never mentions SetFit; it is deleted rather than re-cited. All three are in
`phase1-2-audit.md` and all three are visible in §1 rather than smoothed over.

### Reproduce

```bash
# phase 3 — one probe per model, on the box, under the pinned stack
.venv-embed/bin/python experiments/survey/probe_model.py \
    --model sentence-transformers/all-MiniLM-L6-v2 \
    --out results/survey-2026-11-23/probe-sentence-transformers__all-MiniLM-L6-v2.json

# phase 3 — the table and the pre-registered drop decision
.venv-embed/bin/python experiments/survey/report_phase3.py \
    --probes 'results/survey-2026-11-23/probe-*.json'

# phase 4 — one SetFit sweep per model (5 folds, rowwise, latency pass, determinism refit)
.venv-embed/bin/python experiments/survey/setfit_sweep.py \
    --model sentence-transformers/all-MiniLM-L6-v2 \
    --out results/survey-2026-11-23/sweeps/sentence-transformers__all-MiniLM-L6-v2.json

# phase 4 — the list, one process per model (--parallel 2 is opt-in and distorts latency)
.venv-embed/bin/python experiments/survey/run_survey.py \
    --models sentence-transformers/all-MiniLM-L6-v2,ibm-granite/granite-embedding-30m-english \
    --out-dir results/survey-2026-11-23/sweeps

# phase 4 — the verdict
.venv-embed/bin/python experiments/survey/report_phase4.py \
    --sweeps 'results/survey-2026-11-23/sweeps/*.json'
```

On the box every command is prefixed with that box's interpreter and run through one `ssh` layer:
`& C:\Users\andre\base-model-survey\.venv\Scripts\python.exe <script>`. `--offline` is what a re-run should
use: every model this survey measured is already in that cache.

### Artefacts

`results/` is **gitignored**. The files below live on disk and are cited by path, not committed — nothing here
is reproducible from the repository alone, and the JSON is the evidence.

| path | what it is |
|---|---|
| `docs/base-model-survey-brief-2026-11-23.md` | the four phases and the constraints |
| `docs/ladder-results-2026-11-10.md` | the incumbent (rung 3) and the system to beat (rung 5) |
| `results/survey-2026-11-23/phase1-{A,B,C,D,E}.md` | the five family surveys, 84 rows |
| `results/survey-2026-11-23/phase1-E-verdict.md` | the answer to "is SetFit still the right recipe" |
| `results/survey-2026-11-23/phase2.md` | shortlist, eliminations, estimated latency table |
| `results/survey-2026-11-23/phase1-2-audit.md` | the independent audit of phases 1 and 2 |
| `results/survey-2026-11-23/phase3-env.md` | the box's workspace, versions, and the failures getting there |
| `results/survey-2026-11-23/probe-*.json` | 7 probe artefacts (6 on the box, 1 anchor on the Linux host) |
| `results/survey-2026-11-23/phase3-table.{md,json}` | generated by `report_phase3.py`, ceiling 1.5× by default |
| `results/survey-2026-11-23/sweeps/*.json` | the four phase-4 sweeps (MiniLM, granite, arctic-xs, L3-v2) |
| `results/survey-2026-11-23/sweeps-prev/*.json` | the earlier run, incl. granite's `dtype` failure row |
| `results/survey-2026-11-23/verify/minilm-5fold-rerun.json` | the incumbent's sweep re-run on the Linux host |
| `results/survey-2026-11-23/determinism/` | two cross-process fold-0 runs of the incumbent, identical |
| `results/survey-2026-11-23/phase4-table.{md,json}` | generated by `report_phase4.py` — the verdict above |
| `results/ladder/rung5-ensemble.json` | the fold assignment phase 4 reads, and rung 5's numbers |
| `experiments/survey/README.md` | the harness: probes, sweeps, runners, failure taxonomy, self-tests |
