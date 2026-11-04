> ## Correction, 2026-11-04 — the 93.2% below is a row-wise figure
>
> Section 3 of this document (written 2026-09-21) says the classic-ML models were
> "tested on 382 held-out utterances". That wording is misleading and is left in
> place below only so the history stays readable. The measured facts, re-run from
> the code in this repo (`python -m localalexa eval --split all`):
>
> | split | what training sees | accuracy | 95% CI |
> |---|---|---|---|
> | row-wise | the 200 registry references | 93.2% (356/382) | 88.0–97.6 |
> | phrasing-family held out | same pool, held-out family's slot-fill references dropped | 90.3% (345/382) | 85.0–95.2 |
> | phrasing-family held out, strict | rebuilt from the other 109 families' template fills | 74.9% (286/382) | 66.7–83.1 |
>
> The 382 rows are generated *from the same templates* the registry trains on, and
> 29 of the 200 training references are literal slot fills of those templates. So
> 93.2% measures template overlap as much as it measures understanding: the same
> classifier loses 18.3 points once the held-out family's wording is removed from
> training. Row-wise is the number not to quote; the strict split is the generalisation
> estimate. The intervals resample whole phrasing families (110 clusters), not rows.
>
> A prior audit of this work recorded 92.1% row-wise and 68.6% family-held-out
> (gap −23.5, cluster-bootstrap 95% CI 57.3–79.1) but did not record the split
> mechanics. The numbers above are what the shipped code reproduces; 68.6% falls
> inside the strict split's interval, and the row-wise 92.1% differs from the
> reproducible 93.2% by four rows, so that split's composition is not recoverable.
>
> Separately: `d1_results.json`'s 91.36% (a 3B decision model) was measured on a
> **different task** — smart-home command routing, whose label set does not
> overlap this project's operations. It is kept in `results/historical/` as a
> record and is not comparable to any number above.

---

# RESEARCH.md — classifier stack & vision chains (2026-09-21)

Companion to the experiments in `experiments/`. Covers: Qwen3/3.5 fine-tune bases,
classic ML, lightweight inference stacks, and vision model pairs/chains.

---

## 1. The localAlexa decision layer == a local "System One"

Our Binner cascade (regex -> router -> cactus) is the same pattern as TypeSafe's Jev /
localjev ecosystem (see jevResearch/03): typed decisions (choice + probabilities +
entropy confidence) served by tiny models, reserving LLM calls for actual work.
Jev's `confidence = 1 - H(p)/ln(K)` is implemented in `router_lib.py` verbatim.
Our experiments validate it locally — see FINDINGS.

## 2. Fine-tune base candidates (the "cactus replacement" question)

| Model | Size | Notes |
|---|---|---|
| **Qwen3.5 small series** (0.8B / 1.7B / 4B / 9B) | Mar 2026 release, Apache 2.0 | Native multimodal family, 256K ctx. 0.8B is the direct MiniCPM5-1B competitor for intent/splitter fine-tunes. |
| Qwen3-0.6B / 1.7B | Apr 2025 | Battle-tested edge sizes, huge fine-tune ecosystem. |
| LFM2 family (Liquid AI) | 350M-1.2B | "Most tunable" per distil-labs 15-model/9-task benchmark. Hybrid conv+attention = fast on CPU. |
| MiniCPM5-1B | current scaffold brain | Keep as Tier-4; fine-tuning it for classification wastes its strength (synthesis). |
| ModernBERT (139M/395M) | encoder | The **joint multi-head** candidate: bin head + span head + slot head on one encoder (JointBERT/SLIM pattern), 8K ctx, fast on CPU. |

**Path A (two cactuses) needs:** splitter = seq2seq fine-tune (DialogUSR task shape);
intent = classification fine-tune. Qwen3.5-0.8B or LFM2-450M are the right size class
for both; ModernBERT is the single-model joint alternative.

## 3. Classic ML — EMPIRICALLY TESTED (exp_classic.py, this repo)

Trained on the same 200 references the router uses, tested on 382 held-out utterances:

| Model | Accuracy | p50 latency | Train time |
|---|---|---|---|
| TF-IDF word(1-2) + LogisticRegression | 92.9% | 0.66 ms | 0.1 s |
| TF-IDF char(3-5) + LogisticRegression | 91.6% | 0.81 ms | 0.2 s |
| TF-IDF word(1-2) + LinearSVC | **93.2%** | 0.61 ms | <0.1 s |

- Weakness: no calibrated abstain (none-class detection 0.6-0.8; fixable with a
  max-score threshold = OOD gate, like the router's sim floor).
- Same failure mode as predicted: light_query vs light_set (read/write confusion).
- **Verdict: a legitimate Tier-0.5.** Sub-millisecond, no download, retrained in
  seconds as the registry grows. Between regex (0% cost, brittle) and router (~15 ms).

## 4. Lightweight inference stacks ("opencv and shit" bucket)

- **ONNX Runtime + Optimum**: quantize MiniLM to int8 ONNX -> router latency drops
  well below the 13-17 ms torch number; onnxruntime is already installed.
  `optimum-cli export onnx` or fast-sentence-transformers. Expect 2-5x CPU speedup.
- **OpenVINO**: same idea, Intel-optimized; SigLIP zero-shot classification notebooks
  exist ready-made (docs.openvino.ai) — relevant for the vision tier below.
- **fastText**: the OG sub-word linear classifier, still unbeatable for pure speed
  (<0.1 ms); our TF-IDF+LR numbers say the accuracy gap would be small.
- **OpenCV**: tier-0 vision work — frame differencing, MOG2 background subtraction,
  HOG person detection. Runs everywhere, no NN weights, real-time on a Pi.

## 5. VISION CHAINS — model pairs for the camera problem

Same cascade law as the classifier stack: cheap tier always-on, expensive tier on demand.

### The chain (4 tiers)

```
Tier V0  OpenCV gates          frame diff / MOG2 / HOG person   ~free, always-on
         |  (only wakes the stack on change/person)
Tier V1  tiny encoder          CLIP/SigLIP zero-shot (OpenVINO), YOLO-nano/MobileNet-SSD  ~10-50ms CPU
         |  (what/who is in frame, bounding boxes; zero-shot via label prompts)
Tier V2  small VLM on demand   LFM2.5-VL-450M (cached), SmolVLM 256M/500M/2.2B,
                               moondream2 (cached), gemma-4-E2B + mmproj (cached)   ~0.5-3s
         |  (captioning, VQA, reading text in frame)
Tier V3  scaffold reasoning    MiniCPM5 receives the VLM caption as a tool result
```

### Pairing patterns (how models team up)

1. **Gate + VLM**: OpenCV triggers, VLM explains. 99% of frames never reach a NN.
2. **Encoder + decoder pair**: SigLIP encodes the frame -> matched against text
   prompts zero-shot ("a person holding a soldering iron") -> only ambiguous frames
   escalate to the VLM. SigLIP-2 beats CLIP at equal size.
3. **Detector + VLM**: YOLO-nano finds regions -> VLM gets a cropped region instead
   of the full frame (smaller images = faster VLM tokens).
4. **VLM as captioner, LLM as reasoner**: VLM emits a text caption; the caption enters
   the orchestrator as a tool result (`camera_describe`). Vision becomes just another
   tool bin — the scaffold doesn't even know it's special.
5. **On-device pair (ESP32-CAM)**: ESP-DL / microML person or face detection runs on
   the ESP32 itself; only detections stream to the Pi. Mirrors microWakeWord for audio.

### Local model inventory (already on disk)

| Model | Size | Role |
|---|---|---|
| LFM2.5-VL-450M-Q4_0.gguf + mmproj | ~220 MB | Tier V2 default — tiny VLM via llama.cpp `--mmproj` |
| gemma-4-E2B + mmproj GGUF | ~3.3 GB | Tier V2 quality option |
| moondream2 (HF cache) | ~1.8 B | Alternative V2, strong captioning for size |
| SmolVLM 256M/500M | smallest | If 450M still too slow; Apache, video-capable (SmolVLM2) |
| Meta EUPE encoders | <100M | New compact vision-encoder family for V1 if SigLIP heavy |

### localAlexa use cases, ranked

1. **Presence/presence-wake** (V0+V1): camera satellite wakes the assistant when a
   person appears — the visual equivalent of openWakeWord.
2. **Electronics ingestion** (V1+V2, serves electronicsProjects): component
   recognition on workbench photos, part counting, thumbnail QA, wiring-reading via
   VLM caption. The "grab images + ingredients list" feature in note1.md.
3. **Gesture/state commands** (V1): simple pose/object states as tool triggers.
4. **"What do you see?"** (V2→V3): voice asks about the room; VLM captions, MiniCPM
   reasons — full chain, still local.

## 6. Open questions for next experiments

- Does int8-ONNX MiniLM keep the 93%-class accuracy headroom of the router?
- Train ModernBERT joint (bin+slots) on registry-generated data vs the router — gap?
- Needle3 vs Qwen3.5-0.8B fine-tuned: is the cactus tier even needed, or does
  classic-ML + router cover everything the logs will show?
- V0->V1 gate tuning: false-wake rate of MOG2 on a Pi with real room lighting.
