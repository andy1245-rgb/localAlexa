# Optional experiments — external resources required

Verbatim copies of the four experiment scripts whose dependencies are not part of this
package and are not available offline. They are kept as records of how the numbers in
`results/historical/` were produced; `src/localalexa/` holds the adapted versions of the
scripts that do run offline (`router_lib.py`, `exp_classic.py`, `exp_cascade.py`,
`gen_data.py`).

| script | needs |
|---|---|
| `exp_qwen.py` | a downloaded Qwen small model (`download_qwen.py`, Hugging Face) — CPU probe, no fine-tune |
| `exp_d1.py` | a llama.cpp server (\>= b11500) serving `LiquidAI/d1-3B` at `/v1/systemone` |
| `exp_needle.py` | the Cactus/Needle SDK plus its on-device model, and a warm `agent` handle |
| `exp_router.py` | `sentence-transformers` and the MiniLM weights (installable: `pip install 'localalexa[embeddings]'`) |
| `download_qwen.py` | `huggingface_hub` and network access |

Path assumptions: these were written for the flat `jevResearchCompiled/experiments/`
layout and expect `registry.json` and `data/test.json` **beside the script**. To run one
from here, copy those two files next to it first; nothing in this directory is imported
by the package or by the test suite.

`exp_needle.py`'s metric block (per-tool / per-family accuracy, confusion pairs,
none-detection) is the logic `localalexa.evaluate` was adapted from; the abstain curve,
calibration bins and tau sensitivity in `exp_router.py` were **not** ported — they need
the embedding backend, and the shipped `evaluate` only reports accuracy under the three
splits.

`results/historical/d1_results.json` holds the 91.36% figure quoted in these scripts'
docstrings. That was measured on a different task (smart-home command routing) and is not
comparable to any accuracy in this repo's README.
