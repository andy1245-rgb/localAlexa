# Ladder environment

Two virtualenvs, because the ladder's later rungs need torch and the base
install is deliberately small (numpy + scikit-learn, 61 tests, sub-second fits).

## `.venv` — the repo's documented environment, unchanged

```
python 3.12.14 · numpy 2.5.4 · scikit-learn 1.9.1 · scipy 1.18.1 · pytest 9.2.0
```

Runs everything through rung 2 and rung 6a, all of `experiments/ladder/` except
the two rungs that need torch, and the whole test suite.

## `.venv-embed` — rungs 3 and 4 only

Created with `uv venv .venv-embed --python 3.12`, then:

```
uv pip install --python .venv-embed/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv-embed/bin/python setfit sentence-transformers
uv pip install --python .venv-embed/bin/python -e .
uv pip install --python .venv-embed/bin/python pytest   # to run the suite in this env too
```

```
python 3.12.14 · torch 2.15.0+cpu · transformers 5.22.2 · datasets 5.2.0
setfit 1.2.0 · sentence-transformers 6.2.0 · scikit-learn 1.9.1 · numpy 2.5.4
```

Model weights, downloaded from the Hub at $0: `sentence-transformers/all-MiniLM-L6-v2`
(22M, the SetFit body and the cross-encoder body) and `google/flan-t5-base`
(probed and rejected for paraphrase generation, see below). Nothing was
installed with a paid API; no data was deleted; the base `.venv` was not touched.

`.venv-embed/` and SetFit's scratch output are gitignored.

## The generator probe that closed rung 6b

`google/flan-t5-base` on CPU, `do_sample=True, temperature=1.1, top_p=0.95,
max_new_tokens=24`, three samples per prompt per utterance:

| prompt | utterance | output |
|---|---|---|
| `Rewrite this smart-home voice command in different words: {}` | set a timer for 5 minutes | `timer for 5 minutes` · `turn on a timer for 5 minutes` · `push the p button on the front of your Smart-Home speaker.` |
| same | what's the temperature in the bedroom | `what's the temperature in the bedroom` (twice) · `you're at home for a day and ask what's the temperature of the room` |
| same | add milk to the shopping list | `add milk to the shopping list` · `add milk to shopping list` (twice) |
| `Paraphrase: {}` | is the porch light on | `yes` · `wall` · `h` |
| `Paraphrase: {}` | what meetings do I have today | `what meetings do i have today?` · `what is in my office?` · `i'm on lunch break today.` |

Generation cost: 6–28 s per batch of three on this CPU. Eight hundred candidates
at that rate is over an hour of output like the above — mostly copies of the
input, with the rest drifting off-intent. No `llama.cpp`, no `ollama`, and no
GGUF files on this box, so the generator the plan named is not available here and
the one that is cannot do the job.

That is rung 6's paraphrase half recorded as **unmeasured, with the reason**, not
as a negative result about augmentation. The machinery to use such paraphrases
exists and is fold-aware (`src/localalexa/ladder/augment_rung.py`): rows carry the
reference they came from, a fold drops the paraphrases of any reference it had to
drop, and the intended filter is "keep only rows the current model is unsure
about". It needs an instruct model — a 7B GGUF under llama.cpp, or any hosted
model if the $0 constraint is dropped — and a human spot-check of a 40-row sample,
which is a step no model can substitute for.
