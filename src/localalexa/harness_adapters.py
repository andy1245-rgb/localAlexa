"""The localAlexa side of the shared harness: training hooks, not a second harness.

The harness at ``~/toolInterfacer/bench`` scores predictors it did not train —
``torch_text`` loads a checkpoint and calls ``trainer=module:function`` when a
grouped CV needs a model refitted per fold. This module is that function for the
ladder's SetFit rung, and the checkpoint writer it needs.

The checkpoint contract is the harness's::

    {"state_dict": {"encoder.<key>": tensor, "head.weight": tensor, "head.bias": tensor},
     "labels": [...], "encoder": "sentence-transformers/all-MiniLM-L6-v2",
     "pooling": "mean", "normalize": True, "max_length": 256}

Three details decide whether the port is faithful rather than merely working:

* **normalize=True.** all-MiniLM-L6-v2 has a Normalize module after its mean-pool
  layer, so SetFit's head is trained on unit-norm embeddings. The harness pools
  without normalising by default; without this flag the arm would score a model
  that never existed.
* **the head is the fitted logistic regression, not a new one.** ``model_head`` is
  sklearn's ``LogisticRegression`` over the encoder's embeddings, so its
  ``coef_``/``intercept_`` are the head weights and its ``classes_`` are the label
  order the harness must use.
* **the same training settings as the ladder rung**, imported rather than copied,
  so the arm and the rung cannot drift apart.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from .ladder.setfit_rung import ENCODER, HEAD_ITERATIONS, SetFitPredictor

CHECKPOINT_KEYS = ("state_dict", "labels", "encoder", "pooling", "normalize", "max_length")


def _setfit_args(batch_size: int):
    from setfit import TrainingArguments

    from .classifiers import SEED

    return TrainingArguments(
        batch_size=batch_size,
        num_epochs=1,
        num_iterations=HEAD_ITERATIONS,
        seed=SEED,
        report_to=[],
        output_dir=tempfile.mkdtemp(prefix="setfit-harness-"),
    )


def train_setfit(predictor, rows) -> None:
    """Fit SetFit on ``rows`` and write a ``torch_text`` checkpoint where it is loaded from.

    Called by the harness as ``trainer(predictor, rows)`` once per fold. It writes
    to the path the predictor was built with, so no per-fold file naming is needed
    — the harness drops its loaded state after the call and reloads this one.
    """
    from datasets import Dataset
    from setfit import SetFitModel, Trainer

    if not rows:
        raise ValueError("train_setfit: no rows")
    texts = [row.text for row in rows]
    labels = [row.label for row in rows]
    model = SetFitModel.from_pretrained(ENCODER)
    Trainer(
        model=model,
        train_dataset=Dataset.from_dict({"text": texts, "label": labels}),
        args=_setfit_args(batch_size=16),
    ).train()

    target = getattr(predictor, "checkpoint", "") or Path(tempfile.mkdtemp()) / "setfit.pt"
    write_checkpoint(model, Path(target), rows=len(rows))
    predictor.checkpoint = Path(target)


def write_checkpoint(model, path: Path, *, rows: int = 0) -> Path:
    """Export a fitted ``SetFitModel`` in the harness's checkpoint contract."""
    import torch

    encoder = model.model_body[0].auto_model
    head = model.model_head
    state = {f"encoder.{key}": tensor for key, tensor in encoder.state_dict().items()}
    state["head.weight"] = torch.tensor(head.coef_, dtype=torch.float32)
    state["head.bias"] = torch.tensor(head.intercept_, dtype=torch.float32)
    blob = {
        "state_dict": state,
        "labels": [str(label) for label in head.classes_],
        "encoder": ENCODER,
        "pooling": "mean",
        "normalize": True,
        "max_length": int(model.model_body.max_seq_length),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, str(path))
    return path


def trainer_spec() -> str:
    return "localalexa.harness_adapters:train_setfit"


def arm_specs(checkpoint: str) -> dict[str, str]:
    """The four ladder arms, as harness predictor specs.

    Kept here so a run and its documentation cannot disagree about what was scored:
    ``ladder_port.py`` in the harness prints these next to the numbers.
    """
    setfit = (
        f"torch_text|checkpoint={checkpoint}|encoder={ENCODER}"
        f"|trainer={trainer_spec()}|normalize=true|max_length=256"
    )
    return {
        "r0-word-svc": "sklearn_text|vectorizer=tfidf_word|clf=linsvc|C=1",
        "r2-word-logreg-C100": "sklearn_text|vectorizer=tfidf_word|clf=logreg|C=100",
        "r3-setfit": setfit,
        "r5-ensemble-setfit-z": (
            "ensemble|members="
            "a:sklearn_text~vectorizer=tfidf_word~clf=logreg~C=100"
            f";;b:{setfit}"
            "|mode=mean|standardise=true"
        ),
    }
