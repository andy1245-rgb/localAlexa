"""The encoder's dtype, made explicit instead of inherited from the checkpoint.

``ibm-granite/granite-embedding-30m-english`` ships bf16 weights. Loaded on CPU
it stays bf16, setfit's logistic head trains in float32, and the first pass
through the body raises

    RuntimeError: Found dtype Float but expected BFloat16

which the sweep recorded as ``failed_stage: "harness"`` — the harness's own
dtype mismatch wearing a generic label. Nothing about that row said which model
or which dtype.

The cast belongs on the encoder, once, immediately after it is loaded and before
anything is trained: the body is the thing whose weights are bf16, and casting
there is a property of the measurement ("this model evaluated in float32 on
CPU"), not a patch inside a loss function that would hide which tensor
disagreed. Doing it in the loss would also make the number incomparable with
every other row in the sweep, silently.

So both halves of the contract are checked and written down:

* ``before``/``after`` — the distinct parameter dtypes across the body, before
  and after the cast. A checkpoint that already loads in float32 says
  ``coerced: false`` and is otherwise untouched.
* ``embeddings`` — the dtype the body actually hands the head, read from a real
  forward pass with ``convert_to_tensor=True``. This is a different fact from the
  parameters' dtypes, and the granite probe is the proof: it read
  ``embeddings.dtype: float32`` for that model — through numpy, from a forward
  pass that ran in float32 — while the fit raised ``Found dtype Float but
  expected BFloat16``. A numpy view cannot describe a bf16 tensor at all either
  (measured: ``np.asarray`` raises ``TypeError: Got unsupported ScalarType
  BFloat16``), so the tensor dtype is what gets recorded here, beside the
  parameter dtypes rather than instead of them.

A cast that does not take, or a body that still emits something other than
float32, raises :class:`DtypeCoercionError` carrying the dtype strings that were
found, so the stage that records it can write the actual dtype into the row
instead of a generic message.
"""

from __future__ import annotations

from survey_env import describe_error

FLOAT32 = "float32"


class DtypeCoercionError(RuntimeError):
    """The encoder is not, and could not be made, float32."""

    def __init__(
        self,
        message: str,
        *,
        dtypes: list[str] | None = None,
        embedding_dtypes: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.dtypes = list(dtypes or [])
        self.embedding_dtypes = dict(embedding_dtypes or {})


def _dtype_name(dtype) -> str:
    text = str(dtype)
    return text[len("torch.") :] if text.startswith("torch.") else text


def body_of(model):
    """The ``SentenceTransformer`` inside a ``SetFitModel``, or the model itself."""
    body = getattr(model, "model_body", None)
    return model if body is None else body


def parameter_dtypes(model) -> list[str]:
    """Distinct parameter dtypes, in the order they are first seen."""
    try:
        parameters = list(body_of(model).parameters())
    except Exception:
        return []
    names: list[str] = []
    for parameter in parameters:
        name = _dtype_name(getattr(parameter, "dtype", "unknown"))
        if name not in names:
            names.append(name)
    return names


def device_of(model) -> str:
    return str(getattr(body_of(model), "device", "cpu"))


def embedding_dtypes(model, texts: list[str]) -> dict[str, str]:
    """The dtype of the embeddings the body hands the head, plus numpy's view of it."""
    import numpy as np

    body = body_of(model)
    tensor = body.encode(list(texts), convert_to_tensor=True, show_progress_bar=False)
    block = {"torch": _dtype_name(getattr(tensor, "dtype", "unknown")), "numpy": "unknown"}
    try:
        block["numpy"] = str(np.asarray(tensor).dtype)
    except Exception as exc:
        block["numpy"] = f"unavailable ({describe_error(exc, 80)})"
    return block


def _cast(body, torch) -> None:
    cast = getattr(body, "float", None)
    if callable(cast):
        cast()
        return
    body.to(torch.float32)


def coerce_float32(model, texts: list[str]) -> dict[str, object]:
    """Cast a CPU encoder to float32 and prove it, or raise.

    Returns the record that goes into the artefact under ``load.dtype``; raises
    :class:`DtypeCoercionError` — whose message carries the actual dtype — when
    the cast does not take, the body reports no parameter dtype to check, or the
    embeddings are not float32 afterwards. On a device other than ``cpu`` nothing
    is cast: ``skipped`` says why and ``after`` equals ``before``.
    """
    import torch

    device = device_of(model)
    body = body_of(model)
    before = parameter_dtypes(model)
    record: dict[str, object] = {
        "requested": FLOAT32,
        "device": device,
        "before": before,
        "after": None,
        "coerced": False,
        "embeddings": None,
        "note": (
            "cast on cpu immediately after the load and before any fit: the SetFit "
            "head trains in float32, and nothing in this harness uses an accelerator"
        ),
    }
    if device != "cpu":
        record["after"] = before
        record["skipped"] = f"device is {device}, not cpu; after equals before"
        return record
    if not before:
        raise DtypeCoercionError(
            "could not read any parameter dtype from the encoder, so it cannot be "
            "asserted as float32"
        )
    if [name for name in before if name != FLOAT32]:
        try:
            _cast(body, torch)
        except Exception as exc:
            raise DtypeCoercionError(
                f"could not cast the encoder from {'/'.join(before)} to float32: "
                f"{describe_error(exc)}",
                dtypes=before,
            ) from exc
        record["coerced"] = True

    after = parameter_dtypes(model)
    record["after"] = after
    if not after or [name for name in after if name != FLOAT32]:
        raise DtypeCoercionError(
            f"encoder is still {'/'.join(after) or 'unknown'} after the cast to "
            f"float32 (loaded as {'/'.join(before)}); the SetFit head trains in float32",
            dtypes=after or before,
        )
    try:
        embeddings = embedding_dtypes(model, texts)
    except Exception as exc:
        raise DtypeCoercionError(
            f"encoded a probe text to read the embedding dtype and the body raised: "
            f"{describe_error(exc)}",
            dtypes=after,
        ) from exc
    record["embeddings"] = embeddings
    if embeddings["torch"] != FLOAT32:
        raise DtypeCoercionError(
            f"encoder parameters are float32 but its embeddings are "
            f"{embeddings['torch']} (numpy would call them {embeddings['numpy']}); "
            f"the SetFit head trains in float32",
            dtypes=after,
            embedding_dtypes=embeddings,
        )
    return record
