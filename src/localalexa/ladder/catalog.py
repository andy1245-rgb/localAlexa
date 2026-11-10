"""The ladder's rung catalog, in the order the plan walks it.

Rungs 3–6 need packages the base install does not carry (torch, a transformer
encoder, a generative model). They are imported lazily: asking for a rung whose
dependencies are missing fails with the one-line reason instead of an
ImportError from four frames down, and ``--rungs all`` skips it with that reason
in the artifact rather than quietly reporting a shorter ladder.
"""

from __future__ import annotations

import importlib
import importlib.util

from .run import Rung
from .rungs import LINEAR_RUNGS

OPTIONAL_RUNG_MODULES = (
    ("r3-setfit", "localalexa.ladder.setfit_rung", "rung_3_setfit", ("torch", "setfit")),
    (
        "r4-crossencoder",
        "localalexa.ladder.crossencoder_rung",
        "rung_4_crossencoder",
        ("torch", "transformers"),
    ),
    ("r5-ensemble", "localalexa.ladder.ensemble_rung", "rung_5_ensemble", ()),
    (
        "r5-ensemble-setfit",
        "localalexa.ladder.ensemble_rung",
        "rung_5_ensemble_setfit",
        ("torch", "setfit"),
    ),
    (
        "r5-ensemble-setfit-z",
        "localalexa.ladder.ensemble_rung",
        "rung_5_ensemble_setfit_standardised",
        ("torch", "setfit"),
    ),
    (
        "r6-augment",
        "localalexa.ladder.augment_rung",
        "rung_6_template_augmented",
        (),
    ),
    (
        "r6-augment-capped",
        "localalexa.ladder.augment_rung",
        "rung_6_template_augmented_cap16",
        (),
    ),
    (
        "r6-augment-paraphrase",
        "localalexa.ladder.augment_rung",
        "rung_6_paraphrase_augmented",
        (),
    ),
)


def catalog() -> tuple[list[Rung], list[str]]:
    """Every rung that can run here, plus the reason each other one cannot."""
    rungs: list[Rung] = list(LINEAR_RUNGS)
    skipped: list[str] = []
    for name, module_name, factory_name, requires in OPTIONAL_RUNG_MODULES:
        missing = [dep for dep in requires if importlib.util.find_spec(dep) is None]
        if missing:
            skipped.append(f"{name} unavailable: needs {', '.join(missing)}")
            continue
        try:
            module = importlib.import_module(module_name)
            rungs.append(getattr(module, factory_name)())
        except ImportError as exc:  # pragma: no cover - depends on the local env
            skipped.append(f"{name} unavailable: {exc}")
            continue
    return rungs, skipped


def available_rungs(spec: str = "all") -> list[Rung]:
    rungs, skipped = catalog()
    if spec == "all":
        for reason in skipped:
            print(f"skip {reason}")
        return rungs
    wanted = [name.strip() for name in spec.split(",") if name.strip()]
    by_name = {rung.name: rung for rung in rungs}
    missing = [name for name in wanted if name not in by_name]
    if missing:
        known = sorted(by_name) + [reason.split(" ")[0] for reason in skipped]
        raise SystemExit(f"unknown or unavailable rungs {missing}; known: {known}")
    return [by_name[name] for name in wanted]
