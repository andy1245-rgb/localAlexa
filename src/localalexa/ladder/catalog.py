"""The ladder's rung catalog, in the order the plan walks it.

Rungs 3–6 need packages the base install does not carry (torch, a transformer
encoder, a generative model). They are imported lazily: asking for a rung whose
dependencies are missing fails with the one-line reason instead of an
ImportError from four frames down, and ``--rungs all`` skips it with that reason
in the artifact rather than quietly reporting a shorter ladder.
"""

from __future__ import annotations

import importlib

from .run import Rung
from .rungs import LINEAR_RUNGS

OPTIONAL_RUNG_MODULES = (
    ("r3-setfit", "localalexa.ladder.setfit_rung", "rung_3_setfit"),
    ("r4-crossencoder", "localalexa.ladder.crossencoder_rung", "rung_4_crossencoder"),
    ("r5-ensemble", "localalexa.ladder.ensemble_rung", "rung_5_ensemble"),
    ("r5-ensemble-setfit", "localalexa.ladder.ensemble_rung", "rung_5_ensemble_setfit"),
    ("r5-ensemble-setfit-z", "localalexa.ladder.ensemble_rung", "rung_5_ensemble_setfit_standardised"),
    ("r6-augment", "localalexa.ladder.augment_rung", "rung_6_template_augmented"),
)


def catalog() -> tuple[list[Rung], list[str]]:
    """Every rung that can run here, plus the reason each other one cannot."""
    rungs: list[Rung] = list(LINEAR_RUNGS)
    skipped: list[str] = []
    for name, module_name, factory_name in OPTIONAL_RUNG_MODULES:
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:  # pragma: no cover - depends on the local env
            skipped.append(f"{name} unavailable: {exc}")
            continue
        rungs.append(getattr(module, factory_name)())
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
