"""The classifier ladder: does anything in this repo beat one linear model?

Six rungs, walked in expected-gain order, all scored on one grouped fold
assignment under identical code. A rung that does not beat rung 0 by more than
the fold-to-fold spread has not won, whatever the point estimate says.

See ``docs/classifier-ladder-2026-11-10.md`` for the plan and
``docs/ladder-results-2026-11-10.md`` for what came back.
"""

from .folds import DEFAULT_FOLDS, FoldAssignment, build_folds
from .run import Bench, Rung, block_metrics, evaluate_grouped, evaluate_rowwise, load_bench

__all__ = [
    "DEFAULT_FOLDS",
    "Bench",
    "FoldAssignment",
    "Rung",
    "block_metrics",
    "build_folds",
    "evaluate_grouped",
    "evaluate_rowwise",
    "load_bench",
]
