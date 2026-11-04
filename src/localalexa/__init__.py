"""localAlexa: offline intent classification for a voice assistant.

A three-tier cascade decides what a spoken command means without calling a
large model for routing: hand-written regex (tier 0) -> reference-similarity
router or a TF-IDF linear classifier (tier 1) -> abstain and escalate (tier 2).

The package also ships the measurement code, because the interesting result is
that this project measured its own generalisation penalty: the same classifier
scores far higher row-wise than when whole phrasings are held out.
"""

from .cascade import (
    CascadeRow,
    CompiledPatterns,
    assign_tiers,
    cascade_metrics,
    regex_tier_metrics,
    run_cascade,
)
from .classifiers import MODEL_SPECS, TextClassifier, train_and_score
from .data import (
    CHITCHAT_FAMILY,
    EXPECTED_ROWS,
    NONE_LABEL,
    Registry,
    Row,
    Tool,
    load_registry,
    load_routing_set,
    phrasing_families,
    template_index,
    training_pairs,
)
from .evaluate import (
    SplitResult,
    cluster_bootstrap_ci,
    family_held_out,
    measure,
    row_wise,
)
from .router import Choice, SemanticRouter, decide, entropy_confidence, softmax

__version__ = "0.1.0"

__all__ = [
    "CHITCHAT_FAMILY",
    "EXPECTED_ROWS",
    "MODEL_SPECS",
    "NONE_LABEL",
    "CascadeRow",
    "Choice",
    "CompiledPatterns",
    "Registry",
    "Row",
    "SemanticRouter",
    "SplitResult",
    "TextClassifier",
    "Tool",
    "assign_tiers",
    "cascade_metrics",
    "cluster_bootstrap_ci",
    "decide",
    "entropy_confidence",
    "family_held_out",
    "load_registry",
    "load_routing_set",
    "measure",
    "phrasing_families",
    "regex_tier_metrics",
    "row_wise",
    "run_cascade",
    "softmax",
    "template_index",
    "train_and_score",
    "training_pairs",
    "__version__",
]
