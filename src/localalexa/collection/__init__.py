"""Hands-off data collection where a correction is the only free negative.

The design premise, in the operator's own correction to the earlier plan: *not
correcting is not evidence of correct*, because he will not always notice. So
silence is never a label. What is left is four signals, in order of trust:

| signal | label | why it counts |
|---|---|---|
| the operator corrects the assistant | **strong negative** | an action, not an absence |
| an effect check fails / passes | weak negative / weak positive | the world answered, not the operator's attention |
| a second model agrees | weak positive | correlated with the router, so weak in isolation |
| top two candidates agree or a wide margin | **auto-accepted label** | needs nothing from anyone |

Nothing here prompts, interrupts or asks. Nothing leaves the machine.

Layout:

``RequestLog``       one append per utterance, written before the response
``CollectionTap``    wraps a router; works out the tier, records, returns
``derive``           replays the log into corrections, the reject pile and labels
``summarise``        the ten-second weekly read
``bench``            what all of it costs the request path, in microseconds
"""

from .autoaccept import Acceptance, AutoAcceptRule, normalise_label
from .bench import measure_append, measure_tap
from .corrections import Correction, CorrectionRule, detect
from .derive import Derived, derive
from .log import RequestLog, collect_root, new_session, write_jsonl
from .opinion import SecondOpinion
from .records import Candidate, EffectRecord, LabelRow, RequestRecord
from .redact import EXCLUSIONS, redact
from .summary import Summary, confused_pairs, render, summarise
from .tap import CollectionTap, tier_for

__all__ = [
    "EXCLUSIONS",
    "Acceptance",
    "AutoAcceptRule",
    "Candidate",
    "CollectionTap",
    "Correction",
    "CorrectionRule",
    "Derived",
    "EffectRecord",
    "LabelRow",
    "RequestLog",
    "RequestRecord",
    "SecondOpinion",
    "Summary",
    "collect_root",
    "confused_pairs",
    "derive",
    "detect",
    "measure_append",
    "measure_tap",
    "new_session",
    "normalise_label",
    "redact",
    "render",
    "summarise",
    "tier_for",
    "write_jsonl",
]
