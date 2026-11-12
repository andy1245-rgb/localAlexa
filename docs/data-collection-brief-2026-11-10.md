# Hands-off data collection — corrections are negatives, nothing else is free

2026-11-10. The operator approved option 1 + 2, then raised a correct objection: **he will not always
notice a mistake, so "no correction" cannot be treated as a positive.** This brief is the corrected
design. The earlier framing is wrong and must not be implemented.

## The signal table, and what each is worth

| signal | label | why |
|---|---|---|
| the operator corrects the assistant | **negative, strong** | they noticed and acted. Reliable because it is an action, not an absence |
| the operator does not correct | **nothing** | inattention is indistinguishable from satisfaction. Never a positive |
| the action's effect is verifiable (timer actually created, list actually changed) | **positive, weak** | evidence from the world, independent of attention |
| a second independent model agrees with the routing | **positive, weak** | co-training; useful in bulk, weak in isolation |
| top-2 candidates agree by a wide margin | **label, auto-accepted** | the only fully hands-off source of clean labels |

Everything below is logging and bookkeeping. **No interface, no prompting the operator, no habit.**

## What to build

1. **A request log.** Every utterance the router sees, with: timestamp, the raw text, the chosen
   label, the top-3 candidates with their scores, the confidence, and which tier answered. One
   append-only file, one JSON object per line, written before the response is returned so a crash
   cannot lose the row.
2. **A correction detector.** When the operator re-issues or amends a request shortly after one that
   was acted on — same session, within a short window, referring to the same thing — mark the earlier
   row as corrected. This is a *heuristic* and must be labelled as one in the data: a `corrected_by`
   field, never a silent overwrite. The window and the matching rule are configurable and recorded
   with every row they touch, because the heuristic will be revised and old rows must stay readable.
3. **A reject pile.** Any row below the confidence gate, retained with its top-3. No labelling, no
   interpretation. This is the artefact that answers "which intents are mutually confusable".
4. **The auto-accept rule.** When the top two candidates are the same class after normalisation, or
   when the margin exceeds a stated threshold, write the row to a `labels/` file as an accepted
   label. Record the rule that accepted it. This is the only source of positives that needs nothing
   from the operator.
5. **A weekly summary the operator can read in ten seconds**: rows logged, share below the gate,
   the ten most confused intent pairs, and how many auto-accepted labels were produced. Counts only.

## What this must not do

- **Must not treat silence as a label.** The single most important line in this brief.
- **Must not phone home.** The log is local, the summary is local, nothing is uploaded.
- **Must not slow the request path** measurably. Logging is a file append; if it costs more than a
  few microseconds, say so and buffer it.
- **Must not store more than it needs.** The raw utterance is required for retraining; anything else
  captured incidentally (paths, names of third parties) is dropped at write time, and there is a
  documented exclusion list.

## Deliverable

The logger, the correction heuristic, the auto-accept rule, the weekly summary, a unit test for each
of the five signals above with a hand-constructed case, and `docs/data-collection.md` stating the
signal table, the heuristics and their limits, and the fact that non-correction is deliberately
unused.

## Hard limits

Branch commits only, never push. No network. Do not change `cascade.py` semantics or
`SMART_ASSUMED_ACC` — both are the operator's decisions and both are on his list.
