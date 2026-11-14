# Hands-off data collection — corrections are negatives, silence is nothing

2026-11-10, branch `collect/2026-11-10`. Implements the brief in
`docs/data-collection-brief-2026-11-10.md`. Local only, no network, one append
per utterance in the request path. `cascade.py` semantics and
`SMART_ASSUMED_ACC` were not touched; `tests/test_collection.py
::test_the_log_records_the_tier_the_cascade_would_have_chosen` holds the logger to
`cascade.assign_tiers` on all 382 evaluation rows so the two cannot drift.

## 1. The premise, and the thing it rules out

The operator corrected the earlier design: **not correcting is not evidence of
correct**, because he will not always notice. A silent row is therefore not a
positive, not a weak positive, and not a "probably fine". It is nothing.

That single line is why the positive signals below are three narrow ones rather
than one convenient one. It costs yield — a week of quiet traffic produces almost
no positives — and the weekly summary reports `rows with no signal` explicitly so
the loss is visible instead of being quietly backfilled by an assumption.

## 2. The signal table, as implemented

| signal | label | where it is recognised | strength |
|---|---|---|---|
| the operator corrects the assistant | **negative** | `corrections.detect` — an action, within a window, about the same thing | **strong** |
| an effect check fails | negative | `EffectRecord(verified=False)` from the caller's tooling | weak |
| an effect check passes | positive | `EffectRecord(verified=True)` — the timer exists, the list changed | weak |
| a second model agrees | positive | `opinion.SecondOpinion`, offline over the whole log | weak |
| top two candidates are one class, or the margin is wide | **auto-accepted label** | `autoaccept.decide` — the only source needing nobody | auto |
| **the operator does not correct** | **nothing** | *deliberately absent from the code* | — |

One label per row, the strongest that applies. What got outranked is written into
the row's `detail.suppressed` rather than discarded: "the second model agreed and
the operator corrected it" is evidence about the *agreement rule*, not about the
operator, and throwing that away would hide a bad signal inside a good one.

```
priority  signal                polarity  strength
   50     correction             −1       strong
   40     effect_failed          −1       weak
   30     effect                 +1       weak
   20     auto_accept_*          +1       auto
   10     opinion_agreement      +1       weak
```

## 3. What is logged, and when

`var/collect/requests.jsonl` — one JSON object per line, appended by
`CollectionTap.observe()` **before the response is returned**:

| field | why it is there |
|---|---|
| `id`, `ts`, `at`, `session` | identity and ordering; `session` is what the correction window is scoped to |
| `text` | the redacted utterance — the training example |
| `redactions` | which exclusion rules fired, so a reader knows what was removed |
| `tier` | 0 regex, 1 router, 2 escalated |
| `label` | what was acted on; `null` when it escalated |
| `candidates` | top 3 with scores, exactly as the router ranked them |
| `confidence`, `top_sim`, `margin` | the numbers the gate and the auto-accept rule read |
| `gate`, `sim_threshold`, `conf_threshold` | whether the row is in the reject pile, and the thresholds it was judged against |
| `route_ms` | the router's own decision time, kept out of the logging budget |

`var/collect/effects.jsonl` — one line per effect check, written when the check
happens rather than in the hot path.

Then the derived files, of which there are two kinds:

| file | written by | shape |
|---|---|---|
| `corrections.jsonl` | live, appended per correction found | one row per corrected request, with rule and evidence |
| `labels.jsonl` | live, appended per signal | **events**: every signal that fired, including ones a later one outranks |
| `rejects.jsonl` | live, appended per below-gate row | the pile |
| `labels.reduced.jsonl` | `collect derive`, offline | one label per request, strongest wins |
| `meta.json` | live | which rules and signals produced the files beside it |

The split matters twice. Keeping the events means "auto-accepted by margin, then
corrected by the operator" survives to be audited — a reduction that dropped the
superseded row would destroy the only evidence of whether the margin rule is
safe. And keeping the *reduction* offline is what makes the live path flat: see
§8.

The raw two files are the record; nothing else is authoritative, and every
derived file can be rebuilt from them.

There is no fsync by default. A completed `write(2)` survives the process dying,
which is the crash the brief cares about (the row must not be lost if the process
dies before responding). It does not survive the machine losing power; `sync_every`
exists for a caller who wants that and pays for it.

## 4. The correction detector, and its limits

A correction is the only negative that needs nothing from a model, because it is
an *action*. It is also inferential: "something else, shortly after, about the
same thing" is my reading of an event I cannot see. So every correction carries
`rule` (version, window, thresholds, branches enabled) and `evidence` (which
branch fired, which cue matched, which tokens were shared).

Rule `corr-1`, all four conditions required:

1. same session as the earlier row;
2. the earlier row is the most recent **acted-on** row of that session (tier 0 or
   1 — an escalated row produced no answer to correct);
3. `0 < Δt ≤ 45 s` (default, `--window`) and the two texts are not identical — an
   identical repeat is a repeat;
4. one of three branches fires:
   - **cue** — the later text contains a correction cue ("no", "actually", "I
     meant", "make it", "scratch that", …). Matched on word boundaries, so "no"
     does not match "north";
   - **same operation** — the two rows carry the same label and share at least one
     content token: "a timer for five minutes" → "a timer for ten minutes";
   - **overlap** — token Jaccard ≥ 0.5, whatever the labels say.

Contents words are lowercased tokens minus a small function-word list, so
"*set a timer for the kitchen*" reduces to `{set, timer, kitchen}`.

**Where it is wrong, stated because it will be:** two unrelated requests inside
45 s that share an entity ("lights on" / "lights off") are flagged; a change too
small to move a token ("five minutes" → "5 minutes") is missed; a correction that
arrives in a fresh session is invisible; and a correction of the *escalation*
(where the smart tier answered and got it wrong) has no earlier label to mark, so
it produces nothing. The window and every branch are configurable, and because
detection reads the raw log and writes a derived file, revising the rule re-derives
every correction instead of leaving a mixture of rule vintages in the labels.

One correction per row: if the operator says "no" twice, only the first earlier
row is marked.

## 5. The auto-accept rule, and what it does not mean

Rule `auto-1`, either condition:

- **same class** — the top two candidates normalise to one label (lowercase,
  alphanumeric only, then the alias map). With a single model over a single label
  space this never fires; it is there for ensembles and for routers whose
  candidates carry several spellings of one intent.
- **margin** — `top_sim − runner_up_sim ≥ 0.05` (default, `--margin`).

Rows that escalated are not eligible (no label to accept), and neither are rows the
regex answered (accepting those would record the pattern table back to itself).

The margin rule says the router was *unambiguous*, not that it was *right*. That is
why the strength is `auto`, a third category that is never mixed into the weak
positives: a downstream consumer can weight it without having to know which
rule produced it. In the smoke session the margin rule accepted the operator's
correction utterance ("no i said ten minutes" → `timer_create`) — which is
plausibly correct and entirely unverified, and the labels file says so.

## 6. The reject pile

Every row below the gate, with its top-3 and nothing else: no label, no
interpretation, no guessed intent. It is a derived view of the log rather than a
second write in the request path, so it cannot disagree with the log and costs the
request nothing.

It exists to answer one question — *which intents are mutually confusable* — and
the weekly summary prints the top ten ordered pairs from it.

## 7. The weekly summary

```
last 7 days · 6 rows · 1 sessions
below the gate 3 (50.0%) · tier0 2 / tier1 1 / escalated 3
labels 3 · auto-accepted 1 · corrections 1 · rows with no signal 3
most confused:
  light_query → light_set  1
  shoppinglist_add → shoppinglist_view  1
  shoppinglist_view → shoppinglist_add  1
```

Counts only, four things the brief asked for (rows, share below the gate, ten most
confused pairs, auto-accepted labels) plus two that keep the yield honest:
corrections, and rows that produced nothing at all. `localalexa collect summary
--json` gives the same numbers machine-readable.

## 8. The request path

`localalexa.service.Assistant` is the composition root — the thing firmware or an
HTTP layer calls. `localalexa ask "<utterance>"` runs one request through it from
the shell, which is how it gets exercised today:

```
$ localalexa ask "set a timer for five minutes"
timer_create (tier 0) · confidence 1.000 · no router call
  candidates: timer_create 1.00
recorded 88bcfa72e4f841cc in var/collect/requests.jsonl
settled 1 row(s) in 0.37 ms: 0 correction(s), 0 label(s), wrote nothing

$ localalexa ask "no i meant ten minutes"
reminder_create (tier 1) · confidence 0.78 · 1.4 ms
  candidates: reminder_create 0.78, timer_create 0.03, __none__ 0.03
recorded c464b7fe08fc459b in var/collect/requests.jsonl
settled 1 row(s) in 1.01 ms: 1 correction(s), 2 label(s), wrote corrections.jsonl, labels.jsonl
```

Two separate processes, and the second one found the first row by reading the
log. That is the property the correction detector needs and the reason it runs
off the logged rows rather than off a request list it happens to have in memory.

The order inside `handle` is the contract: tier 0 regex (the router is not even
called when exactly one pattern fires, same as `run_cascade` does with its
residue), else the router, else the log append, else settle, else the response.
A crash between the decision and the answer loses the answer but never the row.

`settle=False` moves the derivation to an explicit call for a caller that would
rather batch it; the row guarantee is unaffected. `record_requests=False` turns
logging off entirely (demos and tests).

## 9. What it costs the request path

Measured on an idle box, 4,000 appends / 1,000 tap calls
(`localalexa collect bench`):

| operation | p50 | p95 | p99 |
|---|---:|---:|---:|
| the append alone (`os.write` of a formatted line) | **0.61 µs** | 1.68 µs | 2.42 µs |
| the whole tap call (redact → rank → format → append) | **15.2 µs** | 16.7 µs | 20.1 µs |

The write is under a microsecond. The other 14.6 µs is redaction (2.9 µs), record
construction (6.0), `json.dumps` (7.7) and the local ISO timestamp (1.2), and it
is 3.9% of a 0.39 ms linear routing decision and 0.2% of a 7.8 ms encoder one.

**Settling is flat, and it took a measurement to make it so.** The first
implementation re-derived the tail on every settle and rewrote the derived files
when they changed. Measured at 1.6 ms with 50 rows in the tail and **15.6 ms with
500** — an order of magnitude more than the routing decision it sits next to, and
growing with traffic:

| tail | settle, rebuilding (rejected) | settle, appending (shipped) |
|---:|---:|---:|
| 50 rows | 1.60 ms | 0.21 ms |
| 200 rows | — | 0.20 ms |
| 500 rows | 15.60 ms | 0.20 ms |

The shipped version appends one line per signal and reduces offline. The remaining
cost is the window scan for the previous acted-on row, which is bounded by the
correction window and not by the log. `tests/test_service.py
::test_settle_stays_flat_as_the_log_grows` fails if it regresses past 5 ms.

No buffering, because buffering is what would break the crash guarantee, and
nothing here needs to be deferred at that size. If it ever does, the cheap move is
a background thread holding a queue and doing the format+write — the record
identity is assigned before the hand-off, so the row can still be found by id.
Redaction is already behind a cheap pre-gate: a rule whose trigger characters are
absent is not run, which took it from 6.8 µs to 2.9 µs and is the only
optimisation here worth having.

A logging failure — full disk, revoked permissions — is counted on the log
(`.errors`), warned about once per distinct message on stderr, and swallowed. The
operator still gets their answer.

## 10. Storage and privacy

Everything is local files under `var/collect/` (override with
`$LOCALALEXA_COLLECT`), mode `0600`, and `var/` is gitignored. Nothing in the
package imports `socket`, `urllib`, `http.client`, `httpx`, `requests`, `aiohttp`,
`ftplib` or `smtplib` — enforced by a test, not by intention.

Redaction happens at write time. The utterance is kept, the matched span is
replaced with a typed marker, and the rule names are recorded on the row:

| rule | matches | becomes |
|---|---|---|
| `url` | `https://`, `www.` | `[url]` |
| `email` | `user@host.tld` | `[email]` |
| `phone` | leading `+`, `(555) 555-5555`, `555-555-5555` | `[phone]` |
| `path_posix` | `/home/…`, `~/…`, `./…` | `[path]` |
| `path_windows` | `C:\…` | `[path]` |
| `ipv4` | dotted quad | `[ip]` |
| `secret` | `sk-…`, `ghp_…`, `AKIA…`, `xox…` | `[secret]` |
| `long_number` | 12+ digits | `[long-number]` |
| — | text over 400 characters | truncated, `truncated` recorded |

Dates and durations survive on purpose: `2026-11-10` and `10 minutes` are the
content of a calendar and a timer command, not incidental. That is a judgement
call and the list is one tuple in `collection/redact.py`, so changing it is one
edit.

**The honest limit:** this is a fixed list of things that *can* be recognised. A
third party's name, a street address, a medication — anything a sentence can
contain that nobody wrote a pattern for — is stored. The mitigation available
without a model is not more regexes; it is that the log is local, private, and the
operator's to delete. Nothing here expires rows automatically: deletion is a
decision, not a background job, and it stays the operator's.

## 11. Failure modes of the design itself

- **Yield is low and bursty.** Quiet traffic yields nothing, which is the intended
  outcome and the reason the summary counts silent rows.
- **Corrections are biased toward noticed mistakes.** A confident wrong answer
  that sounded plausible is exactly what does not get corrected. The negative set
  will over-represent the mistakes that *sounded* wrong.
- **Second-opinion agreement is correlated confirmation.** The opinion model is
  trained on the same 200 references as the router, so agreement is partly shared
  training data. It is counted as weak for that reason, and the one thing it is
  definitely good for is catching an *ensemble* disagreement.
- **Auto-accept labels are unverified by construction.** They are the cheapest
  positives and the only ones that need nobody; a consumer that weights them
  equally with effect-verified rows is misusing them.
- **The detector will be revised.** That is why it writes to a derived file with
  its own rule version, and why the raw log never takes an annotation.

## 12. Reproduce

```bash
.venv/bin/python -m localalexa ask "turn off the hall light"   # one request, logged and settled
.venv/bin/python -m localalexa collect bench          # the request-path cost table
.venv/bin/python -m localalexa collect derive         # rebuild rejects/corrections/labels
.venv/bin/python -m localalexa collect summary        # the ten-second read
.venv/bin/python -m localalexa collect summary --json --days 30
.venv/bin/python experiments/collect/smoke_session.py # end-to-end, temporary root
.venv/bin/python -m pytest tests/test_collection.py   # 37 tests, one per signal
```

Wiring it into a live path is one object:

```python
from localalexa.service import Assistant, default_predictor

predictor, why = default_predictor(registry)      # encoder if installed, else the linear stand-in
assistant = Assistant(registry, predictor=predictor)
response = assistant.handle("turn off the hall light")   # logged, settled, answered
```

and, when the operator's tooling can check an action, one more call:

```python
assistant.append_effect(response.record_id, verified=True, check="timer_registry")
```

A caller that would rather own the turn boundary keeps `settle=False` on
construction and calls `assistant.settle()` when the turn ends.
