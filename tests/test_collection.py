"""Tests for hands-off collection: one per signal, plus the guarantees.

The five signals get a hand-constructed case each. The rest covers the promises
that break silently: the row is on disk before the response returns, a logging
failure cannot fail the request, the tier the log records is the tier the cascade
chose, silence produces no label at all, and nothing here can reach the network.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path

import pytest

from localalexa.cascade import CompiledPatterns, assign_tiers, regex_tier_metrics
from localalexa.collection.autoaccept import AutoAcceptRule, decide as decide_auto, normalise_label
from localalexa.collection.bench import APPEND_BUDGET_US, measure_append
from localalexa.collection.corrections import content_tokens
from localalexa.collection.derive import derive
from localalexa.collection.log import RequestLog, write_jsonl
from localalexa.collection.records import Candidate, EffectRecord, RequestRecord
from localalexa.collection.opinion import SecondOpinion
from localalexa.collection.redact import redact
from localalexa.collection.summary import confused_pairs, render, summarise
from localalexa.collection.tap import CollectionTap, tier_for
from localalexa.data import load_registry, load_routing_set
from localalexa.router import Choice, decide

COLLECTION_DIR = Path(__file__).resolve().parents[1] / "src" / "localalexa" / "collection"


def record(
    index: int,
    text: str,
    label: str | None,
    ts: float,
    *,
    session: str = "s1",
    tier: int = 1,
    margin: float = 0.01,
    confidence: float = 0.6,
    candidates: tuple[Candidate, ...] | None = None,
    gate: bool = False,
) -> RequestRecord:
    if candidates is None:
        candidates = (
            Candidate(label or "escalated", 0.5),
            Candidate("light_set" if label != "light_set" else "light_query", 0.3),
            Candidate("thermostat_set", 0.2),
        )
    return RequestRecord(
        id=f"row-{index:03d}",
        ts=ts,
        session=session,
        text=text,
        tier=tier,
        label=label,
        confidence=confidence,
        top_sim=0.6,
        margin=margin,
        candidates=candidates,
        gate=gate,
        sim_threshold=0.45,
        conf_threshold=0.35,
    )


# ---------------------------------------------------------------- signal 1: correction


def test_a_correction_is_a_strong_negative_on_the_row_it_corrects():
    earlier = record(0, "set a timer for five minutes", "timer_create", ts=1000.0)
    later = record(1, "no, i said ten minutes", "timer_create", ts=1008.0)

    derived = derive([earlier, later])
    assert len(derived.corrections) == 1
    labels = derived.by_id()
    assert set(labels) == {"row-000"}  # the correction utterance itself earns nothing
    label = labels["row-000"]
    assert label.label == "timer_create"
    assert label.polarity == -1
    assert label.strength == "strong"
    assert label.source == "correction"
    assert label.detail["corrected_by"] == "row-001"
    assert label.detail["evidence"]["branch"] == "cue"
    assert label.detail["evidence"]["cue"] == "no"
    # The window and the rule travel with the row, so a later reader can see
    # which heuristic produced it.
    assert label.detail["rule"]["version"] == "corr-1"
    assert label.detail["rule"]["window_seconds"] == 45.0


def test_same_operation_without_a_cue_counts_when_parameters_changed():
    earlier = record(0, "set a timer for five minutes", "timer_create", ts=1000.0)
    later = record(1, "set a timer for ten minutes", "timer_create", ts=1012.0)
    derived = derive([earlier, later])
    assert len(derived.corrections) == 1
    assert derived.corrections[0].evidence["branch"] == "same_operation"


def test_repeating_the_same_utterance_is_not_a_correction():
    earlier = record(0, "set a timer for five minutes", "timer_create", ts=1000.0)
    later = record(1, "set a timer for five minutes", "timer_create", ts=1006.0)
    assert derive([earlier, later]).corrections == ()


def test_a_late_utterance_is_not_a_correction():
    earlier = record(0, "set a timer for five minutes", "timer_create", ts=1000.0)
    later = record(1, "no, ten minutes", "timer_create", ts=1000.0 + 120.0)
    assert derive([earlier, later]).corrections == ()


def test_an_escalated_row_has_nothing_to_correct():
    earlier = record(0, "mumble mumble", None, ts=1000.0, tier=2, gate=True)
    later = record(1, "no, i said ten minutes", "timer_create", ts=1004.0)
    assert derive([earlier, later]).corrections == ()


def test_corrections_do_not_cross_sessions():
    earlier = record(0, "set a timer for five minutes", "timer_create", ts=1000.0, session="a")
    later = record(1, "no, ten minutes", "timer_create", ts=1003.0, session="b")
    assert derive([earlier, later]).corrections == ()


# ---------------------------------------------------------------- signal 2: silence


def test_a_confident_row_with_no_other_signal_produces_no_label():
    # High confidence, answered by the router, nothing else happened. This is the
    # brief's central rule: the operator not complaining is not evidence.
    settled = record(0, "turn on the kitchen lights", "light_set", ts=1000.0, margin=0.001)
    derived = derive([settled])
    assert derived.labels == ()
    assert derived.stats["labels"] == 0
    assert derived.stats["rows_without_signal"] == 1


def test_silence_is_not_counted_as_either_polarity():
    rows = [
        record(0, "turn on the kitchen lights", "light_set", ts=1000.0),
        record(1, "is the porch light on", "light_query", ts=1030.0),
        record(2, "cancel the timer", "timer_cancel", ts=1060.0),
    ]
    derived = derive(rows)
    assert derived.stats["labels_positive"] == 0
    assert derived.stats["labels_negative"] == 0
    assert derived.stats["rows_without_signal"] == 3


# ---------------------------------------------------------------- signal 3: verified effect


def test_a_verified_effect_is_a_weak_positive():
    row = record(0, "set a timer for five minutes", "timer_create", ts=1000.0, margin=0.001)
    effect = EffectRecord(request_id="row-000", ts=1001.0, verified=True, check="timer_registry")
    derived = derive([row], [effect])
    label = derived.by_id()["row-000"]
    assert label.polarity == 1
    assert label.strength == "weak"
    assert label.source == "effect"
    assert label.detail["check"] == "timer_registry"


def test_a_failed_effect_check_is_a_weak_negative():
    row = record(0, "add milk to the shopping list", "shoppinglist_add", ts=1000.0, margin=0.001)
    effect = EffectRecord(
        request_id="row-000", ts=1001.0, verified=False, check="list_diff", detail="no change"
    )
    label = derive([row], [effect]).by_id()["row-000"]
    assert label.polarity == -1
    assert label.source == "effect_failed"


def test_a_correction_outranks_a_verified_effect_and_says_so():
    earlier = record(0, "set a timer for five minutes", "timer_create", ts=1000.0)
    later = record(1, "no, i said ten minutes", "timer_create", ts=1005.0)
    effect = EffectRecord(request_id="row-000", ts=1002.0, verified=True, check="timer_registry")
    label = derive([earlier, later], [effect]).by_id()["row-000"]
    assert label.polarity == -1
    assert label.source == "correction"
    assert label.detail["suppressed"] == ["effect"]


# ---------------------------------------------------------------- signal 4: second opinion


class StubOpinion:
    """A stand-in for the second model: fixed, so the case is the case."""

    def __init__(self, mapping: dict[str, str]) -> None:
        self.mapping = mapping

    def labels(self, texts):
        return [self.mapping.get(text, "__none__") for text in texts]

    def label(self, text):
        return self.mapping.get(text, "__none__")


def test_model_agreement_is_a_weak_positive():
    row = record(0, "turn on the kitchen lights", "light_set", ts=1000.0, margin=0.001)
    opinion = StubOpinion({"turn on the kitchen lights": "light_set"})
    label = derive([row], opinion=opinion).by_id()["row-000"]
    assert label.polarity == 1
    assert label.strength == "weak"
    assert label.source == "opinion_agreement"


def test_model_disagreement_produces_nothing_rather_than_a_negative():
    # A correlated model disagreeing is not an action by anyone; it is not a label.
    row = record(0, "turn on the kitchen lights", "light_set", ts=1000.0, margin=0.001)
    opinion = StubOpinion({"turn on the kitchen lights": "light_query"})
    assert derive([row], opinion=opinion).labels == ()


def test_the_real_second_opinion_fits_and_agrees_with_itself():
    registry = load_registry()
    opinion = SecondOpinion(registry)
    text = "set a timer for five minutes"
    first = opinion.label(text)
    assert first == opinion.label(text)
    assert opinion.agrees(text, first)
    assert not opinion.agrees(text, "__none__")


# ---------------------------------------------------------------- signal 5: top-2 auto-accept


def test_two_spellings_of_one_class_are_auto_accepted():
    row = record(
        0,
        "turn on the kitchen lights",
        "light_set",
        ts=1000.0,
        candidates=(Candidate("light_set", 0.5), Candidate("light-set", 0.3), Candidate("x", 0.1)),
    )
    acceptance = decide_auto(row, AutoAcceptRule(margin=9.9))
    assert acceptance is not None
    assert acceptance.rule == "same_class"
    assert acceptance.label == normalise_label("light_set")


def test_a_wide_margin_is_auto_accepted_and_records_the_rule():
    row = record(0, "dim the kitchen lights", "light_set", ts=1000.0, margin=0.2)
    acceptance = decide_auto(row, AutoAcceptRule(margin_threshold=0.05))
    assert acceptance is not None
    assert acceptance.rule == "margin"
    assert acceptance.rule_version == "auto-1"
    assert acceptance.detail["threshold"] == 0.05


def test_a_narrow_margin_earns_nothing():
    row = record(0, "dim the kitchen lights", "light_set", ts=1000.0, margin=0.001)
    assert decide_auto(row, AutoAcceptRule(margin_threshold=0.05)) is None


def test_escalated_and_regex_rows_are_never_auto_accepted():
    escalated = record(0, "hmm", None, ts=1000.0, tier=2, margin=0.9, gate=True)
    regex = record(1, "set a timer for 5 minutes", "timer_create", ts=1001.0, tier=0, margin=0.9)
    rule = AutoAcceptRule(margin_threshold=0.05)
    assert decide_auto(escalated, rule) is None
    assert decide_auto(regex, rule) is None


def test_auto_accepted_labels_appear_in_the_derived_labels_with_their_rule():
    row = record(0, "dim the kitchen lights", "light_set", ts=1000.0, margin=0.2)
    label = derive([row]).by_id()["row-000"]
    assert label.source == "auto_accept_margin"
    assert label.strength == "auto"
    assert label.detail["rule_version"] == "auto-1"


def test_a_correction_overrides_an_auto_accepted_label():
    earlier = record(0, "dim the kitchen lights", "light_set", ts=1000.0, margin=0.2)
    later = record(1, "no, turn them up", "light_set", ts=1005.0, margin=0.2)
    label = derive([earlier, later]).by_id()["row-000"]
    assert label.polarity == -1 and label.strength == "strong"
    assert "auto_accept_margin" in label.detail["suppressed"]


# ---------------------------------------------------------------- the request path


def test_the_row_is_on_disk_before_the_response_returns(tmp_path):
    log = RequestLog(tmp_path / "requests.jsonl", session="s1")
    tap = CollectionTap(log, id_factory=lambda: uuid.uuid4().hex[:8])
    seen_during = {}

    def predictor(text: str) -> Choice:
        seen_during["rows"] = log.rows()
        return decide([0.7, 0.2, 0.1], ["light_set", "light_query", "x"])

    logged = tap.wrap(predictor)
    choice = logged("turn on the kitchen lights")
    assert choice.top_tool == "light_set"
    assert seen_during["rows"] == []  # nothing written while the router was thinking
    rows = log.rows()  # and written by the time the caller has the answer
    assert len(rows) == 1
    assert rows[0]["text"] == "turn on the kitchen lights"
    assert rows[0]["label"] == "light_set"
    assert rows[0]["candidates"][0]["label"] == "light_set"
    log.close()


def test_a_logging_failure_does_not_fail_the_request(tmp_path, monkeypatch):
    log = RequestLog(tmp_path / "requests.jsonl", session="s1")

    def broken(fd, data):
        raise OSError("disk full")

    monkeypatch.setattr(os, "write", broken)
    tap = CollectionTap(log, id_factory=lambda: "x1")
    choice = tap.wrap(lambda text: decide([0.7, 0.2, 0.1], ["light_set", "light_query", "x"]))(
        "turn on the kitchen lights"
    )
    assert choice.top_tool == "light_set"  # the operator still got an answer
    assert log.errors == 1
    log.close()


def test_batch_wrapping_logs_one_row_per_utterance(tmp_path):
    log = RequestLog(tmp_path / "requests.jsonl", session="s1")
    tap = CollectionTap(log, id_factory=lambda: uuid.uuid4().hex[:8])
    texts = ["turn on the lights", "is the porch light on", "cancel the timer"]

    def predict_batch(batch):
        return [decide([0.7, 0.2, 0.1], ["light_set", "light_query", "timer_cancel"]) for _ in batch]

    choices = tap.wrap_batch(predict_batch)(texts)
    assert len(choices) == 3
    assert [row["text"] for row in log.rows()] == texts
    log.close()


def test_torn_last_line_is_ignored(tmp_path):
    path = tmp_path / "requests.jsonl"
    log = RequestLog(path, session="s1")
    log.append(record(0, "turn on the lights", "light_set", ts=1000.0))
    log.close()
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"kind":"request","id":"half')  # a crash mid-write
    assert len(RequestLog(path).rows()) == 1


def test_the_log_records_the_tier_the_cascade_would_have_chosen():
    registry = load_registry()
    rows = load_routing_set()
    compiled = CompiledPatterns.from_registry(registry)
    hits = regex_tier_metrics(rows, compiled)["_hits"]
    labels = registry.label_ids
    # A spread of similarities wide enough to straddle both gates.
    choices = {
        i: decide(
            [0.62 if j == i % len(labels) else 0.50 - 0.004 * j for j in range(len(labels))],
            labels,
        )
        for i in range(len(rows))
    }
    cascaded = assign_tiers(rows, hits, choices, 0.45, 0.35)
    for i, line in enumerate(cascaded):
        tier, label = tier_for(rows[i].text, choices[i], compiled, 0.45, 0.35)
        assert (tier, label) == (line.tier, line.predicted), rows[i].text


def test_a_regex_row_is_logged_with_the_label_the_regex_chose(tmp_path):
    registry = load_registry()
    log = RequestLog(tmp_path / "requests.jsonl", session="s1")
    tap = CollectionTap(log, registry=registry, id_factory=lambda: "x1")
    row = tap.observe("set a timer for 5 minutes", decide([0.6, 0.2], ["light_set", "light_query"]))
    assert row.tier == 0
    assert row.label == "timer_create"  # the regex hit, not the router's guess
    assert [c.label for c in row.candidates] == ["timer_create"]
    log.close()


# ---------------------------------------------------------------- storage and privacy


def test_redaction_keeps_utterances_and_drops_the_rest():
    assert redact("turn on the kitchen lights") == ("turn on the kitchen lights", ())
    assert redact("call 555-123-4567") == ("call [phone]", ("phone",))
    assert redact("mail me at a@b.co")[0] == "mail me at [email]"
    assert redact("see /home/andrew/notes.txt")[0] == "see [path]"
    assert redact("token sk-ABCDEFGH123")[0] == "token [secret]"


def test_dates_and_durations_survive_redaction():
    # Redaction must not eat the parts of an utterance that carry meaning.
    assert redact("what is on 2026-11-10")[0] == "what is on 2026-11-10"
    assert redact("set a timer for 10 minutes")[0] == "set a timer for 10 minutes"


def test_redaction_names_are_recorded_on_the_row(tmp_path):
    log = RequestLog(tmp_path / "requests.jsonl", session="s1")
    tap = CollectionTap(log, id_factory=lambda: "x1")
    tap.observe("call 555-123-4567 and turn on the lights", None)
    row = log.rows()[0]
    assert row["redactions"] == ["phone"]
    assert "555" not in row["text"]
    log.close()


def test_the_log_file_is_private(tmp_path):
    log = RequestLog(tmp_path / "requests.jsonl", session="s1")
    log.close()
    assert (log.path.stat().st_mode & 0o777) == 0o600


def test_nothing_in_the_collection_package_can_reach_the_network():
    forbidden = ("socket", "urllib", "http.client", "httpx", "requests", "aiohttp", "ftplib", "smtplib")
    for path in sorted(COLLECTION_DIR.glob("*.py")):
        source = path.read_text(encoding="utf-8")
        for name in forbidden:
            assert f"import {name}" not in source, f"{path.name} imports {name}"
            assert f"from {name}" not in source, f"{path.name} imports from {name}"


def test_the_append_stays_within_its_budget():
    stats = measure_append(300)
    assert stats["p50_us"] < APPEND_BUDGET_US


# ---------------------------------------------------------------- the weekly summary


def test_summary_counts_what_the_brief_asks_for():
    rows = [
        record(0, "turn on the kitchen lights", "light_set", ts=time.time() - 10, margin=0.2),
        record(
            1,
            "is the porch light on",
            None,
            ts=time.time() - 5,
            tier=2,
            gate=True,
            candidates=(Candidate("light_query", 0.4), Candidate("light_set", 0.35), Candidate("x", 0.1)),
        ),
        record(
            2,
            "cancel the timer",
            None,
            ts=time.time() - 4,
            tier=2,
            gate=True,
            candidates=(Candidate("timer_cancel", 0.4), Candidate("timer_status", 0.3), Candidate("x", 0.1)),
        ),
    ]
    derived = derive(rows)
    summary = summarise(rows, [label.as_dict() for label in derived.labels], days=7.0)
    assert summary.rows == 3
    assert summary.below_gate == 2
    assert summary.gate_share_pct == pytest.approx(66.7, abs=0.1)
    assert summary.labels_total == 1
    assert summary.labels_by_source == {"auto_accept_margin": 1}
    text = render(summary)
    assert "3 rows" in text
    assert "below the gate 2 (66.7%)" in text
    assert "light_query → light_set" in text
    assert "auto-accepted 1" in text


def test_confused_pairs_come_only_from_below_the_gate():
    rows = [
        record(
            0,
            "is the porch light on",
            None,
            ts=1.0,
            tier=2,
            gate=True,
            candidates=(Candidate("light_query", 0.4), Candidate("light_set", 0.3)),
        ),
        record(
            1,
            "are the lights on",
            None,
            ts=2.0,
            tier=2,
            gate=True,
            candidates=(Candidate("light_query", 0.4), Candidate("light_set", 0.3)),
        ),
        record(2, "dim the lights", "light_set", ts=3.0, margin=0.5),
    ]
    assert confused_pairs(rows) == [("light_query", "light_set", 2)]


def test_summary_renders_with_no_traffic():
    summary = summarise([], [], days=7.0)
    assert summary.rows == 0
    assert "no row fell below the gate" in render(summary)


# ---------------------------------------------------------------- derived artefacts


def test_labels_are_written_one_row_per_request(tmp_path):
    rows = [
        record(0, "dim the kitchen lights", "light_set", ts=1000.0, margin=0.2),
        record(1, "turn off the hall light", "light_set", ts=1001.0, margin=0.2),
    ]
    derived = derive(rows)
    path = write_jsonl(tmp_path / "labels.jsonl", [row.as_dict() for row in derived.labels])
    written = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(written) == 2
    assert {row["request_id"] for row in written} == {"row-000", "row-001"}
    assert all(row["strength"] == "auto" for row in written)


def test_content_tokens_drop_function_words_only():
    assert content_tokens("set a timer for the kitchen") == {"set", "timer", "kitchen"}


def test_summary_accepts_label_rows_as_well_as_dicts():
    row = record(0, "dim the kitchen lights", "light_set", ts=time.time(), margin=0.2)
    derived = derive([row])
    from_dicts = summarise([row], [label.as_dict() for label in derived.labels], days=7.0)
    from_rows = summarise([row], list(derived.labels), days=7.0)
    assert from_dicts.as_dict() == from_rows.as_dict()


def test_the_operator_facing_commands_run_end_to_end(tmp_path, capsys):
    from localalexa.cli import main

    root = tmp_path / "collect"
    log = RequestLog(root / "requests.jsonl", session="s1")
    tap = CollectionTap(log, id_factory=lambda: uuid.uuid4().hex[:8])
    tap.observe("set a timer for five minutes", decide([0.70] + [0.3] * 17, load_registry().label_ids))
    tap.observe("hmm not sure what i meant", decide([0.40] + [0.39] * 17, load_registry().label_ids))
    log.close()

    assert main(["collect", "derive", "--root", str(root)]) == 0
    for name in ("labels.jsonl", "rejects.jsonl", "corrections.jsonl"):
        assert (root / name).is_file(), name
    stats = json.loads(capsys.readouterr().out)["stats"]
    assert stats["rows"] == 2
    assert stats["rows_below_gate"] == 1

    assert main(["collect", "summary", "--root", str(root)]) == 0
    text = capsys.readouterr().out
    assert "2 rows" in text
    assert "below the gate 1" in text
