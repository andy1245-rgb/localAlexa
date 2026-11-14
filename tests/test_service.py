"""Tests for the real request path: one utterance in, one response out, logged.

The two the brief asks for are ``test_a_request_produces_a_row`` and
``test_a_correction_produces_the_corrected_by_link``. The rest pin the properties
that make those two worth having: the row survives a restart, the link is found
from the log rather than from memory, a logging failure does not fail the request,
and settling stays flat as the log grows.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from localalexa.collection.collector import Collector
from localalexa.collection.records import RequestRecord
from localalexa.data import load_registry
from localalexa.router import Choice, decide
from localalexa.service import Assistant

REGISTRY = load_registry()


def stub_choice(top: str, second: str, top_sim: float = 0.70, second_sim: float = 0.62) -> Choice:
    labels = REGISTRY.label_ids
    sims = [0.30] * len(labels)
    sims[labels.index(top)] = top_sim
    sims[labels.index(second)] = second_sim
    return decide(sims, labels)


def fixed(choice: Choice):
    return lambda text: choice


def read(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.fixture()
def assistant(tmp_path):
    made = []

    def build(**kwargs):
        kwargs.setdefault("predictor", fixed(stub_choice("light_set", "light_query")))
        instance = Assistant(REGISTRY, log=tmp_path / "requests.jsonl", **kwargs)
        made.append(instance)
        return instance

    yield build, tmp_path
    for instance in made:
        instance.close()


def test_a_request_produces_a_row(assistant):
    build, root = assistant
    watcher = build()

    response = watcher.handle("make the lamps brighter")

    rows = read(root / "requests.jsonl")
    assert len(rows) == 1
    row = rows[0]
    assert row["text"] == "make the lamps brighter"
    assert row["label"] == "light_set"
    assert row["tier"] == 1
    assert [c["label"] for c in row["candidates"]][:2] == ["light_set", "light_query"]
    assert row["id"] == response.record_id  # the response carries the row it wrote
    assert response.label == "light_set"
    assert not response.escalated


def test_a_correction_produces_the_corrected_by_link(assistant):
    build, root = assistant
    watcher = build()

    first = watcher.handle("make the lamps brighter")
    second = watcher.handle("no make the lamps dimmer")

    corrections = read(root / "corrections.jsonl")
    assert len(corrections) == 1
    assert corrections[0]["earlier_id"] == first.record_id
    assert corrections[0]["later_id"] == second.record_id
    assert corrections[0]["evidence"]["branch"] == "cue"

    events = read(root / "labels.jsonl")
    negative = [row for row in events if row["request_id"] == first.record_id and row["polarity"] < 0]
    assert len(negative) == 1
    assert negative[0]["source"] == "correction"
    assert negative[0]["strength"] == "strong"
    assert negative[0]["detail"]["corrected_by"] == second.record_id


def test_the_link_survives_a_restart_because_it_comes_from_the_log(assistant):
    build, root = assistant
    first_process = build()
    first = first_process.handle("make the lamps brighter")
    first_process.close()

    # A brand-new assistant, with no memory of the first request, finds it by
    # reading the log — which is the point of settling off the logged rows.
    second_process = build(settle=False)
    second = second_process.handle("no make the lamps dimmer")
    second_process.settle()
    second_process.close()

    corrections = read(root / "corrections.jsonl")
    assert [(row["earlier_id"], row["later_id"]) for row in corrections] == [
        (first.record_id, second.record_id)
    ]


def test_a_logging_failure_does_not_fail_the_request(assistant, monkeypatch):
    build, _ = assistant
    watcher = build()

    def broken(fd, data):
        raise OSError("disk full")

    monkeypatch.setattr(os, "write", broken)
    response = watcher.handle("make the lamps brighter")
    assert response.label == "light_set"
    assert watcher.collector.errors >= 1


def test_a_regex_hit_is_answered_without_calling_the_router(assistant):
    build, root = assistant
    calls = []

    def predictor(text):
        calls.append(text)
        return stub_choice("light_set", "light_query")

    watcher = build(predictor=predictor)
    response = watcher.handle("set a timer for five minutes")

    assert calls == []
    assert response.tier == 0
    assert response.label == "timer_create"
    row = read(root / "requests.jsonl")[0]
    assert row["tier"] == 0 and row["label"] == "timer_create"


def test_an_escalation_lands_in_the_reject_pile(assistant):
    build, root = assistant
    watcher = build(predictor=fixed(stub_choice("light_set", "light_query", 0.36, 0.35)))

    response = watcher.handle("hmm what about the thing")

    assert response.escalated and response.label is None
    rejects = read(root / "rejects.jsonl")
    assert len(rejects) == 1
    assert rejects[0]["id"] == response.record_id
    assert "label" not in rejects[0]  # the pile carries no interpretation


def test_a_wide_margin_is_auto_accepted_from_the_logged_row(assistant):
    build, root = assistant
    watcher = build(predictor=fixed(stub_choice("thermostat_set", "light_set", 0.80, 0.40)))

    response = watcher.handle("make the lamps brighter")

    events = read(root / "labels.jsonl")
    auto = [row for row in events if row["request_id"] == response.record_id]
    assert len(auto) == 1
    assert auto[0]["source"] == "auto_accept_margin"
    assert auto[0]["strength"] == "auto"
    assert auto[0]["polarity"] == 1


def test_an_effect_reported_after_the_fact_becomes_a_weak_positive(assistant):
    build, root = assistant
    watcher = build(settle=False)
    response = watcher.handle("make the lamps brighter")

    watcher.append_effect(response.record_id, True, "light_state", "on")
    watcher.settle()

    events = read(root / "labels.jsonl")
    effects = [row for row in events if row["source"] == "effect"]
    assert len(effects) == 1
    assert effects[0]["request_id"] == response.record_id
    assert effects[0]["strength"] == "weak"


def test_settle_stays_flat_as_the_log_grows(tmp_path):
    """The first implementation rebuilt the derived files per settle: 15.6 ms at 500 rows."""
    log_path = tmp_path / "requests.jsonl"
    collector = Collector(log_path, registry=REGISTRY)
    now = time.time()
    for i in range(300):
        collector.log.append(
            RequestRecord(
                id=f"r{i:04d}",
                ts=now - (300 - i) * 5,
                session="s",
                text=f"turn on the kitchen lights {i}",
                tier=1,
                label="light_set",
                confidence=0.6,
                top_sim=0.55,
                margin=0.09,
                candidates=(),
                gate=False,
                sim_threshold=0.45,
                conf_threshold=0.35,
            )
        )
    collector.close()

    collector = Collector(log_path, registry=REGISTRY, tail_rows=300)
    assert len(collector.records) == 300
    samples = []
    for i in range(5):
        collector.record(f"another utterance {i}", stub_choice("light_set", "light_query", 0.80, 0.40))
        started = time.perf_counter()
        collector.settle()
        samples.append((time.perf_counter() - started) * 1000.0)
    collector.close()
    samples.sort()
    assert samples[len(samples) // 2] < 5.0, f"settle median {samples[len(samples) // 2]:.2f} ms at 300 rows"


def test_records_are_not_re_derived_twice(assistant):
    build, root = assistant
    watcher = build(predictor=fixed(stub_choice("thermostat_set", "light_set", 0.80, 0.40)))
    watcher.handle("make the lamps brighter")
    watcher.settle()
    watcher.settle()
    events = read(root / "labels.jsonl")
    assert len([row for row in events if row["source"] == "auto_accept_margin"]) == 1


def test_collection_can_be_turned_off(assistant):
    build, root = assistant
    watcher = build(record_requests=False)
    response = watcher.handle("make the lamps brighter")
    assert response.label == "light_set"
    assert read(root / "requests.jsonl") == []
    assert watcher.settle() is None
