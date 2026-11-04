"""Tests for the data layer: row count, schema, labels, phrasing families."""

from __future__ import annotations

import json

import pytest

from localalexa.data import (
    CHITCHAT_FAMILY,
    EXPECTED_CLASSES,
    EXPECTED_PHRASING_FAMILIES,
    EXPECTED_ROWS,
    EXPECTED_TOOLS,
    NONE_LABEL,
    SchemaError,
    expand_template,
    load_registry,
    load_routing_set,
    phrasing_families,
    slotfill_references,
    template_index,
    training_pairs,
    validate_routing_set,
)


@pytest.fixture(scope="module")
def registry():
    return load_registry()


@pytest.fixture(scope="module")
def rows():
    return load_routing_set()


def test_routing_set_has_the_documented_row_count(rows):
    assert len(rows) == EXPECTED_ROWS == 382


def test_routing_set_schema_and_field_types(rows):
    for row in rows:
        assert isinstance(row.text, str) and row.text.strip()
        assert isinstance(row.tool, str) and row.tool
        assert isinstance(row.family, str) and row.family


def test_every_label_is_known_to_the_registry(registry, rows):
    known = set(registry.label_ids)
    assert {r.tool for r in rows} <= known
    assert len({r.tool for r in rows}) == EXPECTED_CLASSES
    validate_routing_set(rows, registry)


def test_registry_shape(registry):
    assert len(registry.tools) == EXPECTED_TOOLS == 17
    assert len(registry.label_ids) == EXPECTED_CLASSES
    assert len(registry.none_references) == 16
    assert len(training_pairs(registry)) == 200
    assert registry.tool_ids[-1] != NONE_LABEL
    assert registry.label_ids[-1] == NONE_LABEL


def test_tool_family_field_is_populated_and_collapses_to_seven_values(registry, rows):
    assert {t.family for t in registry.tools} == {
        "timers",
        "reminders",
        "lights",
        "climate",
        "knowledge",
        "shopping",
    }
    assert len({r.family for r in rows}) == 7


def test_template_index_recovers_the_phrasing_family_of_every_tool_row(registry, rows):
    index = template_index(registry)
    families = phrasing_families(rows, index)
    assert len(families) == len(rows)
    assert len(set(families)) == EXPECTED_PHRASING_FAMILIES == 110
    unmapped_tool_rows = [
        r.text for r, f in zip(rows, families) if r.tool != NONE_LABEL and f == CHITCHAT_FAMILY
    ]
    assert unmapped_tool_rows == []
    off_scaffold = [f for r, f in zip(rows, families) if r.tool == NONE_LABEL]
    assert set(off_scaffold) == {CHITCHAT_FAMILY}
    assert off_scaffold.count(CHITCHAT_FAMILY) == 25


def test_a_phrasing_family_never_spans_two_tools(registry, rows):
    index = template_index(registry)
    owner = {}
    for text, (tool_id, template) in index.items():
        assert owner.setdefault(template, tool_id) == tool_id
    for row in rows:
        if row.text in index:
            assert index[row.text][0] == row.tool


def test_slotfill_reference_count_is_the_measured_contamination(registry):
    index = template_index(registry)
    contaminated = slotfill_references(registry, index)
    assert len(contaminated) == 29
    assert all(label in registry.label_ids for _, label, _ in contaminated)
    assert all(text in index for text, _, _ in contaminated)
    templates = {template for _, template in index.values()}
    assert {template for _, _, template in contaminated} <= templates


def test_expand_template_covers_the_slot_product(registry):
    tool = registry.tool("timer_create")
    template = tool.templates[0]
    fills = expand_template(tool, template)
    expected = 1
    for values in tool.slots.values():
        expected *= len(values)
    assert len(fills) == expected
    assert all("{" not in fill for fill in fills)
    assert len(set(fills)) == len(fills)


def test_off_scaffold_rows_are_not_template_generated(registry, rows):
    index = template_index(registry)
    none_rows = [r for r in rows if r.tool == NONE_LABEL]
    assert len(none_rows) == 25
    assert [r.text for r in none_rows if r.text in index] == []


def test_malformed_rows_are_rejected(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([{"text": "hello", "label": "x"}]), encoding="utf-8")
    with pytest.raises(SchemaError):
        load_routing_set(bad)


def test_empty_text_is_rejected(tmp_path):
    bad = tmp_path / "empty.json"
    bad.write_text(json.dumps([{"text": "  ", "tool": "x", "family": "y"}]), encoding="utf-8")
    with pytest.raises(SchemaError):
        load_routing_set(bad)


def test_missing_file_names_the_data_directory(tmp_path):
    with pytest.raises(FileNotFoundError) as exc:
        load_routing_set(tmp_path / "absent.json")
    assert "LOCALALEXA_DATA" in str(exc.value)


def test_wrong_row_count_fails_validation(registry, rows):
    with pytest.raises(SchemaError):
        validate_routing_set(rows[:10], registry)


def test_unknown_label_fails_validation(registry, rows):
    from localalexa.data import Row

    broken = list(rows[:-1]) + [Row(text="something new", tool="teleport", family="x")]
    with pytest.raises(SchemaError):
        validate_routing_set(broken, registry)


def test_registry_missing_key_is_rejected(tmp_path):
    path = tmp_path / "registry.json"
    path.write_text(json.dumps({"tools": [], "none_route": {"references": []}}), encoding="utf-8")
    with pytest.raises(SchemaError):
        load_registry(path)
