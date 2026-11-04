"""Loaders for the routing registry and the 382-row evaluation set.

The registry holds 17 tool entries (id, family, entity, commit semantics, regex
patterns, reference utterances, templates, slot vocabularies) plus a
``none_route`` block of off-scaffold references. The evaluation set is 382
utterances generated from the registry's templates, 110 distinct phrasing
families, 18 labels including ``__none__``.
"""

from __future__ import annotations

import itertools
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("LOCALALEXA_DATA", REPO_ROOT / "data"))

NONE_LABEL = "__none__"
CHITCHAT_FAMILY = "__chitchat__"

EXPECTED_ROWS = 382
EXPECTED_TOOLS = 17
EXPECTED_CLASSES = 18
EXPECTED_PHRASING_FAMILIES = 110

ROW_KEYS = {"text", "tool", "family"}


class SchemaError(ValueError):
    """Raised when a data file does not match the documented schema."""


@dataclass(frozen=True)
class Tool:
    id: str
    family: str
    cls: str
    entity_domain: str
    commit: str
    patterns: tuple[str, ...]
    references: tuple[str, ...]
    templates: tuple[str, ...]
    slots: dict[str, tuple[str, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class Registry:
    embedding_model: str
    seed: int
    tools: tuple[Tool, ...]
    none_references: tuple[str, ...]

    @property
    def tool_ids(self) -> list[str]:
        return [t.id for t in self.tools]

    @property
    def label_ids(self) -> list[str]:
        """Every label a router may emit, none-route last."""
        return self.tool_ids + [NONE_LABEL]

    def tool(self, tool_id: str) -> Tool:
        for t in self.tools:
            if t.id == tool_id:
                return t
        raise KeyError(f"unknown tool id: {tool_id!r}")


@dataclass(frozen=True)
class Row:
    text: str
    tool: str
    family: str


def _read_json(path: Path) -> object:
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found; the package reads its data from <repo>/data "
            "(override with the LOCALALEXA_DATA environment variable)"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def load_registry(path: Path | None = None) -> Registry:
    raw = _read_json(Path(path) if path else DATA_DIR / "registry.json")
    if not isinstance(raw, dict):
        raise SchemaError("registry.json must be a JSON object")
    for key in ("embedding_model", "seed", "tools", "none_route"):
        if key not in raw:
            raise SchemaError(f"registry.json missing key {key!r}")
    tools = []
    for entry in raw["tools"]:
        tools.append(
            Tool(
                id=entry["id"],
                family=entry["family"],
                cls=entry["class"],
                entity_domain=entry["entity_domain"],
                commit=entry["commit"],
                patterns=tuple(entry.get("patterns", ())),
                references=tuple(entry.get("references", ())),
                templates=tuple(entry.get("templates", ())),
                slots={k: tuple(v) for k, v in entry.get("slots", {}).items()},
            )
        )
    if len(tools) != EXPECTED_TOOLS:
        raise SchemaError(f"expected {EXPECTED_TOOLS} tools, found {len(tools)}")
    return Registry(
        embedding_model=str(raw["embedding_model"]),
        seed=int(raw["seed"]),
        tools=tuple(tools),
        none_references=tuple(raw["none_route"]["references"]),
    )


def load_routing_set(path: Path | None = None) -> list[Row]:
    raw = _read_json(Path(path) if path else DATA_DIR / "routing_382.json")
    if not isinstance(raw, list):
        raise SchemaError("routing_382.json must be a JSON array of rows")
    rows = []
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict) or set(entry) != ROW_KEYS:
            raise SchemaError(f"row {i}: expected exactly the keys {sorted(ROW_KEYS)}")
        text, tool, family = entry["text"], entry["tool"], entry["family"]
        if not isinstance(text, str) or not text.strip():
            raise SchemaError(f"row {i}: empty text")
        if not isinstance(tool, str) or not tool:
            raise SchemaError(f"row {i}: missing tool label")
        if not isinstance(family, str) or not family:
            raise SchemaError(f"row {i}: missing family")
        rows.append(Row(text=text, tool=tool, family=family))
    return rows


def validate_routing_set(rows: list[Row], registry: Registry) -> None:
    """Cross-check the evaluation set against the registry it was generated from."""
    if len(rows) != EXPECTED_ROWS:
        raise SchemaError(f"expected {EXPECTED_ROWS} rows, found {len(rows)}")
    known = set(registry.label_ids)
    unknown = sorted({r.tool for r in rows} - known)
    if unknown:
        raise SchemaError(f"labels absent from the registry: {unknown}")
    classes = len({r.tool for r in rows})
    if classes != EXPECTED_CLASSES:
        raise SchemaError(f"expected {EXPECTED_CLASSES} classes, found {classes}")


def expand_template(tool: Tool, template: str) -> list[str]:
    """Every slot fill of one template, in slot-vocabulary product order."""
    keys = list(tool.slots)
    if not keys:
        return [template]
    out = []
    for combo in itertools.product(*[tool.slots[k] for k in keys]):
        text = template
        for key, value in zip(keys, combo):
            text = text.replace("{%s}" % key, value)
        out.append(text)
    return out


def template_index(registry: Registry) -> dict[str, tuple[str, str]]:
    """Map every renderable utterance back to its (tool id, template)."""
    index: dict[str, tuple[str, str]] = {}
    for tool in registry.tools:
        for template in tool.templates:
            for text in expand_template(tool, template):
                index.setdefault(text, (tool.id, template))
    return index


def phrasing_families(
    rows: list[Row], index: dict[str, tuple[str, str]]
) -> list[str]:
    """Phrasing family per row: the template that generated it.

    Off-scaffold rows are not template-generated and share one family, so the
    held-out split cannot leak their wording either.
    """
    return [index[r.text][1] if r.text in index else CHITCHAT_FAMILY for r in rows]


def training_pairs(registry: Registry) -> list[tuple[str, str]]:
    """The 200 reference utterances and their labels (none-route last)."""
    pairs = [(ref, tool.id) for tool in registry.tools for ref in tool.references]
    pairs.extend((ref, NONE_LABEL) for ref in registry.none_references)
    return pairs


def slotfill_references(
    registry: Registry, index: dict[str, tuple[str, str]]
) -> list[tuple[str, str, str]]:
    """References that are literal slot fills of a template: (text, label, template).

    These are the contaminated training rows — the generator can produce the
    same wording verbatim for the evaluation set.
    """
    out = []
    for text, label in training_pairs(registry):
        hit = index.get(text)
        if hit is not None:
            out.append((text, label, hit[1]))
    return out
