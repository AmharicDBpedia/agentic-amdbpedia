"""Aliases learned from human-approved review decisions.

The retriever behaves largely like a curated-alias lookup: on the
novel-input experiment (evaluation/novel_inputs/report.md) it answered 21 of
23 infobox fields whose exact Amharic string was already known and 1 of 43
it had never seen. Every approved review item is exactly the evidence that
closes that gap, so each confirmed `templateProperty -> ontologyProperty`
pair is stored here, scoped to its domain class, and consulted *before*
retrieval by `rag.predict.predict_property`.

A plain JSON file (`data/learned_aliases.json`, git-ignored like the
training log), not part of the embedded corpus: a lookup needs no index
rebuild, so an approval takes effect on the very next request.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from logging_config import log_event

LOGGER = logging.getLogger("dbpedia_mapping_assistant.learned_aliases")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEARNED_ALIASES_PATH = PROJECT_ROOT / "data" / "learned_aliases.json"
FILE_VERSION = 1

_LOCK = threading.Lock()
_CACHE: dict[Path, tuple[float, dict[tuple[str, str], str]]] = {}


def normalize_field(name: str) -> str:
    """Spaces, underscores and case don't distinguish infobox field names."""

    return "_".join(name.replace("_", " ").casefold().split())


def _normalize_class(name: str | None) -> str:
    return (name or "").strip().casefold()


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log_event(LOGGER, "learned_aliases.unreadable", path=str(path), error=str(exc))
        return []
    rows = data.get("aliases", []) if isinstance(data, dict) else []
    return [row for row in rows if isinstance(row, dict)]


def _index(path: Path) -> dict[tuple[str, str], str]:
    """`{(class, field): ontology_property}`, re-read only when the file changes."""

    try:
        mtime = path.stat().st_mtime
    except OSError:
        return {}
    cached = _CACHE.get(path)
    if cached and cached[0] == mtime:
        return cached[1]
    index = {
        (_normalize_class(row.get("class")), normalize_field(str(row.get("amharic", "")))): str(
            row["ontology_property"]
        )
        for row in _read_rows(path)
        if row.get("amharic") and row.get("ontology_property")
    }
    _CACHE[path] = (mtime, index)
    return index


def lookup(
    field_name: str, target_class: str | None, *, path: Path = DEFAULT_LEARNED_ALIASES_PATH
) -> str | None:
    """The human-confirmed property for this field, or `None`.

    With a target class, only an alias learned for that same class counts
    (the same Amharic phrase can legitimately mean different properties on
    different infoboxes). Without one, an alias matches only if every class
    it was learned for agrees on the property.
    """

    with _LOCK:
        index = _index(path)
    field = normalize_field(field_name)
    if target_class:
        return index.get((_normalize_class(target_class), field))
    found = {prop for (_cls, alias), prop in index.items() if alias == field}
    return found.pop() if len(found) == 1 else None


def record(
    domain_class: str,
    mappings: Iterable[Mapping[str, Any]],
    *,
    review_id: str | None = None,
    path: Path = DEFAULT_LEARNED_ALIASES_PATH,
) -> int:
    """Store approved mappings; a newer decision for the same field+class wins.

    Returns how many rows were added or changed.
    """

    now = datetime.now(UTC).isoformat()
    changed = 0
    with _LOCK:
        rows = _read_rows(path)
        by_key = {
            (_normalize_class(row.get("class")), normalize_field(str(row.get("amharic", "")))): row
            for row in rows
        }
        for mapping in mappings:
            field = str(mapping.get("templateProperty", "")).strip()
            prop = str(mapping.get("ontologyProperty", "")).strip()
            if not field or not prop:
                continue
            key = (_normalize_class(domain_class), normalize_field(field))
            existing = by_key.get(key)
            if existing and existing.get("ontology_property") == prop:
                continue
            by_key[key] = {
                "class": domain_class,
                "amharic": field,
                "ontology_property": prop,
                "review_id": review_id,
                "learned_at": now,
            }
            changed += 1
        if changed:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"version": FILE_VERSION, "aliases": list(by_key.values())}
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(path)
            _CACHE.pop(path, None)
    if changed:
        log_event(LOGGER, "learned_aliases.recorded", count=changed, domain_class=domain_class)
    return changed
