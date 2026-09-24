"""Infobox extraction and pipeline orchestration (refs implementation.md
16.1/16.2).

**16.1 — extraction**: ported from `agentic-dbpedia`'s `DumpTemplateParser`.
`agentic-dbpedia`'s original parses a whole MediaWiki XML dump, streaming
`<page>` elements. Nothing here does that — the actual use case is a user
pasting one infobox's wikitext directly (`frontend/src/lib/api.ts::previewMapping`
takes a plain string, not a dump path) — so only the wikitext-level parsing
core is ported: `mwparserfromhell` when available, with the same
conservative brace-depth-aware fallback parser `agentic-dbpedia` uses when
it isn't, so behavior matches even on a host without the real parser.

**16.2 — orchestration**: ported from `agentic-dbpedia`'s 4-node
`MappingAgentState` graph shape (`week3_extract_parameters ->
week4_match_properties -> week5_format_syntax -> week6_validate_and_persist`),
wired to this repo's own pieces instead of `agentic-dbpedia`'s: 16.1's
`extract_first_infobox`, 11's `rag.predict.predict_property`, the existing
`generate_mapping_syntax_impl`/`build_mapping_wikitext`, and 14.1's
`create_review_item`. Same optional-`langgraph`-with-sequential-fallback
pattern as the original; the persist node is genuinely async (this repo's
DB layer is async SQLAlchemy throughout), and LangGraph's own `ainvoke()`
runs sync and async nodes together in one graph without any special
handling — verified directly before relying on it.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, TypedDict, cast

from sqlalchemy.ext.asyncio import AsyncSession

from db.session import create_review_item
from logging_config import log_event
from mcp_server.guardrails import is_injection_attempt
from mcp_server.publish import build_mapping_wikitext
from mcp_server.server import MappingEntry, MappingPayload, generate_mapping_syntax_impl
from rag.ontology import AmharicMappingIndex
from rag.predict import PredictionResult, predict_property

LOGGER = logging.getLogger("dbpedia_mapping_assistant.pipeline")

INFOBOX_MARKERS = ("infobox", "info box", "መረጃ", "ሳጥን")


@dataclass(frozen=True, slots=True)
class TemplateField:
    name: str
    value: str


@dataclass(frozen=True, slots=True)
class ExtractedTemplate:
    name: str
    fields: list[TemplateField]


def _is_infobox_like(template_name: str) -> bool:
    normalized = template_name.replace("_", " ").casefold()
    return any(marker in normalized for marker in INFOBOX_MARKERS)


def _split_top_level(body: str) -> list[str]:
    """Split a template body on top-level `|` only — one still inside a
    nested `{{...}}` doesn't count. Mirrors agentic-dbpedia's own fallback
    parser exactly, since the two must agree on edge cases."""

    parts: list[str] = []
    depth = 0
    start = 0
    index = 0

    while index < len(body):
        pair = body[index : index + 2]
        if pair == "{{":
            depth += 1
            index += 2
            continue
        if pair == "}}" and depth:
            depth -= 1
            index += 2
            continue
        if body[index] == "|" and depth == 0:
            parts.append(body[start:index])
            start = index + 1
        index += 1

    parts.append(body[start:])
    return parts


def _extract_top_level_template_bodies(text: str) -> list[str]:
    bodies: list[str] = []
    index = 0
    while True:
        open_index = text.find("{{", index)
        if open_index == -1:
            break
        depth = 1
        cursor = open_index + 2
        while cursor < len(text) and depth:
            pair = text[cursor : cursor + 2]
            if pair == "{{":
                depth += 1
                cursor += 2
                continue
            if pair == "}}":
                depth -= 1
                cursor += 2
                continue
            cursor += 1
        bodies.append(text[open_index + 2 : cursor - 2])
        index = cursor
    return bodies


def _parse_with_fallback(text: str) -> list[ExtractedTemplate]:
    templates: list[ExtractedTemplate] = []

    for body in _extract_top_level_template_bodies(text):
        parts = _split_top_level(body)
        if not parts:
            continue

        template_name = parts[0].strip()
        fields: list[TemplateField] = []

        for raw_part in parts[1:]:
            if "=" not in raw_part:
                continue
            name, value = raw_part.split("=", 1)
            name = name.strip()
            if not name or name.isdecimal():
                continue
            fields.append(TemplateField(name=name, value=value.strip()))

        if template_name and fields:
            templates.append(ExtractedTemplate(name=template_name, fields=fields))

    return templates


def _parse_with_mwparser(text: str) -> list[ExtractedTemplate]:
    import mwparserfromhell

    wikicode = mwparserfromhell.parse(text)
    templates: list[ExtractedTemplate] = []

    for raw_template in wikicode.filter_templates(recursive=False):
        template_name = str(raw_template.name).strip()
        fields: list[TemplateField] = []

        for raw_param in raw_template.params:
            name = str(raw_param.name).strip()
            if not name or name.isdecimal():
                continue
            fields.append(TemplateField(name=name, value=str(raw_param.value).strip()))

        if fields:
            templates.append(ExtractedTemplate(name=template_name, fields=fields))

    return templates


def parse_templates(text: str) -> list[ExtractedTemplate]:
    """Parse every template in `text`, using `mwparserfromhell` when
    installed and the conservative fallback parser otherwise — same
    fallback behavior as `agentic-dbpedia`'s own `DumpTemplateParser`."""

    try:
        templates = _parse_with_mwparser(text)
    except ModuleNotFoundError:
        log_event(LOGGER, "pipeline.parser_fallback")
        templates = _parse_with_fallback(text)
    return templates


def extract_first_infobox(
    wikitext: str, *, mapping_index: AmharicMappingIndex | None = None
) -> ExtractedTemplate | None:
    """The first infobox-like template found in `wikitext`, name and all —
    `extract_infobox` (16.1's original public function) only returns the
    fields, which was enough for that milestone but not for 16.2's
    orchestration, which also needs the template name itself for the
    review item and the published page title.

    A template counts as infobox-like if either `_is_infobox_like`'s
    marker-word heuristic matches its name, OR `mapping_index` (when given)
    already has a real `Mapping am:<name>` page for it — confirmed live
    that the heuristic alone misses real, already-mapped templates whose
    Amharic name doesn't happen to contain "infobox"/"መረጃ"/"ሳጥን" (e.g. the
    Place template, "የቦታ ስም"), which silently dropped every Place infobox
    a user pasted before this second signal was added."""

    def _counts_as_infobox(name: str) -> bool:
        if _is_infobox_like(name):
            return True
        return mapping_index is not None and mapping_index.is_known_template(name)

    infobox_templates = [t for t in parse_templates(wikitext) if _counts_as_infobox(t.name)]
    if not infobox_templates:
        log_event(LOGGER, "pipeline.no_infobox_found")
        return None

    log_event(LOGGER, "pipeline.infobox_extracted", template_name=infobox_templates[0].name)
    return infobox_templates[0]


def extract_infobox(
    wikitext: str, *, mapping_index: AmharicMappingIndex | None = None
) -> list[TemplateField]:
    """Extract the fields of the first infobox-like template found in
    `wikitext`. Returns an empty list if none is found — never raises just
    because the input isn't a recognizable infobox."""

    template = extract_first_infobox(wikitext, mapping_index=mapping_index)
    return template.fields if template is not None else []


try:  # pragma: no cover - exercised when the optional dependency is installed.
    _langgraph_graph = importlib.import_module("langgraph.graph")
except ModuleNotFoundError:  # pragma: no cover - fallback is covered in local tests.
    END = "__end__"
    StateGraph: Any | None = None
else:  # pragma: no cover - depends on optional dependency availability.
    END = _langgraph_graph.END
    StateGraph = _langgraph_graph.StateGraph


class PipelineState(TypedDict, total=False):
    wikitext: str
    domain_class: str
    run_id: str
    session: AsyncSession
    mapping_index: AmharicMappingIndex
    template_name: str
    fields: list[TemplateField]
    predictions: dict[str, PredictionResult]
    mappings: list[dict[str, Any]]
    mapping_wikitext: str
    xml_rules: str
    review_item_id: str | None
    warnings: list[str]
    unmapped_fields: list[dict[str, str]]


def _extract_node(state: PipelineState) -> dict[str, Any]:
    # Loaded before extraction (not after, as this used to be ordered) so
    # `extract_first_infobox` can consult the corpus's own known-template
    # names as a second infobox-recognition signal, alongside the
    # marker-word heuristic -- see that function's docstring.
    state_mapping_index = state.get("mapping_index")
    mapping_index = (
        state_mapping_index
        if state_mapping_index is not None
        else AmharicMappingIndex.from_default_cache()
    )

    template = extract_first_infobox(state["wikitext"], mapping_index=mapping_index)
    warnings = list(state.get("warnings", []))
    updates: dict[str, Any] = {}

    if template is None:
        warnings.append("No infobox-like template found in the given wikitext.")
        return {"template_name": "", "fields": [], "warnings": warnings}

    # Fill in a domain class from the corpus's own `mapToClass` when the
    # caller didn't give one (the Wikipedia-link preview has no
    # target-class input at all, and a bare pasted infobox often won't
    # either). "Thing" is the HTTP layer's own default sentinel for
    # "unspecified". An explicit caller-provided class always wins.
    current_class = state.get("domain_class")
    if current_class in (None, "", "Thing"):
        detected = mapping_index.domain_class_for_template(template.name)
        if detected is not None:
            updates["domain_class"] = detected
            warnings.append(
                f"No target class was given -- using {detected!r} from the published "
                f"mapping for template {template.name!r}."
            )

    kept_fields: list[TemplateField] = []
    skipped = 0
    for field in template.fields:
        # Scoped to this exact template -- not AmharicMappingIndex.lookup(),
        # which matches a field name globally across the whole corpus and
        # confirmed live to silently drop genuinely new fields on unmapped
        # templates (Dam's own "ከፍታ") just because an unrelated template
        # (Place) happens to reuse the same Amharic word for a different
        # property. See AmharicMappingIndex's own docstring for the story.
        if mapping_index.is_already_mapped_on_template(template.name, field.name) is not None:
            skipped += 1
            continue
        kept_fields.append(field)

    if skipped:
        warnings.append(
            f"Skipped {skipped} field(s) already present in the published Amharic mappings."
        )
    if not kept_fields:
        warnings.append("No unmapped fields remained after filtering already-published mappings.")

    return {
        "template_name": template.name,
        "fields": kept_fields,
        "warnings": warnings,
        **updates,
    }


async def _predict_node(state: PipelineState) -> dict[str, Any]:
    """Predict a property for every extracted field.

    `predict_property` is a blocking call (retrieval + an optional
    synchronous LLM rerank over HTTP to Ollama) — run one directly inside
    this `async` node the way the sequential-fallback graph does with a
    sync node, and it blocks the *entire* event loop for as long as it
    takes: no other request (a different browser tab, a health check) can
    be served while it's in flight. `asyncio.to_thread` moves each call
    off the loop so the rest of the server stays responsive.

    Deliberately still sequential across fields, not `asyncio.gather`ed —
    tried that first and it deadlocks (confirmed live): DSPy's
    `dspy.context(lm=lm)` mutates global/thread-shared state per call, and
    two `predict_property` calls racing that from separate
    `asyncio.to_thread` worker threads hang rather than raising. One
    field's LLM rerank still calling out to a real endpoint is a genuinely
    rare, small cost per field (and free entirely once a field's retrieval
    finds no candidates at all, or once rag.predict's own circuit breaker
    has already opened) — not worth reintroducing a hang to shave off."""

    domain_class = state.get("domain_class")
    warnings = list(state.get("warnings", []))
    predictions: dict[str, PredictionResult] = {}
    unmapped_fields: list[dict[str, str]] = []

    for field in state.get("fields", []):
        if is_injection_attempt(field.name):
            warnings.append(f"Rejected suspicious field {field.name!r} as prompt injection.")
            continue
        outcome = await asyncio.to_thread(predict_property, field.name, target_class=domain_class)
        if isinstance(outcome, PredictionResult):
            predictions[field.name] = outcome
        else:
            warnings.append(f"No retrieval candidates found for {field.name!r}.")
            # Carried through to the SSE result event so the frontend can
            # offer an in-chat "search and assign this one yourself" step
            # rather than the field just vanishing into a warning line.
            unmapped_fields.append({"name": field.name, "value": field.value[:80]})

    return {
        "predictions": predictions,
        "warnings": warnings,
        "unmapped_fields": unmapped_fields,
    }


def _format_node(state: PipelineState) -> dict[str, Any]:
    domain_class = state.get("domain_class") or "Thing"
    predictions = state.get("predictions", {})
    warnings = list(state.get("warnings", []))

    mappings: list[dict[str, Any]] = []
    for name, prediction in predictions.items():
        if prediction.llm_proposed:
            # No retrieval score to report -- a fixed low-but-not-zero
            # confidence so the row reads as "weak guess, check this" rather
            # than "0%, definitely wrong".
            confidence = 0.3
            source = "llm"
            warnings.append(
                f"Proposed {prediction.property!r} for {name!r} with the language model "
                f"(no retrieval match) -- verify before approving."
            )
        else:
            confidence = (
                prediction.top_retrieval_result.score if prediction.top_retrieval_result else 0.0
            )
            source = "retrieval"
        mappings.append(
            {
                "templateProperty": name,
                "ontologyProperty": prediction.property,
                "confidence": confidence,
                "source": source,
            }
        )

    if not mappings:
        warnings.append("No mappings were predicted; nothing to format.")
        return {"mappings": [], "mapping_wikitext": "", "xml_rules": "", "warnings": warnings}

    try:
        payload = MappingPayload(
            domain_class=domain_class,
            mappings=[
                MappingEntry(
                    templateProperty=mapping["templateProperty"],
                    ontologyProperty=mapping["ontologyProperty"],
                )
                for mapping in mappings
            ],
        )
        xml_rules = generate_mapping_syntax_impl(payload)
    except Exception as exc:
        warnings.append(f"Could not generate deterministic XML: {exc.__class__.__name__}")
        xml_rules = ""

    mapping_wikitext = build_mapping_wikitext(domain_class, mappings)

    return {
        "mappings": mappings,
        "mapping_wikitext": mapping_wikitext,
        "xml_rules": xml_rules,
        "warnings": warnings,
    }


async def _persist_node(state: PipelineState) -> dict[str, Any]:
    mappings = state.get("mappings", [])
    if not mappings:
        return {"review_item_id": None}

    session = state["session"]
    item = await create_review_item(
        session,
        run_id=state.get("run_id", ""),
        template_name=state.get("template_name", ""),
        domain_class=state.get("domain_class") or "Thing",
        mappings=mappings,
    )
    log_event(LOGGER, "pipeline.review_item_created", review_id=item.id)
    return {"review_item_id": item.id}


PipelineNode = Callable[[PipelineState], dict[str, Any] | Awaitable[dict[str, Any]]]

# Shared by _build_graph() and stream_mapping_pipeline() (16.3's SSE
# endpoint) so both the LangGraph path and the sequential/no-streaming
# fallback report the exact same node names to a caller either way.
NODE_SEQUENCE: list[tuple[str, PipelineNode]] = [
    ("extract_infobox_fields", _extract_node),
    ("predict_properties", _predict_node),
    ("format_mapping_syntax", _format_node),
    ("persist_review_item", _persist_node),
]

NODE_LABELS: dict[str, str] = {
    "extract_infobox_fields": "Extracting infobox fields",
    "predict_properties": "Predicting ontology properties",
    "format_mapping_syntax": "Generating mapping XML",
    "persist_review_item": "Saving to review queue",
}


class _SequentialPipelineGraph:
    """Async-aware sequential fallback for when `langgraph` isn't
    installed — same shape as `agentic-dbpedia`'s own
    `_SequentialMappingAgentGraph`, extended to await async nodes too
    (this pipeline's persist step genuinely needs one)."""

    def __init__(self, nodes: Sequence[tuple[str, PipelineNode]]) -> None:
        self._nodes = nodes

    async def ainvoke(self, initial_state: PipelineState) -> PipelineState:
        state: dict[str, Any] = dict(initial_state)
        for _name, node in self._nodes:
            updates = node(cast(PipelineState, state))
            if isinstance(updates, Awaitable):
                updates = await updates
            state.update(updates)
        return cast(PipelineState, state)


def _build_graph() -> Any:
    if StateGraph is None:
        return _SequentialPipelineGraph(NODE_SEQUENCE)

    workflow = StateGraph(PipelineState)
    for name, node in NODE_SEQUENCE:
        workflow.add_node(name, node)
    workflow.set_entry_point(NODE_SEQUENCE[0][0])
    for (from_name, _), (to_name, _) in zip(NODE_SEQUENCE, NODE_SEQUENCE[1:], strict=False):
        workflow.add_edge(from_name, to_name)
    workflow.add_edge(NODE_SEQUENCE[-1][0], END)
    return workflow.compile()


@dataclass(frozen=True)
class PipelineResult:
    run_id: str
    template_name: str
    domain_class: str
    mappings: list[dict[str, Any]]
    mapping_wikitext: str
    xml_rules: str
    warnings: list[str]
    unmapped_fields: list[dict[str, str]]
    review_item_id: str | None


async def run_mapping_pipeline(
    wikitext: str,
    *,
    domain_class: str,
    session: AsyncSession,
    run_id: str | None = None,
    mapping_index: AmharicMappingIndex | None = None,
) -> PipelineResult:
    """Extract -> predict -> format -> persist, end to end.

    Ported from `agentic-dbpedia`'s 4-node `week3_extract_parameters ->
    week4_match_properties -> week5_format_syntax ->
    week6_validate_and_persist` graph shape, wired to this repo's own
    pieces. `session` is a live `AsyncSession` (14.1) the caller already
    holds — this function never opens or closes one itself, matching how
    `mcp_server/http_app.py`'s route handlers already manage sessions.
    """

    resolved_run_id = run_id or str(uuid.uuid4())
    initial_state = _build_initial_state(
        wikitext,
        domain_class=domain_class,
        session=session,
        run_id=resolved_run_id,
        mapping_index=mapping_index,
    )

    graph = _build_graph()
    final_state = await graph.ainvoke(initial_state)

    log_event(
        LOGGER,
        "pipeline.completed",
        run_id=resolved_run_id,
        mapping_count=len(final_state.get("mappings", [])),
    )

    return PipelineResult(
        run_id=resolved_run_id,
        template_name=final_state.get("template_name", ""),
        # `_extract_node` may have auto-derived this from the template's
        # own `mapToClass` when the caller passed the "Thing" default.
        domain_class=final_state.get("domain_class") or domain_class,
        mappings=final_state.get("mappings", []),
        mapping_wikitext=final_state.get("mapping_wikitext", ""),
        xml_rules=final_state.get("xml_rules", ""),
        warnings=final_state.get("warnings", []),
        unmapped_fields=final_state.get("unmapped_fields", []),
        review_item_id=final_state.get("review_item_id"),
    )


def _build_initial_state(
    wikitext: str,
    *,
    domain_class: str,
    session: AsyncSession,
    run_id: str,
    mapping_index: AmharicMappingIndex | None,
) -> PipelineState:
    return {
        "wikitext": wikitext,
        "domain_class": domain_class,
        "run_id": run_id,
        "session": session,
        # `mapping_index or ...` would be wrong here: AmharicMappingIndex
        # defines __len__, so a deliberately empty (but valid) index is
        # falsy and would get silently replaced by the default cache.
        "mapping_index": (
            mapping_index if mapping_index is not None else AmharicMappingIndex.from_default_cache()
        ),
        "warnings": [],
    }


async def stream_mapping_pipeline(
    wikitext: str,
    *,
    domain_class: str,
    session: AsyncSession,
    run_id: str | None = None,
    mapping_index: AmharicMappingIndex | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """The same extract -> predict -> format -> persist pipeline as
    `run_mapping_pipeline`, yielding one event per completed node instead
    of just the final result — for `mcp_server/http_app.py`'s SSE
    `POST /v1/preview` (16.3).

    Each yielded dict matches `frontend/src/lib/types.ts::AgentStep`
    (`{node, status, detail, timestamp}`) until the final event, which
    matches `frontend/src/lib/api.ts::previewMapping`'s `PreviewEvent`
    "result" branch (`{node: "result", mappings}`) exactly.

    Uses the compiled LangGraph's own `astream()` when `langgraph` is
    installed (yields `{node_name: state_delta}` per step — verified
    directly, not assumed, before relying on it) and a manual per-node
    loop otherwise, so both paths report identical node names via the
    shared `NODE_SEQUENCE`.
    """

    resolved_run_id = run_id or str(uuid.uuid4())
    initial_state = _build_initial_state(
        wikitext,
        domain_class=domain_class,
        session=session,
        run_id=resolved_run_id,
        mapping_index=mapping_index,
    )

    state: dict[str, Any] = dict(initial_state)

    if StateGraph is not None:
        graph = _build_graph()
        async for chunk in graph.astream(initial_state):
            for node_name, delta in chunk.items():
                state.update(delta)
                yield _step_event(node_name)
    else:
        for node_name, node in NODE_SEQUENCE:
            updates = node(cast(PipelineState, state))
            if isinstance(updates, Awaitable):
                updates = await updates
            state.update(updates)
            yield _step_event(node_name)

    log_event(
        LOGGER,
        "pipeline.stream_completed",
        run_id=resolved_run_id,
        mapping_count=len(state.get("mappings", [])),
    )
    # `_format_node` (format_mapping_syntax) already computes real,
    # deterministic MediaWiki mapping XML/wikitext for exactly this result
    # -- it just never left this function before: this was the only place
    # in the whole stack a caller could still reach it (ReviewItem never
    # stores it either, by design -- it's cheaply regenerable from
    # `mappings`/`domain_class` at any time, not data worth duplicating in
    # Postgres). Confirmed live: without this, the "Generating mapping
    # XML" step reported "done" while producing something no caller could
    # ever see. camelCase to match frontend/src/lib/types.ts's convention
    # for every other field this event already carries.
    yield {
        "node": "result",
        "mappings": state.get("mappings", []),
        "mappingWikitext": state.get("mapping_wikitext", ""),
        "xmlRules": state.get("xml_rules", ""),
        # `_persist_node` already computes this (`None` whenever `mappings`
        # ended up empty -- nothing was written) but, like mapping_wikitext
        # above, it never left this function before: a caller had no way
        # to approve/reject the very row this run just created without a
        # separate `GET /v1/reviews` round trip to find its id.
        "reviewItemId": state.get("review_item_id"),
        # Fields retrieval (and the propose-from-scratch LLM fallback)
        # couldn't map -- `{name, value}` each, so the frontend can offer an
        # in-chat "search and assign this one" step instead of the field
        # only showing up as a warning line.
        "unmappedFields": state.get("unmapped_fields", []),
        # Every node's own notes -- fields skipped as already-published,
        # fields with no confident retrieval match, an auto-derived domain
        # class. Confirmed live that without surfacing these, a country
        # infobox whose fields are mostly already mapped looks broken
        # ("only 2 of 28 mapped?") rather than correct ("22 were already
        # published, 4 had no clean property").
        "warnings": state.get("warnings", []),
    }


def _step_event(node_name: str) -> dict[str, Any]:
    return {
        "node": node_name,
        "status": "done",
        "detail": NODE_LABELS.get(node_name, node_name),
        "timestamp": datetime.now(UTC).isoformat(),
    }
