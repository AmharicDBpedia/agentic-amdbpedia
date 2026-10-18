"""HTTP layer for the review queue and mapping pipeline (refs
implementation.md 14.1/14.2/16.3).

A separate Starlette ASGI app from `mcp_server/server.py`'s stdio-based
FastMCP tool interface — this is the general HTTP surface
`frontend/src/lib/api.ts` and `agentic-dbpedia`'s pipeline actually call.
Run it with `uvicorn mcp_server.http_app:create_app --factory` (refs the
`create_app` docstring below for why this is a factory, not a bare
module-level `app` object) -- or `just run-http` (`scripts/run_http.py`),
which does the same thing but also falls back to the next free port if
the preferred one is already taken by a stale server process.

Every response the frontend reads matches `frontend/src/lib/types.ts`'s
shapes field-for-field (camelCase) via `ReviewItem.to_api_dict()` — not
just "close enough". `POST /v1/preview`'s SSE events match
`frontend/src/lib/api.ts::previewMapping`'s `PreviewEvent` type the same
way (refs 16.3) — this is where every `PLANNED` label in that file's own
doc comments becomes `EXISTING`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from config import Settings
from db.models import ReviewItem
from db.session import (
    InvalidReviewStatusError,
    ReviewNotFoundError,
    coverage_stats,
    create_engine,
    create_review_item,
    get_review_item,
    init_models,
    list_review_items,
    session_factory,
    set_review_status,
)
from errors import AssistantValidationError, ClientSafeError
from logging_config import correlation_context, get_correlation_id, log_event
from mcp_server import assistant
from mcp_server.consent import ConsentRequiredError, require_consent
from mcp_server.pipeline import stream_mapping_pipeline
from mcp_server.publish import PublishError, publish_mapping
from mcp_server.server import MAX_PROPERTY_LENGTH, find_semantic_match_impl
from mcp_server.wiki_fetch import (
    WikipediaFetchError,
    extract_template_params,
    fetch_article_wikitext,
    is_template_title,
    synthesize_infobox_wikitext,
    template_display_name,
)
from rag import learned_aliases
from rag.predict import PredictionResult, predict_property, production_predict_options
from rag.retrieval import NoMatchFound
from rag.retrieval import search as default_search
from rag.training_log import DEFAULT_LOG_PATH, log_decision
from scripts.refresh_wiki_cache import refresh_mappings

LOGGER = logging.getLogger("dbpedia_mapping_assistant.http")

REQUIRED_CREATE_FIELDS = ("run_id", "template_name", "domain_class", "mappings")
# frontend/src/lib/api.ts::decideReview's "approved"/"rejected" ->
# db.models.REVIEW_STATUSES.
DECISION_TO_STATUS = {"approved": "approved", "rejected": "rejected"}
# generate_mapping_syntax_impl/MappingPayload.domain_class requires
# ^[A-Z][A-Za-z0-9]*$ -- owl:Thing is the natural default when a preview
# request doesn't name a specific class.
DEFAULT_PREVIEW_DOMAIN_CLASS = "Thing"
MAX_ASSIST_TEXT_LENGTH = 20_000


def _mediawiki_base_url() -> str:
    try:
        return Settings().mediawiki_base_url
    except Exception:  # noqa: BLE001 - only used to build a link
        from config import DEFAULT_MEDIAWIKI_BASE_URL

        return DEFAULT_MEDIAWIKI_BASE_URL


PublishFunc = Callable[..., str]
RefreshMappingsFunc = Callable[..., int]
SearchFunc = Callable[..., Any]
FetchArticleFunc = Callable[..., tuple[str, str]]


def _error_response(error: ClientSafeError, status_code: int) -> JSONResponse:
    payload = error.to_payload()
    payload["correlation_id"] = get_correlation_id()
    return JSONResponse(payload, status_code=status_code)


def _session_factory_from_state(request: Request) -> async_sessionmaker[AsyncSession]:
    factory: async_sessionmaker[AsyncSession] = request.app.state.session_factory
    return factory


async def create_review(request: Request) -> JSONResponse:
    with correlation_context():
        try:
            body: Any = await request.json()
        except Exception:
            return _error_response(AssistantValidationError("Request body must be valid JSON"), 400)

        if not isinstance(body, dict):
            return _error_response(
                AssistantValidationError("Request body must be a JSON object"), 400
            )

        missing = [field for field in REQUIRED_CREATE_FIELDS if field not in body]
        if missing:
            return _error_response(
                AssistantValidationError(f"Missing required field(s): {', '.join(missing)}"), 400
            )

        factory = _session_factory_from_state(request)
        async with factory() as session:
            item = await create_review_item(
                session,
                run_id=body["run_id"],
                template_name=body["template_name"],
                domain_class=body["domain_class"],
                mappings=body["mappings"],
            )

        log_event(LOGGER, "http.review_created", review_id=item.id)
        return JSONResponse(item.to_api_dict(), status_code=201)


async def list_reviews(request: Request) -> JSONResponse:
    with correlation_context():
        status = request.query_params.get("status")
        factory = _session_factory_from_state(request)
        async with factory() as session:
            try:
                items = await list_review_items(session, status=status)
            except InvalidReviewStatusError as exc:
                return _error_response(exc, 400)

        return JSONResponse([item.to_api_dict() for item in items])


async def coverage(request: Request) -> JSONResponse:
    """`frontend/src/lib/api.ts::getCoverageStats` (refs
    frontend/src/lib/types.ts::CoverageStats) -- computed from this repo's
    own review queue, not agentic-dbpedia. See db.session.coverage_stats's
    docstring for why, and for exactly what "coverage" means here.
    """

    with correlation_context():
        factory = _session_factory_from_state(request)
        async with factory() as session:
            stats = await coverage_stats(session)

        return JSONResponse(
            {
                "totalTemplates": stats["total_templates"],
                "mappedTemplates": stats["mapped_templates"],
                "coveragePercent": stats["coverage_percent"],
                "lastRunAt": stats["last_run_at"],
            }
        )


async def get_review(request: Request) -> JSONResponse:
    with correlation_context():
        review_id = request.path_params["review_id"]
        factory = _session_factory_from_state(request)
        async with factory() as session:
            try:
                item: ReviewItem = await get_review_item(session, review_id)
            except ReviewNotFoundError as exc:
                return _error_response(exc, 404)

        return JSONResponse(item.to_api_dict())


async def decide_review(request: Request) -> JSONResponse:
    """Approve or reject a review item, logging one training example per
    mapping row unconditionally (refs 14.2) — every decision becomes
    training data whether or not it later gets published (14.3).

    An optional `corrected_mappings` (same shape as `mappings`) lets a
    reviewer submit an edited answer instead of a bare accept/reject; a
    mapping row whose `ontologyProperty` differs from what was originally
    predicted for that `templateProperty` logs `was_correction: true`.

    An optional `publish: true` alongside `decision: "approved"` is this
    endpoint's explicit consent to the real, outward-facing, hard-to
    -reverse action: it publishes the mapping to the live MediaWiki
    (mcp_server.publish.publish_mapping, gated through
    mcp_server.consent.require_consent) and, on success, sets status to
    "published" instead of "approved" and fires 12.2's eager corpus
    -refresh hook for real. A failed publish leaves status at "approved"
    (the review decision itself still stands) and reports the failure —
    never a silent partial state.

    An optional `reason` (freeform, e.g. why a mapping was rejected) isn't
    stored on the row today — there's no column for it yet — but is
    included on the `http.review_decided` structured log line so it isn't
    silently dropped.
    """

    with correlation_context():
        review_id = request.path_params["review_id"]
        try:
            body: Any = await request.json()
        except Exception:
            return _error_response(AssistantValidationError("Request body must be valid JSON"), 400)

        if not isinstance(body, dict) or "decision" not in body:
            return _error_response(
                AssistantValidationError("Missing required field: decision"), 400
            )

        decision = body["decision"]
        if decision not in DECISION_TO_STATUS:
            return _error_response(
                AssistantValidationError(
                    f"decision must be one of {sorted(DECISION_TO_STATUS)}, got {decision!r}"
                ),
                400,
            )

        corrected_mappings = body.get("corrected_mappings")
        reason = body.get("reason")

        factory = _session_factory_from_state(request)
        async with factory() as session:
            try:
                item = await get_review_item(session, review_id)
            except ReviewNotFoundError as exc:
                return _error_response(exc, 404)

            original_by_template = {
                mapping["templateProperty"]: mapping["ontologyProperty"]
                for mapping in item.mappings
            }
            confirmed_mappings = (
                corrected_mappings if corrected_mappings is not None else item.mappings
            )

            for mapping in confirmed_mappings:
                template_property = mapping["templateProperty"]
                human_confirmed = mapping["ontologyProperty"]
                model_predicted = original_by_template.get(template_property, human_confirmed)
                # The retriever's full candidate list isn't persisted on a
                # ReviewItem today, only the chosen prediction -- logging
                # what was actually decided rather than nothing. Persisting
                # real retrieval candidates here is a reasonable follow-up
                # once M16's pipeline orchestration creates review items
                # with that context available.
                log_decision(
                    f"{item.domain_class}'s {template_property}",
                    [model_predicted],
                    model_predicted=model_predicted,
                    human_confirmed=human_confirmed,
                    run_id=item.run_id,
                    log_path=request.app.state.training_log_path,
                )

            if decision == "approved":
                # Human-confirmed: reused before retrieval on the very next
                # request for this field and class (rag/learned_aliases.py).
                learned_aliases.record(
                    item.domain_class,
                    confirmed_mappings,
                    review_id=review_id,
                    path=request.app.state.learned_aliases_path,
                )

            updated = await set_review_status(session, review_id, DECISION_TO_STATUS[decision])
            if corrected_mappings is not None:
                updated.mappings = corrected_mappings
                await session.commit()
                await session.refresh(updated)

            want_publish = decision == "approved" and bool(body.get("publish"))
            if want_publish:
                publish_func: PublishFunc = request.app.state.publish_func
                try:
                    require_consent(approved=True)(publish_func)(
                        updated.template_name, updated.domain_class, updated.mappings
                    )
                except (PublishError, ConsentRequiredError) as exc:
                    log_event(LOGGER, "http.publish_failed", review_id=review_id, error=str(exc))
                    payload: dict[str, Any] = (
                        dict(exc.to_payload())
                        if isinstance(exc, PublishError)
                        else {
                            "status": "error",
                            "error_type": "consent_required",
                            "message": str(exc),
                        }
                    )
                    payload["correlation_id"] = get_correlation_id()
                    payload["review"] = updated.to_api_dict()
                    return JSONResponse(payload, status_code=502)

                updated = await set_review_status(session, review_id, "published")
                refresh_func: RefreshMappingsFunc = request.app.state.refresh_mappings_func
                refresh_func()
                log_event(LOGGER, "http.review_published", review_id=review_id)

        log_event(
            LOGGER, "http.review_decided", review_id=review_id, decision=decision, reason=reason
        )
        return JSONResponse(updated.to_api_dict())


async def preview_mapping(request: Request) -> Response:
    """SSE stream of the mapping pipeline's progress (refs 16.3), matching
    `frontend/src/lib/api.ts::previewMapping`'s contract exactly: `POST
    {infobox, target_class?}` **or** `POST {url, target_class?}`, response
    `text/event-stream` with one JSON-encoded `data:` line per pipeline
    node (16.2's `stream_mapping_pipeline`) and a final `{"node": "result",
    "mappings": [...]}` event.

    `url` is a real Wikipedia article link (`mcp_server.wiki_fetch`) --
    fetching its wikitext happens *before* `stream_mapping_pipeline`
    starts, reported as its own `fetch_source_article` SSE event, then the
    fetched wikitext runs through the exact same pipeline `infobox` would.
    `extract_first_infobox` already finds "the first infobox-like
    template" within arbitrarily larger wikitext, so handing it a whole
    article is enough -- no need for this endpoint to special-case
    extracting "just the infobox part" itself.

    A `url` may also point straight at a template *definition* page
    (am.wikipedia's `መለጠፊያ:` namespace, the English `Template:`), e.g.
    `.../wiki/መለጠፊያ:የምግብ_መረጃ`. A definition has no filled-in infobox to
    find, so its declared parameters are read instead
    (`extract_template_params`) and rendered as a synthetic infobox
    instance (`synthesize_infobox_wikitext`) that runs through the same
    pipeline -- mapping the template's parameter names directly.
    """

    with correlation_context():
        try:
            body: Any = await request.json()
        except Exception:
            return _error_response(AssistantValidationError("Request body must be valid JSON"), 400)

        infobox = body.get("infobox") if isinstance(body, dict) else None
        url = body.get("url") if isinstance(body, dict) else None
        if not infobox and not url:
            return _error_response(
                AssistantValidationError("Missing required field: infobox or url"), 400
            )

        domain_class = body.get("target_class") or DEFAULT_PREVIEW_DOMAIN_CLASS
        factory = _session_factory_from_state(request)
        fetch_article_func: FetchArticleFunc = request.app.state.fetch_article_func

        def _sse(event: dict[str, Any]) -> bytes:
            return f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode()

        async def event_stream() -> AsyncIterator[bytes]:
            # `infobox` is only ever unset here when `url` is set instead
            # (the validation above rejects both being empty) -- `or ""`
            # is purely to satisfy mypy that this is a `str`, not
            # `str | None`; the `if url:` branch below always overwrites
            # it before it would matter otherwise.
            wikitext: str = infobox or ""
            if url:
                try:
                    title, wikitext = await asyncio.to_thread(fetch_article_func, url)
                except WikipediaFetchError as exc:
                    log_event(LOGGER, "http.preview_fetch_failed", error=str(exc))
                    yield _sse(
                        {"node": "fetch_source_article", "status": "error", "detail": str(exc)}
                    )
                    yield _sse(
                        {
                            "node": "result",
                            "mappings": [],
                            "mappingWikitext": "",
                            "xmlRules": "",
                            "reviewItemId": None,
                            "warnings": [],
                            "unmappedFields": [],
                        }
                    )
                    return
                # A template-namespace page is a definition, not a filled
                # infobox: read its declared parameters and render them as
                # a synthetic infobox instance so the same pipeline maps
                # the template's parameter names directly.
                if is_template_title(title):
                    name = template_display_name(title)
                    params = extract_template_params(wikitext)
                    if not params:
                        yield _sse(
                            {
                                "node": "fetch_source_article",
                                "status": "error",
                                "detail": f'በመለጠፊያ "{name}" ውስጥ ምንም መስክ አልተገኘም።',
                            }
                        )
                        yield _sse(
                            {
                                "node": "result",
                                "mappings": [],
                                "mappingWikitext": "",
                                "xmlRules": "",
                                "reviewItemId": None,
                                "warnings": [],
                                "unmappedFields": [],
                            }
                        )
                        return
                    wikitext = synthesize_infobox_wikitext(name, params)
                    yield _sse(
                        {
                            "node": "fetch_source_article",
                            "status": "done",
                            "detail": f'ከመለጠፊያ "{name}" {len(params)} መስኮች ተነበቡ',
                        }
                    )
                else:
                    yield _sse(
                        {
                            "node": "fetch_source_article",
                            "status": "done",
                            "detail": f'"{title}" ከዊኪፔዲያ ተገኘቷል',
                        }
                    )

            async with factory() as session:
                async for event in stream_mapping_pipeline(
                    wikitext, domain_class=domain_class, session=session
                ):
                    yield _sse(event)

        log_event(LOGGER, "http.preview_started", domain_class=domain_class, from_url=bool(url))
        return StreamingResponse(event_stream(), media_type="text/event-stream")


async def find_semantic_match(request: Request) -> Response:
    """Mirrors the `find_semantic_match` MCP tool over HTTP (refs 16.3),
    matching `frontend/src/lib/api.ts::findSemanticMatch`'s contract
    exactly: `POST {amharic_property, target_class?}` ->
    `find_semantic_match_impl`'s own JSON string, passed through unchanged
    rather than re-encoded (it's already the exact response shape)."""

    with correlation_context():
        try:
            body: Any = await request.json()
        except Exception:
            return _error_response(AssistantValidationError("Request body must be valid JSON"), 400)

        amharic_property = body.get("amharic_property") if isinstance(body, dict) else None
        if not isinstance(amharic_property, str) or not amharic_property.strip():
            return _error_response(
                AssistantValidationError("amharic_property must be a non-empty string"), 400
            )
        if len(amharic_property) > MAX_PROPERTY_LENGTH:
            return _error_response(
                AssistantValidationError(
                    f"amharic_property must be at most {MAX_PROPERTY_LENGTH} characters"
                ),
                400,
            )

        search_func: SearchFunc = request.app.state.search_func
        target_class = body.get("target_class")
        payload = find_semantic_match_impl(amharic_property, target_class, search_func=search_func)
        fallback_func: FallbackFunc | None = request.app.state.fallback_func
        if fallback_func is None:
            return Response(payload, media_type="application/json")
        parsed = json.loads(payload)
        if parsed.get("status") != "no_match":
            return Response(payload, media_type="application/json")
        # Retrieval's gate found nothing confident -- the normal case for
        # Amharic vocabulary not in the corpus. Give the same LLM chain the
        # pipeline uses a chance, and hand back unverified suggestions.
        parsed.update(await asyncio.to_thread(fallback_func, amharic_property, target_class))
        return JSONResponse(parsed)


TEMPLATE_NAME_RE = re.compile(r"^[^\[\]{}|#<>\n]{1,200}$")
PROPERTY_NAME_RE = re.compile(r"^(?:[a-z]+:)?[A-Za-z][A-Za-z0-9_]*$")
TEMPLATE_PROPERTY_RE = re.compile(r"^[^\[\]{}|=\n]{1,120}$")


async def assist_understand(request: Request) -> JSONResponse:
    """`POST {text, target_class?}` -> the plan the chat should run (Groq,
    with a deterministic fallback; see mcp_server/assistant.py)."""

    with correlation_context():
        try:
            body: Any = await request.json()
        except Exception:
            return _error_response(AssistantValidationError("Request body must be valid JSON"), 400)
        text = body.get("text") if isinstance(body, dict) else None
        if not isinstance(text, str) or not text.strip():
            return _error_response(AssistantValidationError("text must be a non-empty string"), 400)
        if len(text) > MAX_ASSIST_TEXT_LENGTH:
            return _error_response(
                AssistantValidationError(
                    f"text must be at most {MAX_ASSIST_TEXT_LENGTH} characters"
                ),
                400,
            )
        plan = await asyncio.to_thread(
            assistant.understand,
            text,
            target_class=body.get("target_class") or None,
            client=request.app.state.chat_client,
        )
        log_event(LOGGER, "http.assist_understood", intent=plan.intent, source=plan.source)
        return JSONResponse(plan.to_api_dict())


async def assist_summarize(request: Request) -> JSONResponse:
    """`POST {kind: "infobox"|"field", result, language?}` -> `{message, source}`."""

    with correlation_context():
        try:
            body: Any = await request.json()
        except Exception:
            return _error_response(AssistantValidationError("Request body must be valid JSON"), 400)
        if not isinstance(body, dict) or not isinstance(body.get("result"), dict):
            return _error_response(AssistantValidationError("result must be an object"), 400)
        kind = body.get("kind")
        if kind not in {"infobox", "field"}:
            return _error_response(AssistantValidationError("kind must be infobox or field"), 400)
        feedback = await asyncio.to_thread(
            assistant.summarize,
            body["result"],
            kind=kind,
            language="en" if body.get("language") == "en" else "am",
            client=request.app.state.chat_client,
        )
        return JSONResponse({"message": feedback.message, "source": feedback.source})


def _publish_validation_error(body: Any) -> str | None:
    if not isinstance(body, dict):
        return "Request body must be an object"
    if body.get("consent") is not True:
        return "consent must be true -- publishing writes to the live mappings wiki"
    template_name = body.get("template_name")
    if not isinstance(template_name, str) or not TEMPLATE_NAME_RE.match(template_name.strip()):
        return "template_name must be a plain template name"
    domain_class = body.get("domain_class")
    if not isinstance(domain_class, str) or not re.match(r"^[A-Z][A-Za-z0-9]*$", domain_class):
        return "domain_class must be a DBpedia class name like Museum"
    mappings = body.get("mappings")
    if not isinstance(mappings, list) or not mappings:
        return "mappings must be a non-empty list"
    for row in mappings:
        if not isinstance(row, dict):
            return "each mapping must be an object"
        template_property = str(row.get("templateProperty", "")).strip()
        ontology_property = str(row.get("ontologyProperty", "")).strip()
        if not TEMPLATE_PROPERTY_RE.match(template_property):
            return f"invalid templateProperty {template_property!r}"
        if not PROPERTY_NAME_RE.match(ontology_property):
            return f"invalid ontologyProperty {ontology_property!r}"
    return None


async def publish_route(request: Request) -> JSONResponse:
    """Publish an edited mapping straight from the chat -- no review step.

    `POST {template_name, domain_class, mappings, consent: true,
    review_item_id?, original_mappings?}`. `consent` must be explicitly
    `true` (the chat only sends it after its confirmation dialog): this is
    a real edit to `Mapping am:<template_name>` on the live wiki. On success
    the mappings become learned aliases, every row is logged as a training
    example (a correction when it differs from `original_mappings`), the
    pipeline's run record is marked published, and the corpus refreshes.
    """

    with correlation_context():
        try:
            body: Any = await request.json()
        except Exception:
            return _error_response(AssistantValidationError("Request body must be valid JSON"), 400)
        problem = _publish_validation_error(body)
        if problem is not None:
            status = 403 if "consent" in problem else 400
            return _error_response(AssistantValidationError(problem), status)

        template_name = body["template_name"].strip()
        domain_class = body["domain_class"]
        mappings = [
            {
                "templateProperty": str(row["templateProperty"]).strip(),
                "ontologyProperty": str(row["ontologyProperty"]).strip(),
            }
            for row in body["mappings"]
        ]

        publish_func: PublishFunc = request.app.state.publish_func
        try:
            title = await asyncio.to_thread(
                require_consent(approved=True)(publish_func), template_name, domain_class, mappings
            )
        except (PublishError, ConsentRequiredError) as exc:
            log_event(LOGGER, "http.publish_failed", error=str(exc))
            payload: dict[str, Any] = (
                dict(exc.to_payload())
                if isinstance(exc, PublishError)
                else {"status": "error", "error_type": "consent_required", "message": str(exc)}
            )
            payload["correlation_id"] = get_correlation_id()
            return JSONResponse(payload, status_code=502)

        original = {
            str(m.get("templateProperty", "")).strip(): str(m.get("ontologyProperty", "")).strip()
            for m in body.get("original_mappings") or []
            if isinstance(m, dict)
        }
        for mapping in mappings:
            predicted = original.get(mapping["templateProperty"], mapping["ontologyProperty"])
            log_decision(
                f"{domain_class}'s {mapping['templateProperty']}",
                [predicted],
                model_predicted=predicted,
                human_confirmed=mapping["ontologyProperty"],
                run_id=body.get("review_item_id"),
                log_path=request.app.state.training_log_path,
            )
        learned_aliases.record(
            domain_class,
            mappings,
            review_id=body.get("review_item_id"),
            path=request.app.state.learned_aliases_path,
        )

        review_item_id = body.get("review_item_id")
        if isinstance(review_item_id, str) and review_item_id:
            factory = _session_factory_from_state(request)
            async with factory() as session:
                try:
                    item = await set_review_status(session, review_item_id, "published")
                    item.mappings = mappings
                    await session.commit()
                except ReviewNotFoundError:
                    log_event(LOGGER, "http.publish_run_record_missing", review_id=review_item_id)

        refresh_func: RefreshMappingsFunc = request.app.state.refresh_mappings_func
        try:
            await asyncio.to_thread(refresh_func)
        except Exception as exc:  # noqa: BLE001 - the edit itself already succeeded
            log_event(LOGGER, "http.refresh_after_publish_failed", error=exc.__class__.__name__)

        base_url = request.app.state.mediawiki_base_url.rstrip("/")
        log_event(LOGGER, "http.published", title=title, mappings=len(mappings))
        return JSONResponse(
            {
                "status": "published",
                "title": title,
                "url": f"{base_url}/index.php/{str(title).replace(' ', '_')}",
            }
        )


routes = [
    Route("/v1/reviews", create_review, methods=["POST"]),
    Route("/v1/reviews", list_reviews, methods=["GET"]),
    Route("/v1/reviews/{review_id}", get_review, methods=["GET"]),
    Route("/v1/reviews/{review_id}/decision", decide_review, methods=["POST"]),
    Route("/v1/preview", preview_mapping, methods=["POST"]),
    Route("/v1/find-semantic-match", find_semantic_match, methods=["POST"]),
    Route("/v1/coverage", coverage, methods=["GET"]),
    Route("/v1/assist/understand", assist_understand, methods=["POST"]),
    Route("/v1/assist/summarize", assist_summarize, methods=["POST"]),
    Route("/v1/publish", publish_route, methods=["POST"]),
]


FallbackFunc = Callable[[str, str | None], dict[str, Any]]


def fallback_payload(outcome: Any, target_class: str | None) -> dict[str, Any]:
    """`/v1/find-semantic-match` fields for a `predict_property` outcome
    reached after retrieval's own `no_match`."""

    if isinstance(outcome, PredictionResult):
        learned = outcome.method == "learned"
        return {
            "status": "ok",
            "matches": [
                {
                    "property": outcome.property,
                    "class": target_class or "",
                    "score": 1.0 if learned else 0.3,
                    "payload": {
                        "source": "learned" if learned else "llm",
                        "model": outcome.model,
                        "reason": outcome.reason,
                    },
                }
            ],
            "suggestions": [],
        }
    return {
        "status": "no_match",
        "matches": [],
        "suggestions": [
            {
                "property": s.property,
                "class": s.ontology_class,
                "score": s.score,
                "payload": {"source": "suggestion"},
            }
            for s in getattr(outcome, "suggestions", ())
        ],
        "llm_error": getattr(outcome, "llm_error", None),
    }


def semantic_fallback(amharic_property: str, target_class: str | None) -> dict[str, Any]:
    """Learned alias -> LLM propose -> LLM select-among-suggestions, without
    re-running the retrieval that just returned `no_match`."""

    outcome = predict_property(
        amharic_property,
        target_class=target_class,
        search_func=lambda *_args, **_kwargs: [NoMatchFound(query=amharic_property)],
        **production_predict_options(),
    )
    return fallback_payload(outcome, target_class)


def create_app(
    *,
    engine: AsyncEngine | None = None,
    settings: Settings | None = None,
    training_log_path: Path | None = None,
    publish_func: PublishFunc = publish_mapping,
    refresh_mappings_func: RefreshMappingsFunc = refresh_mappings,
    search_func: SearchFunc = default_search,
    fetch_article_func: FetchArticleFunc = fetch_article_wikitext,
    fallback_func: FallbackFunc | None = None,
    use_default_fallback: bool = True,
    learned_aliases_path: Path | None = None,
    chat_client: assistant.ChatClient | None = None,
    use_default_chat_client: bool = True,
) -> Starlette:
    """Build the app with a fresh engine/session factory in app.state, so
    tests can inject an isolated (e.g. in-memory SQLite) engine and/or an
    isolated training-log path instead of the real configured ones.

    `publish_func`/`refresh_mappings_func`/`fetch_article_func` default to
    the real mcp_server.publish.publish_mapping /
    scripts.refresh_wiki_cache.refresh_mappings /
    mcp_server.wiki_fetch.fetch_article_wikitext — tests always inject
    fakes for all three, since the real ones perform a live, irreversible
    MediaWiki write and real network fetches respectively. `search_func`
    defaults to the real rag.retrieval.search — tests inject a fake to
    avoid the real embedding-model-backed index. `fallback_func` is what
    `/v1/find-semantic-match` runs on a `no_match` (default: the real LLM
    chain + ungated suggestions, `semantic_fallback`); tests pass
    `use_default_fallback=False` to keep that path offline.
    """

    resolved_engine = engine or create_engine(settings=settings)

    @asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        await init_models(resolved_engine)
        yield

    # frontend/ (SvelteKit, served from a different origin -- e.g.
    # localhost:5173 -- than this app's own localhost:8001) calls this API
    # with fetch() directly from the browser, not from its own backend, so
    # every non-GET request is CORS-preflighted. Without this, the browser
    # never even sends the real POST -- it fails the OPTIONS preflight with
    # a 405 first and safeFetch() in frontend/src/lib/api.ts sees that as
    # an opaque network failure indistinguishable from the server being
    # down entirely (confirmed live: this is exactly what "cross-lingual is
    # not reachable" meant with a server that was actually up and healthy).
    # No cookies/auth headers are ever sent (frontend/README.md's Auth
    # section -- deliberately deferred for this internal tool), so a
    # wildcard origin is safe: there's no credential a third-party origin
    # could ride along with a forged request.
    middleware = [
        Middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )
    ]

    app = Starlette(routes=routes, lifespan=lifespan, middleware=middleware)
    app.state.engine = resolved_engine
    app.state.training_log_path = training_log_path or DEFAULT_LOG_PATH
    app.state.session_factory = session_factory(resolved_engine)
    app.state.publish_func = publish_func
    app.state.refresh_mappings_func = refresh_mappings_func
    app.state.search_func = search_func
    app.state.chat_client = chat_client or (
        assistant.default_chat_client() if use_default_chat_client else None
    )
    app.state.mediawiki_base_url = (
        settings.mediawiki_base_url if settings is not None else _mediawiki_base_url()
    )
    app.state.learned_aliases_path = (
        learned_aliases_path or learned_aliases.DEFAULT_LEARNED_ALIASES_PATH
    )
    app.state.fallback_func = fallback_func or (semantic_fallback if use_default_fallback else None)
    app.state.fetch_article_func = fetch_article_func
    return app


# Deliberately NOT `app = create_app()` at module level. Confirmed live:
# create_engine(settings=None) -> resolve_database_url() -> Settings()
# validates every required field (GROQ_API_KEY has no default -- see its
# own comment in config.py for why that's intentional) the instant this
# module is *imported*, not merely when the server actually starts. Every
# test here imports `create_app` and calls it explicitly with an injected
# engine/settings, so nothing in this codebase actually needs a
# pre-built module-level `app` -- but `uvicorn mcp_server.http_app:app`
# did, and that forced any environment without a real GROQ_API_KEY set
# (a CI unit-test job that has no business needing one) to fail at
# collection time just from importing this file, before a single test
# even ran. `uvicorn ... --factory` calls `create_app()` itself, lazily,
# only when the server actually starts -- exactly where a real
# environment (which does have GROQ_API_KEY) is expected to exist.
