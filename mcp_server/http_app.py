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
from mcp_server.consent import ConsentRequiredError, require_consent
from mcp_server.pipeline import stream_mapping_pipeline
from mcp_server.publish import PublishError, publish_mapping
from mcp_server.server import MAX_PROPERTY_LENGTH, find_semantic_match_impl
from mcp_server.wiki_fetch import WikipediaFetchError, fetch_article_wikitext
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
                yield _sse(
                    {
                        "node": "fetch_source_article",
                        "status": "done",
                        "detail": f'Fetched "{title}" from Wikipedia',
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
        payload = find_semantic_match_impl(
            amharic_property, body.get("target_class"), search_func=search_func
        )
        return Response(payload, media_type="application/json")


routes = [
    Route("/v1/reviews", create_review, methods=["POST"]),
    Route("/v1/reviews", list_reviews, methods=["GET"]),
    Route("/v1/reviews/{review_id}", get_review, methods=["GET"]),
    Route("/v1/reviews/{review_id}/decision", decide_review, methods=["POST"]),
    Route("/v1/preview", preview_mapping, methods=["POST"]),
    Route("/v1/find-semantic-match", find_semantic_match, methods=["POST"]),
    Route("/v1/coverage", coverage, methods=["GET"]),
]


def create_app(
    *,
    engine: AsyncEngine | None = None,
    settings: Settings | None = None,
    training_log_path: Path | None = None,
    publish_func: PublishFunc = publish_mapping,
    refresh_mappings_func: RefreshMappingsFunc = refresh_mappings,
    search_func: SearchFunc = default_search,
    fetch_article_func: FetchArticleFunc = fetch_article_wikitext,
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
    avoid the real embedding-model-backed index.
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
