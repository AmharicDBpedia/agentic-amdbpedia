from __future__ import annotations

import tempfile
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.pool import StaticPool
from starlette.applications import Starlette
from starlette.testclient import TestClient

from db.session import (
    InvalidReviewStatusError,
    ReviewNotFoundError,
    coverage_stats,
    create_review_item,
    get_review_item,
    init_models,
    list_review_items,
    resolve_database_url,
    session_factory,
    set_review_status,
)
from mcp_server.http_app import create_app
from mcp_server.publish import PublishError
from rag.training_log import read_examples

SAMPLE_MAPPINGS = [
    {"templateProperty": "አይካኦ_ኮድ", "ontologyProperty": "icaoLocationIdentifier", "confidence": 0.9}
]


def _in_memory_engine() -> AsyncEngine:
    return create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )


def _make_app(
    *,
    training_log_path: Path | None = None,
    publish_func: Any = None,
    refresh_mappings_func: Any = None,
    learned_aliases_path: Path | None = None,
) -> Starlette:
    kwargs: dict[str, Any] = {
        "engine": _in_memory_engine(),
        "training_log_path": training_log_path,
        "use_default_fallback": False,
        "use_default_chat_client": False,
        "learned_aliases_path": learned_aliases_path
        or Path(tempfile.mkdtemp()) / "learned_aliases.json",
    }
    if publish_func is not None:
        kwargs["publish_func"] = publish_func
    if refresh_mappings_func is not None:
        kwargs["refresh_mappings_func"] = refresh_mappings_func
    return create_app(**kwargs)


@pytest_asyncio.fixture
async def engine() -> AsyncIterator[AsyncEngine]:
    # StaticPool: a single shared connection, so SQLite's :memory: database
    # survives across the multiple separate sessions these tests open --
    # otherwise each new connection would see its own empty database.
    test_engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    await init_models(test_engine)
    yield test_engine
    await test_engine.dispose()


def test_resolve_database_url_reads_the_real_environment_when_no_settings_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression test: `mcp_server/http_app.py`'s module-level `app =
    create_app()` -- the actual object `just run-http` serves -- calls
    `create_engine(settings=None)`. `resolve_database_url` used to
    short-circuit straight to the hardcoded SQLite default whenever
    `settings` was `None`, silently ignoring a real `DATABASE_URL` in the
    environment/`.env` entirely -- found live: every `just run-http` run
    was writing to local SQLite regardless of a correctly-configured
    docker-compose Postgres `DATABASE_URL`."""

    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_placeholder")
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://u:p@some-real-host/db")

    assert resolve_database_url(None) == "postgresql+asyncpg://u:p@some-real-host/db"


@pytest.mark.asyncio
async def test_create_review_item_persists_a_row(engine: AsyncEngine) -> None:
    factory = session_factory(engine)
    async with factory() as session:
        item = await create_review_item(
            session,
            run_id="run-1",
            template_name="Infobox airport",
            domain_class="Airport",
            mappings=SAMPLE_MAPPINGS,
        )

    assert item.id
    assert item.status == "pending_review"
    assert item.run_id == "run-1"


@pytest.mark.asyncio
async def test_list_review_items_returns_created_rows_oldest_first(engine: AsyncEngine) -> None:
    factory = session_factory(engine)
    async with factory() as session:
        await create_review_item(
            session, run_id="r1", template_name="A", domain_class="Airport", mappings=[]
        )
        await create_review_item(
            session, run_id="r2", template_name="B", domain_class="Dam", mappings=[]
        )

    async with factory() as session:
        items = await list_review_items(session)

    assert [item.template_name for item in items] == ["A", "B"]


@pytest.mark.asyncio
async def test_list_review_items_filters_by_status(engine: AsyncEngine) -> None:
    factory = session_factory(engine)
    async with factory() as session:
        created = await create_review_item(
            session, run_id="r1", template_name="A", domain_class="Airport", mappings=[]
        )
        await create_review_item(
            session, run_id="r2", template_name="B", domain_class="Dam", mappings=[]
        )
        await set_review_status(session, created.id, "approved")

    async with factory() as session:
        approved = await list_review_items(session, status="approved")
        pending = await list_review_items(session, status="pending_review")

    assert [item.template_name for item in approved] == ["A"]
    assert [item.template_name for item in pending] == ["B"]


@pytest.mark.asyncio
async def test_list_review_items_rejects_an_unknown_status(engine: AsyncEngine) -> None:
    factory = session_factory(engine)
    async with factory() as session:
        with pytest.raises(InvalidReviewStatusError):
            await list_review_items(session, status="not_a_real_status")


@pytest.mark.asyncio
async def test_get_review_item_raises_not_found_for_a_missing_id(engine: AsyncEngine) -> None:
    factory = session_factory(engine)
    async with factory() as session:
        with pytest.raises(ReviewNotFoundError):
            await get_review_item(session, "does-not-exist")


@pytest.mark.asyncio
async def test_set_review_status_transitions_and_persists(engine: AsyncEngine) -> None:
    factory = session_factory(engine)
    async with factory() as session:
        created = await create_review_item(
            session, run_id="r1", template_name="A", domain_class="Airport", mappings=[]
        )

    async with factory() as session:
        updated = await set_review_status(session, created.id, "approved")

    assert updated.status == "approved"

    async with factory() as session:
        refetched = await get_review_item(session, created.id)
    assert refetched.status == "approved"


@pytest.mark.asyncio
async def test_coverage_stats_on_an_empty_queue(engine: AsyncEngine) -> None:
    factory = session_factory(engine)
    async with factory() as session:
        stats = await coverage_stats(session)

    assert stats == {
        "total_templates": 0,
        "mapped_templates": 0,
        "coverage_percent": 0.0,
        "last_run_at": None,
    }


@pytest.mark.asyncio
async def test_coverage_stats_counts_distinct_templates_and_only_published_as_mapped(
    engine: AsyncEngine,
) -> None:
    factory = session_factory(engine)
    async with factory() as session:
        # Two rows share "Infobox airport" -- still one distinct template.
        published = await create_review_item(
            session,
            run_id="r1",
            template_name="Infobox airport",
            domain_class="Airport",
            mappings=[],
        )
        await create_review_item(
            session,
            run_id="r2",
            template_name="Infobox airport",
            domain_class="Airport",
            mappings=[],
        )
        # approved (not published) shouldn't count as mapped.
        await create_review_item(
            session, run_id="r3", template_name="Infobox bridge", domain_class="Bridge", mappings=[]
        )
        rejected = await create_review_item(
            session, run_id="r4", template_name="Infobox dam", domain_class="Dam", mappings=[]
        )

    async with factory() as session:
        await set_review_status(session, published.id, "published")
        await set_review_status(session, rejected.id, "rejected")

    async with factory() as session:
        stats = await coverage_stats(session)

    assert stats["total_templates"] == 3  # airport, bridge, dam
    assert stats["mapped_templates"] == 1  # only airport has a published row
    assert stats["coverage_percent"] == pytest.approx(33.3)
    assert stats["last_run_at"] is not None


def test_post_v1_reviews_creates_a_row_and_returns_frontend_shaped_json() -> None:
    app = _make_app()

    with TestClient(app) as client:
        response = client.post(
            "/v1/reviews",
            json={
                "run_id": "run-1",
                "template_name": "Infobox airport",
                "domain_class": "Airport",
                "mappings": SAMPLE_MAPPINGS,
            },
        )

        assert response.status_code == 201
        body = response.json()
        # Exactly frontend/src/lib/types.ts::ReviewItem's field set --
        # camelCase, no run_id leaking through (that's DB-internal only).
        assert set(body.keys()) == {
            "id",
            "templateName",
            "domainClass",
            "status",
            "submittedAt",
            "mappings",
        }
        assert body["templateName"] == "Infobox airport"
        assert body["domainClass"] == "Airport"
        assert body["status"] == "pending_review"
        assert body["mappings"] == SAMPLE_MAPPINGS


def test_post_v1_reviews_rejects_missing_required_fields() -> None:
    app = _make_app()

    with TestClient(app) as client:
        response = client.post("/v1/reviews", json={"run_id": "run-1"})

        assert response.status_code == 400
        assert response.json()["error_type"] == "validation"


def test_get_v1_reviews_lists_created_items() -> None:
    app = _make_app()

    with TestClient(app) as client:
        client.post(
            "/v1/reviews",
            json={
                "run_id": "run-1",
                "template_name": "A",
                "domain_class": "Airport",
                "mappings": [],
            },
        )

        response = client.get("/v1/reviews")

        assert response.status_code == 200
        items = response.json()
        assert len(items) == 1
        assert items[0]["templateName"] == "A"


def test_get_v1_coverage_reflects_published_rows_only() -> None:
    app = _make_app(
        publish_func=lambda template_name, domain_class, mappings: f"Mapping am:{template_name}",
        refresh_mappings_func=lambda: 0,
    )

    with TestClient(app) as client:
        empty = client.get("/v1/coverage")
        assert empty.status_code == 200
        assert empty.json() == {
            "totalTemplates": 0,
            "mappedTemplates": 0,
            "coveragePercent": 0.0,
            "lastRunAt": None,
        }

        created = client.post(
            "/v1/reviews",
            json={
                "run_id": "run-1",
                "template_name": "Infobox airport",
                "domain_class": "Airport",
                "mappings": SAMPLE_MAPPINGS,
            },
        ).json()

        still_pending = client.get("/v1/coverage").json()
        assert still_pending["totalTemplates"] == 1
        assert still_pending["mappedTemplates"] == 0

        client.post(
            f"/v1/reviews/{created['id']}/decision",
            json={"decision": "approved", "publish": True},
        )

        after_publish = client.get("/v1/coverage").json()
        assert after_publish["totalTemplates"] == 1
        assert after_publish["mappedTemplates"] == 1
        assert after_publish["coveragePercent"] == 100.0
        assert after_publish["lastRunAt"] is not None


def test_get_v1_reviews_by_id_returns_404_for_missing_review() -> None:
    app = _make_app()

    with TestClient(app) as client:
        response = client.get("/v1/reviews/does-not-exist")

        assert response.status_code == 404
        assert response.json()["error_type"] == "not_found"


def test_decide_review_approves_without_a_correction_and_logs_no_correction(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "training_examples.jsonl"
    app = _make_app(training_log_path=log_path)

    with TestClient(app) as client:
        created = client.post(
            "/v1/reviews",
            json={
                "run_id": "run-1",
                "template_name": "Infobox airport",
                "domain_class": "Airport",
                "mappings": SAMPLE_MAPPINGS,
            },
        ).json()

        response = client.post(
            f"/v1/reviews/{created['id']}/decision", json={"decision": "approved"}
        )

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "approved"
        assert body["mappings"] == SAMPLE_MAPPINGS

    examples = read_examples(log_path)
    assert len(examples) == 1
    assert examples[0].was_correction is False
    assert examples[0].property_class == "icaoLocationIdentifier"
    assert examples[0].run_id == "run-1"


def test_decide_review_with_a_correction_logs_was_correction_true(tmp_path: Path) -> None:
    """The stated 14.2 acceptance criterion: approving a corrected mapping
    logs was_correction: true."""

    log_path = tmp_path / "training_examples.jsonl"
    app = _make_app(training_log_path=log_path)

    with TestClient(app) as client:
        created = client.post(
            "/v1/reviews",
            json={
                "run_id": "run-1",
                "template_name": "Infobox airport",
                "domain_class": "Airport",
                "mappings": SAMPLE_MAPPINGS,
            },
        ).json()

        corrected = [
            {
                "templateProperty": "አይካኦ_ኮድ",
                "ontologyProperty": "iataLocationIdentifier",  # human corrected this
                "confidence": 0.9,
            }
        ]
        response = client.post(
            f"/v1/reviews/{created['id']}/decision",
            json={"decision": "approved", "corrected_mappings": corrected},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "approved"
        assert body["mappings"] == corrected

    examples = read_examples(log_path)
    assert len(examples) == 1
    assert examples[0].was_correction is True
    assert examples[0].model_predicted == "icaoLocationIdentifier"
    assert examples[0].property_class == "iataLocationIdentifier"


def test_decide_review_rejection_still_logs_a_training_example(tmp_path: Path) -> None:
    """13.1's design point: every decision becomes training data whether or
    not it later gets published."""

    log_path = tmp_path / "training_examples.jsonl"
    app = _make_app(training_log_path=log_path)

    with TestClient(app) as client:
        created = client.post(
            "/v1/reviews",
            json={
                "run_id": "run-1",
                "template_name": "Infobox airport",
                "domain_class": "Airport",
                "mappings": SAMPLE_MAPPINGS,
            },
        ).json()

        response = client.post(
            f"/v1/reviews/{created['id']}/decision", json={"decision": "rejected"}
        )

        assert response.status_code == 200
        assert response.json()["status"] == "rejected"

    assert len(read_examples(log_path)) == 1


def test_decide_review_rejects_an_unknown_decision_value(tmp_path: Path) -> None:
    app = _make_app(training_log_path=tmp_path / "training_examples.jsonl")

    with TestClient(app) as client:
        created = client.post(
            "/v1/reviews",
            json={
                "run_id": "run-1",
                "template_name": "A",
                "domain_class": "Airport",
                "mappings": SAMPLE_MAPPINGS,
            },
        ).json()

        response = client.post(f"/v1/reviews/{created['id']}/decision", json={"decision": "maybe"})

        assert response.status_code == 400
        assert response.json()["error_type"] == "validation"


def test_decide_review_returns_404_for_a_missing_review(tmp_path: Path) -> None:
    app = _make_app(training_log_path=tmp_path / "training_examples.jsonl")

    with TestClient(app) as client:
        response = client.post("/v1/reviews/does-not-exist/decision", json={"decision": "approved"})

        assert response.status_code == 404
        assert response.json()["error_type"] == "not_found"


def test_decide_review_with_publish_true_publishes_and_refreshes(tmp_path: Path) -> None:
    publish_calls: list[tuple[str, str, list[dict[str, str]]]] = []
    refresh_calls: list[None] = []

    def fake_publish(template_name: str, domain_class: str, mappings: list[dict[str, str]]) -> str:
        publish_calls.append((template_name, domain_class, mappings))
        return f"Mapping am:{template_name}"

    def fake_refresh() -> int:
        refresh_calls.append(None)
        return 106

    app = _make_app(
        training_log_path=tmp_path / "training_examples.jsonl",
        publish_func=fake_publish,
        refresh_mappings_func=fake_refresh,
    )

    with TestClient(app) as client:
        created = client.post(
            "/v1/reviews",
            json={
                "run_id": "run-1",
                "template_name": "Infobox airport",
                "domain_class": "Airport",
                "mappings": SAMPLE_MAPPINGS,
            },
        ).json()

        response = client.post(
            f"/v1/reviews/{created['id']}/decision", json={"decision": "approved", "publish": True}
        )

        assert response.status_code == 200
        assert response.json()["status"] == "published"

    assert publish_calls == [("Infobox airport", "Airport", SAMPLE_MAPPINGS)]
    assert refresh_calls == [None]


def test_decide_review_without_publish_flag_never_calls_publish(tmp_path: Path) -> None:
    publish_calls: list[Any] = []

    def fake_publish(*args: Any, **kwargs: Any) -> str:
        publish_calls.append((args, kwargs))
        return "should not be called"

    app = _make_app(
        training_log_path=tmp_path / "training_examples.jsonl", publish_func=fake_publish
    )

    with TestClient(app) as client:
        created = client.post(
            "/v1/reviews",
            json={
                "run_id": "run-1",
                "template_name": "A",
                "domain_class": "Airport",
                "mappings": SAMPLE_MAPPINGS,
            },
        ).json()

        response = client.post(
            f"/v1/reviews/{created['id']}/decision", json={"decision": "approved"}
        )

        assert response.json()["status"] == "approved"

    assert publish_calls == []


def test_decide_review_publish_failure_leaves_status_as_approved(tmp_path: Path) -> None:
    def failing_publish(*args: Any, **kwargs: Any) -> str:
        raise PublishError("MediaWiki edit was rejected: simulated failure")

    app = _make_app(
        training_log_path=tmp_path / "training_examples.jsonl", publish_func=failing_publish
    )

    with TestClient(app) as client:
        created = client.post(
            "/v1/reviews",
            json={
                "run_id": "run-1",
                "template_name": "A",
                "domain_class": "Airport",
                "mappings": SAMPLE_MAPPINGS,
            },
        ).json()

        response = client.post(
            f"/v1/reviews/{created['id']}/decision", json={"decision": "approved", "publish": True}
        )

        assert response.status_code == 502
        body = response.json()
        assert body["error_type"] == "publish_failed"
        # The review decision itself still stands even though publish failed.
        assert body["review"]["status"] == "approved"

        refetched = client.get(f"/v1/reviews/{created['id']}").json()
        assert refetched["status"] == "approved"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_review_queue_works_against_a_real_postgres_instance() -> None:
    """Verified against docker-compose's real postgres service, not just
    SQLite — SQLite and Postgres can silently disagree on JSON column
    handling, so this is a real, not hypothetical, compatibility check."""

    real_engine = create_async_engine(
        "postgresql+asyncpg://mapping_assistant:mapping_assistant@localhost:5435/mapping_assistant"
    )
    try:
        await init_models(real_engine)
        factory = session_factory(real_engine)
        async with factory() as session:
            item = await create_review_item(
                session,
                run_id="pg-run-1",
                template_name="Infobox airport (postgres)",
                domain_class="Airport",
                mappings=SAMPLE_MAPPINGS,
            )
            fetched = await get_review_item(session, item.id)
            assert fetched.mappings == SAMPLE_MAPPINGS
            assert fetched.template_name == "Infobox airport (postgres)"

            # Clean up after ourselves -- this hits a real, possibly shared
            # Postgres instance, not a throwaway per-test database.
            await session.delete(fetched)
            await session.commit()
    finally:
        await real_engine.dispose()


def _airport_review(client: TestClient) -> dict[str, Any]:
    body: dict[str, Any] = client.post(
        "/v1/reviews",
        json={
            "run_id": "run-1",
            "template_name": "Infobox airport",
            "domain_class": "Airport",
            "mappings": SAMPLE_MAPPINGS,
        },
    ).json()
    return body


def test_approving_a_corrected_review_teaches_a_learned_alias(tmp_path: Path) -> None:
    from rag.learned_aliases import lookup

    learned = tmp_path / "learned_aliases.json"
    app = _make_app(training_log_path=tmp_path / "log.jsonl", learned_aliases_path=learned)
    corrected = [{"templateProperty": "አይካኦ_ኮድ", "ontologyProperty": "iataLocationIdentifier"}]

    with TestClient(app) as client:
        created = _airport_review(client)
        client.post(
            f"/v1/reviews/{created['id']}/decision",
            json={"decision": "approved", "corrected_mappings": corrected},
        )

    assert lookup("አይካኦ_ኮድ", "Airport", path=learned) == "iataLocationIdentifier"


def test_rejecting_a_review_teaches_nothing(tmp_path: Path) -> None:
    learned = tmp_path / "learned_aliases.json"
    app = _make_app(training_log_path=tmp_path / "log.jsonl", learned_aliases_path=learned)

    with TestClient(app) as client:
        created = _airport_review(client)
        client.post(f"/v1/reviews/{created['id']}/decision", json={"decision": "rejected"})

    assert not learned.exists()
