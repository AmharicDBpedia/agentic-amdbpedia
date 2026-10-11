from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from config import Settings
from mcp_server.consent import ConsentRequiredError, require_consent
from mcp_server.publish import (
    MediaWikiCredentialsError,
    PublishError,
    build_mapping_wikitext,
    merge_mapping_wikitext,
    publish_mapping,
)

SAMPLE_MAPPINGS = [
    {"templateProperty": "አይካኦ_ኮድ", "ontologyProperty": "icaoLocationIdentifier"},
    {"templateProperty": "አያታ_ኮድ", "ontologyProperty": "iataLocationIdentifier"},
]


def _settings(**overrides: Any) -> Settings:
    base = {
        "GROQ_API_KEY": "gsk_test_placeholder",
        "MEDIAWIKI_BOT_USERNAME": "TestBot@publish-bot",
        "MEDIAWIKI_BOT_PASSWORD": "test-bot-password",
    }
    base.update(overrides)
    return Settings(_env_file=None, **base)


class FakeTransport:
    """Scripts the exact login -> csrf -> edit sequence publish_mapping
    drives, without any network access — publishing to the real wiki is
    never something this test suite attempts (see mcp_server/publish.py's
    module docstring)."""

    def __init__(
        self,
        *,
        login_result: str = "Success",
        edit_result: str = "Success",
        omit_login_token: bool = False,
        omit_csrf_token: bool = False,
        existing_page: str | None = None,
    ) -> None:
        self.existing_page = existing_page
        self.login_result = login_result
        self.edit_result = edit_result
        self.omit_login_token = omit_login_token
        self.omit_csrf_token = omit_csrf_token
        self.get_calls: list[dict[str, str]] = []
        self.post_calls: list[dict[str, str]] = []

    def get(self, params: dict[str, str]) -> dict[str, Any]:
        self.get_calls.append(params)
        if params.get("prop") == "revisions":
            if self.existing_page is None:
                return {"query": {"pages": [{"title": params["titles"], "missing": True}]}}
            return {
                "query": {
                    "pages": [
                        {
                            "title": params["titles"],
                            "revisions": [{"slots": {"main": {"content": self.existing_page}}}],
                        }
                    ]
                }
            }
        if params["type"] == "login":
            if self.omit_login_token:
                return {"query": {"tokens": {}}}
            return {"query": {"tokens": {"logintoken": "fake-login-token"}}}
        if self.omit_csrf_token:
            return {"query": {"tokens": {}}}
        return {"query": {"tokens": {"csrftoken": "fake-csrf-token"}}}

    def post(self, data: dict[str, str]) -> dict[str, Any]:
        self.post_calls.append(data)
        if data["action"] == "login":
            return {"login": {"result": self.login_result}}
        return {"edit": {"result": self.edit_result, "title": data["title"]}}


def test_publish_is_refused_without_consent() -> None:
    transport = FakeTransport()

    with pytest.raises(ConsentRequiredError):
        require_consent(approved=False)(publish_mapping)(
            "Infobox airport",
            "Airport",
            SAMPLE_MAPPINGS,
            settings=_settings(),
            transport=transport,
        )

    assert transport.get_calls == []
    assert transport.post_calls == []


def test_publish_succeeds_with_consent_and_a_mocked_successful_edit() -> None:
    transport = FakeTransport()

    title = require_consent(approved=True)(publish_mapping)(
        "Infobox airport", "Airport", SAMPLE_MAPPINGS, settings=_settings(), transport=transport
    )

    assert title == "Mapping am:Infobox airport"
    assert transport.post_calls[-1]["action"] == "edit"
    assert transport.post_calls[-1]["token"] == "fake-csrf-token"
    assert transport.post_calls[-1]["bot"] == "1"


def test_publish_follows_the_login_then_csrf_then_edit_sequence() -> None:
    transport = FakeTransport()

    publish_mapping(
        "Infobox airport", "Airport", SAMPLE_MAPPINGS, settings=_settings(), transport=transport
    )

    assert [call.get("type", call.get("prop")) for call in transport.get_calls] == [
        "login",
        "csrf",
        "revisions",
    ]
    assert [call["action"] for call in transport.post_calls] == ["login", "edit"]
    assert transport.post_calls[0]["lgname"] == "TestBot@publish-bot"
    assert transport.post_calls[0]["lgtoken"] == "fake-login-token"


def test_publish_raises_without_configured_credentials() -> None:
    settings = _settings(MEDIAWIKI_BOT_USERNAME=None, MEDIAWIKI_BOT_PASSWORD=None)

    with pytest.raises(MediaWikiCredentialsError):
        publish_mapping(
            "Infobox airport",
            "Airport",
            SAMPLE_MAPPINGS,
            settings=settings,
            transport=FakeTransport(),
        )


def test_publish_raises_on_a_failed_login() -> None:
    transport = FakeTransport(login_result="Failed")

    with pytest.raises(PublishError, match="login failed"):
        publish_mapping(
            "Infobox airport", "Airport", SAMPLE_MAPPINGS, settings=_settings(), transport=transport
        )


def test_publish_explains_a_wiki_that_denies_api_login() -> None:
    # What mappings.dbpedia.org really answers (checked live): an API error,
    # not a login result -- this used to surface as "unknown error".
    class WriteApiDenied(FakeTransport):
        def post(self, data: dict[str, str]) -> dict[str, Any]:
            if data.get("action") == "login":
                return {"error": {"code": "writeapidenied", "info": "You're not allowed"}}
            return super().post(data)

    with pytest.raises(PublishError, match="writeapidenied.*admin"):
        publish_mapping(
            "Infobox airport",
            "Airport",
            SAMPLE_MAPPINGS,
            settings=_settings(),
            transport=WriteApiDenied(),
        )


def test_publish_raises_when_no_login_token_is_returned() -> None:
    transport = FakeTransport(omit_login_token=True)

    with pytest.raises(PublishError, match="login token"):
        publish_mapping(
            "Infobox airport", "Airport", SAMPLE_MAPPINGS, settings=_settings(), transport=transport
        )


def test_publish_raises_when_no_csrf_token_is_returned() -> None:
    transport = FakeTransport(omit_csrf_token=True)

    with pytest.raises(PublishError, match="CSRF token"):
        publish_mapping(
            "Infobox airport", "Airport", SAMPLE_MAPPINGS, settings=_settings(), transport=transport
        )


def test_publish_raises_when_the_edit_is_rejected() -> None:
    transport = FakeTransport(edit_result="Failure")

    with pytest.raises(PublishError, match="edit was rejected"):
        publish_mapping(
            "Infobox airport", "Airport", SAMPLE_MAPPINGS, settings=_settings(), transport=transport
        )


def test_build_mapping_wikitext_round_trips_through_the_real_parser(tmp_path: Path) -> None:
    """The stated design goal: publish output and rag/ontology.py's parser
    share the same wikitext format."""

    from rag.ontology import AmharicMappingIndex

    wikitext = build_mapping_wikitext("Airport", SAMPLE_MAPPINGS)

    page_xml = (
        "<mediawiki><page><title>Mapping am:Infobox airport</title>"
        f"<revision><text>{wikitext}</text></revision></page></mediawiki>"
    )
    mapping_path = tmp_path / "mapping_am.xml"
    mapping_path.write_text(page_xml, encoding="utf-8")

    index = AmharicMappingIndex.from_mapping_xml(mapping_path)

    icao = index.lookup("አይካኦ_ኮድ")
    assert icao is not None
    assert icao.ontology_property == "icaoLocationIdentifier"
    iata = index.lookup("አያታ_ኮድ")
    assert iata is not None
    assert iata.ontology_property == "iataLocationIdentifier"


EXISTING_PLACE_PAGE = """{{TemplateMapping
 | mapToClass = Place
 | mappings =
\t{{ PropertyMapping | templateProperty = ስም | ontologyProperty = foaf:name | language=am }}
\t{{IntermediateNodeMapping
\t| nodeClass = Demographics
\t| correspondingProperty = demographics
\t| mappings =
\t\t\t{{ PropertyMapping | templateProperty = ሕዝብ_ጠቅላላ | ontologyProperty = populationTotal }}
\t}}
\t{{ PropertyMapping | templateProperty = ከፍታ | ontologyProperty = height | unit = metre }}
}}"""


def test_publishing_onto_an_existing_page_merges_instead_of_replacing_it() -> None:
    transport = FakeTransport(existing_page=EXISTING_PLACE_PAGE)

    publish_mapping(
        "የቦታ መረጃ",
        "City",
        [
            {"templateProperty": "ከፍታ", "ontologyProperty": "elevation"},  # correction
            {"templateProperty": "ሕዝብ ጠቅላላ", "ontologyProperty": "populationTotal"},  # same
            {"templateProperty": "ሰዓት_ክልል", "ontologyProperty": "timeZone"},  # new
        ],
        settings=_settings(),
        transport=transport,
    )

    text = transport.post_calls[-1]["text"]
    # Nothing that was already on the page is lost.
    assert "foaf:name | language=am" in text
    assert "nodeClass = Demographics" in text
    assert text.count("populationTotal") == 1
    # The correction keeps the row's other parameters.
    assert "ontologyProperty = elevation | unit = metre" in text.replace("  ", " ")
    assert "height" not in text
    # The new field is added; the existing class is kept.
    assert "templateProperty = ሰዓት_ክልል | ontologyProperty = timeZone" in text
    assert "mapToClass = Place" in text and "City" not in text


def test_merge_counts_and_round_trips_through_the_real_parser(tmp_path: Path) -> None:
    text, counts = merge_mapping_wikitext(
        EXISTING_PLACE_PAGE, "Place", [{"templateProperty": "ስም", "ontologyProperty": "foaf:name"}]
    )

    assert counts == {"added": 0, "updated": 0, "unchanged": 1}
    assert text == EXISTING_PLACE_PAGE


def test_publishing_identical_rows_onto_an_existing_page_makes_no_edit() -> None:
    transport = FakeTransport(existing_page=EXISTING_PLACE_PAGE)

    publish_mapping(
        "የቦታ መረጃ",
        "Place",
        [{"templateProperty": "ስም", "ontologyProperty": "foaf:name"}],
        settings=_settings(),
        transport=transport,
    )

    assert [call["action"] for call in transport.post_calls] == ["login"]


def test_a_conditional_mapping_page_is_never_overwritten() -> None:
    transport = FakeTransport(existing_page="{{ConditionalMapping | cases = ... }}")

    with pytest.raises(PublishError, match="not a plain TemplateMapping"):
        publish_mapping(
            "የቦታ መረጃ",
            "Place",
            [{"templateProperty": "ስም", "ontologyProperty": "name"}],
            settings=_settings(),
            transport=transport,
        )

    assert [call["action"] for call in transport.post_calls] == ["login"]
