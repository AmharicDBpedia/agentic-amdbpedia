from __future__ import annotations

import json
import urllib.error
from unittest import mock

import pytest

from mcp_server.wiki_fetch import (
    WikipediaFetchError,
    _http_get,
    fetch_article_wikitext,
    parse_article_url,
)

GERD_WIKITEXT = "{{መረጃሳጥን ግድብ\n| ስም = ታላቁ ግድብ\n| ወንዝ = አባይ ወንዝ\n}}\n\n'''ታላቁ ግድብ''' ...\n"


def test_parse_article_url_handles_the_wiki_path_form() -> None:
    base_url, title = parse_article_url("https://am.wikipedia.org/wiki/ታላቁ_የኢትዮጵያ_ሕዳሴ_ግድብ")

    assert base_url == "https://am.wikipedia.org"
    assert title == "ታላቁ_የኢትዮጵያ_ሕዳሴ_ግድብ"


def test_parse_article_url_handles_the_index_php_query_form() -> None:
    base_url, title = parse_article_url(
        "https://am.wikipedia.org/w/index.php?title=ታላቁ_የኢትዮጵያ_ሕዳሴ_ግድብ&action=view"
    )

    assert base_url == "https://am.wikipedia.org"
    assert title == "ታላቁ_የኢትዮጵያ_ሕዳሴ_ግድብ"


def test_parse_article_url_accepts_any_wikipedia_language_edition() -> None:
    base_url, title = parse_article_url("https://en.wikipedia.org/wiki/Ethiopia")

    assert base_url == "https://en.wikipedia.org"
    assert title == "Ethiopia"


@pytest.mark.parametrize(
    "url",
    [
        "https://mappings.dbpedia.org/wiki/Something",  # real wiki, wrong host
        "https://evil.example.com/wikipedia.org/wiki/Trick",  # not actually wikipedia.org
        "https://wikipedia.org.evil.example.com/wiki/Trick",  # suffix trick
        "ftp://am.wikipedia.org/wiki/Title",  # not http(s)
        "not a url at all",
        "",
    ],
)
def test_parse_article_url_rejects_non_wikipedia_hosts(url: str) -> None:
    with pytest.raises(WikipediaFetchError):
        parse_article_url(url)


def test_parse_article_url_rejects_a_link_with_no_title() -> None:
    with pytest.raises(WikipediaFetchError, match="title"):
        parse_article_url("https://am.wikipedia.org/w/index.php?action=view")


def test_fetch_article_wikitext_returns_title_and_wikitext() -> None:
    captured: dict[str, str] = {}

    def fake_fetch(url: str) -> bytes:
        captured["url"] = url
        return GERD_WIKITEXT.encode("utf-8")

    title, wikitext = fetch_article_wikitext(
        "https://am.wikipedia.org/wiki/ታላቁ_የኢትዮጵያ_ሕዳሴ_ግድብ", fetch=fake_fetch
    )

    assert title == "ታላቁ_የኢትዮጵያ_ሕዳሴ_ግድብ"
    assert wikitext == GERD_WIKITEXT
    assert "action=raw" in captured["url"]
    assert "am.wikipedia.org" in captured["url"]


def test_fetch_article_wikitext_wraps_a_fetch_failure() -> None:
    def failing_fetch(url: str) -> bytes:
        raise WikipediaFetchError("HTTP 404: Not Found")

    with pytest.raises(WikipediaFetchError):
        fetch_article_wikitext("https://am.wikipedia.org/wiki/DoesNotExist", fetch=failing_fetch)


def test_fetch_article_wikitext_rejects_empty_body() -> None:
    def empty_fetch(url: str) -> bytes:
        return b""

    with pytest.raises(WikipediaFetchError, match="empty"):
        fetch_article_wikitext("https://am.wikipedia.org/wiki/Blank", fetch=empty_fetch)


def test_fetch_article_wikitext_rejects_non_utf8_body() -> None:
    def bad_encoding_fetch(url: str) -> bytes:
        return b"\xff\xfe not utf-8"

    with pytest.raises(WikipediaFetchError, match="UTF-8"):
        fetch_article_wikitext("https://am.wikipedia.org/wiki/Bad", fetch=bad_encoding_fetch)


@pytest.mark.parametrize(
    ("code", "message"),
    [(404, "Article not found on Wikipedia"), (503, "Wikipedia returned HTTP 503")],
)
def test_http_errors_are_not_reported_as_unreachable(code: int, message: str) -> None:
    error = urllib.error.HTTPError("https://am.wikipedia.org/x", code, "err", {}, None)  # type: ignore[arg-type]
    with (
        mock.patch("urllib.request.urlopen", side_effect=error),
        pytest.raises(WikipediaFetchError, match=message) as excinfo,
    ):
        _http_get("https://am.wikipedia.org/x")
    assert "Could not reach" not in str(excinfo.value)


def test_fetch_article_wikitext_follows_redirects() -> None:
    pages = {
        "ባሕር_ዳር": b"#REDIRECT [[\xe1\x89\xa3\xe1\x88\x95\xe1\x88\xad-\xe1\x8b\xb3\xe1\x88\xad]]",
        "ባሕር-ዳር": "{{መረጃሳጥን ከተማ\n| ስም = ባሕር ዳር\n}}".encode(),
    }
    requested: list[str] = []

    def fake_fetch(raw_url: str) -> bytes:
        import urllib.parse

        title = urllib.parse.parse_qs(urllib.parse.urlparse(raw_url).query)["title"][0]
        requested.append(title)
        return pages[title]

    title, text = fetch_article_wikitext("https://am.wikipedia.org/wiki/ባሕር_ዳር", fetch=fake_fetch)

    assert requested == ["ባሕር_ዳር", "ባሕር-ዳር"]
    assert title == "ባሕር-ዳር"
    assert text.startswith("{{መረጃሳጥን ከተማ")


def test_fetch_article_wikitext_stops_on_a_redirect_loop() -> None:
    def looping_fetch(_raw_url: str) -> bytes:
        return b"#REDIRECT [[Loop]]"

    with pytest.raises(WikipediaFetchError, match="Too many redirects"):
        fetch_article_wikitext("https://am.wikipedia.org/wiki/Loop", fetch=looping_fetch)


def test_transient_connection_errors_are_retried() -> None:
    attempts: list[int] = []

    class FakeResponse:
        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return b"{{Infobox}}"

    def flaky(*_args: object, **_kwargs: object) -> FakeResponse:
        attempts.append(1)
        if len(attempts) < 3:
            raise urllib.error.URLError(OSError(101, "Network is unreachable"))
        return FakeResponse()

    with mock.patch("urllib.request.urlopen", side_effect=flaky):
        assert _http_get("https://am.wikipedia.org/x", sleep=lambda _s: None) == b"{{Infobox}}"
    assert len(attempts) == 3


def test_a_404_is_not_retried() -> None:
    error = urllib.error.HTTPError("https://am.wikipedia.org/x", 404, "nf", {}, None)  # type: ignore[arg-type]
    with (
        mock.patch("urllib.request.urlopen", side_effect=error) as urlopen,
        pytest.raises(WikipediaFetchError, match="Article not found"),
    ):
        _http_get("https://am.wikipedia.org/x", sleep=lambda _s: None)
    assert urlopen.call_count == 1


def test_canonical_template_name_follows_a_template_redirect() -> None:
    from mcp_server.wiki_fetch import canonical_template_name

    payload = {
        "query": {
            "redirects": [{"from": "Template:ካውንቲ መረጃ", "to": "መለጠፊያ:መረጃሳጥን ካውንቲ መረጃ"}],
            "pages": [{"title": "መለጠፊያ:መረጃሳጥን ካውንቲ መረጃ"}],
        }
    }
    canonical_template_name.cache_clear()
    with mock.patch("mcp_server.wiki_fetch._http_get", return_value=json.dumps(payload).encode()):
        assert canonical_template_name("ካውንቲ መረጃ") == "መረጃሳጥን ካውንቲ መረጃ"


def test_canonical_template_name_keeps_the_name_when_offline() -> None:
    from mcp_server.wiki_fetch import canonical_template_name

    canonical_template_name.cache_clear()
    with mock.patch("mcp_server.wiki_fetch._http_get", side_effect=WikipediaFetchError("offline")):
        assert canonical_template_name("የቦታ መረጃ") == "የቦታ መረጃ"


TEMPLATE_DEFINITION = (
    '<includeonly>{| class="infobox hrecipe"\n'
    "! {{{ስም|}}}\n"
    "{{#if: {{{ዓይነት|}}}| ! ዓይነት {{!}} {{{ዓይነት}}} }}\n"
    "{{#if: {{{ዋና_ግብዓቶች|}}}| ! {{{ዋና_ግብዓቶች}}} }}\n"
    "|}</includeonly><noinclude><templatedata>\n"
    '{"params": {"ስም": {}, "ዓይነት": {}, "ሀገር": {}, "ዋና_ግብዓቶች": {}, "ካሎሪ": {}}}\n'
    "</templatedata></noinclude>"
)


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("መለጠፊያ:የምግብ_መረጃ", True),
        ("Template:Infobox food", True),
        ("template:infobox food", True),
        ("ታላቁ_የኢትዮጵያ_ሕዳሴ_ግድብ", False),
        ("Ethiopia", False),
    ],
)
def test_is_template_title(title: str, expected: bool) -> None:
    from mcp_server.wiki_fetch import is_template_title

    assert is_template_title(title) is expected


def test_template_display_name_strips_the_namespace_prefix() -> None:
    from mcp_server.wiki_fetch import template_display_name

    assert template_display_name("መለጠፊያ:የምግብ_መረጃ") == "የምግብ_መረጃ"
    assert template_display_name("Template:Infobox food") == "Infobox food"


def test_extract_template_params_prefers_templatedata() -> None:
    from mcp_server.wiki_fetch import extract_template_params

    assert extract_template_params(TEMPLATE_DEFINITION) == [
        "ስም",
        "ዓይነት",
        "ሀገር",
        "ዋና_ግብዓቶች",
        "ካሎሪ",
    ]


def test_extract_template_params_falls_back_to_triple_braces() -> None:
    from mcp_server.wiki_fetch import extract_template_params

    # No <templatedata> block -- parameters are only visible as references,
    # and a positional {{{1}}} is dropped as having no label to map.
    wikitext = "{{{ስም|}}} {{{የስዕል_ስፋት|240}}} {{{ስም}}} {{{1}}}"
    assert extract_template_params(wikitext) == ["ስም", "የስዕል_ስፋት"]


def test_synthesize_infobox_wikitext_is_recognized_by_the_pipeline() -> None:
    from mcp_server.pipeline import extract_first_infobox
    from mcp_server.wiki_fetch import (
        extract_template_params,
        synthesize_infobox_wikitext,
        template_display_name,
    )

    name = template_display_name("መለጠፊያ:የምግብ_መረጃ")
    params = extract_template_params(TEMPLATE_DEFINITION)
    template = extract_first_infobox(synthesize_infobox_wikitext(name, params))

    assert template is not None
    assert template.name == "የምግብ_መረጃ"
    assert [field.name for field in template.fields] == params
