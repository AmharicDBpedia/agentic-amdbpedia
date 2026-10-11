"""Consent-gated MediaWiki publish (refs implementation.md 14.3).

Publishing a mapping is a real, outward-facing, hard-to-reverse write to
the live mappings.dbpedia.org wiki, under a bot account (Bot Password —
`Special:BotPasswords` — only, never a real user password). `publish_mapping`
itself never checks consent; per `mcp_server.consent`'s own decorator
-factory design, a caller must wrap the call site itself:
`require_consent(approved=user_said_yes)(publish_mapping)(...)`. This
module is never exercised against the real live wiki in tests — every test
here runs against an injected fake transport, matching how
`scripts/refresh_wiki_cache.py` stays testable without live network
access, except here a live call would be a real, irreversible write, so no
"proven live" claim is made or attempted for this one.
"""

from __future__ import annotations

import http.cookiejar
import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Protocol

from config import Settings
from errors import ClientSafeError
from logging_config import log_event

LOGGER = logging.getLogger("dbpedia_mapping_assistant.publish")

_HEADERS = {"User-Agent": "agentic-amdbpedia/publish"}


class PublishError(ClientSafeError):
    """Raised when a publish attempt fails for any reason (network, auth,
    a rejected edit) — never leaks raw credentials or response bodies."""

    def __init__(self, message: str) -> None:
        super().__init__(message=message, error_type="publish_failed")


class MediaWikiCredentialsError(PublishError):
    def __init__(self) -> None:
        super().__init__("MediaWiki bot credentials are not configured")


class MediaWikiTransport(Protocol):
    """GET/POST against the MediaWiki API, sharing session state (cookies)
    across calls — the login -> csrf-token -> edit sequence needs the same
    authenticated session throughout."""

    def get(self, params: dict[str, str]) -> dict[str, Any]: ...
    def post(self, data: dict[str, str]) -> dict[str, Any]: ...


class RealMediaWikiTransport:
    """The real transport: a cookie-jar-aware urllib opener. Never used in
    tests — see this module's docstring."""

    def __init__(self, base_url: str, *, timeout: float = 30.0) -> None:
        self.api_url = f"{base_url.rstrip('/')}/api.php"
        self.timeout = timeout
        self._cookie_jar = http.cookiejar.CookieJar()
        self._opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(self._cookie_jar)
        )

    def get(self, params: dict[str, str]) -> dict[str, Any]:
        query = urllib.parse.urlencode(params)
        request = urllib.request.Request(f"{self.api_url}?{query}", headers=_HEADERS)
        return self._call(request)

    def post(self, data: dict[str, str]) -> dict[str, Any]:
        encoded = urllib.parse.urlencode(data).encode("utf-8")
        request = urllib.request.Request(
            self.api_url, data=encoded, headers=_HEADERS, method="POST"
        )
        return self._call(request)

    def _call(self, request: urllib.request.Request) -> dict[str, Any]:
        try:
            with self._opener.open(request, timeout=self.timeout) as response:  # noqa: S310
                body = response.read()
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            raise PublishError(f"MediaWiki API request failed: {exc.__class__.__name__}") from exc
        try:
            payload: dict[str, Any] = json.loads(body)
        except json.JSONDecodeError as exc:
            raise PublishError("MediaWiki API returned a non-JSON response") from exc
        return payload


def build_mapping_wikitext(domain_class: str, mappings: list[dict[str, str]]) -> str:
    """Build the `{{TemplateMapping | mapToClass = ... | mappings = ...}}`
    wikitext `rag/ontology.py`'s `AmharicMappingIndex` already parses —
    publish output and the parser share the exact same format on purpose,
    so a page this function writes is immediately readable by this repo's
    own corpus the next time it refreshes."""

    lines = ["{{TemplateMapping", f" | mapToClass = {domain_class}", " | mappings ="]
    for mapping in mappings:
        lines.append(
            f"  {{{{PropertyMapping | templateProperty = {mapping['templateProperty']}"
            f" | ontologyProperty = {mapping['ontologyProperty']} }}}}"
        )
    lines.append("}}")
    return "\n".join(lines)


def _field_key(name: str) -> str:
    return "_".join(name.replace("_", " ").split()).casefold()


class MappingMergeError(PublishError):
    """The existing page can't be merged automatically (e.g. a ConditionalMapping)."""


def merge_mapping_wikitext(
    existing: str | None, domain_class: str, mappings: list[dict[str, str]]
) -> tuple[str, dict[str, int]]:
    """Merge `mappings` into an existing `Mapping am:` page instead of replacing it.

    Publishing used to overwrite the whole page with only the rows sent from
    the chat -- on a template that already had a mapping, that silently
    deleted every other property mapping on the live wiki. Now:

    - a field already mapped anywhere on the page (including inside an
      IntermediateNodeMapping) keeps its place and its other parameters
      (`unit`, `language`, ...); only `ontologyProperty` changes;
    - a new field is appended to the top-level `mappings` list;
    - `mapToClass` of an existing page is left as is.

    Returns `(wikitext, {"added": n, "updated": n, "unchanged": n})`.
    """

    if not existing or not existing.strip():
        return build_mapping_wikitext(domain_class, mappings), {
            "added": len(mappings),
            "updated": 0,
            "unchanged": 0,
        }

    import mwparserfromhell

    code = mwparserfromhell.parse(existing)
    roots = [
        t for t in code.filter_templates(recursive=False) if t.name.strip() == "TemplateMapping"
    ]
    if not roots:
        raise MappingMergeError(
            "The existing mapping page is not a plain TemplateMapping; edit it on the wiki instead"
        )
    root = roots[0]
    if not root.has("mappings"):
        root.add("mappings", "\n")

    by_field: dict[str, Any] = {}
    for node in root.get("mappings").value.filter_templates(recursive=True):
        if node.name.strip() == "PropertyMapping" and node.has("templateProperty"):
            by_field.setdefault(_field_key(str(node.get("templateProperty").value)), node)

    counts = {"added": 0, "updated": 0, "unchanged": 0}
    additions: list[str] = []
    for mapping in mappings:
        target: Any = by_field.get(_field_key(mapping["templateProperty"]))
        if target is None:
            additions.append(
                f"  {{{{PropertyMapping | templateProperty = {mapping['templateProperty']}"
                f" | ontologyProperty = {mapping['ontologyProperty']} }}}}"
            )
            counts["added"] += 1
            continue
        current = (
            str(target.get("ontologyProperty").value).strip()
            if target.has("ontologyProperty")
            else ""
        )
        if current == mapping["ontologyProperty"]:
            counts["unchanged"] += 1
            continue
        target.add("ontologyProperty", f" {mapping['ontologyProperty']} ")
        counts["updated"] += 1

    if additions:
        value = root.get("mappings").value
        text = str(value).rstrip()
        root.get("mappings").value = text + "\n" + "\n".join(additions) + "\n"
    return str(code), counts


def _fetch_page_text(transport: MediaWikiTransport, title: str) -> str | None:
    response = transport.get(
        {
            "action": "query",
            "prop": "revisions",
            "rvprop": "content",
            "rvslots": "main",
            "titles": title,
            "formatversion": "2",
            "format": "json",
        }
    )
    pages = response.get("query", {}).get("pages", [])
    if not pages or pages[0].get("missing"):
        return None
    revisions = pages[0].get("revisions") or []
    if not revisions:
        return None
    slot = revisions[0].get("slots", {}).get("main", {})
    content = slot.get("content", revisions[0].get("content"))
    return str(content) if content is not None else None


def _login(transport: MediaWikiTransport, username: str, password: str) -> None:
    token_response = transport.get(
        {"action": "query", "meta": "tokens", "type": "login", "format": "json"}
    )
    try:
        login_token = token_response["query"]["tokens"]["logintoken"]
    except KeyError as exc:
        raise PublishError("Could not obtain a MediaWiki login token") from exc

    login_response = transport.post(
        {
            "action": "login",
            "lgname": username,
            "lgpassword": password,
            "lgtoken": login_token,
            "format": "json",
        }
    )
    result = login_response.get("login", {}).get("result")
    if result != "Success":
        raise PublishError(f"MediaWiki login failed: {result or _api_error(login_response)}")


def _api_error(response: dict[str, Any]) -> str:
    """The wiki's own reason when it answers with an API error instead of a
    login result. mappings.dbpedia.org answers every login with
    `writeapidenied` because anonymous users lack the `writeapi` right."""

    error = response.get("error")
    if not isinstance(error, dict):
        return "unknown error"
    code, info = error.get("code"), error.get("info")
    if code == "writeapidenied":
        return (
            "the wiki does not allow logging in through its API (writeapidenied: "
            "anonymous users lack the 'writeapi' right). A mappings.dbpedia.org "
            "admin has to grant it; your bot permissions are not the problem."
        )
    return f"{code}: {info}" if code else "unknown error"


def _csrf_token(transport: MediaWikiTransport) -> str:
    response = transport.get(
        {"action": "query", "meta": "tokens", "type": "csrf", "format": "json"}
    )
    try:
        token: str = response["query"]["tokens"]["csrftoken"]
    except KeyError as exc:
        raise PublishError("Could not obtain a MediaWiki CSRF token") from exc
    return token


def _edit_page(
    transport: MediaWikiTransport, title: str, text: str, summary: str, token: str
) -> None:
    response = transport.post(
        {
            "action": "edit",
            "title": title,
            "text": text,
            "summary": summary,
            "bot": "1",
            "token": token,
            "format": "json",
        }
    )
    result = response.get("edit", {}).get("result")
    if result != "Success":
        raise PublishError(f"MediaWiki edit was rejected: {result or 'unknown error'}")


def publish_mapping(
    template_name: str,
    domain_class: str,
    mappings: list[dict[str, str]],
    *,
    settings: Settings | None = None,
    transport: MediaWikiTransport | None = None,
    summary: str = "Publish via cross-lingual mapping assistant",
) -> str:
    """Log in as the bot, fetch a CSRF token, and edit `Mapping am:<template_name>`.

    Returns the published page title on success. The caller is responsible
    for the consent gate — see this module's docstring — and for flipping
    the review item's status to "published" and firing the eager corpus
    -refresh hook (12.2) afterward; this function only performs the write.
    """

    resolved_settings = settings or Settings()
    if not resolved_settings.mediawiki_bot_username or not resolved_settings.mediawiki_bot_password:
        raise MediaWikiCredentialsError()

    resolved_transport = transport or RealMediaWikiTransport(resolved_settings.mediawiki_base_url)

    _login(
        resolved_transport,
        resolved_settings.mediawiki_bot_username,
        resolved_settings.mediawiki_bot_password,
    )
    token = _csrf_token(resolved_transport)
    title = f"Mapping am:{template_name}"
    existing = _fetch_page_text(resolved_transport, title)
    wikitext, counts = merge_mapping_wikitext(existing, domain_class, mappings)
    if existing is not None and counts["added"] == 0 and counts["updated"] == 0:
        log_event(LOGGER, "publish.nothing_to_change", title=title)
        return title
    _edit_page(resolved_transport, title, wikitext, summary, token)

    log_event(LOGGER, "publish.completed", title=title, merged=existing is not None, **counts)
    return title
