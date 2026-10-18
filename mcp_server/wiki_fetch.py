"""Fetch a Wikipedia article's wikitext by URL (refs the "paste a Wikipedia
link" chat/preview flow).

Deliberately restricted to `*.wikipedia.org` hosts: this fetches whatever
URL a caller supplies over the live HTTP server, so an unrestricted fetch
would be an open SSRF proxy, not a feature. `action=raw` returns the
article's raw wikitext directly (no JSON envelope, no rendered HTML) --
the exact shape `mcp_server.pipeline.extract_first_infobox` already parses.
Fetching the *whole* article and handing it to the existing pipeline
unmodified is enough: `extract_first_infobox` already finds "the first
infobox-like template" within arbitrarily larger wikitext, so there is no
need to special-case "just the infobox part" of the page here.

Same `Fetcher`-parameter shape as `scripts/refresh_wiki_cache.py::http_get`
on purpose -- tests inject a fake instead of touching the real network,
matching how every other real HTTP call in this repo is already tested.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from functools import lru_cache

from errors import ClientSafeError

_HEADERS = {"User-Agent": "agentic-amdbpedia/wiki-fetch"}
# 15 s was measured live to time out under a busy connection (a large
# download running alongside); retries handle the rest.
DEFAULT_TIMEOUT = 30.0

# Matches "wikipedia.org" itself and any subdomain (am.wikipedia.org,
# en.wikipedia.org, ...) -- this project is Amharic-focused, but nothing
# about extract_first_infobox is Amharic-specific, so no reason to reject
# a different language edition's article.
_ALLOWED_HOST_RE = re.compile(r"^([a-z0-9-]+\.)?wikipedia\.org$", re.IGNORECASE)


class WikipediaFetchError(ClientSafeError):
    """Raised for anything that stops a real article's wikitext from being
    fetched: an unsupported/malformed URL, a page that doesn't exist, or a
    real network failure. Always client-safe -- never leaks the raw
    exception or response body."""

    def __init__(self, message: str) -> None:
        super().__init__(message=message, error_type="wikipedia_fetch_failed")


def parse_article_url(url: str) -> tuple[str, str]:
    """Return `(base_url, page_title)` from a real Wikipedia article URL.

    Accepts both the common `/wiki/<Title>` path form and the
    `?title=<Title>` query form (index.php-style links) -- both are real,
    valid ways an Amharic Wikipedia article link actually looks in
    practice, verified live against am.wikipedia.org."""

    try:
        parsed = urllib.parse.urlsplit(url.strip())
    except ValueError as exc:
        raise WikipediaFetchError(f"Not a valid URL: {url!r}") from exc

    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise WikipediaFetchError(f"Not a valid URL: {url!r}")
    if not _ALLOWED_HOST_RE.match(parsed.hostname or ""):
        raise WikipediaFetchError(
            f"Only wikipedia.org article links are supported, got host {parsed.hostname!r}"
        )

    base_url = f"{parsed.scheme}://{parsed.netloc}"

    if parsed.path.startswith("/wiki/"):
        title = urllib.parse.unquote(parsed.path[len("/wiki/") :])
    else:
        title = urllib.parse.parse_qs(parsed.query).get("title", [""])[0]

    if not title:
        raise WikipediaFetchError(f"Could not find an article title in {url!r}")

    return base_url, title


Fetcher = Callable[[str], bytes]

MAX_REDIRECTS = 3
# "#REDIRECT [[Target]]" / "#REDIRECT [[Target#Section|label]]"; MediaWiki
# accepts the English keyword on every language edition.
_REDIRECT = re.compile(r"^\s*#REDIRECT\s*\[\[([^\]|#]+)", re.IGNORECASE)


# Connection-level failures (DNS, "Network is unreachable", resets) were
# seen live to come and go within seconds and cost whole articles in a
# batch run; they are retried. An HTTP answer (404, 5xx) is never retried.
CONNECT_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 2.0


def _http_get(url: str, *, sleep: Callable[[float], None] = time.sleep) -> bytes:
    request = urllib.request.Request(url, headers=_HEADERS)
    for attempt in range(CONNECT_ATTEMPTS):
        try:
            return _open(request)
        except WikipediaFetchError:
            raise
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            if attempt == CONNECT_ATTEMPTS - 1:
                raise WikipediaFetchError(f"Could not reach {url}: {exc}") from exc
            sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))
    raise WikipediaFetchError(f"Could not reach {url}")  # pragma: no cover


def _open(request: urllib.request.Request) -> bytes:
    url = request.full_url
    try:
        with urllib.request.urlopen(request, timeout=DEFAULT_TIMEOUT) as response:  # noqa: S310 - host allowlisted in parse_article_url
            body: bytes = response.read()
            return body
    except urllib.error.HTTPError as exc:
        # Wikipedia answered -- "Could not reach" would send the user off
        # debugging their network for what is really a typo in the title.
        if exc.code == 404:
            raise WikipediaFetchError(
                "Article not found on Wikipedia -- check the title in the link"
            ) from exc
        raise WikipediaFetchError(f"Wikipedia returned HTTP {exc.code} for {url}") from exc


def fetch_article_wikitext(url: str, *, fetch: Fetcher = _http_get) -> tuple[str, str]:
    """Return `(page_title, wikitext)` for a real Wikipedia article URL.

    `fetch` overrides the real HTTP GET -- tests always inject a fake, the
    same pattern `scripts/refresh_wiki_cache.py`'s own fetch functions
    already use."""

    base_url, title = parse_article_url(url)

    # `action=raw` returns a redirect page's own source ("#REDIRECT
    # [[ባሕር-ዳር]]"), not its target -- confirmed live for ባሕር_ዳር and
    # አዲስ_አበባ_ዩኒቨርሲቲ, which then looked like articles with no infobox.
    for _hop in range(MAX_REDIRECTS + 1):
        raw_url = f"{base_url}/w/index.php?" + urllib.parse.urlencode(
            {"title": title, "action": "raw"}
        )
        body = fetch(raw_url)
        try:
            text = body.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise WikipediaFetchError(f"Article {title!r} was not valid UTF-8") from exc

        if not text.strip():
            raise WikipediaFetchError(f"Article {title!r} appears to be empty or does not exist")

        redirect = _REDIRECT.match(text)
        if redirect is None:
            return title, text
        title = redirect.group(1).strip().replace(" ", "_")

    raise WikipediaFetchError(f"Too many redirects while resolving {url}")


DEFAULT_WIKI_BASE = "https://am.wikipedia.org"


@lru_cache(maxsize=1024)
def canonical_template_name(name: str, base_url: str = DEFAULT_WIKI_BASE) -> str:
    """The template a redirect like `{{ካውንቲ መረጃ}}` points to
    (`መረጃሳጥን ካውንቲ መረጃ`), or `name` itself.

    DBpedia's extraction framework resolves template redirects before it
    looks up `Mapping am:<template>`, so a mapping published under a
    redirect's name is never used. Any failure (offline, API error) keeps
    the name as written.
    """

    if not name.strip():
        return name
    query = urllib.parse.urlencode(
        {
            "action": "query",
            "titles": f"Template:{name.strip()}",
            "redirects": "1",
            "format": "json",
            "formatversion": "2",
        }
    )
    try:
        payload = json.loads(_http_get(f"{base_url.rstrip('/')}/w/api.php?{query}"))
    except Exception:  # noqa: BLE001 - resolution is best-effort
        return name
    pages = payload.get("query", {}).get("pages", [])
    if not pages or pages[0].get("missing"):
        return name
    title = str(pages[0].get("title", ""))
    # Strip the localized namespace ("መለጠፊያ:" on am.wikipedia).
    resolved = title.split(":", 1)[1] if ":" in title else title
    return resolved or name


# Template-definition pages (am.wikipedia's `መለጠፊያ:` namespace, the
# English `Template:`) name their parameters as triple-brace references --
# `{{{ስም|}}}`, `{{{የስዕል_ስፋት|240}}}` -- and usually also list them in a
# `<templatedata>` JSON block. Mapping *from the template page itself* means
# reading those parameter names rather than looking for a filled-in infobox
# instance (a definition has none).
TEMPLATE_NAMESPACE_PREFIXES = ("መለጠፊያ:", "Template:")
_TEMPLATEDATA_RE = re.compile(r"<templatedata>(.*?)</templatedata>", re.IGNORECASE | re.DOTALL)
_TRIPLE_BRACE_PARAM_RE = re.compile(r"\{\{\{\s*([^|{}]+?)\s*[|}]")


def is_template_title(title: str) -> bool:
    """True for a page in a Template namespace -- a template *definition*
    page, not an article that merely uses one."""

    stripped = title.lstrip().casefold()
    return any(stripped.startswith(prefix.casefold()) for prefix in TEMPLATE_NAMESPACE_PREFIXES)


def template_display_name(title: str) -> str:
    """`መለጠፊያ:የምግብ_መረጃ` -> `የምግብ_መረጃ` -- the bare template name DBpedia
    maps under (`Mapping am:<name>`), with the namespace prefix removed."""

    return title.split(":", 1)[1].strip() if ":" in title else title.strip()


def extract_template_params(wikitext: str) -> list[str]:
    """Ordered, de-duplicated parameter names declared by a template
    definition.

    Prefers the `<templatedata>` block (authoritative and noise-free);
    falls back to scanning `{{{name|...}}}` references for templates that
    ship no TemplateData. Positional/numeric parameters are dropped --
    they have no Amharic label to map."""

    names = _params_from_templatedata(wikitext) or _TRIPLE_BRACE_PARAM_RE.findall(wikitext)
    ordered: dict[str, None] = {}
    for name in names:
        cleaned = str(name).strip()
        if cleaned and not cleaned.isdecimal():
            ordered.setdefault(cleaned, None)
    return list(ordered)


def _params_from_templatedata(wikitext: str) -> list[str]:
    match = _TEMPLATEDATA_RE.search(wikitext)
    if match is None:
        return []
    try:
        data = json.loads(match.group(1))
    except (ValueError, TypeError):
        return []
    params = data.get("params") if isinstance(data, dict) else None
    return [str(name) for name in params] if isinstance(params, dict) else []


def synthesize_infobox_wikitext(name: str, params: list[str]) -> str:
    """Render a template name and its parameters as a single infobox
    *instance* (`{{name | p = | q = ...}}`), so a fetched template
    *definition* runs through the exact same extract -> predict -> format
    pipeline a pasted or article infobox does. Values are intentionally
    empty: the pipeline maps parameter *names* (Amharic property labels),
    not their values."""

    body = "".join(f"| {param} = \n" for param in params)
    return f"{{{{{name}\n{body}}}}}"
