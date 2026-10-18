"""Groq as the conversational front and back of the mapping chat.

The mapping itself (retrieval + Gemma via Ollama, see `rag.predict`) never
changes here. Groq only does the two language jobs around it:

1. `understand()` -- turn whatever the user typed into a plan the pipeline
   can run: an infobox, a Wikipedia link, or a single field name pulled out
   of a free-form question ("what is ዋና_አዘጋጅ on a newspaper?"), plus the
   DBpedia class it belongs to. The class is only kept if it is a real
   `OntologyClass:` in the cached ontology.
2. `summarize()` -- turn the pipeline's structured result into a short,
   readable reply: what mapped, what the language model proposed and should
   be checked, what is still unmapped and the closest suggestions, and that
   the table can be edited before publishing. A reply that names a property
   not present in the result is discarded in favour of a deterministic
   summary, so the prose can never contradict the table.

Both degrade to deterministic rules when Groq is unavailable; neither ever
blocks a request. Prompt-injection text is rejected before it reaches Groq.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

from logging_config import log_event
from mcp_server.guardrails import is_injection_attempt

LOGGER = logging.getLogger("dbpedia_mapping_assistant.assistant")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ONTOLOGY_XML = PROJECT_ROOT / "data" / "wiki_cache" / "ontology.xml"

URL_RE = re.compile(r"https?://[a-z\-]+\.(?:m\.)?wikipedia\.org/\S+", re.IGNORECASE)
CLASS_RE = re.compile(r"^[A-Z][A-Za-z0-9]*$")
ETHIOPIC_RE = re.compile(r"[ሀ-፿]")
TEMPLATE_NAME_RE = re.compile(r"\{\{\s*([^|}\n]+)")
FIELD_RE = re.compile(r"^\s*\|\s*([^=|\n]+?)\s*=", re.MULTILINE)
# An infobox-style field name written with underscores, e.g. ዋና_አዘጋጅ.
UNDERSCORED_FIELD_RE = re.compile(r"[\u1200-\u137FA-Za-z0-9]+(?:_[\u1200-\u137FA-Za-z0-9]+)+")
# camelCase identifiers (and prefixed ones like foaf:name) a summary might
# mention -- every one must exist in the result it is summarising.
PROPERTY_TOKEN_RE = re.compile(r"\b(?:[a-z]+:)?[a-z]+(?:[A-Z][a-z0-9]*)+\b")

INTENTS = {"infobox", "url", "field", "unsupported", "rejected"}


class ChatClient(Protocol):
    """The one call this module needs: chat messages in, text out."""

    def complete(self, messages: list[dict[str, str]], *, json_mode: bool) -> str: ...


@dataclass(frozen=True)
class InputPlan:
    intent: str
    # What the pipeline should run on: the raw infobox, the URL, or the field.
    text: str
    target_class: str | None = None
    language: str = "en"
    # Shown to the user when there is nothing to run (greeting, off-topic).
    reply: str | None = None
    source: str = "rules"

    def to_api_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Feedback:
    message: str
    source: str = "rules"
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------- classes ---


@lru_cache(maxsize=1)
def ontology_class_names(path: Path = ONTOLOGY_XML) -> dict[str, str]:
    """`{casefolded: canonical}` for every `OntologyClass:` in the ontology dump."""

    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return {}
    names = re.findall(r"<title>OntologyClass:([^<]+)</title>", text)
    return {name.strip().casefold(): name.strip() for name in names}


def canonical_class(name: str | None, *, classes: dict[str, str] | None = None) -> str | None:
    """A real DBpedia class name in its canonical spelling, or `None`."""

    if not name:
        return None
    cleaned = name.strip().removeprefix("dbo:").removeprefix("dbpedia-owl:")
    known = classes if classes is not None else ontology_class_names()
    if known:
        return known.get(cleaned.casefold())
    return cleaned if CLASS_RE.match(cleaned) else None


# ------------------------------------------------------------ understand ---


def _language_of(text: str) -> str:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return "en"
    ethiopic = sum(bool(ETHIOPIC_RE.match(c)) for c in letters)
    return "am" if ethiopic / len(letters) > 0.5 else "en"


def _rules_plan(text: str, target_class: str | None) -> InputPlan:
    if URL_RE.search(text):
        return InputPlan("url", URL_RE.search(text).group(0), target_class)  # type: ignore[union-attr]
    if "{{" in text:
        return InputPlan("infobox", text, target_class)
    return InputPlan("field", text.strip(), target_class, language=_language_of(text))


UNDERSTAND_PROMPT = """You route messages for an Amharic-Wikipedia-to-DBpedia mapping tool.
The tool maps Amharic infobox field names to DBpedia ontology properties.
Return JSON: {"intent": ..., "field": ..., "target_class": ..., "reply": ...}
- intent "field": the user wants the DBpedia property for one infobox field.
  "field" = that field name copied exactly as the user wrote it (keep Amharic,
  keep underscores). If they ask about a field in a sentence, extract just it --
  not the words around it that name the entity (in "የጋዜጣ ዋና_አዘጋጅ" the field
  is "ዋና_አዘጋጅ"; "የጋዜጣ" just means "of the newspaper").
- intent "infobox": the message contains a wiki template ({{ ... }}).
- intent "unsupported": greetings, chit-chat or anything that is not about
  mapping a field; put a one-sentence answer in "reply" saying what the tool
  can do, in the user's language.
- "target_class": the DBpedia ontology class (CamelCase, e.g. Newspaper,
  SoccerClub, University) the field or infobox describes, if the message or
  the template name makes it clear; otherwise null. For an infobox, infer it
  from the template name (e.g. "መረጃሳጥን ሙዚየም" -> Museum).
Never invent a field that is not in the message."""


def _json_object(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    try:
        value = json.loads(match.group(0) if match else raw)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _normalize_field(value: str) -> str:
    return re.sub(r"[\s_]+", "_", value.strip()).casefold()


def understand(
    text: str,
    *,
    target_class: str | None = None,
    client: ChatClient | None = None,
    classes: dict[str, str] | None = None,
) -> InputPlan:
    """Decide what to run for `text`. A user-supplied class always wins."""

    user_class = canonical_class(target_class, classes=classes) if target_class else None
    if is_injection_attempt(text):
        return InputPlan("rejected", text, user_class, reply="Rejected: prompt-injection attempt.")

    plan = _rules_plan(text, user_class)
    if client is None:
        return plan

    # A URL needs nothing from the model: the class comes from the article.
    if plan.intent == "url":
        return plan

    if plan.intent == "infobox":
        template = TEMPLATE_NAME_RE.search(text)
        fields = FIELD_RE.findall(text)[:15]
        prompt_text = (
            f"Template name: {template.group(1).strip() if template else '?'}\n"
            f"Fields: {', '.join(fields)}"
        )
    else:
        prompt_text = text

    try:
        raw = client.complete(
            [
                {"role": "system", "content": UNDERSTAND_PROMPT},
                {"role": "user", "content": prompt_text},
            ],
            json_mode=True,
        )
    except Exception as exc:  # noqa: BLE001 - routing must never fail a request
        log_event(LOGGER, "assistant.understand_failed", error=exc.__class__.__name__)
        return plan

    data = _json_object(raw)
    inferred_class = user_class or canonical_class(data.get("target_class"), classes=classes)

    if plan.intent == "infobox":
        # The infobox text itself is never rewritten by the model.
        return InputPlan("infobox", text, inferred_class, source="groq")

    intent = data.get("intent")
    if intent == "unsupported" and data.get("reply"):
        return InputPlan(
            "unsupported",
            text,
            inferred_class,
            language=_language_of(text),
            reply=str(data["reply"])[:400],
            source="groq",
        )

    field_name = str(data.get("field") or "").strip()
    # Only accept a field the user actually typed -- never one the model made up.
    if not field_name or _normalize_field(field_name) not in _normalize_field(text):
        field_name = text.strip()
    # A single underscored field name in the message is the field, exactly --
    # the model sometimes keeps a neighbouring word ("የጋዜጣ ዋና_አዘጋጅ").
    underscored = UNDERSCORED_FIELD_RE.findall(text)
    if len(underscored) == 1 and underscored[0] in field_name:
        field_name = underscored[0]
    return InputPlan(
        "field", field_name, inferred_class, language=_language_of(text), source="groq"
    )


# ------------------------------------------------------------- summarize ---


def _rows(result: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    mappings = [m for m in result.get("mappings", []) if isinstance(m, dict)]
    unmapped = [u for u in result.get("unmappedFields", []) if isinstance(u, dict)]
    return mappings, unmapped


def allowed_tokens(result: dict[str, Any]) -> set[str]:
    """Every identifier a summary may mention for this result."""

    mappings, unmapped = _rows(result)
    tokens: set[str] = set()
    for row in mappings:
        tokens.add(str(row.get("ontologyProperty", "")))
        tokens.add(str(row.get("templateProperty", "")))
    for row in unmapped:
        tokens.add(str(row.get("name", "")))
        tokens.update(str(s.get("property", "")) for s in row.get("suggestions", []))
    for match in result.get("matches", []):
        tokens.add(str(match.get("property", "")))
    for suggestion in result.get("suggestions", []):
        tokens.add(str(suggestion.get("property", "")))
    tokens.add(str(result.get("domainClass") or ""))
    expanded = set(tokens)
    for token in tokens:
        expanded.add(token.split(":")[-1])
    return {t for t in expanded if t}


def unknown_properties(message: str, allowed: Iterable[str]) -> list[str]:
    allowed_set = {a.casefold() for a in allowed}
    return [
        token
        for token in PROPERTY_TOKEN_RE.findall(message)
        if token.casefold() not in allowed_set
        and token.split(":")[-1].casefold() not in allowed_set
    ]


def rules_summary(result: dict[str, Any], *, kind: str) -> str:
    """Deterministic Amharic feedback; also the fallback whenever Groq can't
    be used, so the user is always answered in Amharic."""

    if kind == "field":
        matches = result.get("matches", [])
        if matches:
            top = matches[0]
            source = (top.get("payload") or {}).get("source")
            note = (
                " ይህን በቋንቋ ሞዴል የተጠቆመ በመሆኑ ከመጠቀምዎ በፊት ያረጋግጡ።"
                if source == "llm"
                else " ገምጋሚ ለዚሁ ክፍል ይህን ዝምድና ቀደም ሲል አትሞታል።"
                if source == "learned"
                else ""
            )
            return f"ለዚህ መስክ ተስማሚው DBpedia ባህርይ `{top['property']}` ነው።{note}"
        suggestions = [s["property"] for s in result.get("suggestions", [])][:3]
        tail = (
            f" በተመሳሳይነት በጣም ቅርብ የሆኑት {'፣ '.join(f'`{s}`' for s in suggestions)} ናቸው፣ "
            "ነገር ግን አንዳቸውም እርግጠኛ ዝምድና አይደሉም።"
            if suggestions
            else ""
        )
        llm = " የቋንቋ ሞዴሉ ሊገኝ አልቻለም።" if result.get("llm_error") else ""
        return f"ለዚህ መስክ እርግጠኛ የሆነ DBpedia ባህርይ ማግኘት አልቻልኩም።{tail}{llm}"

    mappings, unmapped = _rows(result)
    if not mappings and not unmapped:
        return "በዚህ ማስገቢያ ውስጥ ማዛመድ የሚገባው የመረጃ ሳጥን ማግኘት አልቻልኩም።"
    proposed = [m for m in mappings if m.get("source") == "llm"]
    class_part = f" ለ{result['domainClass']} ክፍል" if result.get("domainClass") else ""
    parts = [f"በድምሩ {len(mappings)} መስኮች{class_part} ተዛምደዋል።"]
    if proposed:
        names = "፣ ".join(f"{m['templateProperty']} → `{m['ontologyProperty']}`" for m in proposed)
        parts.append(f"{len(proposed)} ዝምድናዎች ({names}) በAI ሞዴል የተጠቆሙ ናቸው፤ እባክዎ ያረጋግጡ።")
    if unmapped:
        parts.append(
            f"{len(unmapped)} መስኮች አሁንም ባህርይ ያስፈልጋቸዋል፦ "
            + "፣ ".join(u["name"] for u in unmapped)
            + "። ጥቆማ ይምረጡ ወይም ከታች ይፈልጉ።"
        )
    parts.append("በሠንጠረዡ ውስጥ የፈለጉትን ያርሙ፣ ትክክል ሲሆን «Publish» ይጫኑ።")
    return " ".join(parts)


SUMMARIZE_PROMPT = """You write the final reply of a DBpedia mapping assistant.
You get the mapping result as JSON. Write 2-4 short sentences for the user:
how many fields were mapped, which mappings came from the language model
(source "llm") and need checking, which fields are still unmapped and their
closest suggestions, and that they can edit the table and then press Publish.
Use ONLY property and field names that appear in the JSON, written in
`backticks`. Do not add properties, do not explain DBpedia in general.
Never write internal labels such as "llm", "retrieval", "source" or JSON keys;
say "suggested by the AI model" (or its Amharic equivalent) instead.
Write in {language}."""


def summarize(
    result: dict[str, Any],
    *,
    kind: str,
    language: str = "am",
    client: ChatClient | None = None,
) -> Feedback:
    """Final feedback for a pipeline (`kind="infobox"`) or lookup (`"field"`) result."""

    fallback = rules_summary(result, kind=kind)
    if client is None:
        return Feedback(fallback)

    compact = {
        "domainClass": result.get("domainClass"),
        "mappings": [
            {k: m.get(k) for k in ("templateProperty", "ontologyProperty", "source")}
            for m in result.get("mappings", [])
        ],
        "unmappedFields": [
            {
                "name": u.get("name"),
                "suggestions": [s.get("property") for s in u.get("suggestions", [])],
            }
            for u in result.get("unmappedFields", [])
        ],
        "matches": [
            {"property": m.get("property"), "source": (m.get("payload") or {}).get("source")}
            for m in result.get("matches", [])
        ],
        "suggestions": [s.get("property") for s in result.get("suggestions", [])],
        "llmUnavailable": bool(result.get("llm_error")),
    }
    messages = [
        {
            "role": "system",
            "content": SUMMARIZE_PROMPT.format(
                language="Amharic (አማርኛ)" if language == "am" else "English"
            ),
        },
        {"role": "user", "content": json.dumps(compact, ensure_ascii=False)},
    ]
    try:
        message = client.complete(messages, json_mode=False).strip()
        # Measured live: asked for Amharic, the model still answered in
        # English. One stricter retry; property names stay in Latin script.
        if language == "am" and message and _language_of(message) != "am":
            messages.append({"role": "assistant", "content": message})
            messages.append(
                {
                    "role": "user",
                    "content": "Rewrite that reply in Amharic (አማርኛ). Keep every `backticked` "
                    "name exactly as it is.",
                }
            )
            message = client.complete(messages, json_mode=False).strip() or message
    except Exception as exc:  # noqa: BLE001 - feedback must never fail a request
        log_event(LOGGER, "assistant.summarize_failed", error=exc.__class__.__name__)
        return Feedback(fallback)

    if not message:
        return Feedback(fallback)
    invented = unknown_properties(message, allowed_tokens(result))
    if invented:
        log_event(LOGGER, "assistant.summary_rejected", invented=invented[:5])
        return Feedback(fallback, notes=[f"discarded model reply naming {invented[:3]}"])
    return Feedback(message[:1200], source="groq")


# ---------------------------------------------------------------- client ---


class GroqChatClient:
    """`ChatClient` over the project's Groq SDK wrapper (retries, timeouts)."""

    def __init__(self, groq: Any, model: str) -> None:
        self._groq = groq
        self._model = model

    def complete(self, messages: list[dict[str, str]], *, json_mode: bool) -> str:
        response = self._groq._create_with_retry(
            model=self._model,
            messages=messages,
            response_format={"type": "json_object"} if json_mode else None,
            temperature=0,
            max_completion_tokens=400,
        )
        content = response.choices[0].message.content or ""
        # Reasoning models may prepend a <think> block; it is never user-facing.
        return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()


@lru_cache(maxsize=1)
def cached_chat_client() -> ChatClient | None:
    """One shared Groq client for code paths without app state (the pipeline)."""

    return default_chat_client()


def infer_template_class(template_name: str, field_names: list[str]) -> str | None:
    """The DBpedia class for an infobox, inferred by Groq from its template
    name and field names, or `None` (no Groq, or no real class fits).

    Used for pasted Wikipedia links, which skip `understand()` because the
    template is only known after the article is fetched."""

    client = cached_chat_client()
    if client is None or not template_name:
        return None
    synthetic = (
        "{{" + template_name + "\n" + "".join(f"| {n} = \n" for n in field_names[:15]) + "}}"
    )
    return understand(synthetic, client=client).target_class


def default_chat_client() -> ChatClient | None:
    """A Groq client from `Settings`, or `None` (rules only) if unconfigured."""

    try:
        from config import Settings
        from mcp_server.agent import GroqClient

        settings = Settings()
        return GroqChatClient(GroqClient(settings=settings), settings.groq_model_fast)
    except Exception as exc:  # noqa: BLE001
        log_event(LOGGER, "assistant.groq_unconfigured", error=exc.__class__.__name__)
        return None
