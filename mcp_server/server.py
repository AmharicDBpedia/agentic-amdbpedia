"""FastMCP server for the DBpedia mapping assistant."""

from __future__ import annotations

import json
import logging
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field, ValidationError, field_validator

from config import Settings
from errors import AssistantValidationError, ClientSafeError, RetrievalUnavailableError
from logging_config import (
    configure_logging as configure_json_logging,
)
from logging_config import (
    correlation_context,
    get_correlation_id,
    log_event,
)
from mcp_server.guardrails import is_injection_attempt
from rag.retrieval import NoMatchFound, SearchResult, search

LOGGER = logging.getLogger("dbpedia_mapping_assistant.mcp")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LATEST_METRICS_PATH = PROJECT_ROOT / "evaluation" / "latest_metrics.json"
MAX_PROPERTY_LENGTH = 500

mcp = FastMCP("DBpedia-Mapping-Assistant")


class StartupError(RuntimeError):
    """Raised when the MCP server cannot start safely."""


class MappingEntry(BaseModel):
    templateProperty: str = Field(min_length=1, max_length=MAX_PROPERTY_LENGTH)
    ontologyProperty: str = Field(min_length=1, max_length=MAX_PROPERTY_LENGTH)

    @field_validator("templateProperty", "ontologyProperty")
    @classmethod
    def reject_control_characters(cls, value: str) -> str:
        if any(ord(char) < 32 for char in value):
            raise ValueError("control characters are not allowed")
        return value


class MappingPayload(BaseModel):
    domain_class: str
    mappings: list[MappingEntry] = Field(min_length=1)

    @field_validator("domain_class")
    @classmethod
    def validate_domain_class(cls, value: str) -> str:
        if not value or not value[0].isupper() or not value.replace("_", "").isalnum():
            raise ValueError("domain_class must match ^[A-Z][A-Za-z0-9]*$")
        if "_" in value:
            raise ValueError("domain_class must match ^[A-Z][A-Za-z0-9]*$")
        return value


# Force these two schemas to build now, using this module's own current
# globals(), rather than lazily on first real use. `uv run mcp dev
# mcp_server/server.py` loads this file via
# importlib.util.spec_from_file_location(...) + exec_module(...) without
# ever registering the resulting module in sys.modules (a real bug in the
# mcp SDK's CLI, confirmed directly against a minimal repro of that exact
# loader sequence) -- pydantic's automatic forward-ref resolution for
# `mappings: list[MappingEntry]` needs `sys.modules[cls.__module__]` to
# find `MappingEntry` and silently defers instead, so FastMCP's tool
# registration for generate_mapping_syntax(payload: MappingPayload) later
# blows up with "MappingEntry is not fully defined". `model_rebuild()`
# resolves via the *calling frame's* globals instead of sys.modules, which
# still contains MappingEntry regardless of that missing registration.
# `python -m mcp_server.server` (what tests/integration/test_mcp_protocol.py
# actually uses) doesn't hit this at all -- a normal import always
# registers sys.modules correctly -- which is exactly why this went
# unnoticed until someone ran the literal `mcp dev` CLI command.
MappingEntry.model_rebuild()
MappingPayload.model_rebuild()


def configure_logging() -> None:
    configure_json_logging()


def _safe_error_json(error: ClientSafeError) -> str:
    payload = error.to_payload()
    payload["correlation_id"] = get_correlation_id()
    return json.dumps(payload, ensure_ascii=False)


def validate_startup(
    *,
    settings: Settings | None = None,
    skip_checks: bool | None = None,
) -> None:
    """Fail clearly when required secrets are unavailable.

    Retrieval no longer depends on a reachable external service (refs 10.4):
    the ontology index is embedded in-process, and a model-load failure
    surfaces as a RetrievalUnavailableError at first use rather than a
    startup precondition — checking it here would force a full ~2,948
    -property embedding pass (minutes) into every server start.
    """

    if skip_checks is None:
        skip_checks = os.environ.get("MCP_SERVER_SKIP_STARTUP_CHECKS") == "1"
    if skip_checks:
        LOGGER.info("startup checks skipped")
        return

    try:
        _resolved = settings or Settings()
    except ValidationError as exc:
        raise StartupError(f"Missing or invalid server configuration: {exc}") from exc


def _result_to_payload(result: SearchResult) -> dict[str, Any]:
    return {
        "property": result.property,
        "class": result.ontology_class,
        "score": result.score,
        "payload": result.payload,
    }


def find_semantic_match_impl(
    amharic_property: str,
    target_class: str | None = None,
    *,
    search_func: Any = search,
) -> str:
    """Search DBpedia ontology properties and return JSON for MCP clients."""

    with correlation_context() as correlation_id:
        log_event(LOGGER, "mcp.find_semantic_match.start", target_class=target_class)
        if not amharic_property.strip():
            return _safe_error_json(AssistantValidationError("amharic_property is required"))
        if len(amharic_property) > MAX_PROPERTY_LENGTH:
            return _safe_error_json(
                AssistantValidationError(
                    f"amharic_property must be at most {MAX_PROPERTY_LENGTH} characters"
                )
            )
        if is_injection_attempt(amharic_property):
            log_event(LOGGER, "mcp.find_semantic_match.rejected", reason="prompt_injection")
            return json.dumps(
                {
                    "status": "rejected",
                    "correlation_id": correlation_id,
                    "reason": "prompt_injection",
                    "message": "Rejected: prompt-injection attempt detected.",
                    "matches": [],
                },
                ensure_ascii=False,
            )
        if os.environ.get("MCP_SERVER_TEST_MODE") == "1":
            return json.dumps(
                {
                    "status": "ok",
                    "correlation_id": correlation_id,
                    "matches": [
                        {
                            "property": "icaoLocationIdentifier",
                            "class": target_class or "Airport",
                            "score": 1.0,
                            "payload": {"xsd_type": "xsd:string"},
                        }
                    ],
                },
                ensure_ascii=False,
            )

        try:
            results = search_func(amharic_property, target_class=target_class)
        except ClientSafeError as exc:
            log_event(LOGGER, "mcp.find_semantic_match.error", error_type=exc.error_type)
            return _safe_error_json(exc)
        except Exception as exc:
            log_event(LOGGER, "mcp.find_semantic_match.error", error=exc.__class__.__name__)
            return _safe_error_json(RetrievalUnavailableError())

        if not results or isinstance(results[0], NoMatchFound):
            log_event(LOGGER, "mcp.find_semantic_match.no_match")
            return json.dumps(
                {"status": "no_match", "correlation_id": correlation_id, "matches": []}
            )

        matches = [
            _result_to_payload(result) for result in results if isinstance(result, SearchResult)
        ]
        log_event(LOGGER, "mcp.find_semantic_match.complete", match_count=len(matches))
        return json.dumps(
            {"status": "ok", "correlation_id": correlation_id, "matches": matches},
            ensure_ascii=False,
        )


def generate_mapping_syntax_impl(payload: MappingPayload) -> str:
    """Generate deterministic MediaWiki XML mapping syntax."""

    root = ET.Element("TemplateMapping", {"mapToClass": f"dbo:{payload.domain_class}"})
    for mapping in payload.mappings:
        node = ET.SubElement(root, "PropertyMapping")
        template_property = ET.SubElement(node, "templateProperty")
        template_property.text = mapping.templateProperty
        ontology_property = ET.SubElement(node, "ontologyProperty")
        ontology_property.text = mapping.ontologyProperty

    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode", short_empty_elements=False)


def get_benchmarks_impl(metrics_path: Path = LATEST_METRICS_PATH) -> str:
    """Return latest evaluation metrics or an honest no-data state."""

    if not metrics_path.exists():
        return json.dumps({"status": "no_evaluation_run_yet"})
    return metrics_path.read_text(encoding="utf-8")


@mcp.tool()
def find_semantic_match(amharic_property: str, target_class: str | None = None) -> str:
    """Search DBpedia ontology properties using hybrid retrieval."""

    return find_semantic_match_impl(amharic_property, target_class)


@mcp.tool()
def generate_mapping_syntax(payload: MappingPayload) -> str:
    """Generate valid MediaWiki XML template syntax deterministically."""

    return generate_mapping_syntax_impl(payload)


@mcp.resource("resources://benchmarks/latest")
def get_benchmarks() -> str:
    """Return the latest retrieval/evaluation metrics."""

    return get_benchmarks_impl()


def main() -> int:
    configure_logging()
    try:
        validate_startup()
    except StartupError as exc:
        print(f"Startup error: {exc}", file=sys.stderr)
        return 1
    LOGGER.info("starting FastMCP server")
    mcp.run(transport="stdio")
    LOGGER.info("FastMCP server stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
