"""Documentation contract: the project's substantive technical commitments
have to live somewhere canonical in the repo, and the traceability matrix
has to point at tests that actually exist. Which *file* holds the content
is intentionally flexible so the README can stay short; the fragments just
have to be present in the project docs (README + docs/*.md) taken
together.
"""

from __future__ import annotations

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _project_docs() -> str:
    """README plus every top-level docs/*.md, concatenated. The exact
    distribution across files is a style choice; the technical commitments
    below are the thing we actually test for."""
    parts: list[str] = []
    parts.append((PROJECT_ROOT / "README.md").read_text(encoding="utf-8"))
    for md in sorted((PROJECT_ROOT / "docs").glob("*.md")):
        parts.append(md.read_text(encoding="utf-8"))
    return "\n\n".join(parts)


def test_project_docs_record_domain_and_architecture_commitments() -> None:
    docs = _project_docs()
    required_fragments = [
        "## Domain",
        "## Architecture",
        "## Requirements Traceability",
        "## Future Work",
        "cross-lingual semantic web engineering",
        "Amharic Wikipedia",
        "English DBpedia ontology",
        "Retrieval-augmented generation is required",
        "MCP tools are required",
        "deterministic",
    ]
    missing = [fragment for fragment in required_fragments if fragment not in docs]
    assert missing == []


def test_demo_transcripts_cover_required_paths() -> None:
    demo = (PROJECT_ROOT / "examples" / "demo.md").read_text(encoding="utf-8")
    required_fragments = [
        "Successful Airport Mapping",
        "Low-Confidence No-Match Refusal",
        "Prompt-Injection Guardrail",
        "find_semantic_match",
        "generate_mapping_syntax",
        "Rejected: prompt-injection attempt detected.",
    ]
    missing = [fragment for fragment in required_fragments if fragment not in demo]
    assert missing == []


def test_traceability_matrix_entries_reference_existing_tests() -> None:
    """The traceability matrix (now in docs/traceability.md) must only cite
    tests that actually exist in the suite -- otherwise the matrix is a
    silent lie."""
    matrix = (PROJECT_ROOT / "docs" / "traceability.md").read_text(encoding="utf-8")
    references = re.findall(r"`(tests/[^`]+::test_[^`]+)`", matrix)

    assert references
    for reference in references:
        path_text, test_name = reference.split("::", 1)
        path = PROJECT_ROOT / path_text
        assert path.exists(), reference
        source = path.read_text(encoding="utf-8")
        assert f"def {test_name}" in source or f"async def {test_name}" in source, reference
