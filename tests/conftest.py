from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _no_live_class_inference(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep unit tests offline: the pipeline would otherwise ask Groq for the
    class of every infobox whose template has no published mapping."""

    monkeypatch.setattr("mcp_server.pipeline.infer_template_class", lambda *_args: None)


@pytest.fixture(autouse=True)
def _no_live_template_redirects(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep unit tests offline: template-redirect resolution queries Wikipedia."""

    monkeypatch.setattr("mcp_server.pipeline.canonical_template_name", lambda name: name)
