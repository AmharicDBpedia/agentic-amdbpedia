from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from rag.predict import (
    DEFAULT_MODEL_ALIAS,
    SUPPORTED_MODELS,
    NoCandidatesFound,
    PredictionResult,
    predict_property,
    propose_property,
    resolve_model,
    snap_to_candidate,
)
from rag.retrieval import NoMatchFound, RetrievalCircuitBreaker, SearchResult

ICAO = SearchResult(
    property="icaoLocationIdentifier",
    ontology_class="Airport",
    score=1.0,
    payload={},
)
IATA = SearchResult(
    property="iataLocationIdentifier",
    ontology_class="Airport",
    score=0.6,
    payload={},
)


def _search_func(results: list[Any]) -> Any:
    def _fake(query: str, *, target_class: str | None = None, limit: int = 10) -> list[Any]:
        return results

    return _fake


@dataclass
class FakePrediction:
    property_class: str


class FakeProgram:
    """Mimics a DSPy program's callable interface without any real LLM."""

    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.calls: list[dict[str, str]] = []

    def __call__(self, *, premise: str, candidates: str) -> FakePrediction:
        self.calls.append({"premise": premise, "candidates": candidates})
        return FakePrediction(property_class=self.answer)


class BrokenProgram:
    def __call__(self, *, premise: str, candidates: str) -> FakePrediction:
        raise TimeoutError("ollama unreachable")


class FakeProposeProgram:
    """Mimics the propose-from-scratch DSPy program (different signature
    from the reranker: field_mention + entity_type -> property_class)."""

    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.calls: list[dict[str, str]] = []

    def __call__(self, *, field_mention: str, entity_type: str) -> FakePrediction:
        self.calls.append({"field_mention": field_mention, "entity_type": entity_type})
        return FakePrediction(property_class=self.answer)


class FakeCatalog:
    """Just enough of DbpediaOntologyCatalog for propose_property: a
    case-insensitive `find` that returns something with a `.local_name`."""

    def __init__(self, *known: str) -> None:
        self._by_name = {name.casefold(): name for name in known}

    def find(self, name: str) -> Any:
        canonical = self._by_name.get(name.strip().casefold())
        return SimpleNamespace(local_name=canonical) if canonical else None


def test_default_model_alias_is_gemma4() -> None:
    assert DEFAULT_MODEL_ALIAS == "gemma4"
    assert SUPPORTED_MODELS["gemma4"].litellm_model == "gemini/gemma-4-26b-a4b-it"


def test_resolve_model_looks_up_registry_alias() -> None:
    assert resolve_model("gemma4") == "gemini/gemma-4-26b-a4b-it"
    # Legacy local alias still resolves for anyone running Ollama out-of-tree.
    assert resolve_model("gemma2") == "ollama_chat/gemma2:9b"


def test_resolve_model_passes_through_unknown_names() -> None:
    assert resolve_model("ollama_chat/mistral:7b") == "ollama_chat/mistral:7b"


def test_predict_returns_no_candidates_when_retrieval_finds_nothing() -> None:
    outcome = predict_property(
        "unrelated nonsense",
        search_func=_search_func([NoMatchFound(query="unrelated nonsense")]),
        enable_propose=False,
    )

    assert isinstance(outcome, NoCandidatesFound)


def test_predict_proposes_a_validated_property_when_retrieval_is_empty() -> None:
    program = FakeProposeProgram(answer="topLevelDomain")

    outcome = predict_property(
        "ከፍተኛ_ደረጃ_ከባቢ",
        target_class="Country",
        search_func=_search_func([NoMatchFound(query="ከፍተኛ_ደረጃ_ከባቢ")]),
        circuit_breaker=None,
        propose_program=program,
        catalog=FakeCatalog("topLevelDomain"),
    )

    assert isinstance(outcome, PredictionResult)
    assert outcome.property == "topLevelDomain"
    assert outcome.llm_proposed is True
    assert outcome.used_llm is True
    assert outcome.top_retrieval_result is None
    assert program.calls == [{"field_mention": "ከፍተኛ_ደረጃ_ከባቢ", "entity_type": "Country"}]


def test_predict_rejects_a_proposed_property_that_is_not_in_the_ontology() -> None:
    outcome = predict_property(
        "ከፍተኛ_ደረጃ_ከባቢ",
        search_func=_search_func([NoMatchFound(query="ከፍተኛ_ደረጃ_ከባቢ")]),
        circuit_breaker=None,
        propose_program=FakeProposeProgram(answer="totallyMadeUpProperty"),
        catalog=FakeCatalog("topLevelDomain"),
    )

    assert isinstance(outcome, NoCandidatesFound)


def test_predict_does_not_propose_when_the_circuit_breaker_is_open() -> None:
    breaker = RetrievalCircuitBreaker(failure_threshold=1, reset_after_seconds=60)
    breaker.record_failure()
    program = FakeProposeProgram(answer="topLevelDomain")

    outcome = predict_property(
        "ከፍተኛ_ደረጃ_ከባቢ",
        search_func=_search_func([NoMatchFound(query="ከፍተኛ_ደረጃ_ከባቢ")]),
        circuit_breaker=breaker,
        propose_program=program,
        catalog=FakeCatalog("topLevelDomain"),
    )

    assert isinstance(outcome, NoCandidatesFound)
    assert program.calls == []


def test_predict_does_not_propose_when_disabled() -> None:
    program = FakeProposeProgram(answer="topLevelDomain")

    outcome = predict_property(
        "ከፍተኛ_ደረጃ_ከባቢ",
        search_func=_search_func([NoMatchFound(query="ከፍተኛ_ደረጃ_ከባቢ")]),
        circuit_breaker=None,
        enable_propose=False,
        propose_program=program,
        catalog=FakeCatalog("topLevelDomain"),
    )

    assert isinstance(outcome, NoCandidatesFound)
    assert program.calls == []


def test_propose_property_treats_a_bare_none_answer_as_no_proposal() -> None:
    assert (
        propose_property(
            "ያልታወቀ",
            program=FakeProposeProgram(answer="none"),
            catalog=FakeCatalog("topLevelDomain"),
        )
        is None
    )


def test_propose_property_strips_a_namespace_prefix_before_validating() -> None:
    assert (
        propose_property(
            "ከፍተኛ_ደረጃ_ከባቢ",
            program=FakeProposeProgram(answer="dbo:topLevelDomain"),
            catalog=FakeCatalog("topLevelDomain"),
        )
        == "topLevelDomain"
    )


def test_predict_uses_llm_reranked_answer_when_program_agrees_with_retriever() -> None:
    program = FakeProgram(answer="icaoLocationIdentifier")

    outcome = predict_property(
        "አይካኦ_ኮድ",
        search_func=_search_func([ICAO, IATA]),
        program=program,
        circuit_breaker=None,
    )

    assert isinstance(outcome, PredictionResult)
    assert outcome.property == "icaoLocationIdentifier"
    assert outcome.used_llm is True
    assert outcome.candidates == ["icaoLocationIdentifier", "iataLocationIdentifier"]
    assert program.calls == [
        {"premise": "አይካኦ_ኮድ", "candidates": "icaoLocationIdentifier; iataLocationIdentifier"}
    ]


def test_predict_lets_llm_override_the_top_retrieval_result() -> None:
    program = FakeProgram(answer="iataLocationIdentifier")

    outcome = predict_property(
        "ambiguous code",
        search_func=_search_func([ICAO, IATA]),
        program=program,
        circuit_breaker=None,
    )

    assert isinstance(outcome, PredictionResult)
    assert outcome.property == "iataLocationIdentifier"
    assert outcome.used_llm is True


def test_snap_to_candidate_keeps_an_already_exact_match() -> None:
    assert snap_to_candidate("icaoLocationIdentifier", ["icaoLocationIdentifier"]) == (
        "icaoLocationIdentifier"
    )


def test_snap_to_candidate_is_case_and_whitespace_insensitive() -> None:
    assert (
        snap_to_candidate("  ICAOLocationIdentifier ", ["icaoLocationIdentifier"])
        == "icaoLocationIdentifier"
    )


def test_snap_to_candidate_fixes_a_near_miss_typo() -> None:
    # The guardrail this ports from LLMIntegration's SelectFromCandidates:
    # never trust an LLM answer verbatim if it's close-but-not-exact.
    assert (
        snap_to_candidate(
            "icaoLocationIdentifer", ["icaoLocationIdentifier", "iataLocationIdentifier"]
        )
        == "icaoLocationIdentifier"
    )


def test_snap_to_candidate_falls_back_to_first_candidate_on_a_hallucinated_answer() -> None:
    assert (
        snap_to_candidate(
            "completelyMadeUpProperty", ["icaoLocationIdentifier", "iataLocationIdentifier"]
        )
        == "icaoLocationIdentifier"
    )


def test_snap_to_candidate_with_no_candidates_returns_answer_unchanged() -> None:
    assert snap_to_candidate("anything", []) == "anything"


def test_predict_falls_back_to_retrieval_top1_when_llm_call_fails() -> None:
    outcome = predict_property(
        "አይካኦ_ኮድ",
        search_func=_search_func([ICAO, IATA]),
        program=BrokenProgram(),
        circuit_breaker=None,
    )

    assert isinstance(outcome, PredictionResult)
    assert outcome.property == "icaoLocationIdentifier"
    assert outcome.used_llm is False
    assert "TimeoutError" in outcome.reason
    assert outcome.top_retrieval_result == ICAO


def test_predict_falls_back_when_circuit_breaker_is_open() -> None:
    breaker = RetrievalCircuitBreaker(failure_threshold=1, reset_after_seconds=60)
    breaker.record_failure()
    assert not breaker.allow_request()

    outcome = predict_property(
        "አይካኦ_ኮድ",
        search_func=_search_func([ICAO, IATA]),
        program=FakeProgram(answer="iataLocationIdentifier"),
        circuit_breaker=breaker,
    )

    assert isinstance(outcome, PredictionResult)
    assert outcome.property == "icaoLocationIdentifier"
    assert outcome.used_llm is False
    assert outcome.reason == "LLM reranking temporarily unavailable"


def test_predict_records_failure_on_circuit_breaker_after_llm_error() -> None:
    breaker = RetrievalCircuitBreaker(failure_threshold=1, reset_after_seconds=60)

    predict_property(
        "አይካኦ_ኮድ",
        search_func=_search_func([ICAO]),
        program=BrokenProgram(),
        circuit_breaker=breaker,
    )

    assert not breaker.allow_request()


@pytest.mark.integration
def test_real_dspy_program_construction_and_offline_fallback() -> None:
    """No Ollama is expected to be reachable in CI/test environments — this
    exercises the real (not injected) DSPy program construction path end to
    end and confirms it degrades gracefully rather than raising."""

    outcome = predict_property(
        "አይካኦ_ኮድ",
        search_func=_search_func([ICAO, IATA]),
        circuit_breaker=None,
    )

    assert isinstance(outcome, PredictionResult)
    # Either a real Ollama happened to be reachable (used_llm=True, some
    # candidate chosen) or it wasn't (graceful fallback to top-1) — both are
    # correct outcomes; what must never happen is an unhandled exception.
    assert outcome.property in {"icaoLocationIdentifier", "iataLocationIdentifier"}


# --- model chain, propose -> select-among-suggestions, learned aliases -----


class ModelAwareProgram:
    """Fails under any Ollama model and answers under anything else, so a
    test can tell which model in the chain actually served the call."""

    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.models: list[str] = []

    def __call__(self, **kwargs: str) -> FakePrediction:
        import dspy

        model = dspy.settings.lm.model
        self.models.append(model)
        if model.startswith("ollama"):
            raise ConnectionError("ollama not running")
        return FakePrediction(property_class=self.answer)


def _suggest_func(*names: str) -> Any:
    def _fake(query: str, *, target_class: str | None = None, limit: int = 10) -> list[Any]:
        return [SearchResult(property=n, ontology_class="", score=0.6, payload={}) for n in names][
            :limit
        ]

    return _fake


EMPTY_SEARCH = _search_func([NoMatchFound(query="x")])


def test_rerank_falls_back_to_the_next_model_in_the_chain() -> None:
    program = ModelAwareProgram(answer="iataLocationIdentifier")

    outcome = predict_property(
        "አያታ_ኮድ",
        search_func=_search_func([ICAO, IATA]),
        program=program,
        circuit_breaker=None,
        model_chain=("gemma2", "groq/some-model"),
    )

    assert isinstance(outcome, PredictionResult)
    assert outcome.property == "iataLocationIdentifier"
    assert outcome.model == "groq/some-model"
    assert program.models == ["ollama_chat/gemma2:9b", "groq/some-model"]


def test_rerank_premise_includes_the_target_class() -> None:
    program = FakeProgram(answer="icaoLocationIdentifier")

    predict_property(
        "አይካኦ_ኮድ",
        target_class="Airport",
        search_func=_search_func([ICAO, IATA]),
        program=program,
        circuit_breaker=None,
    )

    assert program.calls[0]["premise"] == "Airport's አይካኦ_ኮድ"


def test_rerank_never_considers_unverified_suggestions() -> None:
    """Measured live: appending ungated alternatives let the LLM override a
    correct curated answer. Confident candidates are reranked on their own."""

    staff = SearchResult(property="numberOfStaff", ontology_class="", score=1.0, payload={})
    program = FakeProgram(answer="numberOfStaff")

    outcome = predict_property(
        "የሰራተኞች_ብዛት",
        target_class="Company",
        search_func=_search_func([staff]),
        suggest_func=_suggest_func("numberOfStaff", "numberOfEmployees"),
        program=program,
        circuit_breaker=None,
    )

    assert program.calls[0]["candidates"] == "numberOfStaff"
    assert isinstance(outcome, PredictionResult)
    assert outcome.method == "rerank"
    assert outcome.top_retrieval_result == staff


def test_select_among_suggestions_runs_when_propose_abstains() -> None:
    select = FakeProgram(answer="publisher")

    outcome = predict_property(
        "አሳታሚ",
        target_class="Book",
        search_func=EMPTY_SEARCH,
        suggest_func=_suggest_func("publisher", "firstPublisher"),
        circuit_breaker=None,
        propose_program=FakeProposeProgram(answer="none"),
        select_program=select,
        catalog=FakeCatalog("publisher"),
    )

    assert isinstance(outcome, PredictionResult)
    assert outcome.property == "publisher"
    assert outcome.method == "llm_select"
    assert outcome.llm_proposed is True
    assert select.calls == [{"premise": "Book's አሳታሚ", "candidates": "publisher; firstPublisher"}]


@pytest.mark.parametrize("answer", ["none", "None.", "completelyMadeUpProperty"])
def test_select_among_suggestions_can_decline(answer: str) -> None:
    outcome = predict_property(
        "የድመቱ_ተወዳጅ_ቀለም",
        search_func=EMPTY_SEARCH,
        suggest_func=_suggest_func("colour", "cargoFuel", "status", "x"),
        circuit_breaker=None,
        propose_program=FakeProposeProgram(answer="none"),
        select_program=FakeProgram(answer=answer),
        catalog=FakeCatalog("colour"),
    )

    assert isinstance(outcome, NoCandidatesFound)
    assert [s.property for s in outcome.suggestions] == ["colour", "cargoFuel", "status"]
    assert outcome.llm_error is None


def test_propose_wins_without_running_select() -> None:
    select = FakeProgram(answer="publisher")

    outcome = predict_property(
        "ደራሲ",
        target_class="Book",
        search_func=EMPTY_SEARCH,
        suggest_func=_suggest_func("writer", "author"),
        circuit_breaker=None,
        propose_program=FakeProposeProgram(answer="author"),
        select_program=select,
        catalog=FakeCatalog("author"),
    )

    assert isinstance(outcome, PredictionResult)
    assert outcome.property == "author"
    assert outcome.method == "llm_propose"
    assert select.calls == []


def test_no_candidates_reports_llm_error_when_every_model_fails() -> None:
    class Down:
        def __call__(self, **_kwargs: str) -> FakePrediction:
            raise ConnectionError("down")

    outcome = predict_property(
        "አሳታሚ",
        search_func=EMPTY_SEARCH,
        suggest_func=_suggest_func("publisher"),
        circuit_breaker=None,
        model_chain=("gemma2", "groq/some-model"),
        propose_program=Down(),
        catalog=FakeCatalog("publisher"),
    )

    assert isinstance(outcome, NoCandidatesFound)
    assert outcome.llm_error is not None
    assert "gemma2" in outcome.llm_error and "groq/some-model" in outcome.llm_error
    assert [s.property for s in outcome.suggestions] == ["publisher"]


def test_learned_alias_short_circuits_retrieval_and_llm() -> None:
    def search_must_not_run(*_args: Any, **_kwargs: Any) -> list[Any]:
        raise AssertionError("retrieval should not run for a learned alias")

    outcome = predict_property(
        "የገጽ_ብዛት",
        target_class="Book",
        search_func=search_must_not_run,
        learned_lookup=lambda field, cls: "numberOfPages" if cls == "Book" else None,
    )

    assert isinstance(outcome, PredictionResult)
    assert outcome.property == "numberOfPages"
    assert outcome.method == "learned"
    assert outcome.used_llm is False


def test_suggestion_failures_never_break_prediction() -> None:
    def broken_suggest(*_args: Any, **_kwargs: Any) -> list[Any]:
        raise RuntimeError("index missing")

    outcome = predict_property(
        "ያልታወቀ",
        search_func=EMPTY_SEARCH,
        suggest_func=broken_suggest,
        circuit_breaker=None,
        propose_program=FakeProposeProgram(answer="none"),
        catalog=FakeCatalog(),
    )

    assert isinstance(outcome, NoCandidatesFound)
    assert outcome.suggestions == ()
