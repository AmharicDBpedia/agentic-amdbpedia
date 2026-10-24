"""Gemma 2 9B retrieve-then-rerank predictor.

Ported from `LLMIntegration/models.py` and `llm_raranker.py` (refs
implementation.md 11.1/11.2). LLMIntegration's own script computed its
top-k candidates directly from a SentenceTransformer over a fixed label
set; here candidates come from `rag.retrieval.search()` instead — the same
real ontology corpus, dense+sparse retrieval, and confidence logic already
used everywhere else in this repo.

The retriever alone already picks a top-1 answer; this module lets an LLM
rerank among the retriever's own top-k candidates and, when retrieval finds
nothing confident (the normal case for Amharic vocabulary not in the
corpus), propose or select a property itself. Every LLM call runs on a
model chain -- by default the local Gemma 2 9B via Ollama, then the hosted
fallback from `Settings.predict_model_chain()` (Groq) -- so an unreachable
Ollama no longer silently disables the whole layer. It is still an
enhancement, never a dependency: when every model fails, or the circuit
breaker is open, prediction degrades to the retriever's own top-1 choice
(or an explicit no-candidates result carrying `llm_error`) rather than a
failed request (refs 11.2).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast

from rapidfuzz import fuzz

from logging_config import log_event
from rag.ontology import DbpediaOntologyCatalog
from rag.retrieval import RetrievalCircuitBreaker, RetrievalResult, SearchResult, search

LOGGER = logging.getLogger("dbpedia_mapping_assistant.predict")

SearchFunc = Callable[..., list[RetrievalResult]]

FUZZY_THRESHOLD = 85
DEFAULT_MODEL_ALIAS = "gemma4"
DEFAULT_TOP_K = 10


@dataclass(frozen=True)
class ModelConfig:
    alias: str
    litellm_model: str
    description: str


SUPPORTED_MODELS: dict[str, ModelConfig] = {
    # Default: Gemma 4 26B A4B (MoE, ~4B active) served by Google AI Studio
    # via LiteLLM. Replaces the previous local gemma2:9b-via-Ollama path,
    # which was dropped from the release stack. Set GEMINI_API_KEY in .env.
    "gemma4": ModelConfig(
        alias="gemma4",
        litellm_model="gemini/gemma-4-26b-a4b-it",
        description="Gemma 4 26B A4B Instruct (default, Gemini API)",
    ),
    "gemma4-31b": ModelConfig(
        alias="gemma4-31b",
        litellm_model="gemini/gemma-4-31b-it",
        description="Gemma 4 31B Instruct (Gemini API)",
    ),
    "gemma3-12b": ModelConfig(
        alias="gemma3-12b",
        litellm_model="gemini/gemma-3-12b-it",
        description="Gemma 3 12B Instruct (Gemini API)",
    ),
    # Legacy local aliases kept for anyone still running Ollama out-of-tree.
    # Not part of the released docker-compose stack.
    "gemma2": ModelConfig(
        alias="gemma2",
        litellm_model="ollama_chat/gemma2:9b",
        description="Gemma 2 9B Instruct (legacy local, Ollama)",
    ),
    "qwen2.5:32b": ModelConfig(
        alias="qwen2.5:32b",
        litellm_model="ollama_chat/qwen2.5:32b",
        description="Qwen 2.5 32B (legacy local, Ollama)",
    ),
    "llama3.1": ModelConfig(
        alias="llama3.1",
        litellm_model="ollama_chat/llama3.1:8b",
        description="Llama 3.1 8B Instruct (legacy local, Ollama)",
    ),
    "llama3.2": ModelConfig(
        alias="llama3.2",
        litellm_model="ollama_chat/llama3.2:3b",
        description="Llama 3.2 3B Instruct (legacy local, Ollama)",
    ),
    "aya": ModelConfig(
        alias="aya",
        litellm_model="ollama_chat/aya-expanse:8b",
        description="Aya Expanse 8B (legacy local, Ollama)",
    ),
}


def resolve_model(name: str) -> str:
    """Resolve a registry alias to its LiteLLM model string.

    Falls back to returning `name` unchanged, so a new Ollama tag or any
    other LiteLLM-compatible model string still works without a registry
    entry.
    """

    config = SUPPORTED_MODELS.get(name)
    return config.litellm_model if config else name


def _normalize(text: str) -> str:
    return " ".join(str(text).strip().casefold().split())


def snap_to_candidate(answer: str, candidates: list[str]) -> str:
    """Never let an answer leave this module unless it's one of the
    retriever's own candidates — the core guardrail ported from
    LLMIntegration's SelectFromCandidates, kept standalone and directly
    testable rather than buried inside the DSPy module closure.

    Always returns one of `candidates` verbatim, never the LLM's own
    echoed casing/whitespace — downstream XML generation is case-sensitive
    about real DBpedia ontology property names, so an exact normalized
    match still snaps to the candidate's canonical spelling rather than
    trusting the answer text as-is (a deliberate change from
    LLMIntegration's original, which only ever fed its output into an
    accuracy metric, not a case-sensitive XML pipeline). A fuzzy-close match
    also snaps to its nearest candidate; anything else falls back to the
    retriever's own top-ranked candidate rather than trusting an
    unrecognized answer.
    """

    if not candidates:
        return answer
    best = max(candidates, key=lambda candidate: fuzz.token_sort_ratio(candidate, answer))
    if _normalize(best) == _normalize(answer) or fuzz.token_sort_ratio(best, answer) >= (
        FUZZY_THRESHOLD
    ):
        return best
    return candidates[0]


class DspyPrediction(Protocol):
    property_class: str


class DspyProgram(Protocol):
    def __call__(self, *, premise: str, candidates: str) -> DspyPrediction: ...


def _build_dspy_program() -> DspyProgram:
    """Construct the real DSPy program. Deferred import: dspy is only ever
    needed when a caller actually wants LLM reranking, not for plain
    retrieval-only usage."""

    import dspy

    class PropertyMappingMC(dspy.Signature):  # type: ignore[misc]
        """Choose the correct canonical English property class for a
        (possibly Amharic) infobox property mention. The premise has the
        form "<entity type>'s <property mention>". You are given candidate
        classes ORDERED BY RETRIEVER CONFIDENCE (most likely first). The
        first candidate is usually correct: keep it unless another
        candidate clearly matches the property mention better. Answer with
        exactly one class copied verbatim from the candidate list."""

        premise: str = dspy.InputField()
        candidates: str = dspy.InputField()
        property_class: str = dspy.OutputField()

    class SelectFromCandidates(dspy.Module):  # type: ignore[misc]
        """Guardrail wrapper: never let the LLM answer with a property that
        isn't one of the retriever's own candidates."""

        def __init__(self, inner: Any) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, **kwargs: Any) -> Any:
            pred = self.inner(**kwargs)
            candidates = [c for c in kwargs.get("candidates", "").split("; ") if c]
            pred.property_class = snap_to_candidate(pred.property_class, candidates)
            return pred

    return SelectFromCandidates(dspy.Predict(PropertyMappingMC))


def _build_select_or_none_program() -> DspyProgram:
    """Rerank for candidates that did NOT pass retrieval's confidence gate.

    Unlike `_build_dspy_program` (whose candidates are already
    high-confidence, so it must always pick one), these come from
    `rag.retrieval.suggest` -- the ungated dense top-k -- and frequently
    contain no correct answer at all. Without an explicit way out, the
    measured behaviour was to pick *something* even for nonsense ("the
    cat's favourite colour" -> `colour`, a prompt injection ->
    `administrator`). `_snap_or_none` turns "none" or any answer outside the
    candidate list into `None`."""

    import dspy

    class SelectPropertyOrNone(dspy.Signature):  # type: ignore[misc]
        """Pick the DBpedia ontology property that an (often Amharic)
        infobox field expresses. The premise has the form "<entity type>'s
        <field name>". The candidates were found by a similarity search and
        are NOT verified: they may all be wrong. Answer with exactly one
        candidate copied verbatim if it clearly means the same thing as the
        field for that entity type; otherwise answer exactly "none"."""

        premise: str = dspy.InputField()
        candidates: str = dspy.InputField()
        property_class: str = dspy.OutputField()

    return cast("DspyProgram", dspy.Predict(SelectPropertyOrNone))


NONE_ANSWERS = {"", "none", "null", "n/a", "no match"}


def _snap_or_none(answer: str, candidates: list[str]) -> str | None:
    """Like `snap_to_candidate`, but an unrecognised or "none" answer means
    "no property" rather than "the first candidate"."""

    cleaned = str(answer).strip().strip("\"'`.").split(":")[-1].strip()
    if _normalize(cleaned) in NONE_ANSWERS or not candidates:
        return None
    best = max(candidates, key=lambda candidate: fuzz.token_sort_ratio(candidate, cleaned))
    if _normalize(best) == _normalize(cleaned) or fuzz.token_sort_ratio(best, cleaned) >= (
        FUZZY_THRESHOLD
    ):
        return best
    return None


# Default per-call timeout. Gemma 2 9B on a CPU-only machine needs several
# seconds per answer (and far longer on the first call, while Ollama loads
# the weights); a hosted fallback answers in well under that. A genuinely
# down Ollama still fails instantly (connection refused), so a generous
# timeout doesn't slow down the fallback to the next model. The pipeline
# passes `Settings.predict_llm_timeout_seconds` explicitly.
LLM_CALL_TIMEOUT_SECONDS = 60.0
# Hosted models (Groq) rate-limit bursts -- measured: ~75% of an unpaced
# burst of calls failed with RateLimitError on the free tier. LiteLLM's own
# retry-with-backoff absorbs that; a local Ollama gets no retries, since a
# refused connection won't succeed on retry.
REMOTE_MODEL_RETRIES = 3


class LLMUnavailableError(RuntimeError):
    """Every model in the chain failed (unreachable, timed out, rate-limited)."""


def _retries_for(model: str) -> int:
    return 0 if resolve_model(model).startswith(("ollama", "ollama_chat")) else REMOTE_MODEL_RETRIES


def _run_on_model_chain(
    models: tuple[str, ...],
    invoke: Callable[[], Any],
    *,
    timeout: float,
    max_tokens: int,
) -> tuple[Any, str]:
    """Run `invoke()` under each model in turn until one succeeds.

    Returns `(result, model_used)`. Raises `LLMUnavailableError` naming
    every model's failure when none succeeds. Responses are never served
    from DSPy's on-disk cache, so every answer is a real model call."""

    import dspy

    failures: list[str] = []
    for model in models:
        lm = dspy.LM(
            resolve_model(model),
            temperature=0.0,
            max_tokens=max_tokens,
            num_retries=_retries_for(model),
            timeout=timeout,
            cache=False,
        )
        try:
            with dspy.context(lm=lm):
                return invoke(), model
        except Exception as exc:  # noqa: BLE001 - try the next model in the chain
            failures.append(f"{model}: {exc.__class__.__name__}")
            log_event(LOGGER, "predict.model_failed", model=model, error=exc.__class__.__name__)
    raise LLMUnavailableError("; ".join(failures) or "no LLM models configured")


def _premise(amharic_property: str, target_class: str | None) -> str:
    """The "<entity type>'s <property mention>" form every prompt documents.

    Without the class, "ቁመት" (height) on a Person and "ከፍታ" (height) on a
    Lake (whose DBpedia property is `elevation`) are indistinguishable."""

    return f"{target_class}'s {amharic_property}" if target_class else amharic_property


def _call_dspy(
    program: DspyProgram,
    amharic_property: str,
    candidates: list[str],
    model_alias: str,
    *,
    timeout: float = LLM_CALL_TIMEOUT_SECONDS,
) -> str:
    """Single-model rerank call; kept for scripts that call it directly."""

    answer, _model = _run_on_model_chain(
        (model_alias,),
        lambda: program(premise=amharic_property, candidates="; ".join(candidates)),
        timeout=timeout,
        max_tokens=256,
    )
    return str(answer.property_class)


_ONTOLOGY_CATALOG: DbpediaOntologyCatalog | None = None


def _ontology_catalog() -> DbpediaOntologyCatalog | None:
    """The real ~2,900-property ontology catalog, built once and reused.

    Only ever needed by the propose-from-scratch fallback, to check that a
    property the LLM names is actually a real DBpedia ontology property --
    so a missing/broken ontology cache degrades that fallback to "no
    proposal" rather than raising.
    """

    global _ONTOLOGY_CATALOG
    if _ONTOLOGY_CATALOG is None:
        try:
            _ONTOLOGY_CATALOG = DbpediaOntologyCatalog.from_default_cache()
        except Exception as exc:  # noqa: BLE001 - any failure here just disables the fallback
            log_event(LOGGER, "predict.ontology_catalog_unavailable", error=exc.__class__.__name__)
            return None
    return _ONTOLOGY_CATALOG


class OntologyLookup(Protocol):
    """The one method the propose fallback needs from an ontology catalog:
    look a name up and get back something with a `.local_name`, or `None`.
    `DbpediaOntologyCatalog` satisfies this structurally; tests pass a fake."""

    def find(self, name: str) -> Any: ...


class ProposeProgram(Protocol):
    def __call__(self, *, field_mention: str, entity_type: str) -> DspyPrediction: ...


def _build_propose_program() -> ProposeProgram:
    """DSPy program for the propose-from-scratch fallback: no retriever
    candidates to choose from, so the LLM names a property outright and the
    caller validates it against the ontology afterwards."""

    import dspy

    class ProposePropertyFromField(dspy.Signature):  # type: ignore[misc]
        """Name the single most likely DBpedia ontology property for a
        (possibly Amharic) infobox field. Answer with just the property's
        camelCase local name (e.g. "topLevelDomain", "foundingDate"), no
        namespace prefix. Answer exactly "none" if no standard DBpedia
        ontology property fits."""

        field_mention: str = dspy.InputField(desc="the infobox field name, often Amharic")
        entity_type: str = dspy.InputField(desc="the DBpedia class of the described entity")
        property_class: str = dspy.OutputField(desc="a camelCase DBpedia property name, or none")

    return cast("ProposeProgram", dspy.Predict(ProposePropertyFromField))


def _propose_with_chain(
    amharic_property: str,
    *,
    target_class: str | None,
    models: tuple[str, ...],
    timeout: float,
    program: ProposeProgram | None,
    catalog: OntologyLookup | None,
) -> tuple[str | None, str | None]:
    """`(validated property or None, model that answered or None)`.

    Raises `LLMUnavailableError` when no model could be reached."""

    resolved_catalog = catalog if catalog is not None else _ontology_catalog()
    if resolved_catalog is None:
        return None, None

    resolved_program = program if program is not None else _build_propose_program()
    prediction, model = _run_on_model_chain(
        models,
        lambda: resolved_program(
            field_mention=amharic_property, entity_type=target_class or "Thing"
        ),
        timeout=timeout,
        max_tokens=64,
    )

    answer = str(prediction.property_class).strip().strip("\"'").lstrip(":").split(":")[-1].strip()
    if not answer or answer.casefold() == "none":
        return None, model

    match = resolved_catalog.find(answer)
    if match is None:
        log_event(LOGGER, "predict.propose_rejected", answer=answer)
        return None, model
    log_event(LOGGER, "predict.proposed", property=match.local_name, model=model)
    return str(match.local_name), model


def propose_property(
    amharic_property: str,
    *,
    target_class: str | None = None,
    model_alias: str = DEFAULT_MODEL_ALIAS,
    model_chain: Sequence[str] | None = None,
    timeout: float = LLM_CALL_TIMEOUT_SECONDS,
    program: ProposeProgram | None = None,
    catalog: OntologyLookup | None = None,
) -> str | None:
    """Ask the LLM to name a DBpedia property outright, then keep the answer
    only if it's a real ontology property. Returns the property's canonical
    local name, or `None` (LLM said "none", the catalog is unavailable, or
    the answer isn't a real property). Raises `LLMUnavailableError` when no
    model in the chain could be reached.

    Measured on 51 unseen Amharic field names via Groq: 34 correct, 1 wrong,
    16 abstained -- and "none" for all four nonsense/injection inputs."""

    models = tuple(model_chain) if model_chain else (model_alias,)
    proposed, _model = _propose_with_chain(
        amharic_property,
        target_class=target_class,
        models=models,
        timeout=timeout,
        program=program,
        catalog=catalog,
    )
    return proposed


@dataclass(frozen=True)
class PredictionResult:
    """The final chosen property, plus enough context to explain the choice."""

    property: str
    used_llm: bool
    candidates: list[str]
    top_retrieval_result: SearchResult | None
    reason: str = ""
    # True for both LLM-only paths below (propose-from-scratch, and
    # select-among-unverified-suggestions): retrieval itself had *no*
    # confident candidate, so a caller should flag the row as "needs a
    # closer look". Distinct from `used_llm`, which also covers the
    # ordinary rerank among confident candidates.
    llm_proposed: bool = False
    # How the property was chosen: "retrieval", "rerank", "learned",
    # "llm_propose" or "llm_select".
    method: str = "retrieval"
    # Which model in the chain actually answered, if any.
    model: str | None = None


@dataclass(frozen=True)
class NoCandidatesFound:
    query: str
    reason: str = "No retrieval candidates to rerank"
    # Ungated dense top-3, for a human to pick from. Never auto-accepted.
    suggestions: tuple[SearchResult, ...] = ()
    # Set when the LLM layer was wanted but no model could be reached, so
    # callers can say "not attempted" instead of implying "no such property".
    llm_error: str | None = None


PredictOutcome = PredictionResult | NoCandidatesFound

SuggestFunc = Callable[..., list[SearchResult]]
LearnedLookup = Callable[[str, str | None], str | None]

DEFAULT_PREDICT_CIRCUIT_BREAKER = RetrievalCircuitBreaker()
SUGGESTIONS_SHOWN = 3


def _safe_suggest(
    suggest_func: SuggestFunc | None, query: str, target_class: str | None, limit: int
) -> list[SearchResult]:
    if suggest_func is None:
        return []
    try:
        return list(suggest_func(query, target_class=target_class, limit=limit))
    except Exception as exc:  # noqa: BLE001 - suggestions are optional
        log_event(LOGGER, "predict.suggest_failed", error=exc.__class__.__name__)
        return []


def predict_property(
    amharic_property: str,
    *,
    target_class: str | None = None,
    model_alias: str = DEFAULT_MODEL_ALIAS,
    model_chain: Sequence[str] | None = None,
    timeout: float = LLM_CALL_TIMEOUT_SECONDS,
    top_k: int = DEFAULT_TOP_K,
    program: DspyProgram | None = None,
    circuit_breaker: RetrievalCircuitBreaker | None = DEFAULT_PREDICT_CIRCUIT_BREAKER,
    search_func: SearchFunc = search,
    suggest_func: SuggestFunc | None = None,
    learned_lookup: LearnedLookup | None = None,
    enable_propose: bool = True,
    propose_program: ProposeProgram | None = None,
    select_program: DspyProgram | None = None,
    catalog: OntologyLookup | None = None,
) -> PredictOutcome:
    """Choose a DBpedia property for one infobox field.

    In order:

    1. `learned_lookup` -- a human already approved this field for this
       class (`rag.learned_aliases`). Returned as-is, no LLM.
    2. `search_func` (hybrid retrieval, gated). If it finds confident
       candidates, the LLM reranks only those, with the target class in the
       premise. Any LLM failure keeps the retriever's top-1. Unverified
       alternatives are deliberately *not* appended here: measured live, that
       let the LLM override a correct curated answer
       (`የድህረ_ምረቃ_ተማሪዎች` -> numberOfGraduateStudents) as often as it fixed
       a cross-class one. A cross-class alias (a Hospital's staff count on a
       Company) is fixed by one reviewer correction instead -- step 1.
    3. Otherwise, when `enable_propose` and the circuit breaker allow it:
       a. propose-from-scratch (validated against the real ontology; may
          answer "none"), then
       b. if it abstained, select among `suggest_func`'s ungated dense
          top-k with an explicit "none" option.
       Measured live on 51 unseen field names: 2 correct before, 40 after.
    4. Else `NoCandidatesFound`, carrying the dense top-3 as suggestions and
       `llm_error` when no model could be reached.

    `model_chain` is tried in order for every LLM call (default: just
    `model_alias`). `program`, `propose_program`, `select_program`,
    `search_func`, `suggest_func` and `learned_lookup` exist so tests run
    fully offline.
    """

    models = tuple(model_chain) if model_chain else (model_alias,)

    if learned_lookup is not None:
        learned = learned_lookup(amharic_property, target_class)
        if learned:
            log_event(LOGGER, "predict.learned_alias", property=learned)
            return PredictionResult(
                property=learned,
                used_llm=False,
                candidates=[learned],
                top_retrieval_result=None,
                reason="Learned from an approved review",
                method="learned",
            )

    results = search_func(amharic_property, target_class=target_class, limit=top_k)
    scored = [result for result in results if isinstance(result, SearchResult)]
    if not scored:
        return _predict_without_candidates(
            amharic_property,
            target_class=target_class,
            models=models,
            timeout=timeout,
            top_k=top_k,
            circuit_breaker=circuit_breaker,
            suggest_func=suggest_func,
            enable_propose=enable_propose,
            propose_program=propose_program,
            select_program=select_program,
            catalog=catalog,
        )

    candidates = [result.property for result in scored]
    top_result = scored[0]

    if circuit_breaker is not None and not circuit_breaker.allow_request():
        log_event(LOGGER, "predict.circuit_open")
        return PredictionResult(
            property=top_result.property,
            used_llm=False,
            candidates=candidates,
            top_retrieval_result=top_result,
            reason="LLM reranking temporarily unavailable",
        )

    try:
        resolved_program = program if program is not None else _build_dspy_program()
        prediction, model = _run_on_model_chain(
            models,
            lambda: resolved_program(
                premise=_premise(amharic_property, target_class), candidates="; ".join(candidates)
            ),
            timeout=timeout,
            max_tokens=256,
        )
    except Exception as exc:
        if circuit_breaker is not None:
            circuit_breaker.record_failure()
        log_event(LOGGER, "predict.llm_unavailable", error=exc.__class__.__name__)
        return PredictionResult(
            property=top_result.property,
            used_llm=False,
            candidates=candidates,
            top_retrieval_result=top_result,
            reason=f"LLM reranking unavailable: {exc}",
        )

    if circuit_breaker is not None:
        circuit_breaker.record_success()
    chosen = snap_to_candidate(str(prediction.property_class), candidates)
    log_event(LOGGER, "predict.llm_reranked", model=model)
    return PredictionResult(
        property=chosen,
        used_llm=True,
        candidates=candidates,
        top_retrieval_result=top_result,
        method="rerank",
        model=model,
    )


def _predict_without_candidates(
    amharic_property: str,
    *,
    target_class: str | None,
    models: tuple[str, ...],
    timeout: float,
    top_k: int,
    circuit_breaker: RetrievalCircuitBreaker | None,
    suggest_func: SuggestFunc | None,
    enable_propose: bool,
    propose_program: ProposeProgram | None,
    select_program: DspyProgram | None,
    catalog: OntologyLookup | None,
) -> PredictOutcome:
    suggestions = _safe_suggest(suggest_func, amharic_property, target_class, top_k)
    shown = tuple(suggestions[:SUGGESTIONS_SHOWN])

    allow_llm = circuit_breaker is None or circuit_breaker.allow_request()
    if not enable_propose:
        return NoCandidatesFound(query=amharic_property, suggestions=shown)
    if not allow_llm:
        return NoCandidatesFound(
            query=amharic_property,
            suggestions=shown,
            llm_error="LLM temporarily disabled after repeated failures",
        )

    try:
        proposed, model = _propose_with_chain(
            amharic_property,
            target_class=target_class,
            models=models,
            timeout=timeout,
            program=propose_program,
            catalog=catalog,
        )
        if proposed is None and suggestions:
            names = [s.property for s in suggestions]
            resolved_select = (
                select_program if select_program is not None else _build_select_or_none_program()
            )
            answer, model = _run_on_model_chain(
                models,
                lambda: resolved_select(
                    premise=_premise(amharic_property, target_class), candidates="; ".join(names)
                ),
                timeout=timeout,
                max_tokens=64,
            )
            selected = _snap_or_none(str(answer.property_class), names)
            if selected is not None:
                if circuit_breaker is not None:
                    circuit_breaker.record_success()
                log_event(LOGGER, "predict.llm_selected", property=selected, model=model)
                return PredictionResult(
                    property=selected,
                    used_llm=True,
                    candidates=names,
                    top_retrieval_result=None,
                    reason="LLM-selected from unverified similarity suggestions",
                    llm_proposed=True,
                    method="llm_select",
                    model=model,
                )
    except Exception as exc:  # noqa: BLE001 - same degradation as a rerank failure
        if circuit_breaker is not None:
            circuit_breaker.record_failure()
        log_event(LOGGER, "predict.propose_unavailable", error=exc.__class__.__name__)
        return NoCandidatesFound(query=amharic_property, suggestions=shown, llm_error=str(exc))

    if circuit_breaker is not None:
        circuit_breaker.record_success()
    if proposed is None:
        return NoCandidatesFound(query=amharic_property, suggestions=shown)
    return PredictionResult(
        property=proposed,
        used_llm=True,
        candidates=[proposed],
        top_retrieval_result=None,
        reason="LLM-proposed (no retrieval candidates)",
        llm_proposed=True,
        method="llm_propose",
        model=model,
    )


def production_predict_options() -> dict[str, Any]:
    """`predict_property` keyword arguments for the running application.

    Reads the model chain and timeout from `Settings` (`PREDICT_MODEL`,
    `PREDICT_FALLBACK_MODEL`, `PREDICT_LLM_TIMEOUT_SECONDS`) and wires in the
    real ungated suggestions and learned-alias lookup. Settings that can't
    be loaded (no `.env` in a test or CI environment) degrade to the local
    default model only, instead of failing the request.

    LiteLLM reads provider keys from the process environment, not from
    `Settings`, so a `GROQ_API_KEY` that only lives in `.env` (e.g. the
    server started without `just`'s dotenv-load) is exported here for the
    hosted fallback to authenticate.
    """

    import os

    from config import DEFAULT_PREDICT_LLM_TIMEOUT_SECONDS, DEFAULT_PREDICT_MODEL, Settings
    from rag import learned_aliases
    from rag.retrieval import suggest

    chain: tuple[str, ...] = (DEFAULT_PREDICT_MODEL,)
    timeout = DEFAULT_PREDICT_LLM_TIMEOUT_SECONDS
    try:
        settings = Settings()
        chain = settings.predict_model_chain()
        timeout = settings.predict_llm_timeout_seconds
        if settings.groq_api_key:
            os.environ.setdefault("GROQ_API_KEY", settings.groq_api_key)
    except Exception as exc:  # noqa: BLE001 - fall back to the local default model
        log_event(LOGGER, "predict.settings_unavailable", error=exc.__class__.__name__)
    return {
        "model_chain": chain,
        "timeout": timeout,
        "suggest_func": suggest,
        "learned_lookup": learned_aliases.lookup,
    }
