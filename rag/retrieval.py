"""In-process retrieval over the real DBpedia ontology property corpus.

Qdrant is gone (refs implementation.md 10.3/10.4): the corpus is small enough
(~2,948 properties) to embed once per process and rank in memory, the same
scale `LLMIntegration/llm_raranker.py` already proved out with `torch.topk`.
Dense (semantic) and sparse (lexical) channels are still fused with
reciprocal rank fusion, replicating Qdrant's own RRF behavior closely enough
that `RRF_SINGLE_CHANNEL_TOP_SCORE` and every existing confidence threshold
keep their original meaning: a document ranked #1 in exactly one channel
scores `1 / (1 + 1) = 0.5`, matching Qdrant's own single-channel RRF score.
"""

from __future__ import annotations

import hashlib
import logging
import os
import pickle
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config import DEFAULT_RETRIEVAL_CONFIDENCE_THRESHOLD, Settings
from errors import RetrievalUnavailableError
from logging_config import log_event
from rag.corpus import PROJECT_ROOT, RetrievalDocument, build_corpus
from rag.embeddings import (
    DEFAULT_DENSE_MODEL,
    DEFAULT_SPARSE_MODEL,
    DenseEmbedder,
    SparseEmbedder,
    SparseVector,
    embed_dense,
    embed_dense_batch,
    embed_query_dense,
    embed_sparse,
    embed_sparse_batch,
)

LOGGER = logging.getLogger("dbpedia_mapping_assistant.retrieval")

# build_index() embeds the whole ~2,948-property corpus in one batched pass
# and this module's own docstring already calls that a "once per process"
# cost -- true within a process, but every fresh `just run-http` pays it
# again from zero (confirmed live: ~10+ minutes on a CPU-only embedder),
# because _INDEX_CACHE only ever lived in RAM. The corpus (DBpedia ontology
# + published Amharic mappings) changes only when someone runs
# `refresh-ontology`/`refresh-mappings`, so a disk cache keyed by a hash of
# the actual document texts plus the embedder model names is safe: it's
# reused across restarts and automatically invalidated the moment the
# corpus content or embedding model actually changes, never served stale.
_INDEX_CACHE_DIR = PROJECT_ROOT / "data" / ".cache"
_INDEX_CACHE_VERSION = "v1"


def _corpus_fingerprint(texts: list[str]) -> str:
    digest = hashlib.sha256()
    digest.update(_INDEX_CACHE_VERSION.encode("utf-8"))
    digest.update(DEFAULT_DENSE_MODEL.encode("utf-8"))
    digest.update(DEFAULT_SPARSE_MODEL.encode("utf-8"))
    for text in texts:
        digest.update(b"\x00")
        digest.update(text.encode("utf-8"))
    return digest.hexdigest()


def _index_cache_path(fingerprint: str) -> Path:
    return _INDEX_CACHE_DIR / f"index_{fingerprint[:32]}.pkl"


def _load_cached_vectors(
    fingerprint: str,
) -> tuple[list[list[float]], list[SparseVector]] | None:
    cache_path = _index_cache_path(fingerprint)
    if not cache_path.is_file():
        return None
    try:
        with cache_path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception as exc:  # a corrupt/foreign-format cache file must never
        # crash retrieval -- fall back to rebuilding exactly as if there
        # were no cache at all.
        log_event(LOGGER, "retrieval.index_cache_unreadable", error=exc.__class__.__name__)
        return None
    if payload.get("fingerprint") != fingerprint:
        return None
    return payload["dense_vectors"], payload["sparse_vectors"]


def _save_cached_vectors(
    fingerprint: str, dense_vectors: list[list[float]], sparse_vectors: list[SparseVector]
) -> None:
    try:
        _INDEX_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path = _index_cache_path(fingerprint)
        tmp_path = cache_path.with_suffix(".pkl.tmp")
        with tmp_path.open("wb") as handle:
            pickle.dump(
                {
                    "fingerprint": fingerprint,
                    "dense_vectors": dense_vectors,
                    "sparse_vectors": sparse_vectors,
                },
                handle,
            )
        tmp_path.replace(cache_path)  # atomic on the same filesystem -- a
        # reader never sees a half-written file.
    except OSError as exc:  # a read-only filesystem or full disk should
        # degrade to "rebuild every time", not crash the server.
        log_event(LOGGER, "retrieval.index_cache_write_failed", error=exc.__class__.__name__)


@dataclass(frozen=True)
class SearchResult:
    property: str
    ontology_class: str
    score: float
    payload: dict[str, Any]


@dataclass(frozen=True)
class NoMatchFound:
    query: str
    reason: str = "No Match Found"


RetrievalResult = SearchResult | NoMatchFound


@dataclass
class RetrievalCircuitBreaker:
    """Small in-process circuit breaker for transient embedding-model outages.

    Retargeted from Qdrant connectivity (10.4 removes that entirely) to the
    real failure mode now: the dense/sparse model failing to load or infer.
    """

    failure_threshold: int = 3
    reset_after_seconds: float = 30.0
    failures: int = 0
    opened_at: float | None = None

    def allow_request(self, now: float | None = None) -> bool:
        current = now if now is not None else time.monotonic()
        if self.opened_at is None:
            return True
        if current - self.opened_at >= self.reset_after_seconds:
            self.failures = 0
            self.opened_at = None
            return True
        return False

    def record_success(self) -> None:
        self.failures = 0
        self.opened_at = None

    def record_failure(self, now: float | None = None) -> None:
        self.failures += 1
        if self.failures >= self.failure_threshold:
            self.opened_at = now if now is not None else time.monotonic()


DEFAULT_CIRCUIT_BREAKER = RetrievalCircuitBreaker()
RRF_SINGLE_CHANNEL_TOP_SCORE = 0.5
PREFETCH_LIMIT = 20  # top-K per channel considered for RRF fusion


def encode_query(
    amharic_text: str,
    *,
    dense_embedder: DenseEmbedder = embed_query_dense,
    sparse_embedder: SparseEmbedder = embed_sparse,
) -> tuple[list[float], SparseVector]:
    """Encode a query with the exact same embedders used for indexing."""

    try:
        return dense_embedder(amharic_text), sparse_embedder(amharic_text)
    except Exception as exc:
        raise RetrievalUnavailableError(f"Embedding model unavailable: {exc}") from exc


def _normalized_alias(value: str) -> str:
    terms = value.casefold().replace("_", " ").split()
    return " ".join(term[1:] if term.startswith("የ") and len(term) > 1 else term for term in terms)


def _has_curated_alias_evidence(query: str, doc: RetrievalDocument) -> bool:
    """Require direct corpus evidence for an ambiguous single-channel RRF hit."""

    normalized_query = _normalized_alias(query)
    aliases = [*doc.amharic_aliases, *doc.english_aliases, doc.property]

    return any(
        normalized_alias
        and (normalized_alias in normalized_query or normalized_query in normalized_alias)
        for alias in aliases
        if (normalized_alias := _normalized_alias(alias))
    )


def _document_payload(doc: RetrievalDocument) -> dict[str, Any]:
    return {
        "curie": doc.curie,
        "uri": doc.uri,
        "label": doc.label,
        "property_type": doc.property_type,
        "amharic_aliases": list(doc.amharic_aliases),
        "english_aliases": list(doc.english_aliases),
    }


@dataclass
class RetrievalIndex:
    """The in-process embedding index: one dense + one sparse vector per doc."""

    documents: list[RetrievalDocument]
    dense_vectors: list[list[float]] = field(repr=False)
    sparse_vectors: list[SparseVector] = field(repr=False)

    def __len__(self) -> int:
        return len(self.documents)


_INDEX_CACHE: RetrievalIndex | None = None
_INDEX_LOCK = threading.Lock()


def build_index(
    *,
    documents: list[RetrievalDocument] | None = None,
    dense_embedder: DenseEmbedder = embed_dense,
    sparse_embedder: SparseEmbedder = embed_sparse,
) -> RetrievalIndex:
    """Embed every corpus document once. Expensive — call get_index() instead
    unless you specifically need a fresh, uncached index (e.g. in tests)."""

    docs = documents if documents is not None else build_corpus()
    texts = [doc.search_text() for doc in docs]

    # Only the real, default embedders are worth disk-caching: test callers
    # pass cheap deterministic fakes (already instant, and must never be
    # cached alongside the real model's vectors), and `documents` being
    # explicitly given is exactly that same "don't touch the shared cache"
    # signal `get_index()` already uses below.
    using_production_embedders = (
        documents is None and dense_embedder is embed_dense and sparse_embedder is embed_sparse
    )
    fingerprint = _corpus_fingerprint(texts) if using_production_embedders else None

    if fingerprint is not None:
        cached = _load_cached_vectors(fingerprint)
        if cached is not None:
            dense_vectors, sparse_vectors = cached
            log_event(LOGGER, "retrieval.index_loaded_from_cache", document_count=len(docs))
            return RetrievalIndex(
                documents=docs, dense_vectors=dense_vectors, sparse_vectors=sparse_vectors
            )

    log_event(LOGGER, "retrieval.index_build_started", document_count=len(docs))

    # The production embedders support batched calls that are an order of
    # magnitude faster on CPU than per-document calls (~2,948 real ontology
    # properties otherwise pays per-call overhead ~2,948 times over). Custom
    # test embedders are plain single-text callables and stay per-document —
    # they're already instant, so batching would only add complexity there.
    if dense_embedder is embed_dense:
        dense_vectors = embed_dense_batch(texts)
    else:
        dense_vectors = [dense_embedder(text) for text in texts]

    if sparse_embedder is embed_sparse:
        sparse_vectors = embed_sparse_batch(texts)
    else:
        sparse_vectors = [sparse_embedder(text) for text in texts]

    log_event(LOGGER, "retrieval.index_build_completed", document_count=len(docs))

    if fingerprint is not None:
        _save_cached_vectors(fingerprint, dense_vectors, sparse_vectors)

    return RetrievalIndex(
        documents=docs, dense_vectors=dense_vectors, sparse_vectors=sparse_vectors
    )


def get_index(
    *,
    documents: list[RetrievalDocument] | None = None,
    dense_embedder: DenseEmbedder = embed_dense,
    sparse_embedder: SparseEmbedder = embed_sparse,
    force_rebuild: bool = False,
) -> RetrievalIndex:
    """Return the process-wide cached index, building it on first use.

    Passing an explicit `documents` corpus (as tests do, with cheap fake
    embedders) always builds a fresh index rather than touching the module
    -wide cache, so tests never fight each other or production traffic over
    a shared singleton.
    """

    global _INDEX_CACHE

    if documents is not None:
        return build_index(
            documents=documents, dense_embedder=dense_embedder, sparse_embedder=sparse_embedder
        )

    with _INDEX_LOCK:
        if _INDEX_CACHE is None or force_rebuild:
            _INDEX_CACHE = build_index(
                dense_embedder=dense_embedder, sparse_embedder=sparse_embedder
            )
        return _INDEX_CACHE


def _dense_similarity(query: list[float], vector: list[float]) -> float:
    """Cosine similarity, simplified to a dot product since embed_dense/
    embed_query_dense both L2-normalize their output."""

    return sum(x * y for x, y in zip(query, vector, strict=True))


def _sparse_similarity(query: SparseVector, vector: SparseVector) -> float:
    doc_weights = dict(zip(vector.indices, vector.values, strict=True))
    return sum(
        weight * doc_weights.get(index, 0.0)
        for index, weight in zip(query.indices, query.values, strict=True)
    )


def _top_ranked(scores: list[float], *, limit: int) -> list[int]:
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [i for i in ranked[:limit] if scores[i] > 0.0]


def _reciprocal_rank_fuse(*rankings: list[int]) -> dict[int, float]:
    """Qdrant-equivalent RRF: rank 1 (best) in one channel contributes
    1 / (1 + 1) = 0.5; a hit in both channels' #1 slot scores 1.0."""

    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_index in enumerate(ranking, start=1):
            fused[doc_index] = fused.get(doc_index, 0.0) + 1.0 / (1 + rank)
    return fused


@dataclass(frozen=True)
class RetrievalSettings:
    retrieval_confidence_threshold: float


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _retrieval_settings(settings: Settings | None) -> RetrievalSettings:
    if settings is not None:
        return RetrievalSettings(
            retrieval_confidence_threshold=settings.retrieval_confidence_threshold
        )
    return RetrievalSettings(
        retrieval_confidence_threshold=_env_float(
            "RETRIEVAL_CONFIDENCE_THRESHOLD", DEFAULT_RETRIEVAL_CONFIDENCE_THRESHOLD
        )
    )


def search(
    query: str,
    *,
    target_class: str | None = None,
    limit: int = 3,
    confidence_threshold: float | None = None,
    dense_embedder: DenseEmbedder = embed_query_dense,
    sparse_embedder: SparseEmbedder = embed_sparse,
    settings: Settings | None = None,
    circuit_breaker: RetrievalCircuitBreaker | None = DEFAULT_CIRCUIT_BREAKER,
    corpus: list[RetrievalDocument] | None = None,
    index: RetrievalIndex | None = None,
) -> list[RetrievalResult]:
    """Search the in-process ontology property index with dense+sparse RRF.

    `target_class` is a soft ranking hint, not a hard filter: the real
    ontology's `rdfs:domain` is sparse and often broader than the class a
    property is actually used on (e.g. `iataLocationIdentifier`'s domain is
    `Infrastructure`, not `Airport`), so excluding non-matching domains would
    silently drop correct answers. For equal RRF scores, direct curated
    alias evidence wins first, then `target_class` breaks any remaining tie.

    `index` lets a caller reuse an already-embedded `RetrievalIndex` across
    many `search()` calls (built once via `build_index()`) instead of paying
    to re-embed the whole corpus on every call, as passing `corpus` alone
    would — mainly useful for tests that run many queries over a fixed
    corpus. `index` takes precedence over `corpus` when both are given.
    """

    if not query.strip():
        return [NoMatchFound(query=query, reason="Empty query")]
    if circuit_breaker is not None and not circuit_breaker.allow_request():
        log_event(LOGGER, "retrieval.circuit_open", target_class=target_class)
        return [NoMatchFound(query=query, reason="Retrieval temporarily unavailable")]
    log_event(LOGGER, "retrieval.start", target_class=target_class, limit=limit)

    resolved_settings = _retrieval_settings(settings)
    threshold = (
        confidence_threshold
        if confidence_threshold is not None
        else resolved_settings.retrieval_confidence_threshold
    )

    try:
        # A custom corpus (tests only) is always embedded with the same
        # dense/sparse embedders used for the query, so an offline test stays
        # fully offline on both sides. Production (corpus=None, index=None)
        # leaves get_index() to use its own passage-side defaults regardless
        # of which query-side embedder was passed in.
        if index is not None:
            retrieval_index = index
        elif corpus is not None:
            retrieval_index = get_index(
                documents=corpus, dense_embedder=dense_embedder, sparse_embedder=sparse_embedder
            )
        else:
            retrieval_index = get_index()
        query_dense, query_sparse = encode_query(
            query, dense_embedder=dense_embedder, sparse_embedder=sparse_embedder
        )
    except RetrievalUnavailableError:
        if circuit_breaker is not None:
            circuit_breaker.record_failure()
        raise
    except Exception as exc:
        if circuit_breaker is not None:
            circuit_breaker.record_failure()
        log_event(LOGGER, "retrieval.unavailable", error=exc.__class__.__name__)
        raise RetrievalUnavailableError("Retrieval index is unavailable") from exc

    dense_scores = [
        _dense_similarity(query_dense, vector) for vector in retrieval_index.dense_vectors
    ]
    sparse_scores = [
        _sparse_similarity(query_sparse, vector) for vector in retrieval_index.sparse_vectors
    ]
    dense_ranked = _top_ranked(dense_scores, limit=PREFETCH_LIMIT)
    sparse_ranked = _top_ranked(sparse_scores, limit=PREFETCH_LIMIT)
    fused = _reciprocal_rank_fuse(dense_ranked, sparse_ranked)

    if circuit_breaker is not None:
        circuit_breaker.record_success()

    if not fused:
        log_event(LOGGER, "retrieval.no_match", threshold=threshold)
        return [NoMatchFound(query=query)]

    def _sort_key(item: tuple[int, float]) -> tuple[float, int, int]:
        doc_index, score = item
        document = retrieval_index.documents[doc_index]
        alias_match = int(_has_curated_alias_evidence(query, document))
        domain = document.domain
        domain_match = int(
            bool(target_class and domain and domain.casefold() == target_class.casefold())
        )
        return (score, alias_match, domain_match)

    ranked = sorted(fused.items(), key=_sort_key, reverse=True)
    top_index, top_score = ranked[0]
    top_doc = retrieval_index.documents[top_index]
    ambiguous_without_alias = (
        top_score <= RRF_SINGLE_CHANNEL_TOP_SCORE
        and not _has_curated_alias_evidence(query, top_doc)
    )
    if top_score < threshold or ambiguous_without_alias:
        log_event(LOGGER, "retrieval.no_match", threshold=threshold)
        return [NoMatchFound(query=query)]

    results: list[RetrievalResult] = []
    for doc_index, score in ranked:
        doc = retrieval_index.documents[doc_index]
        ambiguous_without_alias = (
            score <= RRF_SINGLE_CHANNEL_TOP_SCORE and not _has_curated_alias_evidence(query, doc)
        )
        if score < threshold or ambiguous_without_alias:
            continue
        results.append(
            SearchResult(
                property=doc.property,
                ontology_class=doc.domain or "",
                score=score,
                payload=_document_payload(doc),
            )
        )
        if len(results) >= limit:
            break
    log_event(LOGGER, "retrieval.complete", result_count=len(results))
    return results or [NoMatchFound(query=query)]
