"""Shared dense and sparse embedding helpers."""

from __future__ import annotations

import hashlib
import os
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from errors import RetrievalUnavailableError

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

# dice-research/amharic-property-retriever-afro-xlmr-base — the same model
# LLMIntegration/llm_raranker.py already benchmarks with, fine-tuned for
# Amharic-to-English DBpedia property retrieval specifically (refs 10.2).
DEFAULT_DENSE_MODEL = "dice-research/amharic-property-retriever-afro-xlmr-base"
DEFAULT_SPARSE_MODEL = "Qdrant/bm25"  # a FastEmbed model name, not the Qdrant service
DEFAULT_EMBEDDING_DEVICE = "cpu"
TOKEN_RE = re.compile(r"[\wሀ-፿]+", re.UNICODE)

_DENSE_MODEL_CACHE: dict[str, SentenceTransformer] = {}
_DENSE_MODEL_CACHE_LOCK = threading.Lock()
_SPARSE_MODEL_CACHE: dict[str, Any] = {}
_SPARSE_MODEL_CACHE_LOCK = threading.Lock()


def evict_model_caches() -> None:
    """Drop every cached embedder so the next call reloads fresh instances.

    Needed because an in-flight embedder can wedge (fastembed/ONNX worker
    killed, torch allocator fault, HF tokenizer broken pipe...) and every
    subsequent call then raises `[Errno 32] Broken pipe` indefinitely:
    nothing else ever rebuilds the cached object. Called from
    `retrieval.encode_query`'s except-branch so a transient failure
    recovers on the next request instead of pinning the service to
    `retrieval_unavailable` until restart.
    """

    with _DENSE_MODEL_CACHE_LOCK:
        _DENSE_MODEL_CACHE.clear()
    with _SPARSE_MODEL_CACHE_LOCK:
        _SPARSE_MODEL_CACHE.clear()


@dataclass(frozen=True)
class SparseVector:
    """Sparse vector representation, independent of any vector-database client."""

    indices: list[int]
    values: list[float]


DenseEmbedder = Callable[[str], list[float]]
SparseEmbedder = Callable[[str], SparseVector]


def lexical_sparse_vector(text: str, *, buckets: int = 65536) -> SparseVector:
    """Create a deterministic sparse vector that preserves exact lexical tokens.

    This fallback is intentionally simple and local. Production calls use
    FastEmbed's configured sparse model by default; tests use this function to
    avoid network/model downloads while still exercising sparse retrieval.
    """

    weights: dict[int, float] = {}
    for token in TOKEN_RE.findall(text.casefold()):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest, "big") % buckets
        weights[index] = weights.get(index, 0.0) + 1.0

    return SparseVector(
        indices=sorted(weights), values=[weights[index] for index in sorted(weights)]
    )


def deterministic_dense_vector(text: str, *, size: int = 16) -> list[float]:
    """Create a deterministic dense vector for tests and offline demos.

    Kept as a cheap, network-free stand-in for the real dense model — used by
    tests that need offline determinism, not real semantic similarity.
    """

    vector = [0.0] * size
    for token in TOKEN_RE.findall(text.casefold()):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
        index = int.from_bytes(digest[:4], "big") % size
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign

    norm = sum(value * value for value in vector) ** 0.5
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]


def _load_dense_model(model_name: str) -> SentenceTransformer:
    with _DENSE_MODEL_CACHE_LOCK:
        cached = _DENSE_MODEL_CACHE.get(model_name)
        if cached is not None:
            return cached

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - exercised only if dep missing
            raise RetrievalUnavailableError(
                "sentence-transformers is not installed; cannot load dense embedding model"
            ) from exc

        try:
            model = SentenceTransformer(model_name)
        except Exception as exc:
            raise RetrievalUnavailableError(
                f"Could not load dense embedding model {model_name!r}: {exc}"
            ) from exc

        _DENSE_MODEL_CACHE[model_name] = model
        return model


def _load_sparse_model(model_name: str) -> Any:
    with _SPARSE_MODEL_CACHE_LOCK:
        cached = _SPARSE_MODEL_CACHE.get(model_name)
        if cached is not None:
            return cached

        try:
            from fastembed.sparse import SparseTextEmbedding
        except ImportError as exc:  # pragma: no cover - exercised only if dep missing
            raise RetrievalUnavailableError(
                "fastembed is not installed; cannot load sparse embedding model"
            ) from exc

        try:
            model = SparseTextEmbedding(model_name=model_name, lazy_load=True)
        except Exception as exc:
            raise RetrievalUnavailableError(
                f"Could not load sparse embedding model {model_name!r}: {exc}"
            ) from exc

        _SPARSE_MODEL_CACHE[model_name] = model
        return model


def _embed_dense(
    text: str,
    *,
    model_name: str = DEFAULT_DENSE_MODEL,
    device: str | None = None,
) -> list[float]:
    """Embed text with the real afro-xlmr model, L2-normalized for cosine search."""

    resolved_device = device or os.environ.get("EMBEDDING_DEVICE", DEFAULT_EMBEDDING_DEVICE)
    model = _load_dense_model(model_name)
    vector = model.encode(
        text,
        convert_to_numpy=True,
        normalize_embeddings=True,
        device=resolved_device,
    )
    return [float(value) for value in vector]


def embed_dense(
    text: str,
    model_name: str = DEFAULT_DENSE_MODEL,
    device: str | None = None,
) -> list[float]:
    """Embed an ontology corpus chunk for dense retrieval."""

    return _embed_dense(text, model_name=model_name, device=device)


def embed_dense_batch(
    texts: list[str],
    *,
    model_name: str = DEFAULT_DENSE_MODEL,
    device: str | None = None,
) -> list[list[float]]:
    """Embed many corpus chunks in one batched forward pass.

    Building the retrieval index one document at a time (~2,948 real
    ontology properties) via repeated single-text `encode()` calls is
    dominated by Python/CPU-batch overhead, not model compute — batching
    is an order of magnitude faster on CPU. Used by rag/retrieval.py's
    build_index() for the production embedder path (refs 10.3).
    """

    if not texts:
        return []
    resolved_device = device or os.environ.get("EMBEDDING_DEVICE", DEFAULT_EMBEDDING_DEVICE)
    model = _load_dense_model(model_name)
    vectors = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        device=resolved_device,
        batch_size=64,
    )
    return [[float(value) for value in vector] for vector in vectors]


def embed_query_dense(
    text: str,
    model_name: str = DEFAULT_DENSE_MODEL,
    device: str | None = None,
) -> list[float]:
    """Embed a user query for dense retrieval."""

    return _embed_dense(text, model_name=model_name, device=device)


def dense_vector_size(model_name: str = DEFAULT_DENSE_MODEL) -> int:
    """The embedding dimension of the loaded model — computed, never hardcoded."""

    model = _load_dense_model(model_name)
    size = model.get_embedding_dimension()
    if size is None:  # pragma: no cover - defensive, SentenceTransformer always reports this
        raise RetrievalUnavailableError(f"Could not determine embedding size for {model_name!r}")
    return size


def embed_sparse(text: str, model_name: str = DEFAULT_SPARSE_MODEL) -> SparseVector:
    """Embed text with FastEmbed's sparse model."""

    try:
        embedding = next(_load_sparse_model(model_name).query_embed([text]))
    except RetrievalUnavailableError:
        raise
    except Exception as exc:
        raise RetrievalUnavailableError(f"Sparse embedding model unavailable: {exc}") from exc
    return SparseVector(
        indices=[int(index) for index in embedding.indices.tolist()],
        values=[float(value) for value in embedding.values.tolist()],
    )


def embed_sparse_batch(
    texts: list[str], model_name: str = DEFAULT_SPARSE_MODEL
) -> list[SparseVector]:
    """Embed many texts with FastEmbed's sparse model in one batched call."""

    if not texts:
        return []
    try:
        embeddings = list(_load_sparse_model(model_name).passage_embed(texts))
    except RetrievalUnavailableError:
        raise
    except Exception as exc:
        raise RetrievalUnavailableError(f"Sparse embedding model unavailable: {exc}") from exc
    return [
        SparseVector(
            indices=[int(index) for index in embedding.indices.tolist()],
            values=[float(value) for value in embedding.values.tolist()],
        )
        for embedding in embeddings
    ]
