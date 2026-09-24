from __future__ import annotations

from pathlib import Path

import pytest

import rag.retrieval as retrieval
from rag.corpus import RetrievalDocument
from rag.embeddings import SparseVector, deterministic_dense_vector, lexical_sparse_vector
from rag.retrieval import NoMatchFound, SearchResult, encode_query, search

AIRPORT_ICAO = RetrievalDocument(
    property="icaoLocationIdentifier",
    curie="dbo:icaoLocationIdentifier",
    uri="http://dbpedia.org/ontology/icaoLocationIdentifier",
    label="icao location identifier",
    property_type="DatatypeProperty",
    domain="Airport",
    amharic_aliases=("አይካኦ_ኮድ",),
    english_aliases=("ICAO",),
)
DAM_ELEVATION = RetrievalDocument(
    property="elevation",
    curie="dbo:elevation",
    uri="http://dbpedia.org/ontology/elevation",
    label="elevation",
    property_type="DatatypeProperty",
    domain="Airport",
    amharic_aliases=("ከፍታ",),
    english_aliases=("airport elevation",),
)
DAM_OPENING_DATE = RetrievalDocument(
    property="openingDate",
    curie="dbo:openingDate",
    uri="http://dbpedia.org/ontology/openingDate",
    label="opening date",
    property_type="DatatypeProperty",
    domain="Dam",
    amharic_aliases=("የመክፈቻ_ቀን", "የግድብ_መክፈቻ_ቀን"),
    english_aliases=("opening date",),
)

TEST_CORPUS = [AIRPORT_ICAO, DAM_ELEVATION, DAM_OPENING_DATE]


def _fake_dense(text: str) -> list[float]:
    return deterministic_dense_vector(text, size=16)


def test_encode_query_uses_supplied_shared_embedders() -> None:
    calls: list[str] = []

    def dense(text: str) -> list[float]:
        calls.append(f"dense:{text}")
        return [1.0, 0.0]

    def sparse(text: str) -> SparseVector:
        calls.append(f"sparse:{text}")
        return SparseVector(indices=[1], values=[1.0])

    dense_vector, sparse_vector = encode_query(
        "አይካኦ_ኮድ", dense_embedder=dense, sparse_embedder=sparse
    )

    assert dense_vector == [1.0, 0.0]
    assert sparse_vector == SparseVector(indices=[1], values=[1.0])
    assert calls == ["dense:አይካኦ_ኮድ", "sparse:አይካኦ_ኮድ"]


def test_search_finds_exact_alias_match_via_sparse_channel() -> None:
    results = search(
        "አይካኦ_ኮድ ICAO",
        target_class="Airport",
        corpus=TEST_CORPUS,
        dense_embedder=_fake_dense,
        sparse_embedder=lexical_sparse_vector,
        confidence_threshold=0.1,
    )

    assert isinstance(results[0], SearchResult)
    assert results[0].property == "icaoLocationIdentifier"


def test_search_with_custom_embedders_does_not_require_groq_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    results = search(
        "አይካኦ_ኮድ ICAO",
        target_class="Airport",
        corpus=TEST_CORPUS,
        dense_embedder=_fake_dense,
        sparse_embedder=lexical_sparse_vector,
        confidence_threshold=0.1,
    )

    assert isinstance(results[0], SearchResult)
    assert results[0].property == "icaoLocationIdentifier"


def test_search_low_score_returns_no_match() -> None:
    results = search(
        "totally unrelated nonsense phrase about volcanoes",
        corpus=TEST_CORPUS,
        dense_embedder=_fake_dense,
        sparse_embedder=lexical_sparse_vector,
        confidence_threshold=0.5,
    )

    assert results == [NoMatchFound(query="totally unrelated nonsense phrase about volcanoes")]


def test_search_single_channel_rrf_without_alias_evidence_returns_no_match() -> None:
    # "elevation" only matches lexically on generic filler tokens shared with
    # the query, not on any curated alias, so the ambiguity guard must reject
    # it even though it is the single top-ranked hit.
    results = search(
        "የቡና ጣዕም መለኪያ",
        target_class="Airport",
        corpus=TEST_CORPUS,
        dense_embedder=_fake_dense,
        sparse_embedder=lexical_sparse_vector,
        confidence_threshold=0.35,
    )

    assert results == [NoMatchFound(query="የቡና ጣዕም መለኪያ")]


def test_search_single_channel_rrf_with_curated_alias_is_accepted() -> None:
    results = search(
        "የግድብ_መክፈቻ_ቀን",
        target_class="Dam",
        corpus=TEST_CORPUS,
        dense_embedder=_fake_dense,
        sparse_embedder=lexical_sparse_vector,
        confidence_threshold=0.35,
    )

    assert isinstance(results[0], SearchResult)
    assert results[0].property == "openingDate"


def test_search_empty_query_returns_no_match() -> None:
    results = search("   ", corpus=TEST_CORPUS, dense_embedder=_fake_dense)

    assert results == [NoMatchFound(query="   ", reason="Empty query")]


def test_search_target_class_breaks_ties_between_equally_ranked_documents() -> None:
    # dense_only_match wins solely on the dense channel (rank #1, no lexical
    # overlap with the query); sparse_only_match wins solely on the sparse
    # channel (an exact alias match, but an unrelated dense embedding). Both
    # therefore fuse to the same single-channel RRF score, 0.5 — a real tie
    # that only target_class can break.
    dense_only_match = RetrievalDocument(
        property="propA",
        curie="dbo:propA",
        uri="http://dbpedia.org/ontology/propA",
        label="unrelated label alpha",
        property_type="DatatypeProperty",
        domain="Airport",
    )
    sparse_only_match = RetrievalDocument(
        property="propB",
        curie="dbo:propB",
        uri="http://dbpedia.org/ontology/propB",
        label="unrelated label beta",
        property_type="DatatypeProperty",
        domain="Dam",
        amharic_aliases=("የጋራ_ቃል",),
    )

    def dense_embedder(text: str) -> list[float]:
        if text == "የጋራ_ቃል" or "alpha" in text:
            return [1.0, 0.0]
        return [0.0, 1.0]

    results = search(
        "የጋራ_ቃል",
        target_class="Dam",
        corpus=[dense_only_match, sparse_only_match],
        dense_embedder=dense_embedder,
        sparse_embedder=lexical_sparse_vector,
        confidence_threshold=0.1,
    )

    assert isinstance(results[0], SearchResult)
    assert results[0].property == "propB"


def test_search_exact_alias_breaks_tie_before_broader_domain_hint() -> None:
    dense_only_target_match = RetrievalDocument(
        property="iataAirlineCode",
        curie="dbo:iataAirlineCode",
        uri="http://dbpedia.org/ontology/iataAirlineCode",
        label="airline code",
        property_type="DatatypeProperty",
        domain="Airport",
    )
    sparse_alias_match = RetrievalDocument(
        property="iataLocationIdentifier",
        curie="dbo:iataLocationIdentifier",
        uri="http://dbpedia.org/ontology/iataLocationIdentifier",
        label="location identifier",
        property_type="DatatypeProperty",
        domain="Infrastructure",
        amharic_aliases=("አያታ_ኮድ",),
        english_aliases=("IATA",),
    )

    def dense_embedder(text: str) -> list[float]:
        if text == "አያታ_ኮድ IATA" or "airline" in text:
            return [1.0, 0.0]
        return [0.0, 1.0]

    results = search(
        "አያታ_ኮድ IATA",
        target_class="Airport",
        corpus=[dense_only_target_match, sparse_alias_match],
        dense_embedder=dense_embedder,
        sparse_embedder=lexical_sparse_vector,
        confidence_threshold=0.1,
    )

    assert isinstance(results[0], SearchResult)
    assert results[0].property == "iataLocationIdentifier"


def test_search_omits_ambiguous_single_channel_alternatives() -> None:
    good = RetrievalDocument(
        property="length",
        curie="dbo:length",
        uri="http://dbpedia.org/ontology/length",
        label="length",
        property_type="DatatypeProperty",
        domain="Bridge",
        amharic_aliases=("ርዝመት",),
    )
    dense_only = RetrievalDocument(
        property="penisLength",
        curie="dbo:penisLength",
        uri="http://dbpedia.org/ontology/penisLength",
        label="unrelated dense result",
        property_type="DatatypeProperty",
        domain="Person",
    )
    sparse_only = RetrievalDocument(
        property="lineLength",
        curie="dbo:lineLength",
        uri="http://dbpedia.org/ontology/lineLength",
        label="unrelated sparse result",
        property_type="DatatypeProperty",
        domain="RouteOfTransportation",
    )
    index = retrieval.RetrievalIndex(
        documents=[good, dense_only, sparse_only],
        dense_vectors=[[0.8, 0.0], [1.0, 0.0], [0.0, 1.0]],
        sparse_vectors=[
            SparseVector(indices=[1], values=[0.8]),
            SparseVector(indices=[2], values=[1.0]),
            SparseVector(indices=[1], values=[1.0]),
        ],
    )

    results = search(
        "ርዝመት",
        target_class="Bridge",
        index=index,
        dense_embedder=lambda _text: [1.0, 0.0],
        sparse_embedder=lambda _text: SparseVector(indices=[1], values=[1.0]),
        confidence_threshold=0.35,
    )

    assert [result.property for result in results if isinstance(result, SearchResult)] == ["length"]


def test_index_disk_cache_roundtrips_and_is_fingerprint_scoped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """build_index() re-embedding the whole ~2,948-property ontology corpus
    from scratch on every process start took 10+ minutes on a CPU-only
    embedder (confirmed live) -- the disk cache in rag/retrieval.py exists
    specifically to make every restart after the first one fast. Exercises
    the cache functions directly (not through the real embedder, which
    would need a live network + model download) but with the exact same
    fingerprint/save/load path build_index() itself calls."""

    monkeypatch.setattr(retrieval, "_INDEX_CACHE_DIR", tmp_path)

    fingerprint = retrieval._corpus_fingerprint(["hello", "world"])
    assert retrieval._load_cached_vectors(fingerprint) is None  # nothing cached yet

    dense_vectors = [[0.1, 0.2], [0.3, 0.4]]
    sparse_vectors = [
        SparseVector(indices=[1, 2], values=[0.5, 0.5]),
        SparseVector(indices=[3], values=[1.0]),
    ]
    retrieval._save_cached_vectors(fingerprint, dense_vectors, sparse_vectors)

    loaded = retrieval._load_cached_vectors(fingerprint)
    assert loaded == (dense_vectors, sparse_vectors)

    # A different corpus -- e.g. after `refresh-ontology`/`refresh-mappings`
    # actually changes the published mappings -- must never serve stale
    # vectors for content that was never embedded.
    other_fingerprint = retrieval._corpus_fingerprint(["hello", "different"])
    assert retrieval._load_cached_vectors(other_fingerprint) is None


def test_index_disk_cache_survives_a_corrupt_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cache file from an old format, or truncated by a killed process
    mid-write, must degrade to "rebuild the index" -- never crash the
    server that would otherwise be serving real requests."""

    monkeypatch.setattr(retrieval, "_INDEX_CACHE_DIR", tmp_path)

    fingerprint = retrieval._corpus_fingerprint(["hello"])
    cache_path = retrieval._index_cache_path(fingerprint)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(b"not a valid pickle")

    assert retrieval._load_cached_vectors(fingerprint) is None
