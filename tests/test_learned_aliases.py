from __future__ import annotations

import json
from pathlib import Path

from rag.learned_aliases import lookup, normalize_field, record


def test_record_then_lookup_is_class_scoped(tmp_path: Path) -> None:
    path = tmp_path / "learned.json"

    added = record(
        "Book",
        [{"templateProperty": "የገጽ_ብዛት", "ontologyProperty": "numberOfPages"}],
        review_id="r1",
        path=path,
    )

    assert added == 1
    assert lookup("የገጽ ብዛት", "Book", path=path) == "numberOfPages"
    assert lookup("የገጽ_ብዛት", "Film", path=path) is None


def test_lookup_without_class_requires_agreement(tmp_path: Path) -> None:
    path = tmp_path / "learned.json"
    record("Lake", [{"templateProperty": "ከፍታ", "ontologyProperty": "elevation"}], path=path)
    record("Building", [{"templateProperty": "ከፍታ", "ontologyProperty": "height"}], path=path)
    record("Book", [{"templateProperty": "ደራሲ", "ontologyProperty": "author"}], path=path)

    assert lookup("ከፍታ", None, path=path) is None
    assert lookup("ደራሲ", None, path=path) == "author"


def test_newer_decision_replaces_older_one(tmp_path: Path) -> None:
    path = tmp_path / "learned.json"
    record("Company", [{"templateProperty": "ገቢ", "ontologyProperty": "income"}], path=path)
    record("Company", [{"templateProperty": "ገቢ", "ontologyProperty": "revenue"}], path=path)

    assert lookup("ገቢ", "Company", path=path) == "revenue"
    rows = json.loads(path.read_text(encoding="utf-8"))["aliases"]
    assert len(rows) == 1


def test_unchanged_and_incomplete_rows_are_not_rewritten(tmp_path: Path) -> None:
    path = tmp_path / "learned.json"
    record("Book", [{"templateProperty": "ደራሲ", "ontologyProperty": "author"}], path=path)

    assert (
        record("Book", [{"templateProperty": "ደራሲ", "ontologyProperty": "author"}], path=path) == 0
    )
    assert record("Book", [{"templateProperty": "", "ontologyProperty": "x"}], path=path) == 0


def test_missing_or_corrupt_file_means_no_aliases(tmp_path: Path) -> None:
    assert lookup("ደራሲ", "Book", path=tmp_path / "absent.json") is None
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert lookup("ደራሲ", "Book", path=bad) is None


def test_normalize_field_ignores_case_spaces_and_underscores() -> None:
    assert normalize_field(" GDP_በነፍስ  ወከፍ ") == "gdp_በነፍስ_ወከፍ"
