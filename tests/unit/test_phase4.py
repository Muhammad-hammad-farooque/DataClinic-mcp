"""Phase 4 tests: cleaning, undo and the refusal policy.

Every test here mutates data, so each builds its own server: the shared,
module-scoped server of the read-only Phase 2 tests must never see a change.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eda_mcp.config import Settings, load_settings
from eda_mcp.errors import EDAError
from eda_mcp.mutations import apply
from eda_mcp.server import build_server

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def call(server, tool, args):  # type: ignore[no-untyped-def]
    result = asyncio.run(server.call_tool(tool, args))
    payload = result[1] if isinstance(result, tuple) else result
    return payload.structured_content


@pytest.fixture()
def settings() -> Settings:
    return load_settings(log_level="WARNING", allowed_paths=[FIXTURES.parents[1]])


@pytest.fixture()
def server(settings: Settings):  # type: ignore[no-untyped-def]
    srv = build_server(settings)
    call(srv, "load_dataset", {"source": str(FIXTURES / "messy.csv"), "alias": "m"})
    return srv


def clean(server, *operations):  # type: ignore[no-untyped-def]
    return call(server, "clean_data", {"alias": "m", "operations": list(operations)})


FRAME = pd.DataFrame(
    {
        "age": [20.0, np.nan, 40.0, -999.0, 60.0, np.nan],
        "city": [" Paris", "paris ", "PARIS", "Rome", None, "Rome"],
        "score": [1.0, 2.0, 3.0, 4.0, 5.0, 100.0],
    }
)


# --------------------------------------------------------------------------
# operations, one by one


def test_fill_missing_methods() -> None:
    median, outcomes = apply(FRAME, [{"op": "fill_missing", "column": "age", "method": "median"}])
    assert median["age"].isna().sum() == 0
    assert median.loc[1, "age"] == FRAME["age"].median()
    assert outcomes[0].cells_changed == 2

    constant, _ = apply(
        FRAME, [{"op": "fill_missing", "column": "age", "method": "constant", "value": 0}]
    )
    assert constant.loc[1, "age"] == 0

    flagged, outcomes = apply(FRAME, [{"op": "fill_missing", "column": "age", "flag": True}])
    assert flagged["age_was_missing"].tolist() == [0, 1, 0, 0, 0, 1]
    assert outcomes[0].columns_added == ["age_was_missing"]


def test_mode_is_chosen_independently_of_row_order() -> None:
    frame = pd.DataFrame({"c": ["b", "a", None, "a", "b"]})  # a and b tie
    for shuffled in (frame, frame.iloc[::-1]):
        filled, _ = apply(shuffled, [{"op": "fill_missing", "column": "c", "method": "mode"}])
        assert filled.loc[2, "c"] == "a"  # ties go to the smallest value


def test_replace_values_matches_json_string_keys_to_numbers() -> None:
    replaced, outcomes = apply(
        FRAME, [{"op": "replace_values", "column": "age", "mapping": {"-999": None}}]
    )
    assert np.isnan(replaced.loc[3, "age"]) and outcomes[0].cells_changed == 1


def test_text_operations() -> None:
    stripped, _ = apply(FRAME, [{"op": "strip_whitespace", "column": "city"}])
    assert stripped["city"].tolist()[:3] == ["Paris", "paris", "PARIS"]
    merged, outcomes = apply(
        FRAME,
        [{"op": "strip_whitespace", "column": "city"}, {"op": "merge_variants", "column": "city"}],
    )
    # three spellings, one count each: the tie goes to the first in sort order
    assert set(merged["city"].dropna()) == {"PARIS", "Rome"}
    assert outcomes[1].cells_changed == 2
    lowered, _ = apply(FRAME, [{"op": "standardize_case", "column": "city", "case": "lower"}])
    assert lowered.loc[2, "city"] == "paris"


def test_outliers_drop_clip_and_flag() -> None:
    clipped, outcomes = apply(
        FRAME, [{"op": "remove_outliers", "column": "score", "action": "clip"}]
    )
    assert clipped["score"].max() < 100 and outcomes[0].cells_changed == 1
    flagged, _ = apply(FRAME, [{"op": "remove_outliers", "column": "score"}])  # default: flag
    assert flagged["score_outlier"].tolist() == [0, 0, 0, 0, 0, 1]
    dropped, outcomes = apply(
        FRAME, [{"op": "remove_outliers", "column": "score", "action": "drop"}]
    )
    assert len(dropped) == 5 and outcomes[0].rows_removed == 1


def test_cast_and_parse() -> None:
    frame = pd.DataFrame(
        {"n": ["1,200", "3", "x"], "d": ["2024-01-02", "2024-02-03", "2024-03-04"]}
    )
    cast, outcomes = apply(frame, [{"op": "cast_type", "column": "n", "to": "numeric"}])
    assert cast["n"].tolist()[:2] == [1200, 3] and pd.isna(cast.loc[2, "n"])
    assert "1 value(s) could not be converted" in (outcomes[0].note or "")
    parsed, _ = apply(frame, [{"op": "parse_dates", "column": "d"}])
    assert str(parsed["d"].dtype).startswith("datetime64")


def test_drop_rows_uses_the_query_language() -> None:
    kept, outcomes = apply(FRAME, [{"op": "drop_rows", "where": "age < 0"}])
    assert len(kept) == 5 and outcomes[0].rows_removed == 1


def test_rename_refuses_duplicate_names() -> None:
    renamed, _ = apply(FRAME, [{"op": "rename_columns", "mapping": {"city": "town"}}])
    assert "town" in renamed.columns
    with pytest.raises(EDAError):
        apply(FRAME, [{"op": "rename_columns", "mapping": {"city": "age"}}])


# --------------------------------------------------------------------------
# refusal policy and atomicity

SPARSE = FRAME.assign(
    sparse=[1.0, np.nan, np.nan, np.nan, np.nan, 2.0],
    gappy=[np.nan, np.nan, np.nan, np.nan, 1.0, 1.0],
)


@pytest.mark.parametrize(
    ("operation", "reason"),
    [
        ({"op": "drop_rows", "where": "score < 50"}, "over the 50% limit"),
        ({"op": "drop_missing", "column": "gappy"}, "over the 50% limit"),
        ({"op": "fill_missing", "column": "sparse"}, "imputing would invent most of it"),
        ({"op": "drop_columns", "columns": list(SPARSE.columns)}, "every column"),
        ({"op": "cast_type", "column": "city", "to": "numeric"}, "would turn 100%"),
    ],
)
def test_refusal_policy(operation, reason) -> None:  # type: ignore[no-untyped-def]
    after, outcomes = apply(SPARSE, [operation])
    assert outcomes[0].refused is not None and reason in outcomes[0].refused.reason
    pd.testing.assert_frame_equal(after, SPARSE)  # a refusal changes nothing


def test_dropping_most_of_the_data_is_refused() -> None:
    frame = pd.DataFrame({"a": range(10), "b": range(10), "c": [1.0] + [np.nan] * 9})
    _, outcomes = apply(frame, [{"op": "drop_columns", "columns": ["a", "b"]}])  # 20 of 21 cells
    assert outcomes[0].refused is not None
    assert "95% of the remaining data" in outcomes[0].refused.reason
    kept, outcomes = apply(frame, [{"op": "drop_columns", "column": "a"}])  # 10 of 21: fine
    assert outcomes[0].refused is None and list(kept.columns) == ["b", "c"]


def test_a_refused_operation_leaves_no_partial_change() -> None:
    """fill_missing over [fine, too sparse] must not keep the first column's fill."""
    after, outcomes = apply(SPARSE, [{"op": "fill_missing", "columns": ["age", "sparse"]}])
    assert outcomes[0].refused is not None
    assert after["age"].isna().sum() == 2  # untouched, not half-filled


def test_original_frame_is_never_modified() -> None:
    before = FRAME.copy()
    apply(
        FRAME,
        [
            {"op": "fill_missing", "column": "age"},
            {"op": "strip_whitespace", "column": "city"},
            {"op": "remove_outliers", "column": "score", "action": "clip"},
            {"op": "drop_rows", "where": "age < 0"},
        ],
    )
    pd.testing.assert_frame_equal(FRAME, before)


@pytest.mark.parametrize(
    ("operations", "message"),
    [
        ([], "non-empty list"),
        ([{"column": "age"}], "has no op"),
        ([{"op": "explode"}], "unknown operation"),
        ([{"op": "fill_missing", "colum": "age"}], "unknown parameter(s) colum"),
        ([{"op": "drop_columns", "column": "nope"}], "no column named 'nope'"),
        ([{"op": "fill_missing", "column": "country", "method": "mean"}], "needs a numeric column"),
        ([{"op": "fill_missing", "column": "age", "method": "knn"}], "not available"),
    ],
)
def test_malformed_batches_change_nothing(server, operations, message) -> None:  # type: ignore[no-untyped-def]
    payload = call(server, "clean_data", {"alias": "m", "operations": operations})
    assert payload["error"]["code"] in ("INVALID_OPERATION", "COLUMN_NOT_FOUND")
    assert message in payload["error"]["message"]
    assert call(server, "history", {"alias": "m"})["summary"].startswith("No changes")


def test_errors_name_the_failing_step(server) -> None:  # type: ignore[no-untyped-def]
    payload = clean(
        server,
        {"op": "drop_duplicates"},
        {"op": "fill_missing", "column": "country", "method": "mean"},
    )
    assert payload["error"]["message"].startswith("operation 2 (fill_missing)")
    assert call(server, "manage_sources", {})["datasets"][0]["shape"] == [5150, 10]


# --------------------------------------------------------------------------
# the tool, history and undo


def test_clean_the_fixture_with_find_issues_advice(server) -> None:  # type: ignore[no-untyped-def]
    before = call(server, "find_issues", {"source": "m"})["counts"]
    payload = clean(
        server,
        {"op": "drop_duplicates"},
        {"op": "drop_columns", "columns": ["notes", "region_code"]},
        {"op": "replace_values", "column": "age", "mapping": {"-5": None}},
        {"op": "merge_variants", "column": "country"},
        {"op": "fill_missing", "column": "age", "method": "median", "flag": True},
        {"op": "flag_missing", "column": "country"},
        {"op": "drop_rows", "where": "price > 0"},  # every row: refused, the rest still apply
    )
    assert payload["shape_before"] == [5150, 10] and payload["shape_after"] == [5000, 10]
    assert len(payload["applied"]) == 6
    assert payload["findings"][0].startswith("MED  refused drop_rows")
    after = call(server, "find_issues", {"source": "m"})
    assert before["HIGH"] == 2 and after["counts"].get("HIGH", 0) == 0
    # the flag is recognised: no circular advice to add it again
    assert any("flagged in country_was_missing" in f for f in after["findings"])
    assert not any("country_was_missing is higher" in f for f in after["findings"])


def test_history_and_undo(server) -> None:  # type: ignore[no-untyped-def]
    clean(server, {"op": "drop_duplicates"})
    clean(server, {"op": "drop_columns", "column": "notes"})
    listed = call(server, "history", {"alias": "m"})
    assert list(listed["steps"]) == ["1", "2"] and listed["shape"] == [5000, 9]
    assert listed["steps"]["1"]["operations"] == ["drop_duplicates all columns: -150 rows"]

    undone = call(server, "history", {"alias": "m", "action": "undo"})
    assert undone["undone"] == 1 and undone["shape"] == [5000, 10]
    undone = call(server, "history", {"alias": "m", "action": "undo", "steps": 5})
    assert undone["undone"] == 1 and undone["shape"] == [5150, 10]
    assert "Only 1 of 5" in undone["summary"]
    nothing = call(server, "history", {"alias": "m", "action": "undo"})
    assert nothing["error"]["code"] == "INVALID_OPERATION"


def test_undo_restores_the_exact_frame(server) -> None:  # type: ignore[no-untyped-def]
    original = call(server, "profile", {"source": "m", "detail": "full"})
    clean(server, {"op": "fill_missing", "column": "age"}, {"op": "strip_whitespace"})
    assert call(server, "profile", {"source": "m", "detail": "full"}) != original
    call(server, "history", {"alias": "m", "action": "undo"})
    assert call(server, "profile", {"source": "m", "detail": "full"}) == original


def test_without_a_snapshot_budget_changes_are_not_undoable() -> None:
    srv = build_server(
        load_settings(log_level="WARNING", allowed_paths=[FIXTURES.parents[1]], max_snapshot_mb=0)
    )
    call(srv, "load_dataset", {"source": str(FIXTURES / "messy.csv"), "alias": "m"})
    payload = call(srv, "clean_data", {"alias": "m", "operations": [{"op": "drop_duplicates"}]})
    assert "Not undoable" in payload["summary"]
    undo = call(srv, "history", {"alias": "m", "action": "undo"})
    assert undo["error"]["code"] == "INVALID_OPERATION"


def test_sources_on_disk_are_never_touched(server) -> None:  # type: ignore[no-untyped-def]
    path = FIXTURES / "messy.csv"
    before = path.read_bytes()
    clean(server, {"op": "drop_duplicates"}, {"op": "drop_columns", "column": "notes"})
    assert path.read_bytes() == before


def test_unknown_dataset_and_action(server) -> None:  # type: ignore[no-untyped-def]
    payload = call(
        server, "clean_data", {"alias": "nope", "operations": [{"op": "drop_duplicates"}]}
    )
    assert payload["error"]["code"] == "SOURCE_NOT_FOUND"
    assert call(server, "history", {"alias": "nope"})["error"]["code"] == "SOURCE_NOT_FOUND"
    redo = call(server, "history", {"alias": "m", "action": "redo"})
    assert redo["error"]["code"] == "INVALID_OPERATION"
