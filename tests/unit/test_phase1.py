"""Phase 1 tests: loading, classification, budgeting, errors, registry.

The invariants matter more than the unit assertions. In particular:
statistics must not depend on row order, and the source file must be
byte-identical after a session.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from eda_mcp.config import Settings, load_settings
from eda_mcp.digest import (
    DEFAULT_BUDGET,
    Finding,
    Response,
    Severity,
    compact,
    estimate_tokens,
    round_sig,
)
from eda_mcp.errors import (
    ColumnNotFoundError,
    EDAError,
    ErrorClass,
    ErrorCode,
    PathNotAllowedError,
    SourceNotFoundError,
    SourceTooLargeError,
    redact,
)
from eda_mcp.loaders import detect_delimiter, detect_encoding, load_file
from eda_mcp.profiling import ColumnKind, classify, column_kinds, orientation
from eda_mcp.registry import Registry
from eda_mcp.server import build_server, default_alias

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


@pytest.fixture(scope="session")
def settings() -> Settings:
    return load_settings(log_level="WARNING", allowed_paths=[FIXTURES.parents[1]])


def call(server, tool, args):  # type: ignore[no-untyped-def]
    """Invoke a tool and return its structured payload."""
    result = asyncio.run(server.call_tool(tool, args))
    payload = result[1] if isinstance(result, tuple) else result
    return payload.structured_content


# --------------------------------------------------------------------------
# errors


def test_error_envelope_shape() -> None:
    payload = SourceTooLargeError("orders", 41_238_904, 5_000_000).to_dict()["error"]
    assert payload["code"] == "SOURCE_TOO_LARGE"
    assert payload["class"] == ErrorClass.USER.value
    assert payload["retryable"] is False
    assert "remedy" in payload


def test_only_infra_errors_are_retryable() -> None:
    assert not SourceNotFoundError("x").retryable
    assert EDAError(ErrorCode.CONNECTION_FAILED, "down").retryable
    assert EDAError(ErrorCode.QUERY_TIMEOUT, "slow").retryable


@pytest.mark.parametrize(
    "text",
    [
        "postgresql://user:hunter2@host/db",
        "password=hunter2",
        "token: ghp_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
    ],
)
def test_credentials_are_redacted(text: str) -> None:
    assert "hunter2" not in redact(text)
    assert "ghp_" not in redact(text)


def test_column_error_truncates_long_column_lists() -> None:
    err = ColumnNotFoundError("nope", [f"c{i}" for i in range(100)])
    assert "+85 more" in (err.remedy or "")


# --------------------------------------------------------------------------
# config


def test_path_traversal_is_blocked(settings: Settings) -> None:
    assert settings.path_allowed(FIXTURES / "messy.csv")
    assert not settings.path_allowed(Path("/etc/passwd"))
    assert not settings.path_allowed(FIXTURES / ".." / ".." / ".." / "Windows")


def test_invalid_log_level_rejected() -> None:
    with pytest.raises(ValueError, match="must be one of"):
        Settings(log_level="LOUD")


# --------------------------------------------------------------------------
# digest


def test_round_sig_keeps_three_figures() -> None:
    assert round_sig(0.6939778779188857) == 0.694
    assert round_sig(1.0) == 1
    assert round_sig(None) is None
    assert round_sig(0) == 0


def test_compact_drops_absences_but_keeps_answers() -> None:
    out = compact({"a": 1, "b": None, "c": "", "d": [], "e": 0, "f": False})
    assert out == {"a": 1, "e": 0, "f": False}


def test_budget_truncates_and_says_so() -> None:
    response = Response("analyze_column")
    for i in range(200):
        response.add(
            Finding(Severity.LOW, f"finding {i} with a reasonably long message", column=f"c{i}")
        )
    payload = response.build()
    assert payload["truncated"]["omitted"] > 0
    assert estimate_tokens(payload) <= 600 * 1.15


def test_findings_rank_high_severity_first() -> None:
    response = Response("find_issues")
    response.add(Finding(Severity.LOW, "low"))
    response.add(Finding(Severity.HIGH, "high"))
    response.add(Finding(Severity.MEDIUM, "medium"))
    findings = response.build()["findings"]
    assert findings[0].startswith("HIGH")
    assert findings[-1].startswith("LOW")


def test_default_budget_applies_to_unlisted_tools() -> None:
    response = Response("something_new")
    for i in range(500):
        response.add(Finding(Severity.INFO, f"message {i} padded out to a realistic length"))
    assert estimate_tokens(response.build()) <= DEFAULT_BUDGET * 1.15


# --------------------------------------------------------------------------
# loaders


def test_encoding_and_delimiter_detection() -> None:
    path = FIXTURES / "messy_semicolon.csv"
    encoding = detect_encoding(path)
    assert encoding == "cp1252"
    assert detect_delimiter(path, encoding) == ";"


def test_coercion_recovers_numbers_and_dates(settings: Settings) -> None:
    df, report = load_file(str(FIXTURES / "messy.csv"), settings)
    assert "revenue" in report.coerced_columns
    assert "signup_date" in report.coerced_columns
    assert pd.api.types.is_numeric_dtype(df["revenue"])
    assert pd.api.types.is_datetime64_any_dtype(df["signup_date"])


def test_disguised_nulls_become_na(settings: Settings) -> None:
    df, _ = load_file(str(FIXTURES / "messy.csv"), settings)
    assert df["revenue"].isna().any()  # "N/A" and "-" were converted


def test_load_outside_allowed_paths_is_refused() -> None:
    narrow = load_settings(allowed_paths=[FIXTURES / "nonexistent"], log_level="WARNING")
    with pytest.raises(PathNotAllowedError):
        load_file(str(FIXTURES / "messy.csv"), narrow)


def test_limit_is_reported(settings: Settings) -> None:
    df, report = load_file(str(FIXTURES / "messy.csv"), settings, limit=100)
    assert len(df) == 100
    assert any("limit" in note for note in report.notes)


# --------------------------------------------------------------------------
# profiling


def test_column_classification() -> None:
    df = pd.DataFrame(
        {
            "ident": [f"ID{i}" for i in range(100)],
            "row_no": range(100),  # consecutive: a surrogate key
            "num": [round(3.5 + (i % 17) * 1.3, 2) for i in range(100)],
            "cat": ["a", "b"] * 50,
            "const": ["x"] * 100,
            "empty": [None] * 100,
            "when": pd.date_range("2024-01-01", periods=100),
            "flag": [True, False] * 50,
        }
    )
    kinds = column_kinds(df)
    assert kinds["ident"] is ColumnKind.IDENTIFIER
    assert kinds["row_no"] is ColumnKind.IDENTIFIER
    assert kinds["num"] is ColumnKind.NUMERIC
    assert kinds["cat"] is ColumnKind.CATEGORICAL
    assert kinds["const"] is ColumnKind.CONSTANT
    assert kinds["empty"] is ColumnKind.EMPTY
    assert kinds["when"] is ColumnKind.DATETIME
    assert kinds["flag"] is ColumnKind.BOOLEAN


def test_unique_numeric_is_not_an_identifier() -> None:
    """Uniqueness alone must not demote a measurement to a key.

    Prices and timestamps are frequently unique; only a consecutive run is
    evidence of a surrogate key.
    """
    unique_prices = pd.Series([100.0 + i * 0.37 for i in range(200)])
    assert classify(unique_prices) is ColumnKind.NUMERIC

    consecutive = pd.Series(range(1000, 1200))
    assert classify(consecutive) is ColumnKind.IDENTIFIER


def test_empty_column_classified_before_dtype() -> None:
    assert classify(pd.Series([None, None], dtype="float64")) is ColumnKind.EMPTY


def test_orientation_flags_real_defects(settings: Settings) -> None:
    df, _ = load_file(str(FIXTURES / "messy.csv"), settings)
    _, findings = orientation(df, column_kinds(df))
    rendered = " | ".join(f.render() for f in findings)
    assert "notes" in rendered  # entirely empty
    assert "region_code" in rendered  # constant
    assert "duplicate" in rendered.lower()  # 150 duplicated rows
    assert any(f.severity is Severity.HIGH for f in findings)


# --------------------------------------------------------------------------
# the invariant that distinguishes this server


def test_statistics_do_not_depend_on_row_order(settings: Settings) -> None:
    """A sorted file must profile identically to an unsorted one.

    Tools that read only the first N rows fail this by a wide margin; on this
    fixture the first-100-rows mean is ~92% below the true mean.
    """
    unsorted, _ = load_file(str(FIXTURES / "messy.csv"), settings)
    ordered, _ = load_file(str(FIXTURES / "messy_sorted.csv"), settings)

    assert unsorted.shape == ordered.shape
    assert round(unsorted["price"].mean(), 6) == round(ordered["price"].mean(), 6)
    assert unsorted["age"].isna().sum() == ordered["age"].isna().sum()
    assert column_kinds(unsorted) == column_kinds(ordered)

    # and confirm the trap the invariant guards against is real
    head_mean = ordered["price"].head(100).mean()
    assert abs(head_mean - ordered["price"].mean()) / ordered["price"].mean() > 0.5


def test_source_file_is_untouched_by_a_session(settings: Settings) -> None:
    path = FIXTURES / "messy.csv"
    before = hashlib.sha256(path.read_bytes()).hexdigest()

    server = build_server(settings)
    call(server, "load_dataset", {"source": str(path)})
    call(server, "manage_sources", {"action": "list"})

    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


# --------------------------------------------------------------------------
# registry


def test_registry_lifecycle() -> None:
    registry = Registry()
    df = pd.DataFrame({"a": [1, 2, 3]})
    registry.add_dataset("sales", df, origin="sales.csv")

    assert registry.get_dataset("sales").shape == (3, 1)
    assert registry.close("sales") == "dataset"
    with pytest.raises(SourceNotFoundError):
        registry.get_dataset("sales")


def test_unique_alias_avoids_collisions() -> None:
    registry = Registry()
    df = pd.DataFrame({"a": [1]})
    registry.add_dataset("data", df, origin="x")
    assert registry.unique_alias("data") == "data_2"
    registry.add_dataset("data_2", df, origin="x")
    assert registry.unique_alias("data") == "data_3"


def test_undo_restores_previous_frame() -> None:
    registry = Registry()
    dataset = registry.add_dataset("d", pd.DataFrame({"a": [1, 2, 3]}), origin="x")
    dataset.snapshot(max_total_mb=100)
    dataset.df = dataset.df.head(1)
    assert dataset.shape == (1, 1)
    assert dataset.restore() == 1
    assert dataset.shape == (3, 1)


def test_snapshots_disabled_when_budget_is_zero() -> None:
    registry = Registry()
    dataset = registry.add_dataset("d", pd.DataFrame({"a": [1]}), origin="x")
    dataset.snapshot(max_total_mb=0)
    assert dataset.snapshots == []


# --------------------------------------------------------------------------
# server


def test_default_alias_from_path() -> None:
    assert default_alias("data/My Sales-2024.csv") == "my_sales_2024"
    assert default_alias("/tmp/.csv") == "dataset"


def test_tools_registered_with_annotations(settings: Settings) -> None:
    server = build_server(settings)
    tools = {t.name: t for t in asyncio.run(server.list_tools())}
    # (readOnly, openWorld) per tool, as tabled in spec section 7.7
    expected = {
        "load_dataset": (True, True),
        "manage_sources": (False, None),
        "profile": (True, True),
        "find_issues": (True, False),
        "analyze_column": (True, False),
        "check_relationships": (True, False),
        "analyze_target": (True, False),
    }
    assert set(tools) == set(expected)
    for name, (read_only, open_world) in expected.items():
        hints = tools[name].annotations
        assert hints.read_only_hint is read_only, name
        if open_world is not None:
            assert hints.open_world_hint is open_world, name


def test_load_dataset_returns_findings_not_just_shape(settings: Settings) -> None:
    server = build_server(settings)
    payload = call(server, "load_dataset", {"source": str(FIXTURES / "messy.csv")})
    assert payload["shape"] == [5150, 10]
    assert payload["findings"]  # the point: no follow-up profile needed
    assert payload["summary"]
    assert payload["read"]["encoding"] == "utf-8"


def test_load_dataset_stays_within_budget(settings: Settings) -> None:
    server = build_server(settings)
    payload = call(server, "load_dataset", {"source": str(FIXTURES / "messy.csv")})
    assert estimate_tokens(payload) <= 800 * 1.15


def test_missing_file_returns_error_envelope(settings: Settings) -> None:
    server = build_server(settings)
    payload = call(server, "load_dataset", {"source": str(FIXTURES / "nope.csv")})
    assert payload["error"]["code"] == "SOURCE_NOT_FOUND"
    assert payload["error"]["retryable"] is False


def test_unsupported_format_returns_error_envelope(settings: Settings) -> None:
    server = build_server(settings)
    payload = call(server, "load_dataset", {"source": str(FIXTURES / "make_fixtures.py")})
    assert payload["error"]["code"] == "UNSUPPORTED_FORMAT"


def test_row_limit_is_enforced() -> None:
    tight = load_settings(
        max_load_rows=10, log_level="WARNING", allowed_paths=[FIXTURES.parents[1]]
    )
    server = build_server(tight)
    payload = call(server, "load_dataset", {"source": str(FIXTURES / "messy.csv")})
    assert payload["error"]["code"] == "SOURCE_TOO_LARGE"


def test_manage_sources_lists_and_closes(settings: Settings) -> None:
    server = build_server(settings)
    call(server, "load_dataset", {"source": str(FIXTURES / "messy.csv"), "alias": "m"})

    listed = call(server, "manage_sources", {"action": "list"})
    assert listed["datasets"][0]["alias"] == "m"

    assert call(server, "manage_sources", {"action": "close", "alias": "m"})["kind"] == "dataset"
    assert "summary" in call(server, "manage_sources", {"action": "list"})


def test_manage_sources_rejects_unknown_action(settings: Settings) -> None:
    server = build_server(settings)
    payload = call(server, "manage_sources", {"action": "destroy"})
    assert payload["error"]["code"] == "INVALID_OPERATION"


def test_responses_never_contain_credentials(settings: Settings) -> None:
    server = build_server(settings)
    payloads = [
        call(server, "load_dataset", {"source": str(FIXTURES / "messy.csv")}),
        call(server, "manage_sources", {"action": "list"}),
        call(server, "load_dataset", {"source": "postgresql://u:secret@h/db"}),
    ]
    blob = json.dumps(payloads)
    assert "secret" not in blob
    assert "postgresql://" not in blob
