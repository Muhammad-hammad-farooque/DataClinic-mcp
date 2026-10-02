"""Phase 2 tests: per-column statistics.

As in Phase 1, the invariants carry the weight: statistics must match pandas
on the full frame and must not depend on row order.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import settings as hyp_settings
from hypothesis import strategies as st

from eda_mcp.config import Settings, load_settings
from eda_mcp.digest import compact, estimate_tokens
from eda_mcp.errors import ColumnNotFoundError
from eda_mcp.issues import column_findings
from eda_mcp.loaders import load_file
from eda_mcp.profiling import (
    ColumnKind,
    column_kinds,
    datetime_stats,
    numeric_stats,
    profile_column,
    profile_frame,
    spelling_variants,
    string_checks,
)
from eda_mcp.server import build_server

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
NOW = pd.Timestamp("2026-10-01")


@pytest.fixture(scope="session")
def settings() -> Settings:
    return load_settings(log_level="WARNING", allowed_paths=[FIXTURES.parents[1]])


@pytest.fixture(scope="session")
def messy(settings: Settings) -> pd.DataFrame:
    df, _ = load_file(str(FIXTURES / "messy.csv"), settings)
    return df


@pytest.fixture(scope="session")
def profiles(messy: pd.DataFrame) -> dict[str, dict]:  # type: ignore[type-arg]
    return {p.name: p.to_dict() for p in profile_frame(messy, column_kinds(messy), now=NOW)}


# --------------------------------------------------------------------------
# numeric


def test_numeric_stats_match_pandas_on_the_full_frame(messy: pd.DataFrame, profiles) -> None:  # type: ignore[no-untyped-def]
    price = messy["price"]
    stats = profiles["price"]
    assert stats["mean"] == pytest.approx(price.mean())
    assert stats["median"] == pytest.approx(price.median())
    assert stats["std"] == pytest.approx(price.std())
    assert stats["skew"] == pytest.approx(price.skew())
    assert stats["max"] == pytest.approx(price.max())


def test_skewed_price_has_outliers(profiles) -> None:  # type: ignore[no-untyped-def]
    price = profiles["price"]
    assert price["skew"] > 1  # lognormal: right-skewed
    assert price["outliers_iqr"] > 0
    assert price["outliers_modified_z"] > 0
    low, high = price["iqr_bounds"]
    assert low < price["median"] < high


def test_impossible_negative_ages_are_counted(messy: pd.DataFrame, profiles) -> None:  # type: ignore[no-untyped-def]
    assert profiles["age"]["negatives"] == int((messy["age"] < 0).sum()) > 0
    assert profiles["age"]["integral"] is True


def test_zero_spread_reports_no_outliers_rather_than_all() -> None:
    stats = numeric_stats(pd.Series([5.0] * 80 + [1.0, 9.0, 100.0] * 5))
    assert "outliers_iqr" not in stats
    assert "outliers_modified_z" not in stats


# --------------------------------------------------------------------------
# categorical and text


def test_country_spelling_variants_are_grouped(profiles) -> None:  # type: ignore[no-untyped-def]
    groups = [set(g) for g in profiles["country"]["variants"]]
    assert {"USA", "usa", "U.S.A."} in groups
    assert {"UK", "uk"} in groups
    assert "France" not in {v for g in groups for v in g}
    assert profiles["country"]["variant_rows"] > 0


def test_variants_respect_unicode_letters() -> None:
    """Accented letters are letters, not punctuation to strip."""
    out = spelling_variants(pd.Series(["España", "espana", "ESPAÑA"]).value_counts())
    assert out["variants"] == [["ESPAÑA", "España"]]  # tied counts break by value


def test_string_checks_find_hidden_defects() -> None:
    s = pd.Series([" padded", "ok", "ok", "1,200", "3.5", 7, " padded"], dtype=object)
    stats = string_checks(s.value_counts())
    # counted per row, though each check ran once per distinct value
    assert stats["whitespace_padded"] == 2
    assert stats["numeric_like"] == 3
    assert stats["mixed_types"] == {"int": 1, "str": 6}
    assert stats["length"]["median"] == s.astype(str).str.len().median()


def test_top_values_are_ranked_and_capped(profiles) -> None:  # type: ignore[no-untyped-def]
    top = profiles["country"]["top"]
    assert len(top) <= 5
    assert list(top.values()) == sorted(top.values(), reverse=True)


# --------------------------------------------------------------------------
# other kinds


def test_datetime_stats() -> None:
    s = pd.Series(pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-10", "2030-01-01"]))
    stats = datetime_stats(s, now=NOW)
    assert stats["min"].startswith("2024-01-01")
    assert stats["future"] == 1
    assert stats["sorted"] is True
    assert stats["median_gap_days"] == 8
    assert stats["max_gap_days"] > 2000


def test_datetime_stats_handle_timezones() -> None:
    s = pd.Series(pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC"))
    assert datetime_stats(s, now=NOW)["future"] == 0


def test_duplicate_rows_do_not_hide_an_identifier(messy: pd.DataFrame) -> None:
    """150 copied rows drop customer_id to 97% unique; it is still a key."""
    assert messy["customer_id"].nunique() / len(messy) < 0.99
    assert column_kinds(messy)["customer_id"] is ColumnKind.IDENTIFIER


def test_identifier_and_constant_columns(profiles) -> None:  # type: ignore[no-untyped-def]
    # 150 duplicated rows repeat 150 customer ids
    assert profiles["customer_id"]["kind"] == "identifier"
    assert profiles["customer_id"]["duplicate_keys"] == 150
    assert profiles["region_code"]["value"] == "EMEA"
    assert profiles["notes"]["missing_pct"] == 100


# --------------------------------------------------------------------------
# robustness


def test_unknown_column_is_rejected(messy: pd.DataFrame) -> None:
    with pytest.raises(ColumnNotFoundError):
        profile_frame(messy, columns=["price", "nope"])


def test_column_subset_keeps_frame_order(messy: pd.DataFrame) -> None:
    names = [p.name for p in profile_frame(messy, columns=["age", "price"])]
    assert names == ["price", "age"]


def test_one_failing_column_does_not_sink_the_rest() -> None:
    df = pd.DataFrame({"good": [1.0, 2.0, 3.0], "bad": ["a", "b", "c"]})
    # Forcing numeric statistics onto text makes that one column fail.
    profiles = profile_frame(df, {"bad": ColumnKind.NUMERIC})
    by_name = {p.name: p for p in profiles}
    assert by_name["bad"].error and "failed" in by_name["bad"].to_dict()
    assert by_name["good"].error is None and by_name["good"].stats["mean"] == 2


def test_non_string_column_labels() -> None:
    df = pd.DataFrame({0: [1.5, 2.5, 3.5], 1: ["x", "y", "x"]})
    assert [p.name for p in profile_frame(df, columns=["0"])] == ["0"]


# --------------------------------------------------------------------------
# invariants


def test_profile_does_not_depend_on_row_order(settings: Settings, messy: pd.DataFrame) -> None:
    """The sorted fixture must profile identically, to reported precision."""
    ordered, _ = load_file(str(FIXTURES / "messy_sorted.csv"), settings)

    def digest(df: pd.DataFrame) -> dict:  # type: ignore[type-arg]
        out = {}
        for p in profile_frame(df, column_kinds(df), now=NOW):
            d = p.to_dict()
            d.pop("sorted", None)  # genuinely a property of the row order
            out[p.name] = compact(d)
        return out

    assert digest(messy) == digest(ordered)


finite = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)


@given(st.lists(finite, min_size=4, max_size=200))
@hyp_settings(max_examples=75, deadline=None)
def test_numeric_stats_match_pandas_and_ignore_order(values: list[float]) -> None:
    s = pd.Series(values)
    stats = profile_column(s, ColumnKind.NUMERIC).stats
    assert stats["mean"] == pytest.approx(s.mean(), rel=1e-9, abs=1e-6)
    # quantile(0.5) and median() may round the midpoint differently in the last bit
    assert stats["median"] == pytest.approx(s.median(), rel=1e-12, abs=1e-12)
    assert stats["min"] == s.min() and stats["max"] == s.max()

    shuffled = profile_column(s.sample(frac=1, random_state=0), ColumnKind.NUMERIC).stats
    assert compact(shuffled) == compact(stats)


def test_profiles_are_json_serialisable(profiles) -> None:  # type: ignore[no-untyped-def]
    import json

    def leaves(obj):  # type: ignore[no-untyped-def]
        if isinstance(obj, dict):
            for v in obj.values():
                yield from leaves(v)
        elif isinstance(obj, list):
            for v in obj:
                yield from leaves(v)
        else:
            yield obj

    # strict: no default=str fallback, so a Timestamp would fail here
    json.dumps(compact(profiles))
    assert not [v for v in leaves(profiles) if isinstance(v, np.generic)]


# --------------------------------------------------------------------------
# issues


def test_binary_flag_is_not_called_skewed() -> None:
    flag = profile_column(pd.Series([0] * 95 + [1] * 5, name="f"), ColumnKind.NUMERIC)
    assert not column_findings(flag)


def test_signed_column_negatives_are_not_flagged() -> None:
    signed = profile_column(pd.Series(np.linspace(-1, 1, 101), name="s"), ColumnKind.NUMERIC)
    assert not any("negative" in f.message for f in column_findings(signed))


def test_stray_negatives_and_skew_are_flagged(messy: pd.DataFrame) -> None:
    by_name = {p.name: column_findings(p) for p in profile_frame(messy, column_kinds(messy))}
    age = [f.render() for f in by_name["age"]]
    price = [f.render() for f in by_name["price"]]
    assert any("52 negative" in line for line in age)
    assert any("right-skewed" in line and "log1p" in line for line in price)


def test_numbers_stored_as_text_are_flagged() -> None:
    text = pd.Series(["1", "2", "3.5", "4,000"] * 30 + ["n/a"], name="t")
    found = column_findings(profile_column(text, ColumnKind.TEXT))
    assert [f.message for f in found] == ["numbers stored as text"]
    assert found[0].affected_rows == 1


def test_future_dates_are_flagged() -> None:
    dates = pd.Series(pd.to_datetime(["2020-01-01", "2021-01-01", "2030-01-01"]), name="d")
    found = column_findings(profile_column(dates, ColumnKind.DATETIME, now=NOW))
    assert found[0].affected_rows == 1 and "future" in found[0].message


def test_every_finding_has_one_recommendation(messy: pd.DataFrame) -> None:
    findings = [f for p in profile_frame(messy, column_kinds(messy)) for f in column_findings(p)]
    assert findings and all(f.recommendation for f in findings)


# --------------------------------------------------------------------------
# profile tool


def call(server, tool, args):  # type: ignore[no-untyped-def]
    result = asyncio.run(server.call_tool(tool, args))
    payload = result[1] if isinstance(result, tuple) else result
    return payload.structured_content


@pytest.fixture()
def server(settings: Settings):  # type: ignore[no-untyped-def]
    srv = build_server(settings)
    call(srv, "load_dataset", {"source": str(FIXTURES / "messy.csv"), "alias": "m"})
    return srv


def test_profile_is_registered_read_only(server) -> None:  # type: ignore[no-untyped-def]
    tools = {t.name: t for t in asyncio.run(server.list_tools())}
    assert tools["profile"].annotations.read_only_hint is True


def test_profile_shows_problem_columns_and_rolls_up_clean(server) -> None:  # type: ignore[no-untyped-def]
    payload = call(server, "profile", {"source": "m"})
    shown = set(payload["columns"])
    assert {"price", "age", "country", "notes", "customer_id"} <= shown
    assert "revenue" not in shown and "revenue" in payload["clean"]
    # worst first: the HIGH columns lead
    assert list(payload["columns"])[:2] == ["customer_id", "notes"]
    assert payload["duplicate_rows"] == 150
    assert any("exact duplicates" in f for f in payload["findings"])


def test_profile_stays_within_budget(server) -> None:  # type: ignore[no-untyped-def]
    for detail in ("brief", "standard", "full"):
        payload = call(server, "profile", {"source": "m", "detail": detail})
        assert estimate_tokens(payload) <= 1500 * 1.15, detail


def test_profile_detail_levels(server) -> None:  # type: ignore[no-untyped-def]
    brief = call(server, "profile", {"source": "m", "detail": "brief"})
    assert "columns" not in brief and brief["findings"]
    full = call(server, "profile", {"source": "m", "detail": "full"})
    assert len(full["columns"]) == 10 and "more_columns" not in full
    assert "clean" not in full


def test_profile_column_subset(server) -> None:  # type: ignore[no-untyped-def]
    payload = call(server, "profile", {"source": "m", "columns": ["revenue", "age"]})
    assert set(payload["columns"]) == {"revenue", "age"}  # clean ones too: they were asked for
    assert "duplicate_rows" not in payload
    assert all(f.split()[1].rstrip(":") in {"revenue", "age"} for f in payload["findings"])


def test_profile_wide_table_is_cut_to_budget(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    wide = pd.DataFrame({f"c{i}": rng.lognormal(0, 1.5, 300) for i in range(120)})
    wide.to_csv(tmp_path / "wide.csv", index=False)
    srv = build_server(load_settings(log_level="WARNING", allowed_paths=[tmp_path]))
    call(srv, "load_dataset", {"source": str(tmp_path / "wide.csv"), "alias": "w"})

    payload = call(srv, "profile", {"source": "w"})
    assert estimate_tokens(payload) <= 1500 * 1.15
    assert "+" in payload["more_columns"]["omitted"]  # names capped, count stated
    assert "columns=" in payload["more_columns"]["remedy"]


def test_profile_identical_on_sorted_copy(settings: Settings) -> None:
    srv = build_server(settings)
    call(srv, "load_dataset", {"source": str(FIXTURES / "messy.csv"), "alias": "a"})
    call(srv, "load_dataset", {"source": str(FIXTURES / "messy_sorted.csv"), "alias": "b"})
    a = call(srv, "profile", {"source": "a"})
    b = call(srv, "profile", {"source": "b"})
    a.pop("dataset"), b.pop("dataset")
    assert a == b


@pytest.mark.parametrize(
    ("args", "code"),
    [
        ({"source": "m", "detail": "verbose"}, "INVALID_OPERATION"),
        ({"source": "nope"}, "SOURCE_NOT_FOUND"),
        ({"source": "m", "columns": ["nope"]}, "COLUMN_NOT_FOUND"),
    ],
)
def test_profile_errors_are_envelopes(server, args, code) -> None:  # type: ignore[no-untyped-def]
    assert call(server, "profile", args)["error"]["code"] == code
