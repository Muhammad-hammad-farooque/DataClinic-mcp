"""Phase 3 tests: database connections and schema exploration.

SQLite runs here, with no server to install. PostgreSQL is exercised by the
integration job against a real server (``-m integration``).
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eda_mcp.config import Settings, load_settings
from eda_mcp.db.connect import env_name, open_engine, read_frame, resolve_dsn
from eda_mcp.db.guard import check
from eda_mcp.digest import estimate_tokens
from eda_mcp.errors import EDAError, ErrorCode
from eda_mcp.server import build_server


def call(server, tool, args):  # type: ignore[no-untyped-def]
    result = asyncio.run(server.call_tool(tool, args))
    payload = result[1] if isinstance(result, tuple) else result
    return payload.structured_content


def make_shop(path: Path) -> Path:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT NOT NULL, country TEXT);
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            customer_id INTEGER REFERENCES customers(id),
            amount REAL DEFAULT 0,
            placed TEXT
        );
        CREATE INDEX ix_orders_customer ON orders(customer_id);
        CREATE VIEW big_orders AS SELECT * FROM orders WHERE amount > 100;
        """
    )
    connection.executemany(
        "INSERT INTO customers VALUES (?, ?, ?)",
        [(i, f"c{i}", ["UK", "FR", "US"][i % 3]) for i in range(50)],
    )
    connection.executemany(
        "INSERT INTO orders VALUES (?, ?, ?, ?)",
        [(i, i % 50, i * 1.5, "2025-01-01") for i in range(500)],
    )
    connection.commit()
    connection.close()
    return path


@pytest.fixture(scope="module")
def shop(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return make_shop(tmp_path_factory.mktemp("db") / "shop.db")


@pytest.fixture()
def settings(shop: Path) -> Settings:
    return load_settings(log_level="WARNING", allowed_paths=[shop.parent])


def dsn_for(path: Path) -> str:
    return "sqlite:///" + path.as_posix()


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    # A developer's own DATABASE_URL must not leak into these tests.
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("EDA_MCP_DSN_SHOP", raising=False)


# --------------------------------------------------------------------------
# credentials


def test_env_name_is_derived_from_the_alias() -> None:
    assert env_name("warehouse") == "EDA_MCP_DSN_WAREHOUSE"
    assert env_name("my-db.2") == "EDA_MCP_DSN_MY_DB_2"


def test_credential_resolution_order(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text('[connections]\nshop = "sqlite:///from-config.db"\n')
    if sys.platform != "win32":
        config.chmod(0o600)

    # 5: the explicit argument, last, with a warning
    assert resolve_dsn("shop", dsn="sqlite:///arg.db", config_path=tmp_path / "none.toml").warning
    # 4: the config file
    found = resolve_dsn("shop", config_path=config)
    assert found.dsn.endswith("from-config.db") and "config.toml" in found.source
    # 3: DATABASE_URL
    monkeypatch.setenv("DATABASE_URL", "sqlite:///from-database-url.db")
    assert resolve_dsn("shop", config_path=config).source.endswith("DATABASE_URL")
    # 2: the per-alias variable
    monkeypatch.setenv("EDA_MCP_DSN_SHOP", "sqlite:///from-alias.db")
    found = resolve_dsn("shop", dsn="sqlite:///arg.db", config_path=config)
    assert found.dsn.endswith("from-alias.db")
    assert found.warning and "takes precedence" in found.warning
    # 1: a variable named explicitly
    monkeypatch.setenv("MY_DSN", "sqlite:///named.db")
    assert resolve_dsn("shop", env_var="MY_DSN", config_path=config).dsn.endswith("named.db")


def test_missing_credentials_name_the_variable_to_set(tmp_path: Path) -> None:
    with pytest.raises(EDAError) as caught:
        resolve_dsn("shop", config_path=tmp_path / "none.toml")
    assert "EDA_MCP_DSN_SHOP" in caught.value.remedy
    with pytest.raises(EDAError):
        resolve_dsn("shop", env_var="NOT_SET_ANYWHERE")


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission bits")
def test_world_readable_config_is_refused(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text('[connections]\nshop = "sqlite:///x.db"\n')
    config.chmod(0o644)
    with pytest.raises(EDAError) as caught:
        resolve_dsn("shop", config_path=config)
    assert caught.value.code is ErrorCode.OPERATION_REFUSED


# --------------------------------------------------------------------------
# engines and the read-only layers


@pytest.mark.parametrize(
    ("dsn", "code"),
    [
        ("sqlite:///:memory:", ErrorCode.INVALID_OPERATION),
        ("mysql://u@h/db", ErrorCode.INVALID_OPERATION),
        ("oracle://u@h/db", ErrorCode.INVALID_OPERATION),
        ("not a url", ErrorCode.INVALID_OPERATION),
    ],
)
def test_unusable_dsns_are_refused(dsn: str, code: ErrorCode, settings: Settings) -> None:
    with pytest.raises(EDAError) as caught:
        open_engine(dsn, settings)
    assert caught.value.code is code


def test_sqlite_paths_are_confined(settings: Settings, tmp_path: Path) -> None:
    outside = make_shop(tmp_path / "elsewhere.db")
    with pytest.raises(EDAError) as caught:
        open_engine(dsn_for(outside), settings)
    assert caught.value.code is ErrorCode.PATH_NOT_ALLOWED


def test_missing_sqlite_file_is_not_created(settings: Settings, shop: Path) -> None:
    missing = shop.parent / "missing.db"
    with pytest.raises(EDAError) as caught:
        open_engine(dsn_for(missing), settings)
    assert caught.value.code is ErrorCode.SOURCE_NOT_FOUND
    assert not missing.exists()


def test_write_connections_are_not_available(settings: Settings, shop: Path) -> None:
    with pytest.raises(EDAError) as caught:
        open_engine(dsn_for(shop), settings, read_only=False)
    assert caught.value.code is ErrorCode.WRITE_NOT_PERMITTED


def test_session_layer_alone_refuses_writes(settings: Settings, shop: Path) -> None:
    """Layer 1 must hold even if the statement guard were bypassed entirely."""
    opened = open_engine(dsn_for(shop), settings)
    try:
        for statement in ("DELETE FROM orders", "DROP TABLE orders", "CREATE TABLE x (a INT)"):
            with pytest.raises(Exception, match="readonly"), opened.engine.connect() as c:
                c.exec_driver_sql(statement)
                c.commit()
    finally:
        opened.engine.dispose()
    assert sqlite3.connect(shop).execute("SELECT COUNT(*) FROM orders").fetchone() == (500,)


def test_read_frame_caps_rows(settings: Settings, shop: Path) -> None:
    opened = open_engine(dsn_for(shop), settings)
    try:
        sql = check("SELECT * FROM orders ORDER BY id", "sqlite", row_cap=11)
        frame, truncated = read_frame(opened.engine, "sqlite", sql, settings, row_cap=10)
        assert len(frame) == 10 and truncated
        assert list(frame.columns) == ["id", "customer_id", "amount", "placed"]
    finally:
        opened.engine.dispose()


def test_sqlite_statement_timeout_interrupts(shop: Path) -> None:
    fast = load_settings(log_level="WARNING", allowed_paths=[shop.parent], statement_timeout=1)
    opened = open_engine(dsn_for(shop), fast)
    endless = (
        "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT COUNT(*) FROM c"
    )
    try:
        with pytest.raises(EDAError) as caught:
            read_frame(opened.engine, "sqlite", check(endless, "sqlite"), fast, row_cap=1)
        assert caught.value.code is ErrorCode.QUERY_TIMEOUT
        assert caught.value.to_dict()["error"]["retryable"] is True
    finally:
        opened.engine.dispose()


# --------------------------------------------------------------------------
# tools


@pytest.fixture()
def server(settings: Settings):  # type: ignore[no-untyped-def]
    return build_server(settings)


def test_connect_via_environment_has_no_warning(
    server, shop: Path, monkeypatch: pytest.MonkeyPatch
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("EDA_MCP_DSN_SHOP", dsn_for(shop))
    payload = call(server, "connect_database", {"alias": "shop"})
    assert payload["dialect"] == "sqlite" and payload["read_only"] is True
    assert payload["credentials"] == "environment variable EDA_MCP_DSN_SHOP"
    assert payload["schemas"] == {"main": {"tables": 2, "views": 1}}
    assert "findings" not in payload


def test_connect_with_dsn_warns_and_alias_cannot_be_reused(server, shop: Path) -> None:  # type: ignore[no-untyped-def]
    payload = call(server, "connect_database", {"alias": "shop", "dsn": dsn_for(shop)})
    assert payload["findings"][0].startswith("MED  the DSN now resides in the conversation")
    again = call(server, "connect_database", {"alias": "shop", "dsn": dsn_for(shop)})
    assert again["error"]["code"] == "INVALID_OPERATION"


def test_explore_schema_levels(server, shop: Path) -> None:  # type: ignore[no-untyped-def]
    call(server, "connect_database", {"alias": "shop", "dsn": dsn_for(shop)})

    # one schema: tables come back straight away
    listing = call(server, "explore_schema", {"connection": "shop"})
    assert listing["schema"] == "main"
    assert listing["tables"]["orders"] == {"kind": "table", "columns": 4}
    assert listing["tables"]["big_orders"]["kind"] == "view"

    table = call(server, "explore_schema", {"connection": "shop", "table": "orders"})
    assert table["columns"][0] == "id INTEGER pk"
    assert table["foreign_keys"] == [{"columns": ["customer_id"], "references": "customers(id)"}]
    assert table["indexes"][0]["columns"] == ["customer_id"]
    assert table["rows"] == 500 and table["row_count_exact"] is True


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ({"connection": "shop", "table": "nope"}, "no table named 'nope'"),
        ({"connection": "shop", "schema": "nope"}, "no schema named 'nope'"),
        ({"connection": "nope"}, "no source named 'nope'"),
    ],
)
def test_explore_schema_errors(server, shop: Path, args, message) -> None:  # type: ignore[no-untyped-def]
    call(server, "connect_database", {"alias": "shop", "dsn": dsn_for(shop)})
    payload = call(server, "explore_schema", args)
    assert payload["error"]["code"] == "SOURCE_NOT_FOUND"
    assert payload["error"]["message"] == message


def test_explore_schema_stays_within_budget(tmp_path: Path) -> None:
    path = tmp_path / "wide.db"
    connection = sqlite3.connect(path)
    for i in range(300):
        connection.execute(f"CREATE TABLE t{i:03d} (a INT, b TEXT, c REAL)")
    connection.execute("CREATE TABLE wide (" + ", ".join(f"col{i} INT" for i in range(400)) + ")")
    connection.commit()
    connection.close()

    srv = build_server(load_settings(log_level="WARNING", allowed_paths=[tmp_path]))
    call(srv, "connect_database", {"alias": "w", "dsn": dsn_for(path)})
    listing = call(srv, "explore_schema", {"connection": "w"})
    assert estimate_tokens(listing) <= 1000 * 1.15
    assert "+" in listing["more_tables"]["omitted"]
    wide = call(srv, "explore_schema", {"connection": "w", "table": "wide"})
    assert estimate_tokens(wide) <= 1000 * 1.15
    assert wide["more_columns"] > 0


def test_manage_sources_lists_and_closes_connections(server, shop: Path) -> None:  # type: ignore[no-untyped-def]
    call(server, "connect_database", {"alias": "shop", "dsn": dsn_for(shop)})
    listed = call(server, "manage_sources", {"action": "list"})
    assert listed["connections"] == [
        {"alias": "shop", "dialect": "sqlite", "database": "shop.db", "read_only": True}
    ]
    assert (
        call(server, "manage_sources", {"action": "close", "alias": "shop"})["kind"] == "connection"
    )
    gone = call(server, "explore_schema", {"connection": "shop"})
    assert gone["error"]["code"] == "SOURCE_NOT_FOUND"


def test_passwords_never_reach_responses_or_logs(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    srv = build_server(load_settings(log_level="DEBUG", allowed_paths=[tmp_path]))
    payloads = [
        call(
            srv,
            "connect_database",
            {"alias": "pg", "dsn": "postgresql://alice:hunter2@127.0.0.1:1/db"},
        ),
        call(srv, "connect_database", {"alias": "my", "dsn": "mysql://alice:hunter2@127.0.0.1/db"}),
        call(
            srv,
            "connect_database",
            {"alias": "bad", "dsn": "postgresql://alice:hunter2@@::nonsense"},
        ),
    ]
    # psycopg absent: DEPENDENCY_MISSING; present: nothing listens on port 1
    assert payloads[0]["error"]["code"] in ("DEPENDENCY_MISSING", "CONNECTION_FAILED")
    captured = capfd.readouterr()
    for blob in (json.dumps(payloads), captured.out, captured.err):
        assert "hunter2" not in blob


# --------------------------------------------------------------------------
# loading tables and queries


@pytest.fixture()
def connected(server, shop: Path):  # type: ignore[no-untyped-def]
    call(server, "connect_database", {"alias": "shop", "dsn": dsn_for(shop)})
    return server


def test_load_table_then_analyse_it(connected) -> None:  # type: ignore[no-untyped-def]
    payload = call(connected, "load_dataset", {"source": "shop.customers"})
    assert payload["dataset"] == "customers" and payload["shape"] == [50, 3]
    assert payload["origin"] == "shop:main.customers"
    assert payload["read"] == {"format": "sqlite", "rows_available": 50}
    # once loaded, it is an ordinary dataset for every analysis tool
    counts = call(connected, "query", {"source": "customers", "expression": "count(by=country)"})
    assert counts["values"] == {"FR": 17, "UK": 17, "US": 16}


def test_schema_qualified_reference_and_types(connected) -> None:  # type: ignore[no-untyped-def]
    payload = call(connected, "load_dataset", {"source": "shop.main.orders", "alias": "o"})
    assert payload["shape"] == [500, 4]
    assert payload["read"]["coerced"] == {"placed": "datetime"}  # text dates parsed


def test_large_table_refused_without_limit(shop: Path) -> None:
    small = load_settings(log_level="WARNING", allowed_paths=[shop.parent], max_load_rows=400)
    srv = build_server(small)
    call(srv, "connect_database", {"alias": "shop", "dsn": dsn_for(shop)})

    refused = call(srv, "load_dataset", {"source": "shop.orders"})
    assert refused["error"]["code"] == "SOURCE_TOO_LARGE"
    assert "profile" in refused["error"]["remedy"]

    limited = call(srv, "load_dataset", {"source": "shop.orders", "limit": 100})
    assert limited["shape"][0] == 100 and limited["read"]["rows_available"] == 500
    assert any("not a random sample" in f for f in limited["findings"])

    query = call(srv, "load_dataset", {"source": "shop", "query": "SELECT * FROM orders"})
    assert query["error"]["code"] == "SOURCE_TOO_LARGE"


def test_load_query_result(connected) -> None:  # type: ignore[no-untyped-def]
    sql = "SELECT country, COUNT(*) AS n FROM customers GROUP BY country ORDER BY country"
    payload = call(connected, "load_dataset", {"source": "shop", "query": sql})
    assert payload["dataset"] == "shop_query" and payload["shape"] == [3, 2]
    assert payload["origin"] == "shop:query"


@pytest.mark.parametrize(
    ("args", "code"),
    [
        ({"source": "shop", "query": "DELETE FROM orders"}, "STATEMENT_REJECTED"),
        ({"source": "shop", "query": "SELECT 1; DROP TABLE orders"}, "STATEMENT_REJECTED"),
        ({"source": "shop", "query": "SELECT * FROM missing_table"}, "INVALID_OPERATION"),
        ({"source": "shop.orders", "options": {"sheet": 1}}, "INVALID_OPERATION"),
        ({"source": "shop.orders", "limit": 0}, "INVALID_OPERATION"),
        ({"source": "shop.nope"}, "SOURCE_NOT_FOUND"),
        ({"source": "shop.nope.orders"}, "SOURCE_NOT_FOUND"),
        ({"source": "elsewhere", "query": "SELECT 1"}, "SOURCE_NOT_FOUND"),
    ],
)
def test_load_from_database_errors(connected, args, code) -> None:  # type: ignore[no-untyped-def]
    assert call(connected, "load_dataset", args)["error"]["code"] == code


def test_loading_never_changes_the_database(connected, shop: Path) -> None:  # type: ignore[no-untyped-def]
    before = sqlite3.connect(shop).execute("SELECT COUNT(*), SUM(amount) FROM orders").fetchone()
    call(connected, "load_dataset", {"source": "shop.orders"})
    call(
        connected,
        "load_dataset",
        {"source": "shop", "query": "SELECT * FROM orders WHERE amount > 1"},
    )
    after = sqlite3.connect(shop).execute("SELECT COUNT(*), SUM(amount) FROM orders").fetchone()
    assert before == after


def test_existing_file_wins_over_a_connection(connected, tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    # "shop.orders" names a real file here, so it must not load the table.
    monkeypatch.chdir(tmp_path)
    (tmp_path / "shop.orders").write_text("not a table")
    payload = call(connected, "load_dataset", {"source": "shop.orders"})
    assert payload["error"]["code"] in ("UNSUPPORTED_FORMAT", "PATH_NOT_ALLOWED")


# --------------------------------------------------------------------------
# push-down profiling


@pytest.fixture(scope="module")
def sales(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, pd.DataFrame]:
    """30k rows, sorted by price: big enough to force sampling at sample_size=2000."""
    rng = np.random.default_rng(0)
    n = 30_000
    df = pd.DataFrame(
        {
            "price": np.round(rng.lognormal(3, 1, n), 2),
            "age": rng.integers(18, 90, n).astype(float),
            "country": rng.choice(["USA", "usa", "U.S.A.", "UK", "France"], n),
            "created": (
                pd.Timestamp("2020-01-01") + pd.to_timedelta(rng.integers(0, 1500, n), unit="D")
            ).strftime("%Y-%m-%d"),
            "region": "EMEA",
        }
    )
    df.loc[rng.random(n) < 0.02, "age"] = -999
    df.loc[rng.random(n) < 0.1, "price"] = np.nan
    df = df.sort_values("price", na_position="first").reset_index(drop=True)
    path = tmp_path_factory.mktemp("sales") / "sales.db"
    with sqlite3.connect(path) as connection:
        df.to_sql("sales", connection, index=False)
    return path, df


@pytest.fixture()
def pushdown(sales):  # type: ignore[no-untyped-def]
    path, _ = sales
    srv = build_server(
        load_settings(log_level="WARNING", allowed_paths=[path.parent], sample_size=2000)
    )
    call(srv, "connect_database", {"alias": "s", "dsn": dsn_for(path)})
    return srv


def test_pushdown_statistics_are_exact(sales) -> None:  # type: ignore[no-untyped-def]
    from eda_mcp.db.connect import open_engine
    from eda_mcp.db.sqlprofile import profile_table
    from eda_mcp.registry import Connection

    path, df = sales
    settings = load_settings(log_level="WARNING", allowed_paths=[path.parent], sample_size=2000)
    opened = open_engine(dsn_for(path), settings)
    try:
        measured = profile_table(Connection("s", "sqlite", opened.engine), None, "sales", settings)
    finally:
        opened.engine.dispose()
    assert measured.rows == len(df) and measured.sampled is not None
    columns = {p.name: p for p in measured.profiles}

    price, truth = columns["price"], df["price"]
    assert (price.count, price.missing, price.unique) == (
        truth.count(),
        truth.isna().sum(),
        truth.nunique(),
    )
    for stat, expected in (
        ("mean", truth.mean()),
        ("std", truth.std()),
        ("min", truth.min()),
        ("max", truth.max()),
    ):
        assert price.stats[stat] == pytest.approx(expected, rel=1e-9), stat
    low, high = price.stats["iqr_bounds"]
    assert price.stats["outliers_iqr"] == int(((truth < low) | (truth > high)).sum())

    age = columns["age"]
    assert age.stats["negatives"] == int((df["age"] < 0).sum())
    assert age.stats["sentinels"] == {-999: int((df["age"] == -999).sum())}

    country = columns["country"]
    expected_top = df["country"].value_counts()
    assert country.stats["top"] == {str(k): int(v) for k, v in expected_top.items()}

    created = columns["created"]
    assert created.stats["min"].startswith(df["created"].min())
    assert created.stats["max"].startswith(df["created"].max())
    assert "median_gap_days" not in created.stats  # a sample's gaps are not the table's


def test_sampled_quartiles_are_unbiased_on_a_sorted_table(sales) -> None:  # type: ignore[no-untyped-def]
    """A truncated over-draw keeps the low end of a sorted table; it must not."""
    from eda_mcp.db.connect import open_engine
    from eda_mcp.db.sqlprofile import profile_table
    from eda_mcp.registry import Connection

    path, df = sales
    settings = load_settings(log_level="WARNING", allowed_paths=[path.parent], sample_size=2000)
    opened = open_engine(dsn_for(path), settings)
    try:
        measured = profile_table(Connection("s", "sqlite", opened.engine), None, "sales", settings)
    finally:
        opened.engine.dispose()
    price = {p.name: p for p in measured.profiles}["price"].stats
    assert price["q1"] == pytest.approx(df["price"].quantile(0.25), rel=0.05)
    assert price["q3"] == pytest.approx(df["price"].quantile(0.75), rel=0.05)


def test_profile_tool_labels_what_was_sampled(pushdown) -> None:  # type: ignore[no-untyped-def]
    payload = call(pushdown, "profile", {"source": "s.sales"})
    sampled = payload["sampled"]
    assert sampled["of"] == 30_000 and sampled["seed"] == 42
    assert sampled["method"] == "rowid hash" and "quartiles" in sampled["estimated"]
    assert payload["shape"] == [30_000, 5]
    assert "duplicate_rows" not in payload  # cannot be judged from a sample
    assert any("placeholder code(s) -999" in f for f in payload["findings"])
    assert estimate_tokens(payload) <= 1500 * 1.15


def test_pushdown_profile_is_reproducible(pushdown) -> None:  # type: ignore[no-untyped-def]
    first = call(pushdown, "profile", {"source": "s.sales", "detail": "full"})
    second = call(pushdown, "profile", {"source": "s.sales", "detail": "full"})
    assert first == second


def test_small_table_profile_matches_loading_it(connected) -> None:  # type: ignore[no-untyped-def]
    in_place = call(connected, "profile", {"source": "shop.orders", "detail": "full"})
    assert "sampled" not in in_place and in_place["duplicate_rows"] == 0
    call(connected, "load_dataset", {"source": "shop.orders"})
    loaded = call(connected, "profile", {"source": "orders", "detail": "full"})
    in_place.pop("dataset"), loaded.pop("dataset")
    assert in_place == loaded


def test_pushdown_column_subset(pushdown) -> None:  # type: ignore[no-untyped-def]
    payload = call(pushdown, "profile", {"source": "s.sales", "columns": ["price", "age"]})
    assert set(payload["columns"]) == {"price", "age"}
    missing = call(pushdown, "profile", {"source": "s.sales", "columns": ["nope"]})
    assert missing["error"]["code"] == "COLUMN_NOT_FOUND"


@pytest.mark.parametrize(
    ("source", "code"),
    [
        ("s.nope", "SOURCE_NOT_FOUND"),
        ("nowhere", "SOURCE_NOT_FOUND"),
        ("s.nope.sales", "SOURCE_NOT_FOUND"),
    ],
)
def test_pushdown_profile_errors(pushdown, source, code) -> None:  # type: ignore[no-untyped-def]
    assert call(pushdown, "profile", {"source": source})["error"]["code"] == code


def test_profiling_never_changes_the_database(pushdown, sales) -> None:  # type: ignore[no-untyped-def]
    path, _ = sales
    query = "SELECT COUNT(*), SUM(price), SUM(age) FROM sales"
    before = sqlite3.connect(path).execute(query).fetchone()
    call(pushdown, "profile", {"source": "s.sales"})
    assert sqlite3.connect(path).execute(query).fetchone() == before


# --------------------------------------------------------------------------
# SQL in query


def test_sql_scalar_rows_and_joins(connected) -> None:  # type: ignore[no-untyped-def]
    count = call(
        connected, "query", {"source": "shop", "expression": "SELECT COUNT(*) AS n FROM orders"}
    )
    assert count["result"] == 500 and count["column"] == "n"

    sql = (
        "SELECT c.country, COUNT(*) AS orders, ROUND(AVG(o.amount), 2) AS avg_amount "
        "FROM orders o JOIN customers c ON c.id = o.customer_id "
        "GROUP BY c.country ORDER BY c.country"
    )
    grouped = call(connected, "query", {"source": "shop", "expression": sql})
    assert grouped["columns"] == ["country", "orders", "avg_amount"]
    assert grouped["returned"] == 3 and "more_rows" not in grouped
    assert all(len(row) == 3 for row in grouped["rows"])


def test_sql_values_are_not_rounded(connected) -> None:  # type: ignore[no-untyped-def]
    # Statistics carry 3 significant figures; values asked for must not.
    sql = "SELECT ROUND(AVG(amount), 2) AS a FROM orders WHERE customer_id % 3 = 0"
    payload = call(connected, "query", {"source": "shop", "expression": sql})
    assert payload["result"] != round(payload["result"])  # kept its decimals
    rows = call(
        connected,
        "query",
        {"source": "shop", "expression": "SELECT amount FROM orders WHERE id = 333"},
    )
    assert rows["result"] == 499.5


def test_sql_row_cap_cannot_be_lifted(connected) -> None:  # type: ignore[no-untyped-def]
    sql = "SELECT * FROM orders ORDER BY amount DESC LIMIT 1000"
    payload = call(connected, "query", {"source": "shop", "expression": sql, "limit": 3})
    assert payload["returned"] == 3 and payload["more_rows"] is True
    assert [row[0] for row in payload["rows"]] == [499, 498, 497]


@pytest.mark.parametrize(
    ("sql", "code"),
    [
        ("DELETE FROM orders", "STATEMENT_REJECTED"),
        ("SELECT 1; DROP TABLE orders", "STATEMENT_REJECTED"),
        ("WITH d AS (DELETE FROM orders RETURNING *) SELECT * FROM d", "STATEMENT_REJECTED"),
        ("SELECT load_extension('evil')", "STATEMENT_REJECTED"),
        ("SELECT * FROM missing_table", "INVALID_OPERATION"),
        ("SELEC oops", "STATEMENT_REJECTED"),
    ],
)
def test_sql_errors(connected, sql, code) -> None:  # type: ignore[no-untyped-def]
    assert call(connected, "query", {"source": "shop", "expression": sql})["error"]["code"] == code


def test_sql_timeout_is_reported(shop: Path) -> None:
    srv = build_server(
        load_settings(log_level="WARNING", allowed_paths=[shop.parent], statement_timeout=1)
    )
    call(srv, "connect_database", {"alias": "shop", "dsn": dsn_for(shop)})
    endless = (
        "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT COUNT(*) FROM c"
    )
    payload = call(srv, "query", {"source": "shop", "expression": endless})
    assert payload["error"]["code"] == "QUERY_TIMEOUT" and payload["error"]["retryable"] is True


def test_sql_stays_within_budget_and_changes_nothing(connected, shop: Path) -> None:  # type: ignore[no-untyped-def]
    before = sqlite3.connect(shop).execute("SELECT COUNT(*), SUM(amount) FROM orders").fetchone()
    payload = call(
        connected, "query", {"source": "shop", "expression": "SELECT * FROM orders", "limit": 100}
    )
    assert estimate_tokens(payload) <= 500 * 1.15 and payload["more_rows"] is True
    after = sqlite3.connect(shop).execute("SELECT COUNT(*), SUM(amount) FROM orders").fetchone()
    assert before == after


def test_dataset_sources_still_use_the_expression_language(connected) -> None:  # type: ignore[no-untyped-def]
    call(connected, "load_dataset", {"source": "shop.customers"})
    payload = call(connected, "query", {"source": "customers", "expression": "count(by=country)"})
    assert payload["values"] == {"FR": 17, "UK": 17, "US": 16}
    sql_on_dataset = call(connected, "query", {"source": "customers", "expression": "SELECT 1"})
    assert sql_on_dataset["error"]["code"] == "INVALID_OPERATION"  # not SQL there
