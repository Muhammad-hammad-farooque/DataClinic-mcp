"""Integration tests against a real PostgreSQL server.

These prove what only a live server can: that the read-only session holds
even when a raw connection tries to switch it off, that statement_timeout
really cancels, that PERCENTILE_CONT and TABLESAMPLE ... REPEATABLE behave
as the push-down profiler assumes, and that planner estimates come back.

They run in CI's ``postgres`` job against a service container, and skip
anywhere ``EDA_TEST_POSTGRES_DSN`` is not set.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

DSN = os.environ.get("EDA_TEST_POSTGRES_DSN", "")

# CI sets EDA_REQUIRE_INTEGRATION so that a missing DSN fails the job loudly
# instead of letting every test here skip and the job pass on nothing.
if os.environ.get("EDA_REQUIRE_INTEGRATION") and not DSN:
    raise RuntimeError("EDA_REQUIRE_INTEGRATION is set but EDA_TEST_POSTGRES_DSN is not")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not DSN, reason="EDA_TEST_POSTGRES_DSN is not set"),
]

ROWS = 20_000


def call(server, tool, args):  # type: ignore[no-untyped-def]
    result = asyncio.run(server.call_tool(tool, args))
    payload = result[1] if isinstance(result, tuple) else result
    return payload.structured_content


@pytest.fixture(scope="module")
def seeded():  # type: ignore[no-untyped-def]
    """A shop schema written through a separate, ordinary connection."""
    import sqlalchemy as sa

    writer = sa.create_engine(DSN.replace("postgresql://", "postgresql+psycopg://", 1))
    rng = np.random.default_rng(0)
    orders = pd.DataFrame(
        {
            "id": np.arange(ROWS),
            "customer_id": rng.integers(0, 50, ROWS),
            "price": np.round(rng.lognormal(3, 1, ROWS), 2),
            "age": rng.integers(18, 90, ROWS).astype(float),
            "placed": pd.Timestamp("2024-01-01")
            + pd.to_timedelta(rng.integers(0, 700, ROWS), unit="D"),
        }
    )
    orders.loc[rng.random(ROWS) < 0.1, "price"] = np.nan
    orders.loc[rng.random(ROWS) < 0.02, "age"] = -999
    # Stored sorted by price: the case that exposes biased sampling.
    orders = orders.sort_values("price", na_position="first").reset_index(drop=True)
    customers = pd.DataFrame(
        {"id": np.arange(50), "name": [f"c{i}" for i in range(50)], "country": ["UK", "FR"] * 25}
    )

    with writer.begin() as c:
        c.exec_driver_sql("DROP SCHEMA IF EXISTS shop CASCADE")
        c.exec_driver_sql("CREATE SCHEMA shop")
        c.exec_driver_sql(
            "CREATE TABLE shop.customers (id INTEGER PRIMARY KEY, name TEXT NOT NULL, country TEXT)"
        )
        c.exec_driver_sql(
            "CREATE TABLE shop.orders (id INTEGER PRIMARY KEY, "
            "customer_id INTEGER REFERENCES shop.customers(id), "
            "price DOUBLE PRECISION, age DOUBLE PRECISION, placed DATE)"
        )
        c.exec_driver_sql("CREATE INDEX ix_orders_customer ON shop.orders(customer_id)")
        c.exec_driver_sql(
            "CREATE VIEW shop.big_orders AS SELECT * FROM shop.orders WHERE price > 100"
        )
    customers.to_sql("customers", writer, schema="shop", if_exists="append", index=False)
    orders.assign(placed=orders["placed"].dt.date).to_sql(
        "orders", writer, schema="shop", if_exists="append", index=False, chunksize=5000
    )
    with writer.begin() as c:
        c.exec_driver_sql("ANALYZE shop.orders")
        c.exec_driver_sql("ANALYZE shop.customers")
    yield orders, writer
    with writer.begin() as c:
        c.exec_driver_sql("DROP SCHEMA IF EXISTS shop CASCADE")
    writer.dispose()


@pytest.fixture()
def server(seeded, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):  # type: ignore[no-untyped-def]
    from eda_mcp.config import load_settings
    from eda_mcp.server import build_server

    monkeypatch.setenv("EDA_MCP_DSN_PG", DSN)
    srv = build_server(
        load_settings(log_level="WARNING", allowed_paths=[tmp_path], sample_size=2000)
    )
    connected = call(srv, "connect_database", {"alias": "pg"})
    assert "error" not in connected, connected
    return srv


def count_orders(writer) -> int:  # type: ignore[no-untyped-def]
    with writer.connect() as c:
        return int(c.exec_driver_sql("SELECT COUNT(*) FROM shop.orders").scalar())


# --------------------------------------------------------------------------
# connection and the read-only layers


def test_connect_reports_the_server(server) -> None:  # type: ignore[no-untyped-def]
    payload = call(server, "manage_sources", {"action": "list"})
    assert payload["connections"][0]["dialect"] == "postgresql"
    shape = call(server, "explore_schema", {"connection": "pg"})
    assert shape["schemas"]["shop"] == {"tables": 2, "views": 1}


def test_session_stays_read_only_even_when_asked_not_to(seeded, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    """Bypass the guard entirely and try to write, then to turn read-only off."""
    from eda_mcp.config import load_settings
    from eda_mcp.db.connect import open_engine

    _, writer = seeded
    before = count_orders(writer)
    opened = open_engine(DSN, load_settings(log_level="WARNING", allowed_paths=[tmp_path]))
    try:
        with pytest.raises(Exception, match="read-only transaction"), opened.engine.connect() as c:
            c.exec_driver_sql("DELETE FROM shop.orders")
        with opened.engine.connect() as c:
            # Allowed as a statement -- but each new transaction re-asserts
            # SET TRANSACTION READ ONLY, so it changes nothing.
            c.exec_driver_sql("SET SESSION CHARACTERISTICS AS TRANSACTION READ WRITE")
            c.commit()
            with pytest.raises(Exception, match="read-only transaction"):
                c.exec_driver_sql("DELETE FROM shop.orders")
    finally:
        opened.engine.dispose()
    assert count_orders(writer) == before


def test_statement_timeout_cancels(seeded, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    from eda_mcp.config import load_settings
    from eda_mcp.db.connect import open_engine, read_frame
    from eda_mcp.db.guard import Checked
    from eda_mcp.errors import EDAError, ErrorCode

    fast = load_settings(log_level="WARNING", allowed_paths=[tmp_path], statement_timeout=1)
    opened = open_engine(DSN, fast)
    start = time.perf_counter()
    try:
        # pg_sleep is refused by the guard; build the statement directly to
        # prove the server-side limit holds on its own.
        with pytest.raises(EDAError) as caught:
            read_frame(
                opened.engine, "postgresql", Checked("SELECT pg_sleep(10)", "postgres"), fast, 1
            )
    finally:
        opened.engine.dispose()
    assert caught.value.code is ErrorCode.QUERY_TIMEOUT
    assert time.perf_counter() - start < 5


def test_wrong_password_never_leaks(seeded, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    from eda_mcp.config import load_settings
    from eda_mcp.server import build_server

    srv = build_server(load_settings(log_level="WARNING", allowed_paths=[tmp_path]))
    secret = "not-the-password-xyz"
    user_part, host_part = DSN.split("@", 1)
    bad = user_part.rsplit(":", 1)[0] + ":" + secret + "@" + host_part
    payload = call(srv, "connect_database", {"alias": "bad", "dsn": bad})
    assert payload["error"]["code"] == "CONNECTION_FAILED"
    assert secret not in json.dumps(payload)


# --------------------------------------------------------------------------
# exploration, loading, profiling, SQL


def test_explore_schema_on_postgres(server) -> None:  # type: ignore[no-untyped-def]
    listing = call(server, "explore_schema", {"connection": "pg", "schema": "shop"})
    assert listing["tables"]["big_orders"]["kind"] == "view"
    assert listing["tables"]["orders"]["rows_estimate"] == pytest.approx(ROWS, rel=0.05)

    table = call(
        server, "explore_schema", {"connection": "pg", "schema": "shop", "table": "orders"}
    )
    assert table["row_count_exact"] is False
    assert table["foreign_keys"] == [{"columns": ["customer_id"], "references": "customers(id)"}]
    assert "id INTEGER not null pk" in table["columns"]


def test_load_table_from_postgres(server) -> None:  # type: ignore[no-untyped-def]
    payload = call(server, "load_dataset", {"source": "pg.shop.customers"})
    assert payload["shape"] == [50, 3] and payload["origin"] == "pg:shop.customers"
    limited = call(server, "load_dataset", {"source": "pg.shop.orders", "limit": 100})
    assert limited["shape"][0] == 100 and limited["read"]["row_count_exact"] is False


def test_pushdown_profile_on_postgres_is_exact(server, seeded) -> None:  # type: ignore[no-untyped-def]
    orders, _ = seeded
    from eda_mcp.config import load_settings
    from eda_mcp.db.connect import open_engine
    from eda_mcp.db.sqlprofile import profile_table
    from eda_mcp.registry import Connection

    settings = load_settings(log_level="WARNING", allowed_paths=[Path.cwd()], sample_size=2000)
    opened = open_engine(DSN, settings)
    try:
        measured = profile_table(
            Connection("pg", "postgresql", opened.engine), "shop", "orders", settings
        )
    finally:
        opened.engine.dispose()
    assert measured.rows == ROWS and measured.sampled["method"] == "bernoulli"
    assert "quartiles" not in measured.sampled["estimated"]  # exact on PostgreSQL

    price = {p.name: p for p in measured.profiles}["price"]
    truth = orders["price"]
    assert (price.count, price.missing) == (truth.count(), truth.isna().sum())
    for stat, expected in (
        ("mean", truth.mean()),
        ("std", truth.std()),
        ("min", truth.min()),
        ("max", truth.max()),
        # PERCENTILE_CONT interpolates linearly, exactly as pandas does
        ("q1", truth.quantile(0.25)),
        ("median", truth.quantile(0.5)),
        ("q3", truth.quantile(0.75)),
    ):
        assert price.stats[stat] == pytest.approx(expected, rel=1e-9), stat
    age = {p.name: p for p in measured.profiles}["age"]
    assert age.stats["sentinels"] == {-999: int((orders["age"] == -999).sum())}


def test_pushdown_profile_tool_is_reproducible(server) -> None:  # type: ignore[no-untyped-def]
    first = call(server, "profile", {"source": "pg.shop.orders", "detail": "full"})
    second = call(server, "profile", {"source": "pg.shop.orders", "detail": "full"})
    assert first == second  # TABLESAMPLE ... REPEATABLE (seed)
    assert first["sampled"]["method"] == "bernoulli"


def test_sql_query_on_postgres(server, seeded) -> None:  # type: ignore[no-untyped-def]
    _, writer = seeded
    before = count_orders(writer)
    sql = (
        "SELECT c.country, COUNT(*) AS n, ROUND(AVG(o.price)::numeric, 2) AS avg_price "
        "FROM shop.orders o JOIN shop.customers c ON c.id = o.customer_id "
        "GROUP BY c.country ORDER BY c.country"
    )
    payload = call(server, "query", {"source": "pg", "expression": sql})
    assert payload["columns"] == ["country", "n", "avg_price"]
    assert sum(row[1] for row in payload["rows"]) == ROWS
    # NUMERIC arrives as Decimal: it must come back as a number, not text
    assert all(isinstance(row[2], float) for row in payload["rows"])

    for hostile in (
        "SELECT pg_read_file('/etc/passwd')",
        "DELETE FROM shop.orders",
        "SELECT pg_sleep(30)",
    ):
        refused = call(server, "query", {"source": "pg", "expression": hostile})
        assert refused["error"]["code"] == "STATEMENT_REJECTED", hostile
    assert count_orders(writer) == before
