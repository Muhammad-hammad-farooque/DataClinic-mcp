"""Loading a database table or query result into the pandas engine.

Once loaded, a table is an ordinary dataset: every analysis tool works on it
unchanged (spec 5.1). Loading is the exception, not the default -- a table
is sized before any row moves, and one above ``max_load_rows`` is refused
unless the caller passes ``limit=``, with ``profile`` suggested instead
(spec 7.1).

Table names reach SQL as quoted identifiers built by sqlglot, and caller
queries pass the statement guard; both run through ``read_frame`` under the
statement timeout and a row cap.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from eda_mcp.config import Settings
from eda_mcp.db.connect import read_frame
from eda_mcp.db.guard import check, sqlglot_dialect
from eda_mcp.db.introspect import describe_table
from eda_mcp.errors import EDAError, ErrorCode, SourceTooLargeError
from eda_mcp.loaders import LoadReport, coerce_types
from eda_mcp.registry import Connection

try:
    from sqlglot import exp
except ImportError:  # pragma: no cover - exercised only without the extra
    exp = None  # type: ignore[assignment]

DEFAULT_SCHEMA = {"sqlite": "main", "postgresql": "public"}


@dataclass(slots=True)
class Loaded:
    df: pd.DataFrame
    report: LoadReport
    origin: str
    default_alias: str
    available: int  # rows the table or query holds
    exact: bool  # whether *available* is an exact count or an estimate


def split_reference(
    source: str, connections: dict[str, Connection]
) -> tuple[str, str | None, str] | None:
    """``conn.table`` or ``conn.schema.table`` -> parts, if *conn* is open.

    Anything else -- a file path, an unknown prefix -- is not a table
    reference, and the caller treats it as a file.
    """
    parts = source.split(".")
    if len(parts) not in (2, 3) or parts[0] not in connections or not all(parts):
        return None
    if len(parts) == 2:
        return parts[0], None, parts[1]
    return parts[0], parts[1], parts[2]


def _count(connection: Connection, sql: str, settings: Settings) -> int:
    dialect = sqlglot_dialect(connection.dialect)
    frame, _ = read_frame(
        connection.engine, connection.dialect, check(sql, dialect, count=True), settings, row_cap=1
    )
    return int(frame.to_numpy()[0, 0])


def _fetch(
    connection: Connection,
    sql: str,
    settings: Settings,
    name: str,
    available: int,
    exact: bool,
    limit: int | None,
) -> tuple[pd.DataFrame, LoadReport]:
    if limit is not None and limit <= 0:
        raise EDAError(ErrorCode.INVALID_OPERATION, "limit must be positive", "e.g. limit=10000")
    if available > settings.max_load_rows and limit is None:
        raise SourceTooLargeError(name, available, settings.max_load_rows)

    cap = min(limit or settings.max_load_rows, settings.max_load_rows)
    dialect = sqlglot_dialect(connection.dialect)
    frame, more = read_frame(
        connection.engine, connection.dialect, check(sql, dialect, row_cap=cap), settings, cap
    )

    report = LoadReport(format=connection.dialect)
    if limit is not None and (more or available > len(frame)):
        # Without ORDER BY a database returns rows in whatever order is
        # cheapest; saying so stops a prefix being mistaken for a sample.
        report.notes.append(
            f"stopped at limit={limit:,} of {'' if exact else '~'}{available:,} rows, in the "
            "database's own order -- not a random sample; statistics cover the rows read"
        )
    return coerce_types(frame, report), report


def load_table(
    connection: Connection,
    schema: str | None,
    table: str,
    settings: Settings,
    limit: int | None = None,
) -> Loaded:
    """Load one table or view, refusing it if it is too large to hold."""
    schema = schema or DEFAULT_SCHEMA.get(connection.dialect, "public")
    # Validates both names and supplies the size: exact on SQLite, the
    # planner's estimate on PostgreSQL.
    detail = describe_table(connection.engine, connection.dialect, schema, table)
    dialect = sqlglot_dialect(connection.dialect)
    qualifier = None if connection.dialect == "sqlite" else schema
    sql = exp.select("*").from_(exp.table_(table, db=qualifier, quoted=True)).sql(dialect=dialect)

    if "rows" in detail:
        available, exact = int(detail["rows"]), bool(detail["row_count_exact"])
    else:
        available, exact = _count(connection, sql, settings), True  # never analysed
    name = f"{connection.alias}.{table}"
    df, report = _fetch(connection, sql, settings, name, available, exact, limit)
    origin = f"{connection.alias}:{schema}.{table}"
    return Loaded(df, report, origin, table, available, exact)


def load_query(
    connection: Connection, sql: str, settings: Settings, limit: int | None = None
) -> Loaded:
    """Load the result of a caller's SELECT, which must pass the guard."""
    available = _count(connection, sql, settings)
    name = f"the query on {connection.alias}"
    df, report = _fetch(connection, sql, settings, name, available, True, limit)
    return Loaded(
        df, report, f"{connection.alias}:query", f"{connection.alias}_query", available, True
    )


__all__ = ["Loaded", "load_query", "load_table", "split_reference"]
