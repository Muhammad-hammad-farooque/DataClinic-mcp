"""Opening database connections: credentials, read-only sessions, limits.

Two of the four read-only layers in spec 6.3 live here:

* **Session** (layer 1). SQLite files open with ``mode=ro``, so the driver
  itself cannot write. PostgreSQL sessions start with
  ``default_transaction_read_only=on`` and every transaction re-asserts
  ``SET TRANSACTION READ ONLY``.
* **Limits** (layer 4). Every statement runs under ``statement_timeout`` --
  natively on PostgreSQL, through a progress handler on SQLite, which has no
  such setting -- and every result is fetched under a row cap.

Credentials are resolved without ever being echoed: responses name the
*source* of a DSN (an environment variable, the config file), never its
value, and every message passes through ``errors.redact`` (spec 6.2).

Supported dialects are SQLite and PostgreSQL, the targets of phase 3. Others
are refused by name rather than half-supported.
"""

from __future__ import annotations

import os
import re
import sqlite3
import stat
import sys
import time
import tomllib
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from eda_mcp.config import CONFIG_PATH, Settings
from eda_mcp.db.guard import Checked
from eda_mcp.errors import (
    DependencyMissingError,
    EDAError,
    ErrorCode,
    OperationRefusedError,
    PathNotAllowedError,
)

try:
    import sqlalchemy as sa
    from sqlalchemy import event
    from sqlalchemy.engine import URL, Engine, make_url
    from sqlalchemy.exc import ArgumentError, DBAPIError, OperationalError
    from sqlalchemy.pool import QueuePool
except ImportError:  # pragma: no cover - exercised only without the extra
    sa = None  # type: ignore[assignment]

SUPPORTED = ("sqlite", "postgresql")
PLANNED = {"mysql", "mariadb", "mssql", "duckdb", "oracle"}

# Bounded pool (spec 13.1): an analysis session needs few connections, and a
# leak should fail fast rather than exhaust the server.
POOL = {"pool_size": 2, "max_overflow": 3, "pool_pre_ping": True, "pool_recycle": 1800}
CONNECT_TIMEOUT = 10
# How many SQLite VM instructions run between timeout checks.
SQLITE_PROGRESS_STEPS = 10_000


@dataclass(slots=True)
class Resolved:
    """A DSN and where it came from. ``source`` is safe to show; ``dsn`` is not."""

    dsn: str
    source: str
    warning: str | None = None


def _require() -> None:
    if sa is None:
        raise DependencyMissingError("sqlalchemy", "sql")


def env_name(alias: str) -> str:
    """The per-alias environment variable: warehouse -> EDA_MCP_DSN_WAREHOUSE."""
    return "EDA_MCP_DSN_" + re.sub(r"[^A-Za-z0-9]", "_", alias).upper()


def _from_config(alias: str, path: Path = CONFIG_PATH) -> str | None:
    """A DSN from ``[connections]`` in the config file, if it is private.

    A file other users can read is refused outright: silently using it would
    teach users that a world-readable secret is acceptable. POSIX permission
    bits do not apply on Windows, where the check is skipped.
    """
    if not path.is_file():
        return None
    if sys.platform != "win32" and path.stat().st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise OperationRefusedError(
            f"reading credentials from {path}",
            "the file is readable by other users",
            f"restrict it with: chmod 600 {path}",
        )
    try:
        with path.open("rb") as handle:
            connections = tomllib.load(handle).get("connections", {})
    except (OSError, tomllib.TOMLDecodeError):
        return None
    value = connections.get(alias) if isinstance(connections, dict) else None
    return value if isinstance(value, str) and value else None


def resolve_dsn(
    alias: str,
    dsn: str | None = None,
    env_var: str | None = None,
    config_path: Path = CONFIG_PATH,
) -> Resolved:
    """Find the DSN for *alias* in the order of spec 6.2.

    1. the environment variable named by *env_var*, if given
    2. ``EDA_MCP_DSN_<ALIAS>``
    3. ``DATABASE_URL``
    4. ``[connections]`` in ``~/.eda-mcp/config.toml``
    5. the explicit *dsn* argument, with a warning that it now sits in the
       conversation history
    """
    if env_var:
        value = os.environ.get(env_var)
        if not value:
            raise EDAError(
                ErrorCode.INVALID_OPERATION,
                f"environment variable {env_var} is not set",
                "set it in the server's environment, then connect again",
            )
        return Resolved(value, f"environment variable {env_var}")

    candidates = (
        (os.environ.get(env_name(alias)), f"environment variable {env_name(alias)}"),
        (os.environ.get("DATABASE_URL"), "environment variable DATABASE_URL"),
        (_from_config(alias, config_path), f"{config_path} [connections]"),
    )
    for value, source in candidates:
        if value:
            note = f"dsn argument ignored: {source} takes precedence" if dsn else None
            return Resolved(value, source, note)

    if dsn:
        return Resolved(
            dsn,
            "dsn argument",
            "the DSN now resides in the conversation history; "
            f"prefer setting {env_name(alias)} in the server's environment",
        )
    raise EDAError(
        ErrorCode.INVALID_OPERATION,
        f"no credentials found for {alias!r}",
        f"set {env_name(alias)} (or DATABASE_URL) in the server's environment, "
        "or pass env_var= naming a variable that holds the DSN",
    )


# --------------------------------------------------------------------------
# engines


def _sqlite_engine(url: URL, settings: Settings) -> Engine:
    database = url.database or ""
    if database in ("", ":memory:"):
        raise EDAError(
            ErrorCode.INVALID_OPERATION,
            "an in-memory SQLite database is empty when opened read-only",
            "point the DSN at a database file: sqlite:///path/to/file.db",
        )
    path = Path(database).expanduser()
    if not settings.path_allowed(path):
        raise PathNotAllowedError(str(path), [str(p) for p in settings.allowed_paths])
    path = path.resolve()
    if not path.is_file():
        raise EDAError(
            ErrorCode.SOURCE_NOT_FOUND,
            f"no SQLite database at {path}",
            "check the path; a read-only connection never creates a file",
        )

    def open_read_only() -> sqlite3.Connection:
        # mode=ro: the driver refuses writes no matter what SQL arrives.
        return sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True, check_same_thread=False)

    return sa.create_engine("sqlite://", creator=open_read_only, poolclass=QueuePool, **POOL)


def _postgres_engine(url: URL, settings: Settings) -> Engine:
    if url.drivername == "postgresql":
        url = url.set(drivername="postgresql+psycopg")
    timeout_ms = settings.statement_timeout * 1000
    try:
        engine = sa.create_engine(
            url,
            connect_args={
                "connect_timeout": CONNECT_TIMEOUT,
                "options": (
                    f"-c default_transaction_read_only=on -c statement_timeout={timeout_ms}"
                ),
            },
            **POOL,
        )
    except ImportError:
        raise DependencyMissingError("psycopg", "postgres") from None

    @event.listens_for(engine, "begin")
    def _read_only_transaction(connection: Any) -> None:
        # Belt and braces with the session default: each transaction asserts
        # read-only itself, before any statement of the caller's runs.
        connection.exec_driver_sql("SET TRANSACTION READ ONLY")

    return engine


@dataclass(slots=True)
class Opened:
    engine: Engine
    dialect: str
    version: str
    database: str


def open_engine(dsn: str, settings: Settings, read_only: bool = True) -> Opened:
    """Create and test an engine for *dsn*; nothing is written, ever, in phase 3."""
    _require()
    if not read_only:
        raise EDAError(
            ErrorCode.WRITE_NOT_PERMITTED,
            "write connections are not available",
            "connect with read_only=True; writing back arrives with export",
        )
    try:
        url = make_url(dsn)
    except ArgumentError:
        raise EDAError(
            ErrorCode.INVALID_OPERATION,
            "the DSN is not a valid database URL",
            "use the form dialect://user@host:port/database or sqlite:///path.db",
        ) from None

    dialect = url.get_backend_name()
    if dialect == "sqlite":
        engine = _sqlite_engine(url, settings)
        database = Path(url.database or "").name
    elif dialect == "postgresql":
        engine = _postgres_engine(url, settings)
        database = url.database or ""
    elif dialect in PLANNED:
        raise EDAError(
            ErrorCode.INVALID_OPERATION,
            f"{dialect} is not supported yet",
            f"supported now: {', '.join(SUPPORTED)}",
        )
    else:
        raise EDAError(
            ErrorCode.INVALID_OPERATION,
            f"unknown database dialect {dialect!r}",
            f"supported: {', '.join(SUPPORTED)}",
        )

    try:
        with engine.connect() as connection:
            if dialect == "sqlite":
                version = str(connection.exec_driver_sql("SELECT sqlite_version()").scalar())
            else:
                version = str(connection.exec_driver_sql("SHOW server_version").scalar())
    except (OperationalError, DBAPIError) as exc:
        engine.dispose()
        raise EDAError(
            ErrorCode.CONNECTION_FAILED,
            f"could not connect: {str(exc.orig).splitlines()[0][:200]}",
            "check that the server is reachable and the credentials are right",
        ) from None
    return Opened(engine, dialect, version, database)


# --------------------------------------------------------------------------
# executing checked SQL


@contextmanager
def _deadline(connection: Any, dialect: str, seconds: int) -> Iterator[None]:
    """Abort the statement after *seconds* on SQLite, which has no timeout setting.

    PostgreSQL enforces ``statement_timeout`` server-side, set at connect.
    """
    if dialect != "sqlite":
        yield
        return
    raw = connection.connection.dbapi_connection
    limit = time.monotonic() + seconds
    raw.set_progress_handler(lambda: int(time.monotonic() > limit), SQLITE_PROGRESS_STEPS)
    try:
        yield
    finally:
        raw.set_progress_handler(None, 0)


def read_frame(
    engine: Engine, dialect: str, checked: Checked, settings: Settings, row_cap: int
) -> tuple[pd.DataFrame, bool]:
    """Run guarded SQL and return at most *row_cap* rows, and whether more existed.

    The transaction is always rolled back: nothing a query does is kept.
    """
    try:
        with (
            engine.connect() as connection,
            _deadline(connection, dialect, settings.statement_timeout),
        ):
            result = connection.exec_driver_sql(checked.sql)
            rows = result.fetchmany(row_cap + 1)
            columns = list(result.keys())
            connection.rollback()
    except (OperationalError, DBAPIError) as exc:
        message = str(exc.orig).splitlines()[0][:200] if exc.orig else str(exc)[:200]
        if "interrupted" in message.lower() or "statement timeout" in message.lower():
            raise EDAError(
                ErrorCode.QUERY_TIMEOUT,
                f"the query ran past statement_timeout ({settings.statement_timeout}s)",
                "narrow it with WHERE or aggregate in SQL, or raise EDA_MCP_STATEMENT_TIMEOUT",
            ) from None
        raise EDAError(
            ErrorCode.INVALID_OPERATION,
            f"the database rejected the query: {message}",
            "check table and column names with explore_schema",
        ) from None
    truncated = len(rows) > row_cap
    return pd.DataFrame(rows[:row_cap], columns=columns), truncated


__all__ = [
    "Opened",
    "Resolved",
    "env_name",
    "open_engine",
    "read_frame",
    "resolve_dsn",
]
