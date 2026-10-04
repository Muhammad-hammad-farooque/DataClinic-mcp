"""The statement guard: model-composed SQL is untrusted input.

This is layer 2 of the four in spec 6.3; the others are a read-only session,
bound parameters for every value the server itself supplies, and a statement
timeout with a row cap. Any one layer failing must not be enough to write.

A statement passes only if all of these hold:

* it parses, in the connection's dialect, to exactly one statement --
  ``SELECT 1; DROP TABLE x`` is two and is refused;
* its root is a query: SELECT, or a UNION / INTERSECT / EXCEPT of them;
* no node anywhere in its tree changes data, schema or session state.
  Walking the whole tree matters: ``WITH d AS (DELETE ... RETURNING *)
  SELECT * FROM d`` has a SELECT at its root, and ``SELECT ... INTO`` and
  ``SELECT ... FOR UPDATE`` are SELECTs that write or lock;
* it calls no function on the deny list -- file access, extension loading,
  sleeps, sequence advances, and functions that execute a string of SQL and
  so would carry an unvalidated statement past this check.

What then runs is SQL **regenerated from the validated tree**, comments
removed, never the caller's original text. Whatever the parser saw is
exactly what executes, so comment smuggling and parser-differential tricks
have nothing to hide in.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from eda_mcp.errors import DependencyMissingError, StatementRejectedError

try:
    import sqlglot
    from sqlglot import exp
    from sqlglot.errors import ParseError, TokenError
except ImportError:  # pragma: no cover - exercised only without the extra
    sqlglot = None  # type: ignore[assignment]

# sqlglot logs a warning whenever it falls back to an opaque Command node;
# the guard rejects those anyway, so the warning is noise on stderr.
logging.getLogger("sqlglot").setLevel(logging.ERROR)

MAX_SQL_LENGTH = 20_000

# SQLAlchemy dialect name -> sqlglot dialect name.
DIALECTS = {
    "postgresql": "postgres",
    "sqlite": "sqlite",
    "mysql": "mysql",
    "mariadb": "mysql",
    "mssql": "tsql",
    "duckdb": "duckdb",
}

# Functions refused in any dialect, lower-cased. Grouped by what they reach.
DENIED_FUNCTIONS = frozenset(
    {
        # file system and server-side files
        "pg_read_file", "pg_read_binary_file", "pg_ls_dir", "pg_stat_file",
        "pg_ls_logdir", "pg_ls_waldir", "pg_ls_tmpdir", "lo_import", "lo_export",
        "load_file", "readfile", "writefile", "edit",
        "read_csv", "read_csv_auto", "read_parquet", "parquet_scan", "read_json",
        "read_json_auto", "read_json_objects", "read_ndjson", "read_ndjson_auto",
        "read_text", "read_blob", "glob", "sniff_csv", "read_xlsx", "iceberg_scan",
        "delta_scan", "openrowset", "opendatasource", "openquery", "bulk",
        # executing a string of SQL, which would bypass this guard
        "query", "query_table", "query_to_xml", "query_to_xml_and_xmlschema",
        "table_to_xml", "cursor_to_xml", "dblink", "dblink_exec", "dblink_send_query",
        "dblink_open", "dblink_connect", "sp_executesql", "exec", "execute",
        # extensions, environment and server control
        "load_extension", "fts3_tokenizer", "getenv", "set_config",
        "pg_reload_conf", "pg_rotate_logfile", "pg_terminate_backend",
        "pg_cancel_backend", "pg_promote", "pg_switch_wal", "xp_cmdshell",
        "sys_exec", "sys_eval",
        # state changes and locks reachable from a SELECT
        "nextval", "setval", "lastval", "pg_advisory_lock", "pg_advisory_xact_lock",
        "pg_try_advisory_lock", "get_lock", "release_lock", "pg_notify",
        # deliberate stalls
        "pg_sleep", "pg_sleep_for", "pg_sleep_until", "sleep", "benchmark",
        "waitfor", "randomblob", "zeroblob",
    }
)  # fmt: skip

# Node types that change data, schema or session, refused anywhere in the tree.
_FORBIDDEN_NAMES = (
    "Insert", "Update", "Delete", "Merge", "Create", "Drop", "Alter",
    "TruncateTable", "Command", "Copy", "Grant", "Revoke", "Set", "Pragma",
    "Attach", "Detach", "Install", "Transaction", "Commit", "Rollback", "Lock",
    "LoadData", "Use", "Kill", "Cache", "Uncache", "Refresh", "Analyze", "Into",
)  # fmt: skip


@dataclass(frozen=True, slots=True)
class Checked:
    """A statement that passed the guard, ready to execute."""

    sql: str  # regenerated from the validated tree, comments removed
    dialect: str


def _require() -> None:
    if sqlglot is None:
        raise DependencyMissingError("sqlglot", "sql")


def sqlglot_dialect(sqlalchemy_dialect: str) -> str:
    """The sqlglot name for a SQLAlchemy dialect; unknown ones are refused."""
    try:
        return DIALECTS[sqlalchemy_dialect]
    except KeyError:
        raise StatementRejectedError(
            f"no SQL guard is defined for the {sqlalchemy_dialect!r} dialect"
        ) from None


def _function_name(node: exp.Func) -> str:
    if isinstance(node, exp.Anonymous):
        return str(node.name).lower()
    return node.sql_name().lower()


def check(sql: str, dialect: str, row_cap: int | None = None, count: bool = False) -> Checked:
    """Validate *sql* for *dialect* (a sqlglot name) and return what to run.

    With *row_cap*, the statement is wrapped so it can return at most that
    many rows. With *count*, it is wrapped as ``SELECT COUNT(*) FROM (...)``
    instead, to size a result before fetching it. Raises
    ``StatementRejectedError`` naming the reason otherwise.
    """
    _require()
    if len(sql) > MAX_SQL_LENGTH:
        raise StatementRejectedError(f"statement longer than {MAX_SQL_LENGTH:,} characters")

    try:
        trees = [t for t in sqlglot.parse(sql, read=dialect) if t is not None]
    except (ParseError, TokenError) as exc:
        raise StatementRejectedError(
            f"could not parse it ({str(exc).splitlines()[0][:120]})"
        ) from None
    except RecursionError:
        raise StatementRejectedError("nested too deeply to parse") from None

    if not trees:
        raise StatementRejectedError("no statement found")
    if len(trees) > 1:
        raise StatementRejectedError(f"{len(trees)} statements found; send exactly one")

    tree = trees[0]
    if not isinstance(tree, (exp.Select, exp.SetOperation)):
        raise StatementRejectedError(f"{type(tree).__name__.upper()} is not a read-only query")

    forbidden = tuple(getattr(exp, name) for name in _FORBIDDEN_NAMES if hasattr(exp, name))
    for node in tree.walk():
        if isinstance(node, forbidden):
            what = type(node).__name__.upper()
            if isinstance(node, exp.Into):
                what = "SELECT ... INTO"
            elif isinstance(node, exp.Lock):
                what = "FOR UPDATE / FOR SHARE locking"
            raise StatementRejectedError(f"{what} is not allowed, even inside a query")
        if isinstance(node, exp.Func):
            name = _function_name(node)
            if name in DENIED_FUNCTIONS:
                raise StatementRejectedError(f"the function {name}() is not allowed")

    if count:
        tree = exp.select(exp.Count(this=exp.Star())).from_(tree.subquery("guarded"))
    elif row_cap is not None:
        tree = exp.select("*").from_(tree.subquery("guarded")).limit(row_cap)
    return Checked(tree.sql(dialect=dialect, comments=False), dialect)


__all__ = ["DENIED_FUNCTIONS", "DIALECTS", "Checked", "check", "sqlglot_dialect"]
