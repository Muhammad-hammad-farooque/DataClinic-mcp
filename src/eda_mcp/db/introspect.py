"""Schema discovery for ``explore_schema``: schemas, then tables, then a table.

Exploration is progressive so a large warehouse is never dumped in one go:
the first call lists schemas, the next a schema's tables, the next one
table's columns, keys and indexes (spec 7.1).

Row counts in listings are planner estimates where the database keeps them
(``pg_class.reltuples``, SQLite's ``sqlite_stat1``) and are labelled as such;
counting every table exactly could take minutes on a warehouse (spec 6.5).
A single SQLite table, being local, is counted exactly.

All identifiers reach SQL through SQLAlchemy's quoting or as bound
parameters, never by string formatting (spec 6.3, layer 3).
"""

from __future__ import annotations

from typing import Any

from eda_mcp.errors import EDAError, ErrorCode

try:
    import sqlalchemy as sa
    from sqlalchemy.engine import Engine
    from sqlalchemy.exc import DBAPIError
except ImportError:  # pragma: no cover - exercised only without the extra
    sa = None  # type: ignore[assignment]

SYSTEM_SCHEMAS = {"information_schema", "pg_catalog", "pg_toast"}


def _user_schemas(inspector: Any, dialect: str) -> list[str]:
    if dialect == "sqlite":
        return ["main"]
    names = inspector.get_schema_names()
    return sorted(
        n for n in names if n not in SYSTEM_SCHEMAS and not n.startswith(("pg_temp", "pg_toast"))
    )


def _estimates(engine: Engine, dialect: str, schema: str) -> dict[str, int]:
    """Planner row estimates per table, where the database keeps them."""
    try:
        with engine.connect() as connection:
            if dialect == "postgresql":
                rows = connection.execute(
                    sa.text(
                        "SELECT c.relname, c.reltuples FROM pg_class c "
                        "JOIN pg_namespace n ON n.oid = c.relnamespace "
                        "WHERE n.nspname = :schema AND c.relkind IN ('r', 'p')"
                    ),
                    {"schema": schema},
                ).all()
                # -1 means never analysed: unknown, not empty.
                return {name: int(count) for name, count in rows if count >= 0}
            has_stats = connection.execute(
                sa.text(
                    "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'sqlite_stat1'"
                )
            ).first()
            if not has_stats:
                return {}
            rows = connection.execute(sa.text("SELECT tbl, stat FROM sqlite_stat1")).all()
            return {name: int(str(stat).split()[0]) for name, stat in rows if stat}
    except DBAPIError:
        return {}


def overview(engine: Engine, dialect: str) -> dict[str, Any]:
    """Every user schema with its table and view counts."""
    inspector = sa.inspect(engine)
    schemas: dict[str, dict[str, int]] = {}
    for schema in _user_schemas(inspector, dialect):
        tables = inspector.get_table_names(schema=schema)
        views = inspector.get_view_names(schema=schema)
        schemas[schema] = {"tables": len(tables), "views": len(views)}
    return {"schemas": schemas, "tables": sum(s["tables"] for s in schemas.values())}


def _not_found(kind: str, name: str, known: list[str]) -> EDAError:
    shown = ", ".join(known[:15]) + (f", ... (+{len(known) - 15} more)" if len(known) > 15 else "")
    return EDAError(
        ErrorCode.SOURCE_NOT_FOUND,
        f"no {kind} named {name!r}",
        f"{kind}s: {shown}" if known else f"this database has no {kind}s",
    )


def _check_schema(inspector: Any, dialect: str, schema: str) -> None:
    known = _user_schemas(inspector, dialect)
    if schema not in known:
        raise _not_found("schema", schema, known)


def tables(engine: Engine, dialect: str, schema: str) -> dict[str, dict[str, Any]]:
    """The tables and views in *schema*, sorted by name."""
    inspector = sa.inspect(engine)
    _check_schema(inspector, dialect, schema)
    estimates = _estimates(engine, dialect, schema)
    listing: dict[str, dict[str, Any]] = {}
    for kind, names in (
        ("table", inspector.get_table_names(schema=schema)),
        ("view", inspector.get_view_names(schema=schema)),
    ):
        for name in names:
            if dialect == "sqlite" and name.startswith("sqlite_"):
                continue  # internal bookkeeping, not user data
            entry: dict[str, Any] = {"kind": kind}
            entry["columns"] = len(inspector.get_columns(name, schema=schema))
            if name in estimates:
                entry["rows_estimate"] = estimates[name]
            listing[name] = entry
    return dict(sorted(listing.items()))


def _render_column(column: Any, primary: set[str]) -> str:
    """One column as a terse line: ``id INTEGER not null pk``."""
    parts = [str(column["name"]), str(column["type"])]
    if not column.get("nullable", True):
        parts.append("not null")
    if column["name"] in primary:
        parts.append("pk")
    if column.get("default") is not None:
        parts.append(f"default {column['default']}")
    return " ".join(parts)


def describe_table(engine: Engine, dialect: str, schema: str, table: str) -> dict[str, Any]:
    """Columns, keys, indexes and size of one table or view."""
    inspector = sa.inspect(engine)
    _check_schema(inspector, dialect, schema)
    known = sorted(
        [*inspector.get_table_names(schema=schema), *inspector.get_view_names(schema=schema)]
    )
    if table not in known:
        raise _not_found("table", table, known)

    primary = set(
        inspector.get_pk_constraint(table, schema=schema).get("constrained_columns") or []
    )
    columns = [_render_column(c, primary) for c in inspector.get_columns(table, schema=schema)]
    detail: dict[str, Any] = {"columns": columns}

    foreign = [
        {
            "columns": fk["constrained_columns"],
            "references": f"{fk['referred_table']}({', '.join(fk['referred_columns'])})",
        }
        for fk in inspector.get_foreign_keys(table, schema=schema)
    ]
    if foreign:
        detail["foreign_keys"] = foreign
    indexes = [
        {"name": ix["name"], "columns": ix["column_names"], "unique": bool(ix.get("unique"))}
        for ix in inspector.get_indexes(table, schema=schema)
    ]
    if indexes:
        detail["indexes"] = indexes

    if dialect == "sqlite":
        # Local and usually modest: an exact count is cheap and worth having.
        try:
            with engine.connect() as connection:
                count = connection.execute(
                    sa.select(sa.func.count()).select_from(sa.table(table))
                ).scalar()
            detail["rows"] = int(count or 0)
            detail["row_count_exact"] = True
        except DBAPIError as exc:
            raise EDAError(
                ErrorCode.INVALID_OPERATION, f"could not count {table}", str(exc.orig)[:200]
            ) from None
    else:
        estimate = _estimates(engine, dialect, schema).get(table)
        if estimate is not None:
            detail["rows"] = estimate
            detail["row_count_exact"] = False
    return detail


__all__ = ["describe_table", "overview", "tables"]
