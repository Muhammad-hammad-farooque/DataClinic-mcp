"""Seeded, reported sampling for statistics that cannot be pushed down.

Skew, kurtosis and the text checks have no portable SQL aggregate, so they
are estimated from a sample -- and the response always says so (spec 6.5).
Two runs on unchanged data draw the same sample (spec 13.4):

* PostgreSQL: ``TABLESAMPLE BERNOULLI (p) REPEATABLE (seed)``.
* SQLite: its ``random()`` cannot be seeded, so rows are chosen by a seeded
  hash of the rowid. The hash scatters selection across the whole table, so
  a table sorted by value is sampled evenly rather than from one end.

SQL is assembled from sqlglot expressions, not strings, and still passes the
statement guard before it runs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import pandas as pd

from eda_mcp.config import Settings
from eda_mcp.db.connect import read_frame
from eda_mcp.db.guard import check, sqlglot_dialect
from eda_mcp.registry import Connection

try:
    from sqlglot import exp
except ImportError:  # pragma: no cover - exercised only without the extra
    exp = None  # type: ignore[assignment]

# The row cap is a safety net only, set well above the expected draw: if a
# LIMIT routinely trimmed the sample, it would keep the rows that come first
# in storage order -- on a table sorted by value, the low end. So the draw
# rate targets n exactly, and the cap almost never binds.
SAFETY_CAP = 2
# Prime modulus for the SQLite rowid hash, and an odd multiplier from
# Knuth's multiplicative hashing.
HASH_MODULUS = 1_000_003
HASH_MULTIPLIER = 2_654_435_761


@dataclass(slots=True)
class Sample:
    frame: pd.DataFrame
    info: dict[str, Any]  # the envelope's "sampled" field


def table_expression(table: str, schema: str | None) -> exp.Table:
    """A fresh, quoted table reference; each query gets its own node."""
    return exp.table_(table, db=schema, quoted=True)


def _number(value: float) -> exp.Expression:
    return exp.Literal.number(value)


def sample_table(
    connection: Connection,
    table: str,
    schema: str | None,
    rows: int,
    settings: Settings,
    is_view: bool = False,
) -> Sample:
    """About ``settings.sample_size`` rows, drawn evenly across the table, reproducibly."""
    n, seed = settings.sample_size, settings.seed
    share = min(1.0, n / max(rows, 1))
    source = table_expression(table, schema)
    reproducible = True

    if connection.dialect == "postgresql":
        source.set(
            "sample",
            exp.TableSample(
                method=exp.var("BERNOULLI"),
                percent=_number(round(share * 100, 6)),
                seed=_number(seed),
            ),
        )
        query = exp.select("*").from_(source)
        method = "bernoulli"
    elif is_view:
        # A view has no rowid to hash, and SQLite's random() takes no seed.
        query = exp.select("*").from_(source).order_by(exp.Rand())
        method = "random"
        reproducible = False
    else:
        hashed = exp.Mod(
            this=exp.Paren(
                this=exp.Add(
                    this=exp.Mul(this=exp.column("rowid"), expression=_number(HASH_MULTIPLIER)),
                    expression=_number(seed),
                )
            ),
            expression=_number(HASH_MODULUS),
        )
        threshold = math.ceil(HASH_MODULUS * share)
        query = (
            exp.select("*").from_(source).where(exp.LT(this=hashed, expression=_number(threshold)))
        )
        method = "rowid hash"

    dialect = sqlglot_dialect(connection.dialect)
    cap = n * SAFETY_CAP
    checked = check(query.sql(dialect=dialect), dialect, row_cap=cap)
    frame, _ = read_frame(connection.engine, connection.dialect, checked, settings, row_cap=cap)
    info: dict[str, Any] = {"n": len(frame), "of": rows, "method": method, "seed": seed}
    if not reproducible:
        info["reproducible"] = False
    return Sample(frame, info)


__all__ = ["Sample", "sample_table", "table_expression"]
