"""Push-down profiling: column statistics computed inside the database.

A table that has not been loaded is profiled where it lives (spec 5.1, 6.4).
Counts, missingness, distinct counts, extremes, means, standard deviations,
zeros, negatives, top values, IQR outlier counts and duplicate keys are
**exact**, from aggregate queries -- one per batch of columns -- that move
no rows. On PostgreSQL the quartiles are exact too (``PERCENTILE_CONT``).

What SQL cannot compute portably -- skew, kurtosis, the text checks,
quartiles on SQLite -- comes from a seeded sample, scaled to the table where
it is a row count, and the response names those statistics (spec 6.5).
Statistics that cannot be scaled honestly, such as the gaps in a date
series or duplicate rows, are left out rather than guessed.

A table no larger than the sample size is simply fetched whole and profiled
in memory: every statistic is then exact and nothing needs labelling.

All SQL is assembled from sqlglot expressions with quoted identifiers and
passes the statement guard before it runs (spec 6.3).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from eda_mcp.config import Settings
from eda_mcp.db.connect import read_frame
from eda_mcp.db.guard import check, sqlglot_dialect
from eda_mcp.db.introspect import describe_table
from eda_mcp.db.sampling import sample_table, table_expression
from eda_mcp.errors import ColumnNotFoundError
from eda_mcp.loaders import LoadReport, coerce_types
from eda_mcp.profiling import IQR_FENCE, ColumnKind, ColumnProfile, column_kinds, profile_frame
from eda_mcp.registry import Connection

try:
    import sqlalchemy as sa
    from sqlglot import exp
except ImportError:  # pragma: no cover - exercised only without the extra
    sa = None  # type: ignore[assignment]

BATCH = 15
TOP_VALUES = 5
DEFAULT_SCHEMA = {"sqlite": "main", "postgresql": "public"}
DOUBLE = {"sqlite": "REAL", "postgresql": "DOUBLE PRECISION"}
ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}")

# Row counts that only the sample can supply; scaled to the table.
SCALED = (
    "whitespace_padded", "blank", "numeric_like", "placeholders", "date_like",
    "variant_rows", "rare_rows", "outliers_modified_z", "future", "before_1900", "true",
)  # fmt: skip
# Facts about a sample that do not describe the table at all.
DROPPED = ("median_gap_days", "max_gap_days", "sorted")


@dataclass(slots=True)
class TableProfile:
    profiles: list[ColumnProfile]
    kinds: dict[str, ColumnKind]
    rows: int
    columns: int
    sampled: dict[str, Any] | None = None
    frame: pd.DataFrame | None = None  # the whole table, when it was small enough to fetch
    exact: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# SQL building


def _col(name: str) -> exp.Column:
    return exp.column(name, quoted=True)


def _number(value: float) -> exp.Expression:
    return exp.Literal.number(value)


def _as_double(name: str, dialect: str) -> exp.Cast:
    return exp.Cast(this=_col(name), to=exp.DataType.build(DOUBLE[dialect], dialect=_glot(dialect)))


def _glot(dialect: str) -> str:
    return sqlglot_dialect(dialect)


def _count_if(condition: exp.Expression) -> exp.Sum:
    return exp.Sum(
        this=exp.Case(
            ifs=[exp.If(this=condition, true=_number(1))],
            default=_number(0),
        )
    )


def _run(connection: Connection, query: exp.Select, settings: Settings, rows: int) -> pd.DataFrame:
    dialect = _glot(connection.dialect)
    checked = check(query.sql(dialect=dialect), dialect, row_cap=rows)
    frame, _ = read_frame(connection.engine, connection.dialect, checked, settings, row_cap=rows)
    return frame


def _is_numeric(sql_type: Any) -> bool:
    return isinstance(sql_type, (sa.Integer, sa.Numeric, sa.Float)) and not isinstance(
        sql_type, sa.Boolean
    )


def _is_date(sql_type: Any) -> bool:
    return isinstance(sql_type, (sa.Date, sa.DateTime))


# --------------------------------------------------------------------------
# exact aggregates


def _aggregates(
    connection: Connection,
    table: str,
    schema: str | None,
    names: list[str],
    types: dict[str, Any],
    samples: dict[str, ColumnProfile],
    settings: Settings,
) -> tuple[int, dict[str, dict[str, Any]]]:
    """Exact per-column statistics, one aggregate query per batch of columns."""
    dialect = connection.dialect
    exact: dict[str, dict[str, Any]] = {name: {} for name in names}
    total = 0
    for start in range(0, len(names), BATCH):
        batch = names[start : start + BATCH]
        select: list[Any] = [exp.Count(this=exp.Star()).as_("n")]
        for i, name in enumerate(batch):
            column = _col(name)

            def named(expression: exp.Expression, label: str, i: int = i) -> Any:
                # "<label>_<column index>": label digits (q1) never blur into the index.
                return expression.as_(f"{label}_{i}")

            select.append(named(exp.Count(this=column.copy()), "p"))
            select.append(named(exp.Count(this=exp.Distinct(expressions=[column.copy()])), "d"))
            if _is_numeric(types[name]):
                shift = samples[name].stats.get("mean", 0.0) if name in samples else 0.0
                deviation = exp.Paren(
                    this=exp.Sub(this=_as_double(name, dialect), expression=_number(shift or 0.0))
                )
                select += [
                    named(exp.Min(this=column.copy()), "lo"),
                    named(exp.Max(this=column.copy()), "hi"),
                    named(exp.Avg(this=_as_double(name, dialect)), "m"),
                    # Squared deviations from the sample mean, not sum(x^2):
                    # stable when values are large but tightly spread.
                    named(exp.Sum(this=exp.Mul(this=deviation, expression=deviation.copy())), "ss"),
                    named(_count_if(exp.EQ(this=column.copy(), expression=_number(0))), "z"),
                    named(_count_if(exp.LT(this=column.copy(), expression=_number(0))), "ng"),
                ]
                if dialect == "postgresql":
                    for q, label in ((0.25, "q1"), (0.5, "q2"), (0.75, "q3")):
                        ordered = exp.Order(expressions=[exp.Ordered(this=column.copy())])
                        percentile = exp.WithinGroup(
                            this=exp.PercentileCont(this=_number(q)), expression=ordered
                        )
                        select.append(named(percentile, label))
            elif _is_date(types[name]) or _iso_text(samples.get(name)):
                select += [
                    named(exp.Min(this=column.copy()), "lo"),
                    named(exp.Max(this=column.copy()), "hi"),
                ]
        query = exp.select(*select).from_(table_expression(table, schema))
        row = _run(connection, query, settings, 1).iloc[0]
        total = int(row["n"])
        for key, value in row.items():
            if key == "n" or value is None or _is_nan(value):
                continue
            label, index = str(key).rsplit("_", 1)
            exact[batch[int(index)]][label] = value
    return total, exact


def _is_nan(value: Any) -> bool:
    return isinstance(value, float) and math.isnan(value)


def _iso_text(profile: ColumnProfile | None) -> bool:
    """A text column whose sampled values are ISO dates sorts correctly as text."""
    return bool(
        profile is not None
        and profile.kind is ColumnKind.DATETIME
        and profile.stats.get("min")
        and ISO_DATE.match(str(profile.stats["min"]))
    )


def _outliers(
    connection: Connection,
    table: str,
    schema: str | None,
    fences: dict[str, tuple[float, float]],
    codes: dict[str, list[int]],
    settings: Settings,
) -> dict[str, dict[str, Any]]:
    """Exact counts beyond the IQR fences, and of each placeholder code seen."""
    names = sorted(set(fences) | set(codes))
    found: dict[str, dict[str, Any]] = {name: {} for name in names}
    for start in range(0, len(names), BATCH):
        batch = names[start : start + BATCH]
        select: list[Any] = []
        for i, name in enumerate(batch):
            column = _col(name)
            if name in fences:
                low, high = fences[name]
                outside = exp.Or(
                    this=exp.LT(this=column.copy(), expression=_number(low)),
                    expression=exp.GT(this=column.copy(), expression=_number(high)),
                )
                select.append(_count_if(outside).as_(f"o_{i}"))
            for j, code in enumerate(codes.get(name, [])):
                select.append(
                    _count_if(exp.EQ(this=column.copy(), expression=_number(code))).as_(
                        f"s_{i}_{j}"
                    )
                )
        if not select:
            continue
        row = _run(
            connection, exp.select(*select).from_(table_expression(table, schema)), settings, 1
        ).iloc[0]
        for i, name in enumerate(batch):
            if f"o_{i}" in row.index:
                found[name]["outliers_iqr"] = int(row[f"o_{i}"] or 0)
            sentinels = {
                code: int(row[f"s_{i}_{j}"] or 0) for j, code in enumerate(codes.get(name, []))
            }
            if sentinels:
                found[name]["sentinels"] = {k: v for k, v in sentinels.items() if v}
    return found


def _top_values(
    connection: Connection, table: str, schema: str | None, name: str, settings: Settings
) -> pd.Series:
    """The most frequent values of one column, exactly, ties broken by value."""
    column = _col(name)
    query = (
        exp.select(column.copy().as_("v"), exp.Count(this=exp.Star()).as_("n"))
        .from_(table_expression(table, schema))
        .where(exp.Not(this=exp.Is(this=column.copy(), expression=exp.Null())))
        .group_by(column.copy())
        .order_by(exp.Ordered(this=exp.column("n"), desc=True), exp.Ordered(this=exp.column("v")))
        .limit(TOP_VALUES)
    )
    frame = _run(connection, query, settings, TOP_VALUES)
    return pd.Series(frame["n"].to_numpy(), index=frame["v"].astype(str))


# --------------------------------------------------------------------------
# merging sample and exact statistics


def _merge(
    profile: ColumnProfile,
    exact: dict[str, Any],
    extra: dict[str, Any],
    top: pd.Series | None,
    rows: int,
    factor: float,
    dialect: str,
) -> None:
    """Overwrite a sample-based profile with exact values, scaling the rest."""
    stats = profile.stats
    for key in DROPPED:
        stats.pop(key, None)
    for key in SCALED:
        if isinstance(stats.get(key), (int, float)) and not isinstance(stats.get(key), bool):
            stats[key] = round(stats[key] * factor)
    if isinstance(stats.get("mixed_types"), dict):
        stats["mixed_types"] = {k: round(v * factor) for k, v in stats["mixed_types"].items()}

    present = int(exact.get("p", 0))
    profile.count, profile.missing, profile.unique = present, rows - present, int(exact.get("d", 0))
    if present == 0:
        profile.kind, profile.stats = ColumnKind.EMPTY, {}
        return

    kind = profile.kind
    if kind is ColumnKind.NUMERIC and "m" in exact:
        mean = float(exact["m"])
        shift = float(stats.get("mean", 0.0) or 0.0)
        if present > 1 and "ss" in exact:
            variance = (float(exact["ss"]) - present * (mean - shift) ** 2) / (present - 1)
            stats["std"] = math.sqrt(max(variance, 0.0))
        stats.update(
            {
                "mean": mean,
                "min": float(exact["lo"]),
                "max": float(exact["hi"]),
                "zeros": int(exact.get("z", 0)),
                "negatives": int(exact.get("ng", 0)),
            }
        )
        if dialect == "postgresql" and "q1" in exact:
            q1, q3 = float(exact["q1"]), float(exact["q3"])
            stats.update({"q1": q1, "median": float(exact["q2"]), "q3": q3})
            iqr = q3 - q1
            if iqr > 0:
                stats["iqr_bounds"] = [q1 - IQR_FENCE * iqr, q3 + IQR_FENCE * iqr]
        stats.pop("outliers_iqr", None)
        stats.pop("sentinels", None)
        stats.update(extra)
    elif kind is ColumnKind.DATETIME and "lo" in exact:
        low, high = pd.Timestamp(exact["lo"]), pd.Timestamp(exact["hi"])
        stats.update(
            {
                "min": low.isoformat(),
                "max": high.isoformat(),
                "span_days": (high - low) / pd.Timedelta(days=1),
            }
        )
    elif kind is ColumnKind.IDENTIFIER:
        stats["duplicate_keys"] = present - profile.unique
    elif top is not None and len(top):
        if kind is ColumnKind.CONSTANT:
            stats["value"] = str(top.index[0])
            stats["dominance_pct"] = int(top.iloc[0]) / present * 100
        elif kind is ColumnKind.CATEGORICAL:
            stats["top"] = {str(k): int(v) for k, v in top.items()}
            stats["top_share_pct"] = int(top.iloc[0]) / present * 100


# --------------------------------------------------------------------------
# entry point


def profile_table(
    connection: Connection,
    schema: str | None,
    table: str,
    settings: Settings,
    columns: list[str] | None = None,
) -> TableProfile:
    """Profile a table where it lives, moving as few rows as possible."""
    schema = schema or DEFAULT_SCHEMA.get(connection.dialect, "public")
    detail = describe_table(connection.engine, connection.dialect, schema, table)
    inspector = sa.inspect(connection.engine)
    meta = inspector.get_columns(table, schema=schema)
    types = {str(c["name"]): c["type"] for c in meta}
    names = list(types)
    if columns:
        for name in columns:
            if name not in types:
                raise ColumnNotFoundError(name, names)
    is_view = table in inspector.get_view_names(schema=schema)
    qualifier = None if connection.dialect == "sqlite" else schema

    # Small enough to hold: fetch it whole, and every statistic is exact.
    estimate = detail.get("rows")
    if estimate is not None and estimate <= settings.sample_size:
        query = exp.select("*").from_(table_expression(table, qualifier))
        dialect = _glot(connection.dialect)
        frame, more = read_frame(
            connection.engine,
            connection.dialect,
            check(query.sql(dialect=dialect), dialect, row_cap=settings.sample_size),
            settings,
            settings.sample_size,
        )
        if not more:
            frame = coerce_types(frame, LoadReport(format=connection.dialect))
            kinds = column_kinds(frame)
            profiles = profile_frame(frame, kinds, columns=columns)
            return TableProfile(profiles, kinds, len(frame), len(names), frame=frame)

    if estimate is None:
        # Never analysed: count exactly rather than sample at a guessed rate.
        counted = exp.select(exp.Count(this=exp.Star()).as_("n")).from_(
            table_expression(table, qualifier)
        )
        estimate = int(_run(connection, counted, settings, 1).to_numpy()[0, 0])
    sample = sample_table(connection, table, qualifier, int(estimate), settings, is_view)
    frame = coerce_types(sample.frame, LoadReport(format=connection.dialect))
    kinds = column_kinds(frame)
    profiles = profile_frame(frame, kinds, columns=columns)
    by_name = {p.name: p for p in profiles}

    wanted = [p.name for p in profiles]
    rows, exact = _aggregates(connection, table, qualifier, wanted, types, by_name, settings)

    fences: dict[str, tuple[float, float]] = {}
    codes: dict[str, list[int]] = {}
    for name in wanted:
        profile = by_name[name]
        if profile.kind is not ColumnKind.NUMERIC or not _is_numeric(types[name]):
            continue
        q1 = exact[name].get("q1", profile.stats.get("q1"))
        q3 = exact[name].get("q3", profile.stats.get("q3"))
        if q1 is not None and q3 is not None and float(q3) > float(q1):
            iqr = float(q3) - float(q1)
            fences[name] = (float(q1) - IQR_FENCE * iqr, float(q3) + IQR_FENCE * iqr)
        if profile.stats.get("sentinels"):
            codes[name] = sorted(int(c) for c in profile.stats["sentinels"])
    extras = _outliers(connection, table, qualifier, fences, codes, settings)

    factor = rows / max(len(frame), 1)
    for name in wanted:
        profile = by_name[name]
        top = None
        if profile.kind in (ColumnKind.CATEGORICAL, ColumnKind.CONSTANT):
            top = _top_values(connection, table, qualifier, name, settings)
        _merge(profile, exact[name], extras.get(name, {}), top, rows, factor, connection.dialect)
        kinds[name] = profile.kind

    estimated = ["skew", "kurtosis", "text checks", "spelling variants", "column kinds"]
    if connection.dialect == "sqlite":
        estimated.insert(2, "quartiles")
    sampled = dict(sample.info)
    sampled["estimated"] = ", ".join(estimated) + " (row counts scaled to the table)"
    return TableProfile(profiles, kinds, rows, len(names), sampled=sampled)


__all__ = ["TableProfile", "profile_table"]
