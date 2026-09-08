"""Column classification and the orientation digest.

Phase 1 provides what ``load_dataset`` needs to answer the obvious next
question in its first response, so the routine follow-up ``profile`` call
never happens. Phase 2 extends this module with distributions and full
statistics.

Every statistic here is computed over the whole frame. Sampling would be
faster and is what makes competing servers wrong on sorted files.

See spec sections 7.1, 10.4 and A.3.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

import pandas as pd
from pandas.api import types as pdt

from eda_mcp.digest import Finding, Severity, round_sig

# A column with almost no repetition is a key, not a feature; one with almost
# no variation carries no signal. Both are reported rather than analysed.
IDENTIFIER_UNIQUE_RATIO = 0.99
CONSTANT_DOMINANCE = 0.99
HIGH_CARDINALITY = 50


class ColumnKind(StrEnum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    BOOLEAN = "boolean"
    TEXT = "text"
    IDENTIFIER = "identifier"
    CONSTANT = "constant"
    EMPTY = "empty"


def _is_row_counter(non_null: pd.Series) -> bool:
    """True when a column is a consecutive integer sequence.

    Distinguishes a surrogate key from a numeric feature that merely happens
    to have no repeated values.
    """
    if len(non_null) < 2 or int(non_null.nunique()) != len(non_null):
        return False
    ordered = non_null.sort_values()
    return bool((ordered.diff().dropna() == 1).all())


def classify(series: pd.Series) -> ColumnKind:
    """Assign the kind that decides how a column is analysed.

    Order matters: emptiness and constancy are checked before dtype, because
    an all-null float column is not usefully "numeric".
    """
    non_null = series.dropna()
    if non_null.empty:
        return ColumnKind.EMPTY

    if non_null.value_counts(normalize=True).iloc[0] >= CONSTANT_DOMINANCE:
        return ColumnKind.CONSTANT

    if pdt.is_bool_dtype(series):
        return ColumnKind.BOOLEAN
    if pdt.is_datetime64_any_dtype(series):
        return ColumnKind.DATETIME

    n_unique = int(non_null.nunique())
    if pdt.is_numeric_dtype(series):
        # Being unique is not enough: prices and measurements are often unique.
        # A surrogate key is unique *and* consecutive, which is a real signature.
        if pdt.is_integer_dtype(series) and _is_row_counter(non_null):
            return ColumnKind.IDENTIFIER
        return ColumnKind.NUMERIC

    if n_unique / len(non_null) >= IDENTIFIER_UNIQUE_RATIO:
        return ColumnKind.IDENTIFIER
    if n_unique > HIGH_CARDINALITY:
        return ColumnKind.TEXT
    return ColumnKind.CATEGORICAL


def column_kinds(df: pd.DataFrame) -> dict[str, ColumnKind]:
    return {str(c): classify(df[c]) for c in df.columns}


def orientation(
    df: pd.DataFrame, kinds: dict[str, ColumnKind]
) -> tuple[dict[str, Any], list[Finding]]:
    """Summarise a freshly loaded frame and flag what needs attention."""
    rows, cols = df.shape
    missing_by_column = df.isna().sum()
    total_cells = rows * cols
    duplicates = int(df.duplicated().sum())

    body: dict[str, Any] = {
        "shape": [int(rows), int(cols)],
        "memory_mb": round_sig(df.memory_usage(deep=True).sum() / 1024**2),
        "columns": {
            kind.value: sorted(c for c, k in kinds.items() if k is kind)
            for kind in ColumnKind
            if any(k is kind for k in kinds.values())
        },
        "missing_cells_pct": round_sig(
            float(missing_by_column.sum()) / total_cells * 100 if total_cells else 0.0
        ),
        "duplicate_rows": duplicates,
    }

    findings: list[Finding] = []

    if rows == 0:
        findings.append(Finding(Severity.HIGH, "the file has no rows"))
        return body, findings

    for column, kind in kinds.items():
        missing = int(missing_by_column[column])
        share = missing / rows
        if share >= 0.6:
            findings.append(
                Finding(
                    Severity.HIGH,
                    f"{share:.0%} missing",
                    column=column,
                    recommendation="add a missingness flag rather than imputing",
                    affected_rows=missing,
                )
            )
        elif share >= 0.2:
            findings.append(
                Finding(
                    Severity.MEDIUM,
                    f"{share:.0%} missing",
                    column=column,
                    recommendation="decide between imputation and a flag",
                    affected_rows=missing,
                )
            )

        if kind is ColumnKind.CONSTANT:
            findings.append(
                Finding(
                    Severity.MEDIUM,
                    "single value throughout",
                    column=column,
                    recommendation="drop; it carries no signal",
                )
            )
        elif kind is ColumnKind.EMPTY:
            findings.append(
                Finding(
                    Severity.HIGH,
                    "entirely empty",
                    column=column,
                    recommendation="drop",
                )
            )
        elif kind is ColumnKind.IDENTIFIER:
            findings.append(
                Finding(
                    Severity.LOW,
                    "unique per row",
                    column=column,
                    recommendation="treat as an identifier, not a feature",
                )
            )

    if duplicates:
        findings.append(
            Finding(
                Severity.MEDIUM,
                f"{duplicates / rows:.0%} of rows are exact duplicates",
                recommendation="drop_duplicates unless repetition is meaningful",
                affected_rows=duplicates,
            )
        )

    if not findings:
        findings.append(Finding(Severity.INFO, "no obvious structural problems"))
    return body, findings


def summarise(df: pd.DataFrame, kinds: dict[str, ColumnKind], findings: list[Finding]) -> str:
    """One sentence stating scale and whether anything needs attention."""
    rows, cols = df.shape
    serious = sum(1 for f in findings if f.severity in (Severity.HIGH, Severity.MEDIUM))
    columns_flagged = len(
        {f.column for f in findings if f.column and f.severity is not Severity.LOW}
    )
    if not serious:
        return f"{cols} columns, {rows:,} rows. No structural problems found."
    return (
        f"{cols} columns, {rows:,} rows. "
        f"{columns_flagged} column(s) need attention before analysis."
    )
