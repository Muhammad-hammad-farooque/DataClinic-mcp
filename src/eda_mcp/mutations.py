"""Cleaning operations for ``clean_data``: a batch, applied as one undoable step.

A call is all-or-nothing on *errors* and per-operation on *refusals*:

* Every operation is checked before any is applied -- an unknown name, a
  misspelt parameter, a missing column -- so a typo in the fifth step never
  leaves the first four half-applied. Execution works on a copy; an
  unexpected failure midway discards it.
* An operation the refusal policy blocks (spec 12.2) is skipped and reported
  with its reason as a finding; the rest of the batch still applies.

Each operation returns what it changed -- rows removed, cells changed,
columns added or removed -- so the response is a delta, never a re-profile
(spec 7.3).

See spec sections 7.3, 12.2 and A.4.3, A.5, A.10, A.18.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as pdt
from scipy import stats as scipy_stats

from eda_mcp.errors import ColumnNotFoundError, EDAError, ErrorCode
from eda_mcp.expressions import evaluate
from eda_mcp.issues import FLAG_SUFFIX

# Refusal policy (spec 12.2).
MAX_ROW_DROP = 0.5
MAX_IMPUTE_MISSING = 0.6
MAX_COLUMN_INFORMATION = 0.9
# Conversions that would blank most of a column are refused like row drops.
MAX_CONVERSION_LOSS = 0.5

OUTLIER_DEFAULTS = {"iqr": 1.5, "zscore": 3.0, "modified_zscore": 3.5}

# Each operation runs on its own copy. Under copy-on-write (always on from
# pandas 3) a shallow copy is free and safe; before it, a shallow copy shares
# column data, so a refused operation could leak into the frame -- deep copy.
COPY_ON_WRITE = int(pd.__version__.split(".")[0]) >= 3 or bool(
    getattr(pd.options.mode, "copy_on_write", False)
)


class RefusedError(Exception):
    """An operation the policy blocks; reported, not raised to the caller."""

    def __init__(self, reason: str, instead: str) -> None:
        super().__init__(reason)
        self.reason, self.instead = reason, instead


@dataclass(slots=True)
class Outcome:
    op: str
    target: str
    refused: RefusedError | None = None
    rows_removed: int = 0
    cells_changed: int = 0
    columns_added: list[str] = field(default_factory=list)
    columns_removed: list[str] = field(default_factory=list)
    note: str | None = None

    def describe(self) -> str:
        """One terse line: ``fill_missing age (median): 1,277 cells``."""
        parts = []
        if self.rows_removed:
            parts.append(f"-{self.rows_removed:,} rows")
        if self.cells_changed:
            parts.append(f"{self.cells_changed:,} cells")
        if self.columns_added:
            parts.append(f"+{', '.join(self.columns_added)}")
        if self.columns_removed:
            parts.append(f"-{len(self.columns_removed)} column(s)")
        line = f"{self.op} {self.target}".rstrip()
        change = ", ".join(parts) or "no change"
        return f"{line}: {change}"


Operation = dict[str, Any]
Handler = Callable[[pd.DataFrame, Operation], tuple[pd.DataFrame, Outcome]]


# --------------------------------------------------------------------------
# helpers


def _invalid(index: int, op: str, message: str, remedy: str) -> EDAError:
    return EDAError(ErrorCode.INVALID_OPERATION, f"operation {index + 1} ({op}): {message}", remedy)


def _columns(df: pd.DataFrame, params: Operation, required: bool = True) -> list[str]:
    """The ``column`` / ``columns`` of an operation, each checked to exist."""
    raw = params.get("columns", params.get("column"))
    if raw is None:
        if required:
            raise EDAError(
                ErrorCode.INVALID_OPERATION, "no column given", "pass column= or columns="
            )
        return []
    names = [raw] if isinstance(raw, str) else list(raw)
    labels = [str(c) for c in df.columns]
    for name in names:
        if name not in labels:
            raise ColumnNotFoundError(str(name), labels)
    return [str(n) for n in names]


def _check_row_drop(df: pd.DataFrame, removed: int) -> None:
    if len(df) and removed / len(df) > MAX_ROW_DROP:
        raise RefusedError(
            f"would remove {removed / len(df):.0%} of rows, over the {MAX_ROW_DROP:.0%} limit",
            "narrow the condition, or split it into smaller steps you can check",
        )


def _is_text(series: pd.Series) -> bool:
    return bool(pdt.is_string_dtype(series.dtype) or pdt.is_object_dtype(series.dtype))


def _numeric(series: pd.Series, name: str, op: str) -> pd.Series:
    if pdt.is_bool_dtype(series.dtype) or not pdt.is_numeric_dtype(series.dtype):
        raise EDAError(
            ErrorCode.INVALID_OPERATION,
            f"{op} needs a numeric column; {name} is {series.dtype}",
            "convert it first with cast_type, or choose a numeric column",
        )
    return series


def _new_name(df: pd.DataFrame, wanted: str) -> str:
    labels = {str(c) for c in df.columns}
    name, n = wanted, 2
    while name in labels:
        name, n = f"{wanted}_{n}", n + 1
    return name


def _mode(series: pd.Series) -> Any:
    counts = series.dropna().value_counts()
    if counts.empty:
        return None
    # Most frequent, ties broken by value, so row order cannot choose.
    return sorted(counts.index, key=lambda v: (-int(counts[v]), str(v)))[0]


# --------------------------------------------------------------------------
# operations


def fill_missing(df: pd.DataFrame, p: Operation) -> tuple[pd.DataFrame, Outcome]:
    names = _columns(df, p, required=False) or [str(c) for c in df.columns[df.isna().any()]]
    method = p.get("method", "auto")
    outcome = Outcome("fill_missing", f"{', '.join(names)} ({method})")
    for name in names:
        series = df[name]
        missing = int(series.isna().sum())
        if not missing:
            continue
        if missing / max(len(series), 1) > MAX_IMPUTE_MISSING:
            raise RefusedError(
                f"{name} is {missing / len(series):.0%} missing; imputing would invent most of it",
                f"use flag_missing on {name} to keep the signal, or drop_columns",
            )
        if p.get("flag"):
            flag = _new_name(df, f"{name}{FLAG_SUFFIX}")
            df[flag] = series.isna().astype("int8")
            outcome.columns_added.append(flag)

        chosen = method
        if chosen == "auto":
            numeric = pdt.is_numeric_dtype(series.dtype) and not pdt.is_bool_dtype(series.dtype)
            chosen = "median" if numeric else "mode"
        if chosen in ("mean", "median"):
            values = _numeric(series, name, f"fill_missing method={chosen}")
            filled = values.fillna(getattr(values, chosen)())
        elif chosen == "mode":
            filled = series.fillna(_mode(series))
        elif chosen == "constant":
            if "value" not in p:
                raise EDAError(
                    ErrorCode.INVALID_OPERATION, "method=constant needs value=", "e.g. value=0"
                )
            filled = series.fillna(p["value"])
        elif chosen in ("ffill", "bfill"):
            filled = series.ffill() if chosen == "ffill" else series.bfill()
        elif chosen == "interpolate":
            filled = _numeric(series, name, "interpolate").interpolate(limit_direction="both")
        elif chosen == "knn":
            raise EDAError(
                ErrorCode.INVALID_OPERATION,
                "knn imputation is not available",
                "use median or mode, with flag=True to keep the missingness signal",
            )
        else:
            raise EDAError(
                ErrorCode.INVALID_OPERATION,
                f"unknown fill method {chosen!r}",
                "use auto, mean, median, mode, constant, ffill, bfill or interpolate",
            )
        outcome.cells_changed += missing - int(filled.isna().sum())
        df[name] = filled
    return df, outcome


def flag_missing(df: pd.DataFrame, p: Operation) -> tuple[pd.DataFrame, Outcome]:
    names = _columns(df, p)
    outcome = Outcome("flag_missing", ", ".join(names))
    for name in names:
        flag = _new_name(df, f"{name}{FLAG_SUFFIX}")
        df[flag] = df[name].isna().astype("int8")
        outcome.columns_added.append(flag)
    return df, outcome


def drop_missing(df: pd.DataFrame, p: Operation) -> tuple[pd.DataFrame, Outcome]:
    names = _columns(df, p, required=False) or None
    how = p.get("how", "any")
    if how not in ("any", "all"):
        raise EDAError(ErrorCode.INVALID_OPERATION, "how must be 'any' or 'all'", "e.g. how='any'")
    kept = df.dropna(subset=names, how=how)
    removed = len(df) - len(kept)
    _check_row_drop(df, removed)
    return kept, Outcome("drop_missing", ", ".join(names or ["any column"]), rows_removed=removed)


def drop_duplicates(df: pd.DataFrame, p: Operation) -> tuple[pd.DataFrame, Outcome]:
    names = _columns(df, p, required=False) or None
    keep = p.get("keep", "first")
    if keep not in ("first", "last"):
        raise EDAError(
            ErrorCode.INVALID_OPERATION, "keep must be 'first' or 'last'", "e.g. keep='first'"
        )
    kept = df.drop_duplicates(subset=names, keep=keep)
    removed = len(df) - len(kept)
    _check_row_drop(df, removed)
    return kept, Outcome(
        "drop_duplicates", ", ".join(names or ["all columns"]), rows_removed=removed
    )


def drop_columns(df: pd.DataFrame, p: Operation) -> tuple[pd.DataFrame, Outcome]:
    names = _columns(df, p)
    if len(names) >= df.shape[1]:
        raise RefusedError("would remove every column", "keep at least one column")
    # "Information" is measured as non-missing cells: dropping columns that
    # hold almost all of the remaining data is almost always a mistake.
    present = df.notna().sum()
    total = int(present.sum())
    share = int(present[names].sum()) / total if total else 0.0
    if share > MAX_COLUMN_INFORMATION:
        raise RefusedError(
            f"those columns hold {share:.0%} of the remaining data, over the "
            f"{MAX_COLUMN_INFORMATION:.0%} limit",
            "drop fewer columns at once, or keep the ones carrying the data",
        )
    return df.drop(columns=names), Outcome("drop_columns", ", ".join(names), columns_removed=names)


def drop_rows(df: pd.DataFrame, p: Operation) -> tuple[pd.DataFrame, Outcome]:
    where = p.get("where")
    if not isinstance(where, str) or not where.strip():
        raise EDAError(
            ErrorCode.INVALID_OPERATION, "drop_rows needs where=", 'e.g. where="age < 0"'
        )
    mask = evaluate(df, where)
    if not isinstance(mask, pd.Series) or not pdt.is_bool_dtype(mask.dtype) or len(mask) != len(df):
        raise EDAError(
            ErrorCode.INVALID_OPERATION,
            "where= must be a per-row condition",
            'e.g. where="age < 0"',
        )
    mask = mask.fillna(False).astype(bool)
    removed = int(mask.sum())
    _check_row_drop(df, removed)
    return df[~mask], Outcome("drop_rows", f"where {where}", rows_removed=removed)


def _outlier_mask(values: pd.Series, method: str, p: Operation) -> tuple[pd.Series, float, float]:
    """Rows outside the bounds, and the bounds themselves (for clipping)."""
    present = values.dropna().astype("float64")
    if method == "percentile":
        low_q, high_q = float(p.get("lower", 0.01)), float(p.get("upper", 0.99))
        low, high = float(present.quantile(low_q)), float(present.quantile(high_q))
    elif method == "iqr":
        k = float(p.get("threshold", OUTLIER_DEFAULTS["iqr"]))
        q1, q3 = float(present.quantile(0.25)), float(present.quantile(0.75))
        low, high = q1 - k * (q3 - q1), q3 + k * (q3 - q1)
    elif method == "zscore":
        k = float(p.get("threshold", OUTLIER_DEFAULTS["zscore"]))
        mean, sd = float(present.mean()), float(present.std())
        low, high = mean - k * sd, mean + k * sd
    elif method == "modified_zscore":
        k = float(p.get("threshold", OUTLIER_DEFAULTS["modified_zscore"]))
        median = float(present.median())
        mad = float(scipy_stats.median_abs_deviation(present))
        spread = k * mad / 0.6745
        low, high = median - spread, median + spread
    else:
        raise EDAError(
            ErrorCode.INVALID_OPERATION,
            f"unknown outlier method {method!r}",
            "use iqr, zscore, modified_zscore or percentile",
        )
    outside = (values < low) | (values > high)
    return outside.fillna(False).astype(bool), low, high


def remove_outliers(df: pd.DataFrame, p: Operation) -> tuple[pd.DataFrame, Outcome]:
    names = _columns(df, p)
    method, action = p.get("method", "iqr"), p.get("action", "flag")
    if action not in ("drop", "clip", "flag"):
        raise EDAError(
            ErrorCode.INVALID_OPERATION, "action must be drop, clip or flag", "e.g. action='clip'"
        )
    outcome = Outcome("remove_outliers", f"{', '.join(names)} ({method}, {action})")
    drop = pd.Series(False, index=df.index)
    for name in names:
        values = _numeric(df[name], name, "remove_outliers")
        mask, low, high = _outlier_mask(values, method, p)
        if action == "drop":
            drop |= mask
        elif action == "clip":
            df[name] = values.clip(low, high)
            outcome.cells_changed += int(mask.sum())
        else:
            flag = _new_name(df, f"{name}_outlier")
            df[flag] = mask.astype("int8")
            outcome.columns_added.append(flag)
    if action == "drop":
        outcome.rows_removed = int(drop.sum())
        _check_row_drop(df, outcome.rows_removed)
        df = df[~drop]
    return df, outcome


def _coerce_key(key: Any, series: pd.Series) -> Any:
    """JSON object keys are always strings; match them to the column's type."""
    if (
        isinstance(key, str)
        and pdt.is_numeric_dtype(series.dtype)
        and not pdt.is_bool_dtype(series.dtype)
    ):
        try:
            number = float(key)
        except ValueError:
            return key
        return int(number) if number.is_integer() else number
    return key


def replace_values(df: pd.DataFrame, p: Operation) -> tuple[pd.DataFrame, Outcome]:
    names = _columns(df, p)
    mapping = p.get("mapping")
    if not isinstance(mapping, dict) or not mapping:
        raise EDAError(
            ErrorCode.INVALID_OPERATION,
            "replace_values needs mapping=",
            'e.g. mapping={"-999": null} to turn a placeholder into missing',
        )
    outcome = Outcome("replace_values", ", ".join(names))
    for name in names:
        series = df[name]
        lookup = {_coerce_key(k, series): (np.nan if v is None else v) for k, v in mapping.items()}
        hit = series.isin(list(lookup))
        outcome.cells_changed += int(hit.sum())
        df[name] = series.where(~hit, series.map(lambda v, lookup=lookup: lookup.get(v, v)))
    return df, outcome


def merge_variants(df: pd.DataFrame, p: Operation) -> tuple[pd.DataFrame, Outcome]:
    """Map spellings that differ only in case or punctuation to the most common one."""
    names = _columns(df, p)
    outcome = Outcome("merge_variants", ", ".join(names))
    for name in names:
        series = df[name]
        if not _is_text(series):
            raise EDAError(
                ErrorCode.INVALID_OPERATION,
                f"{name} is not a text column",
                "use it on text columns",
            )
        groups = _variant_groups(series)
        mapping = {member: group[0] for group in groups for member in group[1:]}
        hit = series.isin(list(mapping))
        outcome.cells_changed += int(hit.sum())
        df[name] = series.where(~hit, series.map(lambda v, mapping=mapping: mapping.get(v, v)))
    return df, outcome


def _variant_groups(series: pd.Series) -> list[list[str]]:
    """Every variant group, most common spelling first (not just the top few)."""
    counts = series.dropna().astype(str).value_counts()
    keys = pd.Series(counts.index).str.casefold().str.replace(r"[\W_]+", "", regex=True)
    groups: list[list[str]] = []
    for key, members in pd.Series(counts.index).groupby(keys.to_numpy(), sort=True):
        if key and len(members) > 1:
            ordered = sorted(members, key=lambda m: (-int(counts[m]), m))
            groups.append(ordered)
    return groups


def rename_columns(df: pd.DataFrame, p: Operation) -> tuple[pd.DataFrame, Outcome]:
    mapping = p.get("mapping")
    if not isinstance(mapping, dict) or not mapping:
        raise EDAError(
            ErrorCode.INVALID_OPERATION,
            "rename_columns needs mapping=",
            'e.g. mapping={"old": "new"}',
        )
    labels = [str(c) for c in df.columns]
    for old in mapping:
        if old not in labels:
            raise ColumnNotFoundError(str(old), labels)
    after = [str(mapping.get(c, c)) for c in labels]
    if len(set(after)) != len(after):
        raise EDAError(
            ErrorCode.INVALID_OPERATION,
            "renaming would create duplicate column names",
            "choose distinct names",
        )
    renamed = df.rename(columns={old: str(new) for old, new in mapping.items()})
    return renamed, Outcome("rename_columns", ", ".join(f"{o}->{n}" for o, n in mapping.items()))


CAST_TARGETS = ("numeric", "float", "int", "str", "bool", "category", "datetime")


def cast_type(df: pd.DataFrame, p: Operation) -> tuple[pd.DataFrame, Outcome]:
    names = _columns(df, p)
    to = p.get("to")
    if to not in CAST_TARGETS:
        raise EDAError(
            ErrorCode.INVALID_OPERATION,
            f"to must be one of {', '.join(CAST_TARGETS)}",
            "e.g. to='numeric'",
        )
    outcome = Outcome("cast_type", f"{', '.join(names)} -> {to}")
    lost_total = 0
    for name in names:
        series = df[name]
        before = int(series.notna().sum())
        if to in ("numeric", "float", "int"):
            text = (
                series.astype("string").str.replace(",", "", regex=False)
                if _is_text(series)
                else series
            )
            converted: pd.Series = pd.to_numeric(text, errors="coerce")
            if to == "float":
                converted = converted.astype("float64")
            elif to == "int":
                converted = converted.round().astype("Int64")
        elif to == "datetime":
            converted = pd.to_datetime(series, errors="coerce", format=p.get("format", "mixed"))
        elif to == "bool":
            truthy = {"true", "yes", "y", "1", "t"}
            falsy = {"false", "no", "n", "0", "f"}
            text = series.astype("string").str.strip().str.lower()
            converted = pd.Series(pd.NA, index=series.index, dtype="boolean")
            converted[text.isin(truthy)] = True
            converted[text.isin(falsy)] = False
        elif to == "category":
            converted = series.astype("category")
        else:
            converted = series.astype("string")
        lost = before - int(converted.notna().sum())
        if before and lost / before > MAX_CONVERSION_LOSS:
            raise RefusedError(
                f"converting {name} to {to} would turn {lost / before:.0%} of its values missing",
                "check the column's contents with analyze_column first",
            )
        lost_total += lost
        outcome.cells_changed += before
        df[name] = converted
    if lost_total:
        outcome.note = f"{lost_total:,} value(s) could not be converted and are now missing"
    return df, outcome


def _text_columns(df: pd.DataFrame, p: Operation) -> list[str]:
    names = _columns(df, p, required=False) or [str(c) for c in df.columns if _is_text(df[c])]
    for name in names:
        if not _is_text(df[name]):
            raise EDAError(
                ErrorCode.INVALID_OPERATION,
                f"{name} is not a text column",
                "use it on text columns",
            )
    return names


def strip_whitespace(df: pd.DataFrame, p: Operation) -> tuple[pd.DataFrame, Outcome]:
    names = _text_columns(df, p)
    outcome = Outcome("strip_whitespace", ", ".join(names) or "no text columns")
    for name in names:
        series = df[name]
        cleaned = series.astype("string").str.strip()
        if p.get("collapse"):
            cleaned = cleaned.str.replace(r"\s+", " ", regex=True)
        changed = (cleaned != series.astype("string")).fillna(False)
        outcome.cells_changed += int(changed.sum())
        df[name] = series.where(~changed, cleaned)
    return df, outcome


def standardize_case(df: pd.DataFrame, p: Operation) -> tuple[pd.DataFrame, Outcome]:
    names = _text_columns(df, p)
    case = p.get("case", "lower")
    if case not in ("lower", "upper", "title"):
        raise EDAError(
            ErrorCode.INVALID_OPERATION, "case must be lower, upper or title", "e.g. case='lower'"
        )
    outcome = Outcome("standardize_case", f"{', '.join(names)} ({case})")
    for name in names:
        series = df[name]
        cased = getattr(series.astype("string").str, case)()
        changed = (cased != series.astype("string")).fillna(False)
        outcome.cells_changed += int(changed.sum())
        df[name] = series.where(~changed, cased)
    return df, outcome


def parse_dates(df: pd.DataFrame, p: Operation) -> tuple[pd.DataFrame, Outcome]:
    """``cast_type(to="datetime")`` under the name the findings use."""
    df, outcome = cast_type(df, {**p, "to": "datetime"})
    outcome.op, outcome.target = "parse_dates", outcome.target.removesuffix(" -> datetime")
    return df, outcome


# name -> (handler, parameters it accepts)
OPERATIONS: dict[str, tuple[Handler, frozenset[str]]] = {
    "fill_missing": (fill_missing, frozenset({"column", "columns", "method", "value", "flag"})),
    "flag_missing": (flag_missing, frozenset({"column", "columns"})),
    "drop_missing": (drop_missing, frozenset({"column", "columns", "how"})),
    "drop_duplicates": (drop_duplicates, frozenset({"column", "columns", "keep"})),
    "drop_columns": (drop_columns, frozenset({"column", "columns"})),
    "drop_rows": (drop_rows, frozenset({"where"})),
    "remove_outliers": (
        remove_outliers,
        frozenset({"column", "columns", "method", "action", "threshold", "lower", "upper"}),
    ),
    "replace_values": (replace_values, frozenset({"column", "columns", "mapping"})),
    "merge_variants": (merge_variants, frozenset({"column", "columns"})),
    "rename_columns": (rename_columns, frozenset({"mapping"})),
    "cast_type": (cast_type, frozenset({"column", "columns", "to", "format"})),
    "strip_whitespace": (strip_whitespace, frozenset({"column", "columns", "collapse"})),
    "standardize_case": (standardize_case, frozenset({"column", "columns", "case"})),
    "parse_dates": (parse_dates, frozenset({"column", "columns", "format"})),
}


def validate(
    operations: list[Operation], registry: dict[str, tuple[Handler, frozenset[str]]]
) -> None:
    """Reject the whole batch on the first malformed operation, before anything runs."""
    if not isinstance(operations, list) or not operations:
        raise EDAError(
            ErrorCode.INVALID_OPERATION,
            "operations must be a non-empty list",
            'e.g. operations=[{"op": "drop_duplicates"}]',
        )
    for index, operation in enumerate(operations):
        if not isinstance(operation, dict) or "op" not in operation:
            raise EDAError(
                ErrorCode.INVALID_OPERATION,
                f"operation {index + 1} has no op",
                'each operation is an object like {"op": "fill_missing", "column": "age"}',
            )
        name = operation["op"]
        if name not in registry:
            raise _invalid(
                index, str(name), "unknown operation", f"available: {', '.join(registry)}"
            )
        unknown = set(operation) - {"op"} - registry[name][1]
        if unknown:
            raise _invalid(
                index,
                name,
                f"unknown parameter(s) {', '.join(sorted(unknown))}",
                f"{name} takes: {', '.join(sorted(registry[name][1]))}",
            )


def _target(operation: Operation) -> str:
    raw = operation.get("columns", operation.get("column", operation.get("where", "")))
    return ", ".join(raw) if isinstance(raw, list) else str(raw)


def apply(
    df: pd.DataFrame,
    operations: list[Operation],
    registry: dict[str, tuple[Handler, frozenset[str]]] = OPERATIONS,
) -> tuple[pd.DataFrame, list[Outcome]]:
    """Run a batch on copies of *df*; the original is never touched.

    Each operation gets its own copy and is kept only if it completes, so an
    operation refused halfway -- the third of three columns over the limit --
    leaves no trace. Copies are shallow where copy-on-write makes that safe.
    """
    validate(operations, registry)
    work = df
    outcomes: list[Outcome] = []
    for index, operation in enumerate(operations):
        name = operation["op"]
        handler = registry[name][0]
        try:
            candidate, outcome = handler(work.copy(deep=not COPY_ON_WRITE), dict(operation))
        except RefusedError as refusal:
            outcomes.append(Outcome(name, _target(operation), refused=refusal))
            continue
        except EDAError as exc:
            # Errors abort the batch: nothing is committed, so name the step.
            exc.message = f"operation {index + 1} ({name}): {exc.message}"
            raise
        work = candidate
        outcomes.append(outcome)
    return work, outcomes


__all__ = ["OPERATIONS", "Outcome", "RefusedError", "apply", "validate"]
