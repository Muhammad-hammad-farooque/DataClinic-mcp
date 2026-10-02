"""Column classification, the orientation digest, and per-column statistics.

``orientation`` gives ``load_dataset`` enough to answer the obvious next
question in its first response, so the routine follow-up ``profile`` call
never happens. ``profile_frame`` computes the full per-column statistics that
``profile``, ``analyze_column`` and ``find_issues`` are built on.

Every statistic here is computed over the whole frame. Sampling would be
faster and is what makes competing servers wrong on sorted files.

Statistics are raw facts, not judgements: deciding that a skew of 2.3 is a
problem belongs to ``issues``. Floats are left unrounded here; the digest
layer rounds at the serialisation boundary.

See spec sections 7.1, 7.2, 10.4, 13.3 and A.3-A.10.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as pdt
from scipy import stats as scipy_stats

from eda_mcp.digest import Finding, Severity, round_sig
from eda_mcp.errors import ColumnNotFoundError

# A column with almost no repetition is a key, not a feature; one with almost
# no variation carries no signal. Both are reported rather than analysed.
IDENTIFIER_UNIQUE_RATIO = 0.99
CONSTANT_DOMINANCE = 0.99
HIGH_CARDINALITY = 50

# Outlier fences: Tukey's 1.5 IQR, and Iglewicz-Hoaglin's 3.5 for the
# modified z-score, which stays honest on the skewed data where z-scores lie.
IQR_FENCE = 1.5
MODIFIED_Z_CUTOFF = 3.5
RARE_SHARE = 0.01
TOP_VALUES = 5
EARLIEST_PLAUSIBLE = pd.Timestamp("1900-01-01")


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
    n = len(non_null)
    if n < 2:
        return False
    # n distinct integers spanning exactly n values are necessarily consecutive.
    # The span is checked first because it is cheap and rejects almost everything.
    if int(non_null.max()) - int(non_null.min()) != n - 1:
        return False
    return int(non_null.nunique()) == n


def _is_constant(non_null: pd.Series) -> bool:
    """True when one value covers at least CONSTANT_DOMINANCE of the rows.

    For orderable columns, a value covering more than half the rows must be
    the median, so comparing against the median alone is exact -- and avoids
    hashing a million distinct floats just to learn none of them dominates.
    """
    if pdt.is_datetime64_any_dtype(non_null) or (
        pdt.is_numeric_dtype(non_null) and not pdt.is_bool_dtype(non_null)
    ):
        share = float((non_null == non_null.median()).mean())
    else:
        share = float(non_null.value_counts().iloc[0]) / len(non_null)
    return share >= CONSTANT_DOMINANCE


def classify(series: pd.Series) -> ColumnKind:
    """Assign the kind that decides how a column is analysed.

    Order matters: emptiness and constancy are checked before dtype, because
    an all-null float column is not usefully "numeric".
    """
    non_null = series.dropna()
    if non_null.empty:
        return ColumnKind.EMPTY
    if _is_constant(non_null):
        return ColumnKind.CONSTANT

    if pdt.is_bool_dtype(series):
        return ColumnKind.BOOLEAN
    if pdt.is_datetime64_any_dtype(series):
        return ColumnKind.DATETIME

    if pdt.is_numeric_dtype(series):
        # Being unique is not enough: prices and measurements are often unique.
        # A surrogate key is unique *and* consecutive, which is a real signature.
        if pdt.is_integer_dtype(series) and _is_row_counter(non_null):
            return ColumnKind.IDENTIFIER
        return ColumnKind.NUMERIC

    n_unique = int(non_null.nunique())
    if n_unique / len(non_null) >= IDENTIFIER_UNIQUE_RATIO:
        return ColumnKind.IDENTIFIER
    if n_unique > HIGH_CARDINALITY:
        return ColumnKind.TEXT
    return ColumnKind.CATEGORICAL


def _near_unique(series: pd.Series) -> bool:
    non_null = series.dropna()
    return len(non_null) > 0 and non_null.nunique() / len(non_null) >= 0.5


def column_kinds(df: pd.DataFrame) -> dict[str, ColumnKind]:
    """Classify every column, judging uniqueness on distinct rows.

    Exact duplicate rows repeat every value in them, so a few hundred copies
    would push a genuine key below the identifier threshold -- hiding the
    key at exactly the moment its duplicates most need reporting.

    A full-row duplicate scan is the most expensive step in classification,
    so it runs only when some string column is near-unique and could be such
    a hidden key.
    """
    kinds = {str(c): classify(df[c]) for c in df.columns}
    suspects = [
        c
        for c in df.columns
        if (
            kinds[str(c)] is ColumnKind.TEXT
            or (kinds[str(c)] is ColumnKind.CATEGORICAL and df[c].count() <= 2 * HIGH_CARDINALITY)
        )
        and _near_unique(df[c])
    ]
    if suspects:
        repeated = df.duplicated()
        if repeated.any():
            distinct = df[~repeated]
            kinds.update({str(c): classify(distinct[c]) for c in suspects})
    return kinds


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


# --------------------------------------------------------------------------
# per-column statistics


@dataclass(slots=True)
class ColumnProfile:
    """Everything known about one column.

    ``error`` is set instead of ``stats`` when the column could not be
    profiled; the rest of the frame is still reported (spec section 13.3).
    """

    name: str
    kind: ColumnKind
    dtype: str
    count: int
    missing: int
    unique: int | None = None  # unknown when the column failed
    stats: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def missing_pct(self) -> float:
        total = self.count + self.missing
        return self.missing / total * 100 if total else 0.0

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "kind": self.kind.value,
            "dtype": self.dtype,
            "count": self.count,
            "missing": self.missing,
            "missing_pct": self.missing_pct,
            "unique": self.unique,
            **self.stats,
        }
        if self.error:
            out["failed"] = self.error
        return out

    def digest(self) -> dict[str, Any]:
        """The response form: dtype and zero missingness are noise to the reader."""
        out = self.to_dict()
        del out["dtype"]
        if not self.missing:
            del out["missing"], out["missing_pct"]
        return out


def _rank(counts: pd.Series) -> pd.Series:
    """Order counts by frequency, ties broken by value, so row order cannot leak in."""
    ranking = pd.DataFrame({"n": counts.to_numpy(), "key": counts.index.map(str)})
    order = ranking.sort_values(["n", "key"], ascending=[False, True], kind="mergesort").index
    return counts.iloc[order.to_numpy()]


def _ranked_counts(series: pd.Series) -> pd.Series:
    return _rank(series.value_counts())


def _quantile(ordered: np.ndarray[Any, np.dtype[np.float64]], p: float) -> float:
    """Linear-interpolated quantile of a sorted array, as pandas computes it."""
    position = p * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower))


def numeric_stats(non_null: pd.Series) -> dict[str, Any]:
    # One sort answers the quantiles, extremes, distinct count and every
    # threshold count below by binary search. It also makes the sums -- and
    # so the mean -- independent of row order down to the last bit.
    ordered = np.sort(non_null.to_numpy(dtype="float64"))
    n = len(ordered)
    q1, median, q3 = (_quantile(ordered, p) for p in (0.25, 0.5, 0.75))
    iqr = q3 - q1
    below_zero = int(np.searchsorted(ordered, 0.0, side="left"))
    # Shape is undefined without spread; say so rather than emit nan.
    has_spread = ordered[0] != ordered[-1]

    stats: dict[str, Any] = {
        "unique": 1 + int(np.count_nonzero(np.diff(ordered))),
        "mean": float(ordered.mean()),
        "std": float(ordered.std(ddof=1)) if n > 1 else None,
        "min": float(ordered[0]),
        "q1": q1,
        "median": median,
        "q3": q3,
        "max": float(ordered[-1]),
        # bias-corrected, matching pandas; excess kurtosis is 0 for a normal
        "skew": float(scipy_stats.skew(ordered, bias=False)) if n > 2 and has_spread else None,
        "excess_kurtosis": (
            float(scipy_stats.kurtosis(ordered, bias=False)) if n > 3 and has_spread else None
        ),
        "zeros": int(np.searchsorted(ordered, 0.0, side="right")) - below_zero,
        "negatives": below_zero,
        "integral": bool(np.all(np.mod(ordered, 1) == 0)),
    }

    # A zero spread makes both fences meaningless: every value off the median
    # would be "an outlier". Report nothing rather than something misleading.
    if iqr > 0:
        low, high = q1 - IQR_FENCE * iqr, q3 + IQR_FENCE * iqr
        outside = (
            int(np.searchsorted(ordered, low, side="left"))
            + n
            - int(np.searchsorted(ordered, high, side="right"))
        )
        stats["outliers_iqr"] = outside
        stats["iqr_bounds"] = [low, high]

    deviation = np.abs(ordered - median)
    mad = float(np.median(deviation))
    if mad > 0:
        # A tiny MAD overflows the score to inf, which is still correctly "an outlier".
        with np.errstate(over="ignore"):
            scores = 0.6745 * deviation / mad
        stats["outliers_modified_z"] = int(np.count_nonzero(scores > MODIFIED_Z_CUTOFF))
    return stats


def boolean_stats(non_null: pd.Series) -> dict[str, Any]:
    true = int(non_null.astype(bool).sum())
    return {
        "unique": int(non_null.nunique()),
        "true": true,
        "true_pct": true / len(non_null) * 100,
    }


def datetime_stats(non_null: pd.Series, now: pd.Timestamp | None = None) -> dict[str, Any]:
    tz = non_null.dt.tz
    if now is None:
        now = pd.Timestamp.now(tz=tz)
    elif tz is not None and now.tzinfo is None:
        now = now.tz_localize(tz)
    earliest = EARLIEST_PLAUSIBLE.tz_localize(tz) if tz is not None else EARLIEST_PLAUSIBLE

    lo, hi = non_null.min(), non_null.max()
    stats: dict[str, Any] = {
        "min": lo.isoformat(),
        "max": hi.isoformat(),
        "span_days": (hi - lo) / pd.Timedelta(days=1),
        "future": int((non_null > now).sum()),
        "before_1900": int((non_null < earliest).sum()),
        "sorted": bool(non_null.is_monotonic_increasing),
    }

    distinct = pd.Series(np.sort(non_null.unique()))
    stats["unique"] = len(distinct)
    if len(distinct) > 1:
        gaps = distinct.diff().dropna().dt.total_seconds() / 86_400
        # The median gap is the series' granularity; the largest is where a
        # supposedly complete series has a hole.
        stats["median_gap_days"] = float(gaps.median())
        stats["max_gap_days"] = float(gaps.max())
    return stats


def string_checks(counts: pd.Series) -> dict[str, Any]:
    """Defects that hide inside text: padding, stray numbers, mixed types.

    Takes value counts rather than rows: each check runs once per distinct
    value and is weighted by its count. The result is exact, and a million-row
    column of six countries costs six string operations, not a million.
    """
    stats: dict[str, Any] = {}
    weights = counts.to_numpy()

    if counts.index.dtype == object:
        type_names = [type(v).__name__ for v in counts.index]
        types = pd.Series(weights).groupby(type_names).sum()
        if len(types) > 1:
            stats["mixed_types"] = {str(k): int(v) for k, v in types.sort_index().items()}

    text = pd.Series(counts.index.map(str), dtype=object)
    lengths = text.str.len().to_numpy()
    stats["length"] = {
        "min": int(lengths.min()),
        "median": float(np.median(np.repeat(lengths, weights))),
        "max": int(lengths.max()),
    }
    stripped = text.str.strip()
    stats["whitespace_padded"] = int(weights[(text != stripped).to_numpy()].sum())
    stats["blank"] = int(weights[(stripped == "").to_numpy()].sum())

    numeric = pd.to_numeric(text.str.replace(",", "", regex=False), errors="coerce")
    stats["numeric_like"] = int(weights[numeric.notna().to_numpy()].sum())
    return stats


def spelling_variants(counts: pd.Series) -> dict[str, Any]:
    """Group values that differ only in case, spacing or punctuation.

    The classic case is ``USA`` / ``usa`` / ``U.S.A.``: three categories to
    pandas, one to anyone reading the data. Takes value counts, so the cost
    tracks cardinality rather than row count.
    """
    counts = _rank(counts.groupby(counts.index.map(str)).sum())
    raw = pd.Series(counts.index, dtype="string")
    keys = raw.str.casefold().str.replace(r"[\W_]+", "", regex=True)

    groups: list[list[str]] = []
    affected = 0
    for key, members in raw.groupby(keys.to_numpy(), sort=False):
        if not key or len(members) < 2:
            continue
        # members keep the frequency order of counts, so the first is dominant
        names = [str(m) for m in members]
        groups.append(names)
        affected += int(counts.loc[names[1:]].sum())

    if not groups:
        return {}
    groups.sort(key=lambda g: (-int(counts.loc[g].sum()), g[0]))
    return {
        "variants": groups[:TOP_VALUES],
        "variant_groups": len(groups),
        "variant_rows": affected,
    }


def categorical_stats(non_null: pd.Series) -> dict[str, Any]:
    counts = _ranked_counts(non_null)
    shares = counts / len(non_null)
    rare = shares[shares < RARE_SHARE]

    stats: dict[str, Any] = {
        "unique": len(counts),
        "top": {str(k): int(v) for k, v in counts.head(TOP_VALUES).items()},
        "top_share_pct": float(shares.iloc[0]) * 100,
    }
    if len(rare):
        stats["rare"] = len(rare)
        stats["rare_rows"] = int(counts[rare.index].sum())
    stats.update(spelling_variants(counts))
    stats.update(string_checks(counts))
    return stats


def text_stats(non_null: pd.Series) -> dict[str, Any]:
    counts = non_null.value_counts()
    stats: dict[str, Any] = {"unique": len(counts)}
    stats.update(string_checks(counts))
    stats.update(spelling_variants(counts))
    return stats


def identifier_stats(non_null: pd.Series) -> dict[str, Any]:
    # A key that repeats is a broken key, and every join on it fans out.
    repeats = int(non_null.duplicated().sum())
    return {"unique": len(non_null) - repeats, "duplicate_keys": repeats}


def constant_stats(non_null: pd.Series) -> dict[str, Any]:
    counts = _ranked_counts(non_null)
    return {
        "unique": len(counts),
        "value": str(counts.index[0]),
        "dominance_pct": int(counts.iloc[0]) / len(non_null) * 100,
    }


def column_stats(
    non_null: pd.Series, kind: ColumnKind, now: pd.Timestamp | None = None
) -> dict[str, Any]:
    """Type-adapted statistics for one column's non-null values.

    Every kind reports ``unique`` from work it already does, sparing a
    separate hashing pass over the column.
    """
    if kind is ColumnKind.EMPTY:
        return {"unique": 0}
    if kind is ColumnKind.NUMERIC:
        return numeric_stats(non_null)
    if kind is ColumnKind.BOOLEAN:
        return boolean_stats(non_null)
    if kind is ColumnKind.DATETIME:
        return datetime_stats(non_null, now)
    if kind is ColumnKind.CATEGORICAL:
        return categorical_stats(non_null)
    if kind is ColumnKind.TEXT:
        return text_stats(non_null)
    if kind is ColumnKind.IDENTIFIER:
        return identifier_stats(non_null)
    return constant_stats(non_null)


def profile_column(
    series: pd.Series, kind: ColumnKind | None = None, now: pd.Timestamp | None = None
) -> ColumnProfile:
    kind = kind or classify(series)
    # Null detection on string columns is a full scan; do it once.
    non_null = series.dropna()
    profile = ColumnProfile(
        name=str(series.name),
        kind=kind,
        dtype=str(series.dtype),
        count=len(non_null),
        missing=len(series) - len(non_null),
    )
    try:
        stats = column_stats(non_null, kind, now)
        profile.unique = stats.pop("unique")
        profile.stats = stats
    except Exception as exc:
        # One unreadable column must not cost the caller the other forty-nine.
        profile.error = f"{type(exc).__name__} while computing {kind.value} statistics"
    return profile


def profile_frame(
    df: pd.DataFrame,
    kinds: dict[str, ColumnKind] | None = None,
    columns: list[str] | None = None,
    now: pd.Timestamp | None = None,
) -> list[ColumnProfile]:
    """Profile the requested columns, in frame order, over every row."""
    # Callers name columns as strings; the frame may hold ints from a headerless file.
    labels = {str(c): c for c in df.columns}
    if columns:
        for column in columns:
            if column not in labels:
                raise ColumnNotFoundError(column, list(labels))
    selected = [name for name in labels if not columns or name in columns]
    kinds = kinds or {}
    return [
        profile_column(df[labels[name]].rename(name), kinds.get(name), now) for name in selected
    ]
