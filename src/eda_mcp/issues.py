"""Quality judgements: turning column statistics into ranked findings.

``profiling`` reports facts -- a skew of 3.9, 52 negative values. This module
decides which facts are problems, how serious each is, and what to do about
it. Keeping the two apart means the thresholds live in one place and the
statistics stay reusable by tools that judge differently.

Each finding carries exactly one recommendation, not a menu (spec 10.6).

See spec sections 7.2, 10.6 and A.3-A.10.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as pdt
from scipy import stats as scipy_stats

from eda_mcp.digest import Finding, Severity
from eda_mcp.profiling import ColumnKind, ColumnProfile, duplicated

# Missingness bands, matching the orientation digest in load_dataset.
MISSING_HIGH = 0.6
MISSING_MEDIUM = 0.2
MISSING_LOW = 0.05

# |skew| above 1 is the conventional line for "strongly skewed" (A.7); below
# 0.5 a distribution is close enough to symmetric.
SKEW_LIMIT = 1.0
SKEW_RESOLVED = 0.5
# Simplest first: parameter-free transforms before fitted ones.
TRANSFORM_PREFERENCE = ("log1p", "sqrt", "square", "boxcox", "yeojohnson")
# Outliers below this share are ordinary tails, not worth a finding alone.
OUTLIER_SHARE = 0.01
# Negatives in an otherwise non-negative column are suspect only while rare;
# a column that is half negative is simply signed.
STRAY_NEGATIVE_SHARE = 0.05

# A text column that parses as numbers or dates this often is mistyped. The
# loader converts at 95%, so this catches what it deliberately left alone.
NUMERIC_TEXT_SHARE = 0.9
DATE_TEXT_SHARE = 0.9

# A sentinel code is suspect when it is common, or sits outside the IQR fences
# where real values rarely are.
SENTINEL_SHARE = 0.05

# Missingness is "related" to another column when the association is at least
# a small-to-medium effect and unlikely to be chance. Effect size decides, not
# the p-value alone: on millions of rows everything is "significant".
RELATION_EFFECT = 0.2
RELATION_P = 0.01
RELATION_MIN_ROWS = 30
# Cells per block when scanning numeric columns, bounding peak memory.
RELATION_BLOCK_CELLS = 8_000_000

# One-hot stays reasonable up to about this many levels.
ONE_HOT_LEVELS = 15
SHORT_LABEL_CHARS = 30


def _missing(profile: ColumnProfile, relation: str | None) -> Finding | None:
    total = profile.count + profile.missing
    if not total or not profile.missing or profile.kind is ColumnKind.EMPTY:
        return None
    share = profile.missing / total
    if share >= MISSING_HIGH:
        severity, advice = Severity.HIGH, "add a missingness flag rather than imputing"
    elif share >= MISSING_MEDIUM:
        severity, advice = Severity.MEDIUM, "decide between imputation and a flag"
    elif share >= MISSING_LOW:
        fill = "median" if profile.kind is ColumnKind.NUMERIC else "mode"
        severity, advice = Severity.LOW, f"impute with the {fill}"
    else:
        return None

    message = f"{share:.0%} missing"
    if relation:
        # Gaps that follow another column are not random; imputing hides the
        # pattern a model could use, so the advice is always a flag (A.4.3).
        message += f", {relation}"
        advice = "add a missingness flag; the gaps are not random"
        if severity is Severity.LOW:
            severity = Severity.MEDIUM
    return Finding(
        severity,
        message,
        column=profile.name,
        recommendation=advice,
        affected_rows=profile.missing,
    )


def _sentinels(profile: ColumnProfile) -> tuple[Finding | None, int]:
    """Placeholder codes such as -999 stored as if they were measurements.

    Returns the finding and how many negative values it accounts for, so the
    generic negative-value check does not report the same rows twice.
    """
    stats, n = profile.stats, profile.count
    low, high = stats.get("iqr_bounds", (-math.inf, math.inf))
    codes: dict[int, int] = {}
    for value, count in stats.get("sentinels", {}).items():
        if count < 2:
            continue
        if value == -1:
            # -1 is a code only when it is the column's sole negative value.
            suspect = count == stats.get("negatives", 0)
        else:
            suspect = count >= SENTINEL_SHARE * n or not low <= value <= high
        if suspect:
            codes[value] = count
    if not codes:
        return None, 0

    listed = ", ".join(str(v) for v in sorted(codes))
    finding = Finding(
        Severity.MEDIUM,
        f"placeholder code(s) {listed} stored as values",
        column=profile.name,
        recommendation="replace with missing",
        affected_rows=sum(codes.values()),
    )
    return finding, sum(c for v, c in codes.items() if v < 0)


def _simplest_transform(transforms: dict[str, float]) -> str:
    """The simplest transform that removes the skew, else the most effective.

    Measured, not assumed. A log needs no fitted parameter to store and
    reapply, so it beats a Box-Cox that is only marginally more symmetric.
    """
    for name in TRANSFORM_PREFERENCE:
        if name in transforms and abs(transforms[name]) < SKEW_RESOLVED:
            return name
    return min(transforms, key=lambda t: (abs(transforms[t]), t))


def _numeric(profile: ColumnProfile) -> list[Finding]:
    stats, name, n = profile.stats, profile.name, profile.count
    findings: list[Finding] = []

    sentinel, explained = _sentinels(profile)
    if sentinel:
        findings.append(sentinel)

    negatives = stats.get("negatives", 0) - explained
    if 0 < negatives <= STRAY_NEGATIVE_SHARE * n:
        findings.append(
            Finding(
                Severity.MEDIUM,
                f"{negatives:,} negative value(s) in an otherwise non-negative column",
                column=name,
                recommendation="treat as invalid and set to missing",
                affected_rows=negatives,
            )
        )

    # A two-valued column is a flag; its "skew" is just class imbalance,
    # which analyze_target reports properly.
    skew = stats.get("skew")
    outliers = stats.get("outliers_iqr", 0)
    if skew is not None and abs(skew) > SKEW_LIMIT and (profile.unique or 0) > 2:
        direction = "right" if skew > 0 else "left"
        message = f"{direction}-skewed ({skew:.2g})"
        if outliers:
            message += f", {outliers:,} outliers beyond 1.5 IQR"
        transforms = stats.get("transforms")
        if transforms:
            best = _simplest_transform(transforms)
            after = round(transforms[best], 2) + 0.0  # + 0.0 turns -0.0 into 0.0
            advice = f"apply {best} (skew {skew:.2g} -> {after:.2f})"
        elif skew > 0 and stats.get("min", -1) >= 0:
            advice = "apply log1p before modelling"
        else:
            advice = "use robust scaling (median/IQR)"
        findings.append(
            Finding(
                Severity.MEDIUM,
                message,
                column=name,
                recommendation=advice,
                affected_rows=outliers or None,
            )
        )
    elif outliers > OUTLIER_SHARE * n:
        findings.append(
            Finding(
                Severity.LOW,
                f"{outliers:,} outliers beyond 1.5 IQR",
                column=name,
                recommendation="verify they are real before removing any",
                affected_rows=outliers,
            )
        )
    return findings


def _textual(profile: ColumnProfile) -> list[Finding]:
    stats, name, n = profile.stats, profile.name, profile.count
    findings: list[Finding] = []

    if stats.get("placeholders"):
        examples = ", ".join(repr(v) for v in stats["placeholder_values"])
        findings.append(
            Finding(
                Severity.MEDIUM,
                f"null placeholders stored as text ({examples})",
                column=name,
                recommendation="treat as missing",
                affected_rows=stats["placeholders"],
            )
        )
    date_like = stats.get("date_like", 0)
    if n and date_like >= DATE_TEXT_SHARE * n:
        findings.append(
            Finding(
                Severity.MEDIUM,
                "dates stored as text",
                column=name,
                recommendation="parse with to_datetime; unparseable values become missing",
                affected_rows=n - date_like or None,
            )
        )
    if stats.get("variant_groups"):
        example = " / ".join(stats["variants"][0])
        findings.append(
            Finding(
                Severity.MEDIUM,
                f"{stats['variant_groups']} value(s) spelled several ways ({example})",
                column=name,
                recommendation="map each group to its most common spelling",
                affected_rows=stats["variant_rows"],
            )
        )
    if stats.get("mixed_types"):
        kinds = ", ".join(stats["mixed_types"])
        findings.append(
            Finding(
                Severity.MEDIUM,
                f"mixed value types ({kinds})",
                column=name,
                recommendation="cast to one type",
            )
        )
    numeric_like = stats.get("numeric_like", 0)
    if n and numeric_like >= NUMERIC_TEXT_SHARE * n:
        findings.append(
            Finding(
                Severity.MEDIUM,
                "numbers stored as text",
                column=name,
                recommendation="convert with to_numeric; non-numeric leftovers become missing",
                affected_rows=n - numeric_like or None,
            )
        )
    if stats.get("blank"):
        findings.append(
            Finding(
                Severity.MEDIUM,
                "blank strings posing as values",
                column=name,
                recommendation="treat as missing",
                affected_rows=stats["blank"],
            )
        )
    if stats.get("whitespace_padded"):
        findings.append(
            Finding(
                Severity.LOW,
                "leading or trailing whitespace",
                column=name,
                recommendation="strip whitespace",
                affected_rows=stats["whitespace_padded"],
            )
        )
    if stats.get("rare"):
        findings.append(
            Finding(
                Severity.LOW,
                f"{stats['rare']} rare categor(ies) under 1%",
                column=name,
                recommendation="group into 'Other' before encoding",
                affected_rows=stats["rare_rows"],
            )
        )
    return findings


def _datetime(profile: ColumnProfile) -> list[Finding]:
    stats, name = profile.stats, profile.name
    findings: list[Finding] = []
    if stats.get("future"):
        findings.append(
            Finding(
                Severity.MEDIUM,
                "dates in the future",
                column=name,
                recommendation="treat as entry errors and set to missing",
                affected_rows=stats["future"],
            )
        )
    if stats.get("before_1900"):
        findings.append(
            Finding(
                Severity.MEDIUM,
                "dates before 1900, likely placeholders",
                column=name,
                recommendation="set to missing",
                affected_rows=stats["before_1900"],
            )
        )
    return findings


def column_findings(profile: ColumnProfile, missing_relation: str | None = None) -> list[Finding]:
    """Everything wrong with one column, each with a single recommended fix.

    *missing_relation* describes what the column's gaps depend on, from
    ``missingness_relations``; only ``find_issues`` pays to compute it.
    """
    name = profile.name
    if profile.error:
        return [
            Finding(
                Severity.LOW,
                f"not profiled ({profile.error})",
                column=name,
                recommendation="inspect with analyze_column",
            )
        ]

    kind = profile.kind
    if kind is ColumnKind.EMPTY:
        return [Finding(Severity.HIGH, "entirely empty", column=name, recommendation="drop")]

    findings: list[Finding] = []
    missing = _missing(profile, missing_relation)
    if missing:
        findings.append(missing)

    if kind is ColumnKind.CONSTANT:
        findings.append(
            Finding(
                Severity.MEDIUM,
                "single value throughout",
                column=name,
                recommendation="drop; it carries no signal",
            )
        )
    elif kind is ColumnKind.IDENTIFIER:
        repeats = profile.stats.get("duplicate_keys", 0)
        if repeats:
            findings.append(
                Finding(
                    Severity.HIGH,
                    "identifier repeats",
                    column=name,
                    recommendation="deduplicate before any join on it",
                    affected_rows=repeats,
                )
            )
        else:
            findings.append(
                Finding(
                    Severity.LOW,
                    "unique per row",
                    column=name,
                    recommendation="treat as an identifier, not a feature",
                )
            )
    elif kind is ColumnKind.NUMERIC:
        findings.extend(_numeric(profile))
    elif kind in (ColumnKind.CATEGORICAL, ColumnKind.TEXT):
        findings.extend(_textual(profile))
    elif kind is ColumnKind.DATETIME:
        findings.extend(_datetime(profile))
    return findings


def frame_findings(df: pd.DataFrame) -> list[Finding]:
    """Problems that belong to the table rather than any one column."""
    rows = len(df)
    if not rows:
        return [Finding(Severity.HIGH, "the table has no rows")]
    duplicates = int(duplicated(df).sum())
    if not duplicates:
        return []
    return [
        Finding(
            Severity.MEDIUM,
            f"{duplicates / rows:.0%} of rows are exact duplicates",
            recommendation="drop_duplicates unless repetition is meaningful",
            affected_rows=duplicates,
        )
    ]


# --------------------------------------------------------------------------
# cross-column checks

Relation = tuple[float, str]


def _as_float(series: pd.Series) -> np.ndarray[Any, np.dtype[np.float64]]:
    if pdt.is_datetime64_any_dtype(series):
        # Days since the earliest value: works for timezone-aware columns,
        # where an int64 cast does not, and NaT becomes NaN.
        series = (series - series.min()) / pd.Timedelta(days=1)
    return series.to_numpy(dtype="float64", na_value=np.nan)


def _significant(r: float, n: int) -> bool:
    t = r * math.sqrt((n - 2) / max(1 - r * r, 1e-12))
    return bool(2 * scipy_stats.t.sf(abs(t), n - 2) < RELATION_P)


def _numeric_relations(
    df: pd.DataFrame, columns: list[str], flags: dict[str, np.ndarray[Any, Any]]
) -> dict[str, Relation]:
    """Strongest point-biserial link between each flag and a numeric column.

    Correlation with pairwise deletion, computed from sums so every target
    costs two matrix-vector products rather than a pass per column pair.
    Columns are scanned in blocks to bound memory on long tables.
    """
    labels = {str(c): c for c in df.columns}
    best: dict[str, Relation] = {}
    step = max(1, RELATION_BLOCK_CELLS // max(len(df), 1))
    for start in range(0, len(columns), step):
        names = columns[start : start + step]
        values = np.column_stack([_as_float(df[labels[c]]) for c in names])
        present = ~np.isnan(values)
        rows = present.sum(axis=0)
        # Centring on each column's own mean makes sum(x) zero over present
        # rows, which removes the cancellation-prone term from the covariance.
        with np.errstate(invalid="ignore"):
            centred = np.where(present, values - np.nanmean(values, axis=0), 0.0)
        del values
        sxx = np.einsum("ij,ij->j", centred, centred)
        present_f = present.astype("float64")

        for target, flag in flags.items():
            share = (flag @ present_f) / np.maximum(rows, 1)
            with np.errstate(divide="ignore", invalid="ignore"):
                r = (flag @ centred) / np.sqrt(sxx * rows * share * (1 - share))
            for j, other in enumerate(names):
                n = int(rows[j])
                value = float(r[j])
                if other == target or not math.isfinite(value) or n < 2 * RELATION_MIN_ROWS:
                    continue
                if abs(value) < RELATION_EFFECT or not _significant(value, n):
                    continue
                if target not in best or abs(value) > best[target][0]:
                    side = "higher" if value > 0 else "lower"
                    best[target] = (abs(value), f"more often where {other} is {side}")
    return best


def _group_relation(
    mask: np.ndarray[Any, Any], other: str, codes: np.ndarray[Any, Any], uniques: pd.Index
) -> Relation | None:
    """Cramer's V between missingness and a categorical column."""
    valid = codes >= 0
    groups = codes[valid]
    missing = np.bincount(groups, weights=mask[valid])
    totals = np.bincount(groups).astype("float64")
    present = np.flatnonzero(totals > 0)
    if len(present) < 2:
        return None
    table = np.vstack([missing[present], totals[present] - missing[present]])
    if (table.sum(axis=1) == 0).any():
        return None
    chi2, p, _, _ = scipy_stats.chi2_contingency(table, correction=False)
    # With two rows (missing / present), V reduces to sqrt(chi2 / n).
    v = math.sqrt(chi2 / table.sum())
    if v < RELATION_EFFECT or p >= RELATION_P:
        return None
    rates = missing[present] / totals[present]
    worst = uniques[present[int(np.argmax(rates))]]
    return v, f"concentrated where {other}={worst}"


def missingness_relations(
    df: pd.DataFrame, kinds: dict[str, ColumnKind], targets: list[str]
) -> dict[str, str]:
    """For each column in *targets*, what its missingness depends on.

    Missing values that cluster by another column are not missing at random
    (A.4.2). Numeric and date columns are tested by point-biserial
    correlation, categorical ones by Cramer's V. Both lie on a 0-1 scale, so
    the strongest relation wins whatever its type.
    """
    labels = {str(c): c for c in df.columns}
    flags: dict[str, np.ndarray[Any, Any]] = {}
    for target in targets:
        flag = df[labels[target]].isna().to_numpy(dtype="float64")
        n_missing = int(flag.sum())
        if min(n_missing, len(flag) - n_missing) >= RELATION_MIN_ROWS:
            flags[target] = flag
    if not flags:
        return {}

    ordered = [n for n, k in kinds.items() if k in (ColumnKind.NUMERIC, ColumnKind.DATETIME)]
    grouped = [n for n, k in kinds.items() if k in (ColumnKind.CATEGORICAL, ColumnKind.BOOLEAN)]
    best = _numeric_relations(df, ordered, flags)

    for other in grouped:
        # Factorise once; every target reuses the codes.
        codes, uniques = pd.factorize(df[labels[other]])
        for target, flag in flags.items():
            if other == target:
                continue
            found = _group_relation(flag, other, codes, pd.Index(uniques))
            if found and (target not in best or found[0] > best[target][0]):
                best[target] = found
    return {target: relation[1] for target, relation in best.items()}


def _duplicate_columns(df: pd.DataFrame, profiles: list[ColumnProfile]) -> list[Finding]:
    """Columns holding exactly the same values under different names.

    Identical columns have identical profiles, so only columns whose profiles
    match are compared value by value -- usually none at all.
    """
    labels = {str(c): c for c in df.columns}
    groups: dict[str, list[str]] = {}
    for p in profiles:
        # Empty and constant columns are already reported, and trivially equal.
        if p.error or p.kind in (ColumnKind.EMPTY, ColumnKind.CONSTANT):
            continue
        signature = repr((p.kind, p.count, p.missing, p.unique, sorted(p.stats.items())))
        groups.setdefault(signature, []).append(p.name)

    findings: list[Finding] = []
    for names in groups.values():
        first = names[0]
        for other in names[1:]:
            if df[labels[first]].equals(df[labels[other]]):
                findings.append(
                    Finding(
                        Severity.MEDIUM,
                        f"identical to {first}",
                        column=other,
                        recommendation="drop it",
                    )
                )
    return findings


def _repeated_records(df: pd.DataFrame, kinds: dict[str, ColumnKind]) -> Finding | None:
    """Rows that match apart from their key: one record entered twice."""
    keys = [c for c in df.columns if kinds.get(str(c)) is ColumnKind.IDENTIFIER]
    rest = [c for c in df.columns if c not in keys]
    if not keys or not rest:
        return None
    extra = int(duplicated(df, subset=rest).sum()) - int(duplicated(df).sum())
    if extra <= 0:
        return None
    named = ", ".join(str(k) for k in keys)
    return Finding(
        Severity.MEDIUM,
        f"rows repeat apart from their identifier ({named})",
        recommendation="drop duplicates on the non-key columns, keeping the first",
        affected_rows=extra,
    )


def find_issues(
    df: pd.DataFrame, kinds: dict[str, ColumnKind], profiles: list[ColumnProfile]
) -> list[Finding]:
    """Every problem in the table: per column, across columns, and whole-table."""
    findings = frame_findings(df)
    rows = len(df)
    if not rows:
        return findings

    targets = [
        p.name
        for p in profiles
        if p.kind is not ColumnKind.EMPTY and p.missing >= MISSING_LOW * rows
    ]
    relations = missingness_relations(df, kinds, targets)

    for p in profiles:
        findings.extend(column_findings(p, relations.get(p.name)))
    findings.extend(_duplicate_columns(df, profiles))
    repeated = _repeated_records(df, kinds)
    if repeated:
        findings.append(repeated)
    return findings


def encoding_advice(profile: ColumnProfile) -> str | None:
    """How to turn a categorical column into model features (A.7, A.19)."""
    levels = profile.unique or 0
    if profile.kind is ColumnKind.BOOLEAN or (
        profile.kind is ColumnKind.CATEGORICAL and levels <= 2
    ):
        return "binary 0/1"
    if profile.kind is ColumnKind.CATEGORICAL:
        if levels <= ONE_HOT_LEVELS:
            return f"one-hot ({levels} columns)"
        return f"frequency or target encoding; one-hot would add {levels} columns"
    if profile.kind is ColumnKind.TEXT:
        # Short repeated labels are a high-cardinality category; long ones are prose.
        if profile.stats.get("length", {}).get("median", 0) <= SHORT_LABEL_CHARS:
            return f"frequency or target encoding; one-hot would add {levels} columns"
        return "free text: derive features (length, keywords) or embed; do not one-hot"
    return None


def needs_attention(findings: list[Finding]) -> bool:
    """True when any finding is serious enough to show the column in full."""
    return any(f.severity in (Severity.HIGH, Severity.MEDIUM) for f in findings)


__all__ = [
    "column_findings",
    "encoding_advice",
    "find_issues",
    "frame_findings",
    "missingness_relations",
    "needs_attention",
]
