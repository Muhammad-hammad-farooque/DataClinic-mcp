"""Quality judgements: turning column statistics into ranked findings.

``profiling`` reports facts -- a skew of 3.9, 52 negative values. This module
decides which facts are problems, how serious each is, and what to do about
it. Keeping the two apart means the thresholds live in one place and the
statistics stay reusable by tools that judge differently.

Each finding carries exactly one recommendation, not a menu (spec 10.6).

See spec sections 7.2, 10.6 and A.3-A.10.
"""

from __future__ import annotations

import pandas as pd

from eda_mcp.digest import Finding, Severity
from eda_mcp.profiling import ColumnKind, ColumnProfile

# Missingness bands, matching the orientation digest in load_dataset.
MISSING_HIGH = 0.6
MISSING_MEDIUM = 0.2
MISSING_LOW = 0.05

# |skew| above 1 is the conventional line for "strongly skewed" (A.7).
SKEW_LIMIT = 1.0
# Outliers below this share are ordinary tails, not worth a finding alone.
OUTLIER_SHARE = 0.01
# Negatives in an otherwise non-negative column are suspect only while rare;
# a column that is half negative is simply signed.
STRAY_NEGATIVE_SHARE = 0.05
# A text column that parses as numbers this often is numbers stored as text.
NUMERIC_TEXT_SHARE = 0.9


def _missing(profile: ColumnProfile) -> Finding | None:
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
    return Finding(
        severity,
        f"{share:.0%} missing",
        column=profile.name,
        recommendation=advice,
        affected_rows=profile.missing,
    )


def _numeric(profile: ColumnProfile) -> list[Finding]:
    stats, name, n = profile.stats, profile.name, profile.count
    findings: list[Finding] = []

    negatives = stats.get("negatives", 0)
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
        if skew > 0 and stats.get("min", -1) >= 0:
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


def column_findings(profile: ColumnProfile) -> list[Finding]:
    """Everything wrong with one column, each with a single recommended fix."""
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
    missing = _missing(profile)
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
    duplicates = int(df.duplicated().sum())
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


def needs_attention(findings: list[Finding]) -> bool:
    """True when any finding is serious enough to show the column in full."""
    return any(f.severity in (Severity.HIGH, Severity.MEDIUM) for f in findings)


__all__ = ["column_findings", "frame_findings", "needs_attention"]
