"""Target analysis: what is being predicted, how well, and what gives it away.

The most expensive mistake in modelling is a feature that encodes the answer:
the model scores perfectly in validation and fails in production. Leakage
is therefore checked before anything else and reported at HIGH severity.

Feature strength uses the shared 0-1 scale of ``relations`` so the ranking
reads the same as ``check_relationships``: |r| (or Spearman, if stronger)
for numeric pairs, the correlation ratio eta for numeric-categorical pairs,
and bias-corrected Cramer's V for categorical pairs.

See spec sections 7.2 and A.15.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from eda_mcp.digest import Finding, Severity
from eda_mcp.issues import simplest_transform
from eda_mcp.profiling import ColumnKind, numeric_stats, spelling_variants, transform_skews
from eda_mcp.relations import cramers_v, group_stats, numeric_pairs

# A whole-number target with this few values is a set of classes, not a quantity.
MAX_CLASS_VALUES = 10
# Majority-class shares that make accuracy misleading (A.15).
SEVERE_IMBALANCE = 0.9
IMBALANCE = 0.75
# A class needs at least this many rows for a model to learn anything of it.
MIN_CLASS_ROWS = 30
# A feature this strongly tied to the target almost certainly encodes it.
LEAK_STRENGTH = 0.95
# Category purity: categories with too few rows are pure by chance.
PURITY_MIN_ROWS = 5
PURITY_SHARE = 0.9
PURITY_CHANCE = 0.5
# Links weaker than this, or not significant, are not worth ranking.
MIN_STRENGTH = 0.05
SIGNIFICANCE = 0.01
CLASSES_SHOWN = 10
SKEW_LIMIT = 1.0

CLASS_KINDS = (ColumnKind.CATEGORICAL, ColumnKind.BOOLEAN)
FEATURE_NUMERIC = (ColumnKind.NUMERIC, ColumnKind.DATETIME)
FEATURE_GROUPED = (ColumnKind.CATEGORICAL, ColumnKind.BOOLEAN)


@dataclass(slots=True)
class Strength:
    feature: str
    strength: float
    detail: str


@dataclass(slots=True)
class TargetReport:
    task: str
    body: dict[str, Any] = field(default_factory=dict)
    findings: list[Finding] = field(default_factory=list)
    ranked: list[Strength] = field(default_factory=list)
    leaks: list[str] = field(default_factory=list)


def task_for(series: pd.Series, kind: ColumnKind) -> str | None:
    """'classification', 'regression', or None when the column cannot be a target."""
    if kind in CLASS_KINDS:
        return "classification"
    if kind is ColumnKind.NUMERIC:
        values = series.dropna()
        integral = bool(np.all(np.mod(values.to_numpy(dtype="float64"), 1) == 0))
        if integral and values.nunique() <= MAX_CLASS_VALUES:
            return "classification"
        return "regression"
    return None


def _as_float(series: pd.Series) -> np.ndarray[Any, np.dtype[np.float64]]:
    if pd.api.types.is_datetime64_any_dtype(series):
        series = (series - series.min()) / pd.Timedelta(days=1)
    return series.to_numpy(dtype="float64", na_value=np.nan)


def _codes(series: pd.Series) -> np.ndarray[Any, Any]:
    return pd.factorize(series, sort=True)[0]


# --------------------------------------------------------------------------
# target health


def _missing_target(name: str, missing: int) -> Finding | None:
    if not missing:
        return None
    return Finding(
        Severity.HIGH,
        "rows with no target value",
        column=name,
        recommendation="drop them before training; never impute a target",
        affected_rows=missing,
    )


def _classification_health(name: str, series: pd.Series) -> tuple[dict[str, Any], list[Finding]]:
    counts = series.value_counts()
    # Largest first, ties by label, so row order cannot change the listing.
    order = sorted(counts.index, key=lambda v: (-int(counts[v]), str(v)))
    counts = counts[order]
    total = int(counts.sum())
    shares = counts / total
    body: dict[str, Any] = {
        "classes": {str(k): int(v) for k, v in counts.head(CLASSES_SHOWN).items()},
    }
    if len(counts) > CLASSES_SHOWN:
        body["classes_not_shown"] = len(counts) - CLASSES_SHOWN

    findings: list[Finding] = []
    majority = float(shares.iloc[0])
    minority = str(counts.index[-1])
    if majority >= SEVERE_IMBALANCE:
        findings.append(
            Finding(
                Severity.HIGH,
                f"severe imbalance: {majority:.0%} of rows are {counts.index[0]}",
                column=name,
                recommendation=(
                    "stratify splits, weight classes, and judge with PR-AUC or F1, not accuracy"
                ),
                affected_rows=int(counts.iloc[1:].sum()),
            )
        )
    elif majority >= IMBALANCE:
        findings.append(
            Finding(
                Severity.MEDIUM,
                f"imbalanced: {majority:.0%} of rows are {counts.index[0]}",
                column=name,
                recommendation="stratify splits and report per-class metrics",
            )
        )
    # Classes that differ only in spelling split one class in two; the model
    # would learn the spelling.
    variants = spelling_variants(counts)
    if variants:
        example = " / ".join(variants["variants"][0])
        findings.append(
            Finding(
                Severity.HIGH,
                f"{variants['variant_groups']} class(es) spelled several ways ({example})",
                column=name,
                recommendation="merge each group into one class before training",
                affected_rows=variants["variant_rows"],
            )
        )
    thin = counts[counts < MIN_CLASS_ROWS]
    if len(thin) and len(counts) > 2:
        findings.append(
            Finding(
                Severity.MEDIUM,
                f"{len(thin)} class(es) with under {MIN_CLASS_ROWS} rows, e.g. {minority}",
                column=name,
                recommendation="merge them into a neighbouring class or collect more rows",
                affected_rows=int(thin.sum()),
            )
        )
    elif len(thin):
        findings.append(
            Finding(
                Severity.HIGH,
                f"class {minority} has only {int(counts.iloc[-1])} rows",
                column=name,
                recommendation="collect more examples; too few to learn from or to validate on",
            )
        )
    return body, findings


def _regression_health(name: str, series: pd.Series) -> tuple[dict[str, Any], list[Finding]]:
    stats = numeric_stats(series)
    body: dict[str, Any] = {
        key: stats[key] for key in ("mean", "std", "min", "median", "max", "skew")
    }
    findings: list[Finding] = []
    skew = stats.get("skew")
    if skew is not None and abs(skew) > SKEW_LIMIT:
        transforms = transform_skews(np.sort(series.to_numpy(dtype="float64")))
        if transforms:
            best = simplest_transform(transforms)
            after = round(transforms[best], 2) + 0.0
            advice = f"model {best}(target) and invert predictions (skew -> {after:.2f})"
        else:
            advice = "use a loss robust to outliers, such as Huber"
        findings.append(
            Finding(
                Severity.MEDIUM,
                f"target is {'right' if skew > 0 else 'left'}-skewed ({skew:.2g})",
                column=name,
                recommendation=advice,
            )
        )
    return body, findings


# --------------------------------------------------------------------------
# feature strength and leakage


def _category_purity(feature: np.ndarray[Any, Any], target: np.ndarray[Any, Any]) -> bool:
    """True when the feature's categories each map to a single target class.

    Purity is judged against chance: with a 94% majority class, a category
    of five rows is all-majority 73% of the time, so many small categories
    would look "pure" without encoding anything. Leakage needs observed
    purity to be high where chance alone would leave it low.
    """
    valid = (feature >= 0) & (target >= 0)
    f, t = feature[valid], target[valid]
    if not len(f):
        return False
    classes = int(t.max()) + 1
    table = np.bincount(f * classes + t, minlength=(int(f.max()) + 1) * classes)
    table = table.reshape(-1, classes)
    sizes = table.sum(axis=1)
    eligible = sizes >= PURITY_MIN_ROWS
    if eligible.sum() < 2 or sizes[eligible].sum() < 0.5 * len(f):
        return False
    pure = (table > 0).sum(axis=1) == 1
    observed = sizes[eligible & pure].sum() / sizes[eligible].sum()
    shares = np.bincount(t, minlength=classes) / len(t)
    chance = sum(int(m) * float((shares ** int(m)).sum()) for m in sizes[eligible])
    expected = chance / sizes[eligible].sum()
    return bool(observed >= PURITY_SHARE and expected <= PURITY_CHANCE)


def feature_strengths(
    df: pd.DataFrame, kinds: dict[str, ColumnKind], target: str, task: str
) -> tuple[list[Strength], list[str]]:
    """Every usable feature's link to the target, and the ones that leak it."""
    labels = {str(c): c for c in df.columns}
    y = df[labels[target]]
    numeric = [c for c, k in kinds.items() if k in FEATURE_NUMERIC and c != target]
    grouped = [c for c, k in kinds.items() if k in FEATURE_GROUPED and c != target]
    found: list[Strength] = []
    leaks: list[str] = []

    if task == "classification":
        classes = _codes(y)
        for x in numeric:
            result = group_stats(classes, _as_float(df[labels[x]]))
            if result is not None and result.p < SIGNIFICANCE:
                eta = math.sqrt(result.eta2)
                found.append(Strength(x, eta, f"eta={eta:.2f}"))
        for c in grouped:
            codes = _codes(df[labels[c]])
            v = cramers_v(classes, codes)
            if v is not None:
                found.append(Strength(c, v, f"V={v:.2f}"))
            if _category_purity(codes, classes):
                leaks.append(c)
    else:
        values = _as_float(y)
        pairs, _ = numeric_pairs(df, [target, *numeric])
        for pair in pairs:
            if target in (pair.a, pair.b):
                other = pair.b if pair.a == target else pair.a
                found.append(Strength(other, pair.strength, pair.detail))
        for c in grouped:
            result = group_stats(_codes(df[labels[c]]), values)
            if result is not None and result.p < SIGNIFICANCE:
                eta = math.sqrt(result.eta2)
                found.append(Strength(c, eta, f"eta={eta:.2f}"))

    leaks.extend(s.feature for s in found if s.strength >= LEAK_STRENGTH and s.feature not in leaks)
    found.sort(key=lambda s: (-s.strength, s.feature))
    return [s for s in found if s.strength >= MIN_STRENGTH], leaks


def analyze_target(
    df: pd.DataFrame, kinds: dict[str, ColumnKind], target: str, task: str
) -> TargetReport:
    labels = {str(c): c for c in df.columns}
    y = df[labels[target]]
    present = y.dropna()
    report = TargetReport(task=task)

    if task == "classification":
        body, health = _classification_health(target, present)
        report.task = f"classification ({present.nunique()} classes)"
    else:
        body, health = _regression_health(target, present)
    report.body.update(body)
    report.body["rows_with_target"] = len(present)

    missing = _missing_target(target, len(y) - len(present))
    if missing:
        report.findings.append(missing)
    report.findings.extend(health)

    ranked, leaks = feature_strengths(df, kinds, target, task)
    by_name = {s.feature: s for s in ranked}
    for name in leaks:
        detail = by_name[name].detail if name in by_name else "categories map 1:1 to classes"
        report.findings.append(
            Finding(
                Severity.HIGH,
                f"probable leakage ({detail} with {target})",
                column=name,
                recommendation="drop it unless its value is known before the outcome",
            )
        )
    report.leaks = leaks
    report.ranked = [s for s in ranked if s.feature not in leaks]
    report.findings.extend(
        Finding(Severity.INFO, f"{s.feature}: {s.detail}") for s in report.ranked
    )
    return report


__all__ = ["TargetReport", "analyze_target", "feature_strengths", "task_for"]
