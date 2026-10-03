"""Relationships between columns: correlation, association, collinearity.

Every measure here is put on one 0-1 strength scale so that pairs of any
type can be ranked together: |r| for numeric pairs, bias-corrected Cramer's
V for categorical pairs, and the correlation ratio eta for mixed pairs.

Only ranked pairs above a threshold are reported -- never a full matrix,
which on fifty columns would be 1,225 numbers the reader cannot use (spec
7.2, 10.5).

Numeric correlations use pairwise deletion and are computed from sums
accumulated over row blocks: a handful of matrix products in place of a pass
per pair, with memory bounded by the block size rather than the table.

See spec sections 7.2, 11 and A.11-A.14.
"""

from __future__ import annotations

import math
import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as pdt
from scipy import stats as scipy_stats

from eda_mcp.digest import Finding, Severity
from eda_mcp.profiling import ColumnKind

# Pairs at least this strong are worth reporting in a whole-table scan; a
# named target lowers the bar, because the caller asked about that column.
REPORT_STRENGTH = 0.5
TARGET_STRENGTH = 0.1
# Numeric columns this correlated carry the same information (A.14).
COLLINEAR = 0.9
VIF_LIMIT = 10.0
# Spearman exceeding Pearson by this much means monotonic but not linear.
NONLINEAR_GAP = 0.1
MIN_PAIR_ROWS = 30
# Group comparisons: Cohen's small effect for eta squared, and significance.
GROUP_EFFECT = 0.01
GROUP_P = 0.01
MAX_GROUPS = 50
GROUPS_SHOWN = 10
# Rows per block when accumulating correlation sums, bounding memory.
BLOCK_ROWS = 100_000

NUMERIC_KINDS = (ColumnKind.NUMERIC, ColumnKind.DATETIME)
GROUP_KINDS = (ColumnKind.CATEGORICAL, ColumnKind.BOOLEAN)


@dataclass(slots=True)
class Pair:
    """One measured relationship, on the shared 0-1 strength scale."""

    a: str
    b: str
    strength: float
    detail: str

    def finding(self) -> Finding:
        return Finding(Severity.INFO, f"{self.a} vs {self.b}: {self.detail}")


# --------------------------------------------------------------------------
# numeric <-> numeric


def _as_float(series: pd.Series) -> np.ndarray[Any, np.dtype[np.float64]]:
    if pdt.is_datetime64_any_dtype(series):
        series = (series - series.min()) / pd.Timedelta(days=1)
    return series.to_numpy(dtype="float64", na_value=np.nan)


def _ranks(values: np.ndarray[Any, np.dtype[np.float64]]) -> np.ndarray[Any, Any]:
    """Average ranks of the present values, as ``scipy.stats.rankdata``.

    One argsort plus run-length arithmetic for ties: several times faster
    than ``rankdata``, and numpy's sort releases the GIL, so columns can be
    ranked on threads. Missing values stay missing.
    """
    out = np.full(values.shape, np.nan)
    present = ~np.isnan(values)
    x = values[present]
    if not len(x):
        return out
    order = np.argsort(x, kind="quicksort")
    ordered = x[order]
    starts = np.flatnonzero(np.r_[True, np.diff(ordered) != 0])
    lengths = np.diff(np.r_[starts, len(ordered)])
    ranks = np.empty(len(x))
    ranks[order] = np.repeat(starts + (lengths - 1) / 2 + 1, lengths)
    out[present] = ranks
    return out


def rank_columns(matrix: np.ndarray[Any, np.dtype[np.float64]]) -> np.ndarray[Any, Any]:
    columns = [np.ascontiguousarray(matrix[:, j]) for j in range(matrix.shape[1])]
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 1) as pool:
        return np.column_stack(list(pool.map(_ranks, columns)))


def pairwise_corr(
    matrix: np.ndarray[Any, np.dtype[np.float64]],
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Pearson correlation of every column pair, with pairwise deletion.

    Returns the correlation matrix and the rows each pair was computed on.
    Matches ``DataFrame.corr()`` but runs as matrix products over row blocks.
    Columns are centred first, which keeps the subtractions stable.

    Columns with no missing values need only the Gram matrix Z'Z between
    them: every row counts and each centred column sums to zero. Only pairs
    involving an incomplete column pay for the masked sums, with M its
    presence mask and Z the zero-filled values:

        n = M'M      sum_x = Z'M      sum_xx = (Z*Z)'M

    so a table with few gappy columns costs little more than a complete one.
    """
    rows, k = matrix.shape
    # A plain mean is NaN exactly for the columns with gaps, which finds them
    # without a separate scan; only those need the slower nanmean.
    means = matrix.mean(axis=0)
    gappy = np.flatnonzero(np.isnan(means))
    g = len(gappy)
    if g:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # an all-missing column
            means[gappy] = np.nanmean(matrix[:, gappy], axis=0)

    sxy = np.zeros((k, k))
    n_g = np.zeros((g, k))  # rows where gappy column a and column j are both present
    sx_g = np.zeros((g, k))  # sum of gappy column a over those rows
    sx_j = np.zeros((k, g))  # sum of column j over those rows
    sxx_g = np.zeros((g, k))
    sxx_j = np.zeros((k, g))
    for start in range(0, rows, BLOCK_ROWS):
        # Centred per block: no full-size copy of the matrix is ever made.
        z = matrix[start : start + BLOCK_ROWS] - means
        if not g:
            sxy += z.T @ z
            continue
        present = np.ones(z.shape)
        present[:, gappy] = ~np.isnan(z[:, gappy])
        mask_g = present[:, gappy]
        z_g = np.where(mask_g > 0, z[:, gappy], 0.0)
        z[:, gappy] = z_g
        sxy += z.T @ z
        n_g += mask_g.T @ present
        sx_g += z_g.T @ present
        sxx_g += (z_g * z_g).T @ present
        sx_j += z.T @ mask_g
        sxx_j += (z * z).T @ mask_g

    n = np.full((k, k), float(rows))
    with np.errstate(divide="ignore", invalid="ignore"):
        variance = np.diag(sxy) / rows
        r = sxy / rows / np.sqrt(np.outer(variance, variance))
        for a, i in enumerate(gappy):
            count = n_g[a]
            mean_i = sx_g[a] / count
            mean_j = sx_j[:, a] / count
            cov = sxy[i] / count - mean_i * mean_j
            var_i = sxx_g[a] / count - mean_i**2
            var_j = sxx_j[:, a] / count - mean_j**2
            row = cov / np.sqrt(var_i * var_j)
            row[(var_i <= 0) | (var_j <= 0)] = np.nan
            r[i, :] = r[:, i] = row
            n[i, :] = n[:, i] = count
    r[n < MIN_PAIR_ROWS] = np.nan
    np.fill_diagonal(r, 1.0)
    return np.clip(r, -1.0, 1.0), n


def numeric_pairs(df: pd.DataFrame, columns: list[str]) -> tuple[list[Pair], np.ndarray[Any, Any]]:
    """Every numeric pair's strength, and the Pearson matrix for collinearity."""
    labels = {str(c): c for c in df.columns}
    if len(columns) < 2:
        return [], np.ones((len(columns), len(columns)))
    values = np.column_stack([_as_float(df[labels[c]]) for c in columns])
    pearson, _ = pairwise_corr(values)
    # Spearman is Pearson on ranks; it sees monotonic curves Pearson misses.
    # Each column is ranked once over all its values rather than re-ranked per
    # pair, which differs from pandas by ~1e-5 where gaps differ, far below
    # the two decimals reported, and avoids k^2 sorts.
    spearman, _ = pairwise_corr(rank_columns(values))

    pairs: list[Pair] = []
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            p, s = pearson[i, j], spearman[i, j]
            if not (math.isfinite(p) and math.isfinite(s)):
                continue
            detail = f"r={p:+.2f}"
            if abs(s) - abs(p) >= NONLINEAR_GAP:
                detail += f", spearman {s:+.2f}: monotonic, not linear"
            pairs.append(Pair(columns[i], columns[j], max(abs(p), abs(s)), detail))
    return pairs, pearson


def collinear_clusters(columns: list[str], pearson: np.ndarray[Any, Any]) -> list[list[str]]:
    """Columns linked by |r| >= COLLINEAR, grouped transitively."""
    parent = list(range(len(columns)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    strong = np.argwhere(np.triu(np.abs(np.nan_to_num(pearson)) >= COLLINEAR, k=1))
    for i, j in strong:
        parent[root(int(i))] = root(int(j))

    groups: dict[int, list[str]] = {}
    for i, name in enumerate(columns):
        groups.setdefault(root(i), []).append(name)
    return [g for g in groups.values() if len(g) > 1]


def variance_inflation(pearson: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """VIF per column: the diagonal of the inverse correlation matrix.

    Catches a column predictable from several others together, which no
    single pairwise correlation reveals. The pseudo-inverse tolerates the
    singular matrix that perfect collinearity produces.
    """
    filled = np.nan_to_num(pearson)
    np.fill_diagonal(filled, 1.0)
    return np.diag(np.linalg.pinv(filled))


# --------------------------------------------------------------------------
# categorical <-> categorical and categorical <-> numeric


def _codes(series: pd.Series) -> tuple[np.ndarray[Any, Any], pd.Index]:
    codes, uniques = pd.factorize(series, sort=True)
    return codes, pd.Index(uniques)


def cramers_v(a: np.ndarray[Any, Any], b: np.ndarray[Any, Any]) -> float | None:
    """Bias-corrected Cramer's V (Bergsma 2013) between two code arrays.

    The uncorrected V rises with the number of levels even for unrelated
    columns; the correction removes that, so high-cardinality pairs are not
    reported as associated by construction.
    """
    valid = (a >= 0) & (b >= 0)
    a, b = a[valid], b[valid]
    n = len(a)
    if n < MIN_PAIR_ROWS:
        return None
    width = int(b.max()) + 1
    table = np.bincount(a * width + b, minlength=(int(a.max()) + 1) * width).reshape(-1, width)
    table = table[table.sum(axis=1) > 0][:, table.sum(axis=0) > 0]
    r, k = table.shape
    if r < 2 or k < 2:
        return None
    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / n
    phi2 = float(((table - expected) ** 2 / expected).sum()) / n
    phi2 = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    r_corr = r - (r - 1) ** 2 / (n - 1)
    k_corr = k - (k - 1) ** 2 / (n - 1)
    denominator = min(r_corr - 1, k_corr - 1)
    return math.sqrt(phi2 / denominator) if denominator > 0 else None


@dataclass(slots=True)
class GroupStats:
    """A numeric column summarised within each group of a categorical one."""

    sizes: np.ndarray[Any, Any]
    means: np.ndarray[Any, Any]
    variances: np.ndarray[Any, Any]
    eta2: float
    p: float


def group_stats(codes: np.ndarray[Any, Any], values: np.ndarray[Any, Any]) -> GroupStats | None:
    """One-way ANOVA from per-group sums: eta squared and its p-value."""
    valid = (codes >= 0) & ~np.isnan(values)
    g, x = codes[valid], values[valid]
    n = len(x)
    if n < MIN_PAIR_ROWS:
        return None
    sizes = np.bincount(g).astype("float64")
    present = sizes > 0
    groups = int(present.sum())
    if groups < 2 or n <= groups:
        return None
    centred = x - x.mean()  # centring keeps the sums of squares stable
    sums = np.bincount(g, weights=centred)
    squares = np.bincount(g, weights=centred * centred)
    with np.errstate(divide="ignore", invalid="ignore"):
        means = sums / sizes
        variances = (squares - sizes * means**2) / (sizes - 1)
    total = float(squares.sum())
    if total <= 0:
        return None
    between = float((sums[present] ** 2 / sizes[present]).sum())
    within = total - between
    eta2 = between / total
    if within <= 0:
        p = 0.0
    else:
        f = (between / (groups - 1)) / (within / (n - groups))
        p = float(scipy_stats.f.sf(f, groups - 1, n - groups))
    return GroupStats(sizes, means + x.mean(), variances, eta2, p)


def _effect_word(eta2: float) -> str:
    # Cohen's conventions for eta squared.
    if eta2 >= 0.14:
        return "large"
    if eta2 >= 0.06:
        return "medium"
    return "small"


# --------------------------------------------------------------------------
# entry points


def _usable(df: pd.DataFrame, kinds: dict[str, ColumnKind]) -> tuple[list[str], list[str]]:
    numeric = [c for c, k in kinds.items() if k in NUMERIC_KINDS]
    grouped = [c for c, k in kinds.items() if k in GROUP_KINDS]
    return numeric, grouped


def all_pairs(
    df: pd.DataFrame, kinds: dict[str, ColumnKind]
) -> tuple[list[Pair], list[str], np.ndarray[Any, Any], dict[str, int]]:
    """Every measurable pair in the table, strongest first.

    Also returns the numeric columns and their Pearson matrix, from which
    collinearity is judged, and how many pairs of each type were tested.
    """
    labels = {str(c): c for c in df.columns}
    numeric, grouped = _usable(df, kinds)
    pairs, pearson = numeric_pairs(df, numeric)
    tested = {"numeric": len(numeric) * (len(numeric) - 1) // 2}

    codes = {c: _codes(df[labels[c]])[0] for c in grouped}
    tested["categorical"] = len(grouped) * (len(grouped) - 1) // 2
    for i, a in enumerate(grouped):
        for b in grouped[i + 1 :]:
            v = cramers_v(codes[a], codes[b])
            if v is not None:
                pairs.append(Pair(a, b, v, f"V={v:.2f}"))

    tested["mixed"] = len(grouped) * len(numeric)
    values = {c: _as_float(df[labels[c]]) for c in numeric}
    for g in grouped:
        for x in numeric:
            result = group_stats(codes[g], values[x])
            if result is not None and result.p < GROUP_P:
                eta = math.sqrt(result.eta2)
                pairs.append(Pair(x, g, eta, f"eta={eta:.2f}, differs by {g}"))

    pairs.sort(key=lambda p: (-p.strength, p.a, p.b))
    return pairs, numeric, pearson, tested


def collinearity_findings(
    df: pd.DataFrame, numeric: list[str], pearson: np.ndarray[Any, Any]
) -> list[Finding]:
    """Clusters of near-interchangeable columns, then hidden multi-column ones."""
    labels = {str(c): c for c in df.columns}
    findings: list[Finding] = []
    clustered: set[str] = set()
    for cluster in collinear_clusters(numeric, pearson):
        clustered.update(cluster)
        # Keep the most complete column; ties go to the first in the frame.
        keep = max(cluster, key=lambda c: (int(df[labels[c]].count()), -numeric.index(c)))
        others = ", ".join(c for c in cluster if c != keep)
        findings.append(
            Finding(
                Severity.MEDIUM,
                f"{', '.join(cluster)} move together (|r| >= {COLLINEAR})",
                recommendation=f"keep {keep}; drop or combine {others}",
            )
        )

    if len(numeric) > 2:
        for name, vif in zip(numeric, variance_inflation(pearson), strict=True):
            if name not in clustered and vif > VIF_LIMIT:
                findings.append(
                    Finding(
                        Severity.MEDIUM,
                        f"VIF {vif:.0f}: largely predictable from the other columns together",
                        column=name,
                        recommendation="drop it or combine it with the columns it depends on",
                    )
                )
    return findings


def target_pairs(df: pd.DataFrame, kinds: dict[str, ColumnKind], target: str) -> list[Pair]:
    """Every column's relationship with *target*, strongest first."""
    labels = {str(c): c for c in df.columns}
    numeric, grouped = _usable(df, kinds)
    pairs: list[Pair] = []
    if target in numeric:
        others = [c for c in numeric if c != target]
        found, _ = numeric_pairs(df, [target, *others])
        pairs.extend(p for p in found if target in (p.a, p.b))
        values = _as_float(df[labels[target]])
        for g in grouped:
            result = group_stats(_codes(df[labels[g]])[0], values)
            if result is not None and result.p < GROUP_P:
                eta = math.sqrt(result.eta2)
                pairs.append(Pair(target, g, eta, f"eta={eta:.2f}, differs by {g}"))
    elif target in grouped:
        codes = _codes(df[labels[target]])[0]
        for g in grouped:
            if g != target:
                v = cramers_v(codes, _codes(df[labels[g]])[0])
                if v is not None:
                    pairs.append(Pair(target, g, v, f"V={v:.2f}"))
        for x in numeric:
            result = group_stats(codes, _as_float(df[labels[x]]))
            if result is not None and result.p < GROUP_P:
                eta = math.sqrt(result.eta2)
                pairs.append(Pair(target, x, eta, f"eta={eta:.2f}, {x} differs by {target}"))

    # Orient every pair as "target vs other" so the lines scan as one list.
    for p in pairs:
        if p.b == target:
            p.a, p.b = p.b, p.a
    pairs.sort(key=lambda p: (-p.strength, p.b))
    return [p for p in pairs if p.strength >= TARGET_STRENGTH]


def compare_groups(
    df: pd.DataFrame,
    kinds: dict[str, ColumnKind],
    group_by: str,
    columns: list[str] | None = None,
) -> tuple[dict[str, int], list[Finding]]:
    """How every other column differs across the groups of *group_by*.

    Numeric columns are compared by one-way ANOVA with eta squared as the
    effect size (Cohen's d for two groups); categorical ones by Cramer's V.
    Only differences that are both significant and at least small are kept,
    ranked by effect size.
    """
    labels = {str(c): c for c in df.columns}
    codes, uniques = _codes(df[labels[group_by]])
    sizes = np.bincount(codes[codes >= 0], minlength=len(uniques))
    order = np.argsort(-sizes, kind="stable")
    shown = {str(uniques[i]): int(sizes[i]) for i in order[:GROUPS_SHOWN]}

    numeric, grouped = _usable(df, kinds)
    wanted = set(columns) if columns else None
    ranked: list[tuple[float, Finding]] = []

    for x in numeric:
        if x == group_by or (wanted and x not in wanted):
            continue
        result = group_stats(codes, _as_float(df[labels[x]]))
        if result is None or result.p >= GROUP_P or result.eta2 < GROUP_EFFECT:
            continue
        valid = result.sizes >= MIN_PAIR_ROWS
        if valid.sum() < 2:
            continue
        idx = np.flatnonzero(valid)
        hi = idx[int(np.argmax(result.means[idx]))]
        lo = idx[int(np.argmin(result.means[idx]))]
        message = f"{x} by {group_by}: eta2={result.eta2:.2f} ({_effect_word(result.eta2)})"
        if len(uniques) == 2:
            pooled = math.sqrt(
                ((result.sizes - 1) * result.variances).sum() / (result.sizes.sum() - 2)
            )
            if pooled > 0:
                d = (result.means[1] - result.means[0]) / pooled
                message += f", d={d:+.2f} ({uniques[1]} vs {uniques[0]})"
        message += (
            f"; highest {uniques[hi]} (mean {result.means[hi]:.3g}),"
            f" lowest {uniques[lo]} (mean {result.means[lo]:.3g})"
        )
        ranked.append((result.eta2, Finding(Severity.INFO, message)))

    for c in grouped:
        if c == group_by or (wanted and c not in wanted):
            continue
        v = cramers_v(codes, _codes(df[labels[c]])[0])
        # V squared is comparable to eta squared as a share of variation.
        if v is not None and v * v >= GROUP_EFFECT:
            ranked.append((v * v, Finding(Severity.INFO, f"{c} by {group_by}: V={v:.2f}")))

    ranked.sort(key=lambda item: -item[0])
    return shown, [f for _, f in ranked]


__all__ = [
    "Pair",
    "all_pairs",
    "collinearity_findings",
    "compare_groups",
    "cramers_v",
    "group_stats",
    "pairwise_corr",
    "target_pairs",
]
