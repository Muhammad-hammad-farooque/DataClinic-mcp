"""Reading data files.

Real files are messy: unknown encodings, sniffed delimiters, nulls disguised
as ``"N/A"`` or ``"-"``, numbers stored as text. The loader handles those on
the way in so that later analysis is not quietly wrong.

Nothing here samples. Statistics computed downstream see every row, which is
the defect that makes competing servers report wrong numbers on sorted files.

See spec sections 7.1 and A.1.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.api import types as pdt

from eda_mcp.config import Settings
from eda_mcp.errors import DependencyMissingError, PathNotAllowedError, UnsupportedFormatError

# Values that mean "missing" but arrive as text. pandas recognises some of
# these already; the rest are common in exports from spreadsheets and BI tools.
NA_VALUES = [
    "",
    " ",
    "NA",
    "N/A",
    "n/a",
    "na",
    "NULL",
    "null",
    "None",
    "none",
    "-",
    "--",
    "?",
    "unknown",
    "Unknown",
    "UNKNOWN",
    "missing",
    "MISSING",
    "nan",
    "NaN",
    "#N/A",
    "#NULL!",
    "<NA>",
]

SUFFIXES = {
    ".csv": "csv",
    ".tsv": "tsv",
    ".txt": "csv",
    ".xlsx": "excel",
    ".xls": "excel",
    ".xlsm": "excel",
    ".parquet": "parquet",
    ".pq": "parquet",
    ".json": "json",
    ".ndjson": "jsonl",
    ".jsonl": "jsonl",
}

ENCODINGS = ("utf-8", "utf-8-sig", "cp1252", "latin-1")


@dataclass(slots=True)
class LoadReport:
    """What the loader had to decide, so the caller can report it honestly."""

    format: str
    encoding: str | None = None
    delimiter: str | None = None
    coerced_columns: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def resolve_path(raw: str, settings: Settings) -> Path:
    """Validate a path against the configured roots.

    Resolved before the check so ``../`` cannot walk out of an allowed root.
    """
    path = Path(raw).expanduser()
    if not settings.path_allowed(path):
        raise PathNotAllowedError(str(path), [str(p) for p in settings.allowed_paths])
    resolved = path.resolve()
    if not resolved.exists():
        from eda_mcp.errors import EDAError, ErrorCode

        raise EDAError(
            ErrorCode.SOURCE_NOT_FOUND,
            f"no such file: {raw}",
            "check the path, or pass one relative to the working directory",
        )
    return resolved


def detect_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix not in SUFFIXES:
        raise UnsupportedFormatError(suffix, sorted(set(SUFFIXES)))
    return SUFFIXES[suffix]


def detect_encoding(path: Path) -> str:
    """Find the first encoding that decodes a sample without error.

    Tried in order of likelihood rather than guessed statistically: latin-1
    decodes any byte sequence, so it sits last as the guaranteed fallback.
    """
    sample = path.read_bytes()[:200_000]
    for encoding in ENCODINGS:
        try:
            sample.decode(encoding)
        except UnicodeDecodeError:
            continue
        return encoding
    return "latin-1"


def detect_delimiter(path: Path, encoding: str) -> str:
    """Sniff the delimiter, falling back to a comma."""
    with path.open("r", encoding=encoding, errors="replace", newline="") as handle:
        sample = handle.read(64_000)
    if not sample:
        return ","
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def coerce_types(df: pd.DataFrame, report: LoadReport) -> pd.DataFrame:
    """Convert text columns that are unambiguously numeric or datetime.

    A column is only converted when nearly every non-null value parses, so a
    numeric column with a stray label is left alone rather than silently
    turned into nulls.
    """
    for column in df.columns:
        series = df[column]
        # pandas 3 gives text columns a dedicated string dtype rather than
        # object, so both must be accepted or coercion silently never runs.
        if not (pdt.is_object_dtype(series) or pdt.is_string_dtype(series)):
            continue
        non_null = series.dropna()
        if non_null.empty:
            continue

        numeric = pd.to_numeric(non_null, errors="coerce")
        if numeric.notna().mean() >= 0.95:
            df[column] = pd.to_numeric(series, errors="coerce")
            report.coerced_columns[str(column)] = "numeric"
            continue

        # Thousands separators are common in spreadsheet exports and stop a
        # column that is plainly numeric from being read as one.
        stripped = non_null.astype("string").str.replace(",", "", regex=False)
        if pd.to_numeric(stripped, errors="coerce").notna().mean() >= 0.95:
            df[column] = pd.to_numeric(
                series.astype("string").str.replace(",", "", regex=False), errors="coerce"
            )
            report.coerced_columns[str(column)] = "numeric (thousands separators removed)"
            continue

        # format="mixed" lets a column hold more than one layout, which is
        # common in hand-maintained exports.
        try:
            parsed = pd.to_datetime(non_null, errors="coerce", format="mixed")
        except (ValueError, TypeError):
            continue
        if parsed.notna().mean() >= 0.95:
            df[column] = pd.to_datetime(series, errors="coerce", format="mixed")
            report.coerced_columns[str(column)] = "datetime"

    return df


def _require(module: str, package: str, extra: str) -> None:
    from importlib.util import find_spec

    if find_spec(module) is None:
        raise DependencyMissingError(package, extra)


def load_file(
    raw_path: str,
    settings: Settings,
    *,
    limit: int | None = None,
    options: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, LoadReport]:
    """Read a file into a DataFrame, reporting every decision made."""
    options = dict(options or {})
    path = resolve_path(raw_path, settings)
    fmt = detect_format(path)
    report = LoadReport(format=fmt)

    if fmt in ("csv", "tsv"):
        encoding = options.pop("encoding", None) or detect_encoding(path)
        delimiter = options.pop("delimiter", None) or (
            "\t" if fmt == "tsv" else detect_delimiter(path, encoding)
        )
        report.encoding = encoding
        report.delimiter = delimiter
        df = pd.read_csv(
            path,
            encoding=encoding,
            sep=delimiter,
            na_values=NA_VALUES,
            keep_default_na=True,
            nrows=limit,
            low_memory=False,
            **options,
        )
    elif fmt == "excel":
        _require("openpyxl", "openpyxl", "excel")
        df = pd.read_excel(path, na_values=NA_VALUES, nrows=limit, **options)
    elif fmt == "parquet":
        _require("pyarrow", "pyarrow", "parquet")
        df = pd.read_parquet(path, **options)
        if limit is not None:
            df = df.head(limit)
    elif fmt == "jsonl":
        df = pd.read_json(path, lines=True, nrows=limit, **options)
    else:  # json
        df = pd.read_json(path, **options)
        if limit is not None:
            df = df.head(limit)

    if limit is not None and len(df) == limit:
        report.notes.append(f"stopped at limit={limit:,}; statistics cover the rows read")

    df = coerce_types(df, report)
    return df, report
