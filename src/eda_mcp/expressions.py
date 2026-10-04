"""A restricted expression language for ``query``.

Expressions use Python syntax but are never handed to ``eval``. They are
parsed to an AST and interpreted here against whitelisted node types: column
names, literals, operators, and a fixed table of functions. There is no
attribute access, subscripting, import, lambda or comprehension, so the
classic escapes -- ``().__class__.__bases__``, ``__import__('os')``,
``open(...)`` -- have no syntax to stand on (spec 12.1).

``DataFrame.query`` / ``DataFrame.eval`` are deliberately not used: their
python engine permits attribute access and is not a security boundary.

Column names that are not identifiers are written in backticks, as in
pandas: ``\\`unit price\\` > 3``.

See spec sections 7.2 and 12.1.
"""

from __future__ import annotations

import ast
import difflib
import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as pdt

from eda_mcp.digest import estimate_tokens
from eda_mcp.errors import EDAError, ErrorCode, OperationRefusedError

MAX_LENGTH = 2_000
MAX_NODES = 200
MAX_STRING = 1_000

BACKTICK = re.compile(r"`([^`]+)`")


# --------------------------------------------------------------------------
# results


@dataclass(slots=True)
class Rows:
    """A row listing: the frame to show and how many rows matched in total."""

    frame: pd.DataFrame
    matched: int


@dataclass(slots=True)
class Grouped:
    """An aggregate computed per group, largest first."""

    values: pd.Series
    function: str


Result = Rows | Grouped | pd.Series | Any


def invalid(message: str, remedy: str) -> EDAError:
    return EDAError(ErrorCode.INVALID_OPERATION, message, remedy)


def refused(what: str) -> OperationRefusedError:
    return OperationRefusedError(
        what,
        "query expressions may only use column names, literals, operators and the listed functions",
        "rewrite the expression without it",
    )


# --------------------------------------------------------------------------
# functions


def _mask(value: Any, length: int) -> pd.Series:
    if not isinstance(value, pd.Series) or not pdt.is_bool_dtype(value.dtype):
        raise invalid("where= must be a true/false condition", "e.g. where=price > 100")
    if len(value) != length:
        raise invalid("where= must be a per-row condition", "compare a column to a value")
    return value.fillna(False).astype(bool)


def _series(value: Any, name: str) -> pd.Series:
    if not isinstance(value, pd.Series):
        raise invalid(f"{name}() needs a column", f"e.g. {name}(price)")
    return value


def _text(value: Any, name: str) -> pd.Series:
    series = _series(value, name)
    if not (pdt.is_string_dtype(series.dtype) or pdt.is_object_dtype(series.dtype)):
        raise invalid(f"{name}() needs a text column", "use it on a categorical or text column")
    return series.astype("string")


def _dates(value: Any, name: str) -> Any:
    series = _series(value, name)
    if not pdt.is_datetime64_any_dtype(series.dtype):
        raise invalid(f"{name}() needs a date column", "use it on a datetime column")
    return series.dt


def _numeric(value: Any, name: str) -> Any:
    if isinstance(value, pd.Series):
        if pdt.is_bool_dtype(value.dtype) or not pdt.is_numeric_dtype(value.dtype):
            raise invalid(f"{name}() needs a numeric column", "use it on a numeric column")
        return value.astype("float64")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise invalid(f"{name}() needs a number", f"e.g. {name}(price)")
    return float(value)


def _unary(fn: Callable[[Any], Any], name: str) -> Callable[..., Any]:
    def apply(value: Any) -> Any:
        with np.errstate(all="ignore"):
            return fn(_numeric(value, name))

    return apply


ELEMENTWISE: dict[str, Callable[..., Any]] = {
    "abs": _unary(np.abs, "abs"),
    "log": _unary(np.log, "log"),
    "log1p": _unary(np.log1p, "log1p"),
    "sqrt": _unary(np.sqrt, "sqrt"),
    "exp": _unary(np.exp, "exp"),
    "round": lambda v, digits=0: np.round(_numeric(v, "round"), int(digits)),
    "isnull": lambda v: _series(v, "isnull").isna(),
    "notnull": lambda v: _series(v, "notnull").notna(),
    "between": lambda v, low, high: _series(v, "between").between(low, high),
    "lower": lambda v: _text(v, "lower").str.lower(),
    "upper": lambda v: _text(v, "upper").str.upper(),
    "strip": lambda v: _text(v, "strip").str.strip(),
    "length": lambda v: _text(v, "length").str.len(),
    "contains": lambda v, text, case=True: _text(v, "contains").str.contains(
        str(text), case=bool(case), regex=False
    ),
    "startswith": lambda v, text: _text(v, "startswith").str.startswith(str(text)),
    "endswith": lambda v, text: _text(v, "endswith").str.endswith(str(text)),
    "year": lambda v: _dates(v, "year").year,
    "month": lambda v: _dates(v, "month").month,
    "day": lambda v: _dates(v, "day").day,
    "weekday": lambda v: _dates(v, "weekday").dayofweek,
    "hour": lambda v: _dates(v, "hour").hour,
}
ELEMENTWISE["isna"] = ELEMENTWISE["isnull"]
ELEMENTWISE["notna"] = ELEMENTWISE["notnull"]

AGGREGATES = ("count", "sum", "mean", "median", "min", "max", "std", "nunique", "quantile")
FUNCTIONS = sorted([*ELEMENTWISE, *AGGREGATES, "rows"])


# --------------------------------------------------------------------------
# interpreter


class _Evaluator:
    def __init__(self, df: pd.DataFrame, aliases: dict[str, str]) -> None:
        self.df = df
        self.labels = {str(c): c for c in df.columns}
        self.aliases = aliases

    # --- names and literals

    def column(self, name: str) -> pd.Series:
        name = self.aliases.get(name, name)
        if name.startswith("__"):
            raise refused(f"the name {name!r}")
        if name not in self.labels:
            close = difflib.get_close_matches(name, list(self.labels), n=3)
            hint = f"did you mean {', '.join(close)}?" if close else "check the column name"
            if name in FUNCTIONS:
                hint = f"{name} is a function; call it as {name}(...)"
            raise invalid(f"no column named {name!r}", hint)
        return self.df[self.labels[name]].rename(name)

    def literal(self, value: Any) -> Any:
        if isinstance(value, str) and len(value) > MAX_STRING:
            raise invalid("string literal too long", f"keep literals under {MAX_STRING} characters")
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        raise refused(f"the literal {value!r}")

    # --- dispatch

    def visit(self, node: ast.AST) -> Any:
        method = getattr(self, f"visit_{type(node).__name__}", None)
        if method is None:
            raise refused(_describe(node))
        return method(node)

    def visit_Expression(self, node: ast.Expression) -> Any:
        return self.visit(node.body)

    def visit_Name(self, node: ast.Name) -> Any:
        if node.id in ("True", "False", "None"):  # pragma: no cover - parsed as Constant
            return {"True": True, "False": False, "None": None}[node.id]
        return self.column(node.id)

    def visit_Constant(self, node: ast.Constant) -> Any:
        return self.literal(node.value)

    def visit_List(self, node: ast.List) -> list[Any]:
        return [self._scalar(item) for item in node.elts]

    visit_Tuple = visit_List

    def _scalar(self, node: ast.AST) -> Any:
        value = self.visit(node)
        if isinstance(value, (pd.Series, Rows, Grouped, list)):
            raise invalid("lists may hold only literal values", 'e.g. country in ["UK", "France"]')
        return value

    # --- operators

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        value = self.visit(node.operand)
        if isinstance(node.op, (ast.Not, ast.Invert)):
            if isinstance(value, pd.Series):
                return ~value.fillna(False).astype(bool)
            return not value
        if isinstance(node.op, ast.USub):
            return -_numeric(value, "-") if not isinstance(value, pd.Series) else -value
        if isinstance(node.op, ast.UAdd):
            return value
        raise refused(_describe(node))  # pragma: no cover

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        left, right = self.visit(node.left), self.visit(node.right)
        op = node.op
        if isinstance(op, (ast.BitAnd, ast.BitOr)):
            return self._logical(op, left, right)
        if _is_text(left) or _is_text(right):
            # Text allows only + with other text. Anything else is refused,
            # whether the text is a literal or a column: "a" * 10**9 and
            # name * 10**9 both build gigabyte strings, and the result of +
            # can be no longer than its inputs combined.
            if not isinstance(op, ast.Add) or not (_is_text(left) and _is_text(right)):
                raise invalid("text supports only + with other text", 'e.g. country + "-x"')
            if isinstance(left, str) and isinstance(right, str):
                return self.literal(left + right)
            try:
                return left + right
            except TypeError:
                raise invalid("text can only be added to text", 'e.g. country + "-x"') from None
        operations: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
            ast.Add: lambda a, b: a + b,
            ast.Sub: lambda a, b: a - b,
            ast.Mult: lambda a, b: a * b,
            ast.Div: lambda a, b: a / b,
            ast.FloorDiv: lambda a, b: a // b,
            ast.Mod: lambda a, b: a % b,
            # Floats, so 10 ** 10 ** 10 overflows to inf instead of building
            # an integer with ten billion digits.
            ast.Pow: lambda a, b: np.power(
                np.float64(a) if _is_number(a) else a.astype("float64"), b
            ),
        }
        if type(op) not in operations:
            raise refused(_describe(node))
        if isinstance(op, ast.Pow) and _is_number(right):
            right = float(right)
        with np.errstate(all="ignore"):
            try:
                return operations[type(op)](left, right)
            except ZeroDivisionError:
                return math.nan
            except TypeError as exc:
                raise invalid(
                    f"cannot apply {_symbol(op)} here ({exc})",
                    "arithmetic needs numbers on both sides",
                ) from None

    def _logical(self, op: ast.operator | ast.boolop, left: Any, right: Any) -> Any:
        if isinstance(left, pd.Series) or isinstance(right, pd.Series):
            a = left.fillna(False).astype(bool) if isinstance(left, pd.Series) else bool(left)
            b = right.fillna(False).astype(bool) if isinstance(right, pd.Series) else bool(right)
            return a & b if isinstance(op, (ast.And, ast.BitAnd)) else a | b
        return (left and right) if isinstance(op, (ast.And, ast.BitAnd)) else (left or right)

    def visit_BoolOp(self, node: ast.BoolOp) -> Any:
        values = [self.visit(v) for v in node.values]
        result = values[0]
        for value in values[1:]:
            result = self._logical(node.op, result, value)
        return result

    def visit_Compare(self, node: ast.Compare) -> Any:
        # Chained comparisons -- 18 <= age < 65 -- become a conjunction.
        left = self.visit(node.left)
        result: Any = None
        for op, comparator in zip(node.ops, node.comparators, strict=True):
            right = self.visit(comparator)
            step = self._compare(op, left, right)
            result = step if result is None else self._logical(ast.And(), result, step)
            left = right
        return result

    def _compare(self, op: ast.cmpop, left: Any, right: Any) -> Any:
        if isinstance(op, (ast.In, ast.NotIn)):
            if not isinstance(right, list):
                raise invalid("in needs a list of values", 'e.g. country in ["UK", "France"]')
            hit = left.isin(right) if isinstance(left, pd.Series) else left in right
            return ~hit if isinstance(op, ast.NotIn) else hit
        compare: dict[type[ast.cmpop], Callable[[Any, Any], Any]] = {
            ast.Eq: lambda a, b: a == b,
            ast.NotEq: lambda a, b: a != b,
            ast.Lt: lambda a, b: a < b,
            ast.LtE: lambda a, b: a <= b,
            ast.Gt: lambda a, b: a > b,
            ast.GtE: lambda a, b: a >= b,
        }
        if type(op) not in compare:
            raise refused("is / is not; use == or isnull()")
        try:
            return compare[type(op)](left, right)
        except TypeError as exc:
            raise invalid(
                f"cannot compare these values ({exc})",
                "compare numbers with numbers, and text with quoted text",
            ) from None

    # --- calls

    def visit_Call(self, node: ast.Call) -> Any:
        if not isinstance(node.func, ast.Name):
            raise refused("calling anything but a listed function")
        name = node.func.id
        if name.startswith("_"):
            raise refused(f"the name {name!r}")
        if name not in FUNCTIONS:
            close = difflib.get_close_matches(name, FUNCTIONS, n=3)
            raise invalid(
                f"unknown function {name}()",
                f"did you mean {', '.join(close)}?"
                if close
                else f"available: {', '.join(FUNCTIONS)}",
            )
        if any(isinstance(a, ast.Starred) for a in node.args) or any(
            k.arg is None for k in node.keywords
        ):
            raise refused("* or ** argument unpacking")
        keywords = {k.arg: k.value for k in node.keywords if k.arg is not None}
        if name == "rows":
            return self._rows(node.args, keywords)
        if name in AGGREGATES:
            return self._aggregate(name, node.args, keywords)
        args = [self.visit(a) for a in node.args]
        kwargs = {k: self.visit(v) for k, v in keywords.items()}
        try:
            return ELEMENTWISE[name](*args, **kwargs)
        except TypeError:
            raise invalid(f"wrong arguments for {name}()", _signature(name)) from None

    def _where(self, keywords: dict[str, ast.expr]) -> pd.Series | None:
        node = keywords.pop("where", None)
        return None if node is None else _mask(self.visit(node), len(self.df))

    def _aggregate(self, name: str, args: list[ast.expr], keywords: dict[str, ast.expr]) -> Any:
        where = self._where(keywords)
        by_node = keywords.pop("by", None)
        q_node = keywords.pop("q", None)
        if keywords:
            raise invalid(f"{name}() does not take {', '.join(keywords)}=", _signature(name))
        if len(args) > (0 if name == "count" else 1) + (1 if name == "quantile" else 0):
            raise invalid(f"too many arguments for {name}()", _signature(name))

        if args:
            values = _series(self.visit(args[0]), name)
        elif name == "count":
            values = pd.Series(1, index=self.df.index)
        else:
            raise invalid(f"{name}() needs a column", _signature(name))

        q = 0.5
        if name == "quantile":
            raw = self.visit(args[1]) if len(args) > 1 else (self.visit(q_node) if q_node else 0.5)
            if not _is_number(raw) or not 0 <= float(raw) <= 1:
                raise invalid("quantile q must be between 0 and 1", "e.g. quantile(price, 0.9)")
            q = float(raw)

        by = None
        if by_node is not None:
            by = _series(self.visit(by_node), "by=")
        if where is not None:
            values = values[where]
            by = by[where] if by is not None else None

        def reduce(series: pd.Series) -> Any:
            if name == "count":
                return int(series.count())
            if name == "nunique":
                return int(series.nunique())
            if name == "quantile":
                return series.quantile(q)
            if pdt.is_bool_dtype(series.dtype) and name in ("sum", "mean"):
                series = series.astype("float64")
            return getattr(series, name)()

        try:
            if by is None:
                return reduce(values)
            grouped = values.groupby(by, observed=True, sort=False).agg(reduce)
        except TypeError:
            raise invalid(f"{name}() cannot summarise this column", _signature(name)) from None
        # Largest first, ties by group label, so row order cannot change it.
        keys = grouped.index.map(str)
        order = sorted(range(len(grouped)), key=lambda i: (_sort_key(grouped.iloc[i]), keys[i]))
        return Grouped(grouped.iloc[np.array(order, dtype=int)], name)

    def _rows(self, args: list[ast.expr], keywords: dict[str, ast.expr]) -> Rows:
        where = self._where(keywords)
        sort_node = keywords.pop("sort", None)
        desc_node = keywords.pop("desc", None)
        if keywords:
            raise invalid(f"rows() does not take {', '.join(keywords)}=", _signature("rows"))
        columns = [_series(self.visit(a), "rows") for a in args]
        frame = pd.concat(columns, axis=1) if columns else self.df.copy(deep=False)
        frame.columns = [str(c) for c in frame.columns]
        if where is not None:
            frame = frame[where]
        if sort_node is not None:
            key = _series(self.visit(sort_node), "sort=")
            descending = bool(self.visit(desc_node)) if desc_node is not None else False
            # Stable, so equal keys keep frame order.
            order = key.loc[frame.index].sort_values(
                ascending=not descending, kind="mergesort", na_position="last"
            )
            frame = frame.loc[order.index]
        return Rows(frame, len(frame))


def _is_text(value: Any) -> bool:
    if isinstance(value, str):
        return True
    return isinstance(value, pd.Series) and (
        pdt.is_string_dtype(value.dtype) or pdt.is_object_dtype(value.dtype)
    )


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.number)) and not isinstance(value, bool)


def _sort_key(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return -number if math.isfinite(number) else math.inf


def _symbol(op: ast.operator) -> str:
    return {
        ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/",
        ast.FloorDiv: "//", ast.Mod: "%", ast.Pow: "**",
    }.get(type(op), type(op).__name__)  # fmt: skip


def _describe(node: ast.AST) -> str:
    names = {
        "Attribute": "attribute access (x.y)",
        "Subscript": "indexing (x[...])",
        "Lambda": "lambda",
        "ListComp": "comprehensions",
        "SetComp": "comprehensions",
        "DictComp": "comprehensions",
        "GeneratorExp": "comprehensions",
        "NamedExpr": "assignment (:=)",
        "JoinedStr": "f-strings",
        "Starred": "* unpacking",
        "Await": "await",
        "Yield": "yield",
        "YieldFrom": "yield",
        "Dict": "dict literals",
        "Set": "set literals",
        "IfExp": "if/else expressions",
        "Slice": "slices",
    }
    return names.get(type(node).__name__, type(node).__name__)


SIGNATURES = {
    "rows": "rows(col, ..., where=cond, sort=col, desc=True)",
    "quantile": "quantile(col, q, where=cond, by=col)",
    "count": "count(col?, where=cond, by=col)",
    "round": "round(col, digits)",
    "between": "between(col, low, high)",
    "contains": 'contains(col, "text", case=False)',
    "startswith": 'startswith(col, "text")',
    "endswith": 'endswith(col, "text")',
}


def _signature(name: str) -> str:
    if name in SIGNATURES:
        return f"usage: {SIGNATURES[name]}"
    if name in AGGREGATES:
        return f"usage: {name}(col, where=cond, by=col)"
    return f"usage: {name}(col)"


# --------------------------------------------------------------------------
# presentation

# Share of the tool budget row listings may use; the envelope needs the rest.
ROWS_SHARE = 0.7
MAX_CELL_CHARS = 60


def jsonable(value: Any) -> Any:
    """One cell or scalar as a compact JSON value."""
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        # Precision is the encoder's decision (digest.compact), not the cell's.
        return value if math.isfinite(value) else None
    if isinstance(value, str) and len(value) > MAX_CELL_CHARS:
        return value[: MAX_CELL_CHARS - 1] + "…"
    if isinstance(value, (bool, int, str)):
        return value
    return jsonable(str(value))


def listing(frame: pd.DataFrame, limit: int, budget: int) -> tuple[dict[str, Any], int]:
    """Rows as a header plus value lists, cut whole-row at the budget."""
    columns = [str(c) for c in frame.columns]
    out: dict[str, Any] = {"columns": columns, "rows": []}
    spent = estimate_tokens(columns)
    for record in frame.head(limit).itertuples(index=False, name=None):
        row = [jsonable(v) for v in record]
        cost = estimate_tokens(row)
        if out["rows"] and spent + cost > budget:
            break
        out["rows"].append(row)
        spent += cost
    return out, len(out["rows"])


def present(
    result: Result, df: pd.DataFrame, expression: str, limit: int, budget: int
) -> tuple[dict[str, Any], str]:
    """Shape any evaluation result into a response body and a summary line."""
    room = int(budget * ROWS_SHARE)
    rows = len(df)

    if isinstance(result, pd.Series) and pdt.is_bool_dtype(result.dtype) and len(result) == rows:
        # A condition: answer "how many", then show the first matches.
        result = Rows(df[result.fillna(False).astype(bool)], int(result.fillna(False).sum()))

    if isinstance(result, Rows):
        rows_out, shown = listing(result.frame, limit, room)
        body = {"matched": result.matched, "of": rows, **rows_out}
        summary = f"{result.matched:,} of {rows:,} rows match ({result.matched / rows:.1%})"
        if shown < result.matched:
            summary += f"; showing the first {shown}"
        return body, summary + "."

    if isinstance(result, Grouped):
        values: dict[str, Any] = {}
        spent = 0
        for key, value in result.values.head(limit).items():
            entry = {str(key): jsonable(value)}
            if values and spent + estimate_tokens(entry) > room:
                break
            values.update(entry)
            spent += estimate_tokens(entry)
        body = {"groups": len(result.values), "values": values}
        summary = f"{result.function} for {len(result.values):,} group(s), largest first"
        if len(values) < len(result.values):
            summary += f"; showing {len(values)}"
        return body, summary + "."

    if isinstance(result, pd.Series):
        present_values = result.dropna()
        body = {"count": len(present_values), "missing": len(result) - len(present_values)}
        numeric = pdt.is_numeric_dtype(result.dtype) and not pdt.is_bool_dtype(result.dtype)
        if numeric and len(present_values):
            finite = present_values.astype("float64")
            body.update(
                {
                    "mean": jsonable(finite.mean()),
                    "min": jsonable(finite.min()),
                    "median": jsonable(finite.median()),
                    "max": jsonable(finite.max()),
                }
            )
        body["first_values"] = [jsonable(v) for v in result.head(min(limit, 10))]
        return body, f"Computed a column of {len(present_values):,} value(s)."

    value = jsonable(result)
    return {"result": value}, f"{expression} = {value}"


def present_sql(
    frame: pd.DataFrame, more: bool, limit: int, budget: int
) -> tuple[dict[str, Any], str]:
    """Shape a SQL result: a single value as itself, anything else as rows.

    The total row count is not known -- counting would run the query twice --
    so the response says only whether more rows exist than were returned.
    """
    if frame.shape == (1, 1) and not more:
        value = jsonable(frame.iloc[0, 0])
        return {"column": str(frame.columns[0]), "result": value}, f"{frame.columns[0]} = {value}"

    rows_out, shown = listing(frame, limit, int(budget * ROWS_SHARE))
    body: dict[str, Any] = {**rows_out, "returned": shown}
    summary = f"{shown} row(s)"
    if more or shown < len(frame):
        body["more_rows"] = True
        summary += "; more exist -- aggregate in SQL, or raise limit= (up to 100)"
    return body, summary + "."


def evaluate(df: pd.DataFrame, expression: str) -> Result:
    """Parse and evaluate *expression* against *df* without ``eval``."""
    if len(expression) > MAX_LENGTH:
        raise invalid("expression too long", f"keep it under {MAX_LENGTH} characters")

    # Backtick-quoted names become placeholders Python can parse.
    aliases: dict[str, str] = {}

    def placeholder(match: re.Match[str]) -> str:
        key = f"col_{len(aliases)}_"
        aliases[key] = match.group(1)
        return key

    source = BACKTICK.sub(placeholder, expression.strip())
    try:
        tree = ast.parse(source, mode="eval")
    except (RecursionError, MemoryError):
        # Pathological nesting can exhaust the parser itself.
        raise invalid("expression nested too deeply", "simplify the expression") from None
    except SyntaxError as exc:
        raise invalid(
            f"cannot parse the expression: {exc.msg}",
            "use Python syntax; quote column names with spaces in backticks",
        ) from None
    if sum(1 for _ in ast.walk(tree)) > MAX_NODES:
        raise invalid("expression too complex", f"keep it under {MAX_NODES} syntax nodes")
    return _Evaluator(df, aliases).visit(tree)


__all__ = ["FUNCTIONS", "Grouped", "Rows", "evaluate", "listing", "present", "present_sql"]
