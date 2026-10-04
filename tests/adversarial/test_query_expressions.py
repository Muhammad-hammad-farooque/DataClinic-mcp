"""Adversarial tests: query expressions must not escape the interpreter.

Every case here is a known way out of a naive ``eval`` sandbox, or a way to
exhaust memory or time. Each must come back as an error envelope -- refused
by policy or rejected as invalid -- and none may run.

See spec sections 12.1 and 16.2.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pandas as pd
import pytest

from eda_mcp.config import load_settings
from eda_mcp.errors import EDAError, ErrorCode
from eda_mcp.expressions import evaluate
from eda_mcp.server import build_server

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"

FRAME = pd.DataFrame({"price": [1.0, 2.0, 3.0], "name": ["a", "b", "c"]})

ESCAPES = [
    # attribute walks to object, subclasses, builtins
    "price.__class__",
    "().__class__.__bases__[0].__subclasses__()",
    "price.__class__.__mro__",
    "name.str.cat",
    "price.values",
    "price.to_csv('out.csv')",
    # imports and builtins
    "__import__('os')",
    "__import__('os').system('echo pwned')",
    "__builtins__",
    "eval('1+1')",
    "exec('x=1')",
    "open('secrets.txt')",
    "getattr(price, '__class__')",
    "globals()",
    "locals()",
    "vars()",
    "compile('1', 'x', 'eval')",
    "breakpoint()",
    "input()",
    # syntax that could carry code
    "lambda: 1",
    "(lambda: __import__('os'))()",
    "[x for x in price]",
    "{x for x in price}",
    "{x: 1 for x in price}",
    "(x for x in price)",
    "(x := 1)",
    "f'{price}'",
    "price[0]",
    "price[::-1]",
    "{'a': 1}",
    "{1, 2}",
    "price if price else name",
    "price is None",
    "*price",
    "await price",
    "yield price",
    # names that do not exist in the language
    "_",
    "__name__",
    "_private(price)",
    "price()",
    "mean(price, **{'by': name})",
    "mean(*[price])",
    # statements are not expressions
    "import os",
    "x = 1",
    "del price",
    "price; __import__('os')",
]

BOMBS = [
    "'a' * 10 ** 9",
    "'a' * 1000000000",
    "name * 1000000000",
    "1000000000 * name",
    "name * price",
    "'%s' % name",
    "'" + "a" * 1_001 + "'",
    "price + " * 300 + "price",
    "-" * 1_500 + "price",  # deep nesting: too many syntax nodes
    "(" * 1_000 + "price" + ")" * 1_000,  # past the parser's own nesting limit
    "x" * 2_001,
]


@pytest.mark.parametrize("expression", ESCAPES)
def test_escape_attempts_are_rejected(expression: str) -> None:
    with pytest.raises(EDAError) as caught:
        evaluate(FRAME, expression)
    assert caught.value.code in (ErrorCode.OPERATION_REFUSED, ErrorCode.INVALID_OPERATION)


@pytest.mark.parametrize("expression", BOMBS)
def test_resource_bombs_are_rejected_quickly(expression: str) -> None:
    start = time.perf_counter()
    with pytest.raises(EDAError):
        evaluate(FRAME, expression)
    assert time.perf_counter() - start < 1.0


def test_huge_powers_overflow_instead_of_hanging() -> None:
    start = time.perf_counter()
    assert evaluate(FRAME, "10 ** 10 ** 10") == float("inf")
    assert evaluate(FRAME, "max(price ** 10 ** 10)") == float("inf")
    assert time.perf_counter() - start < 1.0


def test_dunder_names_are_refused_as_policy() -> None:
    for expression in ("__import__('os')", "_private(price)", "price.__class__"):
        with pytest.raises(EDAError) as caught:
            evaluate(FRAME, expression)
        assert caught.value.code is ErrorCode.OPERATION_REFUSED, expression


def test_nothing_ran_and_no_file_was_written(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.chdir(tmp_path)
    for expression in ("price.to_csv('out.csv')", "open('out.csv', 'w')"):
        with pytest.raises(EDAError):
            evaluate(FRAME, expression)
    assert list(tmp_path.iterdir()) == []


def test_escapes_through_the_tool_return_envelopes() -> None:
    settings = load_settings(log_level="WARNING", allowed_paths=[FIXTURES.parent])
    server = build_server(settings)

    def call(tool: str, args: dict) -> dict:  # type: ignore[type-arg]
        result = asyncio.run(server.call_tool(tool, args))
        payload = result[1] if isinstance(result, tuple) else result
        return payload.structured_content  # type: ignore[no-any-return]

    call("load_dataset", {"source": str(FIXTURES / "messy.csv"), "alias": "m"})
    for expression in ESCAPES + BOMBS:
        payload = call("query", {"source": "m", "expression": expression})
        assert payload["error"]["code"] in ("OPERATION_REFUSED", "INVALID_OPERATION"), expression
        assert payload["error"]["code"] != "INTERNAL_ERROR"
