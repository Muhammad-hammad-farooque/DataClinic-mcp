"""Structured logging to stderr.

stdout carries the MCP protocol: writing anything there corrupts the stream,
so every record goes to stderr as one JSON object per line.

Records describe *what happened*, never what the data contains. No cell
values, no column contents, no credentials.

See spec section 14.
"""

from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from eda_mcp.errors import redact

LOGGER_NAME = "eda_mcp"

# Keys the stdlib LogRecord always carries; anything else was supplied by us
# and belongs in the emitted JSON.
_STANDARD = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "message",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "thread",
        "threadName",
        "taskName",
    }
)


class JSONFormatter(logging.Formatter):
    """Render a record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "event": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _STANDARD and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            # The type and message are useful; a traceback in the transcript is
            # noise and can carry paths we would rather not emit.
            exc_type, exc_value, _ = record.exc_info
            payload["exception"] = getattr(exc_type, "__name__", str(exc_type))
            payload["exception_message"] = redact(str(exc_value))
        return json.dumps(payload, default=str, ensure_ascii=False)


def configure(level: str = "INFO") -> logging.Logger:
    """Install the stderr handler. Safe to call more than once."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False  # never reach a root handler pointed at stdout
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


def new_correlation_id() -> str:
    """Short id tying a returned INTERNAL_ERROR back to a log record."""
    return uuid.uuid4().hex[:16]


@contextmanager
def tool_call(tool: str, **fields: Any) -> Iterator[dict[str, Any]]:
    """Time a tool call and emit one record when it finishes.

    Yields a dict the caller mutates to attach outcome fields (rows touched,
    result token count, truncation) which are merged into the final record.
    """
    logger = get_logger()
    correlation_id = new_correlation_id()
    extra: dict[str, Any] = {}
    started = time.perf_counter()
    try:
        yield extra
    except Exception as exc:
        logger.error(
            "tool_error",
            extra={
                "tool": tool,
                "duration_ms": round((time.perf_counter() - started) * 1000, 1),
                "correlation_id": correlation_id,
                "error_code": getattr(getattr(exc, "code", None), "value", "INTERNAL_ERROR"),
                **fields,
                **extra,
            },
            exc_info=True,
        )
        raise
    logger.info(
        "tool_call",
        extra={
            "tool": tool,
            "duration_ms": round((time.perf_counter() - started) * 1000, 1),
            "correlation_id": correlation_id,
            **fields,
            **extra,
        },
    )
