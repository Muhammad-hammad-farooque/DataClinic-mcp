"""Structured error taxonomy.

Errors are data, not prose: every failure carries a stable code, a human
message, the remedy the caller should try next, and whether retrying could
help. Tool handlers convert these into a response envelope rather than letting
exceptions escape to the transport.

See spec section 9.2.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Any


class ErrorClass(StrEnum):
    """Who is responsible for a failure, which decides how it is presented."""

    USER = "user"  # the request asked for something impossible
    POLICY = "policy"  # deliberately refused; see spec section 12.2
    INFRA = "infra"  # environment failed; may succeed on retry
    CONFIG = "config"  # the deployment is missing something
    BUG = "bug"  # our fault


class ErrorCode(StrEnum):
    SOURCE_NOT_FOUND = "SOURCE_NOT_FOUND"
    SOURCE_TOO_LARGE = "SOURCE_TOO_LARGE"
    COLUMN_NOT_FOUND = "COLUMN_NOT_FOUND"
    INVALID_OPERATION = "INVALID_OPERATION"
    UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"
    OPERATION_REFUSED = "OPERATION_REFUSED"
    WRITE_NOT_PERMITTED = "WRITE_NOT_PERMITTED"
    STATEMENT_REJECTED = "STATEMENT_REJECTED"
    PATH_NOT_ALLOWED = "PATH_NOT_ALLOWED"
    CONNECTION_FAILED = "CONNECTION_FAILED"
    QUERY_TIMEOUT = "QUERY_TIMEOUT"
    MEMORY_LIMIT_EXCEEDED = "MEMORY_LIMIT_EXCEEDED"
    DEPENDENCY_MISSING = "DEPENDENCY_MISSING"
    INTERNAL_ERROR = "INTERNAL_ERROR"


_CLASSES: dict[ErrorCode, ErrorClass] = {
    ErrorCode.SOURCE_NOT_FOUND: ErrorClass.USER,
    ErrorCode.SOURCE_TOO_LARGE: ErrorClass.USER,
    ErrorCode.COLUMN_NOT_FOUND: ErrorClass.USER,
    ErrorCode.INVALID_OPERATION: ErrorClass.USER,
    ErrorCode.UNSUPPORTED_FORMAT: ErrorClass.USER,
    ErrorCode.OPERATION_REFUSED: ErrorClass.POLICY,
    ErrorCode.WRITE_NOT_PERMITTED: ErrorClass.POLICY,
    ErrorCode.STATEMENT_REJECTED: ErrorClass.POLICY,
    ErrorCode.PATH_NOT_ALLOWED: ErrorClass.POLICY,
    ErrorCode.CONNECTION_FAILED: ErrorClass.INFRA,
    ErrorCode.QUERY_TIMEOUT: ErrorClass.INFRA,
    ErrorCode.MEMORY_LIMIT_EXCEEDED: ErrorClass.INFRA,
    ErrorCode.DEPENDENCY_MISSING: ErrorClass.CONFIG,
    ErrorCode.INTERNAL_ERROR: ErrorClass.BUG,
}

# Only infrastructure faults are worth trying again; a refusal or a bad
# argument will fail identically the second time.
_RETRYABLE = {ErrorCode.CONNECTION_FAILED, ErrorCode.QUERY_TIMEOUT}

# Anything resembling a credential is scrubbed before a message is emitted,
# because messages reach both the transcript and the logs.
_SECRET_PATTERNS = (
    # A DSN passed where a path was expected is normalised by pathlib, which
    # turns "postgresql://" into "postgresql:\". Match any slash arrangement
    # so the mangled form is redacted too.
    re.compile(
        r"(?i)\b(?:postgresql|postgres|mysql|mssql|mongodb|redis|amqp)"
        r"(?:\+\w+)?:[\/]{0,2}\S+"
    ),
    # user:password@host in any scheme, including after path mangling.
    re.compile(r"\S*:[^\s:@/]+@\S+"),
    re.compile(r"(?i)\b(password|passwd|pwd|token|secret|api[_-]?key)\s*[=:]\s*\S+"),
    re.compile(r"\bghp_[A-Za-z0-9]{20,}\b"),
)


def redact(text: str) -> str:
    """Remove anything credential-shaped from a message.

    Applied at the boundary rather than at call sites, so a new error path
    cannot accidentally leak a DSN.
    """
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub("[redacted]", text)
    return text


class EDAError(Exception):
    """Base for every failure the server reports deliberately."""

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        remedy: str | None = None,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.code = code
        self.message = redact(message)
        self.remedy = redact(remedy) if remedy else None
        self.details = details or {}
        super().__init__(self.message)

    @property
    def error_class(self) -> ErrorClass:
        return _CLASSES[self.code]

    @property
    def retryable(self) -> bool:
        return self.code in _RETRYABLE

    def to_dict(self) -> dict[str, Any]:
        """Render as the error envelope described in spec section 9.2."""
        payload: dict[str, Any] = {
            "code": self.code.value,
            "class": self.error_class.value,
            "message": self.message,
            "retryable": self.retryable,
        }
        if self.remedy:
            payload["remedy"] = self.remedy
        if self.details:
            payload["details"] = self.details
        return {"error": payload}


class SourceNotFoundError(EDAError):
    def __init__(self, alias: str, known: list[str] | None = None) -> None:
        remedy = f"open sources: {', '.join(known)}" if known else "load one with load_dataset"
        super().__init__(ErrorCode.SOURCE_NOT_FOUND, f"no source named {alias!r}", remedy)


class SourceTooLargeError(EDAError):
    def __init__(self, name: str, rows: int, limit: int) -> None:
        super().__init__(
            ErrorCode.SOURCE_TOO_LARGE,
            f"{name} has ~{rows:,} rows, above max_load_rows ({limit:,})",
            "use profile() for push-down analysis, or pass limit=",
            details={"rows": rows, "limit": limit},
        )


class ColumnNotFoundError(EDAError):
    def __init__(self, column: str, available: list[str]) -> None:
        # Listing every column of a wide frame would blow the budget for a
        # message the caller only needs a hint from.
        shown = ", ".join(available[:15])
        if len(available) > 15:
            shown += f", ... (+{len(available) - 15} more)"
        super().__init__(
            ErrorCode.COLUMN_NOT_FOUND,
            f"no column named {column!r}",
            f"available: {shown}",
        )


class UnsupportedFormatError(EDAError):
    def __init__(self, suffix: str, supported: list[str]) -> None:
        super().__init__(
            ErrorCode.UNSUPPORTED_FORMAT,
            f"cannot read {suffix or 'files without an extension'}",
            f"supported: {', '.join(supported)}",
        )


class DependencyMissingError(EDAError):
    def __init__(self, package: str, extra: str) -> None:
        super().__init__(
            ErrorCode.DEPENDENCY_MISSING,
            f"{package} is required for this operation but is not installed",
            f"install it with: pip install dataclinic-mcp[{extra}]",
        )


class PathNotAllowedError(EDAError):
    def __init__(self, path: str, allowed: list[str]) -> None:
        super().__init__(
            ErrorCode.PATH_NOT_ALLOWED,
            f"{path} is outside the allowed paths",
            f"allowed roots: {', '.join(allowed)}; set EDA_MCP_ALLOWED_PATHS to widen",
        )


class OperationRefusedError(EDAError):
    """A deliberate refusal under the policy in spec section 12.2."""

    def __init__(self, what: str, why: str, instead: str) -> None:
        super().__init__(ErrorCode.OPERATION_REFUSED, f"refused {what}: {why}", instead)


class StatementRejectedError(EDAError):
    """SQL refused by the statement guard before reaching a driver (spec 6.3)."""

    def __init__(self, why: str) -> None:
        super().__init__(
            ErrorCode.STATEMENT_REJECTED,
            f"SQL rejected: {why}",
            "send a single read-only SELECT (or WITH ... SELECT) statement",
        )


class MemoryLimitExceededError(EDAError):
    def __init__(self, needed_mb: float, limit_mb: int) -> None:
        super().__init__(
            ErrorCode.MEMORY_LIMIT_EXCEEDED,
            f"operation needs ~{needed_mb:,.0f} MB, above max_memory_mb ({limit_mb:,})",
            "load fewer rows or columns, or raise EDA_MCP_MAX_MEMORY_MB",
        )
