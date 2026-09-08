"""MCP server: tool registration and dispatch.

Tool handlers are thin. They validate, delegate to a module, and hand the
result to the digest layer; anything more belongs in the module. Every handler
converts a deliberate failure into an error envelope rather than letting it
reach the transport as a protocol error.

See spec sections 7 and 9.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

from eda_mcp import __version__
from eda_mcp.config import Settings, load_settings
from eda_mcp.digest import Finding, Response, Severity
from eda_mcp.errors import EDAError, ErrorCode, SourceTooLargeError
from eda_mcp.instructions import INSTRUCTIONS
from eda_mcp.loaders import load_file
from eda_mcp.logging import configure, get_logger, new_correlation_id, tool_call
from eda_mcp.profiling import column_kinds, orientation, summarise
from eda_mcp.registry import Registry

# Behavioural hints let a client skip confirmation on safe calls. read_only
# means no *source* is modified; destructive is reserved for export, the one
# tool that can overwrite a file or replace a table.
READ_ONLY_EXTERNAL = ToolAnnotations(
    read_only_hint=True, idempotent_hint=True, open_world_hint=True
)
MUTATING = ToolAnnotations(read_only_hint=False, destructive_hint=False, idempotent_hint=False)


def _unexpected(exc: Exception) -> dict[str, Any]:
    """Wrap an unforeseen failure without leaking internals to the caller."""
    correlation_id = new_correlation_id()
    get_logger().exception(
        "internal_error", extra={"correlation_id": correlation_id, "kind": type(exc).__name__}
    )
    return {
        "error": {
            "code": ErrorCode.INTERNAL_ERROR.value,
            "class": "bug",
            "message": "an unexpected error occurred",
            "retryable": False,
            "correlation_id": correlation_id,
        }
    }


def default_alias(source: str) -> str:
    """Derive a usable alias from a path when the caller does not supply one."""
    name = Path(source).name
    # ".csv" is an extension, not a name; Path.stem would return "csv".
    stem = "" if name.startswith(".") else Path(source).stem
    cleaned = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in stem).strip("_")
    return cleaned.lower() or "dataset"


def build_server(settings: Settings | None = None) -> MCPServer:
    """Create the server with the tool set the configuration selects."""
    settings = settings or load_settings()
    configure(settings.log_level)
    registry = Registry()

    server = MCPServer(
        name="dataclinic-mcp",
        title="EDA",
        version=__version__,
        instructions=INSTRUCTIONS,
    )

    @server.tool(
        name="load_dataset",
        description=(
            "Load a CSV, Excel, Parquet or JSON file into the session. Returns shape, "
            "column kinds, missing-data summary and the top findings -- do not call "
            "profile straight after this."
        ),
        annotations=READ_ONLY_EXTERNAL,
    )
    def load_dataset(
        source: str,
        alias: str | None = None,
        limit: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            with tool_call("load_dataset", source=source) as record:
                df, report = load_file(source, settings, limit=limit, options=options)

                if len(df) > settings.max_load_rows:
                    raise SourceTooLargeError(source, len(df), settings.max_load_rows)

                name = registry.unique_alias(alias or default_alias(source))
                dataset = registry.add_dataset(name, df, origin=source)

                kinds = column_kinds(df)
                body, findings = orientation(df, kinds)
                body["dataset"] = name
                body["origin"] = source

                read: dict[str, Any] = {"format": report.format}
                if report.encoding:
                    read["encoding"] = report.encoding
                if report.delimiter:
                    read["delimiter"] = report.delimiter
                if report.coerced_columns:
                    read["coerced"] = report.coerced_columns
                body["read"] = read

                response = Response("load_dataset", body=body)
                response.findings = findings
                for note in report.notes:
                    response.add(Finding(Severity.INFO, note))
                response.summary = summarise(df, kinds, findings)

                payload = response.build()
                record["rows"] = dataset.shape[0]
                record["cols"] = dataset.shape[1]
                record["alias"] = name
                return payload
        except EDAError as exc:
            return exc.to_dict()
        except Exception as exc:  # the tool boundary must not raise
            return _unexpected(exc)

    @server.tool(
        name="manage_sources",
        description=(
            "List the datasets and connections open in this session, or close one to "
            "free memory. action is 'list' or 'close'."
        ),
        annotations=MUTATING,
    )
    def manage_sources(action: str = "list", alias: str | None = None) -> dict[str, Any]:
        try:
            with tool_call("manage_sources", action=action) as record:
                if action == "close":
                    if not alias:
                        raise EDAError(
                            ErrorCode.INVALID_OPERATION,
                            "close requires an alias",
                            "pass alias= naming the source to release",
                        )
                    kind = registry.close(alias)
                    record["closed"] = alias
                    return {"closed": alias, "kind": kind}

                if action != "list":
                    raise EDAError(
                        ErrorCode.INVALID_OPERATION,
                        f"unknown action {action!r}",
                        "use action='list' or action='close'",
                    )

                datasets = [
                    {
                        "alias": d.alias,
                        "shape": list(d.shape),
                        "memory_mb": round(d.bytes / 1024**2, 1),
                        "origin": d.origin,
                        "mutations": len(d.history),
                    }
                    for d in registry.datasets.values()
                ]
                connections = [
                    {"alias": c.alias, "dialect": c.dialect, "read_only": c.read_only}
                    for c in registry.connections.values()
                ]
                record["datasets"] = len(datasets)

                body: dict[str, Any] = {}
                if datasets:
                    body["datasets"] = datasets
                if connections:
                    body["connections"] = connections
                if not body:
                    body["summary"] = "nothing open; start with load_dataset"
                return body
        except EDAError as exc:
            return exc.to_dict()
        except Exception as exc:  # the tool boundary must not raise
            return _unexpected(exc)

    get_logger().info(
        "server_ready",
        extra={"tools": settings.tools.value, "version": __version__},
    )
    return server


__all__ = ["build_server", "default_alias"]
