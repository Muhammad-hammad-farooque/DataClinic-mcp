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
from eda_mcp.digest import BUDGETS, DEFAULT_BUDGET, Finding, Response, Severity, fit
from eda_mcp.errors import ColumnNotFoundError, EDAError, ErrorCode, SourceTooLargeError
from eda_mcp.expressions import evaluate, present
from eda_mcp.instructions import INSTRUCTIONS
from eda_mcp.issues import (
    column_findings,
    encoding_advice,
    frame_findings,
    missingness_relations,
    needs_attention,
)
from eda_mcp.issues import find_issues as detect_issues
from eda_mcp.loaders import load_file
from eda_mcp.logging import configure, get_logger, new_correlation_id, tool_call
from eda_mcp.profiling import (
    ColumnKind,
    column_kinds,
    duplicated,
    orientation,
    profile_frame,
    summarise,
)
from eda_mcp.profiling import analyze_column as deep_profile
from eda_mcp.registry import Registry
from eda_mcp.relations import (
    MAX_GROUPS,
    REPORT_STRENGTH,
    all_pairs,
    collinearity_findings,
    compare_groups,
    target_pairs,
)
from eda_mcp.target import analyze_target as assess_target
from eda_mcp.target import task_for

# Behavioural hints let a client skip confirmation on safe calls. read_only
# means no *source* is modified; destructive is reserved for export, the one
# tool that can overwrite a file or replace a table.
READ_ONLY_EXTERNAL = ToolAnnotations(
    read_only_hint=True, idempotent_hint=True, open_world_hint=True
)
# In-memory analysis touches nothing outside the process (spec 7.7).
READ_ONLY_LOCAL = ToolAnnotations(read_only_hint=True, idempotent_hint=True, open_world_hint=False)
MUTATING = ToolAnnotations(read_only_hint=False, destructive_hint=False, idempotent_hint=False)

DETAIL_LEVELS = ("brief", "standard", "full")
# Column blocks are fixed-size and findings are one line each, so columns get
# a fixed share of the budget and findings absorb the rest.
PROFILE_COLUMN_SHARE = 0.6
ROLLUP_NAMES = 20
# Most rows query lists; beyond this an aggregate answers better.
MAX_QUERY_ROWS = 100
# Kinds that can be related to others, and kinds that can split rows into groups.
RELATABLE_KINDS = (
    ColumnKind.NUMERIC,
    ColumnKind.DATETIME,
    ColumnKind.CATEGORICAL,
    ColumnKind.BOOLEAN,
)
GROUPABLE_KINDS = (ColumnKind.CATEGORICAL, ColumnKind.BOOLEAN, ColumnKind.NUMERIC)
# The lowest severity rank each find_issues filter keeps.
SEVERITY_FLOORS = {
    "all": Severity.INFO.rank,
    "low": Severity.LOW.rank,
    "medium": Severity.MEDIUM.rank,
    "high": Severity.HIGH.rank,
}


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


def name_list(names: list[str]) -> str:
    """Comma-join names, capped so a wide table cannot flood the response."""
    shown = ", ".join(names[:ROLLUP_NAMES])
    extra = len(names) - ROLLUP_NAMES
    return f"{shown} (+{extra} more)" if extra > 0 else shown


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
        name="profile",
        description=(
            "Full-population statistics and ranked findings for a loaded dataset. "
            "detail='standard' shows problem columns in full and rolls up clean ones; "
            "'brief' gives findings only; 'full' shows every column. columns= narrows "
            "to named columns, shown in full."
        ),
        annotations=READ_ONLY_EXTERNAL,
    )
    def profile(
        source: str,
        detail: str = "standard",
        columns: list[str] | None = None,
    ) -> dict[str, Any]:
        try:
            with tool_call("profile", source=source, detail=detail) as record:
                if detail not in DETAIL_LEVELS:
                    raise EDAError(
                        ErrorCode.INVALID_OPERATION,
                        f"unknown detail level {detail!r}",
                        f"use one of {', '.join(DETAIL_LEVELS)}",
                    )
                df = registry.get_dataset(source).df
                kinds = column_kinds(df)
                profiles = profile_frame(df, kinds, columns=columns)

                by_column = {p.name: column_findings(p) for p in profiles}
                findings = [] if columns else frame_findings(df)
                findings.extend(f for fs in by_column.values() for f in fs)

                # Worst column first, so a budget cut drops the least important.
                def badness(name: str) -> tuple[int, int]:
                    fs = by_column[name]
                    worst = min((f.severity.rank for f in fs), default=Severity.INFO.rank)
                    return worst, -max((f.affected_rows or 0 for f in fs), default=0)

                problem = [p.name for p in profiles if needs_attention(by_column[p.name])]
                if detail == "brief":
                    shown: list[str] = []
                elif detail == "full" or columns:
                    shown = [p.name for p in profiles]
                else:
                    shown = problem
                shown = sorted(shown, key=badness)

                digests = {p.name: p.digest() for p in profiles}
                budget = int(BUDGETS["profile"] * PROFILE_COLUMN_SHARE)
                blocks, cut = fit({name: digests[name] for name in shown}, budget)

                body: dict[str, Any] = {
                    "dataset": source,
                    "shape": [int(df.shape[0]), int(df.shape[1])],
                }
                if not columns:
                    body["duplicate_rows"] = int(duplicated(df).sum())
                if blocks:
                    body["columns"] = blocks
                clean = [p.name for p in profiles if p.name not in problem]
                if clean and detail != "full" and not columns:
                    body["clean"] = f"{len(clean)} column(s) with no problems: {name_list(clean)}"
                if cut:
                    body["more_columns"] = {
                        "omitted": name_list(cut),
                        "remedy": "call again with columns=[...] naming them",
                    }

                response = Response(
                    "profile",
                    body=body,
                    findings=findings,
                    remedy="call again with columns=[...] to narrow",
                )
                if columns:
                    response.summary = (
                        f"{len(profiles)} of {df.shape[1]} columns, {df.shape[0]:,} rows. "
                        f"{len(problem)} need attention."
                    )
                else:
                    response.summary = summarise(df, kinds, findings)

                record["columns_shown"] = len(blocks)
                record["findings"] = len(findings)
                return response.build()
        except EDAError as exc:
            return exc.to_dict()
        except Exception as exc:  # the tool boundary must not raise
            return _unexpected(exc)

    @server.tool(
        name="find_issues",
        description=(
            "Every data-quality problem in a loaded dataset, ranked, each with the fix "
            "to apply. Adds cross-column checks profile does not run: missingness tied "
            "to other columns, identical columns, records repeated under new keys. "
            "severity is 'all', 'low', 'medium' or 'high' (the lowest level returned)."
        ),
        annotations=READ_ONLY_LOCAL,
    )
    def find_issues(source: str, severity: str = "all") -> dict[str, Any]:
        try:
            with tool_call("find_issues", source=source, severity=severity) as record:
                if severity not in SEVERITY_FLOORS:
                    raise EDAError(
                        ErrorCode.INVALID_OPERATION,
                        f"unknown severity {severity!r}",
                        f"use one of {', '.join(SEVERITY_FLOORS)}",
                    )
                df = registry.get_dataset(source).df
                kinds = column_kinds(df)
                every = detect_issues(df, kinds, profile_frame(df, kinds))
                floor = SEVERITY_FLOORS[severity]
                kept = [f for f in every if f.severity.rank <= floor]

                counts = {
                    level.value: sum(1 for f in every if f.severity is level)
                    for level in (Severity.HIGH, Severity.MEDIUM, Severity.LOW)
                }
                body: dict[str, Any] = {
                    "dataset": source,
                    "shape": [int(df.shape[0]), int(df.shape[1])],
                    "counts": {k: v for k, v in counts.items() if v},
                }
                response = Response(
                    "find_issues",
                    body=body,
                    findings=kept,
                    remedy="call again with severity='high' or 'medium' to narrow",
                )
                if every:
                    columns_hit = len({f.column for f in every if f.column})
                    response.summary = f"{len(every)} issue(s) across {columns_hit} column(s)."
                else:
                    response.summary = "No issues found."

                record["issues"] = len(every)
                record["returned"] = len(kept)
                return response.build()
        except EDAError as exc:
            return exc.to_dict()
        except Exception as exc:  # the tool boundary must not raise
            return _unexpected(exc)

    @server.tool(
        name="analyze_column",
        description=(
            "Deep dive on one column, adapted to its type: percentiles, histogram, "
            "normality and the best transform for numbers; full value list and encoding "
            "advice for categories; calendar breakdown and gaps for dates; key formats "
            "for identifiers. Includes the column's issues with fixes."
        ),
        annotations=READ_ONLY_LOCAL,
    )
    def analyze_column(source: str, column: str) -> dict[str, Any]:
        try:
            with tool_call("analyze_column", source=source, column=column) as record:
                df = registry.get_dataset(source).df
                labels = {str(c): c for c in df.columns}
                if column not in labels:
                    raise ColumnNotFoundError(column, list(labels))
                # Classified with the whole frame, so the kind matches profile's.
                kinds = column_kinds(df)
                kind = kinds[column]
                profile = deep_profile(df[labels[column]].rename(column), kind)

                relation = None
                if profile.missing:
                    relation = missingness_relations(df, kinds, [column]).get(column)
                findings = column_findings(profile, relation)

                body: dict[str, Any] = {"dataset": source, "column": column}
                body.update(profile.digest())
                encoding = encoding_advice(profile)
                if encoding:
                    body["encoding"] = encoding

                response = Response("analyze_column", body=body, findings=findings)
                serious = sum(1 for f in findings if f.severity is not Severity.LOW)
                response.summary = (
                    f"{column}: {kind.value}, {profile.count:,} values"
                    + (f", {profile.missing_pct:.0f}% missing" if profile.missing else "")
                    + (f". {serious} issue(s) to fix." if serious else ". Nothing to fix.")
                )

                record["kind"] = kind.value
                record["findings"] = len(findings)
                return response.build()
        except EDAError as exc:
            return exc.to_dict()
        except Exception as exc:  # the tool boundary must not raise
            return _unexpected(exc)

    @server.tool(
        name="check_relationships",
        description=(
            "Ranked relationships between columns -- never a full matrix. With no "
            "arguments: strongest pairs of any type plus collinear groups to prune. "
            "target= ranks every column's link to one column. group_by= compares every "
            "column across the groups of a categorical column, with effect sizes."
        ),
        annotations=READ_ONLY_LOCAL,
    )
    def check_relationships(
        source: str, target: str | None = None, group_by: str | None = None
    ) -> dict[str, Any]:
        try:
            with tool_call(
                "check_relationships", source=source, target=target, group_by=group_by
            ) as record:
                df = registry.get_dataset(source).df
                labels = {str(c): c for c in df.columns}
                for name in (target, group_by):
                    if name is not None and name not in labels:
                        raise ColumnNotFoundError(name, list(labels))
                kinds = column_kinds(df)
                body: dict[str, Any] = {"dataset": source, "shape": [*map(int, df.shape)]}

                if target is not None and kinds[target] not in RELATABLE_KINDS:
                    raise EDAError(
                        ErrorCode.INVALID_OPERATION,
                        f"{target} is {kinds[target].value}; it has no relationships to rank",
                        "choose a numeric, date or categorical column as target",
                    )

                if group_by is not None:
                    groups = int(df[labels[group_by]].nunique())
                    if kinds[group_by] not in GROUPABLE_KINDS or groups > MAX_GROUPS:
                        raise EDAError(
                            ErrorCode.INVALID_OPERATION,
                            f"{group_by} has {groups:,} distinct values; too many to group by",
                            f"group by a column with at most {MAX_GROUPS} values, "
                            "or bin this one first",
                        )
                    shown, findings = compare_groups(
                        df, kinds, group_by, [target] if target else None
                    )
                    body["group_by"] = group_by
                    body["groups"] = shown
                    summary = (
                        f"{len(findings)} column(s) differ meaningfully across the "
                        f"{groups} groups of {group_by}."
                    )
                elif target is not None:
                    pairs = target_pairs(df, kinds, target)
                    findings = [p.finding() for p in pairs]
                    body["target"] = target
                    summary = f"{len(pairs)} column(s) related to {target}" + (
                        f"; strongest {pairs[0].b} ({pairs[0].detail})." if pairs else "."
                    )
                else:
                    pairs, numeric, pearson, tested = all_pairs(df, kinds)
                    strong = [p for p in pairs if p.strength >= REPORT_STRENGTH]
                    findings = collinearity_findings(df, numeric, pearson)
                    clusters = len(findings)
                    findings.extend(p.finding() for p in strong)
                    body["pairs_tested"] = tested
                    summary = (
                        f"{len(strong)} strong relationship(s) among "
                        f"{sum(tested.values()):,} pairs tested"
                        + (f"; {clusters} collinearity problem(s)." if clusters else ".")
                    )

                response = Response(
                    "check_relationships",
                    body=body,
                    findings=findings,
                    remedy="call again with target= to focus on one column",
                )
                response.summary = summary
                record["findings"] = len(findings)
                return response.build()
        except EDAError as exc:
            return exc.to_dict()
        except Exception as exc:  # the tool boundary must not raise
            return _unexpected(exc)

    @server.tool(
        name="analyze_target",
        description=(
            "Assess a prediction target: classification or regression, class balance "
            "or skew, every feature ranked by strength, and leakage -- features that "
            "encode the answer. Run before modelling."
        ),
        annotations=READ_ONLY_LOCAL,
    )
    def analyze_target(source: str, target: str) -> dict[str, Any]:
        try:
            with tool_call("analyze_target", source=source, target=target) as record:
                df = registry.get_dataset(source).df
                labels = {str(c): c for c in df.columns}
                if target not in labels:
                    raise ColumnNotFoundError(target, list(labels))
                kinds = column_kinds(df)
                task = task_for(df[labels[target]], kinds[target])
                if task is None:
                    raise EDAError(
                        ErrorCode.INVALID_OPERATION,
                        f"{target} is {kinds[target].value}; it cannot be a prediction target",
                        "choose a numeric or categorical column with more than one value",
                    )

                report = assess_target(df, kinds, target, task)
                body: dict[str, Any] = {"dataset": source, "target": target, "task": report.task}
                body.update(report.body)

                response = Response(
                    "analyze_target",
                    body=body,
                    findings=report.findings,
                    remedy="the weakest features were cut; check_relationships target= lists all",
                )
                rows = report.body["rows_with_target"]
                if report.leaks:
                    lead = f"{len(report.leaks)} probable leak(s): {', '.join(report.leaks)}"
                else:
                    lead = "No leakage found"
                if report.ranked:
                    best = report.ranked[0]
                    tail = f"strongest feature {best.feature} ({best.detail})."
                else:
                    tail = "no feature shows a meaningful link."
                response.summary = f"{report.task}, {rows:,} rows. {lead}; {tail}"

                record["task"] = task
                record["leaks"] = len(report.leaks)
                return response.build()
        except EDAError as exc:
            return exc.to_dict()
        except Exception as exc:  # the tool boundary must not raise
            return _unexpected(exc)

    @server.tool(
        name="query",
        description=(
            "Ask a loaded dataset a precise question with a safe expression (Python "
            'syntax, no code execution). A condition -- price > 100 and country == "UK" '
            "-- counts and lists matching rows. Aggregates: count, sum, mean, median, "
            "min, max, std, nunique, quantile, each with where= and by=, e.g. "
            "mean(price, by=country). rows(col, ..., where=, sort=, desc=) picks columns. "
            "Quote names with spaces in backticks."
        ),
        annotations=READ_ONLY_EXTERNAL,
    )
    def query(source: str, expression: str, limit: int = 20) -> dict[str, Any]:
        try:
            # The expression itself is not logged: it can quote data values.
            with tool_call("query", source=source, expression_chars=len(expression)) as record:
                if not 1 <= limit <= MAX_QUERY_ROWS:
                    raise EDAError(
                        ErrorCode.INVALID_OPERATION,
                        f"limit must be between 1 and {MAX_QUERY_ROWS}",
                        "use an aggregate with by= to summarise more rows than that",
                    )
                df = registry.get_dataset(source).df
                try:
                    result = evaluate(df, expression)
                except (TypeError, ValueError) as exc:
                    # pandas rejecting an operation is the caller's to fix, not a bug.
                    raise EDAError(
                        ErrorCode.INVALID_OPERATION,
                        f"could not evaluate the expression: {str(exc)[:200]}",
                        "check that each operation suits the column's type",
                    ) from None

                budget = BUDGETS.get("query", DEFAULT_BUDGET)
                body, summary = present(result, df, expression, limit, budget)
                response = Response("query", body={"dataset": source, **body})
                response.summary = summary
                record["result"] = type(result).__name__
                return response.build()
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
