"""Session state: the datasets and connections a conversation is working on.

One registry lives for the life of the server process. Datasets are held in
memory and never written back to their origin, so a session can be abandoned
at any point without touching the user's files.

See spec sections 5.2 and 13.2.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from eda_mcp.errors import SourceNotFoundError


@dataclass(slots=True)
class Operation:
    """One recorded mutation, for the audit trail and for undo."""

    name: str
    params: dict[str, Any]
    rows_before: int
    rows_after: int
    cols_before: int
    cols_after: int
    at: float = field(default_factory=time.time)
    undoable: bool = True

    def describe(self) -> str:
        delta_rows = self.rows_after - self.rows_before
        delta_cols = self.cols_after - self.cols_before
        changes = []
        if delta_rows:
            changes.append(f"{delta_rows:+,} rows")
        if delta_cols:
            changes.append(f"{delta_cols:+,} cols")
        suffix = f" ({', '.join(changes)})" if changes else ""
        return f"{self.name}{suffix}"


@dataclass(slots=True)
class Dataset:
    """A frame under analysis, plus everything done to it so far."""

    alias: str
    df: pd.DataFrame
    origin: str
    loaded_at: float = field(default_factory=time.time)
    history: list[Operation] = field(default_factory=list)
    # Snapshots back the undo stack. Bounded by count and by total bytes so a
    # long cleaning session cannot exhaust memory.
    snapshots: list[pd.DataFrame] = field(default_factory=list)

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.df.shape[0]), int(self.df.shape[1]))

    @property
    def bytes(self) -> int:
        return int(self.df.memory_usage(deep=True).sum())

    def snapshot(self, max_total_mb: int, max_depth: int = 10) -> None:
        """Record the current frame so the next mutation can be undone.

        Eviction is visible rather than silent: when the budget is exhausted
        the oldest entry is dropped and its history record is marked
        ``undoable: false``, so ``history`` never claims a reversal it cannot
        perform.
        """
        if max_total_mb <= 0:
            return
        self.snapshots.append(self.df.copy(deep=True))

        limit = max_total_mb * 1024 * 1024
        while self.snapshots and (
            len(self.snapshots) > max_depth
            or sum(int(s.memory_usage(deep=True).sum()) for s in self.snapshots) > limit
        ):
            self.snapshots.pop(0)
            dropped = len(self.history) - len(self.snapshots)
            for op in self.history[:dropped]:
                op.undoable = False
            if len(self.snapshots) <= 1:
                break

    def restore(self, steps: int = 1) -> int:
        """Roll back up to *steps* mutations. Returns how many were undone."""
        undone = 0
        while undone < steps and self.snapshots:
            self.df = self.snapshots.pop()
            if self.history:
                self.history.pop()
            undone += 1
        return undone


@dataclass(slots=True)
class Connection:
    """A live database handle.

    ``credentials`` names where the DSN came from -- safe to show -- and the
    DSN itself is never stored here: the engine holds it, and nothing reads
    it back out.
    """

    alias: str
    dialect: str
    engine: Any
    read_only: bool = True
    version: str = ""
    database: str = ""
    credentials: str = ""
    opened_at: float = field(default_factory=time.time)


@dataclass(slots=True)
class Registry:
    """Everything the current session has open."""

    datasets: dict[str, Dataset] = field(default_factory=dict)
    connections: dict[str, Connection] = field(default_factory=dict)

    def add_dataset(self, alias: str, df: pd.DataFrame, origin: str) -> Dataset:
        dataset = Dataset(alias=alias, df=df, origin=origin)
        self.datasets[alias] = dataset
        return dataset

    def add_connection(self, connection: Connection) -> Connection:
        self.connections[connection.alias] = connection
        return connection

    def get_connection(self, alias: str) -> Connection:
        try:
            return self.connections[alias]
        except KeyError:
            raise SourceNotFoundError(alias, sorted(self.connections)) from None

    def get_dataset(self, alias: str) -> Dataset:
        try:
            return self.datasets[alias]
        except KeyError:
            raise SourceNotFoundError(alias, sorted(self.datasets)) from None

    def close(self, alias: str) -> str:
        """Release a dataset or connection, returning what kind it was."""
        if alias in self.datasets:
            del self.datasets[alias]
            return "dataset"
        connection = self.connections.pop(alias, None)
        if connection is not None:
            dispose = getattr(connection.engine, "dispose", None)
            if callable(dispose):
                dispose()
            return "connection"
        raise SourceNotFoundError(alias, sorted(self.datasets) + sorted(self.connections))

    def close_all(self) -> None:
        """Release every handle. Called on shutdown."""
        for alias in list(self.connections):
            self.close(alias)
        self.datasets.clear()

    def unique_alias(self, preferred: str) -> str:
        """Return *preferred*, suffixed if that name is already taken."""
        if preferred not in self.datasets and preferred not in self.connections:
            return preferred
        n = 2
        while f"{preferred}_{n}" in self.datasets or f"{preferred}_{n}" in self.connections:
            n += 1
        return f"{preferred}_{n}"
