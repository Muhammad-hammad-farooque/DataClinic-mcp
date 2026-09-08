"""Token budgeting and response encoding.

Every tool response passes through here. The encoding rules are not cosmetic:
a tool result re-enters the model's context on every later turn, so bytes
saved once are saved many times over.

Rules applied, in order of value:
  * findings are ranked terse lines, not nested objects
  * floats carry three significant figures, never float64 noise
  * null and absent fields are dropped entirely
  * emission stops at the budget, and says what it omitted

See spec sections 9.1 and 10.5.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

# Runtime budgeting uses a cheap character heuristic. The release benchmark
# (spec section 16.5) measures with the real tokenizer; this only needs to be
# close enough to decide where to stop emitting.
CHARS_PER_TOKEN = 4.0

BUDGETS: dict[str, int] = {
    "profile": 1500,
    "find_issues": 1200,
    "explore_schema": 1000,
    "analyze_target": 800,
    "check_relationships": 800,
    "load_dataset": 800,
    "analyze_column": 600,
}
DEFAULT_BUDGET = 500


class Severity(StrEnum):
    HIGH = "HIGH"
    MEDIUM = "MED"
    LOW = "LOW"
    INFO = "INFO"

    @property
    def rank(self) -> int:
        return {"HIGH": 0, "MED": 1, "LOW": 2, "INFO": 3}[self.value]


def estimate_tokens(value: Any) -> int:
    """Approximate token count for anything we might emit."""
    if isinstance(value, str):
        text = value
    else:
        import json

        text = json.dumps(value, default=str, ensure_ascii=False)
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def round_sig(value: float | int | None, digits: int = 3) -> float | int | None:
    """Round to *digits* significant figures.

    ``0.6939778779188857`` costs about seven tokens; ``0.694`` costs two, and
    no decision is made differently because of the discarded precision.
    """
    if value is None or isinstance(value, bool):
        return None if value is None else value
    if isinstance(value, int):
        return value
    if not math.isfinite(value):
        return None
    if value == 0:
        return 0.0
    rounded = round(value, -math.floor(math.log10(abs(value))) + (digits - 1))
    # Keep integral results integral so "47.0 outliers" reads as "47".
    if rounded.is_integer() and abs(rounded) < 1e15:
        return int(rounded)
    return rounded


def compact(obj: Any, digits: int = 3) -> Any:
    """Recursively drop empty values and round floats.

    Empty strings, empty containers and ``None`` are removed. ``False`` and
    ``0`` are kept: they are answers, not absences.
    """
    if isinstance(obj, dict):
        out = {}
        for key, value in obj.items():
            cleaned = compact(value, digits)
            if cleaned is None:
                continue
            if isinstance(cleaned, (str, list, dict, tuple)) and len(cleaned) == 0:
                continue
            out[key] = cleaned
        return out
    if isinstance(obj, (list, tuple)):
        items = [compact(v, digits) for v in obj]
        return [v for v in items if v is not None]
    if isinstance(obj, float):
        return round_sig(obj, digits)
    return obj


@dataclass(slots=True)
class Finding:
    """One observation about the data, rendered as a single line.

    A nested JSON object per finding costs roughly fifteen tokens in repeated
    keys alone; twenty findings pay that twenty times.
    """

    severity: Severity
    message: str
    column: str | None = None
    recommendation: str | None = None
    affected_rows: int | None = None

    def render(self) -> str:
        parts = [f"{self.severity.value:<4}"]
        if self.column:
            parts.append(f"{self.column}:")
        parts.append(self.message)
        if self.recommendation:
            parts.append(f"-> {self.recommendation}")
        line = " ".join(parts)
        if self.affected_rows is not None:
            line += f" ({self.affected_rows:,} rows)"
        return line


@dataclass(slots=True)
class Response:
    """A tool result assembled under a token budget."""

    tool: str
    body: dict[str, Any] = field(default_factory=dict)
    findings: list[Finding] = field(default_factory=list)
    summary: str | None = None
    resources: list[str] = field(default_factory=list)
    sampled: dict[str, Any] | None = None

    def add(self, finding: Finding) -> None:
        self.findings.append(finding)

    def build(self, budget: int | None = None) -> dict[str, Any]:
        """Render to the envelope in spec section 9.1, honouring the budget."""
        limit = budget if budget is not None else BUDGETS.get(self.tool, DEFAULT_BUDGET)

        payload: dict[str, Any] = compact(dict(self.body))
        if self.sampled:
            payload["sampled"] = compact(self.sampled)
        if self.summary:
            payload["summary"] = self.summary

        # Findings are the only elastic part, so they absorb whatever budget
        # the fixed fields leave behind.
        spent = estimate_tokens(payload)
        ordered = sorted(self.findings, key=lambda f: (f.severity.rank, -(f.affected_rows or 0)))

        emitted: list[str] = []
        omitted = 0
        for finding in ordered:
            line = finding.render()
            cost = estimate_tokens(line) + 2  # list punctuation
            if spent + cost > limit and emitted:
                omitted = len(ordered) - len(emitted)
                break
            emitted.append(line)
            spent += cost

        if emitted:
            payload["findings"] = emitted
        if omitted:
            payload["truncated"] = {
                "omitted": omitted,
                "reason": "token_budget",
                "remedy": "call again with severity= to narrow, or read the full resource",
            }
        if self.resources:
            payload["resources"] = self.resources
        return payload
