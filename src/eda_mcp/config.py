"""Runtime configuration.

Precedence is environment variable, then ~/.eda-mcp/config.toml, then the
default. Validation happens once at startup so a bad value fails immediately
with the offending key named, rather than surfacing mid-analysis.

See spec section 15.
"""

from __future__ import annotations

import sys
import tomllib
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

CONFIG_PATH = Path.home() / ".eda-mcp" / "config.toml"


class ToolGroup(StrEnum):
    """Which tools are registered at startup.

    Fixed for the session: changing the tool list mid-session invalidates the
    client's prompt cache and costs more than the schema it saves.
    """

    CORE = "core"  # files, analysis, cleaning, export
    FULL = "full"  # adds database, plots, reports
    READONLY = "readonly"  # analysis only, no mutation or write


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="EDA_MCP_",
        case_sensitive=False,
        extra="ignore",
    )

    tools: ToolGroup = ToolGroup.FULL

    max_load_rows: int = Field(default=5_000_000, gt=0)
    max_memory_mb: int = Field(default=4096, gt=0)
    max_snapshot_mb: int = Field(default=500, ge=0)
    statement_timeout: int = Field(default=30, gt=0)
    sample_size: int = Field(default=100_000, gt=0)
    seed: int = 42

    allowed_paths: list[Path] = Field(default_factory=lambda: [Path.cwd()])
    log_level: str = "INFO"

    @field_validator("allowed_paths", mode="before")
    @classmethod
    def _split_paths(cls, value: Any) -> Any:
        """Accept an os-path-separator-delimited string from the environment."""
        if isinstance(value, str):
            sep = ";" if sys.platform == "win32" else ":"
            return [Path(p).expanduser() for p in value.split(sep) if p.strip()]
        return value

    @field_validator("allowed_paths")
    @classmethod
    def _resolve_paths(cls, value: list[Path]) -> list[Path]:
        return [p.expanduser().resolve() for p in value]

    @field_validator("log_level")
    @classmethod
    def _check_level(cls, value: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR"}
        upper = value.upper()
        if upper not in allowed:
            raise ValueError(f"must be one of {sorted(allowed)}")
        return upper

    def path_allowed(self, path: Path) -> bool:
        """True when *path* lies under a configured root.

        Resolved first so that ``..`` segments cannot escape the root.
        """
        try:
            resolved = path.expanduser().resolve()
        except (OSError, RuntimeError):
            return False
        return any(resolved == root or root in resolved.parents for root in self.allowed_paths)


def _from_toml(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Read the ``[settings]`` table, ignoring an absent or unreadable file."""
    if not path.is_file():
        return {}
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError):
        return {}
    settings = data.get("settings", {})
    return settings if isinstance(settings, dict) else {}


def load_settings(**overrides: Any) -> Settings:
    """Build settings from file defaults, environment, and explicit overrides.

    pydantic-settings already prefers the environment over values passed as
    defaults here, so the file supplies only what the environment omits.
    """
    return Settings(**{**_from_toml(), **overrides})
