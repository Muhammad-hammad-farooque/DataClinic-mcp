"""EDA MCP Server -- exploratory data analysis over MCP."""

from __future__ import annotations

__version__ = "0.1.0"


def main() -> None:
    """Console entry point: run the server on stdio."""
    from eda_mcp.server import build_server

    build_server().run(transport="stdio")


__all__ = ["__version__", "main"]
