"""LightMCP - A FastAPI-inspired framework for building MCP servers."""

from lightmcp.app import LightMCP
from lightmcp.exceptions import LightMCPError

__version__ = "0.1.0"
__all__ = ["LightMCP", "LightMCPError"]