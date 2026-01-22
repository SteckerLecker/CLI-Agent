"""
Simple MCP Agent Package
Ein einfacher Agent der MCP Server bedienen kann.
"""

from .mcp_manager import MCPManager, MCPServerInstance
from .agent import SimpleMCPAgent

__all__ = ["MCPManager", "MCPServerInstance", "SimpleMCPAgent"]
