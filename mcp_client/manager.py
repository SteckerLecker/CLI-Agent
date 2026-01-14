"""
MCP Server Manager
Loads MCP server configurations from a folder and manages their lifecycle.
"""

import json
from pathlib import Path
from typing import Optional, Any
from dataclasses import dataclass, field
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from rich.console import Console


console = Console()


@dataclass
class MCPServerInstance:
    """Represents a running MCP server instance."""
    name: str
    command: str
    args: list[str]
    env: Optional[dict[str, str]] = None
    session: Optional[ClientSession] = None
    tools: list[dict] = field(default_factory=list)
    _stdio_context: Any = None
    _session_context: Any = None
    _read_stream: Any = None
    _write_stream: Any = None


class MCPManager:
    """Manages multiple MCP servers loaded from configuration files."""

    def __init__(self, config_dir: str = "mcp_configs"):
        self.config_dir = Path(config_dir)
        self.servers: dict[str, MCPServerInstance] = {}
        self._all_tools: list[dict] = []

    def load_configs(self) -> dict[str, dict]:
        """
        Load all MCP server configurations from the config directory.

        Returns:
            Dictionary mapping server names to their configurations.
        """
        configs = {}

        if not self.config_dir.exists():
            console.print(f"[yellow]Konfigurationsverzeichnis '{self.config_dir}' existiert nicht.[/yellow]")
            return configs

        for config_file in self.config_dir.glob("*.json"):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                if "mcpServers" in data:
                    for server_name, server_config in data["mcpServers"].items():
                        configs[server_name] = server_config
                        console.print(f"[dim]Konfiguration geladen: {server_name} aus {config_file.name}[/dim]")

            except json.JSONDecodeError as e:
                console.print(f"[red]Fehler beim Parsen von {config_file}: {e}[/red]")
            except Exception as e:
                console.print(f"[red]Fehler beim Laden von {config_file}: {e}[/red]")

        return configs

    async def start_server(self, name: str, config: dict) -> Optional[MCPServerInstance]:
        """
        Start a single MCP server from its configuration.

        Args:
            name: The server name.
            config: The server configuration dictionary.

        Returns:
            The MCPServerInstance if successful, None otherwise.
        """
        command = config.get("command")
        args = config.get("args", [])
        env = config.get("env")

        if not command:
            console.print(f"[red]Server '{name}': Kein 'command' in der Konfiguration.[/red]")
            return None

        console.print(f"[cyan]Starte MCP Server: {name}...[/cyan]")

        try:
            server = MCPServerInstance(
                name=name,
                command=command,
                args=args,
                env=env
            )

            server_params = StdioServerParameters(
                command=command,
                args=args,
                env=env
            )

            server._stdio_context = stdio_client(server_params)
            server._read_stream, server._write_stream = await server._stdio_context.__aenter__()

            server._session_context = ClientSession(server._read_stream, server._write_stream)
            server.session = await server._session_context.__aenter__()

            await server.session.initialize()

            tools_response = await server.session.list_tools()
            server.tools = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema,
                    "server": name
                }
                for tool in tools_response.tools
            ]

            self.servers[name] = server
            console.print(f"[green]MCP Server '{name}' gestartet. {len(server.tools)} Tools verfuegbar.[/green]")

            return server

        except Exception as e:
            console.print(f"[red]Fehler beim Starten von '{name}': {e}[/red]")
            return None

    async def start_all_servers(self) -> None:
        """Load all configurations and start all MCP servers."""
        configs = self.load_configs()

        if not configs:
            console.print("[yellow]Keine MCP-Server-Konfigurationen gefunden.[/yellow]")
            return

        console.print(f"[cyan]Starte {len(configs)} MCP Server...[/cyan]")

        for name, config in configs.items():
            await self.start_server(name, config)

        self._update_all_tools()
        console.print(f"[green]Alle Server gestartet. Insgesamt {len(self._all_tools)} Tools verfuegbar.[/green]")

    async def stop_server(self, name: str) -> None:
        """Stop a single MCP server."""
        if name not in self.servers:
            return

        server = self.servers[name]
        console.print(f"[cyan]Stoppe MCP Server: {name}...[/cyan]")

        try:
            if server._session_context:
                await server._session_context.__aexit__(None, None, None)
            if server._stdio_context:
                await server._stdio_context.__aexit__(None, None, None)

            del self.servers[name]
            console.print(f"[green]MCP Server '{name}' gestoppt.[/green]")

        except Exception as e:
            console.print(f"[red]Fehler beim Stoppen von '{name}': {e}[/red]")

    async def stop_all_servers(self) -> None:
        """Stop all running MCP servers."""
        server_names = list(self.servers.keys())

        for name in server_names:
            await self.stop_server(name)

        self._all_tools = []
        console.print("[green]Alle MCP Server gestoppt.[/green]")

    def _update_all_tools(self) -> None:
        """Update the combined list of all tools from all servers."""
        self._all_tools = []
        for server in self.servers.values():
            self._all_tools.extend(server.tools)

    def get_all_tools(self) -> list[dict]:
        """Get list of all available tools from all servers."""
        return self._all_tools

    def get_tools_for_openai(self) -> list[dict]:
        """Convert all MCP tools to OpenAI function calling format."""
        openai_tools = []
        for tool in self._all_tools:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description") or "",
                    "parameters": tool.get("input_schema") or {"type": "object", "properties": {}}
                }
            })
        return openai_tools

    def get_server_for_tool(self, tool_name: str) -> Optional[str]:
        """Get the server name that provides a specific tool."""
        for tool in self._all_tools:
            if tool["name"] == tool_name:
                return tool.get("server")
        return None

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """
        Call a tool on the appropriate MCP server.

        Args:
            tool_name: The name of the tool to call.
            arguments: The arguments to pass to the tool.

        Returns:
            The result from the MCP server.
        """
        server_name = self.get_server_for_tool(tool_name)

        if not server_name:
            raise ValueError(f"Tool '{tool_name}' nicht gefunden.")

        server = self.servers.get(server_name)
        if not server or not server.session:
            raise RuntimeError(f"Server '{server_name}' nicht verfuegbar.")

        result = await server.session.call_tool(tool_name, arguments)
        return result

    def list_servers(self) -> list[str]:
        """Get list of all running server names."""
        return list(self.servers.keys())

    def get_server_tools(self, server_name: str) -> list[dict]:
        """Get tools for a specific server."""
        server = self.servers.get(server_name)
        if server:
            return server.tools
        return []
