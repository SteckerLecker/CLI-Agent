"""
Einfacher MCP Server Manager
Laedt MCP Server Konfigurationen aus einem Ordner und verwaltet deren Lebenszyklus.
"""

import json
from pathlib import Path
from typing import Optional, Any
from dataclasses import dataclass, field
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


@dataclass
class MCPServerInstance:
    """Repraesentiert eine laufende MCP Server Instanz."""
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
    """Verwaltet mehrere MCP Server aus Konfigurationsdateien."""

    def __init__(self, config_dir: str = "mcp_configs"):
        self.config_dir = Path(config_dir)
        self.servers: dict[str, MCPServerInstance] = {}
        self._all_tools: list[dict] = []

    def load_configs(self) -> dict[str, dict]:
        """Laedt alle MCP Server Konfigurationen aus dem Konfig-Verzeichnis."""
        configs = {}

        if not self.config_dir.exists():
            print(f"[WARNUNG] Konfigurationsverzeichnis '{self.config_dir}' existiert nicht.")
            return configs

        for config_file in self.config_dir.glob("*.json"):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                if "mcpServers" in data:
                    for server_name, server_config in data["mcpServers"].items():
                        configs[server_name] = server_config
                        print(f"[INFO] Konfiguration geladen: {server_name} aus {config_file.name}")

            except json.JSONDecodeError as e:
                print(f"[FEHLER] Fehler beim Parsen von {config_file}: {e}")
            except Exception as e:
                print(f"[FEHLER] Fehler beim Laden von {config_file}: {e}")

        return configs

    async def start_server(self, name: str, config: dict) -> Optional[MCPServerInstance]:
        """Startet einen einzelnen MCP Server aus seiner Konfiguration."""
        command = config.get("command")
        args = config.get("args", [])
        env = config.get("env")

        if not command:
            print(f"[FEHLER] Server '{name}': Kein 'command' in der Konfiguration.")
            return None

        print(f"[INFO] Starte MCP Server: {name}...")

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
            print(f"[OK] MCP Server '{name}' gestartet. {len(server.tools)} Tools verfuegbar.")

            return server

        except Exception as e:
            print(f"[FEHLER] Fehler beim Starten von '{name}': {e}")
            return None

    async def start_all_servers(self) -> None:
        """Laedt alle Konfigurationen und startet alle MCP Server."""
        configs = self.load_configs()

        if not configs:
            print("[WARNUNG] Keine MCP-Server-Konfigurationen gefunden.")
            return

        print(f"[INFO] Starte {len(configs)} MCP Server...")

        for name, config in configs.items():
            await self.start_server(name, config)

        self._update_all_tools()
        print(f"[OK] Alle Server gestartet. Insgesamt {len(self._all_tools)} Tools verfuegbar.")

    async def stop_server(self, name: str) -> None:
        """Stoppt einen einzelnen MCP Server."""
        if name not in self.servers:
            return

        server = self.servers[name]
        print(f"[INFO] Stoppe MCP Server: {name}...")

        try:
            if server._session_context:
                await server._session_context.__aexit__(None, None, None)
            if server._stdio_context:
                await server._stdio_context.__aexit__(None, None, None)

            del self.servers[name]
            print(f"[OK] MCP Server '{name}' gestoppt.")

        except Exception as e:
            print(f"[FEHLER] Fehler beim Stoppen von '{name}': {e}")

    async def stop_all_servers(self) -> None:
        """Stoppt alle laufenden MCP Server."""
        server_names = list(self.servers.keys())

        for name in server_names:
            await self.stop_server(name)

        self._all_tools = []
        print("[OK] Alle MCP Server gestoppt.")

    def _update_all_tools(self) -> None:
        """Aktualisiert die kombinierte Liste aller Tools von allen Servern."""
        self._all_tools = []
        for server in self.servers.values():
            self._all_tools.extend(server.tools)

    def get_all_tools(self) -> list[dict]:
        """Gibt alle verfuegbaren Tools von allen Servern zurueck."""
        return self._all_tools

    def get_tools_for_openai(self) -> list[dict]:
        """Konvertiert alle MCP Tools ins OpenAI Function Calling Format."""
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
        """Gibt den Servernamen zurueck, der ein bestimmtes Tool bereitstellt."""
        for tool in self._all_tools:
            if tool["name"] == tool_name:
                return tool.get("server")
        return None

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Ruft ein Tool auf dem entsprechenden MCP Server auf."""
        server_name = self.get_server_for_tool(tool_name)

        if not server_name:
            raise ValueError(f"Tool '{tool_name}' nicht gefunden.")

        server = self.servers.get(server_name)
        if not server or not server.session:
            raise RuntimeError(f"Server '{server_name}' nicht verfuegbar.")

        result = await server.session.call_tool(tool_name, arguments)
        return result

    def list_servers(self) -> list[str]:
        """Gibt alle laufenden Servernamen zurueck."""
        return list(self.servers.keys())

    def get_server_tools(self, server_name: str) -> list[dict]:
        """Gibt die Tools fuer einen bestimmten Server zurueck."""
        server = self.servers.get(server_name)
        if server:
            return server.tools
        return []
