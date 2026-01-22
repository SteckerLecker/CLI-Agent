#!/usr/bin/env python3
"""
Simple MCP Agent - Einstiegspunkt
Startet MCP Server aus Konfigurationen und startet den Agent.
"""

import asyncio
import os
from pathlib import Path

from mcp_manager import MCPManager
from agent import SimpleMCPAgent


async def main():
    """Hauptfunktion - startet MCP Server und den Agent."""

    # Pfad zum Konfigurationsverzeichnis (relativ zum Script)
    script_dir = Path(__file__).parent
    config_dir = script_dir / "mcp_configs"

    print(f"[INFO] Konfigurationsverzeichnis: {config_dir}")

    # MCP Manager erstellen und Server starten
    mcp_manager = MCPManager(config_dir=str(config_dir))

    try:
        # Alle MCP Server aus den Konfigurationen starten
        await mcp_manager.start_all_servers()

        # Agent erstellen
        # Konfiguration kann ueber Umgebungsvariablen angepasst werden
        agent = SimpleMCPAgent(
            mcp_manager=mcp_manager,
            base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1"),
            model=os.getenv("OPENAI_MODEL", "qwen3:8b"),
            api_key=os.getenv("OPENAI_API_KEY", "ollama")
        )

        # System-Prompt aktualisieren nachdem Server gestartet wurden
        agent.refresh_system_prompt()

        # Interaktive Chat-Schleife starten
        await agent.run_interactive()

    finally:
        # Alle MCP Server beim Beenden stoppen
        await mcp_manager.stop_all_servers()


if __name__ == "__main__":
    asyncio.run(main())
