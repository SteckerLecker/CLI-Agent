"""
Main entry point for the KI-Agent with Human-in-the-Loop.
Loads MCP servers from configuration files and includes FileSystem Sub-Agent.
"""

import asyncio
import click
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from mcp_client import MCPManager
from agents import HumanInTheLoopAgent, FileSystemAgent


console = Console()

# Default configuration directory
DEFAULT_CONFIG_DIR = Path(__file__).parent / "mcp_configs"


# ============== MCP Server Management ==============

async def start_mcp_servers(config_dir: str = None) -> MCPManager:
    """
    Separate method to start all MCP servers from configuration files.

    Args:
        config_dir: Path to the configuration directory. Defaults to ./mcp_configs

    Returns:
        The initialized MCPManager with all servers started.
    """
    if config_dir is None:
        config_dir = str(DEFAULT_CONFIG_DIR)

    console.print(f"[cyan]Lade MCP Server Konfigurationen aus: {config_dir}[/cyan]")
    mcp_manager = MCPManager(config_dir=config_dir)
    await mcp_manager.start_all_servers()
    return mcp_manager


async def stop_mcp_servers(mcp_manager: MCPManager) -> None:
    """
    Separate method to stop all MCP servers.
    """
    await mcp_manager.stop_all_servers()


# ============== Main Agent (MCP Servers + FileSystem) ==============

async def run_agent_interactive(config_dir: str = None):
    """Run the full agent in interactive mode with all MCP servers."""
    mcp_manager = await start_mcp_servers(config_dir)

    try:
        agent = HumanInTheLoopAgent(mcp_manager)
        await agent.chat_loop()
    finally:
        await stop_mcp_servers(mcp_manager)


async def run_agent_single_task(task: str, config_dir: str = None):
    """Run the full agent with a single task."""
    mcp_manager = await start_mcp_servers(config_dir)

    try:
        agent = HumanInTheLoopAgent(mcp_manager)
        result = await agent.run(task)
        return result
    finally:
        await stop_mcp_servers(mcp_manager)


# ============== FileSystem Sub-Agent Only ==============

async def run_filesystem_agent_interactive():
    """Run only the FileSystem Sub-Agent in interactive mode."""
    from rich.prompt import Confirm

    fs_agent = FileSystemAgent()
    conversation_history = []

    console.print(Panel(
        "[bold]Willkommen beim FileSystem-Agenten![/bold]\n\n"
        "Dieser Agent kann:\n"
        "  - Dateien lesen, schreiben, loeschen, kopieren, verschieben\n"
        "  - Verzeichnisse erstellen, auflisten, durchsuchen\n\n"
        "Schreibende/loeschende Aktionen erfordern Bestaetigung.\n\n"
        "[dim]Befehle:[/dim]\n"
        "[dim]  /new, /clear - Neuen Chat starten[/dim]\n"
        "[dim]  exit, quit   - Beenden[/dim]",
        title="[bold magenta]FileSystem Sub-Agent[/bold magenta]",
        border_style="magenta"
    ))

    while True:
        try:
            user_input = Prompt.ask("\n[bold blue]Sie[/bold blue]")

            if user_input.lower() in ["exit", "quit", "q"]:
                console.print("[yellow]Auf Wiedersehen![/yellow]")
                break

            # New chat commands
            if user_input.lower() in ["/new", "/clear"]:
                if conversation_history:
                    confirm = Confirm.ask(
                        "[yellow]Moechten Sie den aktuellen Chat wirklich loeschen?[/yellow]",
                        default=False
                    )
                    if confirm:
                        conversation_history.clear()
                        console.print("[green]Chat-Verlauf geloescht. Neuer Chat gestartet.[/green]")
                else:
                    console.print("[dim]Chat-Verlauf ist bereits leer.[/dim]")
                continue

            if not user_input.strip():
                continue

            conversation_history.append(user_input)
            await fs_agent.run(user_input)

        except KeyboardInterrupt:
            console.print("\n[yellow]Abgebrochen. Auf Wiedersehen![/yellow]")
            break
        except Exception as e:
            console.print(f"[red]Fehler: {e}[/red]")


async def run_filesystem_agent_single_task(task: str):
    """Run the FileSystem Sub-Agent with a single task."""
    fs_agent = FileSystemAgent()
    result = await fs_agent.run(task)
    return result


# ============== CLI Entry Point ==============

@click.command()
@click.option("--task", "-t", default=None, help="Single task to execute")
@click.option("--interactive", "-i", is_flag=True, help="Run in interactive mode")
@click.option("--filesystem-only", "-f", is_flag=True, help="Use only FileSystem agent (no MCP servers)")
@click.option("--config-dir", "-c", default=None, help="Path to MCP server config directory")
def main(task: str, interactive: bool, filesystem_only: bool, config_dir: str):
    """
    KI-Agent mit Human-in-the-Loop.

    Features:
    - Automatisches Laden von MCP Servern aus Konfigurationsdateien
    - FileSystem Sub-Agent fuer Dateioperationen
    - Human-in-the-Loop fuer gefaehrliche Aktionen

    Konfiguration:
    - MCP Server werden aus ./mcp_configs/*.json geladen
    - Jede JSON-Datei kann mehrere Server definieren

    Beispiele:
        python main.py -i                            # Interaktiver Modus
        python main.py -f -i                         # Nur FileSystem Agent
        python main.py -t "Oeffne google.com"        # Einzelne Aufgabe
        python main.py -c /pfad/zu/configs -i       # Anderer Config-Ordner
    """
    if filesystem_only:
        if task:
            asyncio.run(run_filesystem_agent_single_task(task))
        else:
            asyncio.run(run_filesystem_agent_interactive())
    else:
        if task:
            asyncio.run(run_agent_single_task(task, config_dir))
        else:
            asyncio.run(run_agent_interactive(config_dir))


if __name__ == "__main__":
    main()
