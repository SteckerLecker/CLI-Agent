"""
KI-Agent with Human-in-the-Loop
Uses Ollama (qwen3:8b) via OpenAI-compatible API and MCP servers for various capabilities.
Includes FileSystem Sub-Agent and WebContent Sub-Agent.
"""

import json
from typing import Annotated, TypedDict, Literal
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.panel import Panel
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from mcp_client import MCPManager
from .filesystem_agent import FileSystemAgent
from .webcontent_agent import WebContentAgent


console = Console()


# ============== Sub-Agent Delegation Tools ==============

@tool
def delegate_to_filesystem_agent(task: str) -> str:
    """
    Delegate a file system task to the specialized FileSystem Sub-Agent.
    Use this for any file operations like reading, writing, listing directories, etc.

    Args:
        task: A description of the file system task to perform.
              Examples: "Lies die Datei /path/to/file.txt",
                       "Erstelle eine neue Datei mit dem Inhalt...",
                       "Liste alle Python-Dateien im Verzeichnis..."

    Returns:
        The result from the FileSystem Sub-Agent.
    """
    # This is a placeholder - actual execution happens in the agent
    return task


@tool
def delegate_to_webcontent_agent(task: str) -> str:
    """
    Delegate a web content task to the specialized WebContent Sub-Agent.
    Use this to fetch, read, or summarize content from websites.

    Args:
        task: A description of the web content task to perform.
              Examples: "Hole den Inhalt von https://example.com",
                       "Was steht auf der Webseite www.google.de?",
                       "Fasse die Seite https://news.ycombinator.com zusammen"

    Returns:
        The webpage content as markdown or a summary.
    """
    # This is a placeholder - actual execution happens in the agent
    return task


class AgentState(TypedDict):
    """State for the agent graph."""
    messages: Annotated[list, add_messages]
    pending_tool_calls: list[dict]
    approved_tool_calls: list[dict]
    tool_results: list[dict]


class HumanInTheLoopAgent:
    """KI-Agent with human approval for dangerous actions and sub-agent delegation."""

    DANGEROUS_TOOLS = {
        "browser_click", "browser_type", "browser_navigate", "browser_select_option",
        "browser_drag", "browser_press_key", "browser_file_upload", "browser_handle_dialog",
        "browser_tab_new", "browser_tab_close", "browser_pdf_save", "browser_evaluate"
    }

    SUB_AGENT_TOOLS = {
        "delegate_to_filesystem_agent",
        "delegate_to_webcontent_agent"
    }

    def __init__(self, mcp_manager: MCPManager):
        self.mcp_manager = mcp_manager
        self.filesystem_agent = FileSystemAgent()
        self.webcontent_agent = WebContentAgent()
        self.llm = ChatOpenAI(
            model="qwen3:8b",
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            temperature=0.7,
        )
        self.graph = None
        self.conversation_history: list = []
        self._system_prompt = None

    def clear_history(self) -> None:
        """Clear the conversation history to start a new chat."""
        self.conversation_history = []
        console.print("[green]Chat-Verlauf geloescht. Neuer Chat gestartet.[/green]")

    def _get_system_prompt(self) -> str:
        """Build the system prompt based on available MCP servers."""
        if self._system_prompt:
            return self._system_prompt

        servers = self.mcp_manager.list_servers()
        tools_info = []
        for server in servers:
            server_tools = self.mcp_manager.get_server_tools(server)
            tool_names = [t["name"] for t in server_tools[:5]]
            tools_info.append(f"- {server}: {', '.join(tool_names)}{'...' if len(server_tools) > 5 else ''}")

        mcp_tools_section = "\n".join(tools_info) if tools_info else "Keine MCP Server verfuegbar."

        self._system_prompt = f"""Du bist ein hilfreicher KI-Assistent mit Zugriff auf:

1. MCP Server Tools:
{mcp_tools_section}

2. FileSystem Sub-Agent:
- Nutze delegate_to_filesystem_agent fuer alle Dateioperationen
- Der Sub-Agent kann Dateien lesen, schreiben, loeschen, kopieren, verschieben
- Er kann Verzeichnisse erstellen, auflisten und durchsuchen

3. WebContent Sub-Agent:
- Nutze delegate_to_webcontent_agent um Webseiten-Inhalte abzurufen
- Der Sub-Agent kann Webseiten laden und als Markdown konvertieren
- Nutze dies wenn der Benutzer Informationen von einer Webseite benoetigt

Wichtige Hinweise:
- Beschreibe deine Aktionen klar und warte auf die Ergebnisse
- Bei Dateioperationen delegiere an den FileSystem Sub-Agent
- Bei Webseiten-Anfragen delegiere an den WebContent Sub-Agent

Antworte auf Deutsch."""

        return self._system_prompt

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph agent graph."""
        graph = StateGraph(AgentState)

        graph.add_node("agent", self._agent_node)
        graph.add_node("human_approval", self._human_approval_node)
        graph.add_node("execute_tools", self._execute_tools_node)

        graph.set_entry_point("agent")

        graph.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "human_approval": "human_approval",
                "end": END
            }
        )

        graph.add_conditional_edges(
            "human_approval",
            self._after_approval,
            {
                "execute": "execute_tools",
                "agent": "agent",
                "end": END
            }
        )

        graph.add_edge("execute_tools", "agent")

        return graph.compile()

    def _should_continue(self, state: AgentState) -> Literal["human_approval", "end"]:
        """Determine if we need human approval or should end."""
        if state.get("pending_tool_calls"):
            return "human_approval"
        return "end"

    def _after_approval(self, state: AgentState) -> Literal["execute", "agent", "end"]:
        """Determine next step after human approval."""
        if state.get("approved_tool_calls"):
            return "execute"
        if state.get("pending_tool_calls") == []:
            return "agent"
        return "end"

    async def _agent_node(self, state: AgentState) -> dict:
        """Process messages with the LLM."""
        # Get MCP tools from all servers and add sub-agent delegation tools
        tools = self.mcp_manager.get_tools_for_openai()

        # Add FileSystem Sub-Agent delegation tool
        tools.append({
            "type": "function",
            "function": {
                "name": "delegate_to_filesystem_agent",
                "description": "Delegate a file system task to the specialized FileSystem Sub-Agent. Use this for any file operations like reading, writing, creating, deleting files/directories, searching files, etc.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": "A description of the file system task to perform. Examples: 'Lies die Datei /path/to/file.txt', 'Erstelle eine Datei test.txt mit Inhalt Hello World', 'Liste alle .py Dateien im aktuellen Verzeichnis'"
                        }
                    },
                    "required": ["task"]
                }
            }
        })

        # Add WebContent Sub-Agent delegation tool
        tools.append({
            "type": "function",
            "function": {
                "name": "delegate_to_webcontent_agent",
                "description": "Delegate a web content task to fetch and read website content. Use this when the user wants to know what's on a webpage, needs information from a URL, or wants a website summarized.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": "A description of the web content task. Examples: 'Hole den Inhalt von https://example.com', 'Was steht auf www.google.de?', 'Lade die Seite https://news.ycombinator.com'"
                        }
                    },
                    "required": ["task"]
                }
            }
        })

        llm_with_tools = self.llm.bind_tools(tools)

        response = await llm_with_tools.ainvoke(state["messages"])

        pending_tool_calls = []
        if response.tool_calls:
            for tool_call in response.tool_calls:
                pending_tool_calls.append({
                    "id": tool_call["id"],
                    "name": tool_call["name"],
                    "arguments": tool_call["args"]
                })

        return {
            "messages": [response],
            "pending_tool_calls": pending_tool_calls,
            "approved_tool_calls": [],
            "tool_results": []
        }

    async def _human_approval_node(self, state: AgentState) -> dict:
        """Request human approval for tool calls."""
        pending = state.get("pending_tool_calls", [])
        approved = []

        for tool_call in pending:
            tool_name = tool_call["name"]
            args = tool_call["arguments"]

            needs_approval = tool_name in self.DANGEROUS_TOOLS

            if needs_approval:
                console.print(Panel(
                    f"[bold yellow]Tool:[/bold yellow] {tool_name}\n"
                    f"[bold cyan]Argumente:[/bold cyan]\n{json.dumps(args, indent=2, ensure_ascii=False)}",
                    title="[bold red]Bestaetigung erforderlich[/bold red]",
                    border_style="red"
                ))

                user_approved = Confirm.ask(
                    "[bold]Diese Aktion ausfuehren?[/bold]",
                    default=True
                )

                if user_approved:
                    approved.append(tool_call)
                    console.print("[green]Genehmigt[/green]")
                else:
                    console.print("[red]Abgelehnt[/red]")
            else:
                approved.append(tool_call)

        return {
            "pending_tool_calls": [],
            "approved_tool_calls": approved
        }

    async def _execute_tools_node(self, state: AgentState) -> dict:
        """Execute approved tool calls."""
        approved = state.get("approved_tool_calls", [])
        tool_messages = []

        for tool_call in approved:
            tool_name = tool_call["name"]
            args = tool_call["arguments"]
            tool_id = tool_call["id"]

            console.print(f"[dim]Fuehre aus: {tool_name}...[/dim]")

            try:
                # Check if this is a sub-agent delegation
                if tool_name == "delegate_to_filesystem_agent":
                    task = args.get("task", "")
                    result_text = await self.filesystem_agent.run(task)
                elif tool_name == "delegate_to_webcontent_agent":
                    task = args.get("task", "")
                    result_text = await self.webcontent_agent.run(task)
                else:
                    # Execute MCP tool via manager
                    result = await self.mcp_manager.call_tool(tool_name, args)

                    if hasattr(result, 'content'):
                        content = result.content
                        if isinstance(content, list):
                            result_text = "\n".join(
                                str(item.text) if hasattr(item, 'text') else str(item)
                                for item in content
                            )
                        else:
                            result_text = str(content)
                    else:
                        result_text = str(result)

                tool_messages.append(ToolMessage(
                    content=result_text,
                    tool_call_id=tool_id
                ))

                console.print(f"[green]Erfolgreich: {tool_name}[/green]")

            except Exception as e:
                error_msg = f"Fehler bei {tool_name}: {str(e)}"
                tool_messages.append(ToolMessage(
                    content=error_msg,
                    tool_call_id=tool_id
                ))
                console.print(f"[red]{error_msg}[/red]")

        return {
            "messages": tool_messages,
            "approved_tool_calls": [],
            "tool_results": []
        }

    async def run(self, user_input: str) -> str:
        """Run the agent with user input, maintaining conversation history."""
        if not self.graph:
            self.graph = self._build_graph()

        # Add user message to history
        self.conversation_history.append(HumanMessage(content=user_input))

        # Build messages with system prompt and full history
        messages = [
            SystemMessage(content=self._get_system_prompt()),
            *self.conversation_history
        ]

        initial_state: AgentState = {
            "messages": messages,
            "pending_tool_calls": [],
            "approved_tool_calls": [],
            "tool_results": []
        }

        console.print(Panel(user_input, title="[bold blue]Benutzer[/bold blue]", border_style="blue"))

        final_state = await self.graph.ainvoke(initial_state)

        # Extract the assistant's response from the final state
        last_message = final_state["messages"][-1]
        if isinstance(last_message, AIMessage):
            response = last_message.content
            # Add assistant response to history
            self.conversation_history.append(AIMessage(content=response))
        else:
            response = str(last_message)

        console.print(Panel(response, title="[bold green]Assistent[/bold green]", border_style="green"))

        return response

    def _print_welcome(self):
        """Print the welcome message."""
        servers = self.mcp_manager.list_servers()
        if servers:
            server_list = "\n".join(f"  - {s} ({len(self.mcp_manager.get_server_tools(s))} Tools)" for s in servers)
            mcp_info = f"MCP Server:\n{server_list}"
        else:
            mcp_info = "Keine MCP Server geladen."

        console.print(Panel(
            f"[bold]Willkommen beim KI-Agenten![/bold]\n\n"
            f"{mcp_info}\n"
            f"  - FileSystem Sub-Agent (11 Tools)\n"
            f"  - WebContent Sub-Agent (3 Tools)\n\n"
            f"Gefaehrliche Aktionen erfordern Ihre Bestaetigung.\n\n"
            f"[dim]Befehle:[/dim]\n"
            f"[dim]  /new, /clear - Neuen Chat starten[/dim]\n"
            f"[dim]  exit, quit   - Beenden[/dim]",
            title="[bold cyan]KI-Agent mit Human-in-the-Loop[/bold cyan]",
            border_style="cyan"
        ))

    async def chat_loop(self):
        """Interactive chat loop."""
        self._print_welcome()

        while True:
            try:
                user_input = Prompt.ask("\n[bold blue]Sie[/bold blue]")

                # Exit commands
                if user_input.lower() in ["exit", "quit", "q"]:
                    console.print("[yellow]Auf Wiedersehen![/yellow]")
                    break

                # New chat commands
                if user_input.lower() in ["/new", "/clear"]:
                    if self.conversation_history:
                        confirm = Confirm.ask(
                            "[yellow]Moechten Sie den aktuellen Chat wirklich loeschen?[/yellow]",
                            default=False
                        )
                        if confirm:
                            self.clear_history()
                    else:
                        console.print("[dim]Chat-Verlauf ist bereits leer.[/dim]")
                    continue

                if not user_input.strip():
                    continue

                await self.run(user_input)

            except KeyboardInterrupt:
                console.print("\n[yellow]Abgebrochen. Auf Wiedersehen![/yellow]")
                break
            except Exception as e:
                console.print(f"[red]Fehler: {e}[/red]")
