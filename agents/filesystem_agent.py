"""
FileSystem Sub-Agent
Handles file system operations with human-in-the-loop for dangerous actions.
"""

import os
import json
import shutil
from pathlib import Path
from typing import Annotated, TypedDict, Literal
from rich.console import Console
from rich.prompt import Confirm
from rich.panel import Panel
from openai import OpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from token_manager import get_token_manager
from history_manager import get_history_manager, MessageRole


console = Console()


# ============== FileSystem Tools ==============

@tool
def read_file(file_path: str) -> str:
    """
    Read the contents of a file.

    Args:
        file_path: The path to the file to read.

    Returns:
        The contents of the file as a string.
    """
    path = Path(file_path).expanduser().resolve()
    if not path.exists():
        return f"Fehler: Datei '{file_path}' existiert nicht."
    if not path.is_file():
        return f"Fehler: '{file_path}' ist keine Datei."

    try:
        content = path.read_text(encoding='utf-8')
        return content
    except Exception as e:
        return f"Fehler beim Lesen: {str(e)}"


@tool
def write_file(file_path: str, content: str) -> str:
    """
    Write content to a file. Creates the file if it doesn't exist.

    Args:
        file_path: The path to the file to write.
        content: The content to write to the file.

    Returns:
        Success or error message.
    """
    path = Path(file_path).expanduser().resolve()

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding='utf-8')
        return f"Erfolgreich geschrieben: {file_path}"
    except Exception as e:
        return f"Fehler beim Schreiben: {str(e)}"


@tool
def append_file(file_path: str, content: str) -> str:
    """
    Append content to a file.

    Args:
        file_path: The path to the file.
        content: The content to append.

    Returns:
        Success or error message.
    """
    path = Path(file_path).expanduser().resolve()

    try:
        with open(path, 'a', encoding='utf-8') as f:
            f.write(content)
        return f"Erfolgreich angehaengt: {file_path}"
    except Exception as e:
        return f"Fehler beim Anhaengen: {str(e)}"


@tool
def delete_file(file_path: str) -> str:
    """
    Delete a file.

    Args:
        file_path: The path to the file to delete.

    Returns:
        Success or error message.
    """
    path = Path(file_path).expanduser().resolve()

    if not path.exists():
        return f"Fehler: Datei '{file_path}' existiert nicht."

    try:
        path.unlink()
        return f"Erfolgreich geloescht: {file_path}"
    except Exception as e:
        return f"Fehler beim Loeschen: {str(e)}"


@tool
def list_directory(directory_path: str) -> str:
    """
    List contents of a directory.

    Args:
        directory_path: The path to the directory.

    Returns:
        List of files and directories.
    """
    path = Path(directory_path).expanduser().resolve()

    if not path.exists():
        return f"Fehler: Verzeichnis '{directory_path}' existiert nicht."
    if not path.is_dir():
        return f"Fehler: '{directory_path}' ist kein Verzeichnis."

    try:
        items = []
        for item in sorted(path.iterdir()):
            item_type = "[DIR]" if item.is_dir() else "[FILE]"
            size = "" if item.is_dir() else f" ({item.stat().st_size} bytes)"
            items.append(f"{item_type} {item.name}{size}")

        if not items:
            return f"Verzeichnis '{directory_path}' ist leer."

        return "\n".join(items)
    except Exception as e:
        return f"Fehler beim Auflisten: {str(e)}"


@tool
def create_directory(directory_path: str) -> str:
    """
    Create a directory (including parent directories).

    Args:
        directory_path: The path to the directory to create.

    Returns:
        Success or error message.
    """
    path = Path(directory_path).expanduser().resolve()

    try:
        path.mkdir(parents=True, exist_ok=True)
        return f"Verzeichnis erstellt: {directory_path}"
    except Exception as e:
        return f"Fehler beim Erstellen: {str(e)}"


@tool
def delete_directory(directory_path: str, recursive: bool = False) -> str:
    """
    Delete a directory.

    Args:
        directory_path: The path to the directory to delete.
        recursive: If True, delete directory and all contents. If False, only delete if empty.

    Returns:
        Success or error message.
    """
    path = Path(directory_path).expanduser().resolve()

    if not path.exists():
        return f"Fehler: Verzeichnis '{directory_path}' existiert nicht."
    if not path.is_dir():
        return f"Fehler: '{directory_path}' ist kein Verzeichnis."

    try:
        if recursive:
            shutil.rmtree(path)
        else:
            path.rmdir()
        return f"Verzeichnis geloescht: {directory_path}"
    except OSError as e:
        if "not empty" in str(e).lower() or "directory not empty" in str(e).lower():
            return f"Fehler: Verzeichnis nicht leer. Verwende recursive=True zum Loeschen."
        return f"Fehler beim Loeschen: {str(e)}"
    except Exception as e:
        return f"Fehler beim Loeschen: {str(e)}"


@tool
def copy_file(source_path: str, destination_path: str) -> str:
    """
    Copy a file to a new location.

    Args:
        source_path: The path to the source file.
        destination_path: The path to the destination.

    Returns:
        Success or error message.
    """
    src = Path(source_path).expanduser().resolve()
    dst = Path(destination_path).expanduser().resolve()

    if not src.exists():
        return f"Fehler: Quelldatei '{source_path}' existiert nicht."

    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return f"Kopiert: {source_path} -> {destination_path}"
    except Exception as e:
        return f"Fehler beim Kopieren: {str(e)}"


@tool
def move_file(source_path: str, destination_path: str) -> str:
    """
    Move a file to a new location.

    Args:
        source_path: The path to the source file.
        destination_path: The path to the destination.

    Returns:
        Success or error message.
    """
    src = Path(source_path).expanduser().resolve()
    dst = Path(destination_path).expanduser().resolve()

    if not src.exists():
        return f"Fehler: Quelle '{source_path}' existiert nicht."

    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        return f"Verschoben: {source_path} -> {destination_path}"
    except Exception as e:
        return f"Fehler beim Verschieben: {str(e)}"


@tool
def get_file_info(file_path: str) -> str:
    """
    Get information about a file or directory.

    Args:
        file_path: The path to the file or directory.

    Returns:
        File information as JSON string.
    """
    path = Path(file_path).expanduser().resolve()

    if not path.exists():
        return f"Fehler: '{file_path}' existiert nicht."

    try:
        stat = path.stat()
        info = {
            "name": path.name,
            "path": str(path),
            "type": "directory" if path.is_dir() else "file",
            "size_bytes": stat.st_size,
            "created": stat.st_ctime,
            "modified": stat.st_mtime,
            "readable": os.access(path, os.R_OK),
            "writable": os.access(path, os.W_OK),
            "executable": os.access(path, os.X_OK)
        }
        return json.dumps(info, indent=2)
    except Exception as e:
        return f"Fehler: {str(e)}"


@tool
def search_files(directory_path: str, pattern: str) -> str:
    """
    Search for files matching a pattern in a directory.

    Args:
        directory_path: The directory to search in.
        pattern: Glob pattern to match (e.g., "*.txt", "**/*.py").

    Returns:
        List of matching file paths.
    """
    path = Path(directory_path).expanduser().resolve()

    if not path.exists():
        return f"Fehler: Verzeichnis '{directory_path}' existiert nicht."

    try:
        matches = list(path.glob(pattern))
        if not matches:
            return f"Keine Dateien gefunden fuer Muster: {pattern}"

        results = [str(m) for m in sorted(matches)]
        return "\n".join(results)
    except Exception as e:
        return f"Fehler bei der Suche: {str(e)}"


# ============== FileSystem Agent ==============

class FileSystemAgentState(TypedDict):
    """State for the filesystem agent graph."""
    messages: Annotated[list, add_messages]
    pending_tool_calls: list[dict]
    approved_tool_calls: list[dict]


# All available filesystem tools
FILESYSTEM_TOOLS = [
    read_file,
    write_file,
    append_file,
    delete_file,
    list_directory,
    create_directory,
    delete_directory,
    copy_file,
    move_file,
    get_file_info,
    search_files
]

# Tools that require human approval (dangerous operations)
DANGEROUS_FS_TOOLS = {
    "write_file",
    "append_file",
    "delete_file",
    "delete_directory",
    "move_file"
}


class FileSystemAgent:
    """Sub-Agent for file system operations with human-in-the-loop."""

    def __init__(self, base_url: str = "http://localhost:11434/v1", model: str = "qwen3:8b"):
        self.base_url = base_url
        self.model = model
        self.token_manager = get_token_manager()
        self.history_manager = get_history_manager()
        self.tools = FILESYSTEM_TOOLS
        self.tools_by_name = {t.name: t for t in self.tools}
        self.graph = self._build_graph()

    def _get_client(self) -> OpenAI:
        """Erstellt einen OpenAI Client mit aktuellem Token."""
        return OpenAI(
            base_url=self.base_url,
            api_key=self.token_manager.get_token(),
        )

    def _get_tools_schema(self) -> list[dict]:
        """Convert LangChain tools to OpenAI function format."""
        tools_schema = []
        for t in self.tools:
            tools_schema.append({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.args_schema.schema() if hasattr(t, 'args_schema') else {"type": "object", "properties": {}}
                }
            })
        return tools_schema

    def _messages_to_openai_format(self, messages: list) -> list[dict]:
        """Convert LangChain messages to OpenAI format."""
        openai_messages = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                openai_messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                openai_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                ai_msg = {"role": "assistant", "content": msg.content or ""}
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    ai_msg["tool_calls"] = [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {"name": tc["name"], "arguments": json.dumps(tc["args"])}
                        } for tc in msg.tool_calls
                    ]
                openai_messages.append(ai_msg)
            elif isinstance(msg, ToolMessage):
                openai_messages.append({
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id,
                    "content": msg.content
                })
        return openai_messages

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph agent graph."""
        graph = StateGraph(FileSystemAgentState)

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

    def _should_continue(self, state: FileSystemAgentState) -> Literal["human_approval", "end"]:
        """Determine if we need human approval or should end."""
        if state.get("pending_tool_calls"):
            return "human_approval"
        return "end"

    def _after_approval(self, state: FileSystemAgentState) -> Literal["execute", "agent", "end"]:
        """Determine next step after human approval."""
        if state.get("approved_tool_calls"):
            return "execute"
        if state.get("pending_tool_calls") == []:
            return "agent"
        return "end"

    async def _agent_node(self, state: FileSystemAgentState) -> dict:
        """Process messages with the LLM."""
        openai_messages = self._messages_to_openai_format(state["messages"])
        tools_schema = self._get_tools_schema()

        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=openai_messages,
            tools=tools_schema if tools_schema else None,
            temperature=0.3,
        )

        assistant_message = response.choices[0].message
        content = assistant_message.content or ""

        pending_tool_calls = []
        tool_calls_for_ai_message = []

        if assistant_message.tool_calls:
            for tool_call in assistant_message.tool_calls:
                tc_dict = {
                    "id": tool_call.id,
                    "name": tool_call.function.name,
                    "args": json.loads(tool_call.function.arguments)
                }
                pending_tool_calls.append({
                    "id": tool_call.id,
                    "name": tool_call.function.name,
                    "arguments": json.loads(tool_call.function.arguments)
                })
                tool_calls_for_ai_message.append(tc_dict)

        ai_message = AIMessage(content=content, tool_calls=tool_calls_for_ai_message)

        return {
            "messages": [ai_message],
            "pending_tool_calls": pending_tool_calls,
            "approved_tool_calls": []
        }

    async def _human_approval_node(self, state: FileSystemAgentState) -> dict:
        """Request human approval for dangerous tool calls."""
        pending = state.get("pending_tool_calls", [])
        approved = []

        for tool_call in pending:
            tool_name = tool_call["name"]
            args = tool_call["arguments"]

            needs_approval = tool_name in DANGEROUS_FS_TOOLS

            if needs_approval:
                console.print(Panel(
                    f"[bold yellow]Tool:[/bold yellow] {tool_name}\n"
                    f"[bold cyan]Argumente:[/bold cyan]\n{json.dumps(args, indent=2, ensure_ascii=False)}",
                    title="[bold red]Dateisystem-Aktion - Bestaetigung erforderlich[/bold red]",
                    border_style="red"
                ))

                user_approved = Confirm.ask(
                    "[bold]Diese Dateisystem-Aktion ausfuehren?[/bold]",
                    default=False
                )

                if user_approved:
                    approved.append(tool_call)
                    console.print("[green]Genehmigt[/green]")
                else:
                    console.print("[red]Abgelehnt[/red]")
            else:
                console.print(f"[dim]Auto-genehmigt (lesend): {tool_name}[/dim]")
                approved.append(tool_call)

        return {
            "pending_tool_calls": [],
            "approved_tool_calls": approved
        }

    async def _execute_tools_node(self, state: FileSystemAgentState) -> dict:
        """Execute approved tool calls."""
        approved = state.get("approved_tool_calls", [])
        tool_messages = []

        for tool_call in approved:
            tool_name = tool_call["name"]
            args = tool_call["arguments"]
            tool_id = tool_call["id"]

            console.print(f"[dim]Fuehre aus: {tool_name}...[/dim]")

            try:
                tool_func = self.tools_by_name[tool_name]
                result = await tool_func.ainvoke(args)

                tool_messages.append(ToolMessage(
                    content=str(result),
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
            "approved_tool_calls": []
        }

    async def run(self, task: str) -> str:
        """
        Run the filesystem agent with a specific task.

        Args:
            task: The filesystem task to perform.

        Returns:
            The result of the operation.
        """
        # Log task to central history
        self.history_manager.add_user_message(
            f"[FileSystem Task] {task}",
            agent_name="FileSystemAgent"
        )

        # Get context from central history
        context_summary = self.history_manager.get_context_summary(max_length=1000)

        system_prompt = f"""Du bist ein Dateisystem-Assistent. Du kannst Dateien und Verzeichnisse lesen, schreiben, erstellen, loeschen, kopieren und verschieben.

Verfuegbare Operationen:
- read_file: Datei lesen
- write_file: Datei schreiben/erstellen
- append_file: An Datei anhaengen
- delete_file: Datei loeschen
- list_directory: Verzeichnisinhalt anzeigen
- create_directory: Verzeichnis erstellen
- delete_directory: Verzeichnis loeschen
- copy_file: Datei kopieren
- move_file: Datei verschieben
- get_file_info: Datei-Informationen abrufen
- search_files: Dateien suchen (Glob-Pattern)

Kontext aus vorheriger Konversation:
{context_summary}

Wichtig:
- Sei vorsichtig mit Loeschoperationen
- Bestaetigung wird fuer gefaehrliche Operationen angefordert
- Antworte auf Deutsch"""

        initial_state: FileSystemAgentState = {
            "messages": [
                SystemMessage(content=system_prompt),
                HumanMessage(content=task)
            ],
            "pending_tool_calls": [],
            "approved_tool_calls": []
        }

        console.print(Panel(
            f"[bold]FileSystem Sub-Agent[/bold]\nAufgabe: {task}",
            title="[bold magenta]Sub-Agent gestartet[/bold magenta]",
            border_style="magenta"
        ))

        final_state = await self.graph.ainvoke(initial_state)

        last_message = final_state["messages"][-1]
        if isinstance(last_message, AIMessage):
            response = last_message.content
        else:
            response = str(last_message)

        # Log response to central history
        self.history_manager.add_assistant_message(
            f"[FileSystem Result] {response[:500]}",
            agent_name="FileSystemAgent"
        )

        console.print(Panel(
            response,
            title="[bold magenta]FileSystem Sub-Agent Ergebnis[/bold magenta]",
            border_style="magenta"
        ))

        return response
