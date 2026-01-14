"""
WebContent Sub-Agent
Fetches and converts web content using Docling for use in the chat.
"""

import json
from typing import Annotated, TypedDict, Literal
from rich.console import Console
from rich.panel import Panel
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from docling.document_converter import DocumentConverter


console = Console()


# ============== WebContent Tools ==============

@tool
def fetch_webpage(url: str) -> str:
    """
    Fetch and convert a webpage to markdown format.

    Args:
        url: The URL of the webpage to fetch (e.g., "https://www.example.com").

    Returns:
        The webpage content as markdown text.
    """
    # Ensure URL has a scheme
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    console.print(f"[dim]Lade Webseite: {url}...[/dim]")

    try:
        converter = DocumentConverter()
        result = converter.convert(url)
        markdown_content = result.document.export_to_markdown()

        if not markdown_content or len(markdown_content.strip()) == 0:
            return f"Fehler: Keine Inhalte von {url} extrahiert."

        # Truncate if too long
        max_length = 15000
        if len(markdown_content) > max_length:
            markdown_content = markdown_content[:max_length] + "\n\n[... Inhalt gekuerzt ...]"

        return markdown_content

    except Exception as e:
        return f"Fehler beim Laden von {url}: {str(e)}"


@tool
def fetch_multiple_webpages(urls: list[str]) -> str:
    """
    Fetch and convert multiple webpages to markdown format.

    Args:
        urls: List of URLs to fetch.

    Returns:
        Combined markdown content from all webpages.
    """
    results = []

    for url in urls:
        # Ensure URL has a scheme
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        console.print(f"[dim]Lade Webseite: {url}...[/dim]")

        try:
            converter = DocumentConverter()
            result = converter.convert(url)
            markdown_content = result.document.export_to_markdown()

            if markdown_content and len(markdown_content.strip()) > 0:
                # Truncate individual pages
                max_per_page = 5000
                if len(markdown_content) > max_per_page:
                    markdown_content = markdown_content[:max_per_page] + "\n[... gekuerzt ...]"

                results.append(f"## Inhalt von {url}\n\n{markdown_content}")
            else:
                results.append(f"## {url}\n\nKeine Inhalte extrahiert.")

        except Exception as e:
            results.append(f"## {url}\n\nFehler: {str(e)}")

    return "\n\n---\n\n".join(results)


@tool
def summarize_webpage(url: str, focus: str = "") -> str:
    """
    Fetch a webpage and prepare it for summarization.

    Args:
        url: The URL of the webpage to fetch.
        focus: Optional focus area for the summary (e.g., "pricing", "features").

    Returns:
        The webpage content with summarization instructions.
    """
    # Ensure URL has a scheme
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    console.print(f"[dim]Lade Webseite fuer Zusammenfassung: {url}...[/dim]")

    try:
        converter = DocumentConverter()
        result = converter.convert(url)
        markdown_content = result.document.export_to_markdown()

        if not markdown_content or len(markdown_content.strip()) == 0:
            return f"Fehler: Keine Inhalte von {url} extrahiert."

        # Truncate if too long
        max_length = 10000
        if len(markdown_content) > max_length:
            markdown_content = markdown_content[:max_length] + "\n\n[... Inhalt gekuerzt ...]"

        focus_instruction = f"\nFokus: {focus}" if focus else ""

        return f"Webseiten-Inhalt von {url}:{focus_instruction}\n\n{markdown_content}"

    except Exception as e:
        return f"Fehler beim Laden von {url}: {str(e)}"


# ============== WebContent Agent ==============

class WebContentAgentState(TypedDict):
    """State for the webcontent agent graph."""
    messages: Annotated[list, add_messages]
    pending_tool_calls: list[dict]
    approved_tool_calls: list[dict]
    fetched_content: dict[str, str]


# All available webcontent tools
WEBCONTENT_TOOLS = [
    fetch_webpage,
    fetch_multiple_webpages,
    summarize_webpage
]


class WebContentAgent:
    """Sub-Agent for fetching and processing web content."""

    def __init__(self):
        self.llm = ChatOpenAI(
            model="qwen3:8b",
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            temperature=0.3,
        )
        self.tools = WEBCONTENT_TOOLS
        self.tools_by_name = {t.name: t for t in self.tools}
        self.graph = self._build_graph()
        self.cached_content: dict[str, str] = {}

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph agent graph."""
        graph = StateGraph(WebContentAgentState)

        graph.add_node("agent", self._agent_node)
        graph.add_node("execute_tools", self._execute_tools_node)

        graph.set_entry_point("agent")

        graph.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "execute": "execute_tools",
                "end": END
            }
        )

        graph.add_edge("execute_tools", "agent")

        return graph.compile()

    def _should_continue(self, state: WebContentAgentState) -> Literal["execute", "end"]:
        """Determine if we should execute tools or end."""
        if state.get("pending_tool_calls"):
            return "execute"
        return "end"

    async def _agent_node(self, state: WebContentAgentState) -> dict:
        """Process messages with the LLM."""
        llm_with_tools = self.llm.bind_tools(self.tools)
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
            "approved_tool_calls": []
        }

    async def _execute_tools_node(self, state: WebContentAgentState) -> dict:
        """Execute tool calls (no approval needed for reading web content)."""
        pending = state.get("pending_tool_calls", [])
        tool_messages = []
        fetched = state.get("fetched_content", {})

        for tool_call in pending:
            tool_name = tool_call["name"]
            args = tool_call["arguments"]
            tool_id = tool_call["id"]

            console.print(f"[dim]Fuehre aus: {tool_name}...[/dim]")

            try:
                tool_func = self.tools_by_name[tool_name]
                result = await tool_func.ainvoke(args)

                # Cache the content
                if tool_name == "fetch_webpage":
                    url = args.get("url", "")
                    fetched[url] = result
                    self.cached_content[url] = result

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
            "pending_tool_calls": [],
            "fetched_content": fetched
        }

    async def run(self, task: str) -> str:
        """
        Run the webcontent agent with a specific task.

        Args:
            task: The web content task to perform.

        Returns:
            The result including fetched content.
        """
        system_prompt = """Du bist ein Web-Content-Assistent. Du kannst Webseiten abrufen und deren Inhalte als Markdown bereitstellen.

Verfuegbare Operationen:
- fetch_webpage: Eine einzelne Webseite laden und als Markdown konvertieren
- fetch_multiple_webpages: Mehrere Webseiten gleichzeitig laden
- summarize_webpage: Eine Webseite laden mit optionalem Fokus fuer Zusammenfassung

Wichtig:
- URLs koennen mit oder ohne https:// angegeben werden
- Gib den Inhalt der Webseiten strukturiert zurueck
- Bei langen Inhalten wird automatisch gekuerzt
- Antworte auf Deutsch"""

        initial_state: WebContentAgentState = {
            "messages": [
                SystemMessage(content=system_prompt),
                HumanMessage(content=task)
            ],
            "pending_tool_calls": [],
            "approved_tool_calls": [],
            "fetched_content": {}
        }

        console.print(Panel(
            f"[bold]WebContent Sub-Agent[/bold]\nAufgabe: {task}",
            title="[bold blue]Sub-Agent gestartet[/bold blue]",
            border_style="blue"
        ))

        final_state = await self.graph.ainvoke(initial_state)

        last_message = final_state["messages"][-1]
        if isinstance(last_message, AIMessage):
            response = last_message.content
        else:
            response = str(last_message)

        console.print(Panel(
            response[:500] + "..." if len(response) > 500 else response,
            title="[bold blue]WebContent Sub-Agent Ergebnis[/bold blue]",
            border_style="blue"
        ))

        return response

    def get_cached_content(self, url: str) -> str | None:
        """Get previously fetched content for a URL."""
        return self.cached_content.get(url)

    def clear_cache(self) -> None:
        """Clear the content cache."""
        self.cached_content.clear()
