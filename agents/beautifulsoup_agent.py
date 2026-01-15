"""
BeautifulSoup Web Agent
Fetches and parses web content using BeautifulSoup for use in the chat.
"""

import json
import requests
from typing import Annotated, TypedDict, Literal
from bs4 import BeautifulSoup
from rich.console import Console
from rich.panel import Panel
from openai import OpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from token_manager import get_token_manager
from history_manager import get_history_manager, MessageRole


console = Console()


# ============== BeautifulSoup Tools ==============

@tool
def fetch_webpage_bs(url: str, extract_text: bool = True) -> str:
    """
    Fetch a webpage and extract its content using BeautifulSoup.

    Args:
        url: The URL of the webpage to fetch (e.g., "https://www.example.com").
        extract_text: If True, extract only text content. If False, return cleaned HTML.

    Returns:
        The webpage content as text or cleaned HTML.
    """
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    console.print(f"[dim]Lade Webseite (BeautifulSoup): {url}...[/dim]")

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")

        # Entferne Script und Style Elemente
        for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
            element.decompose()

        if extract_text:
            # Extrahiere nur Text
            text = soup.get_text(separator="\n", strip=True)
            # Entferne mehrfache Leerzeilen
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            content = "\n".join(lines)
        else:
            # Gib bereinigtes HTML zurueck
            content = soup.prettify()

        if not content or len(content.strip()) == 0:
            return f"Fehler: Keine Inhalte von {url} extrahiert."

        # Kuerzen wenn zu lang
        max_length = 15000
        if len(content) > max_length:
            content = content[:max_length] + "\n\n[... Inhalt gekuerzt ...]"

        return content

    except requests.exceptions.Timeout:
        return f"Fehler: Timeout beim Laden von {url}"
    except requests.exceptions.RequestException as e:
        return f"Fehler beim Laden von {url}: {str(e)}"
    except Exception as e:
        return f"Fehler bei der Verarbeitung von {url}: {str(e)}"


@tool
def fetch_webpage_elements(url: str, selector: str) -> str:
    """
    Fetch specific elements from a webpage using CSS selectors.

    Args:
        url: The URL of the webpage to fetch.
        selector: CSS selector to find specific elements (e.g., "article", "div.content", "h1").

    Returns:
        The extracted elements as text.
    """
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    console.print(f"[dim]Lade Elemente '{selector}' von: {url}...[/dim]")

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")
        elements = soup.select(selector)

        if not elements:
            return f"Keine Elemente gefunden fuer Selector: {selector}"

        results = []
        for i, elem in enumerate(elements[:20], 1):  # Maximal 20 Elemente
            text = elem.get_text(separator=" ", strip=True)
            if text:
                results.append(f"{i}. {text[:500]}")  # Max 500 Zeichen pro Element

        return "\n\n".join(results) if results else "Keine Textinhalte in den gefundenen Elementen."

    except requests.exceptions.RequestException as e:
        return f"Fehler beim Laden von {url}: {str(e)}"
    except Exception as e:
        return f"Fehler: {str(e)}"


@tool
def fetch_links(url: str, filter_pattern: str = "") -> str:
    """
    Extract all links from a webpage.

    Args:
        url: The URL of the webpage to fetch.
        filter_pattern: Optional pattern to filter links (e.g., "/article", ".pdf").

    Returns:
        List of found links.
    """
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    console.print(f"[dim]Extrahiere Links von: {url}...[/dim]")

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")
        links = []

        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            text = a_tag.get_text(strip=True)[:100]  # Max 100 Zeichen fuer Link-Text

            # Relative URLs zu absoluten machen
            if href.startswith("/"):
                from urllib.parse import urljoin
                href = urljoin(url, href)

            # Filter anwenden wenn angegeben
            if filter_pattern and filter_pattern not in href:
                continue

            if href.startswith(("http://", "https://")):
                links.append(f"- {text}: {href}" if text else f"- {href}")

        if not links:
            return "Keine Links gefunden" + (f" mit Filter '{filter_pattern}'" if filter_pattern else "")

        # Maximal 50 Links
        if len(links) > 50:
            links = links[:50]
            links.append(f"\n... und {len(links) - 50} weitere Links")

        return "\n".join(links)

    except requests.exceptions.RequestException as e:
        return f"Fehler beim Laden von {url}: {str(e)}"
    except Exception as e:
        return f"Fehler: {str(e)}"


@tool
def fetch_tables(url: str) -> str:
    """
    Extract tables from a webpage.

    Args:
        url: The URL of the webpage to fetch.

    Returns:
        Extracted tables as formatted text.
    """
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    console.print(f"[dim]Extrahiere Tabellen von: {url}...[/dim]")

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")
        tables = soup.find_all("table")

        if not tables:
            return "Keine Tabellen auf dieser Seite gefunden."

        results = []
        for i, table in enumerate(tables[:5], 1):  # Maximal 5 Tabellen
            rows = table.find_all("tr")
            table_data = []

            for row in rows[:20]:  # Maximal 20 Zeilen pro Tabelle
                cells = row.find_all(["th", "td"])
                cell_texts = [cell.get_text(strip=True)[:50] for cell in cells]  # Max 50 Zeichen pro Zelle
                if cell_texts:
                    table_data.append(" | ".join(cell_texts))

            if table_data:
                results.append(f"=== Tabelle {i} ===\n" + "\n".join(table_data))

        return "\n\n".join(results) if results else "Keine Tabelleninhalte extrahiert."

    except requests.exceptions.RequestException as e:
        return f"Fehler beim Laden von {url}: {str(e)}"
    except Exception as e:
        return f"Fehler: {str(e)}"


# ============== BeautifulSoup Agent ==============

class BeautifulSoupAgentState(TypedDict):
    """State for the BeautifulSoup agent graph."""
    messages: Annotated[list, add_messages]
    pending_tool_calls: list[dict]
    approved_tool_calls: list[dict]
    fetched_content: dict[str, str]


# All available BeautifulSoup tools
BEAUTIFULSOUP_TOOLS = [
    fetch_webpage_bs,
    fetch_webpage_elements,
    fetch_links,
    fetch_tables
]


class BeautifulSoupAgent:
    """Sub-Agent for fetching and parsing web content using BeautifulSoup."""

    def __init__(self, base_url: str = "http://localhost:11434/v1", model: str = "qwen3:8b"):
        self.base_url = base_url
        self.model = model
        self.token_manager = get_token_manager()
        self.history_manager = get_history_manager()
        self.tools = BEAUTIFULSOUP_TOOLS
        self.tools_by_name = {t.name: t for t in self.tools}
        self.graph = self._build_graph()
        self.cached_content: dict[str, str] = {}

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
        graph = StateGraph(BeautifulSoupAgentState)

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

    def _should_continue(self, state: BeautifulSoupAgentState) -> Literal["execute", "end"]:
        """Determine if we should execute tools or end."""
        if state.get("pending_tool_calls"):
            return "execute"
        return "end"

    async def _agent_node(self, state: BeautifulSoupAgentState) -> dict:
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

    async def _execute_tools_node(self, state: BeautifulSoupAgentState) -> dict:
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
                if tool_name == "fetch_webpage_bs":
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
        Run the BeautifulSoup agent with a specific task.

        Args:
            task: The web content task to perform.

        Returns:
            The result including fetched content.
        """
        # Log task to central history
        self.history_manager.add_user_message(
            f"[BeautifulSoup Task] {task}",
            agent_name="BeautifulSoupAgent"
        )

        # Get context from central history
        context_summary = self.history_manager.get_context_summary(max_length=1000)

        system_prompt = f"""Du bist ein Web-Scraping-Assistent. Du kannst Webseiten abrufen und deren Inhalte mit BeautifulSoup parsen.

Verfuegbare Operationen:
- fetch_webpage_bs: Eine Webseite laden und Text extrahieren
- fetch_webpage_elements: Bestimmte Elemente per CSS-Selector extrahieren
- fetch_links: Alle Links einer Seite extrahieren
- fetch_tables: Tabellen von einer Seite extrahieren

Kontext aus vorheriger Konversation:
{context_summary}

Wichtig:
- URLs koennen mit oder ohne https:// angegeben werden
- Nutze CSS-Selektoren fuer gezielte Extraktion (z.B. "article", "div.content", "h1")
- Bei langen Inhalten wird automatisch gekuerzt
- Antworte auf Deutsch"""

        initial_state: BeautifulSoupAgentState = {
            "messages": [
                SystemMessage(content=system_prompt),
                HumanMessage(content=task)
            ],
            "pending_tool_calls": [],
            "approved_tool_calls": [],
            "fetched_content": {}
        }

        console.print(Panel(
            f"[bold]BeautifulSoup Sub-Agent[/bold]\nAufgabe: {task}",
            title="[bold cyan]Sub-Agent gestartet[/bold cyan]",
            border_style="cyan"
        ))

        final_state = await self.graph.ainvoke(initial_state)

        last_message = final_state["messages"][-1]
        if isinstance(last_message, AIMessage):
            response = last_message.content
        else:
            response = str(last_message)

        # Log response to central history
        self.history_manager.add_assistant_message(
            f"[BeautifulSoup Result] {response[:500]}",
            agent_name="BeautifulSoupAgent"
        )

        console.print(Panel(
            response[:500] + "..." if len(response) > 500 else response,
            title="[bold cyan]BeautifulSoup Sub-Agent Ergebnis[/bold cyan]",
            border_style="cyan"
        ))

        return response

    def get_cached_content(self, url: str) -> str | None:
        """Get previously fetched content for a URL."""
        return self.cached_content.get(url)

    def clear_cache(self) -> None:
        """Clear the content cache."""
        self.cached_content.clear()
