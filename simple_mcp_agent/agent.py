"""
Einfacher MCP Agent
Verwendet die OpenAI SDK um mit MCP Servern zu interagieren.
"""

import json
from typing import Optional
from openai import OpenAI

from mcp_manager import MCPManager


class SimpleMCPAgent:
    """Ein einfacher Agent der MCP Server Tools nutzen kann."""

    def __init__(
        self,
        mcp_manager: MCPManager,
        base_url: str = "http://localhost:11434/v1",
        model: str = "qwen3:8b",
        api_key: str = "ollama"
    ):
        self.mcp_manager = mcp_manager
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.conversation_history: list[dict] = []
        self._system_prompt = self._build_system_prompt()

    def _get_client(self) -> OpenAI:
        """Erstellt einen OpenAI Client."""
        return OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )

    def _build_system_prompt(self) -> str:
        """Erstellt den System-Prompt basierend auf verfuegbaren MCP Tools."""
        servers = self.mcp_manager.list_servers()
        tools_info = []

        for server in servers:
            server_tools = self.mcp_manager.get_server_tools(server)
            tool_names = [t["name"] for t in server_tools[:10]]
            more_indicator = "..." if len(server_tools) > 10 else ""
            tools_info.append(f"- {server}: {', '.join(tool_names)}{more_indicator}")

        mcp_tools_section = "\n".join(tools_info) if tools_info else "Keine MCP Server verfuegbar."

        return f"""Du bist ein hilfreicher KI-Assistent mit Zugriff auf MCP Server Tools.

Verfuegbare MCP Server und Tools:
{mcp_tools_section}

Wichtige Hinweise:
- Nutze die verfuegbaren Tools um Aufgaben zu erledigen
- Beschreibe deine Aktionen klar
- Antworte auf Deutsch"""

    def refresh_system_prompt(self) -> None:
        """Aktualisiert den System-Prompt (z.B. nach dem Starten neuer Server)."""
        self._system_prompt = self._build_system_prompt()

    def clear_history(self) -> None:
        """Loescht den Konversationsverlauf."""
        self.conversation_history = []
        print("[INFO] Konversationsverlauf geloescht.")

    async def _execute_tool_call(self, tool_name: str, arguments: dict) -> str:
        """Fuehrt einen Tool-Aufruf auf dem MCP Server aus."""
        try:
            result = await self.mcp_manager.call_tool(tool_name, arguments)

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

            return result_text
        except Exception as e:
            return f"Fehler bei Tool '{tool_name}': {str(e)}"

    async def chat(self, user_message: str) -> str:
        """
        Sendet eine Nachricht an den Agent und erhaelt eine Antwort.

        Args:
            user_message: Die Benutzernachricht

        Returns:
            Die Antwort des Agents
        """
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        messages = [
            {"role": "system", "content": self._system_prompt},
            *self.conversation_history
        ]

        tools = self.mcp_manager.get_tools_for_openai()
        client = self._get_client()

        while True:
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools if tools else None,
                temperature=0.7,
            )

            assistant_message = response.choices[0].message

            if assistant_message.tool_calls:
                assistant_dict = {
                    "role": "assistant",
                    "content": assistant_message.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        for tc in assistant_message.tool_calls
                    ]
                }
                messages.append(assistant_dict)

                for tool_call in assistant_message.tool_calls:
                    tool_name = tool_call.function.name
                    arguments = json.loads(tool_call.function.arguments)

                    print(f"[TOOL] Fuehre aus: {tool_name}")
                    result = await self._execute_tool_call(tool_name, arguments)
                    print(f"[TOOL] Ergebnis: {result[:200]}..." if len(result) > 200 else f"[TOOL] Ergebnis: {result}")

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result
                    })
            else:
                response_content = assistant_message.content or ""
                self.conversation_history.append({
                    "role": "assistant",
                    "content": response_content
                })
                return response_content

    async def run_interactive(self) -> None:
        """Startet eine interaktive Chat-Schleife."""
        print("\n" + "=" * 60)
        print("Einfacher MCP Agent")
        print("=" * 60)

        servers = self.mcp_manager.list_servers()
        if servers:
            print("\nVerfuegbare MCP Server:")
            for server in servers:
                tools_count = len(self.mcp_manager.get_server_tools(server))
                print(f"  - {server} ({tools_count} Tools)")
        else:
            print("\nKeine MCP Server geladen.")

        print("\nBefehle:")
        print("  /clear - Konversation loeschen")
        print("  /tools - Alle verfuegbaren Tools anzeigen")
        print("  exit   - Beenden")
        print("=" * 60 + "\n")

        while True:
            try:
                user_input = input("\nSie: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ["exit", "quit", "q"]:
                    print("Auf Wiedersehen!")
                    break

                if user_input.lower() == "/clear":
                    self.clear_history()
                    continue

                if user_input.lower() == "/tools":
                    tools = self.mcp_manager.get_all_tools()
                    print(f"\nVerfuegbare Tools ({len(tools)}):")
                    for tool in tools:
                        desc = tool.get("description", "")[:60]
                        print(f"  - {tool['name']}: {desc}...")
                    continue

                response = await self.chat(user_input)
                print(f"\nAssistent: {response}")

            except KeyboardInterrupt:
                print("\n\nAbgebrochen. Auf Wiedersehen!")
                break
            except Exception as e:
                print(f"\n[FEHLER] {e}")
