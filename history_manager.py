"""
Zentraler History Manager
Verwaltet die Konversationshistorie fuer alle Agents.
"""

import threading
from typing import Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class MessageRole(Enum):
    """Rollen fuer Nachrichten in der Historie."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class HistoryMessage:
    """Eine einzelne Nachricht in der Historie."""
    role: MessageRole
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    agent_name: Optional[str] = None  # Welcher Agent hat diese Nachricht erstellt
    tool_name: Optional[str] = None   # Falls Tool-Aufruf: Name des Tools
    tool_call_id: Optional[str] = None  # Falls Tool-Antwort: ID des Tool-Aufrufs
    metadata: dict = field(default_factory=dict)  # Zusaetzliche Metadaten

    def to_dict(self) -> dict:
        """Konvertiert die Nachricht zu einem Dictionary."""
        return {
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "agent_name": self.agent_name,
            "tool_name": self.tool_name,
            "tool_call_id": self.tool_call_id,
            "metadata": self.metadata
        }

    def to_openai_format(self) -> dict:
        """Konvertiert die Nachricht zum OpenAI API Format."""
        msg = {"role": self.role.value, "content": self.content}
        if self.tool_call_id and self.role == MessageRole.TOOL:
            msg["tool_call_id"] = self.tool_call_id
        return msg


class HistoryManager:
    """
    Singleton History Manager fuer zentrale Historien-Verwaltung.
    Alle Agents koennen auf die gemeinsame Historie zugreifen.
    """

    _instance: Optional['HistoryManager'] = None
    _lock = threading.Lock()

    def __new__(cls) -> 'HistoryManager':
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._history: list[HistoryMessage] = []
        self._history_lock = threading.Lock()
        self._max_messages: int = 100  # Maximale Anzahl Nachrichten
        self._initialized = True

    def add_message(
        self,
        role: MessageRole,
        content: str,
        agent_name: Optional[str] = None,
        tool_name: Optional[str] = None,
        tool_call_id: Optional[str] = None,
        metadata: Optional[dict] = None
    ) -> HistoryMessage:
        """
        Fuegt eine neue Nachricht zur Historie hinzu.

        Args:
            role: Die Rolle der Nachricht (system, user, assistant, tool)
            content: Der Inhalt der Nachricht
            agent_name: Name des Agents, der die Nachricht erstellt hat
            tool_name: Name des Tools (falls Tool-Aufruf)
            tool_call_id: ID des Tool-Aufrufs (falls Tool-Antwort)
            metadata: Zusaetzliche Metadaten

        Returns:
            Die erstellte HistoryMessage
        """
        message = HistoryMessage(
            role=role,
            content=content,
            agent_name=agent_name,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            metadata=metadata or {}
        )

        with self._history_lock:
            self._history.append(message)
            # Alte Nachrichten entfernen wenn Limit erreicht
            if len(self._history) > self._max_messages:
                self._history = self._history[-self._max_messages:]

        return message

    def add_user_message(self, content: str, agent_name: Optional[str] = None) -> HistoryMessage:
        """Fuegt eine Benutzer-Nachricht hinzu."""
        return self.add_message(MessageRole.USER, content, agent_name=agent_name)

    def add_assistant_message(self, content: str, agent_name: Optional[str] = None) -> HistoryMessage:
        """Fuegt eine Assistenten-Nachricht hinzu."""
        return self.add_message(MessageRole.ASSISTANT, content, agent_name=agent_name)

    def add_system_message(self, content: str, agent_name: Optional[str] = None) -> HistoryMessage:
        """Fuegt eine System-Nachricht hinzu."""
        return self.add_message(MessageRole.SYSTEM, content, agent_name=agent_name)

    def add_tool_message(
        self,
        content: str,
        tool_call_id: str,
        tool_name: Optional[str] = None,
        agent_name: Optional[str] = None
    ) -> HistoryMessage:
        """Fuegt eine Tool-Antwort hinzu."""
        return self.add_message(
            MessageRole.TOOL,
            content,
            agent_name=agent_name,
            tool_name=tool_name,
            tool_call_id=tool_call_id
        )

    def get_history(self, limit: Optional[int] = None) -> list[HistoryMessage]:
        """
        Gibt die Historie zurueck.

        Args:
            limit: Optionale Begrenzung der Anzahl (neueste zuerst)

        Returns:
            Liste der HistoryMessages
        """
        with self._history_lock:
            if limit:
                return list(self._history[-limit:])
            return list(self._history)

    def get_history_for_agent(self, agent_name: str, limit: Optional[int] = None) -> list[HistoryMessage]:
        """
        Gibt nur Nachrichten eines bestimmten Agents zurueck.

        Args:
            agent_name: Name des Agents
            limit: Optionale Begrenzung

        Returns:
            Gefilterte Liste der HistoryMessages
        """
        with self._history_lock:
            filtered = [msg for msg in self._history if msg.agent_name == agent_name]
            if limit:
                return filtered[-limit:]
            return filtered

    def get_history_as_openai_messages(self, limit: Optional[int] = None) -> list[dict]:
        """
        Gibt die Historie im OpenAI API Format zurueck.

        Args:
            limit: Optionale Begrenzung

        Returns:
            Liste von Dictionaries im OpenAI Format
        """
        history = self.get_history(limit)
        return [msg.to_openai_format() for msg in history]

    def get_context_summary(self, max_length: int = 2000) -> str:
        """
        Erstellt eine Zusammenfassung der Historie fuer Kontext.

        Args:
            max_length: Maximale Laenge der Zusammenfassung

        Returns:
            Zusammenfassung als String
        """
        with self._history_lock:
            if not self._history:
                return "Keine vorherige Konversation."

            summary_parts = []
            current_length = 0

            # Von hinten nach vorne durchgehen (neueste zuerst)
            for msg in reversed(self._history):
                role_prefix = {
                    MessageRole.USER: "Benutzer",
                    MessageRole.ASSISTANT: "Assistent",
                    MessageRole.SYSTEM: "System",
                    MessageRole.TOOL: "Tool"
                }.get(msg.role, "Unbekannt")

                agent_info = f" ({msg.agent_name})" if msg.agent_name else ""
                line = f"{role_prefix}{agent_info}: {msg.content[:200]}"

                if current_length + len(line) > max_length:
                    break

                summary_parts.insert(0, line)
                current_length += len(line)

            return "\n".join(summary_parts)

    def clear(self) -> None:
        """Loescht die gesamte Historie."""
        with self._history_lock:
            self._history.clear()

    def clear_agent_history(self, agent_name: str) -> int:
        """
        Loescht alle Nachrichten eines bestimmten Agents.

        Args:
            agent_name: Name des Agents

        Returns:
            Anzahl der geloeschten Nachrichten
        """
        with self._history_lock:
            original_count = len(self._history)
            self._history = [msg for msg in self._history if msg.agent_name != agent_name]
            return original_count - len(self._history)

    @property
    def message_count(self) -> int:
        """Gibt die Anzahl der Nachrichten in der Historie zurueck."""
        with self._history_lock:
            return len(self._history)

    @property
    def is_empty(self) -> bool:
        """Prueft ob die Historie leer ist."""
        with self._history_lock:
            return len(self._history) == 0

    def set_max_messages(self, max_messages: int) -> None:
        """Setzt die maximale Anzahl an Nachrichten."""
        with self._history_lock:
            self._max_messages = max_messages
            if len(self._history) > max_messages:
                self._history = self._history[-max_messages:]


# Globale Instanz fuer einfachen Zugriff
_history_manager: Optional[HistoryManager] = None


def get_history_manager() -> HistoryManager:
    """
    Gibt die globale HistoryManager-Instanz zurueck.

    Returns:
        Die HistoryManager Singleton-Instanz.
    """
    global _history_manager
    if _history_manager is None:
        _history_manager = HistoryManager()
    return _history_manager
