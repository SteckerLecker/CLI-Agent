"""
Zentraler Token Manager
Verwaltet Access Tokens fuer alle KI-Anfragen.
"""

import time
import threading
from typing import Optional


class TokenManager:
    """
    Singleton Token Manager fuer zentrale Token-Verwaltung.
    Stellt Access Tokens fuer alle KI-Anfragen bereit.
    """

    _instance: Optional['TokenManager'] = None
    _lock = threading.Lock()

    def __new__(cls) -> 'TokenManager':
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._access_token: Optional[str] = None
        self._token_expiry: Optional[float] = None
        self._token_lock = threading.Lock()
        self._initialized = True

    def _fetch_token(self) -> str:
        """
        Ruft einen neuen Access Token ab.

        PLATZHALTER: Hier die eigentliche Request-Logik einfuegen.

        Returns:
            Der abgerufene Access Token.
        """
        # ============================================
        # PLATZHALTER - Request-Logik hier einfuegen
        # ============================================
        # Beispiel:
        # response = requests.post(
        #     "https://auth.example.com/oauth/token",
        #     json={
        #         "client_id": "...",
        #         "client_secret": "...",
        #         "grant_type": "client_credentials"
        #     }
        # )
        # data = response.json()
        # self._token_expiry = time.time() + data.get("expires_in", 3600)
        # return data["access_token"]
        # ============================================

        # Temporaerer Platzhalter-Token
        self._token_expiry = time.time() + 3600  # 1 Stunde gueltig
        return "PLATZHALTER_ACCESS_TOKEN"

    def _is_token_expired(self) -> bool:
        """Prueft ob der aktuelle Token abgelaufen ist."""
        if self._token_expiry is None:
            return True
        # Token 60 Sekunden vor Ablauf als abgelaufen betrachten
        return time.time() >= (self._token_expiry - 60)

    def get_token(self) -> str:
        """
        Gibt den aktuellen Access Token zurueck.
        Ruft automatisch einen neuen Token ab, falls keiner vorhanden oder abgelaufen.

        Returns:
            Der aktuelle Access Token.
        """
        with self._token_lock:
            if self._access_token is None or self._is_token_expired():
                self._access_token = self._fetch_token()
            return self._access_token

    def refresh_token(self) -> str:
        """
        Erzwingt das Abrufen eines neuen Tokens.

        Returns:
            Der neue Access Token.
        """
        with self._token_lock:
            self._access_token = self._fetch_token()
            return self._access_token

    def invalidate_token(self) -> None:
        """Invalidiert den aktuellen Token (z.B. nach 401-Fehler)."""
        with self._token_lock:
            self._access_token = None
            self._token_expiry = None

    @property
    def has_valid_token(self) -> bool:
        """Prueft ob ein gueltiger Token vorhanden ist."""
        with self._token_lock:
            return self._access_token is not None and not self._is_token_expired()


# Globale Instanz fuer einfachen Zugriff
_token_manager: Optional[TokenManager] = None


def get_token_manager() -> TokenManager:
    """
    Gibt die globale TokenManager-Instanz zurueck.

    Returns:
        Die TokenManager Singleton-Instanz.
    """
    global _token_manager
    if _token_manager is None:
        _token_manager = TokenManager()
    return _token_manager
