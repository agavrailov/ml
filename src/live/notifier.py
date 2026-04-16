"""Telegram alert notifier for the live trading daemon.

Uses only stdlib urllib.request — no extra dependencies.
Configure via environment variables:
    TELEGRAM_BOT_TOKEN  — bot token from @BotFather
    TELEGRAM_CHAT_ID    — chat/group ID to send alerts to

Alert delivery failures are silently swallowed so they never block trading.
"""
from __future__ import annotations

import json
import os
import urllib.request
from typing import Optional


class TelegramNotifier:
    """Sends text alerts to a Telegram chat via the Bot API.

    Instantiate once at startup; call send_alert() from the poll loop.
    If either env var is missing the notifier is a no-op.
    """

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
        timeout: int = 5,
    ) -> None:
        self._token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self._chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "")
        self._timeout = timeout
        self._enabled = bool(self._token and self._chat_id)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def send_alert(self, message: str) -> None:
        """Send *message* to the configured Telegram chat.

        Never raises — all exceptions are silently swallowed.
        """
        if not self._enabled:
            return
        try:
            url = f"https://api.telegram.org/bot{self._token}/sendMessage"
            payload = json.dumps({"chat_id": self._chat_id, "text": message}).encode()
            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=self._timeout)
        except Exception:
            pass


# Module-level singleton — initialised from environment at import time.
# The poll loop can import and call `notifier.send_alert()` directly,
# or construct its own instance with explicit credentials for testing.
notifier = TelegramNotifier()
