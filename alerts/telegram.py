"""
Telegram alerting. Reads credentials from environment variables so
nothing secret lives in source:

    TELEGRAM_BOT_TOKEN   token from @BotFather
    TELEGRAM_CHAT_ID     chat/channel id to post to

Create alerts/.env (gitignored) with these two vars and load it via
python-dotenv, or export them in your shell before running inference.
"""
from __future__ import annotations

import os

import requests
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")


def send_telegram_alert(message: str, image_path: str | None = None) -> None:
    if not BOT_TOKEN or not CHAT_ID:
        print("[telegram] skipped — TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set")
        return

    base_url = f"https://api.telegram.org/bot{BOT_TOKEN}"
    try:
        if image_path:
            with open(image_path, "rb") as f:
                requests.post(
                    f"{base_url}/sendPhoto",
                    data={"chat_id": CHAT_ID, "caption": message},
                    files={"photo": f},
                    timeout=10,
                )
        else:
            requests.post(
                f"{base_url}/sendMessage",
                data={"chat_id": CHAT_ID, "text": message},
                timeout=10,
            )
    except requests.RequestException as exc:
        print(f"[telegram] failed to send alert: {exc}")


if __name__ == "__main__":
    send_telegram_alert("Test alert from factory-defect-detection")
