"""
Email alerting via SMTP. Reads credentials from environment variables:

    SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, ALERT_EMAIL_TO

For Gmail, SMTP_USER/SMTP_PASSWORD must be an app password, not your
regular login password. Put these in alerts/.env (gitignored).
"""
from __future__ import annotations

import os
import smtplib
from email.message import EmailMessage

from dotenv import load_dotenv

load_dotenv()

SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER = os.environ.get("SMTP_USER")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD")
ALERT_EMAIL_TO = os.environ.get("ALERT_EMAIL_TO")


def send_email_alert(subject: str, body: str, image_path: str | None = None) -> None:
    if not SMTP_USER or not SMTP_PASSWORD or not ALERT_EMAIL_TO:
        print("[email] skipped — SMTP_USER / SMTP_PASSWORD / ALERT_EMAIL_TO not set")
        return

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = SMTP_USER
    msg["To"] = ALERT_EMAIL_TO
    msg.set_content(body)

    if image_path:
        with open(image_path, "rb") as f:
            msg.add_attachment(f.read(), maintype="image", subtype="jpeg", filename="detection.jpg")

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.send_message(msg)
    except smtplib.SMTPException as exc:
        print(f"[email] failed to send alert: {exc}")


if __name__ == "__main__":
    send_email_alert("Test alert", "Test alert from factory-defect-detection")
