"""
Gmail tool: fetch unread emails, create drafts (never send).
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from .google_auth import GoogleAuthManager


class GmailTool:
    """Gmail operations: unread summary and draft creation."""

    def __init__(self, auth: GoogleAuthManager):
        self._auth = auth

    async def _fetch_unread(self, max_results: int = 10) -> List[Dict[str, Any]]:
        """Fetch unread emails from INBOX. Returns list of {sender, subject, snippet, date} dicts."""
        def _sync_fetch() -> List[Dict[str, Any]]:
            from googleapiclient.discovery import build

            creds = self._auth.get_credentials()
            service = build("gmail", "v1", credentials=creds)
            results = service.users().messages().list(
                userId="me",
                labelIds=["INBOX"],
                q="is:unread category:primary",
                maxResults=max_results,
            ).execute()
            messages = results.get("messages", [])
            out: List[Dict[str, Any]] = []
            for msg_ref in messages:
                msg = service.users().messages().get(
                    userId="me",
                    id=msg_ref["id"],
                    format="metadata",
                    metadataHeaders=["From", "Subject", "Date"],
                ).execute()
                headers = {h["name"].lower(): h["value"] for h in msg.get("payload", {}).get("headers", [])}
                out.append({
                    "sender": headers.get("from", "Unknown"),
                    "subject": headers.get("subject", "(No subject)"),
                    "snippet": msg.get("snippet", ""),
                    "date": headers.get("date", ""),
                })
            return out

        return await asyncio.to_thread(_sync_fetch)

    async def get_unread_summary(
        self,
        max_emails: int = 10,
        router: Optional[Any] = None,
    ) -> str:
        """Fetch unread emails, build prompt, call ClassifierRouter.generate() for spoken summary."""
        emails = await self._fetch_unread(max_results=max_emails)
        if not emails:
            return "You have no unread emails."

        prompt_parts = ["Unread emails:\n"]
        for i, e in enumerate(emails, 1):
            prompt_parts.append(
                f"{i}. From: {e['sender']}\n   Subject: {e['subject']}\n   Date: {e['date']}\n   Snippet: {e['snippet']}\n"
            )
        prompt = "\n".join(prompt_parts)
        system = (
            "Summarize these unread emails in 2-3 short sentences per email, "
            "suitable for text-to-speech. Be concise and natural."
        )

        if router is not None:
            result = await router.generate(
                [{"role": "user", "content": prompt}],
                system_prompt=system,
                max_tokens=1024,
            )
            return result.text

        # Fallback if no router: simple concatenation
        lines = []
        for e in emails:
            lines.append(f"From {e['sender']}: {e['subject']}. {e['snippet'][:100]}...")
        return " ".join(lines)

    async def draft_email(self, to: str, subject: str, body: str) -> str:
        """Create a Gmail draft (does NOT send). Returns confirmation string."""
        def _sync_draft() -> str:
            from email.mime.text import MIMEText
            import base64
            from googleapiclient.discovery import build

            creds = self._auth.get_credentials()
            service = build("gmail", "v1", credentials=creds)
            message = MIMEText(body)
            message["to"] = to
            message["subject"] = subject
            raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
            draft = service.users().drafts().create(
                userId="me",
                body={"message": {"raw": raw}},
            ).execute()
            return f"Draft created. Subject: {subject}. Recipient: {to}."

        return await asyncio.to_thread(_sync_draft)
