"""
Studio bridge: intent routing for Gmail, Calendar, and voice responses.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional

from pittqlab_utils.llm import ClassifierRouter, OllamaBackend, PortkeyBackend
from pittqlab_utils.llm.pittai import PittAIClient, PittAIModels
from pittqlab_utils.tools import (
    CalendarTool,
    GmailTool,
    GoogleAuthManager,
    VoiceResponse,
)

INTENT_LABELS = [
    "email_read",
    "email_draft",
    "calendar_read",
    "calendar_create",
    "chitchat",
    "unknown",
]


class StudioBridge:
    """Orchestrates intent classification and tool routing for voice/transcript input."""

    def __init__(
        self,
        auth: Optional[GoogleAuthManager] = None,
        router: Optional[ClassifierRouter] = None,
    ):
        self._auth = auth or GoogleAuthManager()
        self._gmail = GmailTool(self._auth)
        self._calendar = CalendarTool(self._auth)
        self._voice = VoiceResponse(lang="en", slow=False)

        if router is not None:
            self._router = router
        else:
            ollama = OllamaBackend(model="llama3.2")
            portkey = PortkeyBackend(model=PittAIModels.CHEAP)
            self._router = ClassifierRouter([ollama, portkey], confidence_threshold=0.75)

        self._pittai = PittAIClient(default_model=PittAIModels.BALANCED)

    async def handle_transcript(
        self,
        transcript: str,
        chat_id: int,
        bot: Any,
    ) -> None:
        """Classify intent and route to appropriate tool handler."""
        result = await self._router.classify(transcript, labels=INTENT_LABELS)
        label = result.label

        if label == "email_read":
            summary = await self._gmail.get_unread_summary(
                max_emails=10,
                router=self._router,
            )
            await self._voice.send_voice_note(bot, chat_id, summary)

        elif label == "email_draft":
            extracted = await self._extract_email_fields(transcript)
            to = extracted.get("to", "")
            subject = extracted.get("subject", "")
            body = extracted.get("body", "")
            if not to or not subject:
                await self._send_text(bot, chat_id, "I need a recipient and subject to draft an email.")
                return
            confirmation = await self._gmail.draft_email(to=to, subject=subject, body=body)
            await self._send_text(bot, chat_id, confirmation)

        elif label == "calendar_read":
            days = await self._extract_days_ahead(transcript)
            summary = await self._calendar.get_events(
                days_ahead=days,
                router=self._router,
            )
            await self._voice.send_voice_note(bot, chat_id, summary)

        elif label == "calendar_create":
            extracted = await self._extract_calendar_fields(transcript)
            title = extracted.get("title", "Event")
            start = extracted.get("start", "")
            end = extracted.get("end", "")
            description = extracted.get("description", "")
            if not start or not end:
                await self._send_text(bot, chat_id, "I need start and end times to create an event.")
                return
            confirmation = await self._calendar.create_event(
                title=title,
                start=start,
                end=end,
                description=description,
            )
            await self._send_text(bot, chat_id, confirmation)

        else:
            # chitchat, unknown
            reply = await self._router.generate(
                [{"role": "user", "content": transcript}],
                system_prompt="Respond briefly and helpfully.",
                max_tokens=256,
            )
            await self._voice.send_voice_note(bot, chat_id, reply.text)

    async def _extract_email_fields(self, transcript: str) -> dict:
        """Extract to, subject, body from transcript via LLM (BALANCED, json_mode)."""
        prompt = (
            f"Extract email fields from this transcript. Return JSON with keys: to, subject, body.\n"
            f"Transcript: {transcript}"
        )
        system = "Extract the recipient (to), subject, and body. Use empty string if not mentioned."
        data = await asyncio.to_thread(
            self._pittai.chat_json,
            prompt,
            system=system,
            model=PittAIModels.BALANCED,
            max_tokens=512,
        )
        if isinstance(data, dict):
            return {
                "to": str(data.get("to", "")),
                "subject": str(data.get("subject", "")),
                "body": str(data.get("body", "")),
            }
        return {"to": "", "subject": "", "body": ""}

    async def _extract_calendar_fields(self, transcript: str) -> dict:
        """Extract title, start, end from transcript via LLM (BALANCED, json_mode)."""
        prompt = (
            f"Extract calendar event fields from this transcript. Return JSON with keys: "
            f"title, start, end, description. Use ISO 8601 for start and end (e.g. 2025-03-17T14:00:00).\n"
            f"Transcript: {transcript}"
        )
        system = "Extract event title, start (ISO 8601), end (ISO 8601), and optional description."
        data = await asyncio.to_thread(
            self._pittai.chat_json,
            prompt,
            system=system,
            model=PittAIModels.BALANCED,
            max_tokens=512,
        )
        if isinstance(data, dict):
            return {
                "title": str(data.get("title", "Event")),
                "start": str(data.get("start", "")),
                "end": str(data.get("end", "")),
                "description": str(data.get("description", "")),
            }
        return {"title": "Event", "start": "", "end": "", "description": ""}

    async def _extract_days_ahead(self, transcript: str) -> int:
        """Extract days_ahead from transcript (default 1)."""
        prompt = (
            f"How many days ahead should we look for calendar events? "
            f"Transcript: {transcript}. Reply with a JSON object: {{\"days\": <number>}}"
        )
        try:
            data = await asyncio.to_thread(
                self._pittai.chat_json,
                prompt,
                system="Extract the number of days. Default to 1 if unclear.",
                model=PittAIModels.BALANCED,
                max_tokens=64,
            )
            if isinstance(data, dict) and "days" in data:
                return max(1, min(7, int(data["days"])))
        except Exception:
            pass
        return 1

    async def _send_text(self, bot: Any, chat_id: int, text: str) -> None:
        """Send text reply. Handles sync or async bot.send_message."""
        result = bot.send_message(chat_id=chat_id, text=text)
        if asyncio.iscoroutine(result):
            await result
        # If sync, it already executed
