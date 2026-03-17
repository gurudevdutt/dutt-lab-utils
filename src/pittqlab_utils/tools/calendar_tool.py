"""
Google Calendar tool: fetch events, create events.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from .google_auth import GoogleAuthManager


def _parse_rfc3339(dt_str: str) -> Optional[datetime]:
    """Parse RFC3339 datetime string."""
    try:
        if "Z" in dt_str or "+" in dt_str or dt_str.count("-") > 2:
            return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        return datetime.fromisoformat(dt_str)
    except (ValueError, TypeError):
        return None


class CalendarTool:
    """Calendar operations: list events and create events."""

    def __init__(self, auth: GoogleAuthManager):
        self._auth = auth

    async def _fetch_events(
        self,
        time_min: datetime,
        time_max: datetime,
    ) -> List[Dict[str, Any]]:
        """Fetch events from primary calendar. Returns list of {title, start, end, location, description} dicts."""
        def _sync_fetch() -> List[Dict[str, Any]]:
            from googleapiclient.discovery import build

            creds = self._auth.get_credentials()
            service = build("calendar", "v3", credentials=creds)
            tmin = time_min.isoformat() + "Z" if time_min.tzinfo is None else time_min.isoformat()
            tmax = time_max.isoformat() + "Z" if time_max.tzinfo is None else time_max.isoformat()
            events_result = service.events().list(
                calendarId="primary",
                timeMin=tmin,
                timeMax=tmax,
                singleEvents=True,
                orderBy="startTime",
            ).execute()
            events = events_result.get("items", [])
            out: List[Dict[str, Any]] = []
            for ev in events:
                start = ev.get("start", {})
                end = ev.get("end", {})
                start_str = start.get("dateTime") or start.get("date", "")
                end_str = end.get("dateTime") or end.get("date", "")
                out.append({
                    "title": ev.get("summary", "(No title)"),
                    "start": start_str,
                    "end": end_str,
                    "location": ev.get("location", ""),
                    "description": ev.get("description", ""),
                })
            return out

        return await asyncio.to_thread(_sync_fetch)

    async def get_events(
        self,
        days_ahead: int = 1,
        router: Optional[Any] = None,
    ) -> str:
        """Fetch events for today + days_ahead, produce spoken summary via ClassifierRouter."""
        now = datetime.now(timezone.utc)
        time_min = now.replace(hour=0, minute=0, second=0, microsecond=0)
        time_max = time_min + timedelta(days=days_ahead + 1)

        events = await self._fetch_events(time_min, time_max)
        if not events:
            return "You have no events scheduled for this period."

        prompt_parts = ["Upcoming calendar events:\n"]
        for i, e in enumerate(events, 1):
            prompt_parts.append(
                f"{i}. {e['title']} from {e['start']} to {e['end']}"
                + (f" at {e['location']}" if e.get("location") else "")
                + "\n"
            )
        prompt = "\n".join(prompt_parts)
        system = (
            "Summarize these calendar events in a natural, spoken style "
            "suitable for text-to-speech. Be concise."
        )

        if router is not None:
            result = await router.generate(
                [{"role": "user", "content": prompt}],
                system_prompt=system,
                max_tokens=1024,
            )
            return result.text

        lines = []
        for e in events:
            lines.append(f"{e['title']} from {e['start']} to {e['end']}")
        return " ".join(lines)

    async def create_event(
        self,
        title: str,
        start: str,
        end: str,
        description: str = "",
    ) -> str:
        """Create a calendar event. start/end are ISO 8601 strings. Returns confirmation."""
        def _sync_create() -> str:
            from googleapiclient.discovery import build

            creds = self._auth.get_credentials()
            service = build("calendar", "v3", credentials=creds)
            body = {
                "summary": title,
                "description": description,
                "start": {"dateTime": start, "timeZone": "UTC"},
                "end": {"dateTime": end, "timeZone": "UTC"},
            }
            service.events().insert(calendarId="primary", body=body).execute()
            return f"Event created: {title} from {start} to {end}."

        return await asyncio.to_thread(_sync_create)
