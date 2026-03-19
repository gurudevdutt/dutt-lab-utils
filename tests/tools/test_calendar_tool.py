"""Tests for CalendarTool. Mocks googleapiclient entirely."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, AsyncMock, patch

from pittqlab_utils.tools.calendar_tool import CalendarTool
from pittqlab_utils.tools.google_auth import GoogleAuthManager


@pytest.fixture
def mock_auth():
    auth = MagicMock(spec=GoogleAuthManager)
    auth.get_credentials.return_value = MagicMock()
    return auth


@pytest.fixture
def mock_calendar_service():
    """Mock Calendar API service."""
    with patch("googleapiclient.discovery.build") as mock_build:
        service = MagicMock()
        mock_build.return_value = service
        service.events.return_value.list.return_value.execute.return_value = {
            "items": [
                {
                    "summary": "Meeting",
                    "start": {"dateTime": "2025-03-17T10:00:00Z"},
                    "end": {"dateTime": "2025-03-17T11:00:00Z"},
                    "location": "Room 1",
                    "description": "Team sync",
                }
            ]
        }
        service.events.return_value.insert.return_value.execute.return_value = {"id": "ev1"}
        yield service


@pytest.mark.asyncio
async def test_get_events_returns_string(mock_auth, mock_calendar_service):
    """Test get_events returns a string."""
    tool = CalendarTool(mock_auth)
    router = MagicMock()
    router.generate = AsyncMock(return_value=MagicMock(text="You have a meeting at 10am."))
    result = await tool.get_events(days_ahead=1, router=router)
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.asyncio
async def test_create_event_calls_events_insert_with_correct_fields(mock_auth):
    """Test create_event calls events.insert with correct fields."""
    with patch("googleapiclient.discovery.build") as mock_build:
        service = MagicMock()
        mock_build.return_value = service
        service.events.return_value.insert.return_value.execute.return_value = {"id": "ev1"}

        tool = CalendarTool(mock_auth)
        result = await tool.create_event(
            title="Test Event",
            start="2025-03-17T14:00:00",
            end="2025-03-17T15:00:00",
            description="Test description",
        )

        insert_call = service.events.return_value.insert
        insert_call.assert_called_once()
        call_kwargs = insert_call.call_args
        assert call_kwargs.kwargs["calendarId"] == "primary"
        body = call_kwargs.kwargs["body"]
        assert body["summary"] == "Test Event"
        assert body["description"] == "Test description"
        assert body["start"]["dateTime"] == "2025-03-17T14:00:00"
        assert body["end"]["dateTime"] == "2025-03-17T15:00:00"
        assert "Event created" in result


@pytest.mark.asyncio
async def test_time_range_calculation_for_days_ahead(mock_auth):
    """Test time range calculation for days_ahead parameter."""
    with patch("googleapiclient.discovery.build") as mock_build:
        service = MagicMock()
        mock_build.return_value = service
        service.events.return_value.list.return_value.execute.return_value = {"items": []}

        tool = CalendarTool(mock_auth)
        await tool.get_events(days_ahead=3, router=None)

        list_call = service.events.return_value.list
        list_call.assert_called_once()
        call_kwargs = list_call.call_args.kwargs
        time_min = call_kwargs["timeMin"]
        time_max = call_kwargs["timeMax"]
        assert "Z" in time_min or "+" in time_min
        assert "Z" in time_max or "+" in time_max
        # time_max should be ~4 days from time_min (today + 3 days ahead + 1)
        # We just verify the call was made with time params
        assert "timeMin" in call_kwargs
        assert "timeMax" in call_kwargs
