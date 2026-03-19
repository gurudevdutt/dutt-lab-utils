"""Tests for GmailTool. Mocks googleapiclient entirely."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from pittqlab_utils.tools.gmail_tool import GmailTool
from pittqlab_utils.tools.google_auth import GoogleAuthManager


@pytest.fixture
def mock_auth():
    auth = MagicMock(spec=GoogleAuthManager)
    auth.get_credentials.return_value = MagicMock()
    return auth


@pytest.fixture
def mock_gmail_service():
    """Mock Gmail API service."""
    with patch("googleapiclient.discovery.build") as mock_build:
        service = MagicMock()
        mock_build.return_value = service

        # messages().list()
        list_result = {"messages": [{"id": "msg1"}, {"id": "msg2"}]}
        service.users.return_value.messages.return_value.list.return_value.execute.return_value = list_result

        # messages().get() for each message
        def get_side_effect(**kwargs):
            m = MagicMock()
            m.execute.return_value = {
                "payload": {
                    "headers": [
                        {"name": "From", "value": "sender@example.com"},
                        {"name": "Subject", "value": "Test Subject"},
                        {"name": "Date", "value": "Mon, 17 Mar 2025 10:00:00"},
                    ]
                },
                "snippet": "Email body snippet here.",
            }
            return m

        service.users.return_value.messages.return_value.get.side_effect = get_side_effect
        yield service


@pytest.mark.asyncio
async def test_get_unread_summary_returns_non_empty_string(mock_auth, mock_gmail_service):
    """Test get_unread_summary returns a non-empty string."""
    tool = GmailTool(mock_auth)
    router = MagicMock()
    router.generate = AsyncMock(return_value=MagicMock(text="You have 2 emails from sender@example.com."))
    result = await tool.get_unread_summary(max_emails=10, router=router)
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.asyncio
async def test_draft_email_calls_drafts_create_never_messages_send(mock_auth):
    """Test draft_email calls drafts.create, never messages.send."""
    with patch("googleapiclient.discovery.build") as mock_build:
        service = MagicMock()
        mock_build.return_value = service
        service.users.return_value.drafts.return_value.create.return_value.execute.return_value = {"id": "draft1"}

        tool = GmailTool(mock_auth)
        result = await tool.draft_email(to="recipient@example.com", subject="Hi", body="Hello")

        service.users.return_value.drafts.return_value.create.assert_called_once()
        # Ensure messages.send was NEVER called
        if hasattr(service.users.return_value.messages.return_value, "send"):
            service.users.return_value.messages.return_value.send.assert_not_called()
        assert "Draft created" in result
        assert "recipient@example.com" in result


@pytest.mark.asyncio
async def test_fetch_unread_returns_correctly_shaped_dicts(mock_auth, mock_gmail_service):
    """Test _fetch_unread returns list of {sender, subject, snippet, date} dicts."""
    tool = GmailTool(mock_auth)
    result = await tool._fetch_unread(max_results=5)
    assert isinstance(result, list)
    for item in result:
        assert "sender" in item
        assert "subject" in item
        assert "snippet" in item
        assert "date" in item
        assert isinstance(item["sender"], str)
        assert isinstance(item["subject"], str)
        assert isinstance(item["snippet"], str)
        assert isinstance(item["date"], str)
