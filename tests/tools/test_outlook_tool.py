"""Tests for OutlookTool. Mocks httpx and PittAIClient."""

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest
import respx

from pittqlab_utils.tools.ms_auth import MicrosoftAuthManager
from pittqlab_utils.tools.outlook_tool import OutlookTool, GRAPH_BASE


@pytest.fixture
def mock_auth():
    auth = MagicMock(spec=MicrosoftAuthManager)
    auth.get_token.return_value = "fake-bearer-token"
    return auth


@pytest.mark.asyncio
@respx.mock
async def test_fetch_unread_returns_correctly_shaped_dicts(mock_auth):
    graph_response = {
        "value": [
            {
                "sender": {
                    "emailAddress": {
                        "name": "Alice",
                        "address": "alice@example.com",
                    }
                },
                "subject": "Hello",
                "bodyPreview": "Preview text",
                "receivedDateTime": "2026-04-01T12:00:00Z",
            }
        ]
    }
    respx.get(f"{GRAPH_BASE}/me/mailFolders/inbox/messages").mock(
        return_value=httpx.Response(200, json=graph_response)
    )

    tool = OutlookTool(mock_auth)
    result = await tool._fetch_unread(max_results=5)

    assert isinstance(result, list)
    assert len(result) == 1
    item = result[0]
    assert item["sender"] == "Alice <alice@example.com>"
    assert item["subject"] == "Hello"
    assert item["snippet"] == "Preview text"
    assert item["date"] == "2026-04-01T12:00:00Z"


@pytest.mark.asyncio
@respx.mock
async def test_get_unread_summary_returns_non_empty_string(mock_auth):
    respx.get(f"{GRAPH_BASE}/me/mailFolders/inbox/messages").mock(
        return_value=httpx.Response(
            200,
            json={
                "value": [
                    {
                        "sender": {"emailAddress": {"address": "b@example.com"}},
                        "subject": "Subj",
                        "bodyPreview": "Snip",
                        "receivedDateTime": "2026-04-01T10:00:00Z",
                    }
                ]
            },
        )
    )

    mock_resp = MagicMock()
    mock_resp.text = "You have one email from b@example.com about Subj."

    with patch("pittqlab_utils.tools.outlook_tool.PittAIClient") as MockClient:
        instance = MockClient.return_value
        instance.chat.return_value = mock_resp

        tool = OutlookTool(mock_auth)
        out = await tool.get_unread_summary(max_emails=10)

    assert isinstance(out, str)
    assert len(out) > 0
    instance.chat.assert_called_once()
    call_kw = instance.chat.call_args
    assert "untrusted external data" in call_kw[0][0]
    assert "<email_content>" in call_kw[0][0]


@pytest.mark.asyncio
@respx.mock
async def test_draft_email_posts_to_messages_not_sendmail(mock_auth):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["path"] = request.url.path
        return httpx.Response(201, json={"id": "msg1"})

    respx.post(f"{GRAPH_BASE}/me/messages").mock(side_effect=handler)

    tool = OutlookTool(mock_auth)
    result = await tool.draft_email(to="to@example.com", subject="S", body="B")

    assert "/me/messages" in captured["path"]
    assert "sendMail" not in captured["url"]
    assert "Draft created" in result
    assert "to@example.com" in result


@pytest.mark.asyncio
@respx.mock
async def test_draft_email_body_contains_is_draft_true(mock_auth):
    captured_body = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_body
        captured_body = json.loads(request.content.decode("utf-8"))
        return httpx.Response(201, json={"id": "m2"})

    respx.post(f"{GRAPH_BASE}/me/messages").mock(side_effect=handler)

    tool = OutlookTool(mock_auth)
    await tool.draft_email(to="x@y.com", subject="Hi", body="Body")

    assert captured_body is not None
    assert captured_body.get("isDraft") is True
    assert captured_body.get("subject") == "Hi"
    assert captured_body.get("body", {}).get("contentType") == "Text"
    assert captured_body.get("body", {}).get("content") == "Body"
    assert captured_body.get("toRecipients") == [{"emailAddress": {"address": "x@y.com"}}]
