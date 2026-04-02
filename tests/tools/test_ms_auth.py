"""Tests for MicrosoftAuthManager. Mocks msal.PublicClientApplication."""

import json
from unittest.mock import MagicMock, patch

import pytest

from pittqlab_utils.tools.ms_auth import MicrosoftAuthManager, MicrosoftAuthError


def test_get_token_raises_when_credentials_missing(tmp_path):
    """Missing env / constructor args -> MicrosoftAuthError."""
    auth = MicrosoftAuthManager(token_path=tmp_path / "ms_token.json")
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(MicrosoftAuthError) as exc_info:
            auth.get_token()
    msg = str(exc_info.value).lower()
    assert "microsoft" in msg or "environment" in msg or "configuration" in msg


def test_get_token_returns_token_string(tmp_path, monkeypatch):
    """Silent acquire returns access_token string; PCA gets client_credential."""
    monkeypatch.setenv("MICROSOFT_CLIENT_ID", "cid")
    monkeypatch.setenv("MICROSOFT_CLIENT_SECRET", "secret")
    monkeypatch.setenv("MICROSOFT_TENANT_ID", "tid")

    auth = MicrosoftAuthManager(token_path=tmp_path / "ms_token.json")
    mock_app = MagicMock()
    mock_app.get_accounts.return_value = [{"username": "u@example.com"}]
    mock_app.acquire_token_silent.return_value = {"access_token": "token-abc-123"}

    with patch("pittqlab_utils.tools.ms_auth.msal.PublicClientApplication") as mock_pca:
        mock_pca.return_value = mock_app
        token = auth.get_token()

    assert token == "token-abc-123"
    mock_pca.assert_called_once()
    assert mock_pca.call_args.kwargs["client_credential"] == "secret"
    assert mock_pca.call_args.kwargs["token_cache"] is auth._cache
    assert mock_pca.call_args.kwargs["authority"] == "https://login.microsoftonline.com/tid"
    mock_app.acquire_token_silent.assert_called_once()
    mock_app.initiate_device_flow.assert_not_called()


def test_uses_cached_token_when_not_expired(tmp_path, monkeypatch):
    """Token file with valid cache: silent path used, no device flow."""
    monkeypatch.setenv("MICROSOFT_CLIENT_ID", "cid")
    monkeypatch.setenv("MICROSOFT_CLIENT_SECRET", "secret")
    monkeypatch.setenv("MICROSOFT_TENANT_ID", "tid")

    token_path = tmp_path / "ms_token.json"
    # Minimal serialized cache blob — MSAL's SerializableTokenCache accepts JSON
    cache_data = {
        "AccessToken": {
            "key1": {
                "credential_type": "AccessToken",
                "secret": "cached-access-token",
                "home_account_id": "hid",
                "environment": "login.microsoftonline.com",
                "client_id": "cid",
                "target": " ".join(
                    [
                        "https://graph.microsoft.com/Mail.Read",
                        "https://graph.microsoft.com/Mail.ReadWrite",
                        "https://graph.microsoft.com/Calendars.Read",
                        "https://graph.microsoft.com/Calendars.ReadWrite",
                    ]
                ),
            }
        },
        "RefreshToken": {},
        "IdToken": {},
        "Account": {},
        "AppMetadata": {},
    }
    token_path.write_text(json.dumps(cache_data), encoding="utf-8")

    auth = MicrosoftAuthManager(token_path=token_path)
    mock_app = MagicMock()
    mock_app.get_accounts.return_value = [{"home_account_id": "hid"}]
    mock_app.acquire_token_silent.return_value = {"access_token": "from-silent"}

    with patch("pittqlab_utils.tools.ms_auth.msal.PublicClientApplication") as mock_pca:
        mock_pca.return_value = mock_app
        token = auth.get_token()

    assert token == "from-silent"
    mock_pca.assert_called_once()
    assert mock_pca.call_args.kwargs["client_credential"] == "secret"
    mock_app.acquire_token_silent.assert_called_once()
    mock_app.initiate_device_flow.assert_not_called()
