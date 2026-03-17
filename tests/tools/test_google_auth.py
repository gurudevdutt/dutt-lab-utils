"""Tests for GoogleAuthManager."""

import pytest
from pathlib import Path

from pittqlab_utils.tools.google_auth import GoogleAuthManager, GoogleAuthError


def test_get_credentials_raises_when_file_missing(tmp_path):
    """Test get_credentials raises clear error when credentials file is missing."""
    auth = GoogleAuthManager(
        credentials_path=tmp_path / "nonexistent.json",
        token_path=tmp_path / "token.json",
    )
    with pytest.raises(GoogleAuthError) as exc_info:
        auth.get_credentials()
    assert "credentials" in str(exc_info.value).lower()
    assert "Setup" in str(exc_info.value) or "google" in str(exc_info.value).lower()
