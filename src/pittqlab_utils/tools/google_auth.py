"""
Google OAuth 2.0 manager for Gmail and Calendar APIs.
Credentials at ~/.menakai/credentials/google_credentials.json
Token stored/refreshed at ~/.menakai/credentials/token.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow


SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.compose",
    "https://www.googleapis.com/auth/calendar.readonly",
    "https://www.googleapis.com/auth/calendar.events",
]

DEFAULT_CREDENTIALS_DIR = Path.home() / ".menakai" / "credentials"
CREDENTIALS_FILE = DEFAULT_CREDENTIALS_DIR / "google_credentials.json"
TOKEN_FILE = DEFAULT_CREDENTIALS_DIR / "token.json"

# Headless-friendly: user copies URL, pastes code back
OOB_REDIRECT_URI = "urn:ietf:wg:oauth:2.0:oob"


class GoogleAuthError(Exception):
    """Raised when credentials are missing or auth fails."""


class GoogleAuthManager:
    """Manages Google OAuth 2.0 credentials for Gmail and Calendar."""

    def __init__(
        self,
        credentials_path: Path = CREDENTIALS_FILE,
        token_path: Path = TOKEN_FILE,
    ):
        self._credentials_path = Path(credentials_path)
        self._token_path = Path(token_path)

    def get_credentials(self) -> Credentials:
        """Return valid Credentials, refreshing silently if expired.

        On first run (no token.json): prints auth URL and waits for user
        to paste the auth code (headless-friendly).

        Raises:
            GoogleAuthError: If credentials file is missing.
        """
        if not self._credentials_path.exists():
            raise GoogleAuthError(
                f"Google credentials file not found: {self._credentials_path}\n"
                "Setup: Create a project in Google Cloud Console, enable Gmail API and "
                "Calendar API, create OAuth 2.0 credentials (Desktop app), download "
                "the JSON, and save it to ~/.menakai/credentials/google_credentials.json"
            )

        creds = None
        if self._token_path.exists():
            creds = self._load_token()

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                creds = self._run_headless_auth()
            self._save_token(creds)

        return creds

    def _load_token(self) -> Credentials:
        """Load credentials from token file."""
        with open(self._token_path, "r") as f:
            data = json.load(f)
        return Credentials.from_authorized_user_info(data, SCOPES)

    def _save_token(self, creds: Credentials) -> None:
        """Save credentials to token file."""
        self._token_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._token_path, "w") as f:
            f.write(creds.to_json())

    def _run_headless_auth(self) -> Credentials:
        """Run headless OAuth flow: print URL, wait for code, exchange."""
        flow = InstalledAppFlow.from_client_secrets_file(
            str(self._credentials_path),
            scopes=SCOPES,
            redirect_uri=OOB_REDIRECT_URI,
        )
        auth_url, _ = flow.authorization_url(
            access_type="offline",
            prompt="consent",
        )
        print("First-time Google auth required. Open this URL in a browser:")
        print(auth_url)
        print("\nAfter authorizing, paste the authorization code here:")
        code = input().strip()
        if not code:
            raise GoogleAuthError("No authorization code provided.")
        flow.fetch_token(code=code)
        return flow.credentials
