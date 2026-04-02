"""
Microsoft OAuth 2.0 (MSAL) for Microsoft Graph: Mail and Calendar.
Token persisted at ~/.menakai/credentials/ms_token.json
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import msal

# Microsoft Graph delegated scopes (v2 format)
SCOPES: List[str] = [
    "https://graph.microsoft.com/Mail.Read",
    "https://graph.microsoft.com/Mail.ReadWrite",
    "https://graph.microsoft.com/Calendars.Read",
    "https://graph.microsoft.com/Calendars.ReadWrite",
]

DEFAULT_CREDENTIALS_DIR = Path.home() / ".menakai" / "credentials"
TOKEN_FILE = DEFAULT_CREDENTIALS_DIR / "ms_token.json"

ENV_CLIENT_ID = "MICROSOFT_CLIENT_ID"
ENV_CLIENT_SECRET = "MICROSOFT_CLIENT_SECRET"
ENV_TENANT_ID = "MICROSOFT_TENANT_ID"


class MicrosoftAuthError(Exception):
    """Raised when Microsoft credentials are missing or authentication fails."""


class MicrosoftAuthManager:
    """MSAL-based Microsoft Graph authentication with device code on first sign-in."""

    def __init__(
        self,
        token_path: Path = TOKEN_FILE,
        client_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        client_secret: Optional[str] = None,
    ):
        self._token_path = Path(token_path)
        self._client_id = client_id
        self._tenant_id = tenant_id
        self._client_secret = client_secret
        self._cache = msal.SerializableTokenCache()
        self._load_cache_from_disk()
        self._app: Optional[msal.PublicClientApplication] = None

    def _load_cache_from_disk(self) -> None:
        if self._token_path.exists():
            try:
                self._cache.deserialize(self._token_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass

    def _persist_cache(self) -> None:
        if self._cache.has_state_changed:
            self._token_path.parent.mkdir(parents=True, exist_ok=True)
            self._token_path.write_text(self._cache.serialize(), encoding="utf-8")

    def _ensure_config(self) -> tuple[str, str, str]:
        cid = self._client_id or os.environ.get(ENV_CLIENT_ID, "").strip()
        secret = self._client_secret or os.environ.get(ENV_CLIENT_SECRET, "").strip()
        tid = self._tenant_id or os.environ.get(ENV_TENANT_ID, "").strip()
        if not cid or not secret or not tid:
            raise MicrosoftAuthError(
                "Microsoft OAuth configuration incomplete. Set these environment variables:\n"
                f"  {ENV_CLIENT_ID} — Azure app (client) ID\n"
                f"  {ENV_CLIENT_SECRET} — client secret (required; passed as "
                "client_credential for tenants that require it with device flow)\n"
                f"  {ENV_TENANT_ID} — directory (tenant) ID\n"
                "Register a public client / native app in Azure Portal and enable "
                "delegated Graph permissions for Mail and Calendars."
            )
        return cid, secret, tid

    def _get_app(self) -> msal.PublicClientApplication:
        if self._app is not None:
            return self._app
        client_id, secret, tenant_id = self._ensure_config()
        authority = f"https://login.microsoftonline.com/{tenant_id}"
        self._app = msal.PublicClientApplication(
            client_id,
            authority=authority,
            client_credential=secret,
            token_cache=self._cache,
        )
        return self._app

    def _result_to_token(self, result: Optional[Dict[str, Any]], context: str) -> str:
        if not result:
            raise MicrosoftAuthError(f"Microsoft authentication failed ({context}): empty result.")
        if "access_token" in result:
            return str(result["access_token"])
        err = result.get("error", "unknown_error")
        desc = result.get("error_description", "")
        raise MicrosoftAuthError(
            f"Microsoft authentication failed ({context}): {err}. {desc}".strip()
        )

    def get_token(self) -> str:
        """Return a valid access token, refreshing silently from cache when possible.

        On first run (no cached token): runs device code flow — prints verification
        URL and code, then blocks until the user completes sign-in in the browser.

        Raises:
            MicrosoftAuthError: If env vars are missing or any auth step fails.
        """
        self._ensure_config()
        app = self._get_app()
        accounts = app.get_accounts()
        result = app.acquire_token_silent(SCOPES, account=accounts[0] if accounts else None)
        if result and "access_token" in result:
            self._persist_cache()
            return str(result["access_token"])

        flow = app.initiate_device_flow(scopes=SCOPES)
        if "user_code" not in flow:
            raise MicrosoftAuthError(
                "Could not start device code flow. "
                f"Response: {flow.get('error_description', flow)}"
            )
        print(flow["message"])
        result = app.acquire_token_by_device_flow(flow)
        self._persist_cache()
        return self._result_to_token(result, "device flow")
