"""
pittqlab_utils.llm.pittai
~~~~~~~~~~~~~~~~~~~~~~~~~
Reusable client for Pitt AI Connect (Portkey gateway).

Usage:
    from pittqlab_utils.llm.pittai import PittAIClient, PittAIModels

    client = PittAIClient()  # reads PITTAI_API_KEY from environment
    response = client.chat("Summarize this abstract.", system="You are a physicist.")
    print(response.text)

    # Multimodal
    response = client.chat("What is shown here?", images=["diagram.png"])
    print(response.text)
"""

from __future__ import annotations

import base64
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

class PittAIModels:
    """Canonical model strings for Pitt AI Connect.

    Use these constants instead of hardcoding model strings in your project.
    Update here when Pitt AI Connect adds or changes model versions.
    """

    # Anthropic Claude
    CLAUDE_SONNET    = "@pitt-ai-connect/us.anthropic.claude-sonnet-4-5-20250929-v1:0"
    CLAUDE_HAIKU     = "@pitt-ai-connect/us.anthropic.claude-haiku-4-5-20251001-v1:0"
    CLAUDE_OPUS      = "@pitt-ai-connect/us.anthropic.claude-opus-4-6-v1"
    CLAUDE_OPUS_4_5 = "@pitt-ai-connect/us.anthropic.claude-opus-4-5-20251101-v1:0"
    CLAUDE_SONNET_4_6 = "@pitt-ai-connect/us.anthropic.claude-sonnet-4-6"

    # Google Gemini
    GEMINI_FLASH          = "@pitt-ai-connect-google-vertex/gemini-2.5-flash"
    GEMINI_FLASH_LITE     = "@pitt-ai-connect-google-vertex/gemini-2.5-flash-lite"
    GEMINI_PRO            = "@pitt-ai-connect-google-vertex/gemini-2.5-pro"

    # OpenAI GPT
    GPT_5p1               = "@pitt-ai-connect-azure-foundry/gpt-5.1"
    GPT_5p2          = "@pitt-ai-connect-azure-foundry/gpt-5.2"
    GPT_5p4          = "@pitt-ai-connect-azure-foundry/gpt-5.4"

    # Recommended tiers for common tasks
    CHEAP    = GEMINI_FLASH_LITE   # bulk extraction, first-pass triage
    BALANCED = GEMINI_FLASH        # general scoring, summarization
    QUALITY  = CLAUDE_SONNET       # nuanced judgment, final scoring
    BEST     = CLAUDE_OPUS         # edge cases, highest stakes


# ---------------------------------------------------------------------------
# Response dataclass
# ---------------------------------------------------------------------------

@dataclass
class PittAIResponse:
    """Structured response from a Pitt AI Connect chat call."""
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    raw: dict = field(default_factory=dict)

    @property
    def usage(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


# ---------------------------------------------------------------------------
# Per-provider API keys (Option A: one key per provider for billing)
# ---------------------------------------------------------------------------

_PROVIDER_ENV_KEYS = {
    "anthropic": "PITTAI_API_KEY_ANTHROPIC",
    "google": "PITTAI_API_KEY_GOOGLE",
    "openai": "PITTAI_API_KEY_OPENAI",
}


def _provider_from_model(model: str) -> str | None:
    """Return provider name from model string, or None if unrecognized."""
    m = model.lower()
    if "anthropic" in m:
        return "anthropic"
    if "google-vertex" in m or "google" in m:
        return "google"
    if "azure-foundry" in m or "openai" in m:
        return "openai"
    return None


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class PittAIClient:
    """Client for Pitt AI Connect via Portkey gateway.

    Supports one API key per provider for separate billing. Optional env vars:
    PITTAI_API_KEY_ANTHROPIC, PITTAI_API_KEY_GOOGLE, PITTAI_API_KEY_OPENAI.
    The key for each request is chosen from the model string; if no provider
    key is set, PITTAI_API_KEY (or api_key) is used.

    Args:
        api_key: Default API key. Defaults to PITTAI_API_KEY env var.
        default_model: Model string when none is specified per-call.
        max_retries: Retry attempts on transient errors.
        retry_delay: Base delay (seconds) between retries (exponential backoff).
        timeout: Request timeout in seconds.
    """

    BASE_URL = "https://api.portkey.ai/v1/chat/completions"

    def __init__(
        self,
        api_key: str | None = None,
        default_model: str = PittAIModels.BALANCED,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        timeout: int = 60,
    ):
        self.api_key = api_key or os.environ.get("PITTAI_API_KEY")
        self._provider_keys: dict[str, str] = {}
        for provider, env_key in _PROVIDER_ENV_KEYS.items():
            val = os.environ.get(env_key)
            if val:
                self._provider_keys[provider] = val
        if not self.api_key and not self._provider_keys:
            raise ValueError(
                "Pitt AI Connect API key not found. "
                "Set PITTAI_API_KEY in your .env or pass api_key= explicitly."
            )
        if not self.api_key:
            self.api_key = next(iter(self._provider_keys.values()))
        self.default_model = default_model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout

    def _get_api_key_for_model(self, model: str) -> str:
        """Return the API key to use for this model (provider-specific or default)."""
        provider = _provider_from_model(model)
        if provider and provider in self._provider_keys:
            return self._provider_keys[provider]
        return self.api_key

    def get_api_key_source_for_model(self, model: str) -> str:
        """Return the env var name used for this model (e.g. PITTAI_API_KEY_GOOGLE). For logging only."""
        provider = _provider_from_model(model)
        if provider and provider in self._provider_keys:
            return _PROVIDER_ENV_KEYS[provider]
        return "PITTAI_API_KEY"

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def chat(
        self,
        prompt: str,
        *,
        system: str = "",
        model: str | None = None,
        max_tokens: int = 1024,
        images: list[Union[str, Path]] | None = None,
        json_mode: bool = False,
    ) -> PittAIResponse:
        """Send a chat message and return a structured response.

        Args:
            prompt: User message text.
            system: Optional system prompt.
            model: Model string override. Defaults to self.default_model.
            max_tokens: Maximum tokens in the response.
            images: Optional list of image file paths for multimodal calls.
            json_mode: If True, instructs the model to respond with valid JSON only.
                       Appends a JSON instruction to the system prompt.

        Returns:
            PittAIResponse with .text, .model, and .usage attributes.

        Raises:
            ValueError: If the API returns an error response.
            requests.Timeout: If the request exceeds self.timeout.
        """
        model = model or self.default_model
        messages = self._build_messages(prompt, system=system, images=images, json_mode=json_mode)
        payload = {"model": model, "messages": messages, "max_tokens": max_tokens}

        raw = self._post_with_retry(payload)
        return self._parse_response(raw, model)

    def chat_json(
        self,
        prompt: str,
        *,
        system: str = "",
        model: str | None = None,
        max_tokens: int = 1024,
    ) -> dict:
        """Convenience wrapper: send a chat message and parse the response as JSON.

        The model is instructed to return only valid JSON with no preamble.
        The response text is parsed and returned as a dict.

        Raises:
            ValueError: If the response cannot be parsed as JSON.
        """
        import json
        response = self.chat(
            prompt, system=system, model=model, max_tokens=max_tokens, json_mode=True
        )
        text = response.text.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Model response could not be parsed as JSON.\n"
                f"Response was: {response.text[:200]}\n"
                f"Error: {e}"
            )
    def chat_with_history(
    self,
    messages: List[Dict[str, str]],
    *,
    model: Optional[str] = None,
    max_tokens: int = 1024,
    json_mode: bool = False,
    ) -> PittAIResponse:
        """Send a full conversation history and return the next response.

        Args:
            messages: List of {"role": "user"|"assistant"|"system", "content": str}
                Build and maintain this list in the calling application.
            model: Model string override. Defaults to self.default_model.
            max_tokens: Maximum tokens in the response.
            json_mode: If True, appends JSON instruction to the last system message
                or inserts one if none exists.

        Returns:
            PittAIResponse — append {"role": "assistant", "content": response.text}
            to your history list to continue the conversation.
        """
        model = model or self.default_model
        msgs = list(messages)  # don't mutate caller's list

        if json_mode:
            json_instruction = (
                "Respond ONLY with valid JSON. No preamble, no explanation, "
                "no markdown code fences. The response must be parseable by json.loads()."
            )
            # Append to existing system message or insert one
            if msgs and msgs[0]["role"] == "system":
                msgs[0] = {
                    "role": "system",
                    "content": msgs[0]["content"] + "\n\n" + json_instruction,
                }
            else:
                msgs.insert(0, {"role": "system", "content": json_instruction})

        payload = {"model": model, "messages": msgs, "max_tokens": max_tokens}
        raw = self._post_with_retry(payload)
        return self._parse_response(raw, model)
    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_messages(
        self,
        prompt: str,
        system: str,
        images: list[Union[str, Path]] | None,
        json_mode: bool,
    ) -> list[dict]:
        messages = []

        # System message
        sys_text = system
        if json_mode:
            json_instruction = (
                "Respond ONLY with valid JSON. No preamble, no explanation, "
                "no markdown code fences. The response must be parseable by json.loads()."
            )
            sys_text = f"{sys_text}\n\n{json_instruction}".strip() if sys_text else json_instruction
        if sys_text:
            messages.append({"role": "system", "content": sys_text})

        # User message — text only or multimodal
        if images:
            content = [{"type": "text", "text": prompt}]
            for img_path in images:
                content.append(self._encode_image(img_path))
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": prompt})

        return messages

    def _encode_image(self, image_path: Union[str, Path]) -> dict:
        """Encode a local image file as a base64 data URL content block."""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")

        suffix = path.suffix.lower()
        media_type_map = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        media_type = media_type_map.get(suffix, "image/png")

        with open(path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")

        return {
            "type": "image_url",
            "image_url": {"url": f"data:{media_type};base64,{encoded}"},
        }

    def _post_with_retry(self, payload: dict) -> dict:
        """POST to the Portkey endpoint with exponential backoff retry."""
        model = payload.get("model", "")
        # Azure/OpenAI backend expects max_completion_tokens instead of max_tokens
        if ("azure-foundry" in model.lower() or "openai" in model.lower()) and "max_tokens" in payload:
            payload["max_completion_tokens"] = payload.pop("max_tokens")
        api_key = self._get_api_key_for_model(model)
        headers = {
            "Content-Type": "application/json",
            "x-portkey-api-key": api_key,
        }
        last_exc: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.BASE_URL,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                if response.status_code == 200:
                    return response.json()

                # Non-retryable client errors
                if response.status_code in (400, 401, 403, 404):
                    raise ValueError(
                        f"Pitt AI Connect API error {response.status_code}: {response.text}"
                    )

                # Retryable server errors (429, 500, 502, 503)
                logger.warning(
                    "Pitt AI Connect request failed (attempt %d/%d): HTTP %d — %s",
                    attempt + 1, self.max_retries, response.status_code, response.text[:200],
                )
                last_exc = ValueError(f"HTTP {response.status_code}: {response.text}")

            except requests.Timeout as e:
                logger.warning("Request timed out (attempt %d/%d)", attempt + 1, self.max_retries)
                last_exc = e
            except requests.ConnectionError as e:
                logger.warning("Connection error (attempt %d/%d): %s", attempt + 1, self.max_retries, e)
                last_exc = e

            if attempt < self.max_retries - 1:
                delay = self.retry_delay * (2 ** attempt)
                logger.info("Retrying in %.1fs...", delay)
                time.sleep(delay)

        raise RuntimeError(
            f"Pitt AI Connect request failed after {self.max_retries} attempts."
        ) from last_exc

    def _parse_response(self, raw: dict, model: str) -> PittAIResponse:
        """Parse the raw API response into a PittAIResponse."""
        try:
            text = raw["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise ValueError(f"Unexpected API response structure: {raw}") from e

        usage = raw.get("usage", {})
        return PittAIResponse(
            text=text,
            model=model,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            raw=raw,
        )
