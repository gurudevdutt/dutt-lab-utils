"""
Ollama async backend: /api/chat and /api/tags, shared httpx client, context manager.
"""

from __future__ import annotations

import json
from typing import Any, Optional

import httpx

from .protocol import (
    ClassifyResult,
    GenerateResult,
    LLMBackend,
    parse_classify_response,
)

OLLAMA_CHAT_URL = "/api/chat"
OLLAMA_TAGS_URL = "/api/tags"


def _classify_system_prompt(labels: list[str]) -> str:
    labels_str = ", ".join(repr(l) for l in labels)
    return (
        "You are a classifier. Reply with ONLY valid JSON, no other text. "
        "Use exactly this shape: {\"label\": \"<one of the labels>\", \"confidence\": <number 0.0 to 1.0>}. "
        f"Allowed labels: {labels_str}. Choose one label and a confidence score."
    )


class OllamaBackend:
    """Async LLM backend for Ollama (local). Uses shared httpx.AsyncClient; supports async context manager."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.2",
        timeout: float = 60.0,
    ):
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
        self._is_available: bool = False

    @property
    def name(self) -> str:
        return f"ollama/{self._model}"

    @property
    def is_available(self) -> bool:
        return self._is_available

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self._base_url, timeout=self._timeout)
        return self._client

    async def __aenter__(self) -> OllamaBackend:
        self._client = httpx.AsyncClient(base_url=self._base_url, timeout=self._timeout)
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def ping(self) -> None:
        try:
            client = self._get_client()
            r = await client.get(OLLAMA_TAGS_URL)
            self._is_available = r.status_code == 200
        except Exception:
            self._is_available = False

    async def classify(
        self,
        text: str,
        labels: Sequence[str],
        *,
        system_prompt: Optional[str] = None,
        max_tokens: int = 256,
    ) -> ClassifyResult:
        labels_list = list(labels)
        system = system_prompt if system_prompt else _classify_system_prompt(labels_list)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": text},
        ]
        payload = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "options": {"num_predict": max_tokens},
        }
        client = self._get_client()
        r = await client.post(OLLAMA_CHAT_URL, json=payload)
        r.raise_for_status()
        data = r.json()
        raw_message = data.get("message", {})
        response_text = raw_message.get("content", "") or ""
        label, confidence = parse_classify_response(response_text, labels_list)
        return ClassifyResult(
            label=label,
            confidence=confidence,
            raw=data,
            backend=self.name,
        )

    async def generate(
        self,
        messages: Sequence[dict],
        *,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
    ) -> GenerateResult:
        msgs: list[dict] = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if isinstance(content, str):
                msgs.append({"role": role, "content": content})
            else:
                msgs.append({"role": role, "content": str(content)})
        payload = {
            "model": self._model,
            "messages": msgs,
            "stream": False,
            "options": {"num_predict": max_tokens},
        }
        client = self._get_client()
        r = await client.post(OLLAMA_CHAT_URL, json=payload)
        r.raise_for_status()
        data = r.json()
        raw_message = data.get("message", {})
        text = raw_message.get("content", "") or ""
        return GenerateResult(
            text=text,
            input_tokens=0,
            finish_reason="stop",
            raw=data,
            backend=self.name,
        )
