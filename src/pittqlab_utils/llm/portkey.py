"""
Portkey (Pitt AI Connect) async adapter: wraps PittAIClient via asyncio.to_thread.
"""

from __future__ import annotations

import asyncio
from typing import Optional, Sequence

from .protocol import (
    ClassifyResult,
    GenerateResult,
    LLMBackend,
    parse_classify_response,
)
from .pittai import PittAIClient, PittAIModels


def _classify_system_prompt(labels: list[str]) -> str:
    labels_str = ", ".join(repr(l) for l in labels)
    return (
        "You are a classifier. Reply with ONLY valid JSON, no other text. "
        "Use exactly this shape: {\"label\": \"<one of the labels>\", \"confidence\": <number 0.0 to 1.0>}. "
        f"Allowed labels: {labels_str}. Choose one label and a confidence score."
    )


def _model_to_short_name(model: str) -> str:
    """e.g. @pitt-ai-connect-google-vertex/gemini-2.5-flash-lite -> gemini-2.5-flash-lite"""
    if "/" in model:
        return model.rsplit("/", 1)[-1]
    return model


class PortkeyBackend:
    """Async adapter over PittAIClient. All sync calls run in thread pool via asyncio.to_thread."""

    def __init__(
        self,
        client: Optional[PittAIClient] = None,
        model: str = PittAIModels.CHEAP,
    ):
        self._client = client if client is not None else PittAIClient()
        self._model = model
        self._is_available: bool = False

    @property
    def name(self) -> str:
        return _model_to_short_name(self._model)

    @property
    def is_available(self) -> bool:
        return self._is_available

    async def ping(self) -> None:
        try:
            await asyncio.to_thread(
                self._client.chat,
                "ping",
                max_tokens=1,
                model=self._model,
            )
            self._is_available = True
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
        response = await asyncio.to_thread(
            self._client.chat,
            text,
            system=system,
            model=self._model,
            max_tokens=max_tokens,
            json_mode=True,
        )
        label, confidence = parse_classify_response(response.text, labels_list)
        return ClassifyResult(
            label=label,
            confidence=confidence,
            raw=response.raw,
            backend=self.name,
        )

    async def generate(
        self,
        messages: Sequence[dict],
        *,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
    ) -> GenerateResult:
        msgs: list[dict[str, str]] = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            msgs.append({"role": role, "content": str(content)})
        response = await asyncio.to_thread(
            self._client.chat_with_history,
            msgs,
            model=self._model,
            max_tokens=max_tokens,
        )
        return GenerateResult(
            text=response.text,
            input_tokens=response.prompt_tokens,
            finish_reason="stop",
            raw=response.raw,
            backend=self.name,
        )
