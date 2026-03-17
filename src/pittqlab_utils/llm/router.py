"""
ClassifierRouter: tries backends in order; confidence threshold and cascade for classify.
"""

from __future__ import annotations

import asyncio
from typing import Optional, Sequence

from .protocol import ClassifyResult, GenerateResult, LLMBackend


class ClassifierRouter:
    """Routes classify/generate over a list of backends. Classify cascades until confidence >= threshold."""

    def __init__(
        self,
        backends: list[LLMBackend],
        confidence_threshold: float = 0.75,
    ):
        self._backends = list(backends)
        self._confidence_threshold = confidence_threshold

    async def ping_all(self) -> None:
        """Ping all backends concurrently."""
        await asyncio.gather(*[b.ping() for b in self._backends])

    async def classify(
        self,
        text: str,
        labels: Sequence[str],
        *,
        system_prompt: Optional[str] = None,
        max_tokens: int = 256,
    ) -> ClassifyResult:
        """Try backends in order; skip unavailable. Return first with confidence >= threshold, else best result."""
        best: Optional[ClassifyResult] = None
        for backend in self._backends:
            if not backend.is_available:
                continue
            try:
                result = await backend.classify(
                    text,
                    labels,
                    system_prompt=system_prompt,
                    max_tokens=max_tokens,
                )
            except Exception:
                continue
            if result.confidence >= self._confidence_threshold:
                return result
            if best is None or result.confidence > best.confidence:
                best = result
        if best is not None:
            return best
        raise RuntimeError("All backends failed or unavailable for classify")

    async def generate(
        self,
        messages: Sequence[dict],
        *,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
    ) -> GenerateResult:
        """First available backend that succeeds wins; no cascade."""
        last_error: Optional[Exception] = None
        for backend in self._backends:
            if not backend.is_available:
                continue
            try:
                return await backend.generate(
                    messages,
                    system_prompt=system_prompt,
                    max_tokens=max_tokens,
                )
            except Exception as e:
                last_error = e
                continue
        raise RuntimeError("All backends failed or unavailable for generate") from last_error
