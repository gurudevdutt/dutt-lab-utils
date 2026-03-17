"""
Unified async LLM backend protocol and result types for intent classification.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable, Sequence


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ClassifyResult:
    """Result of an intent/label classification call."""
    label: str
    confidence: float
    raw: dict
    backend: str


@dataclass(frozen=True)
class GenerateResult:
    """Result of a generate/chat call."""
    text: str
    input_tokens: int
    finish_reason: str
    raw: dict
    backend: str


# ---------------------------------------------------------------------------
# Shared classify response parser
# ---------------------------------------------------------------------------

def parse_classify_response(raw_text: str, labels: Sequence[str]) -> tuple[str, float]:
    """
    Parse LLM response into (label, confidence). Used by Ollama and Portkey backends.
    Fallback: first label, 0.0 confidence. Confidence clamped to [0.0, 1.0].
    Unknown labels replaced with first label.
    """
    if not labels:
        return ("", 0.0)
    first_label = labels[0]
    text = raw_text.strip()

    # 1) Try direct JSON parse
    try:
        data = json.loads(text)
        label = _get_label_from_dict(data, labels, first_label)
        conf = _get_confidence_from_dict(data)
        return (label, _clamp_confidence(conf))
    except (json.JSONDecodeError, TypeError):
        pass

    # 2) Strip markdown code fences and retry
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        data = json.loads(text)
        label = _get_label_from_dict(data, labels, first_label)
        conf = _get_confidence_from_dict(data)
        return (label, _clamp_confidence(conf))
    except (json.JSONDecodeError, TypeError):
        pass

    # 3) Regex extraction
    label_match = re.search(r'"label"\s*:\s*"([^"]*)"', text, re.IGNORECASE)
    conf_match = re.search(r'"confidence"\s*:\s*([\d.]+)', text, re.IGNORECASE)
    if label_match and conf_match:
        raw_label = label_match.group(1).strip()
        label = raw_label if raw_label in labels else first_label
        try:
            conf = float(conf_match.group(1))
        except ValueError:
            conf = 0.0
        return (label, _clamp_confidence(conf))
    if label_match:
        raw_label = label_match.group(1).strip()
        label = raw_label if raw_label in labels else first_label
        return (label, 0.0)

    # 4) Fallback
    return (first_label, 0.0)


def _get_label_from_dict(data: dict, labels: Sequence[str], first_label: str) -> str:
    if not isinstance(data, dict):
        return first_label
    for key in ("label", "Label"):
        if key in data and isinstance(data[key], str):
            val = data[key].strip()
            return val if val in labels else first_label
    return first_label


def _get_confidence_from_dict(data: dict) -> float:
    if not isinstance(data, dict):
        return 0.0
    for key in ("confidence", "confidence_score", "conf"):
        if key in data:
            try:
                return float(data[key])
            except (TypeError, ValueError):
                pass
    return 0.0


def _clamp_confidence(c: float) -> float:
    return max(0.0, min(1.0, c))


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class LLMBackend(Protocol):
    """Async LLM backend for classify and generate. Implementations: OllamaBackend, PortkeyBackend."""

    @property
    def name(self) -> str: ...

    @property
    def is_available(self) -> bool: ...

    async def ping(self) -> None: ...

    async def classify(
        self,
        text: str,
        labels: Sequence[str],
        *,
        system_prompt: Optional[str] = None,
        max_tokens: int = 256,
    ) -> ClassifyResult: ...

    async def generate(
        self,
        messages: Sequence[dict],
        *,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
    ) -> GenerateResult: ...
