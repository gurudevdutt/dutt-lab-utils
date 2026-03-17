"""
Tests for OllamaBackend: ping, classify, parser resilience, HTTP errors, protocol compliance.
Uses respx to mock httpx; no live Ollama server required.
"""

import json
import pytest
import respx
import httpx

from pittqlab_utils.llm import OllamaBackend, LLMBackend, ClassifyResult


# ---------------------------------------------------------------------------
# Ping
# ---------------------------------------------------------------------------

@respx.mock
async def test_ping_success():
    respx.get("http://localhost:11434/api/tags").mock(return_value=httpx.Response(200, json={"models": []}))
    backend = OllamaBackend(base_url="http://localhost:11434")
    await backend.ping()
    assert backend.is_available is True


@respx.mock
async def test_ping_fail():
    respx.get("http://localhost:11434/api/tags").mock(return_value=httpx.Response(500))
    backend = OllamaBackend(base_url="http://localhost:11434")
    await backend.ping()
    assert backend.is_available is False


@respx.mock
async def test_ping_timeout():
    respx.get("http://localhost:11434/api/tags").mock(side_effect=httpx.TimeoutException(""))
    backend = OllamaBackend(base_url="http://localhost:11434")
    await backend.ping()
    assert backend.is_available is False


# ---------------------------------------------------------------------------
# Classify happy path
# ---------------------------------------------------------------------------

@respx.mock
async def test_classify_happy_path():
    respx.post("http://localhost:11434/api/chat").mock(
        return_value=httpx.Response(
            200,
            json={
                "message": {"role": "assistant", "content": '{"label": "A", "confidence": 0.9}'},
                "model": "llama3.2",
            },
        )
    )
    backend = OllamaBackend(base_url="http://localhost:11434")
    result = await backend.classify("hello", ["A", "B"])
    assert isinstance(result, ClassifyResult)
    assert result.label == "A"
    assert result.confidence == 0.9
    assert result.backend == "ollama/llama3.2"
    assert result.raw


# ---------------------------------------------------------------------------
# Parser resilience
# ---------------------------------------------------------------------------

@respx.mock
async def test_classify_strips_markdown_fences():
    respx.post("http://localhost:11434/api/chat").mock(
        return_value=httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": '```json\n{"label": "B", "confidence": 0.7}\n```',
                },
                "model": "llama3.2",
            },
        )
    )
    backend = OllamaBackend(base_url="http://localhost:11434")
    result = await backend.classify("x", ["A", "B"])
    assert result.label == "B"
    assert result.confidence == 0.7


@respx.mock
async def test_classify_unknown_label_fallback():
    respx.post("http://localhost:11434/api/chat").mock(
        return_value=httpx.Response(
            200,
            json={
                "message": {"role": "assistant", "content": '{"label": "Other", "confidence": 0.8}'},
                "model": "llama3.2",
            },
        )
    )
    backend = OllamaBackend(base_url="http://localhost:11434")
    result = await backend.classify("x", ["A", "B"])
    assert result.label == "A"
    assert result.confidence == 0.8


@respx.mock
async def test_classify_confidence_clamp():
    respx.post("http://localhost:11434/api/chat").mock(
        return_value=httpx.Response(
            200,
            json={
                "message": {"role": "assistant", "content": '{"label": "A", "confidence": 1.5}'},
                "model": "llama3.2",
            },
        )
    )
    backend = OllamaBackend(base_url="http://localhost:11434")
    result = await backend.classify("x", ["A"])
    assert result.confidence == 1.0

    respx.post("http://localhost:11434/api/chat").mock(
        return_value=httpx.Response(
            200,
            json={
                "message": {"role": "assistant", "content": '{"label": "A", "confidence": -0.2}'},
                "model": "llama3.2",
            },
        )
    )
    result2 = await backend.classify("x", ["A"])
    assert result2.confidence == 0.0


@respx.mock
async def test_classify_regex_fallback():
    respx.post("http://localhost:11434/api/chat").mock(
        return_value=httpx.Response(
            200,
            json={
                "message": {"role": "assistant", "content": 'Result: "label": "B", "confidence": 0.6'},
                "model": "llama3.2",
            },
        )
    )
    backend = OllamaBackend(base_url="http://localhost:11434")
    result = await backend.classify("x", ["A", "B"])
    assert result.label == "B"
    assert result.confidence == 0.6


@respx.mock
async def test_classify_safe_default():
    respx.post("http://localhost:11434/api/chat").mock(
        return_value=httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": "Sorry, I cannot classify."}, "model": "llama3.2"},
        )
    )
    backend = OllamaBackend(base_url="http://localhost:11434")
    result = await backend.classify("x", ["A", "B"])
    assert result.label == "A"
    assert result.confidence == 0.0


# ---------------------------------------------------------------------------
# HTTP error propagation
# ---------------------------------------------------------------------------

@respx.mock
async def test_classify_http_error_propagates():
    respx.post("http://localhost:11434/api/chat").mock(return_value=httpx.Response(500, text="Server error"))
    backend = OllamaBackend(base_url="http://localhost:11434")
    with pytest.raises(httpx.HTTPStatusError):
        await backend.classify("x", ["A"])


# ---------------------------------------------------------------------------
# System prompt injection
# ---------------------------------------------------------------------------

@respx.mock
async def test_classify_system_prompt_injection():
    respx.post("http://localhost:11434/api/chat").mock(
        return_value=httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": '{"label": "A", "confidence": 1.0}'}, "model": "x"},
        )
    )
    backend = OllamaBackend(base_url="http://localhost:11434")
    await backend.classify("hi", ["A"], system_prompt="You are strict.")
    req = respx.calls.last.request
    body = json.loads(req.content) if req.content else {}
    messages = body.get("messages", [])
    assert any(m.get("role") == "system" and "You are strict" in (m.get("content") or "") for m in messages)


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------

def test_ollama_protocol_compliance():
    backend = OllamaBackend(base_url="http://localhost:11434")
    assert isinstance(backend, LLMBackend)
    assert hasattr(backend, "name")
    assert hasattr(backend, "is_available")
    assert hasattr(backend, "ping")
    assert hasattr(backend, "classify")
    assert hasattr(backend, "generate")
    assert backend.name == "ollama/llama3.2"


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------

@respx.mock
async def test_ollama_context_manager():
    respx.get("http://localhost:11434/api/tags").mock(return_value=httpx.Response(200, json={}))
    async with OllamaBackend(base_url="http://localhost:11434") as backend:
        await backend.ping()
        assert backend.is_available is True
