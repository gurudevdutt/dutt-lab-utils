"""
Tests for pittqlab_utils.llm.pittai

Run:
    pytest tests/test_pittai.py -v
    pytest tests/test_pittai.py -v -m integration   # real API calls
"""

import base64
import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pittqlab_utils.llm.pittai import PittAIClient, PittAIModels, PittAIResponse


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    """Client with a dummy API key for unit tests."""
    return PittAIClient(api_key="test-key-dummy")


def make_mock_response(text: str, model: str = "test-model", status_code: int = 200) -> MagicMock:
    """Build a mock requests.Response object."""
    mock = MagicMock()
    mock.status_code = status_code
    mock.json.return_value = {
        "choices": [{"message": {"content": text}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }
    mock.text = text
    return mock


# ---------------------------------------------------------------------------
# Tier 1: PittAIModels constants
# ---------------------------------------------------------------------------

class TestPittAIModels:
    def test_model_strings_are_strings(self):
        for attr in ("CLAUDE_SONNET", "CLAUDE_HAIKU", "CLAUDE_OPUS",
                     "GEMINI_FLASH", "GEMINI_FLASH_LITE", "GEMINI_PRO",
                     "GPT_5p1", "GPT_5p2", "GPT_5p4"):
            val = getattr(PittAIModels, attr)
            assert isinstance(val, str)
            assert val.startswith("@pitt-ai-connect"), f"{attr} should be a Pitt AI Connect model string"

    def test_tier_aliases_point_to_valid_models(self):
        assert PittAIModels.CHEAP    == PittAIModels.GEMINI_FLASH_LITE
        assert PittAIModels.BALANCED == PittAIModels.GEMINI_FLASH
        assert PittAIModels.QUALITY  == PittAIModels.CLAUDE_SONNET
        assert PittAIModels.BEST     == PittAIModels.CLAUDE_OPUS


# ---------------------------------------------------------------------------
# Tier 1: Client initialization
# ---------------------------------------------------------------------------

class TestClientInit:
    def test_explicit_api_key(self):
        c = PittAIClient(api_key="my-key")
        assert c.api_key == "my-key"

    def test_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("PITTAI_API_KEY", "env-key")
        c = PittAIClient()
        assert c.api_key == "env-key"

    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("PITTAI_API_KEY", raising=False)
        for k in ("PITTAI_API_KEY_ANTHROPIC", "PITTAI_API_KEY_GOOGLE", "PITTAI_API_KEY_OPENAI"):
            monkeypatch.delenv(k, raising=False)
        with pytest.raises(ValueError, match="PITTAI_API_KEY"):
            PittAIClient()

    def test_init_ok_with_only_provider_keys(self, monkeypatch):
        monkeypatch.delenv("PITTAI_API_KEY", raising=False)
        monkeypatch.setenv("PITTAI_API_KEY_ANTHROPIC", "anthropic-key")
        c = PittAIClient()
        assert c.api_key == "anthropic-key"

    def test_default_model_is_balanced(self):
        c = PittAIClient(api_key="k")
        assert c.default_model == PittAIModels.BALANCED

    def test_custom_default_model(self):
        c = PittAIClient(api_key="k", default_model=PittAIModels.QUALITY)
        assert c.default_model == PittAIModels.QUALITY


# ---------------------------------------------------------------------------
# Tier 2: Message building
# ---------------------------------------------------------------------------

class TestMessageBuilding:
    def test_simple_user_message(self, client):
        msgs = client._build_messages("Hello", system="", images=None, json_mode=False)
        assert msgs == [{"role": "user", "content": "Hello"}]

    def test_system_message_prepended(self, client):
        msgs = client._build_messages("Hello", system="Be brief", images=None, json_mode=False)
        assert msgs[0] == {"role": "system", "content": "Be brief"}
        assert msgs[1]["role"] == "user"

    def test_json_mode_appends_instruction(self, client):
        msgs = client._build_messages("Score this", system="", images=None, json_mode=True)
        sys_msg = msgs[0]
        assert sys_msg["role"] == "system"
        assert "json.loads()" in sys_msg["content"]

    def test_json_mode_merges_with_existing_system(self, client):
        msgs = client._build_messages("Score", system="You are an expert", images=None, json_mode=True)
        assert "You are an expert" in msgs[0]["content"]
        assert "JSON" in msgs[0]["content"]

    def test_multimodal_message_structure(self, client, tmp_path):
        img = tmp_path / "test.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)  # minimal PNG header
        msgs = client._build_messages("Describe", system="", images=[img], json_mode=False)
        content = msgs[0]["content"]
        assert isinstance(content, list)
        assert content[0] == {"type": "text", "text": "Describe"}
        assert content[1]["type"] == "image_url"
        assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


# ---------------------------------------------------------------------------
# Tier 2: Image encoding
# ---------------------------------------------------------------------------

class TestImageEncoding:
    def test_png_encoding(self, client, tmp_path):
        img = tmp_path / "photo.png"
        img.write_bytes(b"fakeimagedata")
        block = client._encode_image(img)
        assert block["type"] == "image_url"
        assert "image/png" in block["image_url"]["url"]
        decoded = base64.b64decode(block["image_url"]["url"].split(",")[1])
        assert decoded == b"fakeimagedata"

    def test_jpeg_encoding(self, client, tmp_path):
        img = tmp_path / "photo.jpg"
        img.write_bytes(b"fakejpeg")
        block = client._encode_image(img)
        assert "image/jpeg" in block["image_url"]["url"]

    def test_missing_file_raises(self, client):
        with pytest.raises(FileNotFoundError):
            client._encode_image("/nonexistent/path/image.png")


# ---------------------------------------------------------------------------
# Tier 2: Chat and response parsing
# ---------------------------------------------------------------------------

class TestChat:
    @patch("pittqlab_utils.llm.pittai.requests.post")
    def test_chat_returns_response(self, mock_post, client):
        mock_post.return_value = make_mock_response("Great answer")
        resp = client.chat("What is NV?")
        assert isinstance(resp, PittAIResponse)
        assert resp.text == "Great answer"
        assert resp.total_tokens == 30

    @patch("pittqlab_utils.llm.pittai.requests.post")
    def test_chat_uses_default_model(self, mock_post, client):
        mock_post.return_value = make_mock_response("ok")
        client.chat("Hello")
        payload = mock_post.call_args.kwargs["json"]
        assert payload["model"] == PittAIModels.BALANCED

    @patch("pittqlab_utils.llm.pittai.requests.post")
    def test_chat_model_override(self, mock_post, client):
        mock_post.return_value = make_mock_response("ok")
        client.chat("Hello", model=PittAIModels.QUALITY)
        payload = mock_post.call_args.kwargs["json"]
        assert payload["model"] == PittAIModels.QUALITY

    @patch("pittqlab_utils.llm.pittai.requests.post")
    def test_correct_headers_sent(self, mock_post, client):
        mock_post.return_value = make_mock_response("ok")
        client.chat("Hello")
        headers = mock_post.call_args.kwargs["headers"]
        assert headers["x-portkey-api-key"] == "test-key-dummy"
        assert headers["Content-Type"] == "application/json"

    @patch("pittqlab_utils.llm.pittai.requests.post")
    def test_provider_key_used_for_anthropic_model(self, mock_post, monkeypatch):
        monkeypatch.setenv("PITTAI_API_KEY", "default-key")
        monkeypatch.setenv("PITTAI_API_KEY_ANTHROPIC", "anthropic-key")
        c = PittAIClient()
        mock_post.return_value = make_mock_response("ok")
        c.chat("Hi", model=PittAIModels.CLAUDE_HAIKU)
        assert mock_post.call_args.kwargs["headers"]["x-portkey-api-key"] == "anthropic-key"

    @patch("pittqlab_utils.llm.pittai.requests.post")
    def test_provider_key_used_for_google_model(self, mock_post, monkeypatch):
        monkeypatch.setenv("PITTAI_API_KEY", "default-key")
        monkeypatch.setenv("PITTAI_API_KEY_GOOGLE", "google-key")
        c = PittAIClient()
        mock_post.return_value = make_mock_response("ok")
        c.chat("Hi", model=PittAIModels.GEMINI_FLASH)
        assert mock_post.call_args.kwargs["headers"]["x-portkey-api-key"] == "google-key"

    @patch("pittqlab_utils.llm.pittai.requests.post")
    def test_default_key_used_when_no_provider_key_set(self, mock_post, monkeypatch):
        monkeypatch.setenv("PITTAI_API_KEY", "default-key")
        monkeypatch.delenv("PITTAI_API_KEY_ANTHROPIC", raising=False)
        monkeypatch.delenv("PITTAI_API_KEY_GOOGLE", raising=False)
        monkeypatch.delenv("PITTAI_API_KEY_OPENAI", raising=False)
        c = PittAIClient()
        mock_post.return_value = make_mock_response("ok")
        c.chat("Hi", model=PittAIModels.GEMINI_FLASH)
        assert mock_post.call_args.kwargs["headers"]["x-portkey-api-key"] == "default-key"

    @patch("pittqlab_utils.llm.pittai.requests.post")
    def test_chat_json_parses_valid_json(self, mock_post, client):
        mock_post.return_value = make_mock_response('{"score": 5, "justification": "strong"}')
        result = client.chat_json("Score this CV")
        assert result == {"score": 5, "justification": "strong"}

    @patch("pittqlab_utils.llm.pittai.requests.post")
    def test_chat_json_strips_code_fences(self, mock_post, client):
        mock_post.return_value = make_mock_response('```json\n{"score": 3}\n```')
        result = client.chat_json("Score")
        assert result["score"] == 3

    @patch("pittqlab_utils.llm.pittai.requests.post")
    def test_chat_json_raises_on_invalid_json(self, mock_post, client):
        mock_post.return_value = make_mock_response("Sorry, I cannot score this.")
        with pytest.raises(ValueError, match="parsed as JSON"):
            client.chat_json("Score this")


# ---------------------------------------------------------------------------
# Tier 2: Retry and error handling
# ---------------------------------------------------------------------------

class TestRetryAndErrors:
    @patch("pittqlab_utils.llm.pittai.requests.post")
    @patch("pittqlab_utils.llm.pittai.time.sleep")
    def test_retries_on_500(self, mock_sleep, mock_post, client):
        fail = MagicMock(status_code=500, text="Server error")
        success = make_mock_response("ok")
        mock_post.side_effect = [fail, success]
        resp = client.chat("Hello")
        assert resp.text == "ok"
        assert mock_post.call_count == 2

    @patch("pittqlab_utils.llm.pittai.requests.post")
    @patch("pittqlab_utils.llm.pittai.time.sleep")
    def test_raises_after_max_retries(self, mock_sleep, mock_post, client):
        fail = MagicMock(status_code=503, text="Unavailable")
        mock_post.return_value = fail
        with pytest.raises(RuntimeError, match="3 attempts"):
            client.chat("Hello")

    @patch("pittqlab_utils.llm.pittai.requests.post")
    def test_401_raises_immediately_no_retry(self, mock_post, client):
        mock_post.return_value = MagicMock(status_code=401, text="Unauthorized")
        with pytest.raises(ValueError, match="401"):
            client.chat("Hello")
        assert mock_post.call_count == 1

    @patch("pittqlab_utils.llm.pittai.requests.post")
    def test_malformed_response_raises(self, mock_post, client):
        mock_post.return_value = MagicMock(status_code=200)
        mock_post.return_value.json.return_value = {"unexpected": "structure"}
        with pytest.raises(ValueError, match="response structure"):
            client.chat("Hello")


# ---------------------------------------------------------------------------
# Tier 3: Integration tests (real API, run manually)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestIntegration:
    """Real API calls — require PITTAI_API_KEY in environment.
    Run with: pytest -m integration
    """

    def test_basic_chat(self):
        client = PittAIClient()
        resp = client.chat("Say the word 'photon' and nothing else.")
        assert "photon" in resp.text.lower()
        assert resp.total_tokens > 0

    def test_json_response(self):
        client = PittAIClient()
        result = client.chat_json(
            'Return JSON with keys "element" (string) and "atomic_number" (int) for carbon.'
        )
        assert result["element"].lower() == "carbon"
        assert result["atomic_number"] == 6

    def test_cheap_model(self):
        client = PittAIClient(default_model=PittAIModels.CHEAP)
        resp = client.chat("What is quantum sensing in one sentence?")
        assert len(resp.text) > 10

    def test_gemini_flash(self):
        """Smoke test: Gemini Flash returns a valid response."""
        client = PittAIClient()
        resp = client.chat(
            "In one short sentence, what is quantum sensing?",
            model=PittAIModels.GEMINI_FLASH,
        )
        assert len(resp.text) > 10
        assert resp.total_tokens > 0
        assert "quantum" in resp.text.lower() or "sensing" in resp.text.lower()

    def test_multimodal_requires_image(self, tmp_path):
        """Smoke test — just checks the request is formed correctly."""
        client = PittAIClient()
        img = tmp_path / "blank.png"
        # 1x1 white PNG
        img.write_bytes(
            base64.b64decode(
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8"
                "z8BQDwADhQGAWjR9awAAAABJRU5ErkJggg=="
            )
        )
        resp = client.chat("What color is this image?", images=[img])
        assert isinstance(resp.text, str)
