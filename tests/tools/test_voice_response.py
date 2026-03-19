"""Tests for VoiceResponse. Mocks gTTS, ElevenLabs, and bot.send_voice."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from pittqlab_utils.tools.voice_response import VoiceResponse


@pytest.mark.asyncio
async def test_gtts_backend_produces_ogg_output(tmp_path):
    """Test gTTS backend produces .ogg output."""
    output_path = tmp_path / "test_output.ogg"
    with patch("gtts.gTTS") as mock_gtts:
        with patch(
            "pittqlab_utils.tools.voice_response._mp3_to_ogg"
        ) as mock_convert:
            mock_instance = MagicMock()
            mock_gtts.return_value = mock_instance

            voice = VoiceResponse(backend="gtts", lang="en", slow=False)
            result = await voice.synthesize("Hello world", output_path)

            mock_instance.save.assert_called_once()
            mock_convert.assert_called_once()
            call_args = mock_convert.call_args[0]
            assert call_args[1] == output_path
            assert result == output_path
            mock_gtts.assert_called_once_with(
                text="Hello world", lang="en", slow=False
            )


@pytest.mark.asyncio
async def test_elevenlabs_backend_used_when_key_set(tmp_path):
    """Test ElevenLabs backend is used when ELEVENLABS_API_KEY is set."""
    output_path = tmp_path / "test_output.ogg"
    with patch.dict("os.environ", {"ELEVENLABS_API_KEY": "test-key"}):
        with patch(
            "elevenlabs.client.ElevenLabs"
        ) as mock_elevenlabs_cls:
            mock_client = MagicMock()
            mock_client.text_to_speech.convert.return_value = b"fake-ogg"
            mock_elevenlabs_cls.return_value = mock_client

            voice = VoiceResponse(backend="elevenlabs")
            result = await voice.synthesize("Hello world", output_path)

            mock_elevenlabs_cls.assert_called_once_with(api_key="test-key")
            mock_client.text_to_speech.convert.assert_called_once()
            call_kwargs = mock_client.text_to_speech.convert.call_args.kwargs
            assert call_kwargs["text"] == "Hello world"
            assert call_kwargs["voice_id"] == "21m00Tcm4TlvDq8ikWAM"
            assert call_kwargs["model_id"] == "eleven_turbo_v2"
            assert result == output_path
            assert output_path.read_bytes() == b"fake-ogg"


@pytest.mark.asyncio
async def test_fallback_to_gtts_when_key_missing(tmp_path, caplog):
    """Test fallback to gTTS when ELEVENLABS_API_KEY is not set."""
    output_path = tmp_path / "test_output.ogg"
    with patch.dict("os.environ", {"ELEVENLABS_API_KEY": ""}):
        with patch("gtts.gTTS") as mock_gtts:
            with patch(
                "pittqlab_utils.tools.voice_response._mp3_to_ogg"
            ) as mock_convert:
                mock_instance = MagicMock()
                mock_gtts.return_value = mock_instance

                voice = VoiceResponse(backend="elevenlabs")
                result = await voice.synthesize("Hello world", output_path)

                mock_gtts.assert_called_once()
                mock_convert.assert_called_once()
                assert "ELEVENLABS_API_KEY not set" in caplog.text
                assert "falling back to gTTS" in caplog.text


@pytest.mark.asyncio
async def test_send_voice_note_calls_send_voice_and_cleans_up():
    """Test send_voice_note calls send_voice and cleans up temp file."""
    with patch("gtts.gTTS") as mock_gtts:
        with patch(
            "pittqlab_utils.tools.voice_response._mp3_to_ogg"
        ) as mock_convert:
            with patch("builtins.open", MagicMock()) as mock_open:
                mock_open.return_value.__enter__.return_value = MagicMock()

                mock_instance = MagicMock()
                mock_gtts.return_value = mock_instance

                bot = MagicMock()
                bot.send_voice = MagicMock(return_value=None)

                voice = VoiceResponse(backend="gtts")
                await voice.send_voice_note(
                    bot, chat_id=12345, text="Test message"
                )

                bot.send_voice.assert_called_once()
                call_kwargs = bot.send_voice.call_args.kwargs
                assert call_kwargs["chat_id"] == 12345
                assert "voice" in call_kwargs

                mock_instance.save.assert_called_once()
                mock_convert.assert_called_once()


@pytest.mark.asyncio
async def test_synthesize_creates_file_at_expected_path(tmp_path):
    """Test synthesize creates a file at the expected path (gTTS)."""
    output_path = tmp_path / "test_output.ogg"
    with patch("gtts.gTTS") as mock_gtts:
        with patch(
            "pittqlab_utils.tools.voice_response._mp3_to_ogg"
        ) as mock_convert:
            mock_instance = MagicMock()
            mock_gtts.return_value = mock_instance

            voice = VoiceResponse(lang="en", slow=False)
            result = await voice.synthesize("Hello world", output_path)

            mock_instance.save.assert_called_once()
            mock_convert.assert_called_once()
            assert result == output_path
            mock_gtts.assert_called_once_with(
                text="Hello world", lang="en", slow=False
            )
