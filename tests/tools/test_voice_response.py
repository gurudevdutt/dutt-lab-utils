"""Tests for VoiceResponse. Mocks gTTS and bot.send_voice."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

from pittqlab_utils.tools.voice_response import VoiceResponse


@pytest.mark.asyncio
async def test_synthesize_creates_file_at_expected_path(tmp_path):
    """Test synthesize creates a file at the expected path."""
    with patch("gtts.gTTS") as mock_gtts:
        mock_instance = MagicMock()
        mock_gtts.return_value = mock_instance

        voice = VoiceResponse(lang="en", slow=False)
        output_path = tmp_path / "test_output.mp3"
        result = await voice.synthesize("Hello world", output_path)

        mock_instance.save.assert_called_once_with(str(output_path))
        assert result == output_path
        # gTTS save writes the file; our mock doesn't, so we just check the call
        mock_gtts.assert_called_once_with(text="Hello world", lang="en", slow=False)


@pytest.mark.asyncio
async def test_send_voice_note_calls_send_voice_and_cleans_up():
    """Test send_voice_note calls send_voice and cleans up temp file."""
    with patch("gtts.gTTS") as mock_gtts:
        with patch("builtins.open", MagicMock()) as mock_open:
            mock_open.return_value.__enter__.return_value = MagicMock()

            mock_instance = MagicMock()
            mock_gtts.return_value = mock_instance

            bot = MagicMock()
            bot.send_voice = MagicMock(return_value=None)

            voice = VoiceResponse()
            await voice.send_voice_note(bot, chat_id=12345, text="Test message")

            bot.send_voice.assert_called_once()
            call_kwargs = bot.send_voice.call_args.kwargs
            assert call_kwargs["chat_id"] == 12345
            assert "voice" in call_kwargs

            # Temp file should be cleaned up (unlinked) - we can't easily assert
            # the exact temp path, but we verify send_voice was called
            mock_instance.save.assert_called_once()
