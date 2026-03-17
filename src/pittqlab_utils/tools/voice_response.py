"""
Voice response: synthesize text to mp3 via gTTS, send as Telegram voice note.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Any, Optional


class VoiceResponse:
    """Synthesize text to speech and send as Telegram voice note."""

    def __init__(self, lang: str = "en", slow: bool = False):
        self._lang = lang
        self._slow = slow

    async def synthesize(self, text: str, output_path: Path) -> Path:
        """Use gTTS to synthesize text to mp3 at output_path. Returns output_path."""
        def _sync_synth() -> None:
            from gtts import gTTS

            tts = gTTS(text=text, lang=self._lang, slow=self._slow)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            tts.save(str(output_path))

        await asyncio.to_thread(_sync_synth)
        return output_path

    async def send_voice_note(
        self,
        bot: Any,
        chat_id: int,
        text: str,
    ) -> None:
        """Synthesize text to temp mp3, send as Telegram voice note, clean up."""
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            temp_path = Path(f.name)
        try:
            await self.synthesize(text, temp_path)
            # Pass file path; bot.send_voice accepts path or file. Run in thread for sync bots.
            def _send() -> None:
                with open(temp_path, "rb") as f:
                    bot.send_voice(chat_id=chat_id, voice=f)
            await asyncio.to_thread(_send)
        finally:
            if temp_path.exists():
                temp_path.unlink()
