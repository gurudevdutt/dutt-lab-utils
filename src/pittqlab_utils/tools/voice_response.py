"""
Voice response: synthesize text to ogg via gTTS or ElevenLabs, send as Telegram voice note.
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Literal, Protocol

logger = logging.getLogger(__name__)

# ElevenLabs defaults
ELEVENLABS_DEFAULT_VOICE = "21m00Tcm4TlvDq8ikWAM"  # Rachel
ELEVENLABS_DEFAULT_MODEL = "eleven_turbo_v2"


class VoiceBackend(Protocol):
    """Protocol for TTS backends."""

    async def synthesize(self, text: str, output_path: Path) -> Path:
        """Synthesize text to audio at output_path. Returns output_path."""
        ...


class GTTSSVoice:
    """gTTS backend: synthesizes to mp3, converts to ogg/opus via pydub."""

    def __init__(self, lang: str = "en", slow: bool = False):
        self._lang = lang
        self._slow = slow

    async def synthesize(self, text: str, output_path: Path) -> Path:
        """Use gTTS to synthesize text, convert mp3 to ogg at output_path."""
        def _sync_synth() -> None:
            from gtts import gTTS

            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                mp3_path = Path(f.name)
            try:
                tts = gTTS(text=text, lang=self._lang, slow=self._slow)
                tts.save(str(mp3_path))
                _mp3_to_ogg(mp3_path, output_path)
            finally:
                if mp3_path.exists():
                    mp3_path.unlink()

        await asyncio.to_thread(_sync_synth)
        return output_path


def _mp3_to_ogg(mp3_path: Path, ogg_path: Path) -> None:
    """Convert mp3 to ogg/opus using pydub. Raises clear error if ffmpeg missing."""
    try:
        from pydub import AudioSegment

        audio = AudioSegment.from_mp3(str(mp3_path))
        output_path = Path(ogg_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        audio.export(str(output_path), format="ogg", codec="libopus")
    except FileNotFoundError as e:
        raise RuntimeError(
            "ffmpeg is required for gTTS ogg conversion. "
            "Install it: Ubuntu/Debian: sudo apt install ffmpeg; Mac: brew install ffmpeg"
        ) from e


class ElevenLabsVoice:
    """ElevenLabs backend: synthesizes to ogg/opus natively."""

    def __init__(
        self,
        voice_id: str = ELEVENLABS_DEFAULT_VOICE,
        model_id: str = ELEVENLABS_DEFAULT_MODEL,
    ):
        self._voice_id = voice_id
        self._model_id = model_id

    async def synthesize(self, text: str, output_path: Path) -> Path:
        """Use ElevenLabs to synthesize text to ogg/opus at output_path."""
        def _sync_synth() -> None:
            from elevenlabs.client import ElevenLabs

            client = ElevenLabs(api_key=os.environ.get("ELEVENLABS_API_KEY"))
            audio_bytes = client.text_to_speech.convert(
                voice_id=self._voice_id,
                text=text,
                model_id=self._model_id,
                output_format="opus_48000_64",
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(audio_bytes)

        await asyncio.to_thread(_sync_synth)
        return output_path


class VoiceResponse:
    """Synthesize text to speech and send as Telegram voice note."""

    def __init__(
        self,
        backend: Literal["gtts", "elevenlabs"] = "gtts",
        lang: str = "en",
        slow: bool = False,
    ):
        self._backend_name = backend
        self._lang = lang
        self._slow = slow
        self._backend: VoiceBackend = self._resolve_backend()

    def _resolve_backend(self) -> VoiceBackend:
        if self._backend_name == "elevenlabs":
            if not os.environ.get("ELEVENLABS_API_KEY"):
                logger.warning(
                    "ELEVENLABS_API_KEY not set; falling back to gTTS backend"
                )
                return GTTSSVoice(lang=self._lang, slow=self._slow)
            return ElevenLabsVoice()
        return GTTSSVoice(lang=self._lang, slow=self._slow)

    async def synthesize(self, text: str, output_path: Path) -> Path:
        """Synthesize text to ogg at output_path. Returns output_path."""
        return await self._backend.synthesize(text, output_path)

    async def send_voice_note(
        self,
        bot: Any,
        chat_id: int,
        text: str,
    ) -> None:
        """Synthesize text to temp ogg, send as Telegram voice note, clean up."""
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as f:
            temp_path = Path(f.name)
        try:
            await self.synthesize(text, temp_path)
            def _send() -> None:
                with open(temp_path, "rb") as f:
                    bot.send_voice(chat_id=chat_id, voice=f)
            await asyncio.to_thread(_send)
        finally:
            if temp_path.exists():
                temp_path.unlink()
