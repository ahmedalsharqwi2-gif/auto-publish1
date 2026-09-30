# -*- coding: utf-8 -*-
"""
Google Cloud Text-to-Speech Engine - محرك Google Cloud للنطق
==============================================================
بديل احترافي عالي الجودة لـ SILMA و Edge TTS.
"""

from __future__ import annotations

import os
import logging
import time
from pathlib import Path
from typing import Optional, List

log = logging.getLogger("pipeline")


class GoogleCloudTTS:
    """محرك Google Cloud Text-to-Speech."""

    # قائمة الأصوات العربية المتاحة في Google Cloud
    ARABIC_VOICES = {
        'ar-XA-Standard-A': {'gender': 'MALE', 'quality': 'STANDARD'},
        'ar-XA-Standard-B': {'gender': 'FEMALE', 'quality': 'STANDARD'},
        'ar-XA-Standard-C': {'gender': 'MALE', 'quality': 'STANDARD'},
        'ar-XA-Standard-D': {'gender': 'FEMALE', 'quality': 'STANDARD'},
        'ar-XA-Neural2-A': {'gender': 'MALE', 'quality': 'NEURAL2'},
        'ar-XA-Neural2-B': {'gender': 'FEMALE', 'quality': 'NEURAL2'},
        'ar-XA-Neural2-C': {'gender': 'MALE', 'quality': 'NEURAL2'},
        'ar-XA-Neural2-D': {'gender': 'FEMALE', 'quality': 'NEURAL2'},
    }

    def __init__(
        self,
        credentials_path: Optional[str] = None,
        default_voice: str = 'ar-XA-Neural2-B',
        default_speaking_rate: float = 1.0,
        default_pitch: float = 0.0,
    ):
        """Initialize Google Cloud TTS.
        
        Args:
            credentials_path: Path to Google Cloud credentials JSON file
            default_voice: Default voice to use
            default_speaking_rate: Default speaking rate (0.25 to 4.0)
            default_pitch: Default pitch in Hz (-20.0 to 20.0)
        """
        self.credentials_path = credentials_path or os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        self.default_voice = default_voice
        self.default_speaking_rate = default_speaking_rate
        self.default_pitch = default_pitch
        self.client = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize Google Cloud TTS client."""
        try:
            from google.cloud import texttospeech
            
            if self.credentials_path:
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = self.credentials_path
            
            self.client = texttospeech.TextToSpeechClient()
            log.info("Google Cloud TTS client initialized successfully")
        except ImportError:
            log.error("google-cloud-texttospeech not installed. Install it with: pip install google-cloud-texttospeech")
            self.client = None
        except Exception as e:
            log.error(f"Failed to initialize Google Cloud TTS: {e}")
            self.client = None

    def generate_speech(
        self,
        text: str,
        output_path: Path,
        voice: Optional[str] = None,
        speaking_rate: Optional[float] = None,
        pitch: Optional[float] = None,
        audio_encoding: str = 'MP3',
    ) -> bool:
        """Generate speech from Arabic text.
        
        Args:
            text: Arabic text to convert
            output_path: Path to save audio file
            voice: Voice ID (defaults to self.default_voice)
            speaking_rate: Speaking rate (0.25 to 4.0)
            pitch: Pitch in Hz (-20.0 to 20.0)
            audio_encoding: MP3, LINEAR16, OGG_OPUS
            
        Returns:
            True if successful, False otherwise
        """
        if not self.client:
            log.error("Google Cloud TTS client not initialized")
            return False

        try:
            from google.cloud import texttospeech
            
            voice_id = voice or self.default_voice
            speaking_rate = speaking_rate or self.default_speaking_rate
            pitch = pitch or self.default_pitch

            # Validate voice
            if voice_id not in self.ARABIC_VOICES:
                log.warning(f"Voice {voice_id} not recognized, using default")
                voice_id = self.default_voice

            # Create synthesis input
            synthesis_input = texttospeech.SynthesisInput(text=text)

            # Create voice configuration
            voice = texttospeech.VoiceSelectionParams(
                language_code='ar-XA',
                name=voice_id,
            )

            # Create audio configuration
            audio_config = texttospeech.AudioConfig(
                audio_encoding=getattr(texttospeech.AudioEncoding, audio_encoding),
                speaking_rate=speaking_rate,
                pitch=pitch,
            )

            # Generate speech with bounded exponential retry for rate limits and transient outages.
            log.info(f"Generating speech with voice {voice_id}, rate={speaking_rate}, pitch={pitch}")
            retries = max(1, int(os.getenv("GOOGLE_TTS_RETRIES", "3")))
            response = None
            for attempt in range(1, retries + 1):
                try:
                    response = self.client.synthesize_speech(
                        input=synthesis_input,
                        voice=voice,
                        audio_config=audio_config,
                    )
                    break
                except Exception as exc:
                    code = getattr(exc, "code", lambda: None)()
                    retryable = code in (408, 429, 500, 502, 503, 504) or any(
                        marker in str(exc).lower()
                        for marker in ("rate", "temporar", "timeout", "unavailable")
                    )
                    if not retryable or attempt >= retries:
                        raise
                    delay = min(30, 2 ** (attempt - 1) * 2)
                    log.warning(
                        "Google Cloud TTS transient failure (%s); retry %d/%d in %.1fs",
                        exc, attempt, retries, delay,
                    )
                    time.sleep(delay)

            # Save audio
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'wb') as f:
                f.write(response.audio_content)

            log.info(f"Speech generated successfully: {output_path}")
            return True

        except Exception as e:
            log.error(f"Error generating speech: {e}")
            return False

    def batch_generate(
        self,
        texts: List[str],
        output_dir: Path,
        voice: Optional[str] = None,
        speaking_rate: Optional[float] = None,
    ) -> List[Path]:
        """Generate speech for multiple texts.
        
        Args:
            texts: List of Arabic texts
            output_dir: Directory to save audio files
            voice: Voice ID
            speaking_rate: Speaking rate
            
        Returns:
            List of generated audio file paths
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        generated_files = []

        for i, text in enumerate(texts):
            output_path = output_dir / f"speech_{i:03d}.mp3"
            if self.generate_speech(
                text=text,
                output_path=output_path,
                voice=voice,
                speaking_rate=speaking_rate,
            ):
                generated_files.append(output_path)
            else:
                log.warning(f"Failed to generate speech for text {i}")

        return generated_files

    def list_voices(self) -> dict:
        """Return available Arabic voices."""
        return self.ARABIC_VOICES.copy()

    def set_default_voice(self, voice: str):
        """Set default voice for future generations."""
        if voice in self.ARABIC_VOICES:
            self.default_voice = voice
            log.info(f"Default voice set to {voice}")
        else:
            log.warning(f"Voice {voice} not recognized")
