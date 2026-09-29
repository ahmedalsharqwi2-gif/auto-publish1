# -*- coding: utf-8 -*-
"""
scripts/generate_voice.py - توليد الصوت
========================================
يولد الصوت من النص باستخدام محركات مختلفة مع فحص الجودة.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, Tuple

try:
    from google.cloud import texttospeech
    GOOGLE_TTS_AVAILABLE = True
except ImportError:
    GOOGLE_TTS_AVAILABLE = False

from google_cloud_tts import GoogleCloudTTS
from arabic_tts_quality_checker import ArabicTTSQualityChecker

log = logging.getLogger("pipeline")


class VoiceGenerator:
    """يولد الصوت من النصوص مع فحص الجودة."""

    def __init__(self, output_dir: Path = Path("output")):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize TTS engines
        try:
            self.google_tts = GoogleCloudTTS() if GOOGLE_TTS_AVAILABLE else None
        except:
            self.google_tts = None
            log.warning("Google Cloud TTS not available")
        
        try:
            import edge_tts
            self.edge_tts_available = True
        except ImportError:
            self.edge_tts_available = False
            log.warning("Edge TTS not available")
        
        self.quality_checker = ArabicTTSQualityChecker(min_acceptable_score=0.7)

    def generate_with_google_cloud(
        self,
        text: str,
        output_path: Path,
        voice: str = 'ar-XA-Neural2-B',
        speaking_rate: float = 1.0,
    ) -> Tuple[bool, str]:
        """Generate speech using Google Cloud TTS."""
        if not self.google_tts:
            return False, "Google Cloud TTS not available"
        
        log.info(f"Generating speech with Google Cloud TTS (voice: {voice})")
        success = self.google_tts.generate_speech(
            text=text,
            output_path=output_path,
            voice=voice,
            speaking_rate=speaking_rate,
        )
        
        if success:
            # Check quality
            _, audio_issues = self.quality_checker.check_audio_quality(output_path, text)
            if audio_issues:
                return True, f"Generated with warnings: {audio_issues[0]}"
            return True, "Successfully generated"
        else:
            return False, "Google Cloud TTS generation failed"

    def generate_with_edge_tts(
        self,
        text: str,
        output_path: Path,
        voice: str = "ar-SA-AmmarNeural",
        rate: str = "+10%",
        pitch: str = "0Hz",
    ) -> Tuple[bool, str]:
        """Generate speech using Edge TTS (fallback)."""
        if not self.edge_tts_available:
            return False, "Edge TTS not available"
        
        import edge_tts
        import asyncio
        
        log.info(f"Generating speech with Edge TTS (voice: {voice})")
        
        async def _generate():
            try:
                communicate = edge_tts.Communicate(
                    text=text,
                    voice=voice,
                    rate=rate,
                    pitch=pitch,
                )
                await communicate.save(str(output_path))
                return True
            except Exception as e:
                log.error(f"Edge TTS generation failed: {e}")
                return False
        
        try:
            result = asyncio.run(_generate())
            if result:
                # Check quality
                score, audio_issues = self.quality_checker.check_audio_quality(output_path, text)
                if audio_issues:
                    return True, f"Generated with warnings: {audio_issues[0]}"
                return True, "Successfully generated"
            else:
                return False, "Edge TTS generation failed"
        except Exception as e:
            return False, str(e)

    def generate(
        self,
        text: str,
        output_path: Optional[Path] = None,
        engine: str = "google",
        voice: Optional[str] = None,
    ) -> Tuple[Path, bool]:
        """Generate speech with fallback strategy."""
        if output_path is None:
            output_path = self.output_dir / "narration.mp3"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Try Google Cloud TTS first
        if engine == "google" or not voice:
            voice = voice or 'ar-XA-Neural2-B'
            success, message = self.generate_with_google_cloud(text, output_path, voice)
            if success:
                log.info(f"Audio generation successful: {message}")
                return output_path, True
            else:
                log.warning(f"Google Cloud TTS failed: {message}, trying Edge TTS")
        
        # Fallback to Edge TTS
        if self.edge_tts_available:
            edge_voice = voice or "ar-SA-AmmarNeural"
            success, message = self.generate_with_edge_tts(text, output_path, edge_voice)
            if success:
                log.info(f"Audio generation successful with Edge TTS: {message}")
                return output_path, True
            else:
                log.error(f"Edge TTS also failed: {message}")
                return output_path, False
        
        log.error("No TTS engine available")
        return output_path, False


if __name__ == "__main__":
    generator = VoiceGenerator()
    
    test_text = "السلام عليكم ورحمة الله وبركاته. هذا نص اختبار لتوليد الصوت بجودة عالية."
    output_path, success = generator.generate(test_text)
    
    if success:
        print(f"Audio generated successfully at: {output_path}")
    else:
        print(f"Failed to generate audio")
