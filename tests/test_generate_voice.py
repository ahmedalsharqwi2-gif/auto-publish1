import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.generate_voice import DEFAULT_EDGE_VOICE, VoiceGenerator


class VoiceGeneratorFallbackTests(unittest.TestCase):
    def test_edge_engine_uses_supported_arabic_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            generator = VoiceGenerator(output_dir=Path(tmp))
            generator.edge_tts_available = True
            output = Path(tmp) / "voice.mp3"
            with patch.object(generator, "generate_with_edge_tts", return_value=(True, "ok")) as edge:
                result_path, success = generator.generate("نص عربي", output, engine="edge")
        self.assertTrue(success)
        self.assertEqual(result_path, output)
        edge.assert_called_once_with("نص عربي", output, DEFAULT_EDGE_VOICE)

    def test_google_failure_falls_back_to_edge_config_not_google_voice_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            generator = VoiceGenerator(output_dir=Path(tmp))
            generator.edge_tts_available = True
            output = Path(tmp) / "voice.mp3"
            with (
                patch.dict(os.environ, {"EDGE_TTS_VOICE": "ar-SA-ZariyahNeural"}),
                patch.object(generator, "generate_with_google_cloud", return_value=(False, "no credentials")),
                patch.object(generator, "generate_with_edge_tts", return_value=(True, "ok")) as edge,
            ):
                _, success = generator.generate(
                    "نص عربي",
                    output,
                    engine="google",
                    voice="ar-XA-Neural2-B",
                )
        self.assertTrue(success)
        edge.assert_called_once_with("نص عربي", output, "ar-SA-ZariyahNeural")

    def test_unknown_engine_fails_without_synthesis(self):
        with tempfile.TemporaryDirectory() as tmp:
            generator = VoiceGenerator(output_dir=Path(tmp))
            with patch.object(generator, "generate_with_edge_tts") as edge:
                _, success = generator.generate("نص عربي", engine="unsupported")
        self.assertFalse(success)
        edge.assert_not_called()


if __name__ == "__main__":
    unittest.main()
