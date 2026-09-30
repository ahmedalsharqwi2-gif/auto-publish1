import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from arabic_tts_quality_checker import ArabicTTSQualityChecker
from scripts.generate_voice import edge_fallback_voice, normalize_edge_pitch
from tts_quality import enforce_text_quality, resolve_reference_profile


ROOT = Path(__file__).resolve().parents[1]


class NarrationTextQualityTests(unittest.TestCase):
    @patch("arabic_tts_quality_checker.subprocess.run")
    def test_mp3_duration_uses_ffprobe(self, run):
        run.return_value.stdout = "180.0\n"
        checker = ArabicTTSQualityChecker(min_acceptable_score=0.0)
        class FakeSegment:
            text = " ".join(["كلمة"] * 288)

        class FakeWhisperModel:
            def __init__(self, *args, **kwargs):
                pass

            def transcribe(self, *args, **kwargs):
                return iter([FakeSegment()]), object()

        fake_whisper = types.ModuleType("faster_whisper")
        fake_whisper.WhisperModel = FakeWhisperModel
        fake_numpy = types.ModuleType("numpy")
        fake_scipy = types.ModuleType("scipy")
        fake_scipy.__path__ = []
        fake_scipy_io = types.ModuleType("scipy.io")
        fake_scipy_io.__path__ = []
        fake_wavfile = types.ModuleType("scipy.io.wavfile")
        fake_wavfile.read = lambda _: (16000, [])
        fake_scipy_io.wavfile = fake_wavfile
        with patch.dict(sys.modules, {
            "faster_whisper": fake_whisper,
            "numpy": fake_numpy,
            "scipy": fake_scipy,
            "scipy.io": fake_scipy_io,
            "scipy.io.wavfile": fake_wavfile,
        }), patch.object(Path, "exists", return_value=True):
            score, issues = checker.check_audio_quality(
                Path("sample.mp3"), " ".join(["كلمة"] * 288)
            )
        self.assertEqual(score, 1.0)
        self.assertEqual(issues, [])
        self.assertEqual(run.call_args.args[0][0], "ffprobe")

    def test_google_voice_is_mapped_to_valid_edge_voice(self):
        self.assertEqual(edge_fallback_voice("ar-XA-Neural2-B"), "ar-SA-HamedNeural")
        self.assertEqual(edge_fallback_voice("ar-SA-AmmarNeural"), "ar-SA-AmmarNeural")

    def test_edge_pitch_always_has_required_sign(self):
        self.assertEqual(normalize_edge_pitch("0Hz"), "+0Hz")
        self.assertEqual(normalize_edge_pitch("-5Hz"), "-5Hz")
        self.assertEqual(normalize_edge_pitch("7"), "+7Hz")
        self.assertEqual(normalize_edge_pitch("invalid"), "+0Hz")

    def test_special_character_report_does_not_crash(self):
        checker = ArabicTTSQualityChecker()
        score, issues, warnings = checker.check_text_quality("نص عربي @ $ % &")

        self.assertLess(score, 1.0)
        self.assertTrue(any("رموز خاصة" in issue for issue in issues))

    def test_removes_lingering_latin_tokens_after_revision(self):
        script = "هَذَا نَصٌّ عَرَبِيٌّ جَيِّدٌ يَشْرَحُ الفِكْرَةَ A N O بِوُضُوحٍ."

        cleaned = enforce_text_quality(script, reviser=lambda text, issues: text)

        self.assertNotRegex(cleaned, r"[A-Za-z]")
        self.assertIn("الفِكْرَةَ", cleaned)

    def test_rejects_common_pronoun_verb_mismatch(self):
        with self.assertRaises(Exception):
            enforce_text_quality("هُوَ قَالَتْ ذَلِكَ.")
        with self.assertRaises(Exception):
            enforce_text_quality("هِيَ ذَهَبَ إِلَى الْبَيْتِ.")


class UploadedVoiceProfileTests(unittest.TestCase):
    def test_all_uploaded_voices_resolve_to_existing_audio_and_text(self):
        config_path = ROOT / "assets/voices/voice_profiles.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        profile_ids = {"hossam_uploaded", "marwan", "nassim", "ahmed_z"}

        for profile_id in profile_ids:
            with self.subTest(profile=profile_id):
                wav, text = resolve_reference_profile(profile_id, config_path)
                self.assertTrue(wav.is_file())
                self.assertTrue(text.strip())
                self.assertEqual(config["profiles"][profile_id]["wav"], str(wav.relative_to(ROOT)))


if __name__ == "__main__":
    unittest.main()
