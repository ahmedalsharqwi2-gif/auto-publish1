import json
import unittest
from pathlib import Path

from tts_quality import enforce_text_quality, resolve_reference_profile


ROOT = Path(__file__).resolve().parents[1]


class NarrationTextQualityTests(unittest.TestCase):
    def test_removes_lingering_latin_tokens_after_revision(self):
        script = "هَذَا نَصٌّ عَرَبِيٌّ جَيِّدٌ يَشْرَحُ الفِكْرَةَ A N O بِوُضُوحٍ."

        cleaned = enforce_text_quality(script, reviser=lambda text, issues: text)

        self.assertNotRegex(cleaned, r"[A-Za-z]")
        self.assertIn("الفِكْرَةَ", cleaned)


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
