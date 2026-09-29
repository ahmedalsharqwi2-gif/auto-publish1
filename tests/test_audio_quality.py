import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from arabic_tts_quality_checker import ArabicTTSQualityChecker


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg and ffprobe required")
class AudioDurationQualityTests(unittest.TestCase):
    def test_mp3_duration_is_measured_with_ffprobe(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "sample.mp3"
            subprocess.run(
                [
                    shutil.which("ffmpeg"),
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "sine=frequency=440:duration=2",
                    "-codec:a",
                    "libmp3lame",
                    "-b:a",
                    "32k",
                    str(audio),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            score, issues = ArabicTTSQualityChecker().check_audio_quality(
                audio,
                "واحد اثنان ثلاثة أربعة خمسة",
            )
        self.assertEqual(score, 1.0)
        self.assertEqual(issues, [])

    def test_missing_audio_fails_quality_check(self):
        score, issues = ArabicTTSQualityChecker().check_audio_quality(
            Path("/missing/sample.mp3"),
            "نص اختباري",
        )
        self.assertEqual(score, 0.0)
        self.assertTrue(issues)


if __name__ == "__main__":
    unittest.main()
