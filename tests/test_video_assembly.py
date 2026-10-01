import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.assemble_video import write_ass_subtitles


class VideoAssemblyTests(unittest.TestCase):
    def test_subtitles_use_four_words_per_block_and_two_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "captions.ass"
            write_ass_subtitles(
                "واحد اثنان ثلاثة أربعة خمسة ستة سبعة",
                7.0,
                path,
            )
            text = path.read_text(encoding="utf-8")
            self.assertEqual(text.count("Dialogue:"), 2)
            self.assertIn(r"\N", text)
            self.assertIn("واحد اثنان", text)
            self.assertIn("ثلاثة أربعة", text)
            self.assertIn("Noto Sans Arabic", text)

    def test_captions_are_top_centered_below_mobile_notch_safe_area(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "captions.ass"
            write_ass_subtitles("نص تجريبي للترجمة العربية", 5.0, path)
            text = path.read_text(encoding="utf-8")
        self.assertIn(",8,70,70,260,1", text)

    @patch("scripts.assemble_video.align_words_with_whisper", side_effect=RuntimeError("Whisper alignment too weak: 77/157 words"))
    def test_weak_whisper_alignment_falls_back_to_uniform_timing(self, align):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "captions.ass"
            write_ass_subtitles(
                "واحد اثنان ثلاثة أربعة خمسة ستة سبعة ثمانية",
                8.0,
                path,
                audio_path=Path(directory) / "narration.mp3",
            )
            text = path.read_text(encoding="utf-8")
        align.assert_called_once()
        self.assertEqual(text.count("Dialogue:"), 2)
        self.assertIn("خمسة ستة", text)


if __name__ == "__main__":
    unittest.main()
