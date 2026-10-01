import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
