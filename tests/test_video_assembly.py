import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts.assemble_reel import assemble_reel, build_ass_subtitles, probe_media


HAS_VIDEO_TOOLS = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


@unittest.skipUnless(HAS_VIDEO_TOOLS, "ffmpeg and ffprobe are required for the rendering test")
class ReelAssemblyTests(unittest.TestCase):
    def test_builds_arabic_title_and_caption_events(self):
        ass = build_ass_subtitles("عنوان عربي", "جملة أولى. جملة ثانية؟", 4.0)
        self.assertIn("Style: Caption,Noto Sans Arabic", ass)
        self.assertIn("عنوان عربي", ass)
        self.assertIn("جملة أولى.", ass)
        self.assertIn("Dialogue: 0,", ass)

    def test_renders_playable_vertical_mp4_with_audio(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio_path = root / "sample.wav"
            video_path = root / "reel.mp4"
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
                    "-c:a",
                    "pcm_s16le",
                    str(audio_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            result = assemble_reel(
                audio_path=audio_path,
                narration="هَذَا نَصٌّ عَرَبِيٌّ قَصِيرٌ لِلِاخْتِبَارِ.",
                title="اختبار فيديو عربي",
                output_path=video_path,
            )

            self.assertEqual(result, video_path)
            self.assertGreater(video_path.stat().st_size, 1000)
            metadata = probe_media(video_path)
            video = next(s for s in metadata["streams"] if s["codec_type"] == "video")
            audio = next(s for s in metadata["streams"] if s["codec_type"] == "audio")
            self.assertEqual((video["width"], video["height"]), (1080, 1920))
            self.assertEqual(video["codec_name"], "h264")
            self.assertEqual(audio["codec_name"], "aac")


if __name__ == "__main__":
    unittest.main()
