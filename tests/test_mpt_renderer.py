import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from mpt_renderer import render_with_moneyprinterturbo


class MoneyPrinterTurboAdapterTests(unittest.TestCase):
    def test_renderer_passes_approved_audio_script_and_local_media(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "mpt"
            root.mkdir(parents=True)
            (root / "cli.py").write_text("# test cli\n", encoding="utf-8")
            source_audio = Path(directory) / "narration.mp3"
            source_audio.write_bytes(b"audio")
            output = Path(directory) / "final.mp4"

            def fake_run(command, **_kwargs):
                self.assertIn("--video-script", command)
                self.assertIn("نص عربي معتمد", command)
                self.assertIn("--custom-audio-file", command)
                self.assertIn(str(source_audio.resolve()), command)
                self.assertIn("--video-source", command)
                self.assertIn("local", command)
                task_id = command[command.index("--task-id") + 1]
                task_dir = root / "storage" / "tasks" / task_id
                task_dir.mkdir(parents=True)
                (task_dir / "final-1.mp4").write_bytes(b"video")
                return SimpleNamespace(returncode=0, stdout="MPT_RESULT", stderr="")

            with patch.dict(
                "os.environ",
                {
                    "MONEYPRINTERTURBO_ROOT": str(root),
                    "MPT_PYTHON": "python",
                    "MPT_RENDER_TIMEOUT": "30",
                },
                clear=False,
            ), patch("mpt_renderer._probe_duration", return_value=5.0), patch(
                "mpt_renderer.build_pexels_track", return_value=True
            ), patch("mpt_renderer.subprocess.run", side_effect=fake_run), patch(
                "mpt_renderer._overlay_arabic_subtitles"
            ) as overlay:
                result = render_with_moneyprinterturbo(
                    source_audio,
                    "نص عربي معتمد",
                    output,
                    topic="موضوع علمي",
                )

            self.assertEqual(result, output)
            self.assertEqual(output.read_bytes(), b"video")
            overlay.assert_called_once_with(output, "نص عربي معتمد", 5.0)


if __name__ == "__main__":
    unittest.main()
