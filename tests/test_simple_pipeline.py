import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from scripts.generate_content import ContentGenerator


class ContentGeneratorTests(unittest.TestCase):
    @patch("scripts.generate_content.gemini_chat", return_value="  موضوع جديد  ")
    def test_topic_generation_uses_category_and_returns_clean_text(self, gemini_chat):
        generator = ContentGenerator(min_words=90, max_words=165)
        self.assertEqual(generator.generate_topic("علوم مبسطة"), "موضوع جديد")
        prompt = gemini_chat.call_args.args[0][0]["content"]
        self.assertIn("علوم مبسطة", prompt)

    @patch("scripts.generate_content.gemini_chat", return_value="نَصٌّ عَرَبِيٌّ مُفِيدٌ")
    def test_narration_prompt_uses_configured_word_window(self, gemini_chat):
        generator = ContentGenerator(min_words=90, max_words=165)
        with (
            patch.object(generator.grammar_fixer, "fix_text", side_effect=lambda text: (text, [])),
            patch.object(
                generator.quality_checker,
                "generate_report",
                return_value=SimpleNamespace(is_acceptable=True, issues=[], overall_score=0.95),
            ),
        ):
            narration = generator.generate_narration("موضوع اختباري")
        self.assertEqual(narration, "نَصٌّ عَرَبِيٌّ مُفِيدٌ")
        prompt = gemini_chat.call_args.args[0][0]["content"]
        self.assertIn("90-165 كلمة", prompt)
        self.assertIn("موضوع اختباري", prompt)


class AutoPublishPipelineTests(unittest.TestCase):
    def _report(self):
        return SimpleNamespace(overall_score=0.95, is_acceptable=True, issues=[])

    def test_simple_pipeline_assembles_hosts_and_queues_video(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            audio_path = output_dir / "narration.mp3"
            video_path = output_dir / "reel.mp4"
            narration = "نَصٌّ عَرَبِيٌّ مُفِيدٌ لِمُشَاهَدِي القناة."
            topic = "عنوان المقطع"
            content = SimpleNamespace(
                generate_topic=lambda category: topic,
                generate_narration=lambda received_topic: narration,
            )
            voice = SimpleNamespace(
                output_dir=output_dir,
                generate=Mock(return_value=(audio_path, True)),
            )
            quality = SimpleNamespace(
                check_text=lambda text: (text, self._report()),
                check_audio=lambda path, text: self._report(),
            )
            publisher = SimpleNamespace(publish_to_buffer=Mock(return_value=True))

            with (
                patch.dict(
                    os.environ,
                    {
                        "OUTPUT_DIR": tmp,
                        "MIN_WORDS": "90",
                        "MAX_WORDS": "165",
                        "MIN_QUALITY_SCORE": "0.75",
                        "BUFFER_CHANNEL_IDS": "channel-1, channel-2",
                    },
                    clear=False,
                ),
                patch("main.ContentGenerator", return_value=content),
                patch("main.VoiceGenerator", return_value=voice),
                patch("main.QualityCheckPipeline", return_value=quality),
                patch("main.ContentPublisher", return_value=publisher),
                patch("main.assemble_reel", return_value=video_path) as assemble,
                patch("main.host_video_on_github", return_value="https://github.com/example/reel.mp4") as host,
            ):
                from main import AutoPublishPipeline

                pipeline = AutoPublishPipeline()
                result = pipeline.run(category="علوم", topic=topic)

            self.assertTrue(result)
            voice.generate.assert_called_once_with(
                narration,
                output_path=audio_path,
                engine="edge",
                voice=None,
            )
            assemble.assert_called_once_with(
                audio_path=audio_path,
                narration=narration,
                title=topic,
                output_path=video_path,
            )
            host.assert_called_once_with(video_path)
            publisher.publish_to_buffer.assert_called_once()
            call = publisher.publish_to_buffer.call_args.kwargs
            self.assertEqual(call["video_url"], "https://github.com/example/reel.mp4")
            self.assertEqual(call["channel_ids"], ["channel-1", "channel-2"])

    def test_missing_buffer_channel_ids_stops_before_generation(self):
        content = SimpleNamespace(generate_topic=lambda category: self.fail("generation should not start"))
        with (
            patch.dict(os.environ, {"BUFFER_CHANNEL_IDS": ""}, clear=False),
            patch("main.ContentGenerator", return_value=content),
            patch("main.VoiceGenerator"),
            patch("main.QualityCheckPipeline"),
            patch("main.ContentPublisher"),
        ):
            from main import AutoPublishPipeline

            self.assertFalse(AutoPublishPipeline().run())


if __name__ == "__main__":
    unittest.main()
