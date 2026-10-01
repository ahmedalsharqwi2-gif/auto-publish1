import unittest
from types import SimpleNamespace
from unittest.mock import patch

import main
from scripts.generate_content import ContentGenerator


class CurrentNarrationPolicyTests(unittest.TestCase):
    def test_publish_window_has_valid_order(self):
        self.assertGreater(main.MIN_AUDIO_SECONDS, 0)
        self.assertGreater(main.MAX_AUDIO_SECONDS, main.MIN_AUDIO_SECONDS)

    def test_content_generator_keeps_narration_inside_configured_word_window(self):
        generator = ContentGenerator(min_words=3, max_words=5)
        generated = "واحد اثنان ثلاثة أربعة"
        with (
            patch("scripts.generate_content.llm_chat", return_value=generated),
            patch.object(generator.grammar_fixer, "fix_text", return_value=(generated, [])),
            patch.object(
                generator.quality_checker,
                "generate_report",
                return_value=SimpleNamespace(is_acceptable=True, issues=[], overall_score=1.0),
            ),
        ):
            result = generator.generate_narration("موضوع اختباري")

        self.assertEqual(result, generated)
        self.assertTrue(3 <= len(result.split()) <= 5)

    def test_pipeline_rejects_audio_below_and_above_window(self):
        for duration in (main.MIN_AUDIO_SECONDS - 0.01, main.MAX_AUDIO_SECONDS + 0.01):
            pipeline = main.AutoPublishPipeline()
            with patch.object(main, "probe_duration", return_value=duration):
                report = SimpleNamespace(is_acceptable=True, overall_score=1.0, issues=[], warnings=[])
                pipeline.content_generator.generate_narration = lambda _topic: "نص عربي"
                pipeline.quality_checker.check_text = lambda text: (text, report)
                pipeline.quality_checker.check_audio = lambda _path, _text: report
                pipeline.voice_generator.generate = lambda _text, output_path: (output_path, True)
                self.assertFalse(pipeline.run(topic="موضوع اختباري"))


if __name__ == "__main__":
    unittest.main()
