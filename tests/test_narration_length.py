import unittest
from unittest.mock import patch

from main import (
    MAX_SCRIPT_WORDS,
    OUTRO_MAX_WORDS,
    PipelineError,
    _audio_duration_exceeds_maximum,
    _audio_duration_needs_normalization,
    _validate_final_script_word_count,
    build_topic_user_prompt,
)


class NarrationLengthTests(unittest.TestCase):
    def test_audio_duration_helpers_use_current_limits(self):
        with patch("main.MAX_AUDIO_SECONDS", 90.0), patch("main.TARGET_AUDIO_SECONDS", 89.0):
            self.assertFalse(_audio_duration_exceeds_maximum(90.0))
            self.assertTrue(_audio_duration_exceeds_maximum(90.01))
            self.assertFalse(_audio_duration_needs_normalization(89.0))
            self.assertTrue(_audio_duration_needs_normalization(89.01))

    def test_script_word_count_rejects_only_out_of_range_values(self):
        with self.assertRaises(PipelineError):
            _validate_final_script_word_count(1)
        with self.assertRaisesRegex(PipelineError, "maximum"):
            _validate_final_script_word_count(MAX_SCRIPT_WORDS + 1)
        _validate_final_script_word_count(120)

    def test_prompt_contains_current_script_window(self):
        prompt = build_topic_user_prompt(["موضوع سابق"], ["حقائق علمية صادمة"])
        maximum_before_outro = MAX_SCRIPT_WORDS - OUTRO_MAX_WORDS
        self.assertIn(str(maximum_before_outro), prompt)
        self.assertIn("الموضوعات المستخدمة مؤخرًا", prompt)


if __name__ == "__main__":
    unittest.main()
