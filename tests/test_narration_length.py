import unittest
from unittest.mock import patch

from main import (
    OUTRO_MAX_WORDS,
    OUTRO_MIN_WORDS,
    PipelineError,
    _audio_duration_exceeds_limit,
    _audio_duration_needs_normalization,
    _validate_final_script_word_count,
    build_topic_user_prompt,
)


class MaximumOnlyNarrationLengthTests(unittest.TestCase):
    def test_short_video_duration_is_allowed_and_90_seconds_is_the_only_cap(self):
        with patch("main.MAX_AUDIO_SECONDS", 90.0), patch("main.TARGET_AUDIO_SECONDS", 89.0):
            self.assertFalse(_audio_duration_exceeds_limit(8.5))
            self.assertFalse(_audio_duration_exceeds_limit(90.0))
            self.assertTrue(_audio_duration_exceeds_limit(90.01))
            self.assertFalse(_audio_duration_needs_normalization(8.5))
            self.assertFalse(_audio_duration_needs_normalization(89.0))
            self.assertTrue(_audio_duration_needs_normalization(89.01))

    def test_short_script_with_fixed_outro_is_allowed(self):
        # One factual narration word plus the fixed CTA has no lower-bound rejection.
        self.assertIsNone(_validate_final_script_word_count(OUTRO_MIN_WORDS + 1))

    def test_script_above_maximum_is_still_rejected(self):
        with self.assertRaisesRegex(PipelineError, "maximum"):
            _validate_final_script_word_count(166)

    def test_topic_prompt_requests_only_a_maximum_not_a_minimum(self):
        seed = {
            "id": "science_test",
            "category": "science",
            "subject": "موضوع علمي",
            "angle": "زاوية اختيارية",
            "verified_fact": "حقيقة مثبتة",
        }
        sources = [{"url": "https://example.org/specific-evidence", "text": "دليل"}]
        with patch("main.MAX_SCRIPT_WORDS", 165), patch("main.OUTRO_MAX_WORDS", OUTRO_MAX_WORDS):
            prompt = build_topic_user_prompt(seed, sources)

        maximum_before_outro = 165 - OUTRO_MAX_WORDS
        self.assertIn(f"ولا يتجاوز {maximum_before_outro} كلمة", prompt)
        self.assertNotRegex(prompt, r"narration_script\s+بين\s+\d+\s+و\s+\d+")


if __name__ == "__main__":
    unittest.main()
