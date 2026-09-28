import unittest
from unittest.mock import patch

from main import (
    MIN_SCRIPT_WORDS,
    NARRATION_WORD_SAFETY_BUFFER,
    OUTRO_MIN_WORDS,
    POST_PROOFREAD_TOP_UP_ATTEMPTS,
    PipelineError,
    Topic,
    _top_up_underlength_narration,
)


class PostProofreadNarrationLengthTests(unittest.TestCase):
    @staticmethod
    def words(count):
        return "كلمة " * count

    @staticmethod
    def make_topic(script):
        return Topic(
            hook_text="ما هذا؟",
            narration_script=script,
            title="موضوع البنك",
            caption="شرح موثق.",
            verified_fact="حقيقة البنك المثبتة.",
            source_urls=["https://example.org/article"],
        )

    def test_short_reviewed_script_gets_bounded_source_bound_top_ups(self):
        minimum_before_outro = MIN_SCRIPT_WORDS - OUTRO_MIN_WORDS
        topic = self.make_topic(self.words(minimum_before_outro - 1))
        accessible_sources = ["https://example.org/direct-evidence"]
        expansions = [
            self.words(minimum_before_outro + 5),
            self.words(minimum_before_outro + 8),
        ]
        reviewed_versions = [
            self.words(minimum_before_outro - 1),
            self.words(minimum_before_outro + 1),
        ]

        with (
            patch("main.expand_narration_script", side_effect=expansions) as expand,
            patch("main._proofread_topic_narration", side_effect=reviewed_versions),
        ):
            result = _top_up_underlength_narration(topic, accessible_sources)

        self.assertEqual(len(result.split()), minimum_before_outro + 1)
        self.assertGreaterEqual(len(result.split()) + OUTRO_MIN_WORDS, MIN_SCRIPT_WORDS)
        self.assertEqual(expand.call_count, 2)
        self.assertEqual(expand.call_args.args[1], minimum_before_outro + NARRATION_WORD_SAFETY_BUFFER)
        self.assertEqual(expand.call_args.kwargs["verified_fact"], topic.verified_fact)
        self.assertEqual(expand.call_args.kwargs["source_urls"], accessible_sources)

    def test_top_up_exhaustion_fails_closed_after_bounded_attempts(self):
        minimum_before_outro = MIN_SCRIPT_WORDS - OUTRO_MIN_WORDS
        script = self.words(minimum_before_outro - 1)
        topic = self.make_topic(script)

        with (
            patch("main.expand_narration_script", return_value=script) as expand,
            self.assertRaisesRegex(PipelineError, "bounded top-up attempts"),
        ):
            _top_up_underlength_narration(topic, ["https://example.org/direct-evidence"])

        self.assertEqual(expand.call_count, POST_PROOFREAD_TOP_UP_ATTEMPTS)
        self.assertEqual(topic.narration_script, script)


if __name__ == "__main__":
    unittest.main()
