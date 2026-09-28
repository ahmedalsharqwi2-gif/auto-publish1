import json
import unittest
from unittest.mock import patch

from main import (
    CHANNEL_BRIEF,
    MIN_SCRIPT_WORDS,
    OUTRO_MIN_WORDS,
    PipelineError,
    SYSTEM_PROMPT,
    TOPIC_CATEGORIES,
    Topic,
    _proofread_topic_narration,
    generate_topic,
    proofread_narration_tashkeel,
)


class CurrentTopicAndEditorialTests(unittest.TestCase):
    def test_system_prompt_requires_json_and_arabic_content(self):
        self.assertIn("أخرج JSON صالحًا", SYSTEM_PROMPT)
        self.assertIn("narration_script", SYSTEM_PROMPT)
        self.assertIn("scene_keywords_en", SYSTEM_PROMPT)

    def test_generation_policy_matches_channel_and_excludes_unverifiable_topics(self):
        self.assertIn("أسرار الفضاء", CHANNEL_BRIEF)
        self.assertIn("أعماق المحيطات", CHANNEL_BRIEF)
        self.assertIn(CHANNEL_BRIEF, SYSTEM_PROMPT)
        self.assertNotIn("غرائب دينية موثقة", TOPIC_CATEGORIES)
        self.assertIn("ممنوع المعجزات", SYSTEM_PROMPT)

    def test_proofreader_uses_current_llm_chat_and_preserves_content(self):
        script = "حَقِيقَةٌ " * 60
        with patch(
            "main.llm_chat",
            return_value=json.dumps({"corrected_text": script}, ensure_ascii=False),
        ) as llm:
            result = proofread_narration_tashkeel(
                script,
                canonical_subject="موضوع",
                verified_fact="حقيقة موثقة",
            )
        self.assertEqual(result, script.strip())
        payload = json.loads(llm.call_args.args[0][1]["content"])
        self.assertEqual(payload["canonical_subject"], "موضوع")
        self.assertEqual(payload["verified_fact"], "حقيقة موثقة")

    def test_proofreader_rejects_major_content_deletion(self):
        script = "حَقِيقَةٌ " * 100
        with (
            patch(
                "main.llm_chat",
                return_value=json.dumps({"corrected_text": "حَقِيقَةٌ " * 10}, ensure_ascii=False),
            ),
            self.assertRaisesRegex(PipelineError, "changed script length too much"),
        ):
            proofread_narration_tashkeel(script)

    def test_short_editorial_rewrite_keeps_original_minimum_length_script(self):
        original = "حَقِيقَةٌ " * (MIN_SCRIPT_WORDS - OUTRO_MIN_WORDS)
        shortened = "حَقِيقَةٌ " * 90
        topic = Topic(
            hook_text="مَا هَذِهِ الحَقِيقَةُ؟",
            narration_script=original,
            title="موضوع",
            caption="شرح",
        )
        with (
            patch("main.proofread_narration_tashkeel", return_value=shortened),
            patch("main.enforce_text_quality", side_effect=lambda text, reviser: text),
        ):
            result = _proofread_topic_narration(topic, original)
        self.assertEqual(result, original)

    def test_generate_topic_normalizes_model_output(self):
        hook = "كَيْفَ يَعْمَلُ البَرْقُ؟"
        model_output = {
            "category": "حقائق علمية صادمة",
            "hook_text": hook,
            "narration_script": hook + " " + "شَرْحٌ " * 125,
            "title": "البرق",
            "caption": "شرح علمي موجز.",
            "hashtags": ["#علم"],
            "search_keywords_en": "lightning science",
            "scene_keywords_en": ["lightning storm"] * 10,
        }
        with (
            patch("main.llm_chat", return_value=json.dumps(model_output, ensure_ascii=False)),
            patch("main.load_topic_history", return_value=[]),
            patch("main.topic_is_too_similar", return_value=False),
            patch("main.TOPIC_GENERATION_MAX_ATTEMPTS", 1),
        ):
            topic, sources, errors = generate_topic()

        self.assertIsInstance(topic, Topic)
        self.assertEqual(topic.title, "البرق")
        self.assertEqual(topic.category, "حقائق علمية صادمة")
        self.assertEqual(sources, [])
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
