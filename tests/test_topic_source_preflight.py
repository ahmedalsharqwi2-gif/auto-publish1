import unittest
from pathlib import Path
from unittest.mock import patch

from main import (
    choose_voice_profile,
    load_voice_profile_ids,
    topic_is_too_similar,
    Topic,
)


ROOT = Path(__file__).resolve().parents[1]


class CurrentTopicSelectionTests(unittest.TestCase):
    def test_voice_profiles_load_from_current_config(self):
        profile_ids, default = load_voice_profile_ids(ROOT / "assets/voices/voice_profiles.json")
        self.assertIn(default, profile_ids)
        self.assertGreaterEqual(len(profile_ids), 1)

    def test_voice_rotation_selects_next_profile(self):
        with patch("main.load_topic_history", return_value=[{"voice_profile": "mohamed_elbed"}]), \
             patch("main.VOICE_ROTATION_ENABLED", True):
            selected = choose_voice_profile()
        self.assertNotEqual(selected, "mohamed_elbed")

    def test_topic_similarity_detects_repeated_topic(self):
        topic = Topic(
            hook_text="ما هذه الحقيقة؟",
            narration_script="نص عربي",
            title="موضوع مكرر",
            caption="شرح",
            search_keywords_en="repeated topic",
        )
        history = [{
            "title": "موضوع مكرر",
            "hook_text": "ما هذه الحقيقة؟",
            "search_keywords_en": "repeated topic",
        }]
        self.assertTrue(topic_is_too_similar(topic, history))

    def test_topic_similarity_allows_distinct_topic(self):
        topic = Topic(
            hook_text="كيف يعمل البرق؟",
            narration_script="نص عربي",
            title="البرق",
            caption="شرح",
            search_keywords_en="lightning science",
        )
        self.assertFalse(topic_is_too_similar(topic, [{
            "title": "حياة السلاحف",
            "hook_text": "كيف تعيش السلاحف؟",
            "search_keywords_en": "turtle biology",
        }]))


if __name__ == "__main__":
    unittest.main()
