import tempfile
import unittest
from pathlib import Path

from scripts.topic_history import (
    DuplicateTopicError,
    TopicHistory,
    TopicHistoryError,
    find_duplicate,
    load_history,
    normalize_text,
)


class TopicHistoryTests(unittest.TestCase):
    def test_arabic_diacritics_and_alef_variants_normalize(self):
        self.assertEqual(
            normalize_text("إِشَارَةُ أُورِيُونَ الغامضة"),
            normalize_text("اشاره اوريون الغامضه"),
        )

    def test_same_incident_with_reworded_title_is_detected(self):
        previous = [{
            "title": "لغز اختفاء سفينة ماري سيليست في المحيط الأطلسي",
            "hook": "حادثة اختفاء السفينة ماري سيليست بالمحيط الأطلسي",
        }]
        proposed = {
            "title": "حقيقة اختفاء سفينة ماري سيليست في المحيط الأطلسي",
            "hook": "حادثة اختفاء السفينة ماري سيليست بالمحيط الأطلسي",
        }
        self.assertIsNotNone(find_duplicate(proposed, previous))

    def test_generic_hook_or_caption_does_not_block_a_new_title(self):
        previous = [{
            "title": "كيف تتكون العواصف الشمسية",
            "hook": "قصة مذهلة تكشف سرًا غامضًا وتجيب عن سؤال مهم",
            "caption": "شاهد قصة مذهلة واكتشف الحقيقة",
        }]
        proposed = {
            "title": "كيف تتكون العواصف الرملية في الصحراء",
            "hook": "قصة علمية عن حركة الرياح وحبات الرمل",
        }
        self.assertIsNone(find_duplicate(proposed, previous))

    def test_unrelated_subject_is_not_blocked(self):
        previous = [{"title": "لغز اختفاء سفينة ماري سيليست في المحيط الأطلسي"}]
        proposed = {"title": "كيف يخزن الدماغ الذكريات أثناء النوم"}
        self.assertIsNone(find_duplicate(proposed, previous))

    def test_reservation_is_permanent_and_duplicate_reservation_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "topic_history.json"
            registry = TopicHistory(path)
            candidate = {
                "title": "كيف تتشكل الشفق القطبي فوق القطبين",
                "hook": "تصادم الجسيمات الشمسية مع الغلاف المغناطيسي للأرض",
            }
            reserved = registry.reserve(candidate)
            reloaded = TopicHistory(path)
            self.assertEqual(len(reloaded.entries), 1)
            self.assertEqual(reloaded.entries[0]["id"], reserved["id"])
            with self.assertRaises(DuplicateTopicError):
                reloaded.reserve(candidate)
            reloaded.mark_published(candidate)
            self.assertEqual(TopicHistory(path).entries[0]["status"], "published")

    def test_invalid_history_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "topic_history.json"
            path.write_text("not valid json", encoding="utf-8")
            with self.assertRaises(TopicHistoryError):
                load_history(path)


if __name__ == "__main__":
    unittest.main()


class ScienceTopicGenerationIntegrationTests(unittest.TestCase):
    def test_generator_retries_a_topic_found_in_permanent_history(self):
        from unittest.mock import patch
        from scripts.generate_content import ContentGenerator

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "topic_history.json"
            TopicHistory(path).reserve({"title": "العواصف الترابية على المريخ وعلاقتها بالغبار الكوكبي"})
            generator = ContentGenerator(topic_history_path=path)
            with patch(
                "scripts.generate_content.llm_chat",
                side_effect=[
                    "العواصف الترابية على المريخ وعلاقتها بالغبار الكوكبي",
                    "كيف يستخدم الحبار العملاق الضوء للتمويه في الأعماق",
                ],
            ) as llm:
                topic = generator.generate_topic()
            self.assertEqual(topic, "كيف يستخدم الحبار العملاق الضوء للتمويه في الأعماق")
            self.assertEqual(llm.call_count, 2)
