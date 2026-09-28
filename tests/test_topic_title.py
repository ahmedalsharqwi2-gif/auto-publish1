import json
import re
import unittest
from unittest.mock import patch

from main import (
    SYSTEM_PROMPT,
    PipelineError,
    Topic,
    build_topic_user_prompt,
    generate_topic,
    proofread_narration_tashkeel,
    set_canonical_topic_title,
)


class CanonicalTopicTitleTests(unittest.TestCase):
    def test_system_prompt_forbids_freeform_topic_and_unsupported_shock_details(self):
        self.assertIn("لا تختَر موضوعًا جديدًا", SYSTEM_PROMPT)
        self.assertIn("رقم أو مقارنة أو مفاجأة مصطنعة", SYSTEM_PROMPT)
        self.assertNotIn("مهمتك اختيار فكرة/موضوع", SYSTEM_PROMPT)

    def test_groq_receives_one_ready_seed_and_only_preflighted_sources(self):
        seed = {
            "id": "bat_01_02",
            "category": "عجائب عالم الحيوان",
            "subject": "تحديد الموقع بالصدى",
            "angle": "مقارنة يومية",
            "verified_fact": "تستخدم الخفافيش أصداء الأصوات لتحديد مواقع الأشياء.",
            "source_urls": ["https://blocked.example/root"],
        }
        accessible = [{"url": "https://example.org/direct-article", "text": "Evidence text"}]

        prompt = build_topic_user_prompt(seed, accessible)
        match = re.search(r"\{.*\}", prompt, re.DOTALL)
        self.assertIsNotNone(match)
        payload = json.loads(match.group(0))

        self.assertEqual(payload["subject"], seed["subject"])
        self.assertEqual(payload["verified_fact"], seed["verified_fact"])
        self.assertEqual(payload["category"], seed["category"])
        self.assertEqual(payload["source_urls"], [accessible[0]["url"]])
        self.assertNotIn("https://blocked.example/root", prompt)

    def test_prompt_refuses_to_draft_without_a_preflighted_source(self):
        seed = {
            "id": "x",
            "category": "فئة",
            "subject": "موضوع",
            "angle": "زاوية",
            "verified_fact": "حقيقة",
            "source_urls": ["https://example.org/article"],
        }
        with self.assertRaisesRegex(PipelineError, "preflighted source"):
            build_topic_user_prompt(seed, [])

    def test_proofreader_receives_the_canonical_subject_and_verified_fact(self):
        script = "حَقِيقَةٌ " * 100
        with patch(
            "main._groq_chat",
            return_value=json.dumps({"corrected_text": script}, ensure_ascii=False),
        ) as groq:
            result = proofread_narration_tashkeel(
                script,
                canonical_subject="موضوع البنك",
                verified_fact="حقيقة البنك الثابتة.",
            )

        payload = json.loads(groq.call_args.args[0][1]["content"])
        self.assertEqual(result, script.strip())
        self.assertEqual(payload["canonical_subject"], "موضوع البنك")
        self.assertEqual(payload["verified_fact"], "حقيقة البنك الثابتة.")

    def test_vetted_subject_replaces_unrelated_model_title(self):
        topic = Topic(
            hook_text="ما الذي يحدث؟",
            narration_script="نص عربي موثق.",
            title="سر طحالب القشرية في شعاب حارة",
            caption="شرح علمي.",
        )
        seed = {"subject": "التكافل بين المرجان والطحالب المجهرية"}

        set_canonical_topic_title(topic, seed)

        self.assertEqual(topic.title, seed["subject"])

    def test_generated_topic_keeps_bank_title_category_and_fact(self):
        seed = {
            "id": "bio_04_05",
            "category": "عجائب عالم الحيوان",
            "subject": "تحديد الموقع بالصدى",
            "angle": "زاوية مسجلة",
            "verified_fact": "تستخدم الخفافيش أصداء الأصوات لتحديد مواقع الأشياء.",
            "source_urls": ["https://example.org/direct-article"],
        }
        accessible = [{"url": seed["source_urls"][0], "text": "x" * 350}]
        hook = "كَيْفَ تَعْرِفُ الخَفَافِيشُ مَوْقِعَ الأَشْيَاءِ؟"
        model_output = {
            "category": "فئة اخترعها النموذج",
            "hook_text": hook,
            "narration_script": hook + " " + "شَرْحٌ " * 140,
            "title": "صدى الخفافيش: دقة السنتيمتر",
            "caption": "شرح موجز.",
            "hashtags": ["#خفافيش"],
            "search_keywords_en": "bat echolocation",
            "scene_keywords_en": ["bat in cave"] * 10,
        }
        with (
            patch("main.choose_reachable_topic_seed", return_value=(seed, accessible, [])),
            patch("main._groq_chat", return_value=json.dumps(model_output, ensure_ascii=False)),
            patch("main.load_topic_history", return_value=[]),
            patch("main.topic_is_too_similar", return_value=False),
        ):
            topic, sources, errors = generate_topic()

        self.assertEqual(topic.title, seed["subject"])
        self.assertEqual(topic.category, seed["category"])
        self.assertEqual(topic.verified_fact, seed["verified_fact"])
        self.assertEqual(topic.bank_id, seed["id"])
        self.assertEqual(sources, accessible)
        self.assertEqual(errors, [])

    def test_missing_canonical_subject_is_not_replaced_by_model_guess(self):
        topic = Topic(
            hook_text="ما الذي يحدث؟",
            narration_script="نص عربي موثق.",
            title="عنوان تخميني",
            caption="شرح علمي.",
        )

        with self.assertRaisesRegex(PipelineError, "canonical subject"):
            set_canonical_topic_title(topic, {})


if __name__ == "__main__":
    unittest.main()
