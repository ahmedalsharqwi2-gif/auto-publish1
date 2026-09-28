import unittest
from unittest.mock import patch

from main import SourcePreflightError, choose_reachable_topic_seed, generate_topic


class ReachableTopicSelectionTests(unittest.TestCase):
    def setUp(self):
        self.blocked_url = "https://blocked.example/"
        self.working_url = "https://good.example/science/fact"
        self.working_source = {"url": self.working_url, "text": "evidence " * 60}
        self.topics = [
            {
                "id": "blocked_01",
                "subject": "Blocked fact",
                "verified_fact": "Blocked fact is true.",
                "category": "science",
                "source_urls": [self.blocked_url],
            },
            {
                "id": "blocked_02",
                "subject": "Blocked fact",
                "verified_fact": "Blocked fact is true.",
                "category": "science",
                "source_urls": [self.blocked_url],
            },
            {
                "id": "reachable_01",
                "subject": "Reachable fact",
                "verified_fact": "Reachable fact is true.",
                "category": "engineering",
                "source_urls": [self.working_url],
            },
        ]

    def _choose_first_seed_each_time(self):
        chooser = patch("main.random.SystemRandom")
        mocked = chooser.start()
        self.addCleanup(chooser.stop)
        mocked.return_value.choice.side_effect = lambda items: items[0]

    def test_skips_all_variants_of_an_unreachable_source_set(self):
        self._choose_first_seed_each_time()
        with (
            patch("main.load_topic_bank", return_value=self.topics),
            patch(
                "main.preflight_topic_sources",
                side_effect=[
                    ([], [{"url": self.blocked_url, "error": "403 Forbidden"}]),
                    ([self.working_source], []),
                ],
            ) as preflight,
        ):
            seed, sources, errors = choose_reachable_topic_seed([], set())

        self.assertEqual(seed["id"], "reachable_01")
        self.assertEqual(seed["source_urls"], [self.working_url])
        self.assertEqual(sources, [self.working_source])
        self.assertEqual(errors, [])
        self.assertEqual(preflight.call_count, 2)

    def test_returns_a_working_alternate_and_keeps_preflight_warning(self):
        self._choose_first_seed_each_time()
        alternate_url = "https://good.example/alternate"
        alternate_source = {"url": alternate_url, "text": "evidence " * 60}
        topic = dict(self.topics[0], source_urls=[self.blocked_url, alternate_url])
        with (
            patch("main.load_topic_bank", return_value=[topic]),
            patch(
                "main.preflight_topic_sources",
                return_value=(
                    [alternate_source],
                    [{"url": self.blocked_url, "error": "403 Forbidden"}],
                ),
            ),
        ):
            seed, sources, errors = choose_reachable_topic_seed([], set())

        self.assertEqual(seed["source_urls"], [alternate_url])
        self.assertEqual(sources, [alternate_source])
        self.assertEqual(errors[0]["url"], self.blocked_url)

    def test_raises_before_generation_if_every_source_set_is_unavailable(self):
        self._choose_first_seed_each_time()
        with (
            patch("main.load_topic_bank", return_value=self.topics[:2]),
            patch(
                "main.preflight_topic_sources",
                return_value=([], [{"url": self.blocked_url, "error": "403 Forbidden"}]),
            ) as preflight,
            self.assertRaises(SourcePreflightError),
        ):
            choose_reachable_topic_seed([], set())

        self.assertEqual(preflight.call_count, 1)

    def test_source_exhaustion_stops_before_calling_the_generation_model(self):
        with (
            patch("main.load_topic_history", return_value=[]),
            patch("main.choose_reachable_topic_seed", side_effect=SourcePreflightError("no sources")),
            patch("main._groq_chat") as groq,
            self.assertRaises(SourcePreflightError),
        ):
            generate_topic()

        groq.assert_not_called()


if __name__ == "__main__":
    unittest.main()
