import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import requests

from fact_check import MAX_TOTAL_SOURCE_CHARS, FactCheckError, _groq_json, fact_check_topic, fetch_source, preflight_topic_sources


ROOT = Path(__file__).resolve().parents[1]


class FactCheckSourceFallbackTests(unittest.TestCase):
    def setUp(self):
        self.topic = {
            "title": "Test topic",
            "narration_script": "Arabic narration text",
            "verified_fact": "A supported scientific claim.",
            "source_urls": ["https://example.gov/unavailable", "https://example.gov/available"],
        }
        self.config = {"allowed_domains": ["example.gov"], "minimum_confidence": 0.85}

    def test_one_unavailable_source_does_not_discard_other_evidence(self):
        source = {"url": self.topic["source_urls"][1], "text": "official evidence"}
        verdict = {
            "claims": [{
                "claim": "A supported scientific claim.",
                "verdict": "supported",
                "confidence": 0.95,
                "evidence_quote": "official evidence",
                "source_url": source["url"],
                "reason": "Directly supported.",
            }],
            "overall_reason": "Supported by the accessible official source.",
        }
        with (
            patch("fact_check._load_config", return_value=self.config),
            patch("fact_check.fetch_source", side_effect=[requests.HTTPError("403 Forbidden"), source]),
            patch("fact_check._extract_claims", return_value=[{"claim": "A supported scientific claim."}]) as extract_claims,
            patch("fact_check._judge_claims", return_value=verdict),
        ):
            report = fact_check_topic(self.topic)

        self.assertEqual(report["status"], "PASS")
        self.assertIn(self.topic["title"], extract_claims.call_args.args[0])
        self.assertIn(self.topic["narration_script"], extract_claims.call_args.args[0])
        self.assertEqual(report["sources_fetched"], [source["url"]])
        self.assertEqual(len(report["source_fetch_errors"]), 1)
        self.assertEqual(report["errors"], [])

    def test_rejects_when_every_source_is_unavailable(self):
        with (
            patch("fact_check._load_config", return_value=self.config),
            patch("fact_check.fetch_source", side_effect=requests.HTTPError("403 Forbidden")),
            patch("fact_check._extract_claims") as extract_claims,
        ):
            report = fact_check_topic(self.topic)

        self.assertEqual(report["status"], "REJECT")
        self.assertEqual(report["sources_fetched"], [])
        self.assertEqual(len(report["source_fetch_errors"]), 2)
        self.assertIn("No accessible Fact Check sources", report["errors"][0])
        extract_claims.assert_not_called()

    def test_prefetched_evidence_is_reused_without_refetching(self):
        source = {
            "url": self.topic["source_urls"][1],
            "text": "official evidence " + ("supporting details " * 30),
        }
        verdict = {
            "claims": [{
                "claim": "A supported scientific claim.",
                "verdict": "supported",
                "confidence": 0.95,
                "evidence_quote": "official evidence",
                "source_url": source["url"],
                "reason": "Directly supported.",
            }],
            "overall_reason": "Supported by preflighted evidence.",
        }
        with (
            patch("fact_check._load_config", return_value=self.config),
            patch("fact_check.fetch_source") as fetch,
            patch("fact_check._extract_claims", return_value=[{"claim": "A supported scientific claim."}]),
            patch("fact_check._judge_claims", return_value=verdict),
        ):
            report = fact_check_topic(
                self.topic,
                prefetched_sources=[source],
                preflight_source_errors=[{"url": self.topic["source_urls"][0], "error": "403 Forbidden"}],
            )

        fetch.assert_not_called()
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["sources_fetched"], [source["url"]])
        self.assertEqual(len(report["source_fetch_errors"]), 1)

    def test_preflight_returns_usable_sources_and_structured_failures(self):
        good = {"url": self.topic["source_urls"][1], "text": "x" * 350}
        with (
            patch("fact_check._load_config", return_value=self.config),
            patch("fact_check.fetch_source", side_effect=[requests.HTTPError("403 Forbidden"), good]),
        ):
            sources, errors = preflight_topic_sources(self.topic["source_urls"])

        self.assertEqual(sources, [good])
        self.assertEqual(errors, [{"url": self.topic["source_urls"][0], "error": "403 Forbidden"}])

    def test_fact_check_payload_respects_total_source_character_budget(self):
        sources = [
            {"url": self.topic["source_urls"][0], "text": "A" * 7000},
            {"url": self.topic["source_urls"][1], "text": "B" * 7000},
        ]
        verdict = {
            "claims": [{
                "claim": "A supported scientific claim.",
                "verdict": "supported",
                "confidence": 0.95,
                "evidence_quote": "official evidence",
                "source_url": sources[0]["url"],
                "reason": "Directly supported.",
            }],
            "overall_reason": "Supported by the accessible sources.",
        }
        with (
            patch("fact_check._load_config", return_value=self.config),
            patch("fact_check.fetch_source", side_effect=sources),
            patch("fact_check._extract_claims", return_value=[{"claim": "A supported scientific claim."}]),
            patch("fact_check._judge_claims", return_value=verdict) as judge,
        ):
            report = fact_check_topic(self.topic)

        evidence = judge.call_args.args[3]
        self.assertLessEqual(sum(len(source["text"]) for source in evidence), MAX_TOTAL_SOURCE_CHARS)
        self.assertEqual(report["status"], "PASS")


class GroqRateLimitRetryTests(unittest.TestCase):
    def test_retries_transient_429_using_provider_reset_hint(self):
        limited = SimpleNamespace(
            status_code=429,
            headers={},
            text="Rate limit. Please try again in 9.5925s.",
        )
        success = SimpleNamespace(
            status_code=200,
            headers={},
            text="",
            json=lambda: {"choices": [{"message": {"content": '{"claims": []}'}}]},
        )
        with (
            patch("fact_check.GROQ_API_KEY", "test-key"),
            patch("fact_check.requests.post", side_effect=[limited, success]) as post,
            patch("fact_check.time.sleep") as sleep,
        ):
            result = _groq_json("system", "user", max_tokens=700)

        self.assertEqual(result, {"claims": []})
        self.assertEqual(post.call_count, 2)
        self.assertAlmostEqual(sleep.call_args.args[0], 10.5925)
        self.assertEqual(post.call_args_list[0].kwargs["json"]["max_tokens"], 700)

    def test_persistent_429_is_bounded_and_fails_closed(self):
        limited = SimpleNamespace(status_code=429, headers={"Retry-After": "0"}, text="still limited")
        with (
            patch("fact_check.GROQ_API_KEY", "test-key"),
            patch("fact_check.GROQ_RATE_LIMIT_MAX_RETRIES", 2),
            patch("fact_check.requests.post", return_value=limited) as post,
            patch("fact_check.time.sleep") as sleep,
            self.assertRaisesRegex(FactCheckError, "persisted after 2 retries"),
        ):
            _groq_json("system", "user")

        self.assertEqual(post.call_count, 3)
        self.assertEqual(sleep.call_count, 2)

    def test_generic_noaa_nasa_indexes_are_rejected_before_network_fetch(self):
        indexes = [
            "https://oceanservice.noaa.gov/facts/",
            "https://science.nasa.gov/earth/facts/",
            "https://science.nasa.gov/mars/facts/",
        ]
        with patch("fact_check.requests.get") as get:
            for url in indexes:
                with self.subTest(url=url), self.assertRaisesRegex(FactCheckError, "generic index"):
                    fetch_source(url, ["noaa.gov", "nasa.gov"])
        get.assert_not_called()


class SpecificNOAASourceTests(unittest.TestCase):
    def test_noaa_topics_use_specific_evidence_pages_not_homepage(self):
        data = json.loads((ROOT / "config/topic_bank.json").read_text(encoding="utf-8"))
        expected = {
            "earth_11_": "https://www.weather.gov/safety/lightning-science-overview",
            "earth_12_": "https://www.ncei.noaa.gov/news/planet-postcard-glacial-revelations",
            "eng_05_": "https://www.weather.gov/about/radar",
            "odd_04_": "https://www.nesdis.noaa.gov/about/k-12-education/optical-phenomena/what-causes-rainbow",
        }
        for prefix, url in expected.items():
            matches = [x for x in data["topics"] if x.get("id", "").startswith(prefix)]
            with self.subTest(topic_group=prefix):
                self.assertEqual(len(matches), 10)
                self.assertTrue(all(x.get("source_urls") == [url] for x in matches))

        lightning_comparison = next(x for x in data["topics"] if x.get("id") == "earth_11_02")
        self.assertEqual(lightning_comparison["angle"], "كيف يؤدي تسخين الهواء بالبرق إلى صوت الرعد؟")

        source_config = json.loads((ROOT / "config/fact_sources.json").read_text(encoding="utf-8"))
        self.assertIn("weather.gov", source_config["allowed_domains"])


if __name__ == "__main__":
    unittest.main()
