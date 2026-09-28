import json
import unittest
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]


def allowed_url(url: str, domains: list[str]) -> bool:
    host = (urlparse(url).hostname or "").lower().lstrip(".")
    return any(host == domain or host.endswith(f".{domain}") for domain in domains)


def is_specific_source_url(url: str) -> bool:
    parsed = urlparse(url)
    path = parsed.path.rstrip("/")
    return bool(parsed.scheme == "https" and path and path not in {"/facts", "/news", "/articles"})


class TopicBankSourceCatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.topics = json.loads((ROOT / "config/topic_bank.json").read_text(encoding="utf-8"))["topics"]
        cls.domains = json.loads((ROOT / "config/fact_sources.json").read_text(encoding="utf-8"))["allowed_domains"]

    def test_all_topic_sources_are_specific_https_pages_on_allowed_hosts(self):
        self.assertTrue(self.topics)
        for topic in self.topics:
            with self.subTest(topic_id=topic.get("id")):
                self.assertTrue(topic.get("source_urls"))
                for url in topic["source_urls"]:
                    self.assertEqual(urlparse(url).scheme, "https")
                    self.assertTrue(allowed_url(url, self.domains), url)
                    self.assertTrue(is_specific_source_url(url), url)

    def test_specific_url_and_generic_index_url_are_distinguished(self):
        self.assertTrue(is_specific_source_url("https://oceanservice.noaa.gov/facts/coral_bleach.html"))
        self.assertFalse(is_specific_source_url("https://oceanservice.noaa.gov/facts"))


if __name__ == "__main__":
    unittest.main()
