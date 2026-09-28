import json
import unittest
from pathlib import Path
from urllib.parse import urlparse

from fact_check import _allowed_url


ROOT = Path(__file__).resolve().parents[1]


class TopicBankSourceCatalogTests(unittest.TestCase):
    def test_all_topic_sources_are_specific_https_pages_on_allowed_hosts(self):
        topics = json.loads((ROOT / "config/topic_bank.json").read_text(encoding="utf-8"))["topics"]
        domains = json.loads((ROOT / "config/fact_sources.json").read_text(encoding="utf-8"))["allowed_domains"]

        self.assertTrue(topics)
        for topic in topics:
            urls = topic.get("source_urls", [])
            with self.subTest(topic_id=topic.get("id")):
                self.assertTrue(urls, "topic must have at least one evidence source")
                for url in urls:
                    parsed = urlparse(url)
                    self.assertEqual(parsed.scheme, "https", url)
                    self.assertTrue(parsed.path not in ("", "/"), f"homepage citation: {url}")
                    self.assertTrue(_allowed_url(url, domains), f"source host is not allow-listed: {url}")


if __name__ == "__main__":
    unittest.main()
