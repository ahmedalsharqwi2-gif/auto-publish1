import json
import unittest
from pathlib import Path
from urllib.parse import urlparse

from fact_check import _allowed_url, _is_specific_source_url


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
                    self.assertTrue(_is_specific_source_url(url), f"generic index/listing citation: {url}")

    def test_specific_coral_page_is_allowed_but_noaa_facts_index_is_not(self):
        self.assertTrue(_is_specific_source_url("https://oceanservice.noaa.gov/facts/coral_bleach.html"))
        self.assertFalse(_is_specific_source_url("https://oceanservice.noaa.gov/facts/"))

    def test_tardigrade_topic_uses_accessible_sources_and_supported_fact(self):
        topics = json.loads((ROOT / "config/topic_bank.json").read_text(encoding="utf-8"))["topics"]
        entries = [x for x in topics if x.get("seed_id") == "bio_01"]
        expected_sources = {
            "https://www.nsf.gov/news/how-do-microscopic-creatures-called-tardigrades-survive",
            "https://manoa.hawaii.edu/exploringourfluidearth/biological/what-alive/properties-life/weird-science-cryptobiosis",
        }
        self.assertEqual(len(entries), 10)
        self.assertTrue(all(set(x["source_urls"]) == expected_sources for x in entries))
        self.assertTrue(all("auth1.dpr.ncparks.gov" not in " ".join(x["source_urls"]) for x in entries))
        self.assertEqual(len({x["verified_fact"] for x in entries}), 1)

    def test_soap_bubble_article_url_is_specific_without_index_filename(self):
        topics = json.loads((ROOT / "config/topic_bank.json").read_text(encoding="utf-8"))["topics"]
        entries = [x for x in topics if x.get("seed_id") == "physics_06"]
        self.assertEqual(len(entries), 10)
        urls = {url for x in entries for url in x["source_urls"]}
        self.assertIn("https://micro.magnet.fsu.edu/primer/java/interference/soapbubbles/", urls)
        self.assertFalse(any(url.endswith("index.html") for url in urls))


if __name__ == "__main__":
    unittest.main()
