import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fact_check import (
    MAX_TOTAL_SOURCE_CHARS,
    FactCheckError,
    _extract_claims,
    _json_from_model,
    _judge_claims,
    fact_check_topic,
)


class FactCheckTests(unittest.TestCase):
    def setUp(self):
        self.topic = {
            "title": "موضوع اختباري",
            "caption": "شرح مختصر",
            "narration_script": "هذه حقيقة علمية موثقة.",
            "verified_fact": "هذه حقيقة علمية موثقة.",
        }
        self.source = {
            "url": "https://example.org/article",
            "text": "دليل موثق " * 80,
        }

    def test_json_parser_accepts_fenced_model_output(self):
        parsed = _json_from_model("```json\n{\"claims\": []}\n```")
        self.assertEqual(parsed, {"claims": []})

    def test_json_parser_rejects_missing_json(self):
        with self.assertRaises(FactCheckError):
            _json_from_model("not json")

    def test_fact_check_reuses_prefetched_sources(self):
        verdict = {
            "claims": [{
                "claim": "حقيقة",
                "verdict": "supported",
                "confidence": 0.95,
                "evidence_quote": "دليل موثق",
                "source_url": self.source["url"],
                "reason": "مدعوم بالمصدر",
            }],
            "overall_reason": "مدعوم",
        }
        with (
            patch("fact_check._load_config", return_value={"minimum_confidence": 0.85}),
            patch("fact_check._extract_claims", return_value=[{"claim": "حقيقة"}]),
            patch("fact_check._judge_claims", return_value=verdict),
            patch("fact_check.fetch_web_sources") as fetch,
        ):
            report = fact_check_topic(self.topic, prefetched_sources=[self.source])

        fetch.assert_not_called()
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["sources_fetched"], [self.source["url"]])

    def test_fact_check_rejects_when_no_source_is_available(self):
        topic = dict(self.topic, verified_fact="")
        with (
            patch("fact_check._load_config", return_value={"minimum_confidence": 0.85}),
            patch("fact_check.fetch_web_sources", return_value=[]),
        ):
            report = fact_check_topic(topic)

        self.assertEqual(report["status"], "REJECT")
        self.assertTrue(any("No source available" in error for error in report["errors"]))

    def test_fact_check_limits_total_source_characters(self):
        large_sources = [
            {"url": "https://example.org/one", "text": "A" * 7000},
            {"url": "https://example.org/two", "text": "B" * 7000},
        ]
        verdict = {
            "claims": [{
                "claim": "حقيقة",
                "verdict": "supported",
                "confidence": 0.95,
                "evidence_quote": "A",
                "source_url": large_sources[0]["url"],
                "reason": "مدعوم",
            }],
        }
        with (
            patch("fact_check._load_config", return_value={"minimum_confidence": 0.85}),
            patch("fact_check._extract_claims", return_value=[{"claim": "حقيقة"}]),
            patch("fact_check._judge_claims", return_value=verdict) as judge,
        ):
            report = fact_check_topic(self.topic, prefetched_sources=large_sources)

        evidence = judge.call_args.args[2]
        self.assertLessEqual(sum(len(source["text"]) for source in evidence), MAX_TOTAL_SOURCE_CHARS)
        self.assertEqual(report["status"], "PASS")

    def test_claim_extraction_normalizes_model_claims(self):
        with patch(
            "fact_check._llm_call",
            return_value={"claims": [{"claim": " ادعاء ", "importance": "bad", "numeric": 1}]},
        ):
            claims = _extract_claims("نص")
        self.assertEqual(claims, [{"claim": "ادعاء", "importance": "core", "numeric": True}])

    def test_judge_claims_passes_source_evidence_to_llm(self):
        with patch("fact_check._llm_call", return_value={"claims": []}) as call:
            _judge_claims("نص", [{"claim": "ادعاء"}], [self.source])
        payload = json.loads(call.call_args.args[1])
        self.assertIn(self.source["url"], payload["sources"])


if __name__ == "__main__":
    unittest.main()
