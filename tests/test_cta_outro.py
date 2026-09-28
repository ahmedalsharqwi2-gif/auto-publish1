import unittest

from main import (
    CTA_OUTRO,
    OUTRO_MAX_WORDS,
    OUTRO_MIN_WORDS,
    _append_engagement_outro,
    _find_unmarked_pronoun_suffixes,
)


EXPECTED_OUTRO = "إِذَا أَعْجَبَكَ الْفِيدْيُو، فَاضْغَطْ زِرَّ الإِعْجَابِ، وَلا تَنْسَ مُشَارَكَةَ الْفِيدْيُو"


class FixedCTAOutroTests(unittest.TestCase):
    def test_outro_matches_the_user_text_exactly(self):
        self.assertEqual(CTA_OUTRO, EXPECTED_OUTRO)

    def test_one_exact_outro_is_appended_and_word_budget_is_exact(self):
        script = "هَذَا نَصٌّ قَصِيرٌ."
        final = _append_engagement_outro(script)

        self.assertEqual(final, f"{script} {EXPECTED_OUTRO}")
        self.assertEqual(final.count(EXPECTED_OUTRO), 1)
        self.assertEqual(OUTRO_MIN_WORDS, len(EXPECTED_OUTRO.split()))
        self.assertEqual(OUTRO_MAX_WORDS, OUTRO_MIN_WORDS)

    def test_existing_exact_outro_is_not_duplicated(self):
        script = f"هَذَا نَصٌّ قَصِيرٌ.\n{EXPECTED_OUTRO}"
        self.assertEqual(_append_engagement_outro(script), script)

    def test_fixed_outro_passes_pronunciation_guard(self):
        self.assertEqual(_find_unmarked_pronoun_suffixes(EXPECTED_OUTRO), [])


if __name__ == "__main__":
    unittest.main()
