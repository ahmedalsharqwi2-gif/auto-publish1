import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from arabic_guard import (
    assert_protected_untouched,
    load_protected,
    post_tashkeel,
    validate_narration,
)


class ArabicGuardTests(unittest.TestCase):
    def test_rejects_foreign_and_digits(self):
        kinds = {i.kind for i in validate_narration("في 14 أغسطس carrying a crew ループ")}
        self.assertIn("digit", kinds)
        self.assertIn("foreign_script", kinds)

    def test_accepts_clean_arabic(self):
        self.assertEqual(
            validate_narration("سجّلت ساعةٌ ذريةٌ نبضةً وصلت قبل موعدها."),
            [],
        )

    def test_tashkeel_safety_and_override(self):
        self.assertTrue(
            post_tashkeel(
                "تساءل الفريق",
                "تَسَاءُلِ الْفَرِيقِ",
            ).startswith("تَسَاءَلَ")
        )
        self.assertEqual(post_tashkeel("تساءل الفريق", "شيء آخر"), "تساءل الفريق")

    def test_protected_text_hash_and_exact_match(self):
        text = "إِنَّ مَعَ الْعُسْرِ يُسْرًا"
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "protected.json"
            path.write_text(
                json.dumps(
                    {"texts": [{"id": "test:1", "kind": "quran", "text": text, "sha256": digest}],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            protected = load_protected(str(path))
            assert_protected_untouched(
                [{"kind": "quran", "ref": "test:1", "text": text}],
                protected,
            )
            with self.assertRaises(ValueError):
                assert_protected_untouched(
                    [{"kind": "quran", "ref": "test:1", "text": text + "!"}],
                    protected,
                )


if __name__ == "__main__":
    unittest.main()
