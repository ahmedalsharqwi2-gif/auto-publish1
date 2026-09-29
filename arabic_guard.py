# -*- coding: utf-8 -*-
"""Arabic narration quality gate used before TTS generation."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass

FOREIGN = re.compile(
    r"[A-Za-z\u00C0-\u024F\u0400-\u04FF\u0590-\u05FF"
    r"\u3040-\u30FF\u4E00-\u9FFF\uAC00-\uD7AF]"
)
DIGITS = re.compile(r"[0-9\u0660-\u0669]")
HARAKAT = re.compile(r"[\u064B-\u0652\u0670]")
TATWEEL = "\u0640"
ARABIC_LETTER = re.compile(r"[\u0621-\u064A]")


@dataclass(frozen=True)
class Issue:
    kind: str
    sample: str


def strip_marks(s: str) -> str:
    """Return Arabic text without tashkeel or tatweel."""
    return HARAKAT.sub("", s).replace(TATWEEL, "")


def validate_narration(text: str, min_arabic_ratio: float = 0.85) -> list[Issue]:
    """Validate narration text immediately before it reaches TTS."""
    issues: list[Issue] = []

    for m in FOREIGN.finditer(text):
        start = m.start()
        end = start
        while end < len(text) and FOREIGN.match(text[end]):
            end += 1
        issues.append(Issue("foreign_script", text[start:end]))

    issues.extend(Issue("digit", m.group()) for m in DIGITS.finditer(text))

    letters = [c for c in text if c.isalpha()]
    if letters:
        ratio = sum(bool(ARABIC_LETTER.match(c)) for c in letters) / len(letters)
        if ratio < min_arabic_ratio:
            issues.append(Issue("low_arabic_ratio", f"{ratio:.0%}"))

    seen: set[tuple[str, str]] = set()
    unique: list[Issue] = []
    for issue in issues:
        key = (issue.kind, issue.sample)
        if key not in seen:
            seen.add(key)
            unique.append(issue)
    return unique


def format_feedback(issues: list[Issue]) -> str:
    lines = ["أعد كتابة النص بالعربية الفصحى فقط، وصحّح ما يلي:"]
    for issue in issues[:12]:
        lines.append(f"- {issue.kind}: {issue.sample}")
    lines.append("اكتب الأعداد بالحروف، وترجم أي عبارة أجنبية أو احذفها.")
    return "\n".join(lines)


TASHKEEL_OVERRIDES = {
    "تساءل": "تَسَاءَلَ",
    "سجل": None,
}


def post_tashkeel(original: str, diacritized: str) -> str:
    """Accept tashkeel only when the underlying letters are unchanged."""
    if strip_marks(diacritized) != strip_marks(original):
        return original

    out: list[str] = []
    for word in diacritized.split():
        bare = strip_marks(word)
        fix = TASHKEEL_OVERRIDES.get(bare)
        out.append(fix if fix else word)
    return " ".join(out)


def _norm(s: str) -> str:
    return unicodedata.normalize("NFC", s).strip()


def load_protected(path: str) -> dict[str, str]:
    """Load source-verified Quran/Hadith text and verify its SHA-256."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    result: dict[str, str] = {}
    for item in data["texts"]:
        digest = hashlib.sha256(_norm(item["text"]).encode("utf-8")).hexdigest()
        if digest != item["sha256"]:
            raise ValueError(f"النص الشرعي {item['id']} تغيّر عن المصدر المعتمد")
        result[item["id"]] = _norm(item["text"])
    return result


def assert_protected_untouched(
    script_segments: list[dict], protected: dict[str, str]
) -> None:
    """Require every Quran/Hadith segment to match its verified source exactly."""
    for segment in script_segments:
        if segment.get("kind") in ("quran", "hadith"):
            ref = segment["ref"]
            if ref not in protected:
                raise ValueError(f"النص الشرعي {ref} غير موجود في القائمة المحمية")
            if _norm(segment["text"]) != protected[ref]:
                raise ValueError(f"تعديل غير مسموح في {ref}")
