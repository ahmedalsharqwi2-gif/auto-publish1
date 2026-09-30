from __future__ import annotations
from dataclasses import dataclass
import re
import unicodedata

DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED\u08D3-\u08FF]")
TATWEEL = "ـ"
ARABIC = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")
WORD = re.compile(r"\S+")

@dataclass(frozen=True)
class PreparedText:
    display_text: str
    align_text: str
    display_words: tuple[str, ...]
    align_words: tuple[str, ...]
    span_map: tuple[tuple[int, ...], ...]


def strip_diacritics(text: str) -> str:
    return DIACRITICS.sub("", text)


def normalize_arabic(text: str, *, keep_diacritics: bool = True) -> str:
    text = unicodedata.normalize("NFC", str(text or "")).replace(TATWEEL, "")
    text = re.sub(r"\s+", " ", text).strip()
    if not keep_diacritics:
        text = strip_diacritics(text)
    return text


def _align_word(word: str) -> str:
    word = strip_diacritics(word)
    word = word.translate(str.maketrans({"أ":"ا", "إ":"ا", "آ":"ا", "ى":"ي"}))
    return re.sub(r"^[،؛,:.!؟?]+|[،؛,:.!؟?]+$", "", word).strip()


def prepare_text(text: str) -> PreparedText:
    display = normalize_arabic(text, keep_diacritics=True)
    display_words = tuple(WORD.findall(display))
    align_words = tuple(_align_word(w) for w in display_words)
    align_words = tuple(w for w in align_words if w)
    # One display word maps to one alignment word after deterministic cleanup.
    span_map = tuple((i,) for i, w in enumerate(display_words) if _align_word(w))
    return PreparedText(display, " ".join(align_words), display_words, align_words, span_map)


def arabic_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    return sum(bool(ARABIC.match(c)) for c in letters) / max(1, len(letters))
