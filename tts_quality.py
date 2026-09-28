"""
tts_quality.py — حراسة الجودة لمسار الصوت (SILMA) في خط الإنتاج
==================================================================
يضع هذا الملف بجانب main.py في جذر المستودع. يحتوي على ثلاث طبقات:

1) نظافة النص قبل النطق:  sanitize_for_tts / fix_tashkeel_anomalies /
   find_text_problems / enforce_text_quality
2) تحميل المرجع الصوتي مع نصه المطابق تمامًا:  load_reference
3) توليد SILMA مع فحص المخرَج (تسريب المرجع، كلام مشوّه/ناقص) وإعادة
   المحاولة بـ seed مختلف:  generate_silma_guarded

لا يعتمد على أي شيء من main.py (لا استيراد دائري).
"""

from __future__ import annotations

import difflib
import logging
import os
import re
import subprocess
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable

log = logging.getLogger("pipeline")

# ---------------------------------------------------------------------------
# ثوابت التشكيل
# ---------------------------------------------------------------------------
_SHADDA = "\u0651"
_SUKUN = "\u0652"
_DAGGER = "\u0670"                      # ألف خنجرية
_VOWELS = "\u064B\u064C\u064D\u064E\u064F\u0650"   # تنوين فتح/ضم/كسر + فتحة/ضمة/كسرة
_TASHKEEL_RE = re.compile("[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")

MAX_SENTENCE_WORDS = int(os.getenv("MAX_SENTENCE_WORDS", "30"))
MIN_TASHKEEL_COVERAGE_WARN = float(os.getenv("MIN_TASHKEEL_COVERAGE_WARN", "0.60"))
MIN_TASHKEEL_COVERAGE_FATAL = float(os.getenv("MIN_TASHKEEL_COVERAGE_FATAL", "0.35"))

# تراكيب ظهرت في مخرجات المراجعة الآلية وتدل غالبًا على خلل نحوي واضح.
# تُسجّل كأخطاء حرجة حتى تُعاد مراجعتها قبل التسجيل أو النشر.
_LANGUAGE_RED_FLAGS = (
    (re.compile(r"\bإذا\s+غير\s+موجود(?:ة)?(?:\s+(?:ماء|شيء|سبب))?\b"),
     "تركيب غير سليم: استخدم (إذا لم يوجد/توجد ...) بدل (إذا غير موجود ...)"),
    (re.compile(r"(?:^|[،\s])رفع\s+العمود\b"),
     "تركيب غير سليم في السياق: استخدم (ارتفاع العمود) لا (رفع العمود)"),
    (re.compile(r"\bإذا\s+غير\s+موجود\s+ماء\b"),
     "تركيب غير سليم: استخدم (إذا لم يوجد ماء)"),
)


class TextQualityError(Exception):
    """النص ما زال معيبًا بعد المراجعة — يُفضَّل إيقاف التشغيل على نشر نص سيئ."""


class SilmaQualityError(Exception):
    """لم ينجح أي مرشّح صوتي في اجتياز فحص الجودة."""


def is_arabic_letter(ch: str) -> bool:
    return "\u0621" <= ch <= "\u064A" or ch == "\u0671"


def _is_mark(ch: str) -> bool:
    return unicodedata.category(ch) == "Mn"


# ---------------------------------------------------------------------------
# 1) نظافة النص
# ---------------------------------------------------------------------------
def _fold_presentation_forms(text: str) -> str:
    """بعض النماذج تُخرج أشكال العرض العربية (U+FB50–U+FEFF)؛ نحوّلها لحروفها
    العادية فقط، دون المساس بترتيب علامات التشكيل في بقية النص."""
    return "".join(
        unicodedata.normalize("NFKC", c) if "\uFB50" <= c <= "\uFEFF" else c for c in text
    )


def sanitize_for_tts(text: str) -> tuple[str, int]:
    """ينظّف الرموز التي تُربك محرك النطق. يُرجع (النص، عدد التعديلات التقريبي)."""
    original = text
    text = _fold_presentation_forms(text)
    text = text.replace("\u0640", "")                                         # تطويل
    text = re.sub("[\u200b-\u200f\u202a-\u202e\u2066-\u2069\ufeff]", "", text)  # محارف خفية
    text = text.replace("\u2026", ".").replace("...", ".")
    text = text.replace(",", "،").replace(";", "؛").replace("?", "؟")
    text = re.sub(r"\s[-\u2013\u2014]+\s", "، ", text)                         # شرطة بين كلمتين
    keep = "،.؟!؛:"
    text = "".join(
        ch if (ch.isalnum() or _is_mark(ch) or ch.isspace() or ch in keep) else " "
        for ch in text
    )
    text = re.sub(r"\s+([،.؟!؛:])", r"\1", text)      # لا مسافة قبل علامة الترقيم
    text = re.sub(r"([،.؟!؛:])\1+", r"\1", text)      # علامة مكررة
    text = re.sub(r"\s+", " ", text).strip()
    changed = sum(1 for a, b in zip(original, text) if a != b) + abs(len(original) - len(text))
    return text, changed


def fix_tashkeel_anomalies(text: str) -> tuple[str, int]:
    """يُصلح توليفات التشكيل غير الصالحة التي قد تُنتج نطقًا مشوّهًا:
    - علامات معلّقة على غير حرف عربي (مسافة/بداية النص/ترقيم)
    - حركتان مختلفتان أو أكثر على حرف واحد (تُبقى الأولى)
    - سكون مع حركة، أو سكون مع شدّة (يُحذف السكون)
    - تكرار العلامة نفسها
    لا يعيد ترتيب الشدّة/الحركة ولا يغيّر أي حرف. الإصلاح استدلالي
    (لا يُصحّح الإعراب نفسه — هذا عمل خطوة المراجعة اللغوية)."""
    out: list[str] = []
    fixes = 0
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if not _is_mark(ch):
            out.append(ch)
            i += 1
            continue
        j = i
        while j < n and _is_mark(text[j]):
            j += 1
        run = text[i:j]
        i = j
        if not (out and is_arabic_letter(out[-1])):
            fixes += len(run)
            continue
        real_vowels = [m for m in run if m in _VOWELS]
        allowed = set()
        if _SHADDA in run:
            allowed.add(_SHADDA)
        if real_vowels:
            allowed.add(real_vowels[0])
        elif _SUKUN in run and _SHADDA not in run:
            allowed.add(_SUKUN)
        if _DAGGER in run:
            allowed.add(_DAGGER)
        seen: set[str] = set()
        kept: list[str] = []
        for m in run:
            is_core = m in _VOWELS or m in (_SHADDA, _SUKUN, _DAGGER)
            if is_core and (m not in allowed or m in seen):
                continue
            seen.add(m)
            kept.append(m)
        fixes += len(run) - len(kept)
        out.extend(kept)
    return "".join(out), fixes


def find_text_problems(text: str) -> tuple[list[str], list[str]]:
    """يُرجع (أخطاء_حرجة, تحذيرات). الحرجة تستدعي إعادة مراجعة النص."""
    fatal: list[str] = []
    warn: list[str] = []
    plain = _TASHKEEL_RE.sub("", text)

    foreign = sorted({c for c in plain if c.isalpha() and not is_arabic_letter(c)})
    if foreign:
        fatal.append("يوجد حروف غير عربية داخل النص: " + " ".join(foreign[:10]))
    for pattern, message in _LANGUAGE_RED_FLAGS:
        if pattern.search(plain):
            fatal.append(message)

    tokens = [w.strip("،.؟!؛:") for w in plain.split()]
    tokens = [t for t in tokens if t]
    run = 1
    for a, b in zip(tokens, tokens[1:]):
        run = run + 1 if a == b else 1
        if run >= 3:
            fatal.append(f"تكرار متتالٍ للكلمة نفسها ({a})")
            break
    if re.search(r"([\u0621-\u064A])\1{3,}", plain):
        fatal.append("تكرار غير طبيعي لحرف واحد عدة مرات متتالية")
    long_words = [t for t in tokens if len(t) > 20]
    if long_words:
        fatal.append("كلمات طويلة بشكل مريب (على الأرجح مدموجة أو غير صالحة): " + " ".join(long_words[:3]))

    if re.search(r"[0-9\u0660-\u0669\u06F0-\u06F9]", plain):
        warn.append("يوجد أرقام؛ الأفضل كتابتها بالحروف لضبط النطق والمدة")

    for sentence in re.split(r"[.؟!؛:]+", plain):
        if len(sentence.split()) > MAX_SENTENCE_WORDS:
            warn.append(f"جملة أطول من {MAX_SENTENCE_WORDS} كلمة بلا وقف؛ قد يختلّ الإيقاع")
            break

    words = text.split()
    counted = [w for w in words if sum(is_arabic_letter(c) for c in w) >= 3]
    if counted:
        marked = sum(1 for w in counted if any(_is_mark(c) for c in w))
        coverage = marked / len(counted)
        if coverage < MIN_TASHKEEL_COVERAGE_FATAL:
            fatal.append(f"التشكيل ناقص جدًا ({coverage:.0%} من الكلمات مشكولة)")
        elif coverage < MIN_TASHKEEL_COVERAGE_WARN:
            warn.append(f"التشكيل ناقص ({coverage:.0%} من الكلمات مشكولة)")
    return fatal, warn


def enforce_text_quality(
    script: str, reviser: Callable[[str, list[str]], str] | None = None
) -> str:
    """ينظّف النص، يُصلح شذوذ التشكيل، ثم يفحصه. عند وجود أخطاء حرجة يستدعي
    reviser(النص، قائمة_المشكلات) مرة واحدة (مراجعة لغوية موجَّهة)، ثم يعيد
    الفحص. إن بقي خطأ حرج يرفع TextQualityError."""
    text, n1 = sanitize_for_tts(script)
    text, n2 = fix_tashkeel_anomalies(text)
    if n1 or n2:
        log.info("Text hygiene: %d symbol edit(s), %d tashkeel fix(es)", n1, n2)
    fatal, warn = find_text_problems(text)
    if fatal and reviser is not None:
        log.warning("Narration text has problems, requesting a targeted revision: %s", fatal)
        text = reviser(text, fatal)
        text, _ = sanitize_for_tts(text)
        text, _ = fix_tashkeel_anomalies(text)
        fatal, warn = find_text_problems(text)
    if fatal:
        raise TextQualityError("Narration failed text-quality checks: " + " | ".join(fatal))
    for w in warn:
        log.warning("Narration text warning: %s", w)
    return text


# ---------------------------------------------------------------------------
# مقارنة الكلمات (تتجاهل التشكيل وفروق الهمزات)
# ---------------------------------------------------------------------------
def _canon(word: str) -> str:
    w = _TASHKEEL_RE.sub("", word).replace("\u0640", "")
    w = re.sub("[أإآٱ]", "ا", w)
    w = w.replace("ى", "ي").replace("ة", "ه").replace("ؤ", "و").replace("ئ", "ي")
    return w.strip("،.؟!؛:?,!\"' ").lower()


def normalize_words(text: str) -> list[str]:
    return [c for c in (_canon(w) for w in text.split()) if c]


def media_duration(path: Path) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(r.stdout.strip())


# ---------------------------------------------------------------------------
# 2) المرجع الصوتي + نصه
# ---------------------------------------------------------------------------
REF_MAX_SECONDS = float(os.getenv("SILMA_REF_MAX_SECONDS", "8.5"))
REF_MIN_SECONDS = float(os.getenv("SILMA_REF_MIN_SECONDS", "3.0"))


def load_reference(wav: Path, fallback_text: str = "") -> tuple[Path, str]:
    """يحمّل المرجع ونصه من ملف .txt المجاور له (نفس الاسم). هذا الملف يكتبه
    make_reference_voice.py من النص نفسه الذي وُلِّد منه الصوت، فلا يمكن أن
    يختلف عنه. متغير SILMA_REFERENCE_TEXT مجرد احتياط قديم."""
    wav = wav if wav.is_absolute() else Path.cwd() / wav
    if not wav.exists():
        raise FileNotFoundError(f"SILMA reference audio is missing: {wav}")
    sidecar = wav.with_suffix(".txt")
    if sidecar.exists():
        text = sidecar.read_text(encoding="utf-8").strip()
    else:
        text = (fallback_text or "").strip()
        log.warning("No %s found; falling back to SILMA_REFERENCE_TEXT (risk of text/audio mismatch)", sidecar.name)
    if not text:
        raise ValueError(f"SILMA reference text is empty (expected {sidecar})")
    dur = media_duration(wav)
    if dur > REF_MAX_SECONDS:
        log.warning("SILMA reference is %.1fs (> %.1fs); long references raise leakage/distortion risk", dur, REF_MAX_SECONDS)
    elif dur < REF_MIN_SECONDS:
        log.warning("SILMA reference is only %.1fs (< %.1fs); too short to clone the voice reliably", dur, REF_MIN_SECONDS)
    return wav, text


# ---------------------------------------------------------------------------
# 3) فحص المخرَج الصوتي
# ---------------------------------------------------------------------------
@dataclass
class VoiceCheck:
    score: float
    ok: bool
    leaked: str | None
    words_per_sec: float
    avg_logprob: float | None
    note: str


@lru_cache(maxsize=1)
def _whisper_model():
    from faster_whisper import WhisperModel  # اعتماد اختياري ثقيل
    return WhisperModel(os.getenv("WHISPER_MODEL", "base"), device="cpu", compute_type="int8")


def transcribe_words(audio_path: Path) -> tuple[list[str], float | None]:
    segments, _ = _whisper_model().transcribe(
        str(audio_path), language="ar", word_timestamps=False, vad_filter=False,
        condition_on_previous_text=False,   # يقلّل حلقات الهلوسة في التفريغ الطويل
    )
    segs = list(segments)
    words = normalize_words(" ".join(s.text or "" for s in segs))
    lps = [s.avg_logprob for s in segs if getattr(s, "avg_logprob", None) is not None]
    return words, (sum(lps) / len(lps) if lps else None)


def _similarity(a: list[str], b: list[str]) -> float:
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def detect_reference_leak(heard: list[str], script: list[str], ref: list[str], threshold: float = 0.6) -> str | None:
    """تسريب = ظهور كلمات المرجع في الصوت الناتج وهي ليست في السكريبت.
    يفحص بداية الصوت ونهايته (حيث يحدث التسريب عادة) بتشابه تقريبي، ثم يبحث
    عن أي ثلاث كلمات متتالية من المرجع لا يحتويها السكريبت نفسه، وأخيرًا عن
    أي زوج كلمات من المرجع عند الحافتين ليس في السكريبت — فلا يُرفض
    فيديو لمجرد أن موضوعه يذكر عبارة تشبه نص المرجع."""
    if len(ref) < 3 or not heard:
        return None
    k = len(ref) + 2
    for name, h, s in (("start", heard[:k], script[:k]), ("end", heard[-k:], script[-k:])):
        if _similarity(h, ref) >= threshold and _similarity(s, ref) < 0.4:
            return f"{name}: " + " ".join(h)
    # تسريب جزئي: زوج كلمات من المرجع عند حافّتَي الصوت وليس في السكريبت
    ref_bi = {tuple(ref[i:i + 2]) for i in range(len(ref) - 1)}
    script_bi = {tuple(script[i:i + 2]) for i in range(len(script) - 1)}
    for name, h in (("start", heard[:k]), ("end", heard[-k:])):
        for i in range(len(h) - 1):
            bi = tuple(h[i:i + 2])
            if bi in ref_bi and bi not in script_bi:
                return f"{name}: " + " ".join(bi)
    script_tri = {tuple(script[i:i + 3]) for i in range(len(script) - 2)}
    heard_tri = {tuple(heard[i:i + 3]) for i in range(len(heard) - 2)}
    for i in range(len(ref) - 2):
        tri = tuple(ref[i:i + 3])
        if tri not in script_tri and tri in heard_tri:
            return " ".join(tri)
    return None


def evaluate_tts_audio(audio_path: Path, script_text: str, ref_text: str, min_score: float) -> VoiceCheck:
    script = normalize_words(script_text)
    ref = normalize_words(ref_text)
    heard, avg_lp = transcribe_words(audio_path)
    dur = media_duration(audio_path)
    wps = len(script) / dur if dur > 0 else 0.0

    if not heard:
        return VoiceCheck(0.0, False, None, wps, avg_lp, "no speech recognized (silence or fully garbled audio)")

    coverage = _similarity(script, heard)          # كلمات ناقصة/زائدة/مبدَّلة كلها تخفضه
    leak = detect_reference_leak(heard, script, ref)
    notes: list[str] = []
    score = coverage
    if leak:
        score = 0.0
        notes.append(f"reference leak ({leak})")
    if not (1.0 <= wps <= 3.2):
        score *= 0.8
        notes.append(f"unusual speech rate {wps:.2f} words/s")
    ok = (score >= min_score) and not leak
    return VoiceCheck(score, ok, leak, wps, avg_lp, "; ".join(notes) or "clean")


def generate_silma_guarded(
    text: str,
    out_path: Path,
    reference_wav: Path,
    reference_text: str,
    *,
    speed: float = 1.0,
    base_seed: int = 42,
    attempts: int = 2,
    min_score: float = 0.6,
) -> tuple[Path, VoiceCheck]:
    """يولّد بـ SILMA حتى `attempts` مرات (seed مختلف لكل محاولة)، يفحص كل
    مرشّح، ويحتفظ بالأفضل. يرفع SilmaQualityError إن لم ينجح أي مرشّح — وعندها
    يتحول main.py إلى Edge TTS بدل نشر صوت مشوّه."""
    from silma_tts.api import SilmaTTS

    model = SilmaTTS()          # يُحمَّل مرة واحدة ويُعاد استخدامه بين المحاولات
    best: tuple[Path, VoiceCheck] | None = None
    for i in range(max(1, attempts)):
        seed = base_seed + i * 101
        wav = out_path.with_name(f"{out_path.stem}.try{i}.silma.wav")
        mp3 = out_path.with_name(f"{out_path.stem}.try{i}.mp3")
        model.infer(
            ref_file=str(reference_wav), ref_text=reference_text, gen_text=text,
            file_wave=str(wav), seed=seed, speed=speed,
        )
        conv = subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", str(wav), "-c:a", "libmp3lame", "-b:a", "192k", str(mp3)],
            capture_output=True, text=True,
        )
        wav.unlink(missing_ok=True)
        if conv.returncode != 0 or not mp3.exists() or mp3.stat().st_size == 0:
            log.warning("SILMA attempt %d: ffmpeg conversion failed: %s", i + 1, conv.stderr[-500:])
            continue
        check = evaluate_tts_audio(mp3, text, reference_text, min_score)
        log.info(
            "SILMA attempt %d/%d (seed=%d): score=%.2f ok=%s wps=%.2f avg_logprob=%s — %s",
            i + 1, attempts, seed, check.score, check.ok, check.words_per_sec,
            f"{check.avg_logprob:.2f}" if check.avg_logprob is not None else "n/a", check.note,
        )
        if best is None or check.score > best[1].score:
            if best is not None:
                best[0].unlink(missing_ok=True)
            best = (mp3, check)
        else:
            mp3.unlink(missing_ok=True)
        if check.ok:
            break

    if best is None or not best[1].ok:
        if best is not None:
            best[0].unlink(missing_ok=True)
        raise SilmaQualityError(
            "No SILMA candidate passed the quality check"
            + (f" (best: score={best[1].score:.2f}, {best[1].note})" if best else " (no usable audio produced)")
        )
    best[0].replace(out_path)
    return out_path, best[1]
