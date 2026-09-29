#!/usr/bin/env python3
"""
Automated Content Creation & Publishing Pipeline
==================================================
Generates a short-form vertical video (Arabic voiceover + burned-in subtitles +
Pexels stock footage) from an AI-generated viral topic, then publishes it to
YouTube / TikTok / Facebook via the Buffer API (buffer.com).

Provider: Gemini (primary) -> OpenRouter (optional fallback)
Fact Check: Wikipedia (ar.wikipedia.org)
"""

from __future__ import annotations

import asyncio
import difflib
import hashlib
import json
import logging
import os
import random
import re
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests
from llm_gemini import GEMINI_API_KEY, gemini_chat, gemini_key_kind
from tts_quality import enforce_text_quality, generate_silma_guarded, load_reference, resolve_reference_profile
from fact_check import fact_check_topic

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("pipeline")

WORK_DIR = Path(os.getenv("WORK_DIR", "./work"))
WORK_DIR.mkdir(parents=True, exist_ok=True)
TOPIC_HISTORY_FILE = Path(os.getenv("TOPIC_HISTORY_FILE", "topic_history.json"))
DRY_RUN = os.getenv("DRY_RUN", "false").lower() == "true"
DRY_RUN_INPUT_FILE = Path(os.getenv("DRY_RUN_INPUT_FILE", "tests/fixtures/bad_narration.txt"))

MIN_AUDIO_SECONDS = float(os.getenv("MIN_AUDIO_SECONDS", "60"))
MAX_AUDIO_SECONDS = float(os.getenv("MAX_AUDIO_SECONDS", "90"))
TARGET_AUDIO_SECONDS = float(os.getenv("TARGET_AUDIO_SECONDS", str(max(1.0, MAX_AUDIO_SECONDS - 5.0))))
MAX_SCRIPT_WORDS = int(os.getenv("MAX_SCRIPT_WORDS", "165"))
MIN_SCRIPT_WORDS = int(os.getenv("MIN_SCRIPT_WORDS", "90"))
REQUIRE_EXTERNAL_SOURCES = os.getenv("REQUIRE_EXTERNAL_SOURCES", "false").lower() == "true"
EDITORIAL_REVIEW_ENABLED = os.getenv("EDITORIAL_REVIEW_ENABLED", "true").lower() == "true"
FACT_CHECK_ENABLED = os.getenv("FACT_CHECK_ENABLED", "true").lower() == "true"
TOPIC_GENERATION_MAX_ATTEMPTS = int(os.getenv("TOPIC_GENERATION_MAX_ATTEMPTS", "5"))
HISTORY_LIMIT = int(os.getenv("HISTORY_LIMIT", "1000"))
MIN_SCENE_CLIPS = int(os.getenv("MIN_SCENE_CLIPS", "10"))


def _clean_env(name: str) -> str | None:
    raw = os.getenv(name)
    if raw is None:
        return None
    return re.sub(r"\s+", "", raw) or None


PEXELS_API_KEY = _clean_env("PEXELS_API_KEY")
BUFFER_API_KEY = _clean_env("BUFFER_API_KEY")
BUFFER_CHANNEL_IDS = [c.strip() for c in os.getenv("BUFFER_CHANNEL_IDS", "").split(",") if c.strip()]
GH_RELEASE_TOKEN = _clean_env("GH_RELEASE_TOKEN") or _clean_env("GITHUB_TOKEN")
GITHUB_REPOSITORY = os.getenv("GITHUB_REPOSITORY")

# ---------------------------------------------------------------------------
# LLM provider selection: Gemini (primary) -> OpenRouter (optional fallback)
# ---------------------------------------------------------------------------
OPENROUTER_API_KEY = _clean_env("OPENROUTER_API_KEY")
OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_REFERER = os.getenv(
    "OPENROUTER_REFERER",
    f"https://github.com/{GITHUB_REPOSITORY}" if GITHUB_REPOSITORY else "https://github.com/",
)
OPENROUTER_TITLE = os.getenv("OPENROUTER_TITLE", "Auto Publish Reels")
_OPENROUTER_MODEL_LIST_RAW = os.getenv(
    "OPENROUTER_MODEL",
    "google/gemma-4-26b-a4b-it:free,"
    "google/gemma-4-31b-it:free,"
    "nvidia/nemotron-3-super-120b-a12b:free",
)
OPENROUTER_MODELS = [m.strip() for m in _OPENROUTER_MODEL_LIST_RAW.split(",") if m.strip()]
OPENROUTER_MODEL = OPENROUTER_MODELS[0] if OPENROUTER_MODELS else ""

LLM_MAX_COMPLETION_TOKENS = int(os.getenv("LLM_MAX_COMPLETION_TOKENS", "4000"))
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "120"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))
OPENROUTER_MAX_ATTEMPTS = max(1, int(os.getenv("OPENROUTER_MAX_ATTEMPTS", "2")))

TTS_ENGINE = os.getenv("TTS_ENGINE", "silma").strip().lower()
TTS_VOICE = os.getenv("TTS_VOICE", "ar-EG-SalmaNeural")
EDGE_TTS_RATE = os.getenv("EDGE_TTS_RATE", "-8%")
EDGE_TTS_PITCH = os.getenv("EDGE_TTS_PITCH", "-5Hz")
SILMA_REFERENCE_WAV = Path(os.getenv("SILMA_REFERENCE_WAV", "assets/voice_reference_synthetic.wav"))
SILMA_REFERENCE_TEXT = os.getenv("SILMA_REFERENCE_TEXT", "").strip()
SILMA_REFERENCE_PROFILE = os.getenv("SILMA_REFERENCE_PROFILE", "").strip()
SILMA_VOICE_PROFILES_FILE = Path(os.getenv("SILMA_VOICE_PROFILES_FILE", "assets/voices/voice_profiles.json"))
SILMA_SEED = int(os.getenv("SILMA_SEED", "42"))
SILMA_MAX_ATTEMPTS = int(os.getenv("SILMA_MAX_ATTEMPTS", "2"))
SILMA_MIN_SCORE = float(os.getenv("SILMA_MIN_SCORE", "0.6"))
SILMA_SPEED = float(os.getenv("SILMA_SPEED", "1.0"))
SILMA_GUARD_ENABLED = os.getenv("SILMA_GUARD_ENABLED", "true").lower() == "true"
SILMA_GUARD_MIN_MATCH_WORDS = int(os.getenv("SILMA_GUARD_MIN_MATCH_WORDS", "2"))
VOICE_ROTATION_ENABLED = os.getenv("VOICE_ROTATION_ENABLED", "true").lower() == "true"
VOICE_SELECTION_MODE = os.getenv("VOICE_SELECTION_MODE", "rotation").strip().lower()
AUTO_VOICE_PROFILES = [
    item.strip() for item in os.getenv(
        "AUTO_VOICE_PROFILES", "hossam,hossam_uploaded,marwan,nassim,ahmed_z,egyptian_female"
    ).split(",") if item.strip()
]

PEXELS_SEARCH_ENDPOINT = "https://api.pexels.com/videos/search"
BUFFER_ENDPOINT = "https://api.buffer.com"
GITHUB_API_BASE = "https://api.github.com"

VIDEO_W, VIDEO_H = 1080, 1920
RETRY_BACKOFF_SECONDS = 5

BG_MUSIC_URL = _clean_env("BG_MUSIC_URL")
MUSIC_VOLUME = float(os.getenv("MUSIC_VOLUME", "0.07"))
DEFAULT_BG_MUSIC_URL = (
    "https://raw.githubusercontent.com/effacestudios/Royalty-Free-Music-Pack/master/Bubbles.mp3"
)

REQUIRED_ENV = {
    "PEXELS_API_KEY": PEXELS_API_KEY,
    "BUFFER_API_KEY": BUFFER_API_KEY,
    "GH_RELEASE_TOKEN": GH_RELEASE_TOKEN,
    "GITHUB_REPOSITORY": GITHUB_REPOSITORY,
}


class PipelineError(Exception):
    """Raised for any unrecoverable pipeline failure."""


TOPIC_CATEGORIES = [
    "عجائب عالم الحيوان",
    "غرائب جسم الإنسان والطب",
    "حقائق علمية صادمة",
    "أسرار الفضاء والمحيطات",
    "ظواهر طبيعية نادرة",
    "اختراعات وظواهر تقنية",
    "العلوم القديمة والحديثة",
]

CHANNEL_BRIEF = (
    "وجهتكم الأولى لاكتشاف أسرار الفضاء، والظواهر الكونية المذهلة، وخفايا الطبيعة، "
    "وأعماق المحيطات. نقدم رحلات علمية ممتعة وسريعة لتبسيط أعقد العلوم، "
    "والرد على التساؤلات المحيرة، والغرائب العلمية، والعلوم الحديثة والقديمة، "
    "والظواهر الطبيعية الغريبة، والطب، والفلك، والهندسة، وكل ما هو غريب وموثق "
    "في العلم وعلم الحيوان."
)


@dataclass
class Topic:
    hook_text: str
    narration_script: str
    title: str
    caption: str
    hashtags: list[str] = field(default_factory=list)
    search_keywords_en: str = ""
    scene_keywords_en: list[str] = field(default_factory=list)
    category: str = ""
    verified_fact: str = ""
    source_urls: list[str] = field(default_factory=list)
    voice_profile: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def check_env() -> None:
    if not GEMINI_API_KEY and not OPENROUTER_API_KEY:
        raise PipelineError("GEMINI_API_KEY is required (OPENROUTER_API_KEY alone is only a fallback)")
    if GEMINI_API_KEY:
        kind = gemini_key_kind()
        log.info("Gemini API key detected: %s", kind)
        if kind.startswith("unrecognized"):
            log.warning("Gemini key format looks unusual; if calls fail with 401/403, re-create the secret")
    missing = [k for k, v in REQUIRED_ENV.items() if not v]
    if not BUFFER_CHANNEL_IDS:
        missing.append("BUFFER_CHANNEL_IDS")
    if missing:
        raise PipelineError(f"Missing required environment variables: {', '.join(missing)}")


def with_retries(fn, *args, what: str = "operation", max_retries: int | None = None, **kwargs):
    attempts = max_retries if max_retries is not None else LLM_MAX_RETRIES
    last_err: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            last_err = exc
            log.warning("Attempt %d/%d for %s failed: %s", attempt, attempts, what, exc)
            if attempt < attempts:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)
    raise PipelineError(f"{what} failed after {attempts} attempts: {last_err}") from last_err


def _trim_script_to_word_limit(script: str, max_words: int) -> str:
    words = script.split()
    if len(words) <= max_words:
        return script
    truncated = " ".join(words[:max_words])
    last_boundary = max(truncated.rfind("."), truncated.rfind("؟"), truncated.rfind("!"))
    if last_boundary > len(truncated) * 0.6:
        truncated = truncated[: last_boundary + 1]
    return truncated.strip()


CTA_OUTRO = "إِذَا أَعْجَبَكَ الْفِيدْيُو، فَاضْغَطْ زِرَّ الإِعْجَابِ، وَلا تَنْسَ مُشَارَكَةَ الْفِيدْيُو"
OUTRO_MIN_WORDS = len(CTA_OUTRO.split())
OUTRO_MAX_WORDS = OUTRO_MIN_WORDS


def _append_engagement_outro(script: str) -> str:
    text = script.strip()
    if text.endswith(CTA_OUTRO):
        return text
    return f"{text} {CTA_OUTRO}".strip()


_TASHKEEL_CHARS = "\u0610-\u061A\u064B-\u065F\u06D6-\u06DC\u06DF-\u06E8\u06EA-\u06ED\u0670"
_TASHKEEL_RE = re.compile(f"[{_TASHKEEL_CHARS}]")


def _normalize_for_compare(s: str) -> str:
    s = _TASHKEEL_RE.sub("", s)
    return re.sub(r"\s+", " ", s).strip()


_TASHKEEL_CHECK_ALLOWLIST = {"به", "عليك", "هذه"}


def _find_unmarked_pronoun_suffixes(text: str) -> list[str]:
    offenders = []
    for raw_word in text.split():
        word = raw_word.strip(" ،.!؟\u061F:")
        if not word or word in _TASHKEEL_CHECK_ALLOWLIST:
            continue
        if word[-1] in ("ك", "ه"):
            offenders.append(word)
    return offenders


def _validate_fixed_arabic_strings() -> None:
    offenders = _find_unmarked_pronoun_suffixes(CTA_OUTRO)
    if offenders:
        raise PipelineError(
            f"CTA_OUTRO has word(s) ending in a bare ك/ه with no diacritic: {offenders}"
        )


_validate_fixed_arabic_strings()


def _fuzzy_word_pattern(word: str) -> str:
    gap = f"[{_TASHKEEL_CHARS}]*"
    return gap.join(re.escape(ch) for ch in word)


def _strip_duplicate_hook(hook: str, script: str) -> str:
    norm_hook_words = _normalize_for_compare(hook).split()
    if not norm_hook_words:
        return script
    between = f"[\\s{_TASHKEEL_CHARS}]*"
    pattern = between.join(_fuzzy_word_pattern(w) for w in norm_hook_words)
    cleaned = re.sub(pattern, " ", script)
    return re.sub(r"\s+", " ", cleaned).strip(" ،.!؟\u061F")


def _topic_fingerprint(topic: Topic | dict[str, Any]) -> str:
    values = [topic.get(k, "") if isinstance(topic, dict) else getattr(topic, k, "")
              for k in ("title", "hook_text", "search_keywords_en")]
    return re.sub(r"[^\w\u0600-\u06ff]+", " ", " ".join(map(str, values))).strip().lower()


def load_topic_history() -> list[dict[str, Any]]:
    try:
        data = json.loads(TOPIC_HISTORY_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def load_voice_profile_ids(path: Path | None = None) -> tuple[list[str], str]:
    path = path or SILMA_VOICE_PROFILES_FILE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        profiles = data.get("profiles", {}) if isinstance(data, dict) else {}
        if not isinstance(profiles, dict):
            raise ValueError("profiles must be an object")
        ids = [str(pid) for pid in profiles if str(pid).strip()]
        if not ids:
            raise ValueError("no voice profiles configured")
        default = str(data.get("default", ids[0]))
        if default not in ids:
            default = ids[0]
        return ids, default
    except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
        raise PipelineError(f"Could not load voice profiles {path}: {exc}") from exc


def choose_voice_profile(
    history: list[dict[str, Any]] | None = None,
    topic: Topic | dict[str, Any] | None = None,
) -> str:
    profiles, default = load_voice_profile_ids()
    if SILMA_REFERENCE_PROFILE and SILMA_REFERENCE_PROFILE.lower() != "auto":
        if SILMA_REFERENCE_PROFILE not in profiles:
            raise PipelineError(f"Unknown SILMA_REFERENCE_PROFILE {SILMA_REFERENCE_PROFILE!r}")
        return SILMA_REFERENCE_PROFILE
    if not VOICE_ROTATION_ENABLED:
        selected = SILMA_REFERENCE_PROFILE or default
        if selected not in profiles:
            raise PipelineError(f"Unknown SILMA_REFERENCE_PROFILE {selected!r}")
        return selected
    history = history if history is not None else load_topic_history()
    if VOICE_SELECTION_MODE == "content":
        candidates = [profile for profile in AUTO_VOICE_PROFILES if profile in profiles] or profiles
        current = topic if topic is not None else (history[-1] if history else {})
        topic_text = " ".join(
            str(current.get(field, "") if isinstance(current, dict) else getattr(current, field, ""))
            for field in ("title", "narration_script", "narration")
        )
        female = re.search(r"(امرأة|فتاة|طفلة|زوجة|أميرة|ملكة|ممرضة)", topic_text)
        if female and "egyptian_female" in candidates:
            return "egyptian_female"
        recent = {
            str(item.get("voice_profile", ""))
            for item in history[-3:]
            if item.get("voice_profile")
        }
        available = [profile for profile in candidates if profile not in recent] or candidates
        seed = topic_text.encode("utf-8")
        return available[int.from_bytes(hashlib.sha256(seed).digest()[:4], "big") % len(available)]
    last_profile = next(
        (str(item.get("voice_profile", "")) for item in reversed(history)
         if str(item.get("voice_profile", "")) in profiles),
        None,
    )
    if last_profile is None:
        return default
    return profiles[(profiles.index(last_profile) + 1) % len(profiles)]


def topic_is_too_similar(topic: Topic, history: list[dict[str, Any]]) -> bool:
    candidate = _topic_fingerprint(topic)
    candidate_words = set(candidate.split())
    for old in history:
        old_fp = _topic_fingerprint(old)
        if candidate == old_fp or difflib.SequenceMatcher(None, candidate, old_fp).ratio() >= 0.68:
            return True
        old_words = set(old_fp.split())
        if candidate_words and old_words and len(candidate_words & old_words) / len(candidate_words | old_words) >= 0.55:
            return True
    return False


def remember_topic(topic: Topic) -> None:
    history = load_topic_history()
    history.append({
        "title": topic.title,
        "hook_text": topic.hook_text,
        "search_keywords_en": topic.search_keywords_en,
        "category": topic.category,
        "voice_profile": topic.voice_profile,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    TOPIC_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    TOPIC_HISTORY_FILE.write_text(
        json.dumps(history[-HISTORY_LIMIT:], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def extract_json_block(text: str) -> dict[str, Any]:
    """Extract the JSON object from a model reply.

    Tolerates <think> blocks, markdown fences and reasoning text before/after
    the JSON. Prefers an object that contains one of the expected keys.
    """
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    decoder = json.JSONDecoder()
    found: dict[str, Any] | None = None
    for m in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text[m.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            found = obj
            if "corrected_text" in obj or "hook_text" in obj:
                return obj
    if found is not None:
        return found
    raise PipelineError(f"Could not find JSON object in model response: {text[:300]}")


# ---------------------------------------------------------------------------
# Topic generation (no topic bank: model picks freely from categories)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent(
    """
    أنت كاتب محتوى عربي محترف للريلز القصيرة. اختر موضوعًا واحدًا فقط من مجال القناة
    واكتب نصًا صوتيًا علميًا قابلًا للتحقق عنه.

    وصف القناة:
    {channel_brief}

    قواعد الدقة الصارمة (لا يجوز خرقها):
    - اكتب فقط معلومات عامة مؤكدة ومعروفة، وتجنب أي رقم أو تاريخ أو اسم أو تفصيل
      دقيق قد يكون غير صحيح. عند الشك في تفصيل، احذفه.
    - لا تختلق أي معلومة. إن لم تكن متأكدًا من رقم، اذكر الوصف العام بدل الرقم.
    - إذا ذكرت رقمًا، يجب أن يكون من معلومات عامة موثقة ومشهورة.
    - اجعل الموضوع قائمًا على حقيقة مركزية واحدة يمكن التحقق منها في ويكيبيديا أو
      مصدر علمي موثوق، واجعل كل جملة في النص تخدم هذه الحقيقة.
    - ممنوع المعجزات أو الخوارق أو الأساطير أو الادعاءات الدينية أو التنجيم أو
      المؤامرات أو القصص التي لا يمكن فحصها بمصدر علمي مباشر.
    - لا تستخدم صياغة مثيرة على حساب الدقة، ولا تضف أكثر من حقيقة مركزية واحدة.

    أخرج JSON صالحًا فقط بالمفاتيح التالية:
    {
      "category": "انسخ الفئة التي اخترتها حرفيًا من القائمة المعطاة",
      "subject": "الموضوع الذي اخترته في 3-7 كلمات (يُستخدم كعنوان)",
      "hook_text": "سؤال عربي فصيح قصير لا يتجاوز 12 كلمة، يجذب المشاهد",
      "narration_script": "نص عربي مترابط يبدأ بـ hook_text حرفيًا ويشرح الموضوع، دون خاتمة تفاعلية",
      "title": "انسخ subject حرفيًا كما هو",
      "caption": "جملة أو جملتان تصفان الموضوع دون ادعاء جديد",
      "hashtags": ["#وسم1", "#وسم2", "#وسم3", "#وسم4", "#وسم5"],
      "search_keywords_en": "عبارة إنجليزية قصيرة من كلمتين إلى ثلاث عن الموضوع",
      "scene_keywords_en": ["4 إلى 7 أوصاف إنجليزية قصيرة لمشاهد مرئية مرتبطة بفقرات النص"]
    }

    قواعد الأسلوب:
    - اكتب العربية الفصحى السليمة، وشكّل النص تشكيلًا كاملًا صحيحًا لتوجيه النطق.
    - راجع كل جملة نحويًا: عيّن الفاعل والمفعول، واضبط عائد كل ضمير قبل إخراج النص.
    - طابق الفعل مع الفاعل في التذكير والتأنيث والإفراد والتثنية والجمع، ولا تخلط بين
      الغائب والمخاطب والمتكلم أو بين الماضي والمضارع.
    - ابدأ narration_script بنص hook_text نفسه، وأنهِ الشرح بعد اكتماله.
    - لا تضف طلب إعجاب أو مشاركة أو اشتراك (البرنامج يضيفها تلقائيًا).
    - اجعل scene_keywords_en مرتبطة فعليًا بمحتوى النص.
    """
).replace("{channel_brief}", CHANNEL_BRIEF).strip()


def _openrouter_chat(messages: list[dict[str, str]], max_tokens: int, temperature: float) -> str:
    """OpenRouter with multi-model fallback (optional secondary provider)."""
    if not OPENROUTER_API_KEY:
        raise PipelineError("OPENROUTER_API_KEY is not set")
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": OPENROUTER_REFERER,
        "X-Title": OPENROUTER_TITLE,
    }
    last_error = "no models"
    for model in OPENROUTER_MODELS:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max(max_tokens, 8000),
            "temperature": temperature,
            "reasoning": {"effort": "low", "exclude": True},
        }
        for attempt in range(1, OPENROUTER_MAX_ATTEMPTS + 1):
            try:
                resp = requests.post(OPENROUTER_ENDPOINT, headers=headers, json=payload, timeout=LLM_TIMEOUT)
            except Exception as exc:
                last_error = f"{model}: {exc}"
                log.warning("OpenRouter %s attempt %d: %s", model, attempt, exc)
                continue
            if resp.status_code == 429:
                last_error = f"{model}: 429"
                log.warning("OpenRouter %s rate-limited", model)
                if attempt < OPENROUTER_MAX_ATTEMPTS:
                    time.sleep(8.0)
                continue
            if resp.status_code != 200:
                last_error = f"{model}: HTTP {resp.status_code} {resp.text[:200]}"
                log.warning("OpenRouter %s failed: %s", model, last_error)
                break
            data = resp.json()
            content = ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
            if not content.strip():
                last_error = f"{model}: empty content"
                log.warning("OpenRouter %s returned empty content (attempt %d)", model, attempt)
                continue
            log.info("OpenRouter: using model %s", model)
            return content
    raise PipelineError(f"All OpenRouter models failed: {last_error}")


def llm_chat(messages: list[dict[str, str]], max_tokens: int = LLM_MAX_COMPLETION_TOKENS,
             temperature: float = 0.7) -> str:
    """Dispatch to Gemini first; if it fails and OpenRouter is configured, try that."""
    if GEMINI_API_KEY:
        try:
            return gemini_chat(messages, max_tokens=max_tokens, temperature=temperature,
                               timeout=LLM_TIMEOUT)
        except Exception as exc:
            log.warning("Gemini failed (%s)", exc)
            if not OPENROUTER_API_KEY:
                raise PipelineError(f"Gemini failed and no fallback is configured: {exc}") from exc
            log.warning("Trying OpenRouter fallback")
    if OPENROUTER_API_KEY:
        return _openrouter_chat(messages, max_tokens, temperature)
    raise PipelineError("No LLM provider available")


PROOFREAD_SYSTEM_PROMPT = textwrap.dedent(
    """
    أنت محرر عربي فصيح ومدقق علمي صارم. سيصلك نص قصير مرشح للنشر. أعد كتابته
    كاملاً بعد إصلاح النحو والصرف والأسلوب، واحذف أي ادعاء غير موثق.

    قواعد:
    1) اكتب عربية فصحى سليمة فقط. ممنوع العامية والتراكيب المترجمة حرفيًا.
    2) لا تخترع مصطلحًا أو رقمًا أو اسمًا. احذف أي ادعاء لا يمكن التحقق منه.
    3) صحح الإعراب وعلامات الترقيم، واحذف الحشو.
    4) حافظ على سؤال البداية وموضوع النص. صحح الصياغة فقط.
    5) ضع التشكيل الكامل المناسب للنطق.
    6) ضع حركة الإعراب الأخيرة عند الحاجة، وشكّل الأفعال والضمائر والكلمات الملتبسة
       تشكيلًا واضحًا؛ لا تترك كلمة عربية مهمة بلا تشكيل إذا كان لها أكثر من قراءة.
    7) إذا وُجد مفتاح issues_to_fix فأصلح كل مشكلة مذكورة فيه.
    8) اترك النص ينتهي بعد اكتمال الشرح، دون خاتمة تفاعلية.

    أجب حصراً بكائن JSON بمفتاح واحد:
    {"corrected_text": "النص العربي الكامل بعد المراجعة"}
    """
).strip()


CONTENT_RED_FLAGS = (
    "السيلينس", "الشهرات الجوية", "المحتلة بالدقيق", "البركان الثلجي",
    "الهزات الرقمية", "مغنطيس قمري يستخلص", "مغناطيس قمري يستخلص",
    "انفجار نجمي قريب", "الدراسات الأسترادية", "تحمية الكوكب",
    "القمر يتفجر في سطوحه",
)


def find_content_red_flag(text: str) -> str | None:
    plain = re.sub(r"[\u064B-\u065F\u0670]", "", text or "")
    sentences = re.split(r"(?<=[.!؟])\s+", plain)
    for flag in CONTENT_RED_FLAGS:
        for sentence in sentences:
            if flag not in sentence:
                continue
            negated = re.search(
                r"(?:لا|ليس|ليست|غير|لا توجد|لا يوجد|لا يعتمد|لا تسمى|لا يسمى|لا يطلق|لا تطلق)"
                r"[^.!؟]{0,100}" + re.escape(flag), sentence
            )
            corrected = re.search(
                re.escape(flag) + r"[^.!؟]{0,100}(?:غير دقيق|غير صحيحة|غير صحيح|لا وجود|ليست.*حقيقة)",
                sentence,
            )
            if not (negated or corrected):
                return flag
    return None


def proofread_narration_tashkeel(
    script: str,
    issues: list[str] | None = None,
    *,
    canonical_subject: str | None = None,
    verified_fact: str | None = None,
) -> str:
    def _call() -> str:
        payload: dict[str, Any] = {"text": script}
        if issues:
            payload["issues_to_fix"] = issues
        if canonical_subject:
            payload["canonical_subject"] = canonical_subject
        if verified_fact:
            payload["verified_fact"] = verified_fact
        messages = [
            {"role": "system", "content": PROOFREAD_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        raw_text = llm_chat(messages, max_tokens=LLM_MAX_COMPLETION_TOKENS, temperature=0.15)
        parsed = extract_json_block(raw_text)
        corrected = str(parsed.get("corrected_text", "")).strip()
        if not corrected:
            raise PipelineError("Editorial review returned an empty corrected_text")
        return corrected

    corrected = with_retries(_call, what="Arabic editorial review")
    orig_words = len(_normalize_for_compare(script).split())
    new_words = len(_normalize_for_compare(corrected).split())
    minimum_preserved_words = max(1, (orig_words * 70 + 99) // 100)
    maximum_review_words = max(1, (orig_words * 135) // 100)
    if new_words < minimum_preserved_words or new_words > maximum_review_words:
        raise PipelineError(
            f"Editorial review changed script length too much ({orig_words} -> {new_words} words)"
        )
    if len(re.findall(r"[\u0600-\u06FF]", corrected)) < max(1, new_words * 2):
        raise PipelineError("Editorial review returned insufficient Arabic text")
    log.info("Editorial gate passed (%d -> %d words)", orig_words, new_words)
    return corrected


def _validate_final_script_word_count(word_count: int) -> None:
    if word_count < MIN_SCRIPT_WORDS:
        raise PipelineError(
            f"Final narration is {word_count} words; minimum is {MIN_SCRIPT_WORDS}"
        )
    if word_count > MAX_SCRIPT_WORDS:
        raise PipelineError(
            f"Final narration is {word_count} words; maximum is {MAX_SCRIPT_WORDS}"
        )


def build_topic_user_prompt(recent_topics: list[str], recent_categories: list[str]) -> str:
    pre_outro_max = MAX_SCRIPT_WORDS - OUTRO_MAX_WORDS
    pre_outro_min = max(MIN_SCRIPT_WORDS - OUTRO_MAX_WORDS, 40)
    categories_str = "\n".join(f"- {c}" for c in TOPIC_CATEGORIES)
    recent_topics_str = ", ".join(recent_topics[-15:]) if recent_topics else "(لا يوجد)"
    recent_cats_str = ", ".join(recent_categories[-2:]) if recent_categories else "(لا يوجد)"
    return (
        "اختر موضوعًا واحدًا مثيرًا من الفئات التالية فقط:\n"
        f"{categories_str}\n\n"
        f"الموضوعات المستخدمة مؤخرًا (تجنبها): {recent_topics_str}\n"
        f"الفئات المستخدمة مؤخرًا (تجنبها): {recent_cats_str}\n\n"
        "اكتب narration_script يشرح الموضوع بوضوح، بحيث يكون عدد كلماته "
        f"بين {pre_outro_min} و{pre_outro_max} كلمة قبل العبارة الختامية الثابتة "
        f"(المطلوب النهائي بين {MIN_SCRIPT_WORDS} و{MAX_SCRIPT_WORDS} كلمة مع العبارة).\n"
        "ابدأ narration_script بـ hook_text نفسه. اجعل scene_keywords_en بين 4 و7 عبارات.\n"
        "أخرج JSON فقط."
    )


def generate_topic() -> tuple[Topic, list[dict[str, str]], list[dict[str, str]]]:
    log.info("Generating a fresh topic via LLM (provider: %s)...",
             "gemini" if GEMINI_API_KEY else "openrouter")

    def _parse_topic(raw_text: str) -> tuple[Topic, int]:
        parsed = extract_json_block(raw_text)
        topic = Topic(
            hook_text=str(parsed["hook_text"]).strip(),
            narration_script=str(parsed.get("narration_script", "")).strip(),
            title=str(parsed.get("title", parsed.get("subject", ""))).strip(),
            caption=str(parsed.get("caption", "")).strip(),
            hashtags=list(parsed.get("hashtags", [])),
            search_keywords_en=str(parsed.get("search_keywords_en", "nature abstract")).strip(),
            scene_keywords_en=[str(x).strip() for x in parsed.get("scene_keywords_en", []) if str(x).strip()],
            category=str(parsed.get("category", "")).strip(),
        )
        if not topic.hook_text:
            raise PipelineError("Model returned an empty hook_text")
        if not topic.title:
            raise PipelineError("Model returned an empty title")
        topic.narration_script = _strip_duplicate_hook(topic.hook_text, topic.narration_script)
        topic.narration_script = f"{topic.hook_text} {topic.narration_script}".strip()
        return topic, len(topic.narration_script.split())

    def _call() -> Topic:
        history = load_topic_history()
        recent_topics = [h.get("title", "") for h in history[-15:] if h.get("title")]
        recent_categories = [h.get("category", "") for h in history[-5:] if h.get("category")]
        user_msg = build_topic_user_prompt(recent_topics, recent_categories)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        raw_text = llm_chat(messages, temperature=0.6)
        topic, word_count = _parse_topic(raw_text)

        if topic.category not in TOPIC_CATEGORIES:
            raise PipelineError(f"Model returned unknown category: {topic.category!r}")

        pre_outro_max = MAX_SCRIPT_WORDS - OUTRO_MAX_WORDS
        if word_count > pre_outro_max:
            log.warning("Script too long (%d words); trimming to %d", word_count, pre_outro_max)
            topic.narration_script = _trim_script_to_word_limit(topic.narration_script, pre_outro_max)
            word_count = len(topic.narration_script.split())

        _validate_final_script_word_count(word_count + OUTRO_MIN_WORDS)

        if len(topic.scene_keywords_en) < MIN_SCENE_CLIPS:
            raise PipelineError(
                f"Model returned fewer than {MIN_SCENE_CLIPS} scene keywords"
            )
        if topic_is_too_similar(topic, history):
            raise PipelineError("Generated topic is too similar to a previously published topic")
        return topic

    topic = with_retries(_call, what="topic generation", max_retries=TOPIC_GENERATION_MAX_ATTEMPTS)
    log.info("Topic generated: %s (category: %s, %d-word script)",
             topic.title, topic.category, len(topic.narration_script.split()))
    return topic, [], []


def _proofread_topic_narration(topic: Topic, script: str) -> str:
    """Editorial review. If the reviewer LLM fails, fall back to the original script.

    This is safe because fact_check_topic() still runs afterwards and blocks
    publishing of anything it cannot support from Wikipedia.
    """
    minimum_pre_outro_words = max(MIN_SCRIPT_WORDS - OUTRO_MIN_WORDS, 1)

    def keep_minimum_length(candidate: str) -> str:
        candidate_words = len(candidate.split())
        if candidate_words >= minimum_pre_outro_words:
            return candidate
        original_words = len(script.split())
        if original_words >= minimum_pre_outro_words:
            log.warning(
                "Editorial review shortened narration below the minimum "
                "(%d -> %d words); keeping the original script",
                original_words,
                candidate_words,
            )
            return script
        return candidate

    try:
        corrected = proofread_narration_tashkeel(
            script,
            canonical_subject=topic.title,
            verified_fact=topic.verified_fact or None,
        )
        corrected = keep_minimum_length(corrected)
        return enforce_text_quality(
            corrected,
            reviser=lambda text, issues: proofread_narration_tashkeel(
                text, issues=issues,
                canonical_subject=topic.title,
                verified_fact=topic.verified_fact or None,
            ),
        )
    except PipelineError as exc:
        log.error("Editorial review failed; the original script will be quality-checked strictly: %s", exc)
        return keep_minimum_length(enforce_text_quality(script))


# ---------------------------------------------------------------------------
# Pexels
# ---------------------------------------------------------------------------

def search_pexels_video(keywords: str, exclude: set[str] | None = None) -> str:
    log.info("Searching Pexels: %r", keywords)
    exclude = exclude or set()

    def _call() -> str:
        resp = requests.get(
            PEXELS_SEARCH_ENDPOINT,
            headers={"Authorization": PEXELS_API_KEY},
            params={"query": keywords, "orientation": "portrait", "size": "large", "per_page": 40},
            timeout=30,
        )
        if resp.status_code != 200:
            raise PipelineError(f"Pexels API error {resp.status_code}: {resp.text[:500]}")
        videos = resp.json().get("videos", [])
        if not videos:
            raise PipelineError(f"No Pexels results for {keywords!r}")
        for video in videos:
            files = [
                f for f in video.get("video_files", [])
                if f.get("width") and f.get("height") and f["height"] > f["width"]
            ]
            if not files:
                continue
            files.sort(key=lambda f: f["width"], reverse=True)
            best = files[0]
            if best["link"] not in exclude:
                return best["link"]
        raise PipelineError("No usable portrait clip found")

    return with_retries(_call, what="Pexels search")


def search_pexels_videos(keywords_list: list[str], topic_context: str = "") -> list[str]:
    urls: list[str] = []
    for keywords in keywords_list:
        combined = f"{topic_context} {keywords}".strip() if topic_context else keywords
        try:
            url = search_pexels_video(combined, exclude=set(urls))
        except PipelineError as exc:
            log.warning("Skipping scene %r: %s", keywords, exc)
            continue
        if url not in urls:
            urls.append(url)
    if len(urls) < MIN_SCENE_CLIPS:
        raise PipelineError(
            f"Only {len(urls)} distinct scene clips found; need {MIN_SCENE_CLIPS}"
        )
    return urls


def _load_word_timings(timings_path: Path) -> list[dict[str, Any]]:
    if not timings_path.exists():
        return []
    try:
        data = json.loads(timings_path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def compute_scene_durations(timings: list[dict[str, Any]], total_duration: float, n_scenes: int) -> list[float]:
    if n_scenes <= 1:
        return [total_duration]
    words = [w for w in timings if w.get("text")]
    if not words:
        return [total_duration / n_scenes] * n_scenes
    offsets = [max(float(w.get("offset", 0)), 0.0) for w in words]
    texts = [str(w.get("text", "")) for w in words]
    ideal_points = [total_duration * i / n_scenes for i in range(1, n_scenes)]
    window = max(total_duration / n_scenes * 0.4, 1.0)
    cut_points: list[float] = []
    for ideal in ideal_points:
        nearby = [o for o in offsets if abs(o - ideal) <= window]
        if not nearby:
            cut_points.append(ideal)
            continue
        sentence_ends = [
            o for o, t in zip(offsets, texts)
            if abs(o - ideal) <= window and t[-1:] in "؟?.!،"
        ]
        cut_points.append(min(sentence_ends or nearby, key=lambda o: abs(o - ideal)))
    cut_points = sorted(cut_points)
    bounds = [0.0] + cut_points + [total_duration]
    return [max(bounds[i + 1] - bounds[i], 0.3) for i in range(n_scenes)]


def build_multishot_background(clips: list[Path], segment_durations: list[float], out_path: Path) -> Path:
    inputs: list[str] = []
    filters: list[str] = []
    for i, (clip, segment) in enumerate(zip(clips, segment_durations)):
        inputs += ["-stream_loop", "-1", "-i", str(clip)]
        filters.append(
            f"[{i}:v]trim=duration={segment:.3f},setpts=PTS-STARTPTS,"
            f"scale={VIDEO_W}:{VIDEO_H}:force_original_aspect_ratio=increase,"
            f"crop={VIDEO_W}:{VIDEO_H},fps=30[v{i}]"
        )
    filters.append("".join(f"[v{i}]" for i in range(len(segment_durations))) +
                   f"concat=n={len(segment_durations)}:v=1:a=0[outv]")
    cmd = ["ffmpeg", "-y", *inputs, "-filter_complex", ";".join(filters),
           "-map", "[outv]", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
           "-an", str(out_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not out_path.exists() or out_path.stat().st_size == 0:
        raise PipelineError(f"ffmpeg multi-shot background failed: {result.stderr[-2000:]}")
    return out_path


def download_file(url: str, dest: Path) -> Path:
    log.info("Downloading %s", url)
    def _call() -> Path:
        with requests.get(url, stream=True, timeout=120) as resp:
            if resp.status_code != 200:
                raise PipelineError(f"Download failed ({resp.status_code})")
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
        return dest
    return with_retries(_call, what=f"download {url}")


def generate_ambient_music(dest: Path, duration: float = 90.0) -> Path | None:
    try:
        cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-f", "lavfi", "-i", f"sine=frequency=110:sample_rate=44100:duration={duration}",
            "-f", "lavfi", "-i", f"sine=frequency=164.81:sample_rate=44100:duration={duration}",
            "-filter_complex",
            "[0:a]volume=0.10,afade=t=in:st=0:d=4,afade=t=out:st=86:d=4[a];"
            "[1:a]volume=0.055,afade=t=in:st=0:d=4,afade=t=out:st=86:d=4[b];"
            "[a][b]amix=inputs=2:normalize=0,lowpass=f=900,volume=0.8[out]",
            "-map", "[out]", "-c:a", "libmp3lame", "-b:a", "96k", str(dest),
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        if dest.exists() and dest.stat().st_size > 0:
            return dest
    except Exception as exc:
        log.warning("Could not generate ambient music: %s", exc)
    return None


def get_bg_music(dest_dir: Path) -> Path | None:
    repo_root = Path(__file__).resolve().parent
    for name in ("music", "Music", "assets/music", "audio", "bg_music"):
        d = (repo_root / name).resolve()
        if not d.is_dir():
            continue
        found = sorted(p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in (".mp3", ".m4a", ".wav"))
        if found:
            chosen = random.choice(found)
            log.info("Background music: %s", chosen)
            return chosen
    candidates = [u.strip() for u in (BG_MUSIC_URL or DEFAULT_BG_MUSIC_URL).split(",") if u.strip()]
    if candidates:
        dest = dest_dir / "music.mp3"
        try:
            download_file(random.choice(candidates), dest)
            return dest
        except Exception as exc:
            log.warning("Could not fetch music: %s", exc)
    return generate_ambient_music(dest_dir / "ambient_pad.mp3")


# ---------------------------------------------------------------------------
# TTS
# ---------------------------------------------------------------------------

def _norm_arabic_words(text: str) -> list[str]:
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text or "")
    text = re.sub(r"[^\w\u0600-\u06FF]+", " ", text, flags=re.UNICODE)
    return [w for w in text.lower().split() if w]


def generate_edge_tts(text: str, out_path: Path, voice: str = TTS_VOICE) -> Path:
    log.info("Generating Edge TTS with voice %s", voice)
    import edge_tts
    async def _run():
        communicate = edge_tts.Communicate(text, voice, rate=EDGE_TTS_RATE, pitch=EDGE_TTS_PITCH)
        timings = []
        with open(out_path, "wb") as audio_file:
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_file.write(chunk["data"])
                elif chunk["type"] == "WordBoundary":
                    timings.append({
                        "text": chunk.get("text", ""),
                        "offset": chunk.get("offset", 0) / 10_000_000,
                        "duration": chunk.get("duration", 0) / 10_000_000,
                    })
        out_path.with_suffix(".timings.json").write_text(
            json.dumps(timings, ensure_ascii=False), encoding="utf-8"
        )
    def _call() -> Path:
        asyncio.run(_run())
        if not out_path.exists() or out_path.stat().st_size == 0:
            raise PipelineError("Edge TTS produced empty audio")
        return out_path
    return with_retries(_call, what="Edge TTS generation")


def generate_tts(text: str, out_path: Path, voice: str = TTS_VOICE,
                 reference_profile: str | None = None) -> Path:
    if TTS_ENGINE == "silma":
        try:
            selected_profile = reference_profile or SILMA_REFERENCE_PROFILE
            if selected_profile:
                reference, ref_text = resolve_reference_profile(selected_profile, SILMA_VOICE_PROFILES_FILE)
            else:
                reference, ref_text = load_reference(SILMA_REFERENCE_WAV, SILMA_REFERENCE_TEXT)
            generate_silma_guarded(
                text, out_path, reference, ref_text,
                speed=SILMA_SPEED, base_seed=SILMA_SEED,
                attempts=SILMA_MAX_ATTEMPTS, min_score=SILMA_MIN_SCORE,
            )
            out_path.with_suffix(".timings.json").write_text("[]", encoding="utf-8")
            return out_path
        except Exception:
            log.exception("SILMA failed; switching to Edge TTS")
            out_path.unlink(missing_ok=True)
            return generate_edge_tts(text, out_path, voice)
    return generate_edge_tts(text, out_path, voice)


def get_media_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


def _audio_duration_exceeds_maximum(d: float) -> bool:
    return d > MAX_AUDIO_SECONDS


def _audio_duration_needs_normalization(d: float) -> bool:
    return d > TARGET_AUDIO_SECONDS


def _reel_duration_needs_padding(d: float) -> bool:
    return d < MIN_AUDIO_SECONDS


def normalize_narration_duration(audio_path: Path, target_seconds: float) -> float:
    original = get_media_duration(audio_path)
    if original <= 0 or abs(original - target_seconds) < 0.15:
        return original
    tempo = original / target_seconds
    temp_path = audio_path.with_name(audio_path.stem + ".normalized.mp3")
    result = subprocess.run([
        "ffmpeg", "-y", "-v", "error", "-i", str(audio_path),
        "-filter:a", f"atempo={tempo:.6f}", "-c:a", "libmp3lame", "-b:a", "192k",
        str(temp_path),
    ], capture_output=True, text=True)
    if result.returncode != 0 or not temp_path.exists() or temp_path.stat().st_size == 0:
        raise PipelineError(f"Could not normalize audio: {result.stderr[-1000:]}")
    temp_path.replace(audio_path)
    timings_path = audio_path.with_suffix(".timings.json")
    if timings_path.exists():
        try:
            timings = json.loads(timings_path.read_text(encoding="utf-8"))
            scale = target_seconds / original
            for item in timings:
                item["offset"] = float(item.get("offset", 0)) * scale
                item["duration"] = float(item.get("duration", 0)) * scale
            timings_path.write_text(json.dumps(timings, ensure_ascii=False), encoding="utf-8")
        except Exception as exc:
            log.warning("Could not rescale timings: %s", exc)
    final = get_media_duration(audio_path)
    log.info("Audio normalized: %.1fs → %.1fs", original, final)
    return final


WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL", "base")
LOW_CONFIDENCE_THRESHOLD = float(os.getenv("PRONUNCIATION_CONFIDENCE_THRESHOLD", "0.4"))


def align_words_with_whisper(audio_path: Path, script_words: list[str]):
    from faster_whisper import WhisperModel
    log.info("Aligning subtitles with Whisper (%s)...", WHISPER_MODEL_SIZE)
    model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(str(audio_path), language="ar", word_timestamps=True, vad_filter=False)
    whisper_words: list[tuple[str, float, float, float]] = []
    for seg in segments:
        for w in (seg.words or []):
            t = (w.word or "").strip()
            if t:
                whisper_words.append((t, float(w.start), float(w.end), float(getattr(w, "probability", 1.0))))
    if not whisper_words:
        raise PipelineError("Whisper produced no timestamps")

    def _norm(w: str) -> str:
        return _normalize_for_compare(w).strip(" ،.!؟\u061F").lower()

    script_norm = [_norm(w) for w in script_words]
    whisper_norm = [_norm(w) for w, _, _, _ in whisper_words]
    matcher = difflib.SequenceMatcher(None, script_norm, whisper_norm, autojunk=False)
    timings: list[dict[str, Any] | None] = [None] * len(script_words)
    for _tag, i1, i2, j1, j2 in matcher.get_matching_blocks():
        for k in range(i2 - i1):
            if i1 + k >= len(script_words) or j1 + k >= len(whisper_words):
                continue
            _, start, end, prob = whisper_words[j1 + k]
            timings[i1 + k] = {"text": script_words[i1 + k], "offset": start,
                               "duration": max(end - start, 0.05), "probability": prob}
    known_indices = [i for i, t in enumerate(timings) if t is not None]
    if not known_indices:
        raise PipelineError("Could not align any words")
    if len(known_indices) < len(script_words) * 0.5:
        raise PipelineError("Alignment too unreliable")
    for i in range(len(timings)):
        if timings[i] is not None:
            continue
        prev_i = max((k for k in known_indices if k < i), default=None)
        next_i = min((k for k in known_indices if k > i), default=None)
        if prev_i is None:
            base = timings[next_i]
            offset = max(base["offset"] - 0.2 * (next_i - i), 0.0)
        elif next_i is None:
            base = timings[prev_i]
            offset = base["offset"] + base["duration"] * (i - prev_i)
        else:
            prev_end = timings[prev_i]["offset"] + timings[prev_i]["duration"]
            next_start = timings[next_i]["offset"]
            span = max(next_start - prev_end, 0.05)
            offset = prev_end + span * (i - prev_i) / (next_i - prev_i)
        timings[i] = {"text": script_words[i], "offset": offset, "duration": 0.3, "probability": None}
    known_set = set(known_indices)
    flagged_words = []
    for i in range(len(script_words)):
        if i not in known_set:
            flagged_words.append({"index": i, "text": script_words[i],
                                  "approx_seconds": round(timings[i]["offset"], 2),
                                  "reason": "not_recognized"})
            continue
        prob = timings[i].get("probability")
        if prob is not None and prob < LOW_CONFIDENCE_THRESHOLD:
            flagged_words.append({"index": i, "text": script_words[i],
                                  "approx_seconds": round(timings[i]["offset"], 2),
                                  "reason": "low_confidence", "confidence": round(prob, 2)})
    return timings, flagged_words


def realign_subtitles_with_whisper(narration_path: Path, script_text: str):
    timings_path = narration_path.with_suffix(".timings.json")
    try:
        aligned, flagged = align_words_with_whisper(narration_path, script_text.split())
        timings_path.write_text(json.dumps(aligned, ensure_ascii=False), encoding="utf-8")
        log.info("Whisper-aligned timings written")
        return True, flagged
    except Exception:
        log.error("WHISPER ALIGNMENT FAILED", exc_info=True)
        return False, []


def _ass_escape(text: str) -> str:
    return text.replace("{", "").replace("}", "").replace("\n", " ").replace("\r", " ")


DISPLAY_PUNCTUATION = str.maketrans(
    "".join([".", ",", "،", "؛", ":", "!", "?", "؟", "…", "-", "—", "_",
             "(", ")", "[", "]", "{", "}", '"', "«", "»", "/", "\\"]),
    " " * 23,
)


def _clean_display_words(words: list[str]) -> list[str]:
    cleaned = [_TASHKEEL_RE.sub("", w.translate(DISPLAY_PUNCTUATION)).strip() for w in words]
    return [w for w in cleaned if w]


def build_subtitles(text: str, duration: float, out_path: Path,
                    timings_path: Path | None = None) -> Path:
    timing_data = []
    if timings_path and timings_path.exists():
        try:
            timing_data = json.loads(timings_path.read_text(encoding="utf-8"))
        except Exception as exc:
            log.warning("Could not read timings: %s", exc)
    timed_words = [x for x in timing_data if x.get("text")]
    if timed_words:
        words = [w["text"] for w in timed_words]
        offsets = [max(float(w.get("offset", 0)), 0.0) for w in timed_words]
        durations = [max(float(w.get("duration", 0)), 0.0) for w in timed_words]
        chunks = []
        for i in range(0, len(words), 8):
            end_i = min(i + 8, len(words))
            start = offsets[i]
            end = offsets[end_i] if end_i < len(words) else max(
                offsets[end_i - 1] + durations[end_i - 1], duration)
            chunks.append((words[i:end_i], max(start, 0), min(max(end, start + 0.25), duration)))
    else:
        words = text.split()
        word_chunks = [words[i:i + 8] for i in range(0, len(words), 8)] or [words]
        def _w(chunk):
            weight = sum(len(w) for w in chunk) + len(chunk)
            if chunk and chunk[-1][-1:] in "؟?.!":
                weight += 6
            return max(weight, 1.0)
        weights = [_w(c) for c in word_chunks]
        total = sum(weights) or 1.0
        chunks = []
        cursor = 0.0
        for chunk, w in zip(word_chunks, weights):
            start = cursor
            cursor += duration * (w / total)
            chunks.append((chunk, start, cursor))
    def fmt(t: float) -> str:
        t = max(t, 0.0)
        h, m, s = int(t // 3600), int((t % 3600) // 60), int(t % 60)
        cs = int(round((t - int(t)) * 100))
        return f"{h:d}:{m:02}:{s:02}.{cs:02}"
    rtl = "\u200f"
    events = []
    for chunk_words, start, end in chunks:
        display_words = _clean_display_words(chunk_words)
        if not display_words:
            continue
        split_at = max(1, (len(display_words) + 1) // 2)
        line1 = rtl + _ass_escape(" ".join(display_words[:split_at]))
        chunk_text = line1
        if len(display_words) > 1:
            chunk_text += "\\N" + rtl + _ass_escape(" ".join(display_words[split_at:]))
        events.append(f"Dialogue: 0,{fmt(start)},{fmt(end)},Caption,,0,0,0,,{chunk_text}")
    ass_content = textwrap.dedent(f"""\
        [Script Info]
        ScriptType: v4.00+
        PlayResX: {VIDEO_W}
        PlayResY: {VIDEO_H}
        WrapStyle: 2
        ScaledBorderAndShadow: yes

        [V4+ Styles]
        Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
        Style: Caption,Arial,64,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,3,0,8,60,60,260,1

        [Events]
        Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
        """) + "\n".join(events) + "\n"
    out_path.write_text(ass_content, encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# Video assembly
# ---------------------------------------------------------------------------

def assemble_video(bg_video: Path, narration: Path, subtitle_path: Path,
                   out_path: Path, music_path: Path | None = None) -> Path:
    log.info("Assembling final video...")
    audio_duration = get_media_duration(narration)
    reel_duration = max(audio_duration, MIN_AUDIO_SECONDS)
    subtitle_filter_path = str(subtitle_path).replace("\\", "/").replace(":", "\\:")
    vf = (
        f"scale={VIDEO_W}:{VIDEO_H}:force_original_aspect_ratio=increase,"
        f"crop={VIDEO_W}:{VIDEO_H},subtitles='{subtitle_filter_path}'"
    )
    audio_inputs = ["-i", str(narration)]
    if music_path is not None:
        audio_inputs += ["-i", str(music_path)]
    OUTPUT_FPS = 30
    if music_path is not None:
        filter_complex = (
            f"[2:a]aloop=loop=-1:size=2e9,atrim=0:{reel_duration:.2f},"
            f"afade=t=in:st=0:d=1.5,afade=t=out:st={max(reel_duration - 1.5, 0):.2f}:d=1.5,"
            f"volume={MUSIC_VOLUME}[music];"
            f"[1:a]apad=whole_dur={reel_duration:.2f}[narration];"
            f"[narration][music]amix=inputs=2:duration=first:dropout_transition=2:normalize=0[aout]"
        )
        cmd = ["ffmpeg", "-y", "-stream_loop", "-1", "-i", str(bg_video), *audio_inputs,
               "-t", f"{reel_duration:.2f}", "-vf", vf, "-r", str(OUTPUT_FPS),
               "-filter_complex", filter_complex, "-map", "0:v:0", "-map", "[aout]",
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
               "-c:a", "aac", "-b:a", "192k", str(out_path)]
    else:
        cmd = ["ffmpeg", "-y", "-stream_loop", "-1", "-i", str(bg_video), *audio_inputs,
               "-t", f"{reel_duration:.2f}", "-vf", vf, "-r", str(OUTPUT_FPS),
               "-filter_complex", f"[1:a]apad=whole_dur={reel_duration:.2f}[aout]",
               "-map", "0:v:0", "-map", "[aout]",
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
               "-c:a", "aac", "-b:a", "192k", str(out_path)]
    def _call():
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise PipelineError(f"ffmpeg failed: {result.stderr[-2000:]}")
        if not out_path.exists() or out_path.stat().st_size == 0:
            raise PipelineError("ffmpeg produced empty output")
        return out_path
    return with_retries(_call, what="video assembly")


# ---------------------------------------------------------------------------
# GitHub hosting
# ---------------------------------------------------------------------------

def _ensure_repo_is_public(api_base: str, headers: dict[str, str]) -> None:
    resp = requests.get(api_base, headers=headers, timeout=30)
    if resp.status_code == 200 and resp.json().get("private"):
        raise PipelineError("Repository is Private; Buffer cannot fetch release assets.")


def host_video_on_github(video_path: Path, run_id: str) -> str:
    if not GH_RELEASE_TOKEN or not GITHUB_REPOSITORY:
        raise PipelineError("GH_RELEASE_TOKEN/GITHUB_REPOSITORY not set")
    api_base = f"{GITHUB_API_BASE}/repos/{GITHUB_REPOSITORY}"
    tag = f"auto-publish-{run_id}"
    headers = {"Authorization": f"Bearer {GH_RELEASE_TOKEN}", "Accept": "application/vnd.github+json"}
    _ensure_repo_is_public(api_base, headers)
    def _create():
        resp = requests.post(f"{api_base}/releases", headers=headers, json={
            "tag_name": tag, "name": f"Auto-publish video {run_id}",
            "body": "Temporary release for Buffer hosting. Safe to delete.",
            "draft": False, "prerelease": False}, timeout=30)
        if resp.status_code not in (200, 201):
            raise PipelineError(f"GitHub release failed: {resp.status_code} {resp.text[:300]}")
        return resp.json()
    release = with_retries(_create, what="create GitHub release")
    upload_url = release["upload_url"].split("{")[0]
    def _upload():
        with open(video_path, "rb") as f:
            resp = requests.post(upload_url, headers={**headers, "Content-Type": "video/mp4"},
                                 params={"name": video_path.name}, data=f, timeout=300)
        if resp.status_code not in (200, 201):
            raise PipelineError(f"GitHub upload failed: {resp.status_code} {resp.text[:300]}")
        return resp.json()
    asset = with_retries(_upload, what="upload video asset")
    url = asset["browser_download_url"]
    log.info("Video hosted: %s", url)
    return url


# ---------------------------------------------------------------------------
# Buffer publishing
# ---------------------------------------------------------------------------

_BUFFER_MUTATION = """
mutation CreatePost($channelId: ChannelId!, $text: String!, $videoUrl: String!, $metadata: PostInputMetaData) {
  createPost(input: {
    text: $text
    channelId: $channelId
    schedulingType: automatic
    mode: shareNow
    assets: [{ video: { url: $videoUrl } }]
    metadata: $metadata
  }) {
    ... on PostActionSuccess { post { id text dueAt } }
    ... on MutationError { message }
  }
}
"""

_GET_CHANNEL_QUERY = """
query GetChannel($id: ChannelId!) {
  channel(input: { id: $id }) { id service }
}
"""

YOUTUBE_CATEGORY_ID = os.getenv("YOUTUBE_CATEGORY_ID", "24")


def get_channel_service(channel_id: str) -> str:
    def _call():
        resp = requests.post(BUFFER_ENDPOINT, headers={
            "Authorization": f"Bearer {BUFFER_API_KEY}",
            "Content-Type": "application/json"}, json={
            "query": _GET_CHANNEL_QUERY, "variables": {"id": channel_id}}, timeout=30)
        if resp.status_code != 200:
            raise PipelineError(f"Buffer lookup failed: {resp.status_code}")
        data = resp.json()
        if data.get("errors"):
            raise PipelineError(f"Buffer error: {data['errors']}")
        service = ((data.get("data") or {}).get("channel") or {}).get("service")
        if not service:
            raise PipelineError(f"Could not resolve service for {channel_id}")
        return service
    return with_retries(_call, what=f"lookup channel {channel_id}")


def build_channel_metadata(service: str, topic: Topic) -> dict[str, Any] | None:
    s = (service or "").lower()
    if s == "youtube":
        return {"youtube": {"title": topic.title[:100], "categoryId": YOUTUBE_CATEGORY_ID}}
    if s == "facebook":
        return {"facebook": {"type": "reel"}}
    return None


def publish_video(video_path: Path, topic: Topic, channel_ids: list[str]) -> dict[str, Any]:
    log.info("Publishing to %d channel(s)", len(channel_ids))
    hashtags_str = " ".join(topic.hashtags)
    text = f"{topic.title}\n\n{topic.caption}\n\n{hashtags_str}".strip()
    run_id = time.strftime("%Y%m%d-%H%M%S")
    video_url = host_video_on_github(video_path, run_id)
    results: dict[str, Any] = {}
    failed: list[str] = []
    for channel_id in channel_ids:
        service = get_channel_service(channel_id)
        metadata = build_channel_metadata(service, topic)
        def _call(cid=channel_id, md=metadata):
            resp = requests.post(BUFFER_ENDPOINT, headers={
                "Authorization": f"Bearer {BUFFER_API_KEY}",
                "Content-Type": "application/json"}, json={
                "query": _BUFFER_MUTATION,
                "variables": {"channelId": cid, "text": text, "videoUrl": video_url, "metadata": md}},
                timeout=60)
            if resp.status_code != 200:
                raise PipelineError(f"Buffer API error {resp.status_code}: {resp.text[:500]}")
            return resp.json()
        response = with_retries(_call, what=f"publish to {channel_id}")
        if response.get("errors"):
            failed.append(channel_id)
            results[channel_id] = {"success": False, "errors": response["errors"]}
            continue
        payload = (response.get("data") or {}).get("createPost") or {}
        if "message" in payload and "post" not in payload:
            failed.append(channel_id)
            results[channel_id] = {"success": False, "error": payload["message"]}
        else:
            results[channel_id] = {"success": True, "post": payload.get("post", {})}
            log.info("Published to %s", channel_id)
    if failed:
        raise PipelineError(f"Publishing failed for: {', '.join(failed)}")
    return results


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_pipeline() -> None:
    check_env()
    if DRY_RUN:
        if not DRY_RUN_INPUT_FILE.exists():
            raise PipelineError(f"Dry-run file missing: {DRY_RUN_INPUT_FILE}")
        raw_text = DRY_RUN_INPUT_FILE.read_text(encoding="utf-8").strip()
        corrected = proofread_narration_tashkeel(raw_text)
        corrected = enforce_text_quality(
            corrected,
            reviser=lambda t, i: proofread_narration_tashkeel(t, issues=i),
        )
        red_flag = find_content_red_flag(corrected)
        print(json.dumps({"passed": red_flag is None, "red_flag": red_flag,
                          "corrected_text": corrected}, ensure_ascii=False, indent=2))
        if red_flag:
            raise PipelineError(f"Dry-run rejected: {red_flag}")
        return

    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = WORK_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log.info("Run directory: %s", run_dir)

    topic, prefetched_sources, preflight_errors = generate_topic()

    if EDITORIAL_REVIEW_ENABLED:
        topic.narration_script = _proofread_topic_narration(topic, topic.narration_script)
    else:
        log.warning("Editorial LLM review disabled; applying strict local Arabic quality gate")
    topic.narration_script = enforce_text_quality(
        topic.narration_script,
        reviser=lambda text, issues: proofread_narration_tashkeel(
            text, issues=issues, canonical_subject=topic.title,
            verified_fact=topic.verified_fact or None,
        ),
    )
    topic.narration_script = _append_engagement_outro(topic.narration_script)
    final_word_count = len(topic.narration_script.split())
    _validate_final_script_word_count(final_word_count)
    red_flag = find_content_red_flag(topic.narration_script)
    if red_flag:
        raise PipelineError(f"Rejected hallucinated content term: {red_flag}")

    if FACT_CHECK_ENABLED:
        fact_report = fact_check_topic(
            topic,
            run_dir / "fact_check.json",
            prefetched_sources=prefetched_sources,
            preflight_source_errors=preflight_errors,
        )
        if fact_report.get("status") != "PASS":
            raise PipelineError(
                "Fact Check failed. Details: "
                f"{fact_report.get('errors', [])}. "
                f"Claim verdicts: {[c.get('verdict') for c in fact_report.get('claims', [])]}"
            )
        log.info("Fact Check passed: %d supported claims", len(fact_report.get("claims", [])))
    else:
        fact_report = {
            "status": "SKIPPED",
            "reason": "FACT_CHECK_ENABLED=false; simple workflow mode",
            "title": topic.title,
            "errors": [],
        }
        (run_dir / "fact_check.json").write_text(
            json.dumps(fact_report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        log.warning("LLM Fact Check disabled; local red-flag and length checks remain active")

    (run_dir / "topic.json").write_text(
        json.dumps(topic.__dict__, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    voice_profile = choose_voice_profile(topic=topic)
    topic.voice_profile = voice_profile
    (run_dir / "topic.json").write_text(
        json.dumps(topic.__dict__, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log.info("Voice profile: %s", voice_profile)

    max_attempts = 3
    narration_path = None
    audio_duration = 0.0
    for attempt in range(1, max_attempts + 1):
        narration_path = generate_tts(topic.narration_script, run_dir / "narration.mp3",
                                       reference_profile=voice_profile)
        audio_duration = get_media_duration(narration_path)
        log.info("Narration: %.1fs (attempt %d/%d)", audio_duration, attempt, max_attempts)
        if _audio_duration_needs_normalization(audio_duration):
            audio_duration = normalize_narration_duration(narration_path, TARGET_AUDIO_SECONDS)
        if not _audio_duration_exceeds_maximum(audio_duration):
            break
        if attempt == max_attempts:
            raise PipelineError(f"Audio still over {MAX_AUDIO_SECONDS}s after {max_attempts} attempts")

    whisper_aligned, pronunciation_flags = realign_subtitles_with_whisper(narration_path, topic.narration_script)
    (run_dir / "whisper_alignment_status.json").write_text(
        json.dumps({"whisper_aligned": whisper_aligned}, ensure_ascii=False), encoding="utf-8"
    )
    (run_dir / "pronunciation_review.json").write_text(
        json.dumps(pronunciation_flags, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    remember_topic(topic)
    scene_urls = search_pexels_videos(topic.scene_keywords_en, topic.search_keywords_en)
    scene_paths = [download_file(u, run_dir / f"scene_{i:02d}.mp4") for i, u in enumerate(scene_urls)]
    word_timings = _load_word_timings(narration_path.with_suffix(".timings.json"))
    reel_duration = max(audio_duration, MIN_AUDIO_SECONDS)
    if _reel_duration_needs_padding(audio_duration):
        log.info("Padding reel to %.1fs (audio was %.1fs)", reel_duration, audio_duration)
    segment_durations = compute_scene_durations(word_timings, reel_duration, len(scene_paths))
    bg_video_path = build_multishot_background(scene_paths, segment_durations, run_dir / "background.mp4")
    music_path = get_bg_music(run_dir)
    subtitle_path = build_subtitles(
        topic.narration_script, audio_duration, run_dir / "subtitles.ass",
        timings_path=narration_path.with_suffix(".timings.json"),
    )
    final_video_path = assemble_video(bg_video_path, narration_path, subtitle_path,
                                      run_dir / "final.mp4", music_path=music_path)
    final_duration = get_media_duration(final_video_path)
    if final_duration < MIN_AUDIO_SECONDS - 0.25 or final_duration > MAX_AUDIO_SECONDS:
        raise PipelineError(
            f"Assembled video is {final_duration:.2f}s; required window is "
            f"{MIN_AUDIO_SECONDS:.0f}..{MAX_AUDIO_SECONDS:.0f}s"
        )
    log.info("Video assembled: %.2fs", final_duration)

    result = publish_video(final_video_path, topic, BUFFER_CHANNEL_IDS)
    (run_dir / "publish_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log.info("Published successfully")


def main() -> int:
    try:
        run_pipeline()
    except PipelineError as exc:
        log.error("Pipeline failed: %s", exc)
        return 1
    except Exception as exc:
        log.exception("Unexpected error: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
