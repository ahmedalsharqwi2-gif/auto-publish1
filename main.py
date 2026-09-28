#!/usr/bin/env python3
"""
Automated Content Creation & Publishing Pipeline
==================================================
Generates a short-form vertical video (Arabic voiceover + burned-in subtitles +
Pexels stock footage) from an AI-generated viral topic, then publishes it to
YouTube / TikTok / Facebook via the Buffer API (buffer.com).

Provider: OpenRouter (multi-model fallback list of free models)
"""

from __future__ import annotations

import asyncio
import difflib
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
from tts_quality import enforce_text_quality, generate_silma_guarded, load_reference, resolve_reference_profile
from fact_check import fact_check_topic, preflight_topic_sources

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
TOPIC_BANK_FILE = Path(os.getenv("TOPIC_BANK_FILE", "config/topic_bank.json"))
DRY_RUN = os.getenv("DRY_RUN", "false").lower() == "true"
DRY_RUN_INPUT_FILE = Path(os.getenv("DRY_RUN_INPUT_FILE", "tests/fixtures/bad_narration.txt"))

MIN_AUDIO_SECONDS = float(os.getenv("MIN_AUDIO_SECONDS", "60"))
MAX_AUDIO_SECONDS = float(os.getenv("MAX_AUDIO_SECONDS", "90"))
TARGET_AUDIO_SECONDS = float(os.getenv("TARGET_AUDIO_SECONDS", str(max(1.0, MAX_AUDIO_SECONDS - 5.0))))
MAX_SCRIPT_WORDS = int(os.getenv("MAX_SCRIPT_WORDS", "165"))
MIN_SCRIPT_WORDS = int(os.getenv("MIN_SCRIPT_WORDS", "120"))

REQUIRE_EXTERNAL_SOURCES = os.getenv("REQUIRE_EXTERNAL_SOURCES", "false").lower() == "true"
TOPIC_GENERATION_MAX_ATTEMPTS = int(os.getenv("TOPIC_GENERATION_MAX_ATTEMPTS", "5"))
HISTORY_LIMIT = int(os.getenv("HISTORY_LIMIT", "1000"))
MIN_SCENE_CLIPS = int(os.getenv("MIN_SCENE_CLIPS", "10"))


def _clean_env(name: str) -> str | None:
    raw = os.getenv(name)
    if raw is None:
        return None
    cleaned = re.sub(r"\s+", "", raw)
    return cleaned or None


PEXELS_API_KEY = _clean_env("PEXELS_API_KEY")

BUFFER_API_KEY = _clean_env("BUFFER_API_KEY")
BUFFER_CHANNEL_IDS = [
    c.strip() for c in os.getenv("BUFFER_CHANNEL_IDS", "").split(",") if c.strip()
]

GH_RELEASE_TOKEN = _clean_env("GH_RELEASE_TOKEN") or _clean_env("GITHUB_TOKEN")
GITHUB_REPOSITORY = os.getenv("GITHUB_REPOSITORY")

# ---------------------------------------------------------------------------
# OpenRouter configuration (multi-model fallback for maximum stability)
# ---------------------------------------------------------------------------
OPENROUTER_API_KEY = _clean_env("OPENROUTER_API_KEY")
OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_REFERER = os.getenv(
    "OPENROUTER_REFERER",
    f"https://github.com/{GITHUB_REPOSITORY}" if GITHUB_REPOSITORY else "https://github.com/",
)
OPENROUTER_TITLE = os.getenv("OPENROUTER_TITLE", "Auto Publish Reels")
OPENROUTER_RATE_LIMIT_MAX_RETRIES = int(os.getenv("OPENROUTER_RATE_LIMIT_MAX_RETRIES", "4"))

# A prioritized list of free models to try in order. If one fails (404,
# reasoning-only response, rate limit, etc.), the pipeline automatically
# moves on to the next. The single model can still be pinned by setting
# OPENROUTER_MODEL env var explicitly to a single model id.
_OPENROUTER_MODEL_LIST_RAW = os.getenv(
    "OPENROUTER_MODEL",
    "qwen/qwen-2.5-7b-instruct:free,"
    "mistralai/mistral-nemo:free,"
    "meta-llama/llama-3.2-3b-instruct:free,"
    "google/gemma-2-9b-it:free,"
    "microsoft/phi-3-mini-128k-instruct:free",
)
OPENROUTER_MODELS = [m.strip() for m in _OPENROUTER_MODEL_LIST_RAW.split(",") if m.strip()]
OPENROUTER_MODEL = OPENROUTER_MODELS[0]  # kept for logging/compat

GROQ_MODEL = OPENROUTER_MODEL
GROQ_MAX_COMPLETION_TOKENS = int(os.getenv("GROQ_MAX_COMPLETION_TOKENS", "2200"))

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

PEXELS_SEARCH_ENDPOINT = "https://api.pexels.com/videos/search"
TOP_RELEVANT_CANDIDATES = int(os.getenv("TOP_RELEVANT_CANDIDATES", "8"))
BUFFER_ENDPOINT = "https://api.buffer.com"
GITHUB_API_BASE = "https://api.github.com"

VIDEO_W, VIDEO_H = 1080, 1920
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 5

BG_MUSIC_URL = _clean_env("BG_MUSIC_URL")
MUSIC_VOLUME = float(os.getenv("MUSIC_VOLUME", "0.07"))
MUSIC_DIR = Path(__file__).resolve().parent / "music"
DEFAULT_BG_MUSIC_URL = (
    "https://raw.githubusercontent.com/effacestudios/Royalty-Free-Music-Pack/master/Bubbles.mp3"
)

REQUIRED_ENV = {
    "OPENROUTER_API_KEY": OPENROUTER_API_KEY,
    "PEXELS_API_KEY": PEXELS_API_KEY,
    "BUFFER_API_KEY": BUFFER_API_KEY,
    "GH_RELEASE_TOKEN": GH_RELEASE_TOKEN,
    "GITHUB_REPOSITORY": GITHUB_REPOSITORY,
}


class PipelineError(Exception):
    """Raised for any unrecoverable pipeline failure."""


class SourcePreflightError(PipelineError):
    """Raised when no unused topic has any accessible evidence source."""


TOPIC_CATEGORIES = [
    "غرائب دينية موثقة",
    "عجائب عالم الحيوان",
    "غرائب جسم الإنسان والطب",
    "حقائق علمية صادمة",
    "أسرار الفضاء والمحيطات",
    "ظواهر طبيعية نادرة",
    "قصص تاريخية غريبة",
    "حضارات وعادات وثقافات غير مألوفة",
    "اختراعات وظواهر تقنية",
    "أماكن غامضة",
    "حقائق نفسية واجتماعية",
]


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
    bank_id: str = ""
    verified_fact: str = ""
    source_urls: list[str] = field(default_factory=list)
    voice_profile: str = ""


def set_canonical_topic_title(topic: Topic, seed: dict[str, Any]) -> None:
    subject = str(seed.get("subject", "")).strip()
    if not subject:
        raise PipelineError("Selected topic-bank seed is missing its canonical subject")
    topic.title = subject


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def check_env() -> None:
    missing = [k for k, v in REQUIRED_ENV.items() if not v]
    if not BUFFER_CHANNEL_IDS:
        missing.append("BUFFER_CHANNEL_IDS")
    if missing:
        raise PipelineError(f"Missing required environment variables: {', '.join(missing)}")


def with_retries(fn, *args, what: str = "operation", max_retries: int | None = None, **kwargs):
    attempts = max_retries if max_retries is not None else MAX_RETRIES
    last_err: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fn(*args, **kwargs)
        except SourcePreflightError:
            raise
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
    s = re.sub(r"\s+", " ", s).strip()
    return s


_TASHKEEL_CHECK_ALLOWLIST = {
    "به",
    "عليك",
    "هذه",
}


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
            f"CTA_OUTRO has word(s) ending in a bare ك/ه with no diacritic "
            f"on that letter: {offenders}. Add the missing tashkeel or a "
            f"justified entry to _TASHKEEL_CHECK_ALLOWLIST. Full string: {CTA_OUTRO!r}"
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
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ،.!؟\u061F")
    return cleaned


def _topic_fingerprint(topic: Topic | dict[str, Any]) -> str:
    values = [topic.get(k, "") if isinstance(topic, dict) else getattr(topic, k, "")
              for k in ("title", "hook_text", "search_keywords_en")]
    return re.sub(r"[^\w\u0600-\u06ff]+", " ", " ".join(map(str, values))).strip().lower()


def load_topic_bank() -> list[dict[str, Any]]:
    try:
        data = json.loads(TOPIC_BANK_FILE.read_text(encoding="utf-8"))
        topics = data.get("topics", []) if isinstance(data, dict) else []
        if not isinstance(topics, list) or len(topics) < 730:
            raise PipelineError(f"Topic bank must contain at least 730 entries: {TOPIC_BANK_FILE}")
        return [x for x in topics if isinstance(x, dict) and x.get("id") and x.get("verified_fact")]
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(f"Could not load topic bank {TOPIC_BANK_FILE}: {exc}") from exc


def _topic_source_signature(topic: dict[str, Any]) -> tuple[str, ...]:
    return tuple(sorted({str(url).strip() for url in topic.get("source_urls", []) if str(url).strip()}))


def choose_topic_seed(
    history: list[dict[str, Any]],
    blocked_categories: set[str] | None = None,
    excluded_source_signatures: set[tuple[str, ...]] | None = None,
) -> dict[str, Any]:
    bank = load_topic_bank()
    used = {str(x.get("bank_id", "")) for x in history if x.get("bank_id")}
    excluded = excluded_source_signatures or set()
    available = [
        x for x in bank
        if str(x.get("id")) not in used and _topic_source_signature(x) not in excluded
    ]
    blocked = blocked_categories or set()
    varied = [x for x in available if str(x.get("category", "")) not in blocked]
    if varied:
        available = varied
    if not available:
        if excluded:
            raise SourcePreflightError(
                f"No unused topic has accessible citations after checking {len(excluded)} unique source set(s)"
            )
        raise PipelineError("All topic-bank entries have already been used; archive or reset topic_history.json")
    return random.SystemRandom().choice(available)


def choose_reachable_topic_seed(
    history: list[dict[str, Any]],
    blocked_categories: set[str] | None = None,
    excluded_source_signatures: set[tuple[str, ...]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, str]], list[dict[str, str]]]:
    excluded = excluded_source_signatures if excluded_source_signatures is not None else set()
    while True:
        seed = choose_topic_seed(history, blocked_categories, excluded)
        if not REQUIRE_EXTERNAL_SOURCES:
            selected = dict(seed)
            selected["source_urls"] = []
            log.info(
                "Using vetted topic-bank fact for %s; external source fetching is disabled",
                seed["id"],
            )
            return selected, [], []
        signature = _topic_source_signature(seed)
        sources, errors = preflight_topic_sources(list(seed.get("source_urls", [])))
        if sources:
            selected = dict(seed)
            selected["source_urls"] = [source["url"] for source in sources]
            if errors:
                log.warning(
                    "Topic seed %s has %d inaccessible citation(s); proceeding with %d fetched source(s)",
                    seed["id"], len(errors), len(sources),
                )
            return selected, sources, errors
        if not REQUIRE_EXTERNAL_SOURCES:
            selected = dict(seed)
            selected["source_urls"] = []
            log.warning(
                "Topic seed %s has no reachable external source; continuing with the vetted "
                "topic-bank fact only",
                seed["id"],
            )
            return selected, [], errors
        excluded.add(signature)
        log.warning("Skipping topic seed %s before generation: all cited sources are inaccessible", seed["id"])


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
        ids = [str(profile_id) for profile_id in profiles if str(profile_id).strip()]
        if not ids:
            raise ValueError("no voice profiles configured")
        default = str(data.get("default", ids[0]))
        if default not in ids:
            default = ids[0]
        return ids, default
    except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
        raise PipelineError(f"Could not load voice profiles {path}: {exc}") from exc


def choose_voice_profile(history: list[dict[str, Any]] | None = None) -> str:
    profiles, default = load_voice_profile_ids()
    if not VOICE_ROTATION_ENABLED:
        selected = SILMA_REFERENCE_PROFILE or default
        if selected not in profiles:
            raise PipelineError(
                f"Unknown SILMA_REFERENCE_PROFILE {selected!r}; available profiles: {profiles}"
            )
        return selected

    history = history if history is not None else load_topic_history()
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
    history.append({"title": topic.title, "hook_text": topic.hook_text,
                    "search_keywords_en": topic.search_keywords_en,
                    "category": topic.category,
                    "bank_id": topic.bank_id,
                    "voice_profile": topic.voice_profile,
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    TOPIC_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    TOPIC_HISTORY_FILE.write_text(json.dumps(history[-HISTORY_LIMIT:], ensure_ascii=False, indent=2), encoding="utf-8")


def extract_json_block(text: str) -> dict[str, Any]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise PipelineError(f"Could not find JSON object in model response: {text[:300]}")
    return json.loads(match.group(0))


# ---------------------------------------------------------------------------
# Step 1: Select a vetted topic-bank entry, then draft its script
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent(
    """
    أنت محرر سكريبتات عربية، ولست مولّد موضوعات. البرنامج اختار مسبقًا موضوعًا
    من بنك مُراجع؛ لا تختَر موضوعًا جديدًا، ولا تستبدل الكائن أو الظاهرة، ولا تضف
    إلى العنوان صفة أو رقمًا أو مقارنة غير موجودة في سجل البنك.

    مصدر الحقيقة الوحيد هو verified_fact والصفحات المحددة في source_urls التي
    وصلت في رسالة المستخدم. angle تلميح أسلوبي اختياري، وليس دليلًا أو إذنًا
    بإضافة مقارنة. إذا تعارضت زاوية مع الحقيقة أو لم تسندها المصادر، تجاهل الزاوية
    واشرح الحقيقة مباشرة. عند الشك احذف التفصيل بدل تخمينه.

    أخرج JSON صالحًا فقط بالمفاتيح التالية:
    {
      "category": "انسخ الفئة الواردة في سجل البنك حرفيًا",
      "hook_text": "سؤال عربي فصيح قصير، لا يتجاوز 12 كلمة، ولا يفترض حقيقة غير موثقة",
      "narration_script": "نص عربي مترابط يبدأ بـ hook_text حرفيًا ويشرح الحقيقة المسجلة فقط، دون خاتمة تفاعلية",
      "title": "انسخ subject الوارد في سجل البنك حرفيًا دون إضافة أي كلمة",
      "caption": "جملة أو جملتان تصفان الموضوع والحقيقة نفسها دون ادعاء جديد",
      "hashtags": ["#وسم1", "#وسم2", "#وسم3", "#وسم4", "#وسم5"],
      "search_keywords_en": "عبارة إنجليزية قصيرة من كلمتين إلى ثلاث عن الموضوع المحدد",
      "scene_keywords_en": ["4 إلى 7 أوصاف إنجليزية قصيرة لمشاهد مرئية مرتبطة فعلًا بفقرات النص"]
    }

    قواعد الدقة:
    - لا تضف أرقامًا أو قياسات أو دقة أو آلية سببية أو مقارنة أو أسماء جديدة ما لم
      تذكرها verified_fact أو تدعمها صراحة صفحة من source_urls.
    - لا تجعل السؤال الافتتاحي يوحي بادعاء أقوى من الحقيقة المسجلة. لا يلزم أن
      يحتوي على رقم أو مقارنة أو مفاجأة مصطنعة؛ سؤال واضح وصادق أفضل من هوك مضلل.
    - لا تضف معلومات عامة عن الموضوع لمجرد إكمال عدد الكلمات؛ اشرح الحقيقة بالقدر
      الكافي فقط دون حشو، واحذف أي جملة لا يمكن ردّها إلى دليل.
    - اكتب العربية الفصحى السليمة. شكّل hook_text وnarration_script تشكيلًا كاملًا
      صحيحًا قدر الإمكان لتوجيه النطق، ولا تستخدم العامية أو ألفاظًا مخترعة.
    - يبدأ narration_script بنص hook_text نفسه، وينتهي بعد اكتمال الشرح؛ لا تضف
      طلب إعجاب أو مشاركة أو اشتراك، فالبرنامج يضيف العبارة الختامية الثابتة.
    - التزم بالحد الأعلى للكلمات وعدد المشاهد المطلوب فقط.
    - اجعل كلمات البحث والمشاهد الإنجليزية تصف الشيء المذكور فعلًا في الموضوع أو
      النص، ولا تستخدم كلمات عامة أو لقطات لا علاقة لها به.

    تذكير: لا تغيّر الموضوع أو الحقيقة أو الفئة أو العنوان. إذا لم تسمح الأدلة
    بنص مثير، فاكتب نصًا بسيطًا صحيحًا ولا تخترع الإثارة.
    """
).strip()


# ---------------------------------------------------------------------------
# OpenRouter chat with automatic multi-model fallback
# ---------------------------------------------------------------------------
def _groq_chat(
    messages: list[dict[str, str]],
    max_completion_tokens: int = GROQ_MAX_COMPLETION_TOKENS,
    temperature: float = 0.75,
) -> str:
    """OpenRouter chat-completion with automatic multi-model fallback.

    Iterates through OPENROUTER_MODELS in order. For each candidate:
      - 404 (model not available)  -> try next model immediately
      - 429 (rate limit)           -> brief wait, then try next model
      - 200 with null content      -> try next model (reasoning-only bug)
      - 200 with JSON content      -> return it

    Raises only if every model in the list fails.
    """
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": OPENROUTER_REFERER,
        "X-Title": OPENROUTER_TITLE,
    }

    last_error: str = "no models attempted"
    for model_idx, model_name in enumerate(OPENROUTER_MODELS, start=1):
        payload: dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "response_format": {"type": "json_object"},
            "max_tokens": max_completion_tokens,
            "temperature": temperature,
        }

        for rate_limit_attempt in range(1, OPENROUTER_RATE_LIMIT_MAX_RETRIES + 1):
            try:
                resp = requests.post(
                    OPENROUTER_ENDPOINT,
                    headers=headers,
                    json=payload,
                    timeout=120,
                )
            except Exception as exc:
                last_error = f"{model_name}: request failed: {exc}"
                log.warning("Model %d/%d (%s) request failed: %s",
                            model_idx, len(OPENROUTER_MODELS), model_name, exc)
                break

            if resp.status_code == 404:
                last_error = f"{model_name}: 404 unavailable"
                log.warning(
                    "Model %d/%d (%s) unavailable (404); trying next model",
                    model_idx, len(OPENROUTER_MODELS), model_name,
                )
                break

            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                wait_seconds = 8.0
                if retry_after:
                    try:
                        wait_seconds = float(retry_after)
                    except ValueError:
                        pass
                else:
                    match = re.search(r"try again in\s+([0-9.]+)s", resp.text, re.IGNORECASE)
                    if match:
                        wait_seconds = float(match.group(1))
                wait_seconds = min(max(wait_seconds, 3.0), 60.0)
                log.warning(
                    "Model %d/%d (%s) rate-limited; waiting %.1fs (%d/%d)",
                    model_idx, len(OPENROUTER_MODELS), model_name,
                    wait_seconds, rate_limit_attempt, OPENROUTER_RATE_LIMIT_MAX_RETRIES,
                )
                time.sleep(wait_seconds + 1.0)
                if rate_limit_attempt < OPENROUTER_RATE_LIMIT_MAX_RETRIES:
                    continue
                last_error = f"{model_name}: persistent 429"
                break

            if resp.status_code != 200:
                last_error = f"{model_name}: HTTP {resp.status_code} {resp.text[:200]}"
                log.warning(
                    "Model %d/%d (%s) returned HTTP %d; trying next model",
                    model_idx, len(OPENROUTER_MODELS), model_name, resp.status_code,
                )
                break

            data = resp.json()
            try:
                choice = data.get("choices", [{}])[0]
                finish_reason = choice.get("finish_reason")
                raw_text = (choice.get("message") or {}).get("content") or ""
            except Exception as exc:
                last_error = f"{model_name}: malformed response: {exc}"
                log.warning("Model %s malformed response: %s", model_name, exc)
                break

            if not raw_text.strip():
                last_error = f"{model_name}: empty content (finish_reason={finish_reason})"
                log.warning(
                    "Model %d/%d (%s) returned empty content (finish_reason=%s); "
                    "trying next model",
                    model_idx, len(OPENROUTER_MODELS), model_name, finish_reason,
                )
                break

            log.info(
                "OpenRouter: using model %d/%d (%s)",
                model_idx, len(OPENROUTER_MODELS), model_name,
            )
            return raw_text

    raise PipelineError(
        f"All {len(OPENROUTER_MODELS)} OpenRouter models failed. Last error: {last_error}"
    )


PROOFREAD_SYSTEM_PROMPT = textwrap.dedent(
    """
    أنت محرر عربي فصيح ومدقق علمي صارم، ولست مدقق تشكيل فقط. سيصلك نص قصير
    مرشح للنشر. أعد كتابته كاملاً بعد إصلاح النحو والصرف والأسلوب، والتحقق من
    المنطق العلمي والتاريخي داخل النص.

    قواعد لا يجوز خرقها:
    إذا احتوى مدخل المستخدم على canonical_subject وverified_fact فهما مرجعان ثابتان:
    لا تغيّر الموضوع أو الحقيقة، ولا تستبدلهما بظاهرة أخرى؛ اقتصر على إصلاح اللغة
    والتشكيل وحذف الادعاء غير المدعوم، وسيحسم فحص الأدلة النهائي صلاحية النص.
    1) اكتب عربية فصحى سليمة فقط؛ ممنوع العامية، التراكيب المترجمة حرفياً،
       الكلمات الوهمية، أو الجمل التي لا معنى لها.
    2) لا تخترع مصطلحاً أو وحدة قياس أو آلية فيزيائية أو معلومة جديدة. إذا كان
       ادعاء في المسودة لا يطابق verified_fact، احذفه أو صححه دون تغيير الموضوع
       ودون إدخال حقيقة بديلة من الذاكرة.
       تنبيه حاسم: عبارات مثل «البركان الثلجي» و«وحدة السيلينس» و«الشهرات
       الجوية» و«الماء المحتل بالدقيق» ليست مصطلحات علمية مقبولة في هذا النص؛
       احذفها تماماً ولا تعِد تسميتها أو شرحها. استبدل الفكرة بجملة صريحة مثل:
       «هذا الوصف المتداول غير دقيق؛ فالبركان لا يصنع الثلج من دون ماء، وقد
       تتكون بلورات جليدية في العمود البركاني بسبب رطوبة الجو». لا تنقل هذا المثال
       إلى النص إلا إذا كان موضوع البنك هو البراكين وكانت مصادره تدعمه.
    3) صحح التطابق والإعراب والضمائر وعلامات الترقيم، واجعل كل جملة قابلة
       للفهم عند قراءتها منفردة. احذف الحشو والتكرار.
    4) لا تضف أرقاماً أو أسماء أو نتائج إلا إذا ذكرها verified_fact أو دعمتها
       صراحةً مصادر الموضوع؛ عند الشك احذف التفصيل.
    5) حافظ على سؤال البداية وموضوع البنك ومعناه. صحح الصياغة فقط، ولا تستبدل
       الموضوع أو الحقيقة بظاهرة أخرى. اترك النص ينتهي بعد اكتمال الشرح، ولا تضف
       أي خاتمة تفاعلية أو دعوة للإعجاب أو المشاركة.
    6) ضع التشكيل الكامل المناسب للنطق، لكن لا تجعل التشكيل يغطي خطأً لغوياً؛
       صحة الكلمات والمعنى أولاً.
    7) إذا وُجد مفتاح issues_to_fix في الرسالة فأصلح كل مشكلة مذكورة فيه صراحةً:
       احذف أي حرف غير عربي، وأكمل التشكيل الناقص على كل كلمة، واكتب الأرقام بالحروف.
    8) راجع كل جملة نحويًا قبل إخراجها. ممنوع تركيب «إذا غير موجود ...»؛ اكتب «إذا لم يوجد/توجد ...».
       وممنوع استعمال «رفع العمود» عندما يكون المقصود «ارتفاع العمود». لا تُخرج جملة
       غير مكتملة أو عبارة تبدو مترجمة حرفيًا، حتى لو كان معناها العام مفهومًا.

    أجب حصراً بكائن JSON بمفتاح واحد:
    {"corrected_text": "النص العربي الكامل بعد المراجعة"}
    """
).strip()


CONTENT_RED_FLAGS = (
    "السيلينس", "الشهرات الجوية", "المحتلة بالدقيق", "البركان الثلجي",
    "الهزات الرقمية", "مغنطيس قمري يستخلص", "مغناطيس قمري يستخلص",
    "انفجار نجمي قريب", "حدث انفجار نجمي", "الدراسات الأسترادية",
    "تأثر ميكانيكي وجوي", "مهنته هي تحمية الكوكب", "تحمية الكوكب",
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
        raw_text = _groq_chat(messages, max_completion_tokens=GROQ_MAX_COMPLETION_TOKENS, temperature=0.15)
        parsed = extract_json_block(raw_text)
        corrected = str(parsed.get("corrected_text", "")).strip()
        if not corrected:
            raise PipelineError("Editorial review returned an empty corrected_text")
        return corrected

    corrected = with_retries(_call, what="Arabic and factual editorial review")
    orig_words = len(_normalize_for_compare(script).split())
    new_words = len(_normalize_for_compare(corrected).split())
    minimum_preserved_words = max(1, (orig_words * 70 + 99) // 100)
    maximum_review_words = max(1, (orig_words * 135) // 100)
    if new_words < minimum_preserved_words or new_words > maximum_review_words:
        raise PipelineError(
            f"Editorial review changed script length too much ({orig_words} -> {new_words} words; "
            f"allowed {minimum_preserved_words}..{maximum_review_words})"
        )
    if len(re.findall(r"[\u0600-\u06FF]", corrected)) < max(1, new_words * 2):
        raise PipelineError("Editorial review returned insufficient Arabic text")
    log.info("Arabic/scientific editorial gate passed (%d -> %d words)", orig_words, new_words)
    return corrected


def _validate_final_script_word_count(word_count: int) -> None:
    if word_count < MIN_SCRIPT_WORDS:
        raise PipelineError(
            f"Final narration is {word_count} words; minimum is {MIN_SCRIPT_WORDS} "
            "including the fixed outro (needed for a 60s+ reel)"
        )
    if word_count > MAX_SCRIPT_WORDS:
        raise PipelineError(
            f"Final narration is {word_count} words; maximum is {MAX_SCRIPT_WORDS} "
            "including the fixed outro"
        )


def build_topic_user_prompt(seed: dict[str, Any], accessible_sources: list[dict[str, str]]) -> str:
    source_urls = [str(source.get("url", "")).strip() for source in accessible_sources]
    source_urls = [url for url in source_urls if url]
    if not source_urls and REQUIRE_EXTERNAL_SOURCES:
        raise PipelineError("Cannot write a topic without at least one preflighted source")

    pre_outro_max = MAX_SCRIPT_WORDS - OUTRO_MAX_WORDS
    pre_outro_min = max(MIN_SCRIPT_WORDS - OUTRO_MAX_WORDS, 40)
    bank_entry = {
        "bank_id": seed["id"],
        "category": seed["category"],
        "subject": seed["subject"],
        "angle": seed["angle"],
        "verified_fact": seed["verified_fact"],
        "source_urls": source_urls,
    }
    return (
        "اكتب نص الفيديو فقط للمدخل الجاهز التالي من بنك الموضوعات. الموضوع والفئة "
        "والعنوان محددة مسبقًا ولا يجوز تغييرها أو اقتراح موضوع بديل. استخدم angle "
        "كتلميح أسلوبي اختياري فقط؛ لا تضف مقارنة أو رقمًا أو تفصيلًا لا يسنده verified_fact "
        "أو مصدر متاح. يجب أن يبقى كل ادعاء في السؤال والنص والكابشن ضمن الدليل المرفق.\n\n"
        + json.dumps(bank_entry, ensure_ascii=False)
        + (
            f"\n\nاكتب narration_script بحيث يشرح الحقيقة بالقدر الكافي، ويكون عدد كلماته "
            f"بين {pre_outro_min} و{pre_outro_max} كلمة قبل العبارة الختامية الثابتة "
            f"(المطلوب النهائي بين {MIN_SCRIPT_WORDS} و{MAX_SCRIPT_WORDS} كلمة مع العبارة الختامية). "
            "ابدأه بـ hook_text نفسه. أعد category وtitle كما هما تمامًا من المدخل، "
            "واجعل scene_keywords_en بين 4 و7 عبارات مرتبطة بصريًا بالنص. أخرج JSON فقط."
        )
    )


def generate_topic() -> tuple[Topic, list[dict[str, str]], list[dict[str, str]]]:
    log.info("Writing script for a pre-vetted topic-bank entry via OpenRouter (%s)...", OPENROUTER_MODEL)
    prefetched_sources: list[dict[str, str]] = []
    preflight_source_errors: list[dict[str, str]] = []
    excluded_source_signatures: set[tuple[str, ...]] = set()

    def _parse_topic(raw_text: str) -> tuple[Topic, int]:
        parsed = extract_json_block(raw_text)
        topic = Topic(
            hook_text=parsed["hook_text"].strip(),
            narration_script=parsed.get("narration_script", "").strip(),
            title=str(parsed.get("title", "")).strip(),
            caption=parsed.get("caption", "").strip(),
            hashtags=list(parsed.get("hashtags", [])),
            search_keywords_en=parsed.get("search_keywords_en", "nature abstract").strip(),
            scene_keywords_en=[str(x).strip() for x in parsed.get("scene_keywords_en", []) if str(x).strip()],
            category=str(parsed.get("category", "")).strip(),
        )
        if not topic.hook_text:
            raise PipelineError("Model returned an empty hook_text")

        topic.narration_script = _strip_duplicate_hook(topic.hook_text, topic.narration_script)
        topic.narration_script = f"{topic.hook_text} {topic.narration_script}".strip()

        return topic, len(topic.narration_script.split())

    def _call() -> Topic:
        history = load_topic_history()
        recent_categories = [h.get("category", "") for h in history[-5:] if h.get("category")]
        nonlocal prefetched_sources, preflight_source_errors
        seed, prefetched_sources, preflight_source_errors = choose_reachable_topic_seed(
            history,
            set(recent_categories[-2:]),
            excluded_source_signatures,
        )
        user_msg = build_topic_user_prompt(seed, prefetched_sources)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        raw_text = _groq_chat(messages, temperature=0.35)
        topic, word_count = _parse_topic(raw_text)
        set_canonical_topic_title(topic, seed)
        log.info("Using vetted topic-bank subject as the video title: %s", topic.title)
        topic.bank_id = str(seed["id"])
        topic.verified_fact = str(seed["verified_fact"])
        topic.source_urls = [str(x) for x in seed.get("source_urls", [])]
        topic.category = str(seed["category"])

        pre_outro_max = MAX_SCRIPT_WORDS - OUTRO_MAX_WORDS

        if word_count > pre_outro_max:
            log.warning(
                "narration_script too long (%d words > %d incl. outro headroom); trimming at a "
                "sentence boundary to stay within the Facebook Reels 90s cap",
                word_count, pre_outro_max,
            )
            topic.narration_script = _trim_script_to_word_limit(topic.narration_script, pre_outro_max)
            word_count = len(topic.narration_script.split())

        final_word_count = word_count + OUTRO_MIN_WORDS
        _validate_final_script_word_count(final_word_count)

        if len(topic.scene_keywords_en) < MIN_SCENE_CLIPS:
            raise PipelineError(
                f"Model returned fewer than {MIN_SCENE_CLIPS} scene keywords; visual/text alignment is required"
            )
        if topic.category and recent_categories and topic.category in recent_categories[-2:]:
            raise PipelineError(
                f"Generated category {topic.category!r} repeats one of the last 2 used categories "
                f"{recent_categories[-2:]!r}; forcing a retry with a different domain"
            )
        if topic_is_too_similar(topic, load_topic_history()):
            raise PipelineError("Generated topic is too similar to a previously published topic")
        return topic

    topic = with_retries(_call, what="OpenRouter topic generation", max_retries=TOPIC_GENERATION_MAX_ATTEMPTS)
    final_word_count = len(topic.narration_script.split()) + OUTRO_MIN_WORDS
    log.info(
        "Topic generated: %s (%d-word script, ~%.0fs at 1.75 words/sec)",
        topic.title, final_word_count, final_word_count / 1.75,
    )
    return topic, prefetched_sources, preflight_source_errors


def _proofread_topic_narration(topic: Topic, script: str) -> str:
    corrected = proofread_narration_tashkeel(
        script,
        canonical_subject=topic.title,
        verified_fact=topic.verified_fact,
    )
    return enforce_text_quality(
        corrected,
        reviser=lambda text, issues: proofread_narration_tashkeel(
            text,
            issues=issues,
            canonical_subject=topic.title,
            verified_fact=topic.verified_fact,
        ),
    )


def _build_evidence_only_fallback(topic: Topic) -> None:
    subject = topic.title.strip()
    fact = topic.verified_fact.strip()
    if not subject or not fact:
        raise PipelineError("Cannot build evidence-only fallback without title and verified_fact")
    topic.hook_text = f"مَا الحَقِيقَةُ المُوَثَّقَةُ عَنْ {subject}؟"
    topic.narration_script = f"{topic.hook_text} {fact}".strip()
    topic.caption = fact
    topic.hashtags = []
    log.warning(
        "Fact Check rejected the generated draft; replaced it with an evidence-only script "
        "from the vetted topic-bank fact and will re-check before publishing"
    )


# ---------------------------------------------------------------------------
# Step 2: Background footage (Pexels)
# ---------------------------------------------------------------------------

def search_pexels_video(keywords: str, exclude: set[str] | None = None) -> str:
    log.info("Searching Pexels for vertical footage: %r", keywords)
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
        data = resp.json()
        videos = data.get("videos", [])
        if not videos:
            raise PipelineError(f"No Pexels results for keywords: {keywords!r}")

        ordered_videos = videos
        for video in ordered_videos:
            files = [
                f for f in video.get("video_files", [])
                if f.get("width") and f.get("height") and f["height"] > f["width"]
            ]
            if not files:
                continue
            files.sort(key=lambda f: f["width"], reverse=True)
            best = files[0]
            if best["link"] in exclude:
                continue
            return best["link"]
        raise PipelineError(
            "No usable portrait-orientation video found in Pexels results "
            "(either none had a portrait file, or all were already used by another scene)"
        )

    return with_retries(_call, what="Pexels search")


def search_pexels_videos(keywords_list: list[str], topic_context: str = "") -> list[str]:
    urls: list[str] = []
    for keywords in keywords_list:
        combined = f"{topic_context} {keywords}".strip() if topic_context else keywords
        try:
            url = search_pexels_video(combined, exclude=set(urls))
        except PipelineError as exc:
            log.warning("Skipping scene %r because its topic-specific query failed: %s", keywords, exc)
            continue
        if url not in urls:
            urls.append(url)
    if len(urls) < MIN_SCENE_CLIPS:
        raise PipelineError(
            f"Only {len(urls)} distinct scene clips found; need at least {MIN_SCENE_CLIPS} "
            "to build a varied video without repeating unrelated footage"
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
        even = total_duration / n_scenes
        return [even] * n_scenes

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
        chosen = min(sentence_ends or nearby, key=lambda o: abs(o - ideal))
        cut_points.append(chosen)

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
            f"scale={VIDEO_W}:{VIDEO_H}:force_original_aspect_ratio=increase,crop={VIDEO_W}:{VIDEO_H},fps=30[v{i}]"
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
    log.info("Downloading %s -> %s", url, dest)

    def _call() -> Path:
        with requests.get(url, stream=True, timeout=120) as resp:
            if resp.status_code != 200:
                raise PipelineError(f"Download failed ({resp.status_code}) for {url}")
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
        return dest

    return with_retries(_call, what=f"download {url}")


def generate_ambient_music(dest: Path, duration: float = 90.0) -> Path | None:
    try:
        cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-f", "lavfi", "-i",
            "sine=frequency=110:sample_rate=44100:duration=" + str(duration),
            "-f", "lavfi", "-i",
            "sine=frequency=164.81:sample_rate=44100:duration=" + str(duration),
            "-filter_complex",
            "[0:a]volume=0.10,afade=t=in:st=0:d=4,afade=t=out:st=86:d=4[a];"
            "[1:a]volume=0.055,afade=t=in:st=0:d=4,afade=t=out:st=86:d=4[b];"
            "[a][b]amix=inputs=2:normalize=0,lowpass=f=900,volume=0.8[out]",
            "-map", "[out]", "-c:a", "libmp3lame", "-b:a", "96k", str(dest),
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        if dest.exists() and dest.stat().st_size > 0:
            log.info("Background music: generated offline ambient pad at %s", dest)
            return dest
    except Exception as exc:
        log.warning("Could not generate offline ambient music: %s", exc)
    return None


def get_bg_music(dest_dir: Path) -> Path | None:
    repo_root = Path(__file__).resolve().parent
    candidate_dirs = []
    seen = set()
    for name in ("music", "Music", "assets/music", "assets/Music", "audio", "bg_music"):
        d = (repo_root / name).resolve()
        if d not in seen:
            seen.add(d)
            candidate_dirs.append(d)

    audio_exts = (".mp3", ".m4a", ".wav", ".aac")
    all_found: list[Path] = []
    for d in candidate_dirs:
        if not d.is_dir():
            log.info("Background music: no folder at %s", d)
            continue
        found = sorted(p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in audio_exts)
        log.info("Background music: checked %s — %d audio file(s) found", d, len(found))
        all_found.extend(found)

    if all_found:
        chosen = random.choice(all_found)
        log.info("Background music: using %s", chosen)
        return chosen

    candidates = [u.strip() for u in (BG_MUSIC_URL or DEFAULT_BG_MUSIC_URL).split(",") if u.strip()]
    if candidates:
        url = random.choice(candidates)
        dest = dest_dir / "music.mp3"
        try:
            download_file(url, dest)
            log.info("Background music: %s", url)
            return dest
        except Exception as exc:
            log.warning("Could not fetch background music — trying offline fallback: %s", exc)

    generated = generate_ambient_music(dest_dir / "ambient_pad.mp3")
    if generated:
        return generated

    log.info(
        "No background music available after local, GitHub, and offline fallback — "
        "checked %s (recursively) for %s files.",
        ", ".join(str(d) for d in candidate_dirs), "/".join(audio_exts),
    )
    return None


# ---------------------------------------------------------------------------
# Step 3: Voiceover (edge-tts) + subtitles
# ---------------------------------------------------------------------------

def _norm_arabic_words(text: str) -> list[str]:
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text or "")
    text = re.sub(r"[^\w\u0600-\u06FF]+", " ", text, flags=re.UNICODE)
    return [w for w in text.lower().split() if w]


def detect_silma_reference_leak(audio_path: Path) -> str | None:
    if not SILMA_GUARD_ENABLED or not SILMA_REFERENCE_TEXT:
        return None
    from faster_whisper import WhisperModel
    model = WhisperModel(os.getenv("WHISPER_MODEL", "base"), device="cpu", compute_type="int8")
    segments, _ = model.transcribe(str(audio_path), language="ar", word_timestamps=False, vad_filter=False)
    heard = _norm_arabic_words(" ".join(seg.text or "" for seg in segments))
    ref = _norm_arabic_words(SILMA_REFERENCE_TEXT)
    minimum = max(2, min(SILMA_GUARD_MIN_MATCH_WORDS, len(ref)))
    for size in range(len(ref), minimum - 1, -1):
        phrase = ref[-size:]
        for i in range(len(heard) - size + 1):
            if heard[i:i + size] == phrase:
                return " ".join(phrase)
    return None


def generate_edge_tts(text: str, out_path: Path, voice: str = TTS_VOICE) -> Path:
    log.info("Generating Edge TTS fallback narration with voice %s", voice)
    import edge_tts
    async def _run():
        communicate = edge_tts.Communicate(text, voice, rate=EDGE_TTS_RATE, pitch=EDGE_TTS_PITCH)
        timings = []
        with open(out_path, "wb") as audio_file:
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_file.write(chunk["data"])
                elif chunk["type"] == "WordBoundary":
                    timings.append({"text": chunk.get("text", ""),
                                    "offset": chunk.get("offset", 0) / 10_000_000,
                                    "duration": chunk.get("duration", 0) / 10_000_000})
        out_path.with_suffix(".timings.json").write_text(json.dumps(timings, ensure_ascii=False), encoding="utf-8")
    def _call() -> Path:
        asyncio.run(_run())
        if not out_path.exists() or out_path.stat().st_size == 0:
            raise PipelineError("Edge TTS fallback produced an empty audio file")
        return out_path
    return with_retries(_call, what="Edge TTS fallback generation")


def generate_tts(
    text: str,
    out_path: Path,
    voice: str = TTS_VOICE,
    reference_profile: str | None = None,
) -> Path:
    if TTS_ENGINE == "silma":
        try:
            selected_profile = reference_profile or SILMA_REFERENCE_PROFILE
            if selected_profile:
                reference, ref_text = resolve_reference_profile(
                    selected_profile, SILMA_VOICE_PROFILES_FILE
                )
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
            log.exception("SILMA failed or every candidate was rejected; switching to Edge TTS")
            out_path.unlink(missing_ok=True)
            return generate_edge_tts(text, out_path, voice)
    return generate_edge_tts(text, out_path, voice)


def get_media_duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(path),
        ],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


def _audio_duration_exceeds_limit(duration_seconds: float) -> bool:
    return duration_seconds < MIN_AUDIO_SECONDS or duration_seconds > MAX_AUDIO_SECONDS


def _reel_duration_needs_padding(duration_seconds: float) -> bool:
    return duration_seconds < MIN_AUDIO_SECONDS


def _audio_duration_exceeds_maximum(duration_seconds: float) -> bool:
    return duration_seconds > MAX_AUDIO_SECONDS


def _audio_duration_needs_normalization(duration_seconds: float) -> bool:
    return duration_seconds > TARGET_AUDIO_SECONDS


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
        raise PipelineError(f"Could not normalize narration duration: {result.stderr[-1000:]}")
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
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            log.warning("Could not rescale subtitle timings: %s", exc)
    final_duration = get_media_duration(audio_path)
    log.info("Narration normalized from %.1fs to %.1fs", original, final_duration)
    return final_duration


WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL", "base")
LOW_CONFIDENCE_THRESHOLD = float(os.getenv("PRONUNCIATION_CONFIDENCE_THRESHOLD", "0.4"))


def align_words_with_whisper(
    audio_path: Path, script_words: list[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from faster_whisper import WhisperModel

    log.info("Force-aligning subtitles to the rendered audio with Whisper (%s)...", WHISPER_MODEL_SIZE)
    model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(
        str(audio_path), language="ar", word_timestamps=True, vad_filter=False,
    )

    whisper_words: list[tuple[str, float, float, float]] = []
    for segment in segments:
        for w in (segment.words or []):
            text = (w.word or "").strip()
            if text:
                whisper_words.append((text, float(w.start), float(w.end), float(getattr(w, "probability", 1.0))))
    if not whisper_words:
        raise PipelineError("Whisper produced no word-level timestamps")

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
            _, start, end, probability = whisper_words[j1 + k]
            timings[i1 + k] = {
                "text": script_words[i1 + k],
                "offset": start,
                "duration": max(end - start, 0.05),
                "probability": probability,
            }

    known_indices = [i for i, t in enumerate(timings) if t is not None]
    if not known_indices:
        raise PipelineError("Could not align any script words against the Whisper transcript")
    if len(known_indices) < len(script_words) * 0.5:
        raise PipelineError(
            f"Only {len(known_indices)}/{len(script_words)} script words matched the Whisper "
            "transcript; alignment is too unreliable to trust for this run"
        )

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

    log.info(
        "Whisper alignment: %d/%d script words matched directly, %d interpolated",
        len(known_indices), len(script_words), len(script_words) - len(known_indices),
    )

    known_set = set(known_indices)
    flagged_words: list[dict[str, Any]] = []
    for i in range(len(script_words)):
        if i not in known_set:
            flagged_words.append({
                "index": i, "text": script_words[i],
                "approx_seconds": round(timings[i]["offset"], 2),
                "reason": "not_recognized",
            })
            continue
        prob = timings[i].get("probability")
        if prob is not None and prob < LOW_CONFIDENCE_THRESHOLD:
            flagged_words.append({
                "index": i, "text": script_words[i],
                "approx_seconds": round(timings[i]["offset"], 2),
                "reason": "low_confidence", "confidence": round(prob, 2),
            })
    return timings, flagged_words


def realign_subtitles_with_whisper(narration_path: Path, script_text: str) -> tuple[bool, list[dict[str, Any]]]:
    timings_path = narration_path.with_suffix(".timings.json")
    try:
        aligned, flagged_words = align_words_with_whisper(narration_path, script_text.split())
        timings_path.write_text(json.dumps(aligned, ensure_ascii=False), encoding="utf-8")
        log.info("Whisper-aligned timings written to %s", timings_path)
        if flagged_words:
            log.warning(
                "PRONUNCIATION REVIEW — %d word(s) flagged for review (possible mispronunciation, "
                "worth listening to first): %s",
                len(flagged_words),
                ", ".join(
                    f"{w['text']}@{w['approx_seconds']}s[{w['reason']}]" for w in flagged_words
                ),
            )
        return True, flagged_words
    except Exception:
        log.error(
            "WHISPER ALIGNMENT FAILED — falling back to edge-tts's own word timings, which "
            "are known to run ahead of the actual audio. Captions in this video will likely "
            "be out of sync. Full error below:",
            exc_info=True,
        )
        return False, []


def _ass_escape(text: str) -> str:
    return text.replace("{", "").replace("}", "").replace("\n", " ").replace("\r", " ")


DISPLAY_PUNCTUATION = str.maketrans(
    "".join([
        ".", ",", "،", "؛", ":", "!", "?", "؟", "…", "-", "—", "_",
        "(", ")", "[", "]", "{", "}", '"', "«", "»", "/", "\\",
    ]),
    " " * 23,
)


def _clean_display_words(words: list[str]) -> list[str]:
    cleaned = [
        _TASHKEEL_RE.sub("", w.translate(DISPLAY_PUNCTUATION)).strip()
        for w in words
    ]
    return [w for w in cleaned if w]


def build_subtitles(text: str, duration: float, out_path: Path, timings_path: Path | None = None) -> Path:
    timing_data = []
    if timings_path and timings_path.exists():
        try:
            timing_data = json.loads(timings_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("Could not read TTS word timings; using fallback timing: %s", exc)

    timed_words = [x for x in timing_data if x.get("text")]

    if timed_words:
        words = [w["text"] for w in timed_words]
        offsets = [max(float(w.get("offset", 0)), 0.0) for w in timed_words]
        durations = [max(float(w.get("duration", 0)), 0.0) for w in timed_words]
        chunks = []
        for i in range(0, len(words), 8):
            end_i = min(i + 8, len(words))
            start = offsets[i]
            if end_i < len(words):
                end = offsets[end_i]
            else:
                end = max(offsets[end_i - 1] + durations[end_i - 1], duration)
            chunks.append((words[i:end_i], max(start, 0), min(max(end, start + 0.25), duration)))
    else:
        log.warning("No TTS word timings available; using proportional fallback (captions may drift)")
        words = text.split()
        word_chunks = [words[i:i + 8] for i in range(0, len(words), 8)] or [words]

        def _chunk_weight(chunk: list[str]) -> float:
            weight = sum(len(w) for w in chunk) + len(chunk)
            if chunk and chunk[-1][-1:] in "؟?.!":
                weight += 6
            return max(weight, 1.0)

        weights = [_chunk_weight(c) for c in word_chunks]
        total_weight = sum(weights) or 1.0
        chunks = []
        cursor = 0.0
        for chunk, w in zip(word_chunks, weights):
            start = cursor
            cursor += duration * (w / total_weight)
            chunks.append((chunk, start, cursor))

    def fmt(t: float) -> str:
        t = max(t, 0.0)
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = int(t % 60)
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
            line2 = rtl + _ass_escape(" ".join(display_words[split_at:]))
            chunk_text += "\\N" + line2
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
# Step 4: Video assembly (ffmpeg)
# ---------------------------------------------------------------------------

def assemble_video(
    bg_video: Path, narration: Path, subtitle_path: Path, out_path: Path, music_path: Path | None = None,
) -> Path:
    log.info("Assembling final video...")
    audio_duration = get_media_duration(narration)
    reel_duration = max(audio_duration, MIN_AUDIO_SECONDS)

    subtitle_filter_path = str(subtitle_path).replace("\\", "/").replace(":", "\\:")

    vf = (
        f"scale={VIDEO_W}:{VIDEO_H}:force_original_aspect_ratio=increase,"
        f"crop={VIDEO_W}:{VIDEO_H},"
        f"subtitles='{subtitle_filter_path}'"
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
        cmd = [
            "ffmpeg", "-y",
            "-stream_loop", "-1", "-i", str(bg_video),
            *audio_inputs,
            "-t", f"{reel_duration:.2f}",
            "-vf", vf,
            "-r", str(OUTPUT_FPS),
            "-filter_complex", filter_complex,
            "-map", "0:v:0", "-map", "[aout]",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-c:a", "aac", "-b:a", "192k",
            str(out_path),
        ]
    else:
        cmd = [
            "ffmpeg", "-y",
            "-stream_loop", "-1", "-i", str(bg_video),
            *audio_inputs,
            "-t", f"{reel_duration:.2f}",
            "-vf", vf,
            "-r", str(OUTPUT_FPS),
            "-filter_complex", f"[1:a]apad=whole_dur={reel_duration:.2f}[aout]",
            "-map", "0:v:0", "-map", "[aout]",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-c:a", "aac", "-b:a", "192k",
            str(out_path),
        ]

    def _call() -> Path:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise PipelineError(f"ffmpeg failed: {result.stderr[-2000:]}")
        if not out_path.exists() or out_path.stat().st_size == 0:
            raise PipelineError("ffmpeg produced an empty output file")
        return out_path

    return with_retries(_call, what="video assembly")


# ---------------------------------------------------------------------------
# Step 5: Host the video publicly (GitHub Release asset)
# ---------------------------------------------------------------------------

def _ensure_repo_is_public(api_base: str, headers: dict[str, str]) -> None:
    resp = requests.get(api_base, headers=headers, timeout=30)
    if resp.status_code == 200 and resp.json().get("private"):
        raise PipelineError(
            "This repository is Private. GitHub Release assets in a private repo can't be "
            "downloaded without authentication, so Buffer's bot can't fetch the video from its "
            "public URL. Fix: make the repository Public."
        )


def host_video_on_github(video_path: Path, run_id: str) -> str:
    if not GH_RELEASE_TOKEN or not GITHUB_REPOSITORY:
        raise PipelineError(
            "GH_RELEASE_TOKEN / GITHUB_REPOSITORY not set — both are required to host "
            "the video publicly for Buffer."
        )

    api_base = f"{GITHUB_API_BASE}/repos/{GITHUB_REPOSITORY}"
    tag = f"auto-publish-{run_id}"
    headers = {
        "Authorization": f"Bearer {GH_RELEASE_TOKEN}",
        "Accept": "application/vnd.github+json",
    }
    _ensure_repo_is_public(api_base, headers)

    def _create_release() -> dict[str, Any]:
        resp = requests.post(
            f"{api_base}/releases",
            headers=headers,
            json={
                "tag_name": tag,
                "name": f"Auto-publish video {run_id}",
                "body": "Temporary release created only to give Buffer a public URL for this video. Safe to delete.",
                "draft": False,
                "prerelease": False,
            },
            timeout=30,
        )
        if resp.status_code not in (200, 201):
            raise PipelineError(f"GitHub release creation failed ({resp.status_code}): {resp.text[:500]}")
        return resp.json()

    release = with_retries(_create_release, what="create GitHub release for video hosting")
    upload_url = release["upload_url"].split("{")[0]

    def _upload_asset() -> dict[str, Any]:
        with open(video_path, "rb") as f:
            resp = requests.post(
                upload_url,
                headers={**headers, "Content-Type": "video/mp4"},
                params={"name": video_path.name},
                data=f,
                timeout=300,
            )
        if resp.status_code not in (200, 201):
            raise PipelineError(f"GitHub release asset upload failed ({resp.status_code}): {resp.text[:500]}")
        return resp.json()

    asset = with_retries(_upload_asset, what="upload video asset to GitHub release")
    url = asset["browser_download_url"]
    log.info("Video hosted publicly at: %s", url)
    return url


# ---------------------------------------------------------------------------
# Step 6: Publish (Buffer)
# ---------------------------------------------------------------------------

_BUFFER_CREATE_POST_MUTATION = """
mutation CreatePost($channelId: ChannelId!, $text: String!, $videoUrl: String!, $metadata: PostInputMetaData) {
  createPost(
    input: {
      text: $text
      channelId: $channelId
      schedulingType: automatic
      mode: shareNow
      assets: [{ video: { url: $videoUrl } }]
      metadata: $metadata
    }
  ) {
    ... on PostActionSuccess {
      post { id text dueAt }
    }
    ... on MutationError {
      message
    }
  }
}
"""

_GET_CHANNEL_QUERY = """
query GetChannel($id: ChannelId!) {
  channel(input: { id: $id }) {
    id
    service
  }
}
"""

YOUTUBE_CATEGORY_ID = os.getenv("YOUTUBE_CATEGORY_ID", "24")


def get_channel_service(channel_id: str) -> str:
    def _call() -> str:
        resp = requests.post(
            BUFFER_ENDPOINT,
            headers={
                "Authorization": f"Bearer {BUFFER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={"query": _GET_CHANNEL_QUERY, "variables": {"id": channel_id}},
            timeout=30,
        )
        if resp.status_code != 200:
            raise PipelineError(
                f"Buffer API error {resp.status_code} while looking up channel {channel_id}: {resp.text[:500]}"
            )
        data = resp.json()
        if data.get("errors"):
            raise PipelineError(f"Buffer API error looking up channel {channel_id}: {data['errors']}")
        service = ((data.get("data") or {}).get("channel") or {}).get("service")
        if not service:
            raise PipelineError(f"Could not resolve 'service' for Buffer channel {channel_id}: {data}")
        return service

    return with_retries(_call, what=f"look up Buffer channel {channel_id}")


def build_channel_metadata(service: str, topic: Topic) -> dict[str, Any] | None:
    service = (service or "").lower()
    if service == "youtube":
        return {"youtube": {"title": topic.title[:100], "categoryId": YOUTUBE_CATEGORY_ID}}
    if service == "facebook":
        return {"facebook": {"type": "reel"}}
    return None


def publish_video(video_path: Path, topic: Topic, channel_ids: list[str]) -> dict[str, Any]:
    log.info("Publishing to Buffer channels: %s", channel_ids)
    hashtags_str = " ".join(topic.hashtags)
    text = f"{topic.title}\n\n{topic.caption}\n\n{hashtags_str}".strip()

    run_id = time.strftime("%Y%m%d-%H%M%S")
    video_url = host_video_on_github(video_path, run_id)

    results: dict[str, Any] = {}
    failed: list[str] = []

    for channel_id in channel_ids:
        service = get_channel_service(channel_id)
        metadata = build_channel_metadata(service, topic)
        log.info("Channel %s resolved to service=%s metadata=%s", channel_id, service, metadata)

        def _call(channel_id: str = channel_id, metadata: dict[str, Any] | None = metadata) -> dict[str, Any]:
            resp = requests.post(
                BUFFER_ENDPOINT,
                headers={
                    "Authorization": f"Bearer {BUFFER_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "query": _BUFFER_CREATE_POST_MUTATION,
                    "variables": {
                        "channelId": channel_id,
                        "text": text,
                        "videoUrl": video_url,
                        "metadata": metadata,
                    },
                },
                timeout=60,
            )
            if resp.status_code != 200:
                raise PipelineError(f"Buffer API error {resp.status_code}: {resp.text[:500]}")
            return resp.json()

        response = with_retries(_call, what=f"publish to Buffer channel {channel_id}")
        log.info("Buffer response for channel %s: %s", channel_id, json.dumps(response, ensure_ascii=False)[:800])

        if response.get("errors"):
            log.error("✘ channel %s FAILED: %s", channel_id, response["errors"])
            failed.append(channel_id)
            results[channel_id] = {"success": False, "errors": response["errors"]}
            continue

        payload = (response.get("data") or {}).get("createPost") or {}
        if "message" in payload and "post" not in payload:
            log.error("✘ channel %s FAILED: %s", channel_id, payload["message"])
            failed.append(channel_id)
            results[channel_id] = {"success": False, "error": payload["message"]}
        else:
            post = payload.get("post", {})
            log.info("✔ channel %s published (post id: %s)", channel_id, post.get("id"))
            results[channel_id] = {"success": True, "post": post}

    if failed:
        raise PipelineError(f"Publishing failed for Buffer channel(s): {', '.join(failed)}")
    return results


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_pipeline() -> None:
    check_env()
    if DRY_RUN:
        if not DRY_RUN_INPUT_FILE.exists():
            raise PipelineError(f"Dry-run input file is missing: {DRY_RUN_INPUT_FILE}")
        raw_text = DRY_RUN_INPUT_FILE.read_text(encoding="utf-8").strip()
        if not raw_text:
            raise PipelineError("Dry-run input text is empty")
        corrected = proofread_narration_tashkeel(raw_text)
        corrected = enforce_text_quality(
            corrected,
            reviser=lambda text, issues: proofread_narration_tashkeel(text, issues=issues),
        )
        red_flag = find_content_red_flag(corrected)
        result = {
            "dry_run": True,
            "input_file": str(DRY_RUN_INPUT_FILE),
            "input_word_count": len(_normalize_for_compare(raw_text).split()),
            "corrected_word_count": len(_normalize_for_compare(corrected).split()),
            "red_flag": red_flag,
            "passed": red_flag is None,
            "corrected_text": corrected,
        }
        out = WORK_DIR / "dry_run_result.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print("===== DRY RUN RESULT =====")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if red_flag:
            raise PipelineError(f"Dry-run rejected corrected text because of: {red_flag}")
        print("✅ Dry-run passed: no TTS, footage, or publishing was executed.")
        return
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = WORK_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log.info("Run directory: %s", run_dir)

    topic, prefetched_sources, preflight_source_errors = generate_topic()

    topic.narration_script = _proofread_topic_narration(topic, topic.narration_script)
    topic.narration_script = _append_engagement_outro(topic.narration_script)
    final_word_count = len(topic.narration_script.split())
    _validate_final_script_word_count(final_word_count)
    red_flag = find_content_red_flag(topic.narration_script)
    if red_flag:
        raise PipelineError(f"Rejected hallucinated or nonstandard content term: {red_flag}")

    fact_report = fact_check_topic(
        topic,
        run_dir / "fact_check.json",
        prefetched_sources=prefetched_sources,
        preflight_source_errors=preflight_source_errors,
    )
    if fact_report.get("status") != "PASS":
        _build_evidence_only_fallback(topic)
        topic.narration_script = _append_engagement_outro(topic.narration_script)
        fact_report = fact_check_topic(
            topic,
            run_dir / "fact_check.json",
            prefetched_sources=prefetched_sources,
            preflight_source_errors=preflight_source_errors,
        )
        if fact_report.get("status") != "PASS":
            raise PipelineError(
                "Fact Check rejected both the generated script and the evidence-only fallback; "
                "audio and publishing are blocked. "
                f"Details: {fact_report.get('errors', [])}"
            )
    log.info("Fact Check passed: %d supported claim(s)", len(fact_report.get("claims", [])))

    (run_dir / "topic.json").write_text(
        json.dumps(topic.__dict__, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    voice_profile = choose_voice_profile()
    topic.voice_profile = voice_profile
    (run_dir / "topic.json").write_text(
        json.dumps(topic.__dict__, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log.info("Selected rotating voice profile for this video: %s", voice_profile)

    max_duration_attempts = 3
    narration_path = None
    audio_duration = 0.0
    for attempt in range(1, max_duration_attempts + 1):
        narration_path = generate_tts(
            topic.narration_script,
            run_dir / "narration.mp3",
            reference_profile=voice_profile,
        )
        audio_duration = get_media_duration(narration_path)
        log.info("Narration audio duration: %.1fs (attempt %d/%d)", audio_duration, attempt, max_duration_attempts)
        if _audio_duration_needs_normalization(audio_duration):
            log.warning(
                "Narration exceeds the %.1fs safe target at %.1fs; correcting speed to %.1fs",
                TARGET_AUDIO_SECONDS, audio_duration, TARGET_AUDIO_SECONDS,
            )
            audio_duration = normalize_narration_duration(narration_path, TARGET_AUDIO_SECONDS)
        if not _audio_duration_exceeds_maximum(audio_duration):
            break
        log.warning(
            "Narration remains over %.1fs after normalization (%.1fs); retrying TTS only (attempt %d/%d)",
            MAX_AUDIO_SECONDS, audio_duration, attempt, max_duration_attempts,
        )
        if attempt == max_duration_attempts:
            raise PipelineError(
                f"Narration audio still exceeds the {MAX_AUDIO_SECONDS:.0f}s maximum after "
                f"{max_duration_attempts} attempts — aborting rather than publish an overlong video"
            )

    whisper_aligned, pronunciation_flags = realign_subtitles_with_whisper(narration_path, topic.narration_script)
    (run_dir / "whisper_alignment_status.json").write_text(
        json.dumps({"whisper_aligned": whisper_aligned}, ensure_ascii=False), encoding="utf-8"
    )
    (run_dir / "pronunciation_review.json").write_text(
        json.dumps(pronunciation_flags, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    remember_topic(topic)
    scene_urls = search_pexels_videos(topic.scene_keywords_en, topic.search_keywords_en)
    scene_paths = [download_file(url, run_dir / f"scene_{i:02d}.mp4") for i, url in enumerate(scene_urls)]
    word_timings = _load_word_timings(narration_path.with_suffix(".timings.json"))
    reel_duration = max(audio_duration, MIN_AUDIO_SECONDS)
    if _reel_duration_needs_padding(audio_duration):
        log.info(
            "Narration is %.1fs; extending the final reel with background audio/video to %.1fs "
            "without adding unverified narration",
            audio_duration, reel_duration,
        )
    segment_durations = compute_scene_durations(word_timings, reel_duration, len(scene_paths))
    bg_video_path = build_multishot_background(scene_paths, segment_durations, run_dir / "background.mp4")
    music_path = get_bg_music(run_dir)

    subtitle_path = build_subtitles(
        topic.narration_script,
        audio_duration,
        run_dir / "subtitles.ass",
        timings_path=narration_path.with_suffix(".timings.json"),
    )

    final_video_path = assemble_video(
        bg_video_path, narration_path, subtitle_path, run_dir / "final.mp4", music_path=music_path,
    )
    final_video_duration = get_media_duration(final_video_path)
    if final_video_duration < MIN_AUDIO_SECONDS - 0.25 or final_video_duration > MAX_AUDIO_SECONDS:
        raise PipelineError(
            f"Assembled video is {final_video_duration:.2f}s; required window is "
            f"{MIN_AUDIO_SECONDS:.0f}..{MAX_AUDIO_SECONDS:.0f}s. Publishing is blocked."
        )
    log.info(
        "Assembled video duration: %.2fs (required window %.0f..%.0fs)",
        final_video_duration, MIN_AUDIO_SECONDS, MAX_AUDIO_SECONDS,
    )

    result = publish_video(final_video_path, topic, BUFFER_CHANNEL_IDS)
    (run_dir / "publish_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log.info("Published successfully: %s", json.dumps(result, ensure_ascii=False)[:500])


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
