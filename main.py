#!/usr/bin/env python3
"""
Automated Content Creation & Publishing Pipeline
==================================================
Generates a short-form vertical video (Arabic voiceover + burned-in subtitles +
Pexels stock footage) from an AI-generated viral topic, then publishes it to
YouTube / TikTok / Facebook via the Buffer API (buffer.com) — using Buffer's
free plan, which covers exactly 3 connected channels.

Since Buffer's API doesn't accept a direct file upload (it needs a public
HTTPS URL for the video), this pipeline temporarily hosts the finished video
as a GitHub Release asset in this same repository, hands that URL to Buffer,
then Buffer fetches and publishes it.

Designed to run unattended inside GitHub Actions (see
.github/workflows/auto_publish.yml), but works fine locally too:

    python main.py

Required environment variables (set as GitHub Secrets in CI):
    GROQ_API_KEY                  free key from console.groq.com/keys
    PEXELS_API_KEY
    BUFFER_API_KEY                from publish.buffer.com/settings/api
    BUFFER_CHANNEL_IDS            comma-separated Buffer channel IDs, one per
                                  connected platform (YouTube, TikTok, Facebook)
    GH_RELEASE_TOKEN              a token with 'contents: write' on this repo,
                                  used to upload the video as a Release asset.
                                  In GitHub Actions this can just be the
                                  built-in secrets.GITHUB_TOKEN — see the
                                  workflow file.

Optional:
    GROQ_MODEL            Groq model id (default: openai/gpt-oss-120b)

Optional environment variables:
    TTS_VOICE            edge-tts voice (default: ar-EG-SalmaNeural)
    WORK_DIR             scratch directory (default: ./work)
    LOG_LEVEL            default: INFO
    BG_MUSIC_URL         direct MP3/audio URL(s), comma-separated, for background
                          music. Falls back to bundled CC-BY tracks if unset.
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
MAX_AUDIO_SECONDS = float(os.getenv("MAX_AUDIO_SECONDS", "90"))
TARGET_AUDIO_SECONDS = float(os.getenv("TARGET_AUDIO_SECONDS", str(max(1.0, MAX_AUDIO_SECONDS - 1.0))))
MAX_SCRIPT_WORDS = int(os.getenv("MAX_SCRIPT_WORDS", "165"))
# Topic generation gets a larger retry budget than generic calls because
# rate limits and malformed model responses can otherwise exhaust the run.
TOPIC_GENERATION_MAX_ATTEMPTS = int(os.getenv("TOPIC_GENERATION_MAX_ATTEMPTS", "5"))
HISTORY_LIMIT = int(os.getenv("HISTORY_LIMIT", "1000"))
MIN_SCENE_CLIPS = int(os.getenv("MIN_SCENE_CLIPS", "10"))

def _clean_env(name: str) -> str | None:
    """Read an env var and strip ALL whitespace/newline characters from it.

    GitHub Secrets very often pick up an invisible trailing newline or space
    (copy-pasted from a browser/terminal, or set via `echo "key" | gh secret
    set ...` instead of `printf`). Python's http.client refuses to send any
    header whose value contains raw \\r/\\n or leading/trailing whitespace,
    which is exactly the 'Invalid leading whitespace, reserved character(s),
    or return character(s) in header value' error. None of these secrets
    (API keys, username) should legitimately contain whitespace, so it's
    always safe to strip it all out here.
    """
    raw = os.getenv(name)
    if raw is None:
        return None
    cleaned = re.sub(r"\s+", "", raw)
    return cleaned or None


PEXELS_API_KEY = _clean_env("PEXELS_API_KEY")

# --- Publishing (Buffer) ---------------------------------------------------
# Buffer's free plan covers exactly 3 connected channels, which is all this
# pipeline needs (YouTube + TikTok + Facebook). Unlike upload-post.com,
# Buffer's API takes a public HTTPS URL for the video rather than a raw file
# upload, so we host the finished video as a GitHub Release asset first (see
# host_video_on_github below) and hand Buffer that URL.
BUFFER_API_KEY = _clean_env("BUFFER_API_KEY")
BUFFER_CHANNEL_IDS = [
    c.strip() for c in os.getenv("BUFFER_CHANNEL_IDS", "").split(",") if c.strip()
]

# A token with 'contents: write' on this repo, used only to create a Release
# and upload the video as an asset (so Buffer has a public URL to fetch it
# from). In GitHub Actions this is normally just the built-in
# secrets.GITHUB_TOKEN, passed through as GH_RELEASE_TOKEN in the workflow.
GH_RELEASE_TOKEN = _clean_env("GH_RELEASE_TOKEN") or _clean_env("GITHUB_TOKEN")
# GITHUB_REPOSITORY (e.g. "your-user/your-repo") is set automatically by
# GitHub Actions on every run — no need to define it yourself in the workflow.
GITHUB_REPOSITORY = os.getenv("GITHUB_REPOSITORY")

# --- Topic-generation model (Groq) --------------------------------------
# Groq's free tier needs just an API key (console.groq.com/keys) — no
# service account, no OAuth, no "AQ. key" style breakage like AI Studio's
# Gemini keys. It's also faster and, on open models like Llama 3.3 70B /
# GPT-OSS 120B, comparably or more capable than Gemini Flash for this task.
GROQ_API_KEY = _clean_env("GROQ_API_KEY")

TTS_ENGINE = os.getenv("TTS_ENGINE", "silma").strip().lower()
TTS_VOICE = os.getenv("TTS_VOICE", "ar-EG-SalmaNeural")
EDGE_TTS_RATE = os.getenv("EDGE_TTS_RATE", "-8%")
EDGE_TTS_PITCH = os.getenv("EDGE_TTS_PITCH", "-5Hz")
SILMA_REFERENCE_WAV = Path(os.getenv("SILMA_REFERENCE_WAV", "assets/voice_reference_synthetic.wav"))
SILMA_REFERENCE_TEXT = os.getenv("SILMA_REFERENCE_TEXT", "").strip()  # legacy fallback only
SILMA_REFERENCE_PROFILE = os.getenv("SILMA_REFERENCE_PROFILE", "").strip()
SILMA_VOICE_PROFILES_FILE = Path(os.getenv("SILMA_VOICE_PROFILES_FILE", "assets/voices/voice_profiles.json"))
SILMA_SEED = int(os.getenv("SILMA_SEED", "42"))
SILMA_MAX_ATTEMPTS = int(os.getenv("SILMA_MAX_ATTEMPTS", "2"))
SILMA_MIN_SCORE = float(os.getenv("SILMA_MIN_SCORE", "0.6"))
SILMA_SPEED = float(os.getenv("SILMA_SPEED", "1.0"))
SILMA_GUARD_ENABLED = os.getenv("SILMA_GUARD_ENABLED", "true").lower() == "true"
SILMA_GUARD_MIN_MATCH_WORDS = int(os.getenv("SILMA_GUARD_MIN_MATCH_WORDS", "2"))

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_MAX_COMPLETION_TOKENS = int(os.getenv("GROQ_MAX_COMPLETION_TOKENS", "2200"))
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
# A Groq 429 used to be turned into a PipelineError immediately, which meant
# it silently ate one of generate_topic()'s own limited content-generation
# attempts (see with_retries) even though it has nothing to do with content
# quality. _groq_chat now absorbs 429s itself, up to this many tries, so a
# transient rate limit no longer costs a real "the script came out too
# short" retry.
GROQ_RATE_LIMIT_MAX_RETRIES = int(os.getenv("GROQ_RATE_LIMIT_MAX_RETRIES", "4"))
PEXELS_SEARCH_ENDPOINT = "https://api.pexels.com/videos/search"
# How many of Pexels's own top (most-relevant) results to randomize among.
# search_pexels_video() used to shuffle across the FULL up-to-15-result page
# before picking one, which just as often picked a result Pexels itself
# ranked 14th (loosely related at best) as the top match — this is why
# scenes frequently had nothing to do with the script's topic. Keeping this
# small preserves Pexels's relevance ranking while still giving some
# variety across runs that reuse the same search phrase.
TOP_RELEVANT_CANDIDATES = int(os.getenv("TOP_RELEVANT_CANDIDATES", "8"))
BUFFER_ENDPOINT = "https://api.buffer.com"
GITHUB_API_BASE = "https://api.github.com"

VIDEO_W, VIDEO_H = 1080, 1920  # 9:16 vertical
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 5

# --- Background music -----------------------------------------------------
# incompetech.com's old direct-download URLs (used previously) have gone
# dead — the site now serves its player via JavaScript, so those links
# 404'd on every run. Local files are the reliable fix: no network fetch at
# all, so there's nothing to go stale or get blocked later.
#
# Drop your own royalty-free .mp3 files into a "music/" folder at the repo
# root (next to main.py — actions/checkout brings it along automatically).
# One is picked at random each run, so adding a handful of tracks gives you
# variety for free. Good, verified no-attribution-required sources: Mixkit
# (mixkit.co/free-stock-music) or Pixabay (pixabay.com/music) — download the
# file in your browser, commit it into music/.
#
# BG_MUSIC_URL (a GitHub Secret or plain env var, comma-separated for
# several options) is still supported as a fallback if you'd rather host
# your track elsewhere. If neither is available, the pipeline simply
# publishes without music instead of failing the run.
BG_MUSIC_URL = _clean_env("BG_MUSIC_URL")
MUSIC_VOLUME = float(os.getenv("MUSIC_VOLUME", "0.07"))
MUSIC_DIR = Path(__file__).resolve().parent / "music"
# Verified public repository track used only when no local track or user URL is
# configured. If GitHub is unreachable, generate_ambient_music() is used.
DEFAULT_BG_MUSIC_URL = (
    "https://raw.githubusercontent.com/effacestudios/Royalty-Free-Music-Pack/master/Bubbles.mp3"
)

REQUIRED_ENV = {
    "GROQ_API_KEY": GROQ_API_KEY,
    "PEXELS_API_KEY": PEXELS_API_KEY,
    "BUFFER_API_KEY": BUFFER_API_KEY,
    "GH_RELEASE_TOKEN": GH_RELEASE_TOKEN,
    "GITHUB_REPOSITORY": GITHUB_REPOSITORY,
}


class PipelineError(Exception):
    """Raised for any unrecoverable pipeline failure."""


class SourcePreflightError(PipelineError):
    """Raised when no unused topic has any accessible evidence source."""


# The exact 11 categories listed in SYSTEM_PROMPT's opening paragraph, kept
# here too so remember_topic()/_call() can track which ones were used
# recently and stop the model from repeating one (see recent_categories in
# generate_topic()). If SYSTEM_PROMPT's category list is ever edited, update
# this list to match.
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
    hook_text: str          # one strange/curious Arabic question (the opening hook — no answer)
    narration_script: str    # full multi-paragraph script that gets spoken + subtitled
    title: str               # Video title
    caption: str              # Caption for the post
    hashtags: list[str] = field(default_factory=list)
    search_keywords_en: str = ""  # English keywords for Pexels search
    scene_keywords_en: list[str] = field(default_factory=list)
    category: str = ""       # one of TOPIC_CATEGORIES — used to force domain rotation
    bank_id: str = ""        # selected entry from config/topic_bank.json
    verified_fact: str = ""  # source-backed fact supplied to the generator
    source_urls: list[str] = field(default_factory=list)


def set_canonical_topic_title(topic: Topic, seed: dict[str, Any]) -> None:
    """Use the vetted topic-bank subject as the published title, never a model invention."""
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
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            log.warning("Attempt %d/%d for %s failed: %s", attempt, attempts, what, exc)
            if attempt < attempts:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)
    raise PipelineError(f"{what} failed after {attempts} attempts: {last_err}") from last_err


def _trim_script_to_word_limit(script: str, max_words: int) -> str:
    """Hard safety net: if the model overshoots MAX_SCRIPT_WORDS, cut the
    script down at the nearest sentence boundary at or before the limit
    instead of publishing an over-length (and therefore Buffer-rejected)
    video."""
    words = script.split()
    if len(words) <= max_words:
        return script
    truncated = " ".join(words[:max_words])
    # Prefer cutting at the end of a full sentence if one exists reasonably
    # close to the limit, so the narration doesn't stop mid-thought.
    last_boundary = max(truncated.rfind("."), truncated.rfind("؟"), truncated.rfind("!"))
    if last_boundary > len(truncated) * 0.6:
        truncated = truncated[: last_boundary + 1]
    return truncated.strip()


# One fixed closing phrase chosen by the user. It is spoken by TTS and appears
# in the burned-in subtitles on every video; there are no alternate CTAs.
CTA_OUTRO = "إِذَا أَعْجَبَكَ الْفِيدْيُو، فَاضْغَطْ زِرَّ الإِعْجَابِ، وَلا تَنْسَ مُشَارَكَةَ الْفِيدْيُو"
OUTRO_MIN_WORDS = len(CTA_OUTRO.split())
OUTRO_MAX_WORDS = OUTRO_MIN_WORDS


def _append_engagement_outro(script: str) -> str:
    """Append the user's exact closing phrase once, never a random variant."""
    text = script.strip()
    if text.endswith(CTA_OUTRO):
        return text
    return f"{text} {CTA_OUTRO}".strip()


_TASHKEEL_CHARS = "\u0610-\u061A\u064B-\u065F\u06D6-\u06DC\u06DF-\u06E8\u06EA-\u06ED\u0670"
_TASHKEEL_RE = re.compile(f"[{_TASHKEEL_CHARS}]")


def _normalize_for_compare(s: str) -> str:
    """Strip tashkeel/diacritics and collapse whitespace so hook_text can be
    compared against narration_script even when the model re-typed the
    diacritics slightly differently (or dropped them)."""
    s = _TASHKEEL_RE.sub("", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# --- Automated self-check: catch a missing pronoun diacritic before it ships ---
# The fixed CTA's attached pronoun suffix (أَعْجَبَكَ) is checked here. The
# "attached pronoun suffixes must always be diacritized" rule is enforced in
# code so an edit cannot silently reintroduce a mispronunciation. A
# rule that only lives in a comment relies on someone re-reading and
# re-checking it by eye every time the string is touched — exactly what
# just failed twice. Enforcing it in code instead means a future edit that
# reintroduces the same mistake fails LOUDLY, at import time, before a
# single Groq/Pexels/TTS call is made, rather than shipping a mispronounced
# video that a viewer has to point out.
#
# Attached pronouns are ك (you) and ه (him/it); a diacritic on that letter
# is written as an EXTRA character immediately after it (Unicode combining
# marks follow their base letter), so if ك or ه is genuinely the last
# character of a word, it carries no vowel of its own. This can also
# legitimately trip on a word where ك/ه is a plain ROOT letter rather than
# an attached pronoun (e.g. a word that simply happens to end in ك or ه
# with no real pronunciation risk) — those go in
# _TASHKEEL_CHECK_ALLOWLIST below, each with a one-line reason, rather than
# silently special-cased, so the allowlist stays a deliberate, reviewable
# decision instead of quietly regrowing into the same blind spot.
_TASHKEEL_CHECK_ALLOWLIST = {
    "به",     # extremely common function word (bihi) — always reads the
              # same way unvocalized; already left unmarked elsewhere in
              # this same text (e.g. عليك, لنا) by the same convention.
    "عليك",   # same as above — a fixed, unambiguous collocation.
    "هذه",    # the ه here is part of the demonstrative's own spelling
              # (hādhihi), not an attached pronoun — always reads one way.
}


def _find_unmarked_pronoun_suffixes(text: str) -> list[str]:
    """Return every word in `text` that ends in a bare ك or ه (no
    diacritic on that letter) and isn't in _TASHKEEL_CHECK_ALLOWLIST."""
    offenders = []
    for raw_word in text.split():
        word = raw_word.strip(" ،.!؟\u061F:")
        if not word or word in _TASHKEEL_CHECK_ALLOWLIST:
            continue
        if word[-1] in ("ك", "ه"):
            offenders.append(word)
    return offenders


def _validate_fixed_arabic_strings() -> None:
    """Run once at import time over the fixed user-selected CTA.
    Deliberately NOT applied to Groq's own narration_script
    output: that text is dynamic and this same check would false-positive
    constantly on legitimate root-letter ك/ه endings in arbitrary content;
    Groq's output is governed instead by the tashkeel rules inside
    SYSTEM_PROMPT."""
    offenders = _find_unmarked_pronoun_suffixes(CTA_OUTRO)
    if offenders:
        raise PipelineError(
            f"CTA_OUTRO has word(s) ending in a bare ك/ه with no diacritic "
            f"on that letter: {offenders}. Add the missing tashkeel or a "
            f"justified entry to _TASHKEEL_CHECK_ALLOWLIST. Full string: {CTA_OUTRO!r}"
        )


_validate_fixed_arabic_strings()


def _fuzzy_word_pattern(word: str) -> str:
    """Regex matching `word` inside the original (diacritic-bearing) text,
    tolerating any tashkeel marks interleaved between its letters."""
    gap = f"[{_TASHKEEL_CHARS}]*"
    return gap.join(re.escape(ch) for ch in word)


def _strip_duplicate_hook(hook: str, script: str) -> str:
    """Remove any near-duplicate occurrence of hook_text left inside
    narration_script — wherever the model put it (start, middle, or end,
    e.g. mistakenly used as a closing "interactive question" instead of the
    opening hook) — so it isn't spoken/shown twice once we prepend it
    ourselves. Safe to call unconditionally: does nothing if no match."""
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
    # A random choice is safe here because the durable bank_id in history is the
    # actual uniqueness guard; reruns never silently reuse a selected entry.
    return random.SystemRandom().choice(available)


def choose_reachable_topic_seed(
    history: list[dict[str, Any]],
    blocked_categories: set[str] | None = None,
    excluded_source_signatures: set[tuple[str, ...]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, str]], list[dict[str, str]]]:
    """Choose a seed only after at least one of its allow-listed sources is fetched."""
    excluded = excluded_source_signatures if excluded_source_signatures is not None else set()
    while True:
        seed = choose_topic_seed(history, blocked_categories, excluded)
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
        excluded.add(signature)
        log.warning("Skipping topic seed %s before generation: all cited sources are inaccessible", seed["id"])

def load_topic_history() -> list[dict[str, Any]]:
    try:
        data = json.loads(TOPIC_HISTORY_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        return []


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
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    TOPIC_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    TOPIC_HISTORY_FILE.write_text(json.dumps(history[-HISTORY_LIMIT:], ensure_ascii=False, indent=2), encoding="utf-8")


def extract_json_block(text: str) -> dict[str, Any]:
    """Pull a JSON object out of a model response that may be wrapped in prose or fences."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise PipelineError(f"Could not find JSON object in model response: {text[:300]}")
    return json.loads(match.group(0))


# ---------------------------------------------------------------------------
# Step 1: Select a vetted topic-bank entry, then draft its script (Groq)
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
    - لا يوجد حد أدنى لطول النص؛ التزم بالحد الأعلى للكلمات وعدد المشاهد المطلوب فقط.
    - اجعل كلمات البحث والمشاهد الإنجليزية تصف الشيء المذكور فعلًا في الموضوع أو
      النص، ولا تستخدم كلمات عامة أو لقطات لا علاقة لها به.

    تذكير: لا تغيّر الموضوع أو الحقيقة أو الفئة أو العنوان. إذا لم تسمح الأدلة
    بنص مثير، فاكتب نصًا بسيطًا صحيحًا ولا تخترع الإثارة.
    """
).strip()

# Facebook Reels hard-caps posts at 90 seconds. Narration may be shorter; only
# the actual TTS duration is constrained, with a small speed-normalization
# margin for any audio that exceeds the maximum. MAX_SCRIPT_WORDS is an
# additional safety ceiling, not a target or minimum.
def _groq_chat(
    messages: list[dict[str, str]],
    max_completion_tokens: int = GROQ_MAX_COMPLETION_TOKENS,
    temperature: float = 0.75,
) -> str:
    """Shared Groq chat-completion call, forcing a JSON-object response.
    Used by both generate_topic (topic/script generation) and
    proofread_narration_tashkeel (the tashkeel proofreading pass) — pulled
    out to module level so a second Groq-backed step doesn't need its own
    copy of the same request/retry/429-handling logic."""
    payload = {
        "model": GROQ_MODEL,
        "messages": messages,
        "response_format": {"type": "json_object"},
        # Keep the completion budget bounded to reduce Groq TPM usage.
        "max_completion_tokens": max_completion_tokens,
        "reasoning_effort": "low",
        "temperature": temperature,
    }

    # 429s are handled in their own loop, separate from with_retries, so a
    # rate limit never costs the caller one of its own (much scarcer)
    # attempts — see GROQ_RATE_LIMIT_MAX_RETRIES above. Only a persistent
    # rate limit (or a non-429 failure) is raised out to the caller.
    for rate_limit_attempt in range(1, GROQ_RATE_LIMIT_MAX_RETRIES + 1):
        resp = requests.post(
            GROQ_ENDPOINT,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60,
        )
        if resp.status_code == 429:
            match = re.search(r"Please try again in\s+([0-9.]+)s", resp.text)
            wait_seconds = min(max(float(match.group(1)) if match else 15.0, 3.0), 90.0)
            log.warning(
                "Groq rate limit (429); waiting %.1fs before retry (%d/%d) — not counted "
                "against the caller's own retry budget",
                wait_seconds, rate_limit_attempt, GROQ_RATE_LIMIT_MAX_RETRIES,
            )
            time.sleep(wait_seconds + 1.0)
            continue
        if resp.status_code != 200:
            raise PipelineError(f"Groq API error {resp.status_code}: {resp.text[:500]}")
        data = resp.json()
        raw_text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        if not raw_text:
            raise PipelineError(f"Unexpected Groq response shape: {data}")
        return raw_text

    raise PipelineError(
        f"Groq API rate limit (429) persisted after {GROQ_RATE_LIMIT_MAX_RETRIES} internal retries"
    )


# --- Stage 1 (preventive): proofread tashkeel before the TTS ever records it ---
# The pipeline now has two independent layers of tashkeel QA, and they
# catch different things on purpose:
#   Stage 1 — proofread_narration_tashkeel() (this section): a dedicated
#     Groq proofreading pass that reviews the narration_script Groq JUST
#     generated, fixing missing/wrong diacritics BEFORE generate_tts ever
#     records it. Preventive — a fixed word is never spoken wrong at all.
#   Stage 2 — align_words_with_whisper()'s pronunciation_review (see that
#     section): runs on the ACTUAL rendered audio, after the fact. It
#     catches whatever Stage 1 missed, AND anything that isn't a tashkeel
#     problem at all (a TTS engine quirk unrelated to how the text was
#     vocalized). The two layers are complementary, not redundant — Stage
#     1 reduces how often Stage 2 has anything to flag; it can't replace it,
#     because no proofreading pass over TEXT can catch an engine-level
#     mispronunciation that only shows up in the AUDIO.
# --- Stage 1: mandatory Arabic + factual editorial review before TTS ---
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
    # Known fabricated/nonstandard terms from earlier generations.
    "السيلينس", "الشهرات الجوية", "المحتلة بالدقيق", "البركان الثلجي",
    # Fabricated lunar/astronomical mechanisms observed in production.
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
    """Mandatory editorial gate: grammar, meaning, factual plausibility, and tashkeel."""
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
    if new_words < max(70, int(orig_words * 0.70)) or new_words > int(orig_words * 1.35):
        raise PipelineError(
            f"Editorial review changed script length too much ({orig_words} -> {new_words} words)"
        )
    if len(re.findall(r"[\u0600-\u06FF]", corrected)) < max(40, int(new_words * 2.0)):
        raise PipelineError("Editorial review returned insufficient Arabic text")
    log.info("Arabic/scientific editorial gate passed (%d -> %d words)", orig_words, new_words)
    return corrected
def _validate_final_script_word_count(word_count: int) -> None:
    """Enforce only the final-script maximum; short videos are allowed."""
    if word_count > MAX_SCRIPT_WORDS:
        raise PipelineError(
            f"Final narration is {word_count} words; maximum is {MAX_SCRIPT_WORDS} "
            "including the fixed outro"
        )


def build_topic_user_prompt(seed: dict[str, Any], accessible_sources: list[dict[str, str]]) -> str:
    """Give Groq one fixed bank topic and only citations already fetched successfully."""
    source_urls = [str(source.get("url", "")).strip() for source in accessible_sources]
    source_urls = [url for url in source_urls if url]
    if not source_urls:
        raise PipelineError("Cannot write a topic without at least one preflighted source")

    pre_outro_max = MAX_SCRIPT_WORDS - OUTRO_MAX_WORDS
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
        + f"\n\nاكتب narration_script باختصار طبيعي يشرح الحقيقة بالقدر الكافي دون حشو، ولا يتجاوز {pre_outro_max} كلمة قبل العبارة الختامية الثابتة. "
        "ابدأه بـ hook_text نفسه. أعد category وtitle كما هما تمامًا من المدخل، "
        "واجعل scene_keywords_en بين 4 و7 عبارات مرتبطة بصريًا بالنص. أخرج JSON فقط."
    )


def generate_topic() -> tuple[Topic, list[dict[str, str]], list[dict[str, str]]]:
    log.info("Writing script for a pre-vetted topic-bank entry via Groq (%s)...", GROQ_MODEL)
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
            raise PipelineError("Groq returned an empty hook_text")

        # The prompt *asks* the model to open narration_script with
        # hook_text verbatim, but nothing enforced that — the model sometimes
        # drifts: rewords it, drops it, or (observed in
        # production) leaves it dangling at the END as the "closing
        # interactive question" instead of the opening hook. Don't trust the
        # model's placement at all: unconditionally strip any near-duplicate
        # of hook_text out of narration_script (wherever it ended up) and
        # prepend the real hook_text ourselves, so it's always first and
        # never doubled.
        topic.narration_script = _strip_duplicate_hook(topic.hook_text, topic.narration_script)
        topic.narration_script = f"{topic.hook_text} {topic.narration_script}".strip()

        return topic, len(topic.narration_script.split())

    def _call() -> Topic:
        history = load_topic_history()
        # Topic and category rotation happen here in code against the reviewed
        # bank. Groq receives exactly one selected entry and cannot choose a
        # different subject or category.
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

        # MAX_SCRIPT_WORDS bounds the final narration including the fixed CTA.
        # There is intentionally no minimum: short, evidence-backed scripts
        # are preferable to padding a video with unnecessary narration.
        pre_outro_max = MAX_SCRIPT_WORDS - OUTRO_MAX_WORDS

        if word_count > pre_outro_max:
            log.warning(
                "narration_script too long (%d words > %d incl. outro headroom); trimming at a "
                "sentence boundary to stay within the Facebook Reels 90s cap",
                word_count, pre_outro_max,
            )
            topic.narration_script = _trim_script_to_word_limit(topic.narration_script, pre_outro_max)
            word_count = len(topic.narration_script.split())

        # Account for the fixed closing phrase in the final video duration,
        # but append it only after text proofreading/cleanup in the pipeline.
        final_word_count = word_count + OUTRO_MIN_WORDS
        _validate_final_script_word_count(final_word_count)

        if len(topic.scene_keywords_en) < MIN_SCENE_CLIPS:
            raise PipelineError(
                f"Groq returned fewer than {MIN_SCENE_CLIPS} scene keywords; visual/text alignment is required"
            )
        if topic.category and recent_categories and topic.category in recent_categories[-2:]:
            raise PipelineError(
                f"Generated category {topic.category!r} repeats one of the last 2 used categories "
                f"{recent_categories[-2:]!r}; forcing a retry with a different domain"
            )
        if topic_is_too_similar(topic, load_topic_history()):
            raise PipelineError("Generated topic is too similar to a previously published topic")
        return topic

    topic = with_retries(_call, what="Groq topic generation", max_retries=TOPIC_GENERATION_MAX_ATTEMPTS)
    final_word_count = len(topic.narration_script.split()) + OUTRO_MIN_WORDS
    log.info(
        "Topic generated: %s (%d-word script, ~%.0fs at 1.75 words/sec)",
        topic.title, final_word_count, final_word_count / 1.75,
    )
    return topic, prefetched_sources, preflight_source_errors


def _proofread_topic_narration(topic: Topic, script: str) -> str:
    """Apply the same factual/Arabic review and TTS hygiene to a topic script."""
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

        # Only shuffle among the top N most-relevant results (Pexels returns
        # them ranked by relevance already) instead of the entire page, so we
        # never fall back to a loosely-related result ranked far down just
        # because it happened to have a portrait file first. If none of the
        # top candidates have a usable portrait file, fall through to the
        # rest of the page (still in Pexels's original relevance order)
        # rather than failing the scene outright.
        # Keep Pexels' relevance ranking intact. Randomizing these results can
        # replace a highly relevant clip with a generic one, which is exactly
        # what caused unrelated footage to appear in earlier runs.
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
            # Similar scene queries (e.g. all prefixed with the same
            # search_keywords_en context) frequently rank the same handful
            # of Pexels videos at the top for every single scene. Returning
            # the first portrait match unconditionally meant several scenes
            # could silently resolve to the exact same clip, which then
            # collapsed to fewer than MIN_SCENE_CLIPS distinct URLs once
            # de-duplicated in search_pexels_videos — failing the whole run
            # even though Pexels actually had enough different videos
            # further down the same results page. Skipping anything already
            # used by an earlier scene keeps walking down the ranked list
            # instead of failing later on a duplicate.
            if best["link"] in exclude:
                continue
            return best["link"]
        raise PipelineError(
            "No usable portrait-orientation video found in Pexels results "
            "(either none had a portrait file, or all were already used by another scene)"
        )

    return with_retries(_call, what="Pexels search")


def search_pexels_videos(keywords_list: list[str], topic_context: str = "") -> list[str]:
    """Fetch one portrait clip per semantic scene, avoiding duplicate URLs.

    Each scene query is combined with the topic's own search_keywords_en
    (e.g. "ancient egypt pyramids" + "stone carving") so Pexels doesn't just
    match the scene keyword in isolation and pull back generic footage with
    no real connection to the video's actual subject — the combined query
    is tried first and only falls back to the bare scene keyword if it
    returns nothing. Already-picked URLs are passed as `exclude` to every
    call so that scene queries which happen to look similar to Pexels (e.g.
    because topic_context is long or repetitive) can't silently collapse
    onto the same handful of clips — search_pexels_video keeps walking down
    the ranked results instead of returning a duplicate.
    """
    urls: list[str] = []
    for keywords in keywords_list:
        combined = f"{topic_context} {keywords}".strip() if topic_context else keywords
        try:
            # Do not fall back to a bare/generic scene query: that fallback was
            # the source of random footage unrelated to the video's subject.
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
    """Return n_scenes segment durations (summing to total_duration) for the
    background-video cuts, with each cut point snapped to a real TTS word
    boundary — and nudged onto the nearest sentence-ending pause when one is
    close by — instead of a blind equal-time split.

    A pure `total_duration / n_scenes` split (the previous behaviour) cuts
    to the next visual scene at an arbitrary instant that has no relation to
    what is actually being said at that moment, which is why the footage
    could feel out of sync with the narration even though the *audio* itself
    was perfectly in sync. Snapping cuts to word starts means a visual
    change never lands mid-word, and preferring a nearby sentence-ending
    boundary (a natural speaking pause) makes the cut coincide with a real
    beat in the narration whenever one is available.
    """
    if n_scenes <= 1:
        return [total_duration]
    words = [w for w in timings if w.get("text")]
    if not words:
        # No timing data available (e.g. TTS didn't report word boundaries) —
        # fall back to the old equal split rather than failing the run.
        even = total_duration / n_scenes
        return [even] * n_scenes

    offsets = [max(float(w.get("offset", 0)), 0.0) for w in words]
    texts = [str(w.get("text", "")) for w in words]
    ideal_points = [total_duration * i / n_scenes for i in range(1, n_scenes)]
    # How far from the ideal, equal-split point we're willing to look for a
    # better (word- or sentence-boundary) cut point.
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
    """Create a sequence of distinct portrait shots, one per clip, each held
    for its own segment_durations[i] seconds (see compute_scene_durations —
    these no longer need to be equal-length)."""
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
    """Create a quiet, license-free ambient pad with FFmpeg as an offline fallback."""
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
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not generate offline ambient music: %s", exc)
    return None


def get_bg_music(dest_dir: Path) -> Path | None:
    """Best-effort background-music selection. Never raises — a music
    problem should never fail the whole pipeline; it just publishes without
    music. Local files committed into the repo are tried first (no network
    call, so nothing to 404), then BG_MUSIC_URL, then silence.

    Checks several common folder names/locations (case-insensitive) and
    searches them recursively, since a single hardcoded "music/" folder at
    the repo root is a common source of silent misses if the folder was
    committed with a different name/casing or nested a level deep. Every
    candidate path checked is logged so a run with no music shows exactly
    where it looked."""
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
        except Exception as exc:  # noqa: BLE001
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
    """Transcribe generated audio and detect a contiguous phrase copied from
    SILMA's reference text. Returns the leaked phrase, or None."""
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


def generate_tts(text: str, out_path: Path, voice: str = TTS_VOICE) -> Path:
    """Generate SILMA, guard its audio, and fall back to Edge when needed."""
    if TTS_ENGINE == "silma":
        try:
            if SILMA_REFERENCE_PROFILE:
                reference, ref_text = resolve_reference_profile(
                    SILMA_REFERENCE_PROFILE, SILMA_VOICE_PROFILES_FILE
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
        except Exception:  # noqa: BLE001
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
    """Return true only when the actual voiceover exceeds the hard cap."""
    return duration_seconds > MAX_AUDIO_SECONDS


def _audio_duration_needs_normalization(duration_seconds: float) -> bool:
    """Leave short clips untouched; only normalize audio above the safety target."""
    return duration_seconds > TARGET_AUDIO_SECONDS


def normalize_narration_duration(audio_path: Path, target_seconds: float) -> float:
    """Fit TTS into a safe duration without generating another topic.

    atempo preserves the voice while making small speed corrections. Word
    boundary timestamps are scaled by the same factor so subtitles remain in
    sync after the correction.
    """
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


# ---------------------------------------------------------------------------
# Step 3b: Force-align subtitles to the actual rendered audio (Whisper)
# ---------------------------------------------------------------------------
# edge-tts self-reports "WordBoundary" timings while it synthesizes speech,
# and build_subtitles used to trust those numbers directly. In production
# this drifted badly: Azure's Arabic neural voices do not report per-word
# timing reliably, and the more tashkeel the input text carries (needed for
# correct pronunciation — see CTA_OUTRO / SYSTEM_PROMPT), the more
# those self-reported boundaries can be off, with the error compounding
# over a ~90-second video until the captions are visibly out of sync with
# what's actually being said.
#
# align_words_with_whisper() replaces that self-reported metadata with an
# independent, ground-truth measurement: it runs speech recognition
# (faster-whisper) on the ACTUAL final audio waveform and reads back real
# start/end times for each recognized word. Whisper's own transcription is
# only ever used to obtain timestamps — the words displayed and spoken
# always remain the original, tashkeel-bearing script text. Because
# Whisper's Arabic transcription has no diacritics and can occasionally
# mis-hear a word, its output is diff-aligned (tashkeel/diacritic-
# insensitive) against our own known script word-by-word; any script word
# that doesn't confidently match gets an interpolated timestamp from its
# nearest matched neighbours, so every word still ends up with a timing.
#
# Pronunciation QA (see below) is built on top of this same alignment, and
# it's worth being honest about what it can and can't catch. Comparing
# TEXT after stripping diacritics from both sides can only notice a
# completely different word being recognized (a dropped/extra/wrong word,
# or silence) — it is structurally blind to a word whose CONSONANTS were
# recognized correctly but whose SHORT VOWELS were off, because the
# diacritic-stripped spelling is identical either way. That vowel-only
# case is exactly the bug class already found by ear (جرس/جَرَس،
# ليصلَك/ليصلَكَ) — a pure text-match check would not have caught either
# one. faster-whisper's per-word `probability` (an acoustic confidence
# score, not a text-match score) is used as a second, complementary
# signal for exactly this blind spot: a word whose actual vowels sounded
# unusual tends to score lower confidence even when the recognized text
# still happens to match, since the score reflects what was actually
# heard, not just which letters got written down.
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL", "base")
# Below this acoustic confidence, a MATCHED word is still flagged for
# pronunciation review (see PRONUNCIATION_CONFIDENCE_THRESHOLD note above).
# Chosen conservatively low so normal ASR uncertainty (accents, minor
# audio artifacts) doesn't flood the report with false alarms — only
# words Whisper was genuinely unsure about get flagged.
LOW_CONFIDENCE_THRESHOLD = float(os.getenv("PRONUNCIATION_CONFIDENCE_THRESHOLD", "0.4"))


def align_words_with_whisper(
    audio_path: Path, script_words: list[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from faster_whisper import WhisperModel  # imported lazily; heavy optional dependency

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

    # Interpolate the handful of words Whisper didn't confidently match, so
    # every script word still ends up with a usable timestamp.
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

    # Pronunciation QA — two complementary signals, see the module comment
    # above align_words_with_whisper for why both are needed:
    #   "not_recognized" — Whisper's TEXT didn't match the script word at
    #     all (a different/dropped/extra word, or silence). Catches gross
    #     errors; blind to correct-consonants-wrong-vowel mistakes.
    #   "low_confidence"  — the text matched, but Whisper's ACOUSTIC score
    #     for that word was weak, which vowel-only mispronunciation can
    #     still trigger even though the written result looks identical.
    # This doesn't replace listening to the video, but it turns "many
    # words are being mispronounced" from a vague, ear-only complaint into
    # a concrete, per-run list of specific word indices to check first.
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
    return timings, flagged_words  # type: ignore[return-value]


def realign_subtitles_with_whisper(narration_path: Path, script_text: str) -> tuple[bool, list[dict[str, Any]]]:
    """Best-effort: overwrite narration_path's .timings.json with
    Whisper-derived timings, and return a pronunciation-QA list alongside
    the success flag. Never raises — if Whisper isn't installed, the model
    can't be fetched (e.g. no network), or alignment quality is too low,
    the existing edge-tts timings are left in place and the pipeline
    continues rather than failing the whole run over a subtitle-quality
    enhancement.

    edge-tts's own self-reported WordBoundary timestamps are known to run
    noticeably AHEAD of when the word is actually audible in Arabic neural
    voices (see align_words_with_whisper's module docstring) — this is
    exactly the "captions run ahead of the voice" symptom. If this function
    silently falls back, the video ships with that same known-bad timing,
    so the fallback is logged at ERROR (not warning) with the full
    traceback, and the caller records whether Whisper actually ran so it's
    checkable per-run without grepping the full log.
    """
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
    except Exception:  # noqa: BLE001
        log.error(
            "WHISPER ALIGNMENT FAILED — falling back to edge-tts's own word timings, which "
            "are known to run ahead of the actual audio. Captions in this video will likely "
            "be out of sync. Full error below:",
            exc_info=True,
        )
        return False, []


def _ass_escape(text: str) -> str:
    """Strip characters that have special meaning inside an ASS Dialogue
    Text field: '{' / '}' open/close an override block, and a raw newline
    would break the one-line-per-event .ass format."""
    return text.replace("{", "").replace("}", "").replace("\n", " ").replace("\r", " ")


# علامات تُحذف من الترجمة المعروضة على الشاشة فقط، وليس من النص المنطوق —
# النقطة والفاصلة وعلامات الاقتباس وما شابهها تبدو كإشارة واضحة لمحتوى
# مولَّد بالذكاء الاصطناعي عند ظهورها حرفيًا في ترجمة فيديو قصير. النص
# المُمرَّر لـ generate_tts (topic.narration_script) يبقى بعلاماته كاملة
# دون أي تغيير، لأن edge-tts يستخدمها لصناعة السكتات الصحيحة بين الجمل؛
# هذا التنظيف يُطبَّق فقط على نسخة الكلمات المعروضة في ملف الـ .ass.
DISPLAY_PUNCTUATION = str.maketrans(
    "".join([
        ".", ",", "،", "؛", ":", "!", "?", "؟", "…", "-", "—", "_",
        "(", ")", "[", "]", "{", "}", '"', "«", "»", "/", "\\",
    ]),
    " " * 23,
)


def _clean_display_words(words: list[str]) -> list[str]:
    # التشكيل الكامل (انظر SYSTEM_PROMPT) ضروري لضبط نطق edge-tts، لكن عرضه
    # حرفياً في الترجمة على الشاشة كان سيُظهر كل حركة/شدة فوق كل حرف تقريباً
    # — غير مقروء بصرياً في فيديو قصير. _TASHKEEL_RE (مُعرَّف أعلى الملف)
    # يُستخدم هنا لحذفه من نسخة العرض فقط؛ النص الأصلي الممرَّر لـ
    # generate_tts يبقى بتشكيله الكامل دون أي تغيير.
    cleaned = [
        _TASHKEEL_RE.sub("", w.translate(DISPLAY_PUNCTUATION)).strip()
        for w in words
    ]
    return [w for w in cleaned if w]


def build_subtitles(text: str, duration: float, out_path: Path, timings_path: Path | None = None) -> Path:
    """Build a two-line, RTL, word-timed .ass caption file synced to the
    edge-tts narration.

    This used to emit a plain .srt and rely on ffmpeg's `subtitles` filter
    to auto-convert it to ASS at render time via `force_style`. That
    auto-conversion silently falls back to a legacy 384x288 script
    resolution, and `original_size` does NOT reliably compensate for that
    (verified by rendering test frames: the caption came out far smaller
    and further left than the style values implied). Authoring a real
    .ass file directly, with PlayResX/PlayResY set to the *actual* output
    resolution, removes that guesswork entirely — the numbers below are
    real pixels on the real frame.
    """
    timing_data = []
    if timings_path and timings_path.exists():
        try:
            timing_data = json.loads(timings_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("Could not read TTS word timings; using fallback timing: %s", exc)

    timed_words = [x for x in timing_data if x.get("text")]

    if timed_words:
        # Build captions straight from what edge-tts itself reports it
        # spoke, in its own order and with its own offsets — this is the
        # only source guaranteed to match the audio. Previously the script
        # was re-split independently with text.split() and matched to the
        # TTS boundaries purely by index; any tokenization mismatch
        # (Arabic-Indic digits, elongated tashkeel, a merged/split token —
        # all common) silently shifted every caption after that point out
        # of sync with the voice.
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

        # Weight each chunk by character length instead of slicing the
        # audio into equal-length pieces per chunk of 8 words — a chunk of
        # eight short words takes noticeably less time to say than one of
        # eight long words, so equal slicing drifted increasingly out of
        # sync as the video went on. A chunk ending in sentence-final
        # punctuation (a natural pause point — e.g. right after the hook's
        # "؟") also gets extra weight to approximate the pause a real
        # speaker takes there, which is exactly where drift was most
        # noticeable (right at the start, after the hook question).
        def _chunk_weight(chunk: list[str]) -> float:
            weight = sum(len(w) for w in chunk) + len(chunk)  # +1/word for inter-word gaps
            if chunk and chunk[-1][-1:] in "؟?.!":
                weight += 6  # approximate end-of-sentence pause
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
        # ASS timestamp: H:MM:SS.cc (centiseconds, hour NOT zero-padded).
        t = max(t, 0.0)
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = int(t % 60)
        cs = int(round((t - int(t)) * 100))
        return f"{h:d}:{m:02}:{s:02}.{cs:02}"

    rtl = "\u200f"  # RIGHT-TO-LEFT MARK — forces each caption line to lay out RTL
    events = []
    for chunk_words, start, end in chunks:
        # Punctuation is stripped here (display only) — chunk_words itself,
        # and the TTS input elsewhere, keep every mark unchanged.
        display_words = _clean_display_words(chunk_words)
        if not display_words:
            continue
        split_at = max(1, (len(display_words) + 1) // 2)
        line1 = rtl + _ass_escape(" ".join(display_words[:split_at]))
        chunk_text = line1
        if len(display_words) > 1:
            line2 = rtl + _ass_escape(" ".join(display_words[split_at:]))
            chunk_text += "\\N" + line2  # \N = forced ASS line break
        events.append(f"Dialogue: 0,{fmt(start)},{fmt(end)},Caption,,0,0,0,,{chunk_text}")

    # Style tuned for a VIDEO_W x VIDEO_H (1080x1920) vertical frame:
    #   Alignment=8   -> anchors the block to TOP-center (7/8/9 = top row,
    #                    8 = horizontally centered within the margins)
    #   MarginV=260   -> distance from the top edge, in real pixels (since
    #                    PlayResY == VIDEO_H) — sits comfortably below the
    #                    phone's front-camera cutout
    #   MarginL/R=60  -> symmetric, so Alignment=8 centers on the true
    #                    frame center rather than an off-center box
    #   Fontsize=64   -> readable at this resolution without dominating
    #                    the screen; tune up/down to taste
    #   Outline=3, Shadow=0, BorderStyle=1 -> crisp white text with a
    #                    solid black outline, no separate drop shadow
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

    # Escape path for ffmpeg's subtitles filter (colon needs escaping on all platforms)
    subtitle_filter_path = str(subtitle_path).replace("\\", "/").replace(":", "\\:")

    vf = (
        f"scale={VIDEO_W}:{VIDEO_H}:force_original_aspect_ratio=increase,"
        f"crop={VIDEO_W}:{VIDEO_H},"
        # All styling (font size, TOP-center position, RTL-safe margins)
        # lives inside the .ass file itself (see build_subtitles), authored
        # directly at PlayResX/PlayResY = VIDEO_W x VIDEO_H — i.e. the real
        # output resolution. No force_style/original_size juggling needed
        # here, which is what was silently mispositioning the captions
        # before (verified with rendered test frames).
        f"subtitles='{subtitle_filter_path}'"
    )

    audio_inputs = ["-i", str(narration)]
    if music_path is not None:
        audio_inputs += ["-i", str(music_path)]

    # Force a constant output frame rate explicitly. Without this the output
    # simply inherits whatever the Pexels source reports, and stock footage
    # is frequently VFR (variable frame rate) — looping it with
    # -stream_loop creates an irregular-duration frame at each loop seam,
    # which pulls the *average* frame rate ffprobe/Facebook measures below
    # the nominal value (e.g. a "24fps" source can average ~23.9 once
    # looped/cut). That's what triggered Facebook Reels' hard
    # "frame rate must be at least 24 fps" rejection. -r as an output
    # option forces ffmpeg to duplicate/drop frames as needed to hit an
    # exact, constant rate — 30fps here, safely clear of the 24fps floor
    # rather than sitting right on the edge of it.
    OUTPUT_FPS = 30

    if music_path is not None:
        # Ducked mix: keep narration at full volume, music quiet underneath,
        # loop/trim the music to the narration's exact length, short fades
        # at the start/end so it doesn't cut off abruptly.
        filter_complex = (
            f"[2:a]aloop=loop=-1:size=2e9,atrim=0:{audio_duration:.2f},"
            f"afade=t=in:st=0:d=1.5,afade=t=out:st={max(audio_duration - 1.5, 0):.2f}:d=1.5,"
            f"volume={MUSIC_VOLUME}[music];"
            f"[1:a][music]amix=inputs=2:duration=first:dropout_transition=2:normalize=0[aout]"
        )
        cmd = [
            "ffmpeg", "-y",
            "-stream_loop", "-1", "-i", str(bg_video),
            *audio_inputs,
            "-t", f"{audio_duration:.2f}",
            "-vf", vf,
            "-r", str(OUTPUT_FPS),
            "-filter_complex", filter_complex,
            "-map", "0:v:0", "-map", "[aout]",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            str(out_path),
        ]
    else:
        cmd = [
            "ffmpeg", "-y",
            "-stream_loop", "-1", "-i", str(bg_video),
            *audio_inputs,
            "-t", f"{audio_duration:.2f}",
            "-vf", vf,
            "-r", str(OUTPUT_FPS),
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
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
# Buffer's API doesn't accept a raw file upload for posts — it needs a public,
# unauthenticated HTTPS URL that it can fetch the video from (see
# https://developers.buffer.com/guides/hosting-media.html). We get that for
# free by publishing the finished video as an asset on a new GitHub Release
# in this same repository; GitHub serves release assets over a public HTTPS
# URL with no login required.

def _ensure_repo_is_public(api_base: str, headers: dict[str, str]) -> None:
    """GitHub Release assets on a PRIVATE repo require an authenticated
    request to download — Buffer's fetch bot has no such credentials, so it
    gets a generic 'Video could not be read from its URL.' error. Fail fast
    with a clear message instead of letting that opaque error surface later."""
    resp = requests.get(api_base, headers=headers, timeout=30)
    if resp.status_code == 200 and resp.json().get("private"):
        raise PipelineError(
            "This repository is Private. GitHub Release assets in a private repo can't be "
            "downloaded without authentication, so Buffer's bot can't fetch the video from its "
            "public URL (this is what causes Buffer's 'Video could not be read from its URL.' "
            "error). Fix: make the repository Public — Settings -> General -> Danger Zone -> "
            "Change visibility -> Make public."
        )


def host_video_on_github(video_path: Path, run_id: str) -> str:
    if not GH_RELEASE_TOKEN or not GITHUB_REPOSITORY:
        raise PipelineError(
            "GH_RELEASE_TOKEN / GITHUB_REPOSITORY not set — both are required to host "
            "the video publicly for Buffer. Make sure the workflow has "
            "'permissions: contents: write' and passes GH_RELEASE_TOKEN: "
            "${{ secrets.GITHUB_TOKEN }} to this step."
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
    # upload_url comes back as a URI template, e.g. ".../assets{?name,label}" — strip the template part.
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

# YouTube requires a category on every video post. 24 = Entertainment, a
# reasonable default for viral short-form facts content. Override via the
# YOUTUBE_CATEGORY_ID env var if you'd rather use another category
# (e.g. 27 = Education, 28 = Science & Technology).
YOUTUBE_CATEGORY_ID = os.getenv("YOUTUBE_CATEGORY_ID", "24")


def get_channel_service(channel_id: str) -> str:
    """Ask Buffer which platform (youtube/facebook/tiktok/...) a channel ID
    belongs to, so we know which network-specific fields it requires."""

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
    """Different networks require different per-post fields on a video post
    (this is why the same generic mutation failed with network-specific
    'X is required' errors). Only the channel's own network needs metadata."""
    service = (service or "").lower()
    if service == "youtube":
        return {"youtube": {"title": topic.title[:100], "categoryId": YOUTUBE_CATEGORY_ID}}
    if service == "facebook":
        # A 9:16 vertical video published to a Facebook Page is a Reel.
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

    # Stage 1 tashkeel QA (see the block comment above
    # proofread_narration_tashkeel) — runs BEFORE anything is recorded, so
    # a caught diacritic mistake is never actually spoken. topic.json below
    # is written with the already-proofread script, so the saved record
    # always matches exactly what generate_tts receives.
    topic.narration_script = _proofread_topic_narration(topic, topic.narration_script)
    # Proofreading is intentionally complete before appending the fixed CTA,
    # so no language model can rewrite, shorten, or remove the user's exact words.
    topic.narration_script = _append_engagement_outro(topic.narration_script)
    final_word_count = len(topic.narration_script.split())
    _validate_final_script_word_count(final_word_count)
    red_flag = find_content_red_flag(topic.narration_script)
    if red_flag:
        raise PipelineError(f"Rejected hallucinated or nonstandard content term: {red_flag}")

    # Hard publication gate: no TTS, video assembly, release upload, or
    # Buffer request may happen until every extracted scientific claim is
    # supported by an allow-listed source at the configured confidence.
    fact_report = fact_check_topic(
        topic,
        run_dir / "fact_check.json",
        prefetched_sources=prefetched_sources,
        preflight_source_errors=preflight_source_errors,
    )
    if fact_report.get("status") != "PASS":
        raise PipelineError(
            "Fact Check rejected the final script; audio and publishing are blocked. "
            f"Details: {fact_report.get('errors', [])}"
        )
    log.info("Fact Check passed: %d supported claim(s)", len(fact_report.get("claims", [])))

    (run_dir / "topic.json").write_text(
        json.dumps(topic.__dict__, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Generate narration first so the final video length is measured from the
    # actual TTS audio, then choose enough distinct visual scenes to cover it.
    max_duration_attempts = 3
    narration_path = None
    audio_duration = 0.0
    for attempt in range(1, max_duration_attempts + 1):
        narration_path = generate_tts(topic.narration_script, run_dir / "narration.mp3")
        audio_duration = get_media_duration(narration_path)
        log.info("Narration audio duration: %.1fs (attempt %d/%d)", audio_duration, attempt, max_duration_attempts)
        if _audio_duration_needs_normalization(audio_duration):
            log.warning(
                "Narration exceeds the %.1fs safe target at %.1fs; correcting speed to %.1fs",
                TARGET_AUDIO_SECONDS, audio_duration, TARGET_AUDIO_SECONDS,
            )
            audio_duration = normalize_narration_duration(narration_path, TARGET_AUDIO_SECONDS)
        if not _audio_duration_exceeds_limit(audio_duration):
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
        # Keep the same topic; a duration mismatch is a voice-speed issue, not
        # a reason to spend another Groq request or risk topic repetition.

    # Re-derive word timings from the actual final audio (Whisper) instead of
    # trusting edge-tts's own self-reported WordBoundary metadata — see
    # realign_subtitles_with_whisper's docstring for why. Must run after any
    # atempo speed correction above, so it aligns against the exact audio
    # that will be published.
    whisper_aligned, pronunciation_flags = realign_subtitles_with_whisper(narration_path, topic.narration_script)
    (run_dir / "whisper_alignment_status.json").write_text(
        json.dumps({"whisper_aligned": whisper_aligned}, ensure_ascii=False), encoding="utf-8"
    )
    # Pronunciation QA report for this specific video — see
    # align_words_with_whisper's docstring. Written even when the list is
    # empty, so "no flags this run" is a visible, checkable fact rather
    # than an absent file that looks the same as "never checked".
    (run_dir / "pronunciation_review.json").write_text(
        json.dumps(pronunciation_flags, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    remember_topic(topic)
    scene_urls = search_pexels_videos(topic.scene_keywords_en, topic.search_keywords_en)
    scene_paths = [download_file(url, run_dir / f"scene_{i:02d}.mp4") for i, url in enumerate(scene_urls)]
    word_timings = _load_word_timings(narration_path.with_suffix(".timings.json"))
    segment_durations = compute_scene_durations(word_timings, audio_duration, len(scene_paths))
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
    if _audio_duration_exceeds_limit(final_video_duration):
        raise PipelineError(
            f"Assembled video is {final_video_duration:.2f}s; the maximum is "
            f"{MAX_AUDIO_SECONDS:.0f}s. Publishing is blocked."
        )
    log.info("Assembled video duration: %.2fs (maximum %.0fs)", final_video_duration, MAX_AUDIO_SECONDS)

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
    except Exception as exc:  # noqa: BLE001
        log.exception("Unexpected error: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
