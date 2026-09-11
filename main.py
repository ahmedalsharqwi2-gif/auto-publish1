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
MIN_AUDIO_SECONDS = float(os.getenv("MIN_AUDIO_SECONDS", "85"))
MAX_AUDIO_SECONDS = float(os.getenv("MAX_AUDIO_SECONDS", "89"))
MIN_SCRIPT_WORDS = int(os.getenv("MIN_SCRIPT_WORDS", "145"))
MAX_SCRIPT_WORDS = int(os.getenv("MAX_SCRIPT_WORDS", "165"))
HISTORY_LIMIT = int(os.getenv("HISTORY_LIMIT", "200"))

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

TTS_VOICE = os.getenv("TTS_VOICE", "ar-EG-SalmaNeural")

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_MAX_COMPLETION_TOKENS = int(os.getenv("GROQ_MAX_COMPLETION_TOKENS", "2200"))
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
PEXELS_SEARCH_ENDPOINT = "https://api.pexels.com/videos/search"
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


@dataclass
class Topic:
    hook_text: str          # one strange/curious Arabic question (the opening hook — no answer)
    narration_script: str    # full multi-paragraph script that gets spoken + subtitled
    title: str               # Video title
    caption: str              # Caption for the post
    hashtags: list[str] = field(default_factory=list)
    search_keywords_en: str = ""  # English keywords for Pexels search
    scene_keywords_en: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def check_env() -> None:
    missing = [k for k, v in REQUIRED_ENV.items() if not v]
    if not BUFFER_CHANNEL_IDS:
        missing.append("BUFFER_CHANNEL_IDS")
    if missing:
        raise PipelineError(f"Missing required environment variables: {', '.join(missing)}")


def with_retries(fn, *args, what: str = "operation", **kwargs):
    last_err: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            log.warning("Attempt %d/%d for %s failed: %s", attempt, MAX_RETRIES, what, exc)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)
    raise PipelineError(f"{what} failed after {MAX_RETRIES} attempts: {last_err}") from last_err


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


_TASHKEEL_CHARS = "\u0610-\u061A\u064B-\u065F\u06D6-\u06DC\u06DF-\u06E8\u06EA-\u06ED\u0670"
_TASHKEEL_RE = re.compile(f"[{_TASHKEEL_CHARS}]")


def _normalize_for_compare(s: str) -> str:
    """Strip tashkeel/diacritics and collapse whitespace so hook_text can be
    compared against narration_script even when the model re-typed the
    diacritics slightly differently (or dropped them)."""
    s = _TASHKEEL_RE.sub("", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


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
# Step 1: Topic generation (Groq)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent(
    """
    أنت خبير عالمي في التسويق الفيروسي (Viral Marketing) وصناعة محتوى الفيديوهات
    القصيرة (Shorts/Reels/TikTok) باللغة العربية. مهمتك اختيار فكرة/موضوع واحد فقط
    لفيديو اليوم من أحد المجالات التالية: الغرائب والعجائب، الحقائق العلمية الصادمة،
    أسرار الفضاء، أسرار المحيطات.

    اختر موضوعاً لم يُستهلك بشكل مبتذل، وله معدل جذب مرتفع (High Hook Rate) في أول
    3 ثوانٍ (High-CTR). أجب حصراً بكائن JSON صالح دون أي نص إضافي أو علامات كود،
    بالمفاتيح التالية:

    {
      "hook_text": "سؤال واحد فقط، غريب وغير متوقع ومثير للفضول، بالعربية الفصحى المبسطة، يُفتتح به الفيديو. يجب أن يُصاغ حرفياً كسؤال ينتهي بعلامة استفهام (؟)، ولا يكشف الإجابة إطلاقاً، ولا يتجاوز 15 كلمة. الهدف الوحيد منه أن يجعل المشاهد غير قادر على تجاوز الفيديو قبل معرفة الإجابة. استخدم تشكيلاً جزئياً وخفيفاً فقط (وليس تشكيلاً كاملاً) في المواضع التي قد يلتبس نطقها بدونه",
      "narration_script": "السكريبت الكامل الذي سيُروى بصوت التعليق ويظهر كترجمة على الفيديو. يبدأ بـ hook_text حرفياً ثم يجيب عنه بتفاصيل موثوقة ومثيرة في فقرات مترابطة، وينتهي بخاتمة قصيرة. يجب أن يكون بين 145 و165 كلمة عربية، لإنتاج فيديو بين 85 و89 ثانية دون تجاوز 90 ثانية، مقسم إلى جمل قصيرة واضحة.",
      "title": "عنوان جذاب قصير بالعربية",
      "caption": "كابشن للمنشور بالعربية، 1-3 جمل",
      "hashtags": ["#وسم1", "#وسم2", "#وسم3", "#وسم4", "#وسم5"],
      "search_keywords_en": "2-4 English keywords for the main subject",
      "scene_keywords_en": ["5-7 English searches, one per visual scene, in the exact order of the narration; each must visibly represent the paragraph it accompanies"]
    }

    تعليمات إلزامية بخصوص الهوك (لا تتجاهلها):
    - hook_text يجب أن يكون دائماً سؤالاً غريباً بصيغة استفهامية حقيقية (وليس جملة إخبارية صادمة)، مثل: "لماذا لا تستطيع...؟" أو "ما السبب الحقيقي وراء...؟" أو "هل تعلم ماذا يحدث لو...؟".
    - لا تكشف الإجابة في hook_text إطلاقاً — الإجابة تأتي فقط داخل narration_script، بعد إعادة صياغة السؤال نفسه حرفياً في بدايته.
    - تجنّب الأسئلة المستهلكة أو المتوقعة؛ اختر زاوية غريبة وغير شائعة حتى لو كان الموضوع نفسه معروفاً، بحيث يشعر المشاهد أنه *يجب* أن يعرف الإجابة.

    تعليمات إلزامية بخصوص الطول (لا تتجاهلها):
    - حقل narration_script يجب أن يحتوي على 145-165 كلمة عربية، بما يستهدف مدة صوتية بين 85 و89 ثانية دون تجاوز 90 ثانية.
    - scene_keywords_en إلزامي: كل عبارة يجب أن تمثل جزءاً محدداً من النص، ولا تستخدم كلمات عامة لا علاقة لها بالموضوع.
    - عدّ الكلمات فعلياً قبل إنهاء الإجابة، ولا تُسلّم نصاً أطول أو أقصر من المطلوب.

    تعليمات إلزامية بخصوص التشكيل (لا تتجاهلها):
    - استخدم تشكيلاً جزئياً وخفيفاً (Selective/Light Tashkeel) فقط في المواضع التي قد يلتبس نطقها أو معناها بدون تشكيل (كلمات متشابهة رسماً ومختلفة نطقاً، أفعال قد تُقرأ بأكثر من صيغة، كلمات نادرة، إلخ).
    - لا تضع تشكيلاً على كل حرف في كل كلمة — هذا غير مطلوب، ويجعل الترجمة النصية الظاهرة على الشاشة مزدحمة بصرياً دون داعٍ.
    - اترك الكلمات الواضحة النطق بدون أي تشكيل، وتجنّب تشكيل أواخر الكلمات إعرابياً إلا إذا كان ضرورياً فعلاً لتفادي التباس حقيقي في المعنى أو النطق.
    """
).strip()

# Measured from real production runs: ar-EG-SalmaNeural speaks Arabic at
# roughly 1.7-1.8 words/second — much slower than a naive estimate would
# suggest. Facebook Reels hard-caps posts at 90 seconds, so actual TTS duration
# is checked and must remain in the 85-89 second safety window. MAX_SCRIPT_WORDS is a
# hard ceiling — scripts longer than this get trimmed at a sentence boundary
# as a safety net, and MAX_AUDIO_SECONDS is a second, final safety net
# checked against the *actual* generated audio duration before publishing.
def generate_topic() -> Topic:
    log.info("Generating viral topic via Groq (%s)...", GROQ_MODEL)

    def _groq_chat(messages: list[dict[str, str]]) -> str:
        payload = {
            "model": GROQ_MODEL,
            "messages": messages,
            "response_format": {"type": "json_object"},
            # GPT-OSS may spend completion tokens on reasoning before emitting
            # JSON; 1536 was too small and caused json_validate_failed.
            "max_completion_tokens": 4096,
            "reasoning_effort": "low",
            "temperature": 0.9,
        }
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
            log.warning("Groq rate limit (429); waiting %.1fs before retry", wait_seconds)
            time.sleep(wait_seconds + 1.0)
            raise PipelineError(f"Groq API rate limit (429) after waiting {wait_seconds:.1f}s")
        if resp.status_code != 200:
            raise PipelineError(f"Groq API error {resp.status_code}: {resp.text[:500]}")
        data = resp.json()
        raw_text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        if not raw_text:
            raise PipelineError(f"Unexpected Groq response shape: {data}")
        return raw_text

    def _parse_topic(raw_text: str) -> tuple[Topic, int]:
        parsed = extract_json_block(raw_text)
        topic = Topic(
            hook_text=parsed["hook_text"].strip(),
            narration_script=parsed.get("narration_script", "").strip(),
            title=parsed["title"].strip(),
            caption=parsed.get("caption", "").strip(),
            hashtags=list(parsed.get("hashtags", [])),
            search_keywords_en=parsed.get("search_keywords_en", "nature abstract").strip(),
            scene_keywords_en=[str(x).strip() for x in parsed.get("scene_keywords_en", []) if str(x).strip()],
        )
        if not topic.hook_text:
            raise PipelineError("Groq returned an empty hook_text")

        # The prompt *asks* the model to open narration_script with
        # hook_text verbatim, but nothing enforced that — at temperature=0.9
        # the model frequently drifts: rewords it, drops it, or (observed in
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
        user_msg = (
            "أعطني فكرة فيديو جديدة بصيغة JSON كما هو محدد. "
            f"يجب أن يكون narration_script بين {MIN_SCRIPT_WORDS} و{MAX_SCRIPT_WORDS} كلمة، "
            "ويجب أن يحتوي scene_keywords_en على 5 إلى 7 مشاهد مرتبطة مباشرة بفقرات النص. "
            "لا تكرر أياً من الموضوعات السابقة التالية: "
            + json.dumps([x.get("title", "") for x in load_topic_history()[-40:]], ensure_ascii=False)
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        raw_text = _groq_chat(messages)
        topic, word_count = _parse_topic(raw_text)

        # If the script comes back short, don't throw the whole topic away —
        # ask the model, in the same conversation, to expand what it already
        # wrote. This fixes the actual problem (under-length output) instead
        # of just re-rolling the dice on a fresh topic with the same prompt.
        # One compact repair request is enough; repeated full JSON rewrites
        # consume the free-tier tokens-per-minute budget very quickly.
        expand_attempts = 0
        while word_count < MIN_SCRIPT_WORDS and expand_attempts < 1:
            expand_attempts += 1
            log.warning(
                "narration_script too short (%d words); asking model to expand in place (attempt %d/1)...",
                word_count, expand_attempts,
            )
            messages.append({"role": "assistant", "content": raw_text})
            messages.append({
                "role": "user",
                "content": (
                    f"السكريبت الذي كتبته يحتوي على {word_count} كلمة فقط، وهذا أقل من المطلوب. "
                    f"أعد كتابة نفس كائن JSON بالكامل، مع الإبقاء على hook_text كما هو حرفياً، "
                    f"لكن وسّع narration_script بتفاصيل مرتبطة مباشرة بالموضوع حتى يصل إلى {MIN_SCRIPT_WORDS}-{MAX_SCRIPT_WORDS} كلمة، "
                    "وأضف 5-7 scene_keywords_en مرتبطة بفقرات النص. أجب حصراً بكائن JSON صالح."
                ),
            })
            raw_text = _groq_chat(messages)
            topic, word_count = _parse_topic(raw_text)

        if word_count < MIN_SCRIPT_WORDS:
            # Still short after giving the model a chance to expand in place —
            # treat this as a failed attempt so with_retries tries a fresh
            # topic from scratch instead of shipping a too-short script.
            raise PipelineError(
                f"narration_script too short ({word_count} words, need >= {MIN_SCRIPT_WORDS}) "
                "— would produce a video under the safe length for Facebook Reels"
            )

        if word_count > MAX_SCRIPT_WORDS:
            log.warning(
                "narration_script too long (%d words > %d); trimming at a sentence boundary "
                "to stay within the Facebook Reels 90s cap",
                word_count, MAX_SCRIPT_WORDS,
            )
            topic.narration_script = _trim_script_to_word_limit(topic.narration_script, MAX_SCRIPT_WORDS)

        if len(topic.scene_keywords_en) < 5:
            raise PipelineError("Groq returned fewer than 5 scene keywords; visual/text alignment is required")
        if topic_is_too_similar(topic, load_topic_history()):
            raise PipelineError("Generated topic is too similar to a previously published topic")
        return topic

    topic = with_retries(_call, what="Groq topic generation")
    log.info(
        "Topic generated: %s (%d-word script, ~%.0fs at 1.75 words/sec)",
        topic.title, len(topic.narration_script.split()), len(topic.narration_script.split()) / 1.75,
    )
    return topic


# ---------------------------------------------------------------------------
# Step 2: Background footage (Pexels)
# ---------------------------------------------------------------------------

def search_pexels_video(keywords: str) -> str:
    log.info("Searching Pexels for vertical footage: %r", keywords)

    def _call() -> str:
        resp = requests.get(
            PEXELS_SEARCH_ENDPOINT,
            headers={"Authorization": PEXELS_API_KEY},
            params={"query": keywords, "orientation": "portrait", "size": "large", "per_page": 15},
            timeout=30,
        )
        if resp.status_code != 200:
            raise PipelineError(f"Pexels API error {resp.status_code}: {resp.text[:500]}")
        data = resp.json()
        videos = data.get("videos", [])
        if not videos:
            raise PipelineError(f"No Pexels results for keywords: {keywords!r}")

        random.shuffle(videos)
        for video in videos:
            files = [
                f for f in video.get("video_files", [])
                if f.get("width") and f.get("height") and f["height"] > f["width"]
            ]
            if not files:
                continue
            files.sort(key=lambda f: f["width"], reverse=True)
            best = files[0]
            return best["link"]
        raise PipelineError("No portrait-orientation video files found in Pexels results")

    return with_retries(_call, what="Pexels search")


def search_pexels_videos(keywords_list: list[str]) -> list[str]:
    """Fetch one portrait clip per semantic scene, avoiding duplicate URLs."""
    urls: list[str] = []
    for keywords in keywords_list:
        try:
            url = search_pexels_video(keywords)
            if url not in urls:
                urls.append(url)
        except PipelineError as exc:
            log.warning("No clip for scene %r: %s", keywords, exc)
    if len(urls) < 5:
        raise PipelineError(f"Only {len(urls)} distinct scene clips found; refusing to repeat unrelated footage")
    return urls


def build_multishot_background(clips: list[Path], duration: float, out_path: Path) -> Path:
    """Create a sequence of distinct portrait shots covering the narration duration."""
    segment = duration / len(clips)
    inputs: list[str] = []
    filters: list[str] = []
    for i, clip in enumerate(clips):
        inputs += ["-stream_loop", "-1", "-i", str(clip)]
        filters.append(
            f"[{i}:v]trim=duration={segment:.3f},setpts=PTS-STARTPTS,"
            f"scale={VIDEO_W}:{VIDEO_H}:force_original_aspect_ratio=increase,crop={VIDEO_W}:{VIDEO_H},fps=30[v{i}]"
        )
    filters.append("".join(f"[v{i}]" for i in range(len(clips))) +
                   f"concat=n={len(clips)}:v=1:a=0[outv]")
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

def generate_tts(text: str, out_path: Path, voice: str = TTS_VOICE) -> Path:
    log.info("Generating TTS narration with voice %s", voice)
    import edge_tts  # imported lazily so the script can still be linted without the dep

    async def _run():
        communicate = edge_tts.Communicate(text, voice)
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
            raise PipelineError("TTS produced an empty audio file")
        return out_path

    return with_retries(_call, what="TTS generation")


def get_media_duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(path),
        ],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


def _ass_escape(text: str) -> str:
    """Strip characters that have special meaning inside an ASS Dialogue
    Text field: '{' / '}' open/close an override block, and a raw newline
    would break the one-line-per-event .ass format."""
    return text.replace("{", "").replace("}", "").replace("\n", " ").replace("\r", " ")


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
        split_at = max(1, (len(chunk_words) + 1) // 2)
        line1 = rtl + _ass_escape(" ".join(chunk_words[:split_at]))
        chunk_text = line1
        if len(chunk_words) > 1:
            line2 = rtl + _ass_escape(" ".join(chunk_words[split_at:]))
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
            f"volume=0.15[music];"
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
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = WORK_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log.info("Run directory: %s", run_dir)

    topic = generate_topic()
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
        if MIN_AUDIO_SECONDS <= audio_duration <= MAX_AUDIO_SECONDS:
            break
        log.warning(
            "Narration audio is %.1fs, target is %.0f-%.0fs — generating a fresh topic (attempt %d/%d)",
            audio_duration, MIN_AUDIO_SECONDS, MAX_AUDIO_SECONDS, attempt, max_duration_attempts,
        )
        if attempt == max_duration_attempts:
            raise PipelineError(
                f"Narration audio did not reach the target {MIN_AUDIO_SECONDS:.0f}-{MAX_AUDIO_SECONDS:.0f}s range "
                f"after {max_duration_attempts} attempts — aborting rather than publish the wrong length"
            )
        topic = generate_topic()
        (run_dir / "topic.json").write_text(
            json.dumps(topic.__dict__, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    remember_topic(topic)
    scene_urls = search_pexels_videos(topic.scene_keywords_en)
    scene_paths = [download_file(url, run_dir / f"scene_{i:02d}.mp4") for i, url in enumerate(scene_urls)]
    bg_video_path = build_multishot_background(scene_paths, audio_duration, run_dir / "background.mp4")
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
