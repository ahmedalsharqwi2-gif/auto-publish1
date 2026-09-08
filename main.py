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
# Prefer a direct GitHub Raw URL. Multiple URLs may be comma-separated.
BG_MUSIC_GITHUB_RAW_URL = _clean_env("BG_MUSIC_GITHUB_RAW_URL")
MUSIC_VOLUME = float(os.getenv("MUSIC_VOLUME", "0.18"))
MUSIC_DIR = Path(__file__).resolve().parent / "music"

FONT_PATH = Path(__file__).resolve().parent / "Cairo-Bold.ttf"
CAIRO_FONT_URL = os.getenv(
    "CAIRO_FONT_URL",
    "https://raw.githubusercontent.com/google/fonts/main/ofl/cairo/Cairo%5Bslnt,wght%5D.ttf",
)
FONT_SIZE = 60
TEXT_Y = 320

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
    hook_text: str          # 1-2 short shocking Arabic sentences (the opening hook only)
    narration_script: str    # full multi-paragraph script that gets spoken + subtitled
    title: str               # Video title
    caption: str              # Caption for the post
    hashtags: list[str] = field(default_factory=list)
    search_keywords_en: str = ""  # English keywords for Pexels search


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
      "hook_text": "جملة أو جملتان قصيرتان وصادمتان بالعربية الفصحى المبسطة تُفتتح بها السكريبت، لا تتجاوز 20 كلمة إجمالاً، ويجب أن تكون مُشكّلة بالكامل (تشكيل كامل لكل حرف: فتحة/ضمة/كسرة/سكون/شدة/تنوين)",
      "narration_script": "السكريبت الكامل الذي سيُروى بصوت التعليق ويظهر كترجمة على الفيديو. يجب أن يبدأ بنص hook_text نفسه حرفياً (بنفس التشكيل)، ثم يكمل بحقيقة صادمة واحدة فقط بإيجاز شديد بلا أي حشو أو تفاصيل جانبية، وينتهي بخاتمة قصيرة جداً أو سؤال تفاعلي من كلمات قليلة. يجب ألا يقل إجمالي عدد الكلمات عن 90 كلمة ولا يزيد عن 130 كلمة بالعربية الفصحى المبسطة (فيديو قصير جداً جداً ومكثّف بشدة يصلح لجميع منصات النشر بما فيها Facebook Reels ذات الحد الأقصى 90 ثانية)، مقسم إلى جمل قصيرة جداً وواضحة تصلح للترجمة النصية على الشاشة. يجب أن يكون النص بأكمله مُشكّلاً تشكيلاً كاملاً وصحيحاً نحوياً (كل كلمة، وليس فقط أواخر الكلمات) حتى ينطقه محرك تحويل النص إلى كلام بشكل صحيح ومفهوم",
      "title": "عنوان جذاب قصير بالعربية",
      "caption": "كابشن للمنشور بالعربية، 1-3 جمل",
      "hashtags": ["#وسم1", "#وسم2", "#وسم3", "#وسم4", "#وسم5"],
      "search_keywords_en": "2-4 English keywords describing matching vertical stock footage, e.g. 'deep ocean underwater'"
    }

    تعليمات إلزامية بخصوص الطول (لا تتجاهلها):
    - حقل narration_script يجب أن يحتوي على 90-130 كلمة عربية بالضبط تقريباً — ليس أكثر وليس أقل. هذا فيديو قصير جداً (Micro-short)، وليس فيديو Shorts عادياً.
    - الفيديو النهائي يُنشر على Facebook Reels التي تفرض حداً أقصى صارماً بـ90 ثانية فعلية للصوت المسموع، وسرعة الراوي أبطأ مما يبدو (حوالي 1.7-1.8 كلمة/ثانية فقط)، لذلك يجب الالتزام الصارم بحد 130 كلمة كسقف مطلق.
    - عدّ الكلمات فعلياً قبل إنهاء الإجابة، ولا تُسلّم نصاً أطول أو أقصر من المطلوب.

    تعليمات إلزامية بخصوص التشكيل (لا تتجاهلها):
    - كل من hook_text وnarration_script يجب أن يكونا مُشكّلين تشكيلاً كاملاً (Full Arabic Diacritics/Tashkeel) على كل حرف تقريباً في كل كلمة، وليس فقط الحركة الإعرابية الأخيرة، وذلك حتى تنطقهما محركات تحويل النص إلى كلام (TTS) بنطق صحيح وواضح.
    - استخدم الحركات القياسية (فتحة، ضمة، كسرة، سكون، شدة، تنوين بالفتح/الضم/الكسر) بدقة نحوية سليمة.
    - لا تترك أي كلمة بدون تشكيل، حتى الكلمات القصيرة والحروف مثل (مِنْ، فِي، عَلَى، وَ، لَا).
    """
).strip()

# Measured from real production runs: ar-EG-SalmaNeural speaks Arabic at
# roughly 1.7-1.8 words/second — much slower than a naive estimate would
# suggest. Facebook Reels hard-caps posts at 90 seconds, so the script must
# stay well under that: 90-130 words keeps spoken narration in the ~55-75s
# range even at the slower end of the measured rate. MAX_SCRIPT_WORDS is a
# hard ceiling — scripts longer than this get trimmed at a sentence boundary
# as a safety net, and MAX_AUDIO_SECONDS is a second, final safety net
# checked against the *actual* generated audio duration before publishing.
MIN_SCRIPT_WORDS = 80
MAX_SCRIPT_WORDS = 135
MAX_AUDIO_SECONDS = 82.0


def generate_topic() -> Topic:
    log.info("Generating viral topic via Groq (%s)...", GROQ_MODEL)

    def _groq_chat(messages: list[dict[str, str]], strict_json: bool = True) -> str:
        payload = {
            "model": GROQ_MODEL,
            "messages": messages,
            # GPT-OSS may spend completion tokens on hidden reasoning before
            # emitting JSON. 1536 was too small for a 90-130 word Arabic script.
            "max_completion_tokens": 4096,
            "reasoning_effort": "low",
            "temperature": 0.9,
        }
        if strict_json:
            payload["response_format"] = {"type": "json_object"}
        resp = requests.post(
            GROQ_ENDPOINT,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60,
        )
        if resp.status_code != 200:
            # Some Groq/model combinations reject strict JSON mode even though
            # they can return a valid JSON object in ordinary text mode.
            # Retry once without response_format; extract_json_block() below
            # already handles markdown fences and surrounding prose.
            if strict_json and resp.status_code == 400 and "json_validate_failed" in resp.text:
                log.warning("Groq strict JSON mode failed; retrying in text mode")
                return _groq_chat(messages, strict_json=False)
            raise PipelineError(f"Groq API error {resp.status_code}: {resp.text[:800]}")
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
        )
        if not topic.hook_text:
            raise PipelineError("Groq returned an empty hook_text")
        return topic, len(topic.narration_script.split())

    def _call() -> Topic:
        user_msg = (
            "أعطني فكرة فيديو اليوم بصيغة JSON كما هو محدد. "
            "تذكير مهم: narration_script يجب أن يكون بين 90 و130 كلمة عربية بالضبط — "
            "هذا الشرط أهم من أي شرط آخر في الطلب، وسيتم رفض أي إجابة أطول أو أقصر."
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
        expand_attempts = 0
        while word_count < MIN_SCRIPT_WORDS and expand_attempts < 2:
            expand_attempts += 1
            log.warning(
                "narration_script too short (%d words); asking model to expand in place (attempt %d/2)...",
                word_count, expand_attempts,
            )
            messages.append({"role": "assistant", "content": raw_text})
            messages.append({
                "role": "user",
                "content": (
                    f"السكريبت الذي كتبته يحتوي على {word_count} كلمة فقط، وهذا أقل من المطلوب. "
                    f"أعد كتابة نفس كائن JSON بالكامل، مع الإبقاء على hook_text كما هو حرفياً، "
                    f"لكن وسّع narration_script قليلاً بإضافة تفصيلة واحدة إضافية بسيطة "
                    f"حتى يصل إلى 100-130 كلمة. أجب حصراً بكائن JSON صالح."
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

    music_urls = ",".join(filter(None, [BG_MUSIC_GITHUB_RAW_URL, BG_MUSIC_URL]))
    if music_urls:
        candidates = [u.strip() for u in music_urls.split(",") if u.strip()]
        if candidates:
            url = random.choice(candidates)
            dest = dest_dir / "music.mp3"
            try:
                # GitHub links must be Raw/content links, not an HTML repository page.
                if "github.com" in url and "/raw/" not in url and "raw.githubusercontent.com" not in url:
                    raise PipelineError("GitHub music URL is not a direct Raw URL")
                download_file(url, dest)
                if dest.stat().st_size < 1024:
                    raise PipelineError("Downloaded music file is unexpectedly small")
                log.info("Background music: %s", url)
                return dest
            except Exception as exc:  # noqa: BLE001
                log.warning("Could not fetch BG_MUSIC_URL track — publishing without music: %s", exc)
                return None

    log.info(
        "No background music configured — checked %s (recursively) and GitHub Raw URL for "
        "%s files, and BG_MUSIC_URL is unset. Add audio files to one of "
        "those folders (committed to the repo) or set BG_MUSIC_URL.",
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
        await communicate.save(str(out_path))

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


def build_srt(text: str, duration: float, out_path: Path) -> Path:
    """Naive proportional-timing SRT: splits text into short chunks and
    spaces them evenly across the audio duration. Each chunk is kept as a
    SINGLE line (small word count per chunk) so it reliably fits within the
    subtitle safe-area at the configured font size without wrapping onto a
    second line — a wrapped/forced two-line block is what was pushing
    captions above the top of the frame. Good enough for short-form hook
    videos; swap in a forced-aligner/Whisper timestamp pass for
    frame-perfect sync."""
    words = text.split()
    # Kept small (3 words) so a single line comfortably fits inside
    # MarginL/MarginR at the configured FontSize, even for wider Arabic
    # glyphs, without libass auto-wrapping it onto a second line.
    chunk_size = 3
    word_chunks = [words[i:i + chunk_size] for i in range(0, len(words), chunk_size)] or [words]
    per_chunk = duration / len(word_chunks)

    def fmt(t: float) -> str:
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = int(t % 60)
        ms = int((t - int(t)) * 1000)
        return f"{h:02}:{m:02}:{s:02},{ms:03}"

    lines = []
    for i, chunk_words in enumerate(word_chunks):
        start = i * per_chunk
        end = (i + 1) * per_chunk
        chunk_text = " ".join(chunk_words)
        lines.append(str(i + 1))
        lines.append(f"{fmt(start)} --> {fmt(end)}")
        lines.append(chunk_text)
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def ensure_arabic_font() -> Path:
    """Download the real Cairo Arabic font once, with a clear failure message."""
    if FONT_PATH.exists() and FONT_PATH.stat().st_size > 10_000:
        return FONT_PATH
    try:
        log.info("Downloading Arabic font: %s", CAIRO_FONT_URL)
        with requests.get(CAIRO_FONT_URL, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            tmp = FONT_PATH.with_suffix(".tmp")
            with tmp.open("wb") as out:
                for chunk in resp.iter_content(1 << 16):
                    if chunk:
                        out.write(chunk)
            if tmp.stat().st_size < 10_000:
                raise PipelineError("Downloaded Cairo font is invalid or incomplete")
            tmp.replace(FONT_PATH)
    except (requests.RequestException, OSError) as exc:
        raise PipelineError(f"Could not download Cairo-Bold.ttf: {exc}") from exc
    return FONT_PATH


def _arabic_display(line: str) -> str:
    """Shape Arabic first, then apply bidi display ordering (never reverse manually)."""
    import arabic_reshaper
    from bidi.algorithm import get_display

    configuration = {"delete_harakat": False, "support_ligatures": True}
    reshaper = arabic_reshaper.ArabicReshaper(configuration=configuration)
    return get_display(reshaper.reshape(line))


def build_subtitle_video(text: str, duration: float, out_dir: Path) -> Path:
    """Render correctly shaped Arabic text at y=320 into timed transparent PNGs."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise PipelineError(
            "Missing Pillow. Install all dependencies with: "
            "python -m pip install -r requirements.txt"
        ) from exc

    try:
        import arabic_reshaper  # noqa: F401
        from bidi.algorithm import get_display  # noqa: F401
    except ImportError as exc:
        raise PipelineError(
            "Missing Arabic text dependencies. Install with: "
            "python -m pip install -r requirements.txt"
        ) from exc

    font = ImageFont.truetype(str(ensure_arabic_font()), FONT_SIZE)
    # The official Google Fonts file is variable; select its Bold instance when
    # Pillow exposes variation controls, while retaining a safe fallback.
    if hasattr(font, "set_variation_by_name"):
        try:
            font.set_variation_by_name("Bold")
        except (OSError, ValueError):
            log.warning("Could not select Cairo Bold variation; using default instance")
    words = text.split()
    chunks = [words[i:i + 4] for i in range(0, len(words), 4)] or [[]]
    per_chunk = duration / len(chunks)
    image_paths: list[Path] = []

    for index, chunk in enumerate(chunks):
        # Word wrap is performed on original Arabic text, before shaping/bidi.
        original_line = " ".join(chunk)
        rendered_line = _arabic_display(original_line)
        image = Image.new("RGBA", (VIDEO_W, VIDEO_H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        bbox = draw.textbbox((0, 0), rendered_line, font=font, stroke_width=4)
        text_width = bbox[2] - bbox[0]
        x = (VIDEO_W - text_width) / 2 - bbox[0]
        draw.text(
            (x, TEXT_Y), rendered_line, font=font, fill="white",
            stroke_width=4, stroke_fill="black", anchor=None,
        )
        path = out_dir / f"subtitle_{index:04d}.png"
        image.save(path, "PNG")
        image_paths.append(path)

    concat = out_dir / "subtitles.concat.txt"
    with concat.open("w", encoding="utf-8") as f:
        for index, path in enumerate(image_paths):
            safe_path = str(path.resolve()).replace("'", "'\\''")
            f.write(f"file '{safe_path}'\n")
            segment = per_chunk if index < len(image_paths) - 1 else duration - per_chunk * index
            f.write(f"duration {max(segment, 0.01):.6f}\n")
        # concat requires the final file to be repeated to honor the last duration.
        if image_paths:
            safe_path = str(image_paths[-1].resolve()).replace("'", "'\\''")
            f.write(f"file '{safe_path}'\n")

    # qtrle preserves the transparent alpha channel and is supported in MOV;
    # MP4 cannot store qtrle reliably on common FFmpeg builds.
    subtitle_video = out_dir / "subtitles.mov"
    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat),
        "-vf", f"scale={VIDEO_W}:{VIDEO_H},format=rgba", "-t", f"{duration:.3f}",
        "-c:v", "qtrle", str(subtitle_video),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not subtitle_video.exists():
        raise PipelineError(f"Subtitle rendering failed: {result.stderr[-1500:]}")
    return subtitle_video


# ---------------------------------------------------------------------------
# Step 4: Video assembly (ffmpeg)
# ---------------------------------------------------------------------------

def assemble_video(
    bg_video: Path, narration: Path, subtitle_video: Path, out_path: Path,
    music_path: Path | None = None,
) -> Path:
    log.info("Assembling final video...")
    audio_duration = get_media_duration(narration)

    vf = (
        f"scale={VIDEO_W}:{VIDEO_H}:force_original_aspect_ratio=increase,"
        f"crop={VIDEO_W}:{VIDEO_H}[base]"
    )

    audio_inputs = ["-i", str(subtitle_video), "-i", str(narration)]
    if music_path is not None:
        audio_inputs += ["-i", str(music_path)]

    if music_path is not None:
        # Ducked mix: keep narration at full volume, music quiet underneath,
        # loop/trim the music to the narration's exact length, short fades
        # at the start/end so it doesn't cut off abruptly.
        filter_complex = (
            f"[3:a]aloop=loop=-1:size=2e9,atrim=0:{audio_duration:.2f},"
            f"afade=t=in:st=0:d=1.5,afade=t=out:st={max(audio_duration - 1.5, 0):.2f}:d=1.5,"
            f"volume={max(0.01, min(MUSIC_VOLUME, 1.0)):.3f}[music];"
            f"[2:a][music]amix=inputs=2:duration=first:dropout_transition=2:normalize=0[aout];"
            f"[0:v]{vf};[1:v]format=rgba[subs];[base][subs]overlay=0:0:format=auto[vout]"
        )
        cmd = [
            "ffmpeg", "-y",
            "-stream_loop", "-1", "-i", str(bg_video),
            *audio_inputs,
            "-t", f"{audio_duration:.2f}",
            "-filter_complex", filter_complex,
            "-map", "[vout]", "-map", "[aout]",
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
            "-filter_complex", f"[0:v]{vf};[1:v]format=rgba[subs];[base][subs]overlay=0:0:format=auto[vout]",
            "-map", "[vout]", "-map", "2:a:0",
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

    bg_video_url = search_pexels_video(topic.search_keywords_en)
    bg_video_path = download_file(bg_video_url, run_dir / "background.mp4")
    music_path = get_bg_music(run_dir)

    # Generating the narration is comparatively cheap, so if the actual
    # spoken audio comes out longer than the Facebook Reels safety cap, retry
    # with a fresh (hopefully shorter) topic a couple of times before giving
    # up on this run entirely — cheaper than re-downloading footage/music.
    max_duration_attempts = 3
    narration_path = None
    audio_duration = 0.0
    for attempt in range(1, max_duration_attempts + 1):
        narration_path = generate_tts(topic.narration_script, run_dir / "narration.mp3")
        audio_duration = get_media_duration(narration_path)
        log.info("Narration audio duration: %.1fs (attempt %d/%d)", audio_duration, attempt, max_duration_attempts)
        if audio_duration <= MAX_AUDIO_SECONDS:
            break
        log.warning(
            "Narration audio is %.1fs, over the %.0fs Facebook Reels safety cap — "
            "generating a fresh, shorter topic (attempt %d/%d)",
            audio_duration, MAX_AUDIO_SECONDS, attempt, max_duration_attempts,
        )
        if attempt == max_duration_attempts:
            raise PipelineError(
                f"Narration audio stayed over the {MAX_AUDIO_SECONDS:.0f}s safety cap after "
                f"{max_duration_attempts} attempts — aborting this run rather than publish an "
                "over-length video"
            )
        topic = generate_topic()
        (run_dir / "topic.json").write_text(
            json.dumps(topic.__dict__, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    subtitle_video_path = build_subtitle_video(
        topic.narration_script, audio_duration, run_dir
    )

    final_video_path = assemble_video(
        bg_video_path, narration_path, subtitle_video_path,
        run_dir / "final.mp4", music_path=music_path,
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
