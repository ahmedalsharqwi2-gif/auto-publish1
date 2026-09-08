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
# Optional: set BG_MUSIC_URL (a GitHub Secret or plain env var) to a direct,
# publicly-downloadable audio file URL to always use your own track. You can
# pass several URLs separated by commas and one will be picked at random each
# run. If it's unset, the pipeline falls back to the small set of stable,
# long-standing Kevin MacLeod / incompetech.com tracks below (CC BY 4.0 —
# free to use, attribution required. If you'd rather not deal with
# attribution, set BG_MUSIC_URL to a CC0 track of your own instead).
BG_MUSIC_URL = _clean_env("BG_MUSIC_URL")
DEFAULT_BG_MUSIC_TRACKS = [
    ("https://incompetech.com/music/royalty-free/mp3-royaltyfree/Cipher.mp3", "Cipher by Kevin MacLeod"),
    ("https://incompetech.com/music/royalty-free/mp3-royaltyfree/Investigations.mp3", "Investigations by Kevin MacLeod"),
    ("https://incompetech.com/music/royalty-free/mp3-royaltyfree/Mystery%20Sax.mp3", "Mystery Sax by Kevin MacLeod"),
]

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
      "hook_text": "جملة أو جملتان قصيرتان وصادمتان بالعربية الفصحى المبسطة تُفتتح بها السكريبت، لا تتجاوز 25 كلمة إجمالاً",
      "narration_script": "السكريبت الكامل الذي سيُروى بصوت التعليق ويظهر كترجمة على الفيديو. يجب أن يبدأ بنص hook_text نفسه حرفياً، ثم يكمل بشرح شيّق ومكثّف (أهم حقيقة أو حقيقتين صادمتين فقط، بلا حشو) وينتهي بخاتمة قوية أو سؤال تفاعلي قصير. يجب ألا يقل إجمالي عدد الكلمات عن 130 كلمة ولا يزيد عن 170 كلمة بالعربية الفصحى المبسطة (فيديو قصير جداً ومكثّف يصلح لجميع منصات النشر بما فيها Facebook Reels ذات الحد الأقصى 90 ثانية)، مقسم إلى عدة جمل قصيرة وواضحة تصلح للترجمة النصية على الشاشة",
      "title": "عنوان جذاب قصير بالعربية",
      "caption": "كابشن للمنشور بالعربية، 1-3 جمل",
      "hashtags": ["#وسم1", "#وسم2", "#وسم3", "#وسم4", "#وسم5"],
      "search_keywords_en": "2-4 English keywords describing matching vertical stock footage, e.g. 'deep ocean underwater'"
    }

    تعليمات إلزامية بخصوص الطول (لا تتجاهلها):
    - حقل narration_script يجب أن يحتوي على 130-170 كلمة عربية بالضبط تقريباً — ليس أكثر وليس أقل.
    - الفيديو النهائي يُنشر على Facebook Reels التي تفرض حداً أقصى صارماً بـ90 ثانية، لذلك يجب أن يبقى السكريبت مختصراً ومكثفاً (فكرة واحدة قوية، بلا استطراد).
    - عدّ الكلمات فعلياً قبل إنهاء الإجابة، ولا تُسلّم نصاً أطول أو أقصر من المطلوب.
    """
).strip()

# edge-tts narration speaks at roughly 2.0-2.4 Arabic words/second for the
# ar-EG-SalmaNeural voice (use the conservative low end for safety margins).
# Facebook Reels hard-caps posts at 90 seconds, so the script must stay well
# under that: 130-170 words keeps the spoken narration in the ~55-80s range
# even at the slower end of that rate. MAX_SCRIPT_WORDS is a hard ceiling —
# scripts longer than this get trimmed at a sentence boundary as a safety
# net, and MAX_AUDIO_SECONDS is a second, final safety net checked against
# the *actual* generated audio duration before we ever try to publish.
MIN_SCRIPT_WORDS = 110
MAX_SCRIPT_WORDS = 190
MAX_AUDIO_SECONDS = 85.0


def generate_topic() -> Topic:
    log.info("Generating viral topic via Groq (%s)...", GROQ_MODEL)

    def _groq_chat(messages: list[dict[str, str]]) -> str:
        payload = {
            "model": GROQ_MODEL,
            "messages": messages,
            "response_format": {"type": "json_object"},
            "max_completion_tokens": 1536,
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
        )
        if not topic.hook_text:
            raise PipelineError("Groq returned an empty hook_text")
        return topic, len(topic.narration_script.split())

    def _call() -> Topic:
        user_msg = (
            "أعطني فكرة فيديو اليوم بصيغة JSON كما هو محدد. "
            "تذكير مهم: narration_script يجب ألا يقل عن 360 كلمة عربية — "
            "هذا الشرط أهم من أي شرط آخر في الطلب، وسيتم رفض أي إجابة أقصر."
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
                    f"لكن وسّع narration_script بإضافة تفاصيل/أمثلة/مقارنات/حقائق إضافية ذات صلة "
                    f"حتى يصل إلى 380-420 كلمة على الأقل. أجب حصراً بكائن JSON صالح."
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
        "Topic generated: %s (%d-word script, ~%.0fs at 2.0 words/sec)",
        topic.title, len(topic.narration_script.split()), len(topic.narration_script.split()) / 2.0,
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
    """Best-effort background-music download. Never raises — a music problem
    should never fail the whole pipeline; it just publishes without music,
    same as before."""
    if BG_MUSIC_URL:
        candidates = [(u.strip(), "your BG_MUSIC_URL track") for u in BG_MUSIC_URL.split(",") if u.strip()]
    else:
        candidates = DEFAULT_BG_MUSIC_TRACKS

    if not candidates:
        return None

    url, label = random.choice(candidates)
    dest = dest_dir / "music.mp3"
    try:
        download_file(url, dest)
        log.info("Background music: %s", label)
        return dest
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not fetch background music (%s) — publishing without music: %s", label, exc)
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
    """Naive proportional-timing SRT: splits text into chunks and spaces them
    evenly across the audio duration. Good enough for short-form hook videos;
    swap in a forced-aligner/Whisper timestamp pass for frame-perfect sync."""
    words = text.split()
    chunk_size = 4
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)] or [text]
    per_chunk = duration / len(chunks)

    def fmt(t: float) -> str:
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = int(t % 60)
        ms = int((t - int(t)) * 1000)
        return f"{h:02}:{m:02}:{s:02},{ms:03}"

    lines = []
    for i, chunk in enumerate(chunks):
        start = i * per_chunk
        end = (i + 1) * per_chunk
        lines.append(str(i + 1))
        lines.append(f"{fmt(start)} --> {fmt(end)}")
        lines.append(chunk)
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# Step 4: Video assembly (ffmpeg)
# ---------------------------------------------------------------------------

def assemble_video(
    bg_video: Path, narration: Path, srt_path: Path, out_path: Path, music_path: Path | None = None,
) -> Path:
    log.info("Assembling final video...")
    audio_duration = get_media_duration(narration)

    # Escape path for ffmpeg's subtitles filter (colon needs escaping on all platforms)
    srt_filter_path = str(srt_path).replace("\\", "/").replace(":", "\\:")

    vf = (
        f"scale={VIDEO_W}:{VIDEO_H}:force_original_aspect_ratio=increase,"
        f"crop={VIDEO_W}:{VIDEO_H},"
        # `original_size` MUST be set to the real output frame size. Without
        # it, ffmpeg's subtitles filter (via libass) assumes the legacy
        # default script resolution of 384x288 and scales/positions the text
        # for that instead of the actual 1080x1920 frame — which is exactly
        # what pushed the burned-in captions up into the middle of the video
        # instead of the intended lower third.
        f"subtitles='{srt_filter_path}':original_size={VIDEO_W}x{VIDEO_H}:force_style="
        "'FontName=Arial,FontSize=20,PrimaryColour=&H00FFFFFF,"
        "OutlineColour=&H00000000,BorderStyle=3,Outline=2,Alignment=2,MarginV=180'"
    )

    audio_inputs = ["-i", str(narration)]
    if music_path is not None:
        audio_inputs += ["-i", str(music_path)]

    if music_path is not None:
        # Ducked mix: keep narration at full volume, music quiet underneath,
        # loop/trim the music to the narration's exact length, short fades
        # at the start/end so it doesn't cut off abruptly.
        filter_complex = (
            f"[2:a]aloop=loop=-1:size=2e9,atrim=0:{audio_duration:.2f},"
            f"afade=t=in:st=0:d=1.5,afade=t=out:st={max(audio_duration - 1.5, 0):.2f}:d=1.5,"
            f"volume=0.18[music];"
            f"[1:a][music]amix=inputs=2:duration=first:dropout_transition=2:normalize=0[aout]"
        )
        cmd = [
            "ffmpeg", "-y",
            "-stream_loop", "-1", "-i", str(bg_video),
            *audio_inputs,
            "-t", f"{audio_duration:.2f}",
            "-vf", vf,
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

    srt_path = build_srt(topic.narration_script, audio_duration, run_dir / "subtitles.srt")

    final_video_path = assemble_video(
        bg_video_path, narration_path, srt_path, run_dir / "final.mp4", music_path=music_path,
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
