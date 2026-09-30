"""Download and prepare topic-specific portrait clips from the Pexels API."""
from __future__ import annotations

import math
import os
import subprocess
from pathlib import Path
from urllib.parse import urlparse

import requests

PEXELS_SEARCH_URL = "https://api.pexels.com/videos/search"
CLIP_SECONDS = 6.0
MIN_CLIPS = 11
MAX_CLIPS = 30


def visual_queries(topic: str) -> list[str]:
    """Return several closely related English queries for the Arabic topic."""
    text = (topic or "").lower()
    mapping = (
        (("فضاء", "فلك", "نجوم", "كواكب", "كون"), [
            "space stars galaxy", "astronomy telescope", "nebula planets night sky",
        ]),
        (("محيط", "بحر", "أعماق", "ماء"), [
            "ocean underwater", "deep sea marine life", "waves coral reef",
        ]),
        (("طبيعة", "غابة", "حيوان", "حيوانات"), [
            "nature forest wildlife", "mountain landscape river", "animals close up nature",
        ]),
        (("طب", "جسم", "دماغ", "مرض"), [
            "medical laboratory", "human body science", "microscope cells research",
        ]),
        (("هندسة", "فيزياء", "تقنية", "اختراع", "روبوت"), [
            "technology science laboratory", "robot engineering machine", "physics experiment energy",
        ]),
    )
    for words, queries in mapping:
        if any(word in text for word in words):
            return queries
    default = os.getenv("PEXELS_DEFAULT_QUERY", "science laboratory technology")
    return [default, "scientific research laboratory", "technology experiment"]


def visual_query(topic: str) -> str:
    """Backward-compatible primary query used by callers and tests."""
    return visual_queries(topic)[0]


def search_portrait_videos(api_key: str, query: str, per_page: int = 80) -> list[str]:
    response = requests.get(
        PEXELS_SEARCH_URL,
        headers={"Authorization": api_key},
        # Do not restrict the API to portrait: relevant landscape footage is
        # safely center-cropped to 9:16 by _normalize_clip below.
        params={"query": query, "size": "large", "per_page": per_page},
        timeout=30,
    )
    response.raise_for_status()
    urls: list[str] = []
    for video in response.json().get("videos", []):
        files = video.get("video_files") or []
        candidates = [
            item for item in files
            if item.get("link")
            and item.get("width", 0) >= 540
            and item.get("height", 0) >= 540
        ]
        # Prefer portrait, then choose the highest usable resolution.
        candidates.sort(
            key=lambda item: (
                item.get("height", 0) >= item.get("width", 0),
                item.get("width", 0) * item.get("height", 0),
            ),
            reverse=True,
        )
        if candidates:
            urls.append(candidates[0]["link"])
    return list(dict.fromkeys(urls))


def _download(url: str, destination: Path) -> None:
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            destination.unlink(missing_ok=True)
            with requests.get(url, stream=True, timeout=(20, 180)) as response:
                response.raise_for_status()
                with destination.open("wb") as output:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            output.write(chunk)
            if destination.stat().st_size > 0:
                return
        except (OSError, requests.RequestException) as exc:
            last_error = exc
            print(f"⚠️ إعادة تنزيل مقطع Pexels {attempt}/3 بعد انقطاع الشبكة.")
    if last_error:
        raise last_error


def _normalize_clip(source: Path, destination: Path, duration: float) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error", "-stream_loop", "-1", "-i", str(source),
            "-t", f"{duration:.3f}",
            "-vf", "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,setsar=1",
            "-an", "-r", "30", "-c:v", "libx264", "-preset", "veryfast", "-crf", "24",
            str(destination),
        ],
        check=True,
    )


def build_pexels_track(api_key: str, topic: str, duration: float, output_path: Path) -> bool:
    """Build a full-length track from unique, topic-specific clips only."""
    if not api_key:
        return False
    workdir = output_path.parent / "pexels_clips"
    workdir.mkdir(parents=True, exist_ok=True)
    required = max(MIN_CLIPS, math.ceil(duration / CLIP_SECONDS))
    try:
        urls: list[str] = []
        seen: set[str] = set()
        for query in visual_queries(topic):
            for url in search_portrait_videos(api_key, query):
                if url not in seen:
                    seen.add(url)
                    urls.append(url)
                if len(urls) >= min(MAX_CLIPS, required + 5):
                    break
            if len(urls) >= min(MAX_CLIPS, required + 5):
                break
        if len(urls) < required:
            print(f"⚠️ Pexels أعاد {len(urls)} مقاطع فقط، والمطلوب {required} مقطعًا؛ لن نكرر مقطعًا.")
            return False

        normalized: list[Path] = []
        remaining = duration
        for index, url in enumerate(urls[:required]):
            if remaining <= 0:
                break
            suffix = Path(urlparse(url).path).suffix or ".mp4"
            raw = workdir / f"raw_{index}{suffix}"
            clip = workdir / f"clip_{index}.mp4"
            try:
                _download(url, raw)
                segment_duration = min(CLIP_SECONDS, remaining)
                _normalize_clip(raw, clip, segment_duration)
                normalized.append(clip)
                remaining -= segment_duration
            except (OSError, requests.RequestException, subprocess.CalledProcessError) as exc:
                print(f"⚠️ تخطي مقطع Pexels غير صالح ({exc}).")

        if len(normalized) < required or remaining > 0.05:
            print(f"⚠️ تم تجهيز {len(normalized)} مقاطع فقط؛ لن نعيد أي مقطع لتغطية المدة.")
            return False

        concat_list = workdir / "concat.txt"
        concat_list.write_text(
            "\n".join(f"file '{path.resolve()}'" for path in normalized) + "\n",
            encoding="utf-8",
        )
        subprocess.run(
            [
                "ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
                "-i", str(concat_list), "-t", f"{duration:.3f}", "-an",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "24", "-pix_fmt", "yuv420p",
                str(output_path),
            ],
            check=True,
        )
        return output_path.exists() and output_path.stat().st_size > 0
    except (OSError, requests.RequestException, subprocess.CalledProcessError, ValueError) as exc:
        print(f"⚠️ تعذر جلب مقاطع Pexels ({exc}) — استخدام الخلفية الاحتياطية.")
        return False
