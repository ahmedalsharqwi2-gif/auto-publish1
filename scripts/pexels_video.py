"""Download and prepare portrait stock clips from the Pexels video API."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from urllib.parse import urlparse

import requests

PEXELS_SEARCH_URL = "https://api.pexels.com/videos/search"


def visual_query(topic: str) -> str:
    """Map Arabic editorial topics to stable English stock-video queries."""
    text = (topic or "").lower()
    mapping = (
        (("فضاء", "فلك", "نجوم", "كواكب", "كون"), "space stars galaxy"),
        (("محيط", "بحر", "أعماق", "ماء"), "ocean underwater"),
        (("طبيعة", "غابة", "حيوان", "حيوانات"), "nature forest wildlife"),
        (("طب", "جسم", "دماغ", "مرض"), "medical laboratory human body"),
        (("هندسة", "فيزياء", "تقنية", "اختراع", "روبوت"), "technology science laboratory"),
    )
    for words, query in mapping:
        if any(word in text for word in words):
            return query
    return os.getenv("PEXELS_DEFAULT_QUERY", "science technology nature")


def search_portrait_videos(api_key: str, query: str, per_page: int = 15) -> list[str]:
    response = requests.get(
        PEXELS_SEARCH_URL,
        headers={"Authorization": api_key},
        params={"query": query, "orientation": "portrait", "size": "large", "per_page": per_page},
        timeout=30,
    )
    response.raise_for_status()
    urls: list[str] = []
    for video in response.json().get("videos", []):
        files = video.get("video_files") or []
        candidates = [
            item for item in files
            if item.get("link") and item.get("width", 0) >= 540 and item.get("height", 0) >= item.get("width", 0)
        ]
        candidates.sort(key=lambda item: (item.get("width", 0), item.get("height", 0)), reverse=True)
        if candidates:
            urls.append(candidates[0]["link"])
    return list(dict.fromkeys(urls))


def _download(url: str, destination: Path) -> None:
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with destination.open("wb") as output:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    output.write(chunk)


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
    """Build a full-length portrait track; return False for safe visual fallback."""
    if not api_key:
        return False
    workdir = output_path.parent / "pexels_clips"
    workdir.mkdir(parents=True, exist_ok=True)
    try:
        urls = search_portrait_videos(api_key, visual_query(topic))
        if not urls:
            return False
        remaining = duration
        normalized: list[Path] = []
        for index, url in enumerate(urls[:5]):
            if remaining <= 0:
                break
            suffix = Path(urlparse(url).path).suffix or ".mp4"
            raw = workdir / f"raw_{index}{suffix}"
            clip = workdir / f"clip_{index}.mp4"
            _download(url, raw)
            segment_duration = min(8.0, remaining)
            _normalize_clip(raw, clip, segment_duration)
            normalized.append(clip)
            remaining -= segment_duration
        if not normalized:
            return False
        # If the API returned too little footage, loop the prepared visual track.
        concat_list = workdir / "concat.txt"
        concat_list.write_text("\n".join(f"file '{path.resolve()}'" for path in normalized) + "\n", encoding="utf-8")
        subprocess.run(
            [
                "ffmpeg", "-y", "-v", "error", "-stream_loop", "-1", "-f", "concat", "-safe", "0",
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
