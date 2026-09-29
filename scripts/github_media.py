"""Host completed reels on public GitHub Releases for Buffer to fetch."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

import requests

log = logging.getLogger("pipeline")
GITHUB_API_BASE = "https://api.github.com"
MAX_RELEASE_ASSET_BYTES = 2 * 1024 * 1024 * 1024


class MediaHostingError(RuntimeError):
    """Raised when an MP4 cannot be hosted at a public stable URL."""


def host_video_on_github(
    video_path: Path,
    *,
    token: str | None = None,
    repository: str | None = None,
    run_id: str | None = None,
    run_attempt: str | None = None,
    api_base: str = GITHUB_API_BASE,
) -> str:
    """Upload an MP4 to a public release and return its browser download URL.

    Release assets are intentionally retained: Buffer may fetch queued media
    hours or days after the post is created.
    """
    video_path = Path(video_path)
    token = token or os.getenv("GH_RELEASE_TOKEN") or os.getenv("GITHUB_TOKEN")
    repository = repository or os.getenv("GITHUB_REPOSITORY")
    run_id = run_id or os.getenv("GITHUB_RUN_ID")
    run_attempt = run_attempt or os.getenv("GITHUB_RUN_ATTEMPT", "1")
    if not token or not repository:
        raise MediaHostingError("GH_RELEASE_TOKEN/GITHUB_TOKEN and GITHUB_REPOSITORY are required")
    if not video_path.is_file() or video_path.stat().st_size == 0:
        raise MediaHostingError(f"Video file is missing or empty: {video_path}")
    if video_path.stat().st_size > MAX_RELEASE_ASSET_BYTES:
        raise MediaHostingError("Video exceeds GitHub's 2 GB release-asset limit")

    safe_run_id = re.sub(r"[^A-Za-z0-9._-]", "-", run_id or str(os.getpid()))
    safe_attempt = re.sub(r"[^A-Za-z0-9._-]", "-", run_attempt)
    tag = f"auto-publish-{safe_run_id}-attempt-{safe_attempt}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    repository_url = f"{api_base.rstrip('/')}/repos/{repository}"

    repo_response = requests.get(repository_url, headers=headers, timeout=30)
    if repo_response.status_code != 200:
        raise MediaHostingError(
            f"Could not verify public GitHub repository: HTTP {repo_response.status_code}"
        )
    if repo_response.json().get("private"):
        raise MediaHostingError("Repository is private; Buffer cannot fetch its release assets")

    release_response = requests.post(
        f"{repository_url}/releases",
        headers=headers,
        json={
            "tag_name": tag,
            "name": f"Auto-publish reel {safe_run_id}-{safe_attempt}",
            "body": "Public media asset used by Buffer for an auto-published reel. Kept available for queued posts.",
            "draft": False,
            "prerelease": False,
        },
        timeout=30,
    )
    if release_response.status_code not in (200, 201):
        raise MediaHostingError(
            f"GitHub release creation failed: HTTP {release_response.status_code} "
            f"{release_response.text[:500]}"
        )
    release = release_response.json()
    upload_url = str(release.get("upload_url", "")).split("{", 1)[0]
    if not upload_url:
        raise MediaHostingError("GitHub did not return a release upload URL")

    with video_path.open("rb") as video_file:
        upload_response = requests.post(
            upload_url,
            headers={**headers, "Content-Type": "video/mp4"},
            params={"name": video_path.name},
            data=video_file,
            timeout=300,
        )
    if upload_response.status_code not in (200, 201):
        raise MediaHostingError(
            f"GitHub video upload failed: HTTP {upload_response.status_code} "
            f"{upload_response.text[:500]}"
        )
    asset_url = upload_response.json().get("browser_download_url")
    if not isinstance(asset_url, str) or not asset_url.startswith("https://"):
        raise MediaHostingError("GitHub did not return a public HTTPS asset URL")
    log.info("Hosted reel as a public GitHub release asset: %s", asset_url)
    return asset_url
