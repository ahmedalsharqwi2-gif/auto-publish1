#!/usr/bin/env python3
"""Safely renew an exhausted topic bank from Google Trends and YouTube.

The default is a read-only dry run. A candidate is writable only when both
providers returned evidence, it passes the local duplicate guard, and its
score is at least the configured threshold.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote_plus

import requests

try:
    from pytrends.request import TrendReq
except ImportError:  # Optional at import time so dry runs can fail closed.
    TrendReq = None

try:
    from scripts.topic_history import DuplicateTopicError, TopicHistory, find_duplicate
except ImportError:  # pragma: no cover - useful when executed from scripts/.
    from topic_history import DuplicateTopicError, TopicHistory, find_duplicate

MIN_SCORE = 70
DEFAULT_MAX_CANDIDATES = 20
USER_AGENT = "topic-bank-renewer/1.0 (+https://github.com/ahmedalsharqwi2-gif)"

@dataclass(frozen=True)
class Candidate:
    title: str
    hook: str
    queries: tuple[str, ...]
    source_url: str
    evidence: str
    trend_score: float
    youtube_score: float
    score: int
    story_type: str = ""


def clean(value: Any, limit: int = 500) -> str:
    return " ".join(str(value or "").split())[:limit].strip()


def query_groups(value: str | Iterable[str]) -> list[tuple[str, ...]]:
    """Parse aliases while keeping all aliases in one scoring group."""
    if isinstance(value, str):
        raw = re.split(r"[;\n]", value)
    else:
        raw = list(value)
    groups: list[tuple[str, ...]] = []
    for item in raw:
        aliases = tuple(clean(x, 120) for x in re.split(r"[,|]", str(item)) if clean(x))
        if aliases:
            groups.append(aliases)
    return groups


def merge_related_trends(values: dict[str, float], groups: Iterable[Iterable[str]] | None = None) -> dict[str, float]:
    """Merge aliases for one idea without averaging a strong signal with zeros.

    Google Trends often returns one useful alias and zeros for its translation.
    The group score is therefore the maximum observed alias, while unrelated
    query groups remain separate. This fixes the old per-query threshold bug.
    """
    if not groups:
        return {clean(k, 120): float(v or 0) for k, v in values.items()}
    merged: dict[str, float] = {}
    used: set[str] = set()
    for group in groups:
        aliases = [clean(q, 120) for q in group if clean(q)]
        scores = [float(values.get(q, 0) or 0) for q in aliases]
        if aliases:
            merged[" / ".join(aliases)] = max(scores, default=0.0)
            used.update(aliases)
    for key, value in values.items():
        if key not in used:
            merged[clean(key, 120)] = float(value or 0)
    return merged


def _trend_client() -> Any:
    if TrendReq is None:
        return None
    try:
        return TrendReq(hl="en-US", tz=0, timeout=(5, 20), retries=2, backoff_factor=0.3)
    except TypeError:  # Older pytrends versions do not accept timeout.
        return TrendReq(hl="en-US", tz=0, retries=2, backoff_factor=0.3)


def fetch_trends(groups: list[tuple[str, ...]], geo: str, timeframe: str) -> dict[str, float]:
    client = _trend_client()
    if client is None:
        return {}
    result: dict[str, float] = {}
    for group in groups:
        try:
            client.build_payload(list(group), timeframe=timeframe, geo=geo, gprop="")
            frame = client.interest_over_time()
            if frame is None or frame.empty:
                continue
            for query in group:
                if query in frame:
                    result[query] = float(frame[query].mean())
        except Exception as exc:
            print(f"TREND_WARNING query_group={group!r}: {exc}", file=sys.stderr)
    return result


def fetch_youtube(queries: list[str], api_key: str, region: str, max_results: int = 10) -> list[dict[str, Any]]:
    if not api_key:
        return []
    headers = {"Accept": "application/json", "User-Agent": USER_AGENT}
    params = {"part": "snippet", "type": "video", "order": "relevance", "maxResults": max_results,
              "regionCode": region, "q": queries[0], "key": api_key}
    try:
        response = requests.get("https://www.googleapis.com/youtube/v3/search", params=params,
                                headers=headers, timeout=20)
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError) as exc:
        print(f"YOUTUBE_WARNING query={queries[0]!r}: {exc}", file=sys.stderr)
        return []
    items = []
    for item in data.get("items", []):
        snippet = item.get("snippet") or {}
        video_id = ((item.get("id") or {}).get("videoId"))
        if not video_id:
            continue
        items.append({"title": clean(snippet.get("title"), 220), "description": clean(snippet.get("description"), 600),
                      "published_at": clean(snippet.get("publishedAt"), 40),
                      "url": f"https://www.youtube.com/watch?v={quote_plus(video_id)}"})
    return items


def _normalize_key(text: str) -> str:
    return re.sub(r"[^\w\u0600-\u06ff]+", " ", text.casefold()).strip()


def build_candidates(trend_values: dict[str, float], youtube_items: list[dict[str, Any]], groups: list[tuple[str, ...]], kind: str) -> list[Candidate]:
    if not trend_values or not youtube_items:
        return []
    merged = merge_related_trends(trend_values, groups)
    peak = max(merged.values(), default=0.0)
    if peak <= 0:
        return []
    grouped_items: dict[str, dict[str, Any]] = {}
    for item in youtube_items:
        title = clean(item.get("title"), 220)
        if not title:
            continue
        key = _normalize_key(title)
        if key and key not in grouped_items:
            grouped_items[key] = item
    candidates: list[Candidate] = []
    for idx, item in enumerate(grouped_items.values()):
        title = item["title"]
        # Stronger signals are the trend group peak and an actual YouTube hit.
        trend_raw = max(merged.values())
        trend_points = min(25.0, 25.0 * trend_raw / peak)
        youtube_points = min(25.0, 10.0 + min(15.0, len(item.get("description", "")) / 40.0))
        freshness_points = 10.0 if item.get("published_at", "").startswith(str(datetime.now(timezone.utc).year)) else 5.0
        source_points = 20.0 if item.get("url") else 0.0
        visual_points = 10.0 if item.get("description") else 0.0
        short_points = 10.0 if 3 <= len(title.split()) <= 14 else 5.0
        score = round(trend_points + youtube_points + freshness_points + source_points + visual_points + short_points)
        story_type = ""
        if kind == "horror":
            story_type = "sci_fi" if any(word in title.casefold() for word in ("fiction", "خيال", "رواية")) else "true_case"
        hook = item.get("description") or f"ما الذي يجعل موضوع «{title}» جديرًا بالبحث؟"
        candidates.append(Candidate(title=title, hook=clean(hook, 500), queries=tuple(q for g in groups for q in g),
                                    source_url=item["url"], evidence=clean(item.get("description"), 500),
                                    trend_score=round(trend_raw, 2), youtube_score=round(youtube_points, 2),
                                    score=score, story_type=story_type))
    return sorted(candidates, key=lambda c: (-c.score, c.title))


def _bank_titles(path: Path) -> set[str]:
    if not path.exists():
        return set()
    text = path.read_text(encoding="utf-8")
    return {clean(match) for match in re.findall(r"^\|\s*\d+\s*\|\s*([^|]+?)\s*\|", text, re.M)}


def _next_priority(path: Path) -> int:
    titles = _bank_titles(path)
    if not path.exists():
        return 1
    values = [int(x) for x in re.findall(r"^\|\s*(\d+)\s*\|", path.read_text(encoding="utf-8"), re.M)]
    return max(values, default=len(titles)) + 1


def append_to_bank(path: Path, candidate: Candidate) -> None:
    if not path.exists():
        raise RuntimeError(f"topic bank missing: {path}")
    text = path.read_text(encoding="utf-8")
    priority = _next_priority(path)
    keywords = "، ".join(candidate.queries[:4])
    row = f"| {priority} | {candidate.title.replace('|', '/')} | {candidate.hook.replace('|', '/')} | {keywords} |\n"
    if not text.endswith("\n"):
        text += "\n"
    path.write_text(text + row, encoding="utf-8")


def bank_exhausted(bank: Path, history: TopicHistory) -> bool:
    titles = _bank_titles(bank)
    if not titles:
        return True
    return all(find_duplicate({"title": title}, history.entries) for title in titles)


def candidate_record(candidate: Candidate) -> dict[str, Any]:
    return {"title": candidate.title, "hook": candidate.hook, "source": candidate.source_url,
            "score": candidate.score, "trend_score": candidate.trend_score, "youtube_score": candidate.youtube_score,
            "queries": list(candidate.queries), "story_type": candidate.story_type}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", default="TOPIC_BANK.md")
    parser.add_argument("--history", default=None)
    parser.add_argument("--kind", choices=("science", "horror", "history"), required=True)
    parser.add_argument("--geo", default=os.getenv("TRENDS_GEO", "EG"))
    parser.add_argument("--timeframe", default=os.getenv("TRENDS_TIMEFRAME", "today 3-m"))
    parser.add_argument("--min-score", type=int, default=int(os.getenv("TOPIC_MIN_SCORE", MIN_SCORE)))
    parser.add_argument("--max-candidates", type=int, default=DEFAULT_MAX_CANDIDATES)
    parser.add_argument("--write", action="store_true", help="Persist accepted candidates; otherwise dry-run")
    parser.add_argument("--force", action="store_true", help="Renew even if unused bank rows remain")
    args = parser.parse_args(argv)
    bank = Path(args.bank)
    history_path = Path(args.history or ("topic_history.json" if args.kind == "science" else "state/topic_history.json"))
    history = TopicHistory(history_path)
    if not args.force and not bank_exhausted(bank, history):
        print("TOPIC_RENEWAL: bank_not_exhausted; no-op")
        return 0
    groups = query_groups(os.getenv("TOPIC_RENEWAL_QUERIES", {
        "science": "AI, artificial intelligence;quantum computing, الحوسبة الكمية;space mission, مهمة فضائية;robotics, robotics medicine",
        "horror": "unexplained disappearance, اختفاء غامض;abandoned places, أماكن مهجورة;unsolved mystery, لغز لم يحل;strange signals, إشارات غامضة",
        "history": "Islamic history, التاريخ الإسلامي;Andalus, الأندلس;Abbasid Baghdad, بغداد العباسية;Mamluk battles, معارك المماليك",
    }[args.kind]))
    trend_values = fetch_trends(groups, args.geo, args.timeframe)
    queries = [q for group in groups for q in group]
    youtube_items = fetch_youtube(queries, os.getenv("YOUTUBE_API_KEY") or os.getenv("GOOGLE_API_KEY", ""), args.geo)
    candidates = [c for c in build_candidates(trend_values, youtube_items, groups, args.kind) if c.score >= args.min_score]
    accepted: list[Candidate] = []
    existing_titles = _bank_titles(bank)
    for candidate in candidates[:args.max_candidates]:
        if candidate.title in existing_titles or find_duplicate({"title": candidate.title, "hook": candidate.hook}, history.entries):
            print(f"REJECT duplicate: {candidate.title}")
            continue
        accepted.append(candidate)
    print(json.dumps({"bank": str(bank), "history": str(history_path), "trend_queries": len(trend_values),
                      "youtube_results": len(youtube_items), "accepted": [candidate_record(c) for c in accepted],
                      "dry_run": not args.write}, ensure_ascii=False, indent=2))
    if args.write:
        for candidate in accepted:
            append_to_bank(bank, candidate)
            print(f"TOPIC_ACCEPTED score={candidate.score} title={candidate.title}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
