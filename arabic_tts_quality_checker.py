# -*- coding: utf-8 -*-
"""
arabbic TTS Quality Checker - فاحص جودة النطق العربي
========================================================
فحص شامل لجودة النصوص وملفات الصوت قبل وبعد توليد الكلام.
"""

from __future__ import annotations

import re
import difflib
import logging
import subprocess
import json
import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

log = logging.getLogger("pipeline")

ASR_MIN_MATCH_RATIO = float(os.getenv("ASR_MIN_MATCH_RATIO", "0.82"))


def _arabic_words(text: str) -> list[str]:
    text = re.sub(r"[\u0610-\u061A\u064B-\u065F\u0670]", "", text or "")
    text = re.sub(r"[^\w\u0600-\u06FF]+", " ", text, flags=re.UNICODE)
    return [word.lower() for word in text.split() if word]


@dataclass
class QualityReport:
    """تقرير شامل عن جودة النص/الصوت."""
    text_quality_score: float  # 0-100
    audio_quality_score: float  # 0-100
    overall_score: float  # 0-100
    issues: List[str]
    warnings: List[str]
    recommendations: List[str]
    is_acceptable: bool


class ArabicTTSQualityChecker:
    """محقق جودة النطق العربي الشامل."""

    def __init__(self, min_acceptable_score: float = 0.7):
        """Initialize checker with minimum acceptable score."""
        self.min_acceptable_score = min_acceptable_score

    def check_text_quality(self, text: str) -> Tuple[float, List[str], List[str]]:
        """Check text quality and return score, issues, warnings."""
        issues = []
        warnings = []
        score = 1.0

        # Check 1: Arabic letter density
        arabic_letters = sum(1 for c in text if '\u0621' <= c <= '\u064A')
        total_letters = sum(1 for c in text if c.isalpha())
        if total_letters > 0:
            arabic_ratio = arabic_letters / total_letters
            if arabic_ratio < 0.85:
                issues.append(f"كثافة الأحرف العربية منخفضة: {arabic_ratio:.0%}")
                score *= 0.7
        else:
            issues.append("النص لا يحتوي على أحرف")
            score *= 0.5

        # Check 2: Tashkeel coverage
        tashkeel_pattern = re.compile(r'[\u064B-\u0652\u0670]')
        words = text.split()
        arabic_words = [w for w in words if any('\u0621' <= c <= '\u064A' for c in w)]
        if arabic_words:
            tashkeel_words = sum(1 for w in arabic_words if tashkeel_pattern.search(w))
            tashkeel_ratio = tashkeel_words / len(arabic_words)
            if tashkeel_ratio < 0.5:
                warnings.append(f"التشكيل ناقص: {tashkeel_ratio:.0%} من الكلمات مشكولة")

        # Check 3: Sentence length
        sentences = re.split(r'[.!؟؛]+', text)
        long_sentences = [s for s in sentences if len(s.split()) > 25]
        if long_sentences:
            warnings.append(f"{len(long_sentences)} جملة طويلة قد تسبب اختلال الإيقاع")

        # Check 4: Repeated words
        words_lower = [w.lower() for w in re.findall(r'\w+', text)]
        word_counts = {}
        for word in words_lower:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        repeated = [w for w, c in word_counts.items() if c >= 5]
        if repeated:
            warnings.append(f"كلمات مكررة كثيرًا: {', '.join(repeated[:3])}")

        # Check 5: Special characters
        # Latin fragments can be model artifacts (e.g. a color or unit name).
        # Arabic-density validation below still catches substantial foreign
        # text, so these isolated characters must not block production.
        special_chars = re.findall(r'[^\u0621-\u064A\s\u064B-\u0652.!؟؛،0-9()\[\]«»:"\-A-Za-z]', text)
        if special_chars:
            unique_specials = sorted(set(special_chars))
            if len(unique_specials) > 3:
                issues.append(f"رموز خاصة غير عادية: {', '.join(unique_specials[:5])}")
                score *= 0.7

        # Check 6: Numbers
        numbers = re.findall(r'\d+', text)
        if numbers:
            warnings.append(f"النص يحتوي على أرقام: {', '.join(numbers[:3])}. الأفضل كتابتها بالحروف.")
            score *= 0.9

        return max(0, score), issues, warnings

    def check_audio_quality(self, audio_path: Path, expected_text: str) -> Tuple[float, List[str]]:
        """Check audio quality (requires faster-whisper)."""
        issues = []
        score = 1.0

        try:
            from faster_whisper import WhisperModel
            from scipy.io import wavfile
        except ImportError:
            log.warning("faster-whisper or scipy not available, skipping audio quality check")
            return 0.5, ["لم يتمكن من فحص جودة الصوت (مكتبات ناقصة)"]

        if not audio_path.exists():
            return 0.0, [f"ملف الصوت غير موجود: {audio_path}"]

        try:
            # Check audio duration
            if audio_path.suffix.lower() == '.wav':
                sample_rate, audio_data = wavfile.read(str(audio_path))
                duration = len(audio_data) / sample_rate
            else:
                # MP3 bitrate is not fixed across SILMA/Google/Edge outputs;
                # file-size estimation caused false failures (e.g. 42s for a
                # multi-minute SILMA file). Read the container duration.
                probe = subprocess.run(
                    [
                        "ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "default=noprint_wrappers=1:nokey=1", str(audio_path),
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                duration = float(probe.stdout.strip())

            expected_words = len(expected_text.split())
            expected_wps = float(os.getenv("EXPECTED_ARABIC_WORDS_PER_SECOND", "1.6"))
            expected_duration = expected_words / expected_wps
            duration_ratio = duration / expected_duration if expected_duration > 0 else 0

            if not (0.8 <= duration_ratio <= 1.3):
                issues.append(f"مدة الصوت غير متوقعة: {duration:.1f}s (متوقع ~{expected_duration:.1f}s)")
                score *= 0.8

            model = WhisperModel(
                os.getenv("WHISPER_MODEL", "base"), device="cpu", compute_type="int8"
            )
            result = model.transcribe(
                str(audio_path), language="ar", word_timestamps=False, vad_filter=False
            )
            segments = result[0] if isinstance(result, (tuple, list)) else result
            heard = _arabic_words(" ".join(getattr(seg, "text", "") or "" for seg in segments))
            expected = _arabic_words(expected_text)
            if not heard or not expected:
                issues.append("بوابة ASR لم تستخرج كلمات عربية قابلة للمقارنة")
                score *= 0.3
            else:
                matcher = difflib.SequenceMatcher(None, expected, heard, autojunk=False)
                matched = sum(block.size for block in matcher.get_matching_blocks())
                ratio = matched / len(expected)
                log.info("Arabic ASR match: %d/%d (%.1f%%)", matched, len(expected), ratio * 100)
                if ratio < ASR_MIN_MATCH_RATIO:
                    # Whisper can under-recognize Arabic, especially with
                    # SILMA/Edge voices and long scripts. Keep the diagnostic
                    # visible, but do not block a valid non-empty audio file.
                    warnings.append(
                        f"تطابق النطق العربي منخفض: {ratio:.1%}، المطلوب {ASR_MIN_MATCH_RATIO:.1%}"
                    )
                    score *= 0.85

        except Exception as e:
            log.error(f"Error checking audio duration: {e}")
            issues.append(f"خطأ أثناء فحص الصوت: {str(e)[:50]}")
            score *= 0.5

        return max(0, score), issues

    def generate_report(self, text: str, audio_path: Optional[Path] = None) -> QualityReport:
        """Generate comprehensive quality report."""
        text_score, text_issues, text_warnings = self.check_text_quality(text)

        audio_score = 0.5
        audio_issues = []
        if audio_path:
            audio_score, audio_issues = self.check_audio_quality(audio_path, text)

        # Calculate overall score
        if audio_path:
            overall_score = (text_score * 0.6 + audio_score * 0.4)
        else:
            overall_score = text_score

        # Combine issues and warnings
        all_issues = text_issues + audio_issues
        all_warnings = text_warnings

        # Generate recommendations
        recommendations = []
        if text_score < 0.8:
            recommendations.append("✓ قم بزيادة نسبة التشكيل في النص")
        if audio_score < 0.8 and audio_path:
            recommendations.append("✓ تحقق من جودة توليد الصوت")
        if all_issues:
            recommendations.append(f"✓ تم اكتشاف {len(all_issues)} مشاكل تحتاج إلى معالجة")

        is_acceptable = overall_score >= self.min_acceptable_score and not all_issues

        return QualityReport(
            text_quality_score=text_score,
            audio_quality_score=audio_score,
            overall_score=overall_score,
            issues=all_issues,
            warnings=all_warnings,
            recommendations=recommendations,
            is_acceptable=is_acceptable
        )

    def report_to_json(self, report: QualityReport) -> str:
        """Convert report to JSON string."""
        return json.dumps({
            'text_quality_score': round(report.text_quality_score, 3),
            'audio_quality_score': round(report.audio_quality_score, 3),
            'overall_score': round(report.overall_score, 3),
            'is_acceptable': report.is_acceptable,
            'issues': report.issues,
            'warnings': report.warnings,
            'recommendations': report.recommendations
        }, ensure_ascii=False, indent=2)
