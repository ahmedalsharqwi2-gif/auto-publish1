# -*- coding: utf-8 -*-
"""
arabbic TTS Quality Checker - فاحص جودة النطق العربي
========================================================
فحص شامل لجودة النصوص وملفات الصوت قبل وبعد توليد الكلام.
"""

from __future__ import annotations

import re
import logging
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict

log = logging.getLogger("pipeline")


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
                score *= 0.85
            elif tashkeel_ratio < 0.3:
                issues.append(f"التشكيل ناقص جدًا: {tashkeel_ratio:.0%}")
                score *= 0.5

        # Check 3: Sentence length
        sentences = re.split(r'[.!؟؛]+', text)
        long_sentences = [s for s in sentences if len(s.split()) > 25]
        if long_sentences:
            warnings.append(f"{len(long_sentences)} جملة طويلة قد تسبب اختلال الإيقاع")
            score *= 0.9

        # Check 4: Repeated words
        words_lower = [w.lower() for w in re.findall(r'\w+', text)]
        word_counts = {}
        for word in words_lower:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        repeated = [w for w, c in word_counts.items() if c >= 5]
        if repeated:
            warnings.append(f"كلمات مكررة كثيرًا: {', '.join(repeated[:3])}")
            score *= 0.9

        # Check 5: Special characters
        special_chars = re.findall(r'[^\u0621-\u064A\s\u064B-\u0652.!؟؛،]', text)
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
            import numpy as np
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
                # For MP3 files, use simple estimation
                file_size = audio_path.stat().st_size
                duration = file_size / (128 * 1024)  # Rough estimate for 128kbps MP3

            expected_words = len(expected_text.split())
            expected_duration = expected_words / 2.5  # Average 2.5 words per second in Arabic
            duration_ratio = duration / expected_duration if expected_duration > 0 else 0

            if not (0.8 <= duration_ratio <= 1.3):
                issues.append(f"مدة الصوت غير متوقعة: {duration:.1f}s (متوقع ~{expected_duration:.1f}s)")
                score *= 0.8

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
