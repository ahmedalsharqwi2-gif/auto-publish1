# -*- coding: utf-8 -*-
"""
scripts/quality_check.py - فحص الجودة الشامل
============================================
يفحص جودة النصوص والملفات الصوتية قبل النشر.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Tuple

from arabic_grammar_fixer import ArabicGrammarFixer
from arabic_tts_quality_checker import ArabicTTSQualityChecker, QualityReport

log = logging.getLogger("pipeline")


class QualityCheckPipeline:
    """خط أنابيب شامل لفحص الجودة."""

    def __init__(self, min_acceptable_score: float = 0.75):
        self.grammar_fixer = ArabicGrammarFixer()
        self.tts_checker = ArabicTTSQualityChecker(min_acceptable_score)

    def check_text(
        self,
        text: str,
        auto_fix: bool = True,
    ) -> Tuple[str, QualityReport]:
        """Check and optionally fix text quality."""
        log.info("Starting text quality check...")
        
        # Fix grammar issues
        if auto_fix:
            fixed_text, fixes = self.grammar_fixer.fix_text(text)
            if fixes:
                log.info(f"Applied {len(fixes)} grammar fixes:")
                for fix in fixes:
                    log.debug(f"  - {fix}")
        else:
            fixed_text = text
        
        # Generate quality report
        report = self.tts_checker.generate_report(fixed_text)
        
        # Log results
        log.info(f"Text quality score: {report.text_quality_score:.2f}/1.0")
        
        if report.issues:
            log.warning(f"Found {len(report.issues)} issue(s):")
            for issue in report.issues:
                log.warning(f"  - {issue}")
        
        if report.warnings:
            log.info(f"Found {len(report.warnings)} warning(s):")
            for warning in report.warnings:
                log.info(f"  - {warning}")
        
        if report.recommendations:
            log.info("Recommendations:")
            for rec in report.recommendations:
                log.info(f"  {rec}")
        
        return fixed_text, report

    def check_audio(
        self,
        audio_path: Path,
        expected_text: str,
    ) -> QualityReport:
        """Check audio quality."""
        log.info(f"Checking audio quality: {audio_path}")
        report = self.tts_checker.generate_report(expected_text, audio_path)
        
        log.info(f"Audio quality score: {report.audio_quality_score:.2f}/1.0")
        log.info(f"Overall quality score: {report.overall_score:.2f}/1.0")
        
        if not report.is_acceptable:
            log.warning("Audio quality below acceptable threshold")
            if report.issues:
                for issue in report.issues:
                    log.warning(f"  - {issue}")
        
        return report

    def check_pipeline(
        self,
        text: str,
        audio_path: Optional[Path] = None,
        auto_fix: bool = True,
        require_acceptable: bool = True,
    ) -> Tuple[str, QualityReport]:
        """Complete quality check pipeline."""
        log.info("Running complete quality check pipeline...")
        
        # Check text
        fixed_text, text_report = self.check_text(text, auto_fix)
        
        # Check audio if provided
        if audio_path and audio_path.exists():
            audio_report = self.check_audio(audio_path, fixed_text)
            # Merge reports
            merged_report = QualityReport(
                text_quality_score=text_report.text_quality_score,
                audio_quality_score=audio_report.audio_quality_score,
                overall_score=(text_report.text_quality_score * 0.6 + audio_report.audio_quality_score * 0.4),
                issues=text_report.issues + audio_report.issues,
                warnings=text_report.warnings + audio_report.warnings,
                recommendations=text_report.recommendations + audio_report.recommendations,
                is_acceptable=(text_report.is_acceptable and audio_report.is_acceptable),
            )
            final_report = merged_report
        else:
            final_report = text_report
        
        # Final decision
        if require_acceptable and not final_report.is_acceptable:
            log.error("Quality check failed - content not acceptable for publishing")
        else:
            log.info("Quality check passed - content ready for publishing")
        
        return fixed_text, final_report


if __name__ == "__main__":
    checker = QualityCheckPipeline()
    
    test_text = """السلام عليكم ورحمة الله وبركاته. هذا نص جميل وطويل نسبياً يحتوي على معلومات مفيدة.
    النص يجب أن يكون مشكولاً بشكل صحيح. هنا نرى استخدام الكلمات العربية الفصحى.
    التشكيل مهم جداً لتحسين جودة النطق في محركات تحويل النص إلى كلام."""
    
    fixed_text, report = checker.check_pipeline(test_text)
    print(f"\nFixed text:\n{fixed_text}")
    print(f"\nQuality Report:\n{report}")
