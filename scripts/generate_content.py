# -*- coding: utf-8 -*-
"""
scripts/generate_content.py - توليد المحتوى
============================================
يولد الموضوع والنص الخاص بالفيديو.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional

from llm_gemini import gemini_chat
from arabic_grammar_fixer import ArabicGrammarFixer
from arabic_tts_quality_checker import ArabicTTSQualityChecker

log = logging.getLogger("pipeline")


class ContentGenerator:
    """يولد المحتوى الأساسي (نص وموضوع)."""

    def __init__(self, min_words: int = 300, max_words: int = 1000):
        self.min_words = min_words
        self.max_words = max_words
        self.grammar_fixer = ArabicGrammarFixer()
        self.quality_checker = ArabicTTSQualityChecker()

    def generate_topic(self, category: str = "") -> str:
        """توليد موضوع جديد."""
        log.info("Generating topic for category: %s", category or "general")
        prompt = f"توليد موضوع فريد ومثير للاهتمام باللغة العربية الفصحى {'للفئة: ' + category if category else ''}. الموضوع يجب أن يكون مفيداً وقابلاً للنشر على وسائل التواصل."
        
        try:
            response = gemini_chat([{"role": "user", "content": prompt}])
            topic = response.strip()
            log.info("Generated topic: %s", topic[:100])
            return topic
        except Exception as e:
            log.error(f"Failed to generate topic: {e}")
            raise

    def generate_narration(self, topic: str) -> str:
        """توليد النص الروائي."""
        log.info("Generating narration for topic")
        
        prompt = f"""اكتب نصاً روائياً بالعربية الفصحى حول: {topic}

المتطلبات:
- يكون النص بين {self.min_words}-{self.max_words} كلمة
- استخدم لغة واضحة وسهلة النطق
- تجنب الكلمات الأجنبية والأرقام
- اجعل التشكيل (الحركات) على معظم الكلمات
- الجمل قصيرة ومفهومة
- بدون رموز خاصة

النص:"""

        try:
            response = gemini_chat([{"role": "user", "content": prompt}])
            narration = response.strip()
            
            # Fix grammar and quality
            fixed_narration, grammar_fixes = self.grammar_fixer.fix_text(narration)
            if grammar_fixes:
                log.info(f"Applied {len(grammar_fixes)} grammar fixes")
            
            # Check quality
            report = self.quality_checker.generate_report(fixed_narration)
            log.info(f"Content quality score: {report.overall_score:.2f}/1.0")
            
            if report.issues:
                log.warning(f"Quality issues found: {report.issues}")
            
            if not report.is_acceptable:
                log.warning("Content quality below acceptable threshold, attempting revision...")
                # Request revision
                revision_prompt = f"الرجاء إصلاح الأخطاء التالية في النص:\n{chr(10).join(report.issues)}\n\nالنص الأصلي:\n{fixed_narration}"
                response = gemini_chat([{"role": "user", "content": revision_prompt}])
                fixed_narration = response.strip()
            
            return fixed_narration
            
        except Exception as e:
            log.error(f"Failed to generate narration: {e}")
            raise


if __name__ == "__main__":
    generator = ContentGenerator()
    topic = generator.generate_topic("تاريخ وحكمة")
    print(f"Topic: {topic}")
    
    narration = generator.generate_narration(topic)
    print(f"\nNarration: {narration}")
