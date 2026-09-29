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

from llm_gemini import llm_chat
from arabic_grammar_fixer import ArabicGrammarFixer
from arabic_tts_quality_checker import ArabicTTSQualityChecker

log = logging.getLogger("pipeline")

CHANNEL_BRIEF = (
    "وجهتك الأولى لاكتشاف أسرار الفضاء والظواهر الكونية المذهلة، "
    "وخفايا الطبيعة وأعماق المحيطات. نقدم رحلات علمية ممتعة وسريعة "
    "لتبسيط أعقد العلوم والهندسة والطب والفلك والفيزياء، والإجابة عن "
    "التساؤلات التي تحير العقول."
)

# موضوعات ذات قابلية عالية للمشاهدة، مع شرط أن تكون قابلة للتحقق وليست عناوين
# صادمة مضللة. يختار النموذج منها ويدوّر بينها بدل تكرار فئة واحدة.
TOPIC_CATEGORIES = (
    "أسرار الفضاء والكون والثقوب السوداء والكواكب الغريبة",
    "ظواهر كونية مذهلة يمكن شرحها ببساطة",
    "أعماق المحيطات والكائنات البحرية النادرة",
    "خفايا الطبيعة والحيوانات والقدرات المدهشة للكائنات",
    "أسرار جسم الإنسان والطب والدماغ والنوم والذاكرة",
    "الفيزياء في حياتنا اليومية والأشياء التي تحدث من حولنا",
    "الهندسة والتقنية والذكاء الاصطناعي ومستقبل الحياة",
    "ألغاز علمية حقيقية وحوادث تاريخية حُسم تفسيرها بالدليل",
    "الأرض والبراكين والزلازل والمناخ والظواهر الجوية",
    "اختراعات وتجارب علمية غيّرت العالم وكيف تعمل",
    "مقارنات علمية سريعة: ماذا يحدث لو تغيرت قاعدة في الطبيعة؟",
    "ماذا يحدث لو اختفت الجاذبية أو الأكسجين أو المجال المغناطيسي لثوانٍ؟",
    "البرق والرعد والضوء: ظواهر نراها كثيراً ولا نفهمها جيداً",
    "حيوانات تضيء أو تتنفس بطرق غير متوقعة وكيف تطورت هذه القدرات",
    "أسرار النوم والأحلام والذاكرة بحدود ما يثبته علم الأعصاب",
    "كيف نعرف ما يحدث داخل الأرض أو على كوكب بعيد من دون الوصول إليه؟",
    "الزمن والجاذبية والنسبية في أمثلة يومية سهلة الفهم",
    "ظواهر جوية نادرة مثل البرق الكروي والسحب الغريبة وتفسيرها العلمي",
    "تقنيات المستقبل الواقعية: الطاقة النظيفة والروبوتات والطب الدقيق",
)


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
        categories = "\n".join(f"- {item}" for item in TOPIC_CATEGORIES)
        requested_category = category if category and category not in {"عام", "general"} else "اختر الفئة الأنسب تلقائياً"
        prompt = f"""أنت محرر علمي لقناة عربية قصيرة.
وصف القناة:
{CHANNEL_BRIEF}

الفئة المطلوبة: {requested_category}
الفئات المقترحة للتدوير:
{categories}

اختر فكرة واحدة فقط لفيديو قصير، ثم اكتب عنواناً جذاباً واضحاً من دون مبالغة.
اجعل الفكرة قابلة للتحقق من مصادر علمية موثوقة، وابتعد عن الخرافات ونظريات المؤامرة
والادعاءات الطبية الخطرة. استخدم سؤالاً أو مفارقة أو رقماً موثقاً في العنوان عندما
يكون ذلك طبيعياً، لكن لا تستخدم كلمات مثل "صدمة" أو "لن تصدق" بلا معلومة حقيقية.
أعد الموضوع والعنوان المقترح بالعربية فقط في سطرين."""
        
        try:
            response = llm_chat([{"role": "user", "content": prompt}])
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
            response = llm_chat([{"role": "user", "content": prompt}])
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
                response = llm_chat([{"role": "user", "content": revision_prompt}])
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
