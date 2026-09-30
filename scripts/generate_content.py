# -*- coding: utf-8 -*-
"""
scripts/generate_content.py - توليد المحتوى
============================================
يولد الموضوع والنص الخاص بالفيديو.
"""

import os
import sys
import json
import re
import logging
from pathlib import Path
from typing import Optional

from llm_gemini import pooled_llm_chat as llm_chat
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


def normalize_topic_response(response: str) -> str:
    """Extract a clean topic/title from JSON or Markdown model output."""
    text = re.sub(r"```(?:json|markdown|text)?", "", response or "", flags=re.IGNORECASE)
    text = text.replace("```", "").strip()
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            value = payload.get("topic") or payload.get("title") or payload.get("subject")
            if value:
                return re.sub(r"[*_`\"«»]", "", str(value)).strip()
    except json.JSONDecodeError:
        pass
    for line in text.splitlines():
        clean = re.sub(r"[*_`]", "", line).strip()
        if any(marker in clean for marker in ("العنوان", "الموضوع", "الفكرة")) and ":" in clean:
            return re.sub(r"[\"«»]", "", clean.split(":", 1)[1]).strip()
    return re.sub(r"[*_`\"«»]", "", text.splitlines()[0] if text else "").strip()


def normalize_narration_response(response: str) -> str:
    """Extract narration text and remove transport wrappers from model output."""
    text = re.sub(r"```(?:json|markdown|text)?", "", response or "", flags=re.IGNORECASE)
    text = text.replace("```", "").strip()
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            text = str(payload.get("narration") or payload.get("script") or payload.get("text") or text)
        elif isinstance(payload, str):
            text = payload
    except json.JSONDecodeError:
        # Gemini sometimes returns a JSON-like wrapper with unescaped quotes.
        # Extract the narration value instead of sending {",:} to TTS checks.
        match = re.search(r'"(?:narration|script|text)"\s*:\s*"((?:\\.|[^"\\])*)"', text, re.DOTALL)
        if match:
            try:
                text = json.loads('"' + match.group(1) + '"')
            except json.JSONDecodeError:
                text = match.group(1).replace('\\"', '"').replace('\\n', '\n')
        else:
            text = re.sub(r'^\s*(?:نص السرد|النص|NARRATION|narration)\s*:\s*', '', text, flags=re.IGNORECASE)
    text = text.replace("\\n", "\n").replace("\\t", " ").replace('\\"', '"')
    # Models occasionally emit bidi/control marks or decorative Unicode that
    # is harmless visually but makes the strict Arabic quality gate fail.
    text = re.sub(r"[\u200b-\u200f\u202a-\u202e\ufeff]", "", text)
    text = re.sub(r"[^\u0621-\u064A\u0671-\u06FF\s.!؟؛،0-9()\[\]«»:\"'\-A-Za-z]", " ", text)
    text = re.sub(r"[*_`]", "", text)
    return text.strip().strip('"«»')


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
            topic = normalize_topic_response(response)
            log.info("Generated topic: %s", topic[:100])
            return topic
        except Exception as e:
            log.error(f"Failed to generate topic: {e}")
            raise

    def generate_narration(self, topic: str) -> str:
        """توليد النص الروائي."""
        log.info("Generating narration for topic")
        
        prompt = f"""اكتب نصاً علمياً قصيراً وحيوياً بالعربية الفصحى حول: {topic}

المتطلبات:
- يكون النص بين {self.min_words}-{self.max_words} كلمة
- استخدم لغة واضحة وسهلة النطق
- تجنب الكلمات الأجنبية والأرقام
- اجعل التشكيل (الحركات) على معظم الكلمات
- ابدأ بخطاف أو سؤال علمي واضح في أول جملة، من دون تحية أو مقدمة عامة
- اتبع بنية: معلومة أو مفارقة، ثم شرح مبسط، ثم نتيجة مفاجئة أو تطبيق يومي
- اجعل الجمل قصيرة ومفهومة، وكل جملة تحمل معلومة واحدة قابلة للعرض بصرياً
- لا تكرر الفكرة أو الكلمات، ولا تكتب بأسلوب مقال مدرسي
- بدون رموز خاصة

النص:"""

        try:
            fixed_narration = ""
            for attempt in range(5):
                request = prompt
                if attempt:
                    current_count = len(fixed_narration.split())
                    missing = max(0, self.min_words - current_count)
                    request = (
                        f"النص السابق عدد كلماته {current_count}، وينقصه {missing} كلمة على الأقل. "
                        f"أعد النص كاملًا بين {self.min_words} و{self.max_words} كلمة، "
                        "وأضف شرحًا علميًا وأمثلة مرتبطة بالموضوع بدل الحشو. "
                        "لا تختصره ولا تُرجع ملاحظات خارج النص.\n\n"
                        f"النص السابق:\n{fixed_narration}"
                    )
                response = llm_chat(
                    [{"role": "user", "content": request}],
                    max_tokens=int(os.getenv("LLM_MAX_COMPLETION_TOKENS", "512")),
                )
                narration = normalize_narration_response(response)

                fixed_narration, grammar_fixes = self.grammar_fixer.fix_text(narration)
                if grammar_fixes:
                    log.info(f"Applied {len(grammar_fixes)} grammar fixes")
                word_count = len(fixed_narration.split())
                log.info("Narration length: %d words (required %d–%d)", word_count, self.min_words, self.max_words)
                if self.min_words <= word_count <= self.max_words:
                    break
                if attempt == 0:
                    log.warning("Narration length outside range; requesting a full-length rewrite")
            else:
                raise ValueError(
                    f"النص خارج النطاق بعد ثلاث محاولات: {len(fixed_narration.split())} كلمة، "
                    f"المطلوب {self.min_words}-{self.max_words}"
                )

            report = self.quality_checker.generate_report(fixed_narration)
            log.info(f"Content quality score: {report.overall_score:.2f}/1.0")
            if report.issues:
                log.warning(f"Quality issues found: {report.issues}")
            if not report.is_acceptable:
                log.warning("Content quality below acceptable threshold, attempting revision...")
                revision_prompt = f"الرجاء إصلاح الأخطاء التالية في النص مع الحفاظ على طوله بين {self.min_words} و{self.max_words} كلمة:\n{chr(10).join(report.issues)}\n\nالنص الأصلي:\n{fixed_narration}"
                response = llm_chat(
                    [{"role": "user", "content": revision_prompt}],
                    max_tokens=int(os.getenv("LLM_MAX_COMPLETION_TOKENS", "512")),
                )
                fixed_narration = normalize_narration_response(response)
                revised_words = len(fixed_narration.split())
                if not self.min_words <= revised_words <= self.max_words:
                    raise ValueError(f"النص بعد المراجعة خارج النطاق: {revised_words} كلمة")
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
